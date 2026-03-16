package rag

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/borro/ragcli/internal/input"
	"github.com/borro/ragcli/internal/llm"
	"github.com/borro/ragcli/internal/localize"
	"github.com/borro/ragcli/internal/ragcore"
	"github.com/borro/ragcli/internal/retrieval"
	"github.com/borro/ragcli/internal/verbose"
	openai "github.com/sashabaranov/go-openai"
)

type Options = ragcore.RunOptions

type Turn struct {
	Question string
	Answer   string
}

type Session struct {
	searcher   *ragcore.PreparedSearch
	embedder   llm.EmbeddingRequester
	opts       Options
	indexStats ragcore.SearchStats
}

type Conversation struct {
	chat     llm.ChatRequester
	session  *Session
	turns    []Turn
	baseline []Turn
}

type pipelineStats struct {
	EmbeddingCalls      int
	EmbeddingTokens     int
	CacheHit            bool
	ChunkCount          int
	RetrievedK          int
	AnswerContextChunks int
}

func Run(ctx context.Context, chat llm.ChatRequester, embedder llm.EmbeddingRequester, source input.FileBackedSource, opts Options, question string, plan *verbose.Plan) (string, error) {
	session, err := PrepareSession(ctx, embedder, source, opts, plan)
	if err != nil {
		return "", err
	}
	return session.Answer(ctx, chat, question, nil, plan)
}

func PrepareSession(ctx context.Context, embedder llm.EmbeddingRequester, source input.FileBackedSource, opts Options, plan *verbose.Plan) (*Session, error) {
	indexMeter := plan.Stage("index")
	indexMeter.Start(localize.T("progress.rag.read_input"))
	searcher, searchStats, err := ragcore.PrepareSearch(ctx, embedder, source, ragcore.SearchOptionsFromRunOptions(opts), indexMeter)
	if err != nil {
		return nil, err
	}
	return &Session{
		searcher:   searcher,
		embedder:   embedder,
		opts:       opts,
		indexStats: searchStats,
	}, nil
}

func StartConversation(
	ctx context.Context,
	chat llm.ChatRequester,
	embedder llm.EmbeddingRequester,
	source input.FileBackedSource,
	opts Options,
	question string,
	plan *verbose.Plan,
) (*Conversation, string, error) {
	session, err := PrepareSession(ctx, embedder, source, opts, plan)
	if err != nil {
		return nil, "", err
	}
	answer, err := session.Answer(ctx, chat, question, nil, plan)
	if err != nil {
		return nil, "", err
	}
	turns := []Turn{{Question: strings.TrimSpace(question), Answer: answer}}
	return &Conversation{
		chat:     chat,
		session:  session,
		turns:    append([]Turn(nil), turns...),
		baseline: append([]Turn(nil), turns...),
	}, answer, nil
}

func (c *Conversation) Ask(ctx context.Context, question string, plan *verbose.Plan) (string, error) {
	answer, err := c.session.Answer(ctx, c.chat, question, c.turns, plan)
	if err != nil {
		return "", err
	}
	c.turns = append(c.turns, Turn{
		Question: strings.TrimSpace(question),
		Answer:   answer,
	})
	return answer, nil
}

func (c *Conversation) FollowUpPlan(reporter verbose.Reporter) *verbose.Plan {
	return verbose.NewPlan(reporter, localize.T("progress.mode.rag"),
		verbose.StageDef{Key: "retrieval", Label: localize.T("progress.stage.retrieval"), Slots: 7},
		verbose.StageDef{Key: "final", Label: localize.T("progress.stage.final"), Slots: 7},
	)
}

func (c *Conversation) Reset() {
	if c == nil {
		return
	}
	c.turns = append([]Turn(nil), c.baseline...)
}

func (s *Session) Answer(ctx context.Context, chat llm.ChatRequester, question string, history []Turn, plan *verbose.Plan) (string, error) {
	if s == nil || s.searcher == nil {
		return "", errors.New("rag session is not prepared")
	}
	startedAt := time.Now()
	stats := pipelineStats{}
	retrievalMeter := plan.Stage("retrieval")
	finalMeter := plan.Stage("final")

	stats.EmbeddingCalls += s.indexStats.EmbeddingCalls
	stats.EmbeddingTokens += s.indexStats.EmbeddingTokens
	stats.CacheHit = s.indexStats.CacheHit
	stats.ChunkCount = s.indexStats.ChunkCount
	question = strings.TrimSpace(question)
	query := buildQueryContext(history, question)

	retrievalMeter.Start(localize.T("progress.rag.embed_query"))
	ranked, fusedStats, err := s.searcher.FusedSearchWithHook(ctx, s.embedder, query, "", func() {
		retrievalMeter.Note(localize.T("progress.rag.query_embedding_ready"))
	})
	if err != nil {
		return "", err
	}
	stats.EmbeddingCalls += fusedStats.EmbeddingCalls
	stats.EmbeddingTokens += fusedStats.EmbeddingTokens
	retrievalMeter.Note(localize.T("progress.rag.search_relevant"))
	retrievalMeter.Done(localize.T("progress.rag.retrieval_done", localize.Data{"Candidates": len(ranked), "Evidence": min(len(ranked), s.opts.FinalK)}))
	if ragcore.FusedSearchTooWeak(ranked) {
		slog.Debug("rag retrieval insufficient",
			"duration", float64(time.Since(startedAt).Round(time.Millisecond))/float64(time.Second),
			"chunk_count", stats.ChunkCount,
			"embedding_calls", stats.EmbeddingCalls,
			"embedding_tokens", stats.EmbeddingTokens,
			"cache_hit", stats.CacheHit,
			"retrieved_k", 0,
		)
		return localize.T("rag.answer.insufficient"), nil
	}
	stats.RetrievedK = len(ranked)

	evidence := selectEvidence(ranked, s.opts.FinalK)
	stats.AnswerContextChunks = len(evidence)

	finalMeter.Start(localize.T("progress.rag.final_start"))
	finalMeter.Note(localize.T("progress.rag.final_send", localize.Data{"Count": len(evidence)}))
	answer, chatMetrics, err := synthesizeAnswer(ctx, chat, question, evidence, history)
	if err != nil {
		return "", err
	}
	finalMeter.Note(localize.T("progress.rag.final_citations"))

	slog.Debug("rag processing finished",
		"duration", float64(time.Since(startedAt).Round(time.Millisecond))/float64(time.Second),
		"chunk_count", stats.ChunkCount,
		"embedding_calls", stats.EmbeddingCalls,
		"embedding_tokens", stats.EmbeddingTokens,
		"cache_hit", stats.CacheHit,
		"retrieved_k", stats.RetrievedK,
		"answer_context_chunks", stats.AnswerContextChunks,
		"prompt_tokens", chatMetrics.PromptTokens,
		"completion_tokens", chatMetrics.CompletionTokens,
		"total_tokens", chatMetrics.TotalTokens,
	)

	finalMeter.Done(localize.T("progress.rag.final_done"))
	return appendSources(answer, evidence), nil
}

func selectEvidence(ranked []ragcore.FusedSeedHit, finalK int) []ragcore.FusedSeedHit {
	selected := make([]ragcore.FusedSeedHit, 0, min(finalK, len(ranked)))
	for i, hit := range ranked {
		if len(selected) >= finalK {
			break
		}
		if isAdjacentToSelected(hit, selected) && len(ranked)-i > (finalK-len(selected)) {
			continue
		}
		selected = append(selected, hit)
	}
	if len(selected) < finalK {
		for _, hit := range ranked {
			if len(selected) >= finalK {
				break
			}
			if containsHit(selected, hit) {
				continue
			}
			selected = append(selected, hit)
		}
	}
	return selected
}

func containsHit(selected []ragcore.FusedSeedHit, hit ragcore.FusedSeedHit) bool {
	for _, item := range selected {
		if item.Path == hit.Path && item.ChunkID == hit.ChunkID {
			return true
		}
	}
	return false
}

func isAdjacentToSelected(hit ragcore.FusedSeedHit, selected []ragcore.FusedSeedHit) bool {
	for _, item := range selected {
		if item.Path != hit.Path {
			continue
		}
		delta := item.ChunkID - hit.ChunkID
		if delta >= -1 && delta <= 1 {
			return true
		}
	}
	return false
}

func synthesizeAnswer(ctx context.Context, chat llm.ChatRequester, question string, evidence []ragcore.FusedSeedHit, history []Turn) (string, llm.RequestMetrics, error) {
	contextBlocks := make([]string, 0, len(evidence))
	for i, hit := range evidence {
		contextBlocks = append(contextBlocks, fmt.Sprintf(
			"[source %d] %s:%d-%d\n%s",
			i+1,
			hit.Path,
			hit.StartLine,
			hit.EndLine,
			llm.SanitizeUntrustedBlock(hit.Text),
		))
	}

	req := openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{
			{
				Role: "system",
				Content: localize.T("rag.prompt.answer_system", localize.Data{
					"Policy":       localize.T("rag.prompt.shared_policy"),
					"Conversation": localize.T("rag.prompt.conversation_policy"),
				}),
			},
			{
				Role: "user",
				Content: localize.T("rag.prompt.answer_user", localize.Data{
					"Question":     question,
					"Sources":      strings.Join(contextBlocks, "\n\n---\n\n"),
					"Conversation": formatConversationHistory(history),
				}),
			},
		},
	}

	resp, metrics, err := chat.SendRequestWithMetrics(ctx, req)
	if err != nil {
		return "", llm.RequestMetrics{}, fmt.Errorf("failed to synthesize rag answer: %w", err)
	}
	if resp == nil || len(resp.Choices) == 0 {
		return "", llm.RequestMetrics{}, errors.New("rag answer response has no choices")
	}
	answer := strings.TrimSpace(resp.Choices[0].Message.Content)
	if answer == "" {
		return "", llm.RequestMetrics{}, errors.New("rag answer response is empty")
	}
	return answer, metrics, nil
}

func buildQueryContext(history []Turn, question string) string {
	trimmedQuestion := strings.TrimSpace(question)
	if len(history) == 0 {
		return trimmedQuestion
	}
	questions := make([]string, 0, len(history))
	for _, turn := range history {
		if q := strings.TrimSpace(turn.Question); q != "" {
			questions = append(questions, q)
		}
	}
	if len(questions) == 0 {
		return trimmedQuestion
	}
	return localize.T("rag.prompt.query_context", localize.Data{
		"History": strings.Join(questions, "\n- "),
		"Current": trimmedQuestion,
	})
}

func formatConversationHistory(history []Turn) string {
	if len(history) == 0 {
		return localize.T("rag.prompt.conversation_none")
	}
	parts := make([]string, 0, len(history))
	for index, turn := range history {
		parts = append(parts, localize.T("rag.prompt.conversation_turn", localize.Data{
			"Index":    index + 1,
			"Question": strings.TrimSpace(turn.Question),
			"Answer":   strings.TrimSpace(turn.Answer),
		}))
	}
	return strings.Join(parts, "\n\n")
}

func appendSources(answer string, evidence []ragcore.FusedSeedHit) string {
	citations := make([]retrieval.Citation, 0, len(evidence))
	for _, hit := range evidence {
		citations = append(citations, retrieval.Citation{
			SourcePath: hit.Path,
			StartLine:  hit.StartLine,
			EndLine:    hit.EndLine,
		})
	}
	return retrieval.AppendSources(answer, citations)
}
