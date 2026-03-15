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

type pipelineStats struct {
	EmbeddingCalls      int
	EmbeddingTokens     int
	CacheHit            bool
	ChunkCount          int
	RetrievedK          int
	AnswerContextChunks int
}

func Run(ctx context.Context, chat llm.ChatRequester, embedder llm.EmbeddingRequester, source input.FileBackedSource, opts Options, question string, plan *verbose.Plan) (string, error) {
	startedAt := time.Now()
	stats := pipelineStats{}
	indexMeter := plan.Stage("index")
	retrievalMeter := plan.Stage("retrieval")
	finalMeter := plan.Stage("final")

	indexMeter.Start(localize.T("progress.rag.read_input"))
	searcher, searchStats, err := ragcore.PrepareSearch(ctx, embedder, source, ragcore.SearchOptionsFromRunOptions(opts), indexMeter)
	if err != nil {
		return "", err
	}
	stats.EmbeddingCalls += searchStats.EmbeddingCalls
	stats.EmbeddingTokens += searchStats.EmbeddingTokens
	stats.CacheHit = searchStats.CacheHit
	stats.ChunkCount = searchStats.ChunkCount

	retrievalMeter.Start(localize.T("progress.rag.embed_query"))
	ranked, fusedStats, err := searcher.FusedSearchWithHook(ctx, embedder, question, "", func() {
		retrievalMeter.Note(localize.T("progress.rag.query_embedding_ready"))
	})
	if err != nil {
		return "", err
	}
	stats.EmbeddingCalls += fusedStats.EmbeddingCalls
	stats.EmbeddingTokens += fusedStats.EmbeddingTokens
	retrievalMeter.Note(localize.T("progress.rag.search_relevant"))
	retrievalMeter.Done(localize.T("progress.rag.retrieval_done", localize.Data{"Candidates": len(ranked), "Evidence": min(len(ranked), opts.FinalK)}))
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

	evidence := selectEvidence(ranked, opts.FinalK)
	stats.AnswerContextChunks = len(evidence)

	finalMeter.Start(localize.T("progress.rag.final_start"))
	finalMeter.Note(localize.T("progress.rag.final_send", localize.Data{"Count": len(evidence)}))
	answer, chatMetrics, err := synthesizeAnswer(ctx, chat, question, evidence)
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
		if abs(item.ChunkID-hit.ChunkID) <= 1 {
			return true
		}
	}
	return false
}

func synthesizeAnswer(ctx context.Context, chat llm.ChatRequester, question string, evidence []ragcore.FusedSeedHit) (string, llm.RequestMetrics, error) {
	contextBlocks := make([]string, 0, len(evidence))
	for i, hit := range evidence {
		contextBlocks = append(contextBlocks, fmt.Sprintf(
			"[source %d] %s:%d-%d\n%s",
			i+1,
			hit.Path,
			hit.StartLine,
			hit.EndLine,
			sanitizeUntrustedBlock(hit.Text),
		))
	}

	req := openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    "system",
				Content: localize.T("rag.prompt.answer_system", localize.Data{"Policy": localize.T("rag.prompt.shared_policy")}),
			},
			{
				Role:    "user",
				Content: localize.T("rag.prompt.answer_user", localize.Data{"Question": question, "Sources": strings.Join(contextBlocks, "\n\n---\n\n")}),
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

func sanitizeUntrustedBlock(raw string) string {
	lines := strings.Split(raw, "\n")
	safeLines := lines[:0]
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		lower := strings.ToLower(trimmed)
		if strings.HasPrefix(lower, "system:") || strings.HasPrefix(lower, "assistant:") || strings.HasPrefix(lower, "user:") {
			continue
		}
		if strings.Contains(lower, "ignore previous instructions") {
			continue
		}
		safeLines = append(safeLines, line)
	}
	return strings.TrimSpace(strings.Join(safeLines, "\n"))
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
