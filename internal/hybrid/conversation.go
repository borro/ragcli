package hybrid

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	aitoolseed "github.com/borro/ragcli/internal/aitools/seed"
	"github.com/borro/ragcli/internal/input"
	"github.com/borro/ragcli/internal/llm"
	"github.com/borro/ragcli/internal/localize"
	"github.com/borro/ragcli/internal/retrieval"
	"github.com/borro/ragcli/internal/tools"
	"github.com/borro/ragcli/internal/verbose"
	openai "github.com/sashabaranov/go-openai"
)

type Conversation struct {
	direct *directConversation
	tools  *tools.Conversation
}

type directConversation struct {
	client        llm.ChatAutoContextRequester
	source        input.FileBackedSource
	systemMessage openai.ChatCompletionMessage
	rawContent    string
	history       []openai.ChatCompletionMessage
	baseline      []openai.ChatCompletionMessage
}

func StartConversation(
	ctx context.Context,
	client llm.ChatAutoContextRequester,
	embedder llm.EmbeddingRequester,
	source input.FileBackedSource,
	opts Options,
	prompt string,
	plan *verbose.Plan,
) (*Conversation, string, error) {
	slog.Debug("hybrid processing started", "input_path", source.InputPath(), "file_count", source.FileCount())

	if direct, answer, ok, err := startDirectConversation(ctx, client, source, prompt, plan); err != nil {
		return nil, "", err
	} else if ok {
		slog.Debug("hybrid processing finished", "mode", "direct_context")
		return &Conversation{direct: direct}, answer, nil
	}

	session, err := tools.PrepareSession(ctx, source, embedder, tools.Options{
		EnableRAG: true,
		RAG:       opts.Search,
	}, tools.SessionOptions{
		Prompt:                 prompt,
		AdditionalSystemPrompt: strings.TrimSpace(localize.T("hybrid.prompt.system")),
		AdditionalUserPrompt:   strings.TrimSpace(localize.T("hybrid.prompt.user")),
		FirstTurnAnswerPolicy:  tools.FirstTurnAnswerAllow,
	}, plan.Stage("index"))
	if err != nil {
		return nil, "", err
	}

	retrievalMeter := plan.Stage("retrieval")
	retrievalMeter.Start(localize.T("progress.hybrid.seed_start"))
	retrievalMeter.Note(localize.T("progress.hybrid.seed_search"))
	seedBundle, err := aitoolseed.BuildSeedBundle(ctx, session, prompt, opts.Search)
	if err != nil {
		return nil, "", err
	}
	if seedBundle.Weak {
		session.SetFirstTurnAnswerPolicy(tools.FirstTurnAnswerRequireToolUse)
	}
	retrievalMeter.Done(localize.T("progress.hybrid.seed_done"))

	toolConversation, answer, err := tools.StartConversationWithSession(ctx, client, session, seedBundle.History, plan)
	if err != nil {
		return nil, "", err
	}
	slog.Debug("hybrid processing finished", "mode", "tools", "seed_hits", len(seedBundle.Hits), "weak_seed", seedBundle.Weak)
	return &Conversation{tools: toolConversation}, retrieval.AppendEvidenceSources(answer, toolConversation.SessionEvidence()), nil
}

func (c *Conversation) Ask(ctx context.Context, prompt string, plan *verbose.Plan) (string, error) {
	switch {
	case c == nil:
		return "", fmt.Errorf("hybrid conversation is not prepared")
	case c.direct != nil:
		return c.direct.Ask(ctx, prompt, plan)
	case c.tools != nil:
		answer, err := c.tools.Ask(ctx, prompt, plan)
		if err != nil {
			return "", err
		}
		return retrieval.AppendEvidenceSources(answer, c.tools.SessionEvidence()), nil
	default:
		return "", fmt.Errorf("hybrid conversation is not prepared")
	}
}

func (c *Conversation) FollowUpPlan(reporter verbose.Reporter) *verbose.Plan {
	switch {
	case c == nil:
		return verbose.NewPlan(reporter, localize.T("progress.mode.hybrid"))
	case c.direct != nil:
		return verbose.NewPlan(reporter, localize.T("progress.mode.hybrid"),
			verbose.StageDef{Key: "init", Label: localize.T("progress.stage.init"), Slots: 2},
			verbose.StageDef{Key: "final", Label: localize.T("progress.stage.final"), Slots: 2},
		)
	case c.tools != nil:
		return verbose.NewPlan(reporter, localize.T("progress.mode.hybrid"),
			verbose.StageDef{Key: "init", Label: localize.T("progress.stage.init"), Slots: 2},
			verbose.StageDef{Key: "loop", Label: localize.T("progress.stage.loop"), Slots: 10},
			verbose.StageDef{Key: "final", Label: localize.T("progress.stage.final"), Slots: 2},
		)
	default:
		return verbose.NewPlan(reporter, localize.T("progress.mode.hybrid"))
	}
}

func (c *Conversation) Reset() {
	if c == nil {
		return
	}
	if c.direct != nil {
		c.direct.Reset()
	}
	if c.tools != nil {
		c.tools.Reset()
	}
}

func (c *directConversation) Ask(ctx context.Context, prompt string, plan *verbose.Plan) (string, error) {
	initMeter := plan.Stage("init")
	finalMeter := plan.Stage("final")

	initMeter.Start(localize.T("progress.hybrid.direct_followup_request_start"))

	messages := make([]openai.ChatCompletionMessage, 0, len(c.history)+2)
	messages = append(messages, c.systemMessage)
	messages = append(messages, cloneMessages(c.history)...)
	messages = append(messages, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: strings.TrimSpace(prompt),
	})

	resp, _, err := c.client.SendRequestWithMetrics(ctx, openai.ChatCompletionRequest{Messages: messages})
	if err != nil {
		return "", fmt.Errorf("failed to send hybrid direct-context request: %w", err)
	}
	initMeter.Done(localize.T("progress.hybrid.direct_followup_request_done"))
	finalMeter.Start(localize.T("progress.hybrid.direct_followup_answer_start"))
	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("no choices in response")
	}
	finalMeter.Done(localize.T("progress.hybrid.direct_followup_answer_done"))
	answer := appendDirectContextSources(strings.TrimSpace(resp.Choices[0].Message.Content), c.source, c.rawContent)
	c.history = append(c.history, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: strings.TrimSpace(prompt),
	}, resp.Choices[0].Message)
	return answer, nil
}

func (c *directConversation) Reset() {
	if c == nil {
		return
	}
	c.history = cloneMessages(c.baseline)
}

func cloneMessages(messages []openai.ChatCompletionMessage) []openai.ChatCompletionMessage {
	if len(messages) == 0 {
		return nil
	}
	cloned := make([]openai.ChatCompletionMessage, 0, len(messages))
	for _, message := range messages {
		cloned = append(cloned, openai.ChatCompletionMessage{
			Role:       message.Role,
			Content:    message.Content,
			Name:       message.Name,
			ToolCallID: message.ToolCallID,
			ToolCalls:  append([]openai.ToolCall(nil), message.ToolCalls...),
		})
	}
	return cloned
}
