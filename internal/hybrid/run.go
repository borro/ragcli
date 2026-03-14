package hybrid

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	ragtools "github.com/borro/ragcli/internal/aitools/rag"
	"github.com/borro/ragcli/internal/input"
	"github.com/borro/ragcli/internal/llm"
	"github.com/borro/ragcli/internal/localize"
	"github.com/borro/ragcli/internal/retrieval"
	"github.com/borro/ragcli/internal/tools"
	"github.com/borro/ragcli/internal/verbose"
	openai "github.com/sashabaranov/go-openai"
)

const seedToolCallID = "hybrid-seed-search-rag"

func Run(
	ctx context.Context,
	client llm.ChatRequester,
	embedder llm.EmbeddingRequester,
	source input.FileBackedSource,
	opts Options,
	prompt string,
	plan *verbose.Plan,
) (string, error) {
	slog.Debug("hybrid processing started", "input_path", source.InputPath(), "file_count", source.FileCount())

	session, err := tools.PrepareSession(ctx, source, embedder, tools.Options{
		EnableRAG: true,
		RAG:       opts.Search,
	}, tools.SessionOptions{
		Prompt:                  prompt,
		AdditionalSystemPrompt:  strings.TrimSpace(localize.T("hybrid.prompt.system")),
		AdditionalUserPrompt:    strings.TrimSpace(localize.T("hybrid.prompt.user")),
		WarnOnFirstDirectAnswer: false,
	}, plan.Stage("index"))
	if err != nil {
		return "", err
	}

	retrievalMeter := plan.Stage("retrieval")
	retrievalMeter.Start(localize.T("progress.hybrid.seed_start"))
	seedToolCall, err := seedSearchToolCall(prompt)
	if err != nil {
		return "", err
	}
	retrievalMeter.Note(localize.T("progress.hybrid.seed_search"))
	seedResult, err := session.ExecuteToolCall(ctx, seedToolCall)
	if err != nil {
		return "", fmt.Errorf("failed to execute hybrid seed search: %w", err)
	}
	retrievalMeter.Done(localize.T("progress.hybrid.seed_done"))

	initialHistory := []openai.ChatCompletionMessage{
		{
			Role:      openai.ChatMessageRoleAssistant,
			ToolCalls: []openai.ToolCall{seedToolCall},
		},
		{
			Role:       openai.ChatMessageRoleTool,
			Name:       seedToolCall.Function.Name,
			Content:    seedResult,
			ToolCallID: seedToolCall.ID,
		},
	}

	result, err := session.Run(ctx, client, initialHistory, plan)
	if err != nil {
		return "", err
	}

	slog.Debug("hybrid processing finished", "citations", len(result.Citations))
	return retrieval.AppendSources(result.Answer, result.Citations), nil
}

func seedSearchToolCall(prompt string) (openai.ToolCall, error) {
	arguments, err := ragtools.MarshalJSON(ragtools.SearchParams{
		Query: strings.TrimSpace(prompt),
	})
	if err != nil {
		return openai.ToolCall{}, fmt.Errorf("failed to encode hybrid seed search arguments: %w", err)
	}

	return openai.ToolCall{
		ID:   seedToolCallID,
		Type: "function",
		Function: openai.FunctionCall{
			Name:      "search_rag",
			Arguments: arguments,
		},
	}, nil
}
