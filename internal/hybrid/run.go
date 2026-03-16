package hybrid

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"strings"
	"unicode/utf8"

	"github.com/borro/ragcli/internal/input"
	"github.com/borro/ragcli/internal/llm"
	"github.com/borro/ragcli/internal/localize"
	"github.com/borro/ragcli/internal/retrieval"
	"github.com/borro/ragcli/internal/verbose"
	openai "github.com/sashabaranov/go-openai"
)

func Run(
	ctx context.Context,
	client llm.ChatAutoContextRequester,
	embedder llm.EmbeddingRequester,
	source input.FileBackedSource,
	opts Options,
	prompt string,
	plan *verbose.Plan,
) (string, error) {
	_, answer, err := StartConversation(ctx, client, embedder, source, opts, prompt, plan)
	return answer, err
}

const (
	approxMessageOverheadTokens = 12
	approxRequestReserve        = 64
	directBudgetNumerator       = 3
	directBudgetDenominator     = 5
)

func startDirectConversation(ctx context.Context, client llm.ChatAutoContextRequester, source input.FileBackedSource, prompt string, plan *verbose.Plan) (*directConversation, string, bool, error) {
	if source.IsMultiFile() {
		slog.Debug("hybrid direct context skipped; source is multi-file", "input_path", source.InputPath(), "file_count", source.FileCount())
		return nil, "", false, nil
	}
	indexMeter := plan.Stage("index")
	retrievalMeter := plan.Stage("retrieval")
	initMeter := plan.Stage("init")
	loopMeter := plan.Stage("loop")
	finalMeter := plan.Stage("final")

	slog.Debug("hybrid direct context eligibility check started",
		"input_path", source.InputPath(),
		"display_name", source.DisplayName(),
		"file_count", source.FileCount(),
	)
	indexMeter.Start(localize.T("progress.hybrid.direct_check_start"))

	reader, err := source.Open()
	if err != nil {
		return nil, "", false, err
	}
	defer func() { _ = reader.Close() }()

	raw, err := io.ReadAll(reader)
	if err != nil {
		return nil, "", false, fmt.Errorf("failed to read source for hybrid direct context: %w", err)
	}
	slog.Debug("hybrid direct context source loaded",
		"bytes", len(raw),
		"approx_tokens", approxTokenCount(string(raw)),
	)

	contextLength, err := client.ResolveAutoContextLength(ctx)
	if err != nil || contextLength < 1 {
		indexMeter.Done(localize.T("progress.hybrid.direct_check_skip"))
		slog.Debug("hybrid direct context skipped; context length unavailable", "error", err)
		return nil, "", false, nil
	}

	messages := directContextMessages(prompt, source.DisplayName(), string(raw))
	requestTokens := estimateDirectRequestTokens(messages)
	safeBudget := effectiveDirectApproxTokenBudget(contextLength)
	slog.Debug("hybrid direct context budget estimated",
		"context_length", contextLength,
		"safe_budget", safeBudget,
		"request_tokens", requestTokens,
	)
	if requestTokens > safeBudget {
		indexMeter.Done(localize.T("progress.hybrid.direct_check_skip"))
		slog.Debug("hybrid direct context skipped; input exceeds safe budget",
			"context_length", contextLength,
			"safe_budget", safeBudget,
			"request_tokens", requestTokens,
		)
		return nil, "", false, nil
	}
	indexMeter.Done(localize.T("progress.hybrid.direct_check_done"))
	slog.Debug("hybrid direct context selected",
		"context_length", contextLength,
		"safe_budget", safeBudget,
		"request_tokens", requestTokens,
		"message_count", len(messages),
	)
	retrievalMeter.Done(localize.T("progress.hybrid.direct_retrieval_skip"))
	initMeter.Start(localize.T("progress.hybrid.direct_request_start"))

	slog.Debug("hybrid direct context request started", "message_count", len(messages))
	resp, _, err := client.SendRequestWithMetrics(ctx, openai.ChatCompletionRequest{Messages: messages})
	if err != nil {
		return nil, "", false, fmt.Errorf("failed to send hybrid direct-context request: %w", err)
	}
	initMeter.Done(localize.T("progress.hybrid.direct_request_done"))
	loopMeter.Done(localize.T("progress.hybrid.direct_loop_skip"))
	finalMeter.Start(localize.T("progress.hybrid.direct_answer_start"))
	if len(resp.Choices) == 0 {
		return nil, "", false, fmt.Errorf("no choices in response")
	}
	finalMeter.Done(localize.T("progress.hybrid.direct_answer_done"))
	answer := appendDirectContextSources(strings.TrimSpace(resp.Choices[0].Message.Content), source, string(raw))
	slog.Debug("hybrid direct context request finished",
		"choice_count", len(resp.Choices),
		"answer_chars", len(answer),
	)
	return &directConversation{
		client:        client,
		source:        source,
		systemMessage: messages[0],
		rawContent:    string(raw),
		history: []openai.ChatCompletionMessage{
			messages[1],
			resp.Choices[0].Message,
		},
		baseline: []openai.ChatCompletionMessage{
			messages[1],
			resp.Choices[0].Message,
		},
	}, answer, true, nil
}

func directContextMessages(question, name, content string) []openai.ChatCompletionMessage {
	safeQuestion := strings.TrimSpace(question)
	safeName := strings.TrimSpace(name)
	safeContent := llm.SanitizeUntrustedBlock(content)
	userContent := strings.Join([]string{
		localize.T("hybrid.prompt.direct.user_intro"),
		localize.T("hybrid.prompt.direct.path", localize.Data{"Name": safeName}),
		fmt.Sprintf("%s\n%s", localize.T("hybrid.prompt.direct.question_label"), safeQuestion),
		fmt.Sprintf("%s\n```\n%s\n```", localize.T("hybrid.prompt.direct.content_label"), safeContent),
	}, "\n\n")
	return []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: strings.TrimSpace(localize.T("hybrid.prompt.direct.system"))},
		{Role: openai.ChatMessageRoleUser, Content: userContent},
	}
}

func estimateDirectRequestTokens(messages []openai.ChatCompletionMessage) int {
	total := approxRequestReserve
	for _, message := range messages {
		total += approxMessageOverheadTokens
		total += approxTokenCount(message.Role)
		total += approxTokenCount(message.Content)
	}
	return total
}

func effectiveDirectApproxTokenBudget(maxTokens int) int {
	if maxTokens <= 0 {
		return 0
	}
	budget := (maxTokens * directBudgetNumerator) / directBudgetDenominator
	if budget < 1 {
		return 1
	}
	return budget
}

func approxTokenCount(text string) int {
	if text == "" {
		return 0
	}
	return (utf8.RuneCountInString(text) + 2) / 3
}

func appendDirectContextSources(answer string, source input.FileBackedSource, content string) string {
	lineCount := countLines(content)
	if lineCount < 1 {
		return retrieval.AppendSources(answer, nil)
	}
	return retrieval.AppendSources(answer, []retrieval.Citation{{
		SourcePath: source.DisplayName(),
		StartLine:  1,
		EndLine:    lineCount,
	}})
}

func countLines(text string) int {
	if text == "" {
		return 0
	}
	normalized := strings.ReplaceAll(text, "\r\n", "\n")
	return strings.Count(normalized, "\n") + 1
}
