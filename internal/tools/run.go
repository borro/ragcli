package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/borro/ragcli/internal/aitools"
	"github.com/borro/ragcli/internal/aitools/files"
	"github.com/borro/ragcli/internal/input"
	"github.com/borro/ragcli/internal/llm"
	"github.com/borro/ragcli/internal/localize"
	"github.com/borro/ragcli/internal/verbose"
	openai "github.com/sashabaranov/go-openai"
)

const (
	toolsMaxTurns                 = 20
	toolsMaxConsecutiveNoProgress = 2
	toolsMaxConsecutiveDuplicates = 2
	emptyFinalAnswerRetryLimit    = 1
	stalledToolRetryLimit         = 1
	maxTurnFinalizationRetries    = 1
)

type toolsConfig struct {
	tools         []openai.Tool
	systemMessage openai.ChatCompletionMessage
	userQuestion  openai.ChatCompletionMessage
}

type orchestratorStats struct {
	uniqueLists      int
	uniqueSearches   int
	uniqueReads      int
	cacheHits        int
	duplicateCalls   int
	noProgressRuns   int
	llmCalls         int
	promptTokens     int
	completionTokens int
	totalTokens      int
}

type orchestrationError struct {
	Code    string
	Message string
	Details map[string]any
}

func (e orchestrationError) Error() string {
	return fmt.Sprintf("%s: %s", e.Code, e.Message)
}

type toolLoopState struct {
	registry              aitools.Registry
	seenLines             map[string]struct{}
	consecutiveNoProgress int
	consecutiveDuplicates int
	stats                 orchestratorStats
}

type toolExecutionMeta struct {
	Cached              bool   `json:"cached,omitempty"`
	Duplicate           bool   `json:"duplicate,omitempty"`
	AlreadySeen         bool   `json:"already_seen,omitempty"`
	NovelLines          int    `json:"novel_lines,omitempty"`
	ProgressHint        string `json:"progress_hint,omitempty"`
	SuggestedNextOffset int    `json:"suggested_next_offset,omitempty"`
}

func NewToolsConfig(prompt string, source input.FileBackedSource, definitions []openai.Tool) toolsConfig {
	systemMessage := openai.ChatCompletionMessage{
		Role:    "system",
		Content: localize.T("tools.prompt.system"),
	}

	userQuestion := openai.ChatCompletionMessage{
		Role:    "user",
		Content: buildToolsUserPrompt(prompt, source),
	}

	return toolsConfig{
		tools:         append([]openai.Tool(nil), definitions...),
		systemMessage: systemMessage,
		userQuestion:  userQuestion,
	}
}

func buildToolsUserPrompt(prompt string, source input.FileBackedSource) string {
	lines := []string{
		localize.T("tools.prompt.user.path_attached"),
	}
	if pathName := strings.TrimSpace(source.DisplayName()); pathName != "" {
		lines = append(lines, localize.T("tools.prompt.user.pathname", localize.Data{"Name": pathName}))
	}
	if source.IsMultiFile() {
		lines = append(lines, localize.T("tools.prompt.user.file_count", localize.Data{"Count": source.FileCount()}))
		lines = append(lines, localize.T("tools.prompt.user.list_hint"))
	} else {
		lines = append(lines, localize.T("tools.prompt.user.single_file_hint"))
		if source.Kind() == input.KindStdin {
			lines = append(lines, localize.T("tools.prompt.user.stdin_hint"))
		}
	}
	lines = append(lines, localize.T("tools.prompt.user.question", localize.Data{"Prompt": prompt}))
	return strings.Join(lines, "\n")
}

func Run(ctx context.Context, client llm.ChatRequester, source input.FileBackedSource, prompt string, plan *verbose.Plan) (string, error) {
	startedAt := time.Now()
	readerPath := source.SnapshotPath()
	if readerPath == "" {
		return "", errors.New("tools source path is required")
	}
	slog.Debug("tools processing started", "input_path", source.InputPath(), "reader_path", readerPath, "file_count", source.FileCount())
	initMeter := plan.Stage("init")
	loopMeter := plan.Stage("loop")
	finalMeter := plan.Stage("final")
	initMeter.Start(localize.T("progress.tools.session_start"))

	loopState := newToolLoopState(source)
	toolsConfig := NewToolsConfig(prompt, source, loopState.registry.ToolDefinitions())
	messages := make([]openai.ChatCompletionMessage, 0, 2)
	messages = append(messages, toolsConfig.systemMessage, toolsConfig.userQuestion)
	initMeter.Done(localize.T("progress.tools.session_done"))
	var toolChoiceToSend interface{} = "auto"
	emptyFinalAnswerRetries := 0
	stalledToolRetries := 0
	haveToolResults := false

	logStop := func(reason string, turns int) {
		slog.Debug("tools processing finished",
			"stop_reason", reason,
			"turns", turns,
			"duration", float64(time.Since(startedAt).Round(time.Millisecond))/float64(time.Second),
			"unique_lists", loopState.stats.uniqueLists,
			"unique_searches", loopState.stats.uniqueSearches,
			"unique_reads", loopState.stats.uniqueReads,
			"cache_hits", loopState.stats.cacheHits,
			"duplicate_calls", loopState.stats.duplicateCalls,
			"no_progress_runs", loopState.stats.noProgressRuns,
			"llm_calls", loopState.stats.llmCalls,
			"prompt_tokens", loopState.stats.promptTokens,
			"completion_tokens", loopState.stats.completionTokens,
			"total_tokens", loopState.stats.totalTokens,
		)
	}

	for turn := 1; turn <= toolsMaxTurns; turn++ {
		turnMeter := loopMeter.Slice(turn, toolsMaxTurns)
		turnMeter = turnMeter.WithStage(localize.T("progress.stage.loop"))
		turnMeter.Start(localize.T("progress.tools.turn_start", localize.Data{"Count": len(messages), "Turn": turn, "Total": toolsMaxTurns}))
		req := openai.ChatCompletionRequest{
			Messages:   messages,
			Tools:      toolsConfig.tools,
			ToolChoice: toolChoiceToSend,
		}

		select {
		case <-ctx.Done():
			logStop("context_done", turn-1)
			return "", ctx.Err()
		default:
		}

		slog.Debug("tools turn started",
			"turn", turn,
			"message_count", len(messages),
			"tool_choice", req.ToolChoice,
		)

		resp, metrics, err := client.SendRequestWithMetrics(ctx, req)
		if err != nil {
			logStop("request_error", turn)
			return "", fmt.Errorf("failed to send request: %w", err)
		}
		loopState.stats.llmCalls++
		loopState.stats.promptTokens += metrics.PromptTokens
		loopState.stats.completionTokens += metrics.CompletionTokens
		loopState.stats.totalTokens += metrics.TotalTokens

		if len(resp.Choices) == 0 {
			logStop("empty_choices", turn)
			return "", fmt.Errorf("no choices in response")
		}

		choice := resp.Choices[0]
		messages = append(messages, choice.Message)

		slog.Debug("tools turn received llm response",
			"turn", turn,
			"role", choice.Message.Role,
			"tool_calls_count", len(choice.Message.ToolCalls),
			"has_content", choice.Message.Content != "",
		)
		if len(choice.Message.ToolCalls) > 0 {
			turnMeter.Note(localize.T("progress.tools.calls_requested", localize.Data{"Count": len(choice.Message.ToolCalls)}))
		} else {
			finalMeter.Start(localize.T("progress.tools.text_answer", localize.Data{"Turn": turn}))
		}

		if turn == 1 && readerPath != "" && len(choice.Message.ToolCalls) == 0 && strings.TrimSpace(choice.Message.Content) != "" {
			slog.Warn("tools backend returned a direct answer without using file tools",
				"turn", turn,
				"file_path", source.InputPath(),
				"reader_path", readerPath,
				"message_role", choice.Message.Role,
			)
		}

		if len(choice.Message.ToolCalls) > 0 {
			slog.Debug("received tool calls from model", "turn", turn, "count", len(choice.Message.ToolCalls))
			toolMeter := turnMeter.WithStage(localize.T("progress.tools.call_stage"))
			toolMeter.Start(localize.T("progress.tools.executing_calls", localize.Data{"Count": len(choice.Message.ToolCalls)}))

			results, err := loopState.executeToolCalls(ctx, choice.Message.ToolCalls, toolMeter)
			slog.Debug("tool calls finished", "turn", turn, "result_count", len(results), "has_error", err != nil)
			for i, result := range results {
				messages = append(messages, openai.ChatCompletionMessage{
					Role:       "tool",
					Name:       choice.Message.ToolCalls[i].Function.Name,
					Content:    result,
					ToolCallID: choice.Message.ToolCalls[i].ID,
				})
			}
			haveToolResults = haveToolResults || len(results) > 0
			if err != nil {
				var orchErr orchestrationError
				if errorsAsOrchestration(err, &orchErr) && shouldRetryWithoutTools(orchErr) && stalledToolRetries < stalledToolRetryLimit {
					stalledToolRetries++
					slog.Warn("tool loop stalled, retrying without tools",
						"turn", turn,
						"code", orchErr.Code,
						"attempt", stalledToolRetries,
					)
					messages = append(messages, openai.ChatCompletionMessage{
						Role:    "user",
						Content: localize.T("tools.prompt.stop_calls"),
					})
					toolChoiceToSend = "none"
					continue
				}
				if errorsAsOrchestration(err, &orchErr) {
					logStop(orchErr.Code, turn)
				} else {
					logStop("tool_error", turn)
				}
				return "", fmt.Errorf("failed to execute tool calls: %w", err)
			}

			toolMeter.Done(localize.T("progress.tools.calls_done", localize.Data{"Count": len(results)}))

			continue
		}

		slog.Debug("received final answer from model", "turn", turn)
		if choice.Message.Content == "" {
			if emptyFinalAnswerRetries < emptyFinalAnswerRetryLimit {
				emptyFinalAnswerRetries++
				slog.Warn("received empty final answer from model, retrying without tools", "attempt", emptyFinalAnswerRetries)
				messages = append(messages, openai.ChatCompletionMessage{
					Role:    "user",
					Content: localize.T("tools.prompt.final_retry"),
				})
				toolChoiceToSend = "none"
				continue
			}
			logStop("empty_final_answer", turn)
			return "", fmt.Errorf("received final response without content")
		}
		finalMeter.Done(localize.T("progress.tools.final_done", localize.Data{"Turn": turn}))
		logStop("final_answer", turn)
		return choice.Message.Content, nil
	}

	if haveToolResults {
		finalMeter.Start(localize.T("progress.tools.text_answer", localize.Data{"Turn": toolsMaxTurns + 1}))
		result, err := requestFinalAnswerWithoutTools(ctx, client, messages, toolsConfig.tools, maxTurnFinalizationRetries)
		if err == nil {
			finalMeter.Done(localize.T("progress.tools.final_done", localize.Data{"Turn": toolsMaxTurns + 1}))
			logStop("forced_final_answer", toolsMaxTurns)
			return result, nil
		}
		logStop("orchestration_limit_finalization_failed", toolsMaxTurns)
		return "", err
	}

	logStop("orchestration_limit", toolsMaxTurns)
	return "", orchestrationError{
		Code:    "orchestration_limit",
		Message: fmt.Sprintf("tools orchestration exceeded %d turns without final answer", toolsMaxTurns),
		Details: map[string]any{"max_turns": toolsMaxTurns},
	}
}

func newToolLoopState(source input.FileBackedSource) *toolLoopState {
	var reader files.LineReader
	if source.IsMultiFile() {
		corpusFiles := make([]files.CorpusFile, 0, source.FileCount())
		for _, file := range source.BackingFiles() {
			corpusFiles = append(corpusFiles, files.CorpusFile{
				Path:        file.Path,
				DisplayPath: file.DisplayPath,
			})
		}
		reader = files.NewCorpusReader(corpusFiles)
	} else {
		if source.Kind() == input.KindStdin {
			reader = files.NewNamedFileReader(source.SnapshotPath(), source.DisplayName())
		} else {
			reader = files.NewFileReader(source.SnapshotPath())
		}
	}
	toolset := files.NewTools(reader, files.ToolOptions{MultiFile: source.IsMultiFile()})
	return &toolLoopState{
		registry:  aitools.NewRegistry(toolset...),
		seenLines: make(map[string]struct{}),
	}
}

func summarizeToolMeta(name string, meta toolExecutionMeta, index int, total int) string {
	base := fmt.Sprintf("%s (%d/%d)", strings.TrimSpace(name), index, total)
	switch {
	case meta.NovelLines > 0:
		return localize.T("progress.tools.meta_novel", localize.Data{"Base": base, "Count": meta.NovelLines})
	case meta.Cached:
		return localize.T("progress.tools.meta_cached", localize.Data{"Base": base})
	case meta.AlreadySeen:
		return localize.T("progress.tools.meta_seen", localize.Data{"Base": base})
	case meta.Duplicate:
		return localize.T("progress.tools.meta_duplicate", localize.Data{"Base": base})
	default:
		return localize.T("progress.tools.meta_done", localize.Data{"Base": base})
	}
}

func (s *toolLoopState) executeToolCalls(ctx context.Context, toolCalls []openai.ToolCall, meter verbose.Meter) ([]string, error) {
	results := make([]string, 0, len(toolCalls))
	turnHadProgress := false
	turnAllDuplicates := len(toolCalls) > 0

	for index, call := range toolCalls {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		result, meta, err := s.executeToolCall(ctx, call)
		if err != nil {
			slog.Error("executeTool error", "tool_name", call.Function.Name, "error", err)
			toolErr, marshalErr := files.MarshalJSON(files.AsToolError(err))
			if marshalErr != nil {
				return nil, marshalErr
			}
			results = append(results, toolErr)
			turnAllDuplicates = false
			meter.Report(index+1, len(toolCalls), fmt.Sprintf("%s (%d/%d): ошибка", strings.TrimSpace(call.Function.Name), index+1, len(toolCalls)))
			continue
		}

		if meta.NovelLines > 0 {
			turnHadProgress = true
		}
		if !meta.Duplicate {
			turnAllDuplicates = false
		}

		meter.Report(index+1, len(toolCalls), summarizeToolMeta(call.Function.Name, meta, index+1, len(toolCalls)))
		results = append(results, result)
	}

	if !turnHadProgress {
		s.consecutiveNoProgress++
		s.stats.noProgressRuns++
	} else {
		s.consecutiveNoProgress = 0
	}

	if turnAllDuplicates {
		s.consecutiveDuplicates++
	} else {
		s.consecutiveDuplicates = 0
	}

	if s.consecutiveDuplicates >= toolsMaxConsecutiveDuplicates {
		return results, orchestrationError{
			Code:    "duplicate_call",
			Message: "model repeated duplicate tool calls without new information",
			Details: map[string]any{"threshold": toolsMaxConsecutiveDuplicates},
		}
	}

	if s.consecutiveNoProgress >= toolsMaxConsecutiveNoProgress {
		return results, orchestrationError{
			Code:    "no_progress",
			Message: "tool loop stopped because repeated calls produced no new lines",
			Details: map[string]any{"threshold": toolsMaxConsecutiveNoProgress},
		}
	}

	return results, nil
}

func (s *toolLoopState) executeToolCall(ctx context.Context, call openai.ToolCall) (string, toolExecutionMeta, error) {
	result, err := s.registry.Execute(ctx, call)
	if err != nil {
		return "", toolExecutionMeta{}, err
	}

	if result.Cached {
		s.stats.cacheHits++
		s.stats.duplicateCalls++

		meta := toolExecutionMeta{
			Cached:       true,
			Duplicate:    result.Duplicate,
			AlreadySeen:  true,
			NovelLines:   0,
			ProgressHint: "duplicate tool call; reuse cached result and change strategy instead of repeating the same request",
		}
		if suggested := intHint(result.Hints, files.HintSuggestedNextOffset); suggested > 0 {
			meta.SuggestedNextOffset = suggested
		}

		annotated, err := annotateToolPayload(result.Payload, meta)
		if err != nil {
			return "", toolExecutionMeta{}, err
		}
		return annotated, meta, nil
	}

	lineKeys := make(map[string]struct{}, len(result.ProgressKeys))
	for _, key := range result.ProgressKeys {
		lineKeys[key] = struct{}{}
	}
	novelLines := 0
	for key := range lineKeys {
		if _, seen := s.seenLines[key]; seen {
			continue
		}
		s.seenLines[key] = struct{}{}
		novelLines++
	}

	meta := toolExecutionMeta{
		AlreadySeen: novelLines == 0,
		NovelLines:  novelLines,
	}
	if novelLines == 0 {
		meta.ProgressHint = "tool call returned only already seen lines; broaden or change the search strategy"
	} else {
		meta.ProgressHint = "new lines discovered; read local context or answer if evidence is sufficient"
	}
	if suggested := intHint(result.Hints, files.HintSuggestedNextOffset); suggested > 0 {
		meta.SuggestedNextOffset = suggested
	}

	annotated, err := annotateToolPayload(result.Payload, meta)
	if err != nil {
		return "", toolExecutionMeta{}, err
	}

	switch call.Function.Name {
	case "list_files":
		s.stats.uniqueLists++
	case "search_file":
		s.stats.uniqueSearches++
	case "read_lines", "read_around":
		s.stats.uniqueReads++
	}

	return annotated, meta, nil
}

func annotateToolPayload(raw string, meta toolExecutionMeta) (string, error) {
	if raw == "" {
		return raw, nil
	}

	var payload map[string]any
	if err := json.Unmarshal([]byte(raw), &payload); err != nil {
		return "", fmt.Errorf("failed to decode tool payload for annotation: %w", err)
	}

	if meta.Cached {
		payload["cached"] = true
	}
	if meta.Duplicate {
		payload["duplicate"] = true
	}
	if meta.AlreadySeen {
		payload["already_seen"] = true
	}
	payload["novel_lines"] = meta.NovelLines
	if meta.ProgressHint != "" {
		payload["progress_hint"] = meta.ProgressHint
	}
	if meta.SuggestedNextOffset > 0 {
		payload["suggested_next_offset"] = meta.SuggestedNextOffset
	}

	return files.MarshalJSON(payload)
}

func errorsAsOrchestration(err error, target *orchestrationError) bool {
	return errors.As(err, target)
}

func intHint(hints map[string]any, key string) int {
	if hints == nil {
		return 0
	}
	value, ok := hints[key]
	if !ok {
		return 0
	}
	switch typed := value.(type) {
	case int:
		return typed
	case int32:
		return int(typed)
	case int64:
		return int(typed)
	case float64:
		return int(typed)
	default:
		return 0
	}
}

func shouldRetryWithoutTools(err orchestrationError) bool {
	switch err.Code {
	case "duplicate_call", "no_progress":
		return true
	default:
		return false
	}
}

func requestFinalAnswerWithoutTools(ctx context.Context, client llm.ChatRequester, messages []openai.ChatCompletionMessage, tools []openai.Tool, retries int) (string, error) {
	attemptMessages := append([]openai.ChatCompletionMessage{}, messages...)

	for attempt := 0; attempt <= retries; attempt++ {
		attemptMessages = append(attemptMessages, openai.ChatCompletionMessage{
			Role:    "user",
			Content: localize.T("tools.prompt.stop_calls"),
		})

		resp, _, err := client.SendRequestWithMetrics(ctx, openai.ChatCompletionRequest{
			Messages:   attemptMessages,
			Tools:      tools,
			ToolChoice: "none",
		})
		if err != nil {
			return "", fmt.Errorf("failed to send forced finalization request: %w", err)
		}
		if len(resp.Choices) == 0 {
			return "", fmt.Errorf("forced finalization returned no choices")
		}

		choice := resp.Choices[0]
		attemptMessages = append(attemptMessages, choice.Message)
		if strings.TrimSpace(choice.Message.Content) != "" {
			return choice.Message.Content, nil
		}
	}

	return "", orchestrationError{
		Code:    "orchestration_limit",
		Message: fmt.Sprintf("tools orchestration exceeded %d turns and forced finalization returned no answer", toolsMaxTurns),
		Details: map[string]any{"max_turns": toolsMaxTurns},
	}
}
