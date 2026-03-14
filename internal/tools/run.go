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
	ragtools "github.com/borro/ragcli/internal/aitools/rag"
	"github.com/borro/ragcli/internal/input"
	"github.com/borro/ragcli/internal/llm"
	"github.com/borro/ragcli/internal/localize"
	ragruntime "github.com/borro/ragcli/internal/rag"
	"github.com/borro/ragcli/internal/retrieval"
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

type SessionOptions struct {
	Prompt                  string
	AdditionalSystemPrompt  string
	AdditionalUserPrompt    string
	WarnOnFirstDirectAnswer bool
}

type Session struct {
	config                  toolsConfig
	state                   *toolLoopState
	inputPath               string
	readerPath              string
	warnOnFirstDirectAnswer bool
}

type SessionResult struct {
	Answer    string
	Citations []retrieval.Citation
}

type orchestratorStats struct {
	uniqueLists      int
	uniqueSearches   int
	uniqueRAGSearch  int
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
	seenLinesByDomain     map[string]map[string]struct{}
	citations             []retrieval.Citation
	seenCitations         map[string]struct{}
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

func NewToolsConfig(
	prompt string,
	source input.FileBackedSource,
	opts Options,
	definitions []openai.Tool,
	additionalSystemPrompt string,
	additionalUserPrompt string,
) toolsConfig {
	systemMessage := openai.ChatCompletionMessage{
		Role:    "system",
		Content: buildToolsSystemPrompt(opts, additionalSystemPrompt),
	}

	userQuestion := openai.ChatCompletionMessage{
		Role:    "user",
		Content: buildToolsUserPrompt(prompt, source, opts, additionalUserPrompt),
	}

	return toolsConfig{
		tools:         append([]openai.Tool(nil), definitions...),
		systemMessage: systemMessage,
		userQuestion:  userQuestion,
	}
}

func buildToolsSystemPrompt(opts Options, additionalPrompt string) string {
	lines := []string{strings.TrimSpace(localize.T("tools.prompt.system"))}
	if opts.EnableRAG {
		lines = append(lines, strings.TrimSpace(localize.T("tools.prompt.rag_enabled")))
	}
	if extra := strings.TrimSpace(additionalPrompt); extra != "" {
		lines = append(lines, extra)
	}
	return strings.Join(lines, "\n\n")
}

func buildToolsUserPrompt(prompt string, source input.FileBackedSource, opts Options, additionalPrompt string) string {
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
	if opts.EnableRAG {
		lines = append(lines, localize.T("tools.prompt.user.rag_hint"))
	}
	if extra := strings.TrimSpace(additionalPrompt); extra != "" {
		lines = append(lines, extra)
	}
	lines = append(lines, localize.T("tools.prompt.user.question", localize.Data{"Prompt": prompt}))
	return strings.Join(lines, "\n")
}

func Run(ctx context.Context, client llm.ChatRequester, embedder llm.EmbeddingRequester, source input.FileBackedSource, opts Options, prompt string, plan *verbose.Plan) (string, error) {
	readerPath := source.SnapshotPath()
	if readerPath == "" {
		return "", errors.New("tools source path is required")
	}
	slog.Debug("tools processing started", "input_path", source.InputPath(), "reader_path", readerPath, "file_count", source.FileCount())
	indexMeter := plan.Stage("index")
	session, err := PrepareSession(ctx, source, embedder, opts, SessionOptions{
		Prompt:                  prompt,
		WarnOnFirstDirectAnswer: true,
	}, indexMeter)
	if err != nil {
		return "", err
	}
	result, err := session.Run(ctx, client, nil, plan)
	if err != nil {
		return "", err
	}
	return result.Answer, nil
}

func PrepareSession(
	ctx context.Context,
	source input.FileBackedSource,
	embedder llm.EmbeddingRequester,
	opts Options,
	sessionOpts SessionOptions,
	indexMeter verbose.Meter,
) (*Session, error) {
	readerPath := source.SnapshotPath()
	if readerPath == "" {
		return nil, errors.New("tools source path is required")
	}

	loopState, err := newToolLoopState(ctx, source, embedder, opts, indexMeter)
	if err != nil {
		return nil, err
	}

	config := NewToolsConfig(
		sessionOpts.Prompt,
		source,
		opts,
		loopState.registry.ToolDefinitions(),
		sessionOpts.AdditionalSystemPrompt,
		sessionOpts.AdditionalUserPrompt,
	)

	return &Session{
		config:                  config,
		state:                   loopState,
		inputPath:               source.InputPath(),
		readerPath:              readerPath,
		warnOnFirstDirectAnswer: sessionOpts.WarnOnFirstDirectAnswer,
	}, nil
}

func (s *Session) InitialMessages() []openai.ChatCompletionMessage {
	return []openai.ChatCompletionMessage{
		s.config.systemMessage,
		s.config.userQuestion,
	}
}

func (s *Session) ExecuteToolCall(ctx context.Context, call openai.ToolCall) (string, error) {
	if s == nil || s.state == nil {
		return "", errors.New("tools session is not prepared")
	}
	result, _, err := s.state.executeToolCall(ctx, call)
	if err != nil {
		return "", err
	}
	return result, nil
}

func (s *Session) Run(ctx context.Context, client llm.ChatRequester, initialHistory []openai.ChatCompletionMessage, plan *verbose.Plan) (SessionResult, error) {
	if s == nil || s.state == nil {
		return SessionResult{}, errors.New("tools session is not prepared")
	}

	startedAt := time.Now()
	initMeter := plan.Stage("init")
	loopMeter := plan.Stage("loop")
	finalMeter := plan.Stage("final")

	initMeter.Start(localize.T("progress.tools.session_start"))
	messages := append(s.InitialMessages(), cloneMessages(initialHistory)...)
	initMeter.Done(localize.T("progress.tools.session_done"))

	var toolChoiceToSend interface{} = "auto"
	emptyFinalAnswerRetries := 0
	stalledToolRetries := 0
	haveToolResults := containsToolResponses(initialHistory)

	logStop := func(reason string, turns int) {
		slog.Debug("tool session finished",
			"stop_reason", reason,
			"turns", turns,
			"duration", float64(time.Since(startedAt).Round(time.Millisecond))/float64(time.Second),
			"unique_lists", s.state.stats.uniqueLists,
			"unique_searches", s.state.stats.uniqueSearches,
			"unique_rag_searches", s.state.stats.uniqueRAGSearch,
			"unique_reads", s.state.stats.uniqueReads,
			"cache_hits", s.state.stats.cacheHits,
			"duplicate_calls", s.state.stats.duplicateCalls,
			"no_progress_runs", s.state.stats.noProgressRuns,
			"llm_calls", s.state.stats.llmCalls,
			"prompt_tokens", s.state.stats.promptTokens,
			"completion_tokens", s.state.stats.completionTokens,
			"total_tokens", s.state.stats.totalTokens,
		)
	}

	for turn := 1; turn <= toolsMaxTurns; turn++ {
		turnMeter := loopMeter.Slice(turn, toolsMaxTurns)
		turnMeter = turnMeter.WithStage(localize.T("progress.stage.loop"))
		turnMeter.Start(localize.T("progress.tools.turn_start", localize.Data{"Count": len(messages), "Turn": turn, "Total": toolsMaxTurns}))
		req := openai.ChatCompletionRequest{
			Messages:   messages,
			Tools:      s.config.tools,
			ToolChoice: toolChoiceToSend,
		}

		select {
		case <-ctx.Done():
			logStop("context_done", turn-1)
			return SessionResult{}, ctx.Err()
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
			return SessionResult{}, fmt.Errorf("failed to send request: %w", err)
		}
		s.state.stats.llmCalls++
		s.state.stats.promptTokens += metrics.PromptTokens
		s.state.stats.completionTokens += metrics.CompletionTokens
		s.state.stats.totalTokens += metrics.TotalTokens

		if len(resp.Choices) == 0 {
			logStop("empty_choices", turn)
			return SessionResult{}, fmt.Errorf("no choices in response")
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
		}

		if s.warnOnFirstDirectAnswer && turn == 1 && s.readerPath != "" && len(choice.Message.ToolCalls) == 0 && strings.TrimSpace(choice.Message.Content) != "" {
			slog.Warn("tools backend returned a direct answer without using file tools",
				"turn", turn,
				"file_path", s.inputPath,
				"reader_path", s.readerPath,
				"message_role", choice.Message.Role,
			)
		}

		if len(choice.Message.ToolCalls) > 0 {
			slog.Debug("received tool calls from model", "turn", turn, "count", len(choice.Message.ToolCalls))
			toolMeter := turnMeter.WithStage(localize.T("progress.tools.call_stage"))
			toolMeter.Start(localize.T("progress.tools.executing_calls", localize.Data{"Count": len(choice.Message.ToolCalls)}))

			results, err := s.state.executeToolCalls(ctx, choice.Message.ToolCalls, toolMeter)
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
				return SessionResult{}, fmt.Errorf("failed to execute tool calls: %w", err)
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
			return SessionResult{}, fmt.Errorf("received final response without content")
		}
		finalMeter.Start(localize.T("progress.tools.text_answer", localize.Data{"Turn": turn}))
		finalMeter.Done(localize.T("progress.tools.final_done", localize.Data{"Turn": turn}))
		logStop("final_answer", turn)
		return SessionResult{
			Answer:    choice.Message.Content,
			Citations: s.state.Citations(),
		}, nil
	}

	if haveToolResults {
		finalMeter.Start(localize.T("progress.tools.text_answer", localize.Data{"Turn": toolsMaxTurns + 1}))
		result, err := requestFinalAnswerWithoutTools(ctx, client, messages, s.config.tools, maxTurnFinalizationRetries)
		if err == nil {
			finalMeter.Done(localize.T("progress.tools.final_done", localize.Data{"Turn": toolsMaxTurns + 1}))
			logStop("forced_final_answer", toolsMaxTurns)
			return SessionResult{
				Answer:    result,
				Citations: s.state.Citations(),
			}, nil
		}
		logStop("orchestration_limit_finalization_failed", toolsMaxTurns)
		return SessionResult{}, err
	}

	logStop("orchestration_limit", toolsMaxTurns)
	return SessionResult{}, orchestrationError{
		Code:    "orchestration_limit",
		Message: fmt.Sprintf("tools orchestration exceeded %d turns without final answer", toolsMaxTurns),
		Details: map[string]any{"max_turns": toolsMaxTurns},
	}
}

func newToolLoopState(ctx context.Context, source input.FileBackedSource, embedder llm.EmbeddingRequester, opts Options, indexMeter verbose.Meter) (*toolLoopState, error) {
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
		reader = files.NewNamedFileReader(source.SnapshotPath(), source.DisplayName())
	}
	toolset := files.NewTools(reader, files.ToolOptions{MultiFile: source.IsMultiFile()})
	if opts.EnableRAG {
		if embedder == nil {
			return nil, errors.New("tools rag requires embedding client")
		}
		indexMeter.Start(localize.T("progress.rag.read_input"))
		searcher, _, err := ragruntime.PrepareSearch(ctx, embedder, source, opts.RAG, indexMeter)
		if err != nil {
			return nil, err
		}
		toolset = append(toolset, ragtools.NewTool(searcher, embedder))
	}
	return &toolLoopState{
		registry:          aitools.NewRegistry(toolset...),
		seenLinesByDomain: make(map[string]map[string]struct{}),
		seenCitations:     make(map[string]struct{}),
	}, nil
}

func summarizeToolMeta(label string, meta toolExecutionMeta, index int, total int) string {
	base := fmt.Sprintf("%s (%d/%d)", normalizeToolCallLabel(label), index, total)
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
		callDesc := s.registry.DescribeCall(call)

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
			meter.Report(index+1, len(toolCalls), localize.T("progress.tools.meta_error", localize.Data{
				"Base": fmt.Sprintf("%s (%d/%d)", normalizeToolCallLabel(callDesc.VerboseLabel), index+1, len(toolCalls)),
			}))
			continue
		}

		if meta.NovelLines > 0 {
			turnHadProgress = true
		}
		if !meta.Duplicate {
			turnAllDuplicates = false
		}

		meter.Report(index+1, len(toolCalls), summarizeToolMeta(callDesc.VerboseLabel, meta, index+1, len(toolCalls)))
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

func normalizeToolCallLabel(label string) string {
	trimmed := strings.TrimSpace(label)
	if trimmed == "" {
		return "tool"
	}
	return trimmed
}

func (s *toolLoopState) executeToolCall(ctx context.Context, call openai.ToolCall) (string, toolExecutionMeta, error) {
	result, err := s.registry.Execute(ctx, call)
	if err != nil {
		return "", toolExecutionMeta{}, err
	}
	s.recordCitations(call.Function.Name, result.Payload)

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
	domain := coverageDomain(call.Function.Name)
	seenLines := s.seenLinesByDomain[domain]
	if seenLines == nil {
		seenLines = make(map[string]struct{})
		s.seenLinesByDomain[domain] = seenLines
	}
	novelLines := 0
	for key := range lineKeys {
		if _, seen := seenLines[key]; seen {
			continue
		}
		seenLines[key] = struct{}{}
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
	case "search_rag":
		s.stats.uniqueRAGSearch++
	case "read_lines", "read_around":
		s.stats.uniqueReads++
	}

	return annotated, meta, nil
}

func (s *toolLoopState) Citations() []retrieval.Citation {
	if len(s.citations) == 0 {
		return nil
	}
	return append([]retrieval.Citation(nil), s.citations...)
}

func coverageDomain(toolName string) string {
	switch toolName {
	case "search_rag":
		return "rag"
	case "list_files", "search_file", "read_lines", "read_around":
		return "files"
	default:
		return "default"
	}
}

func (s *toolLoopState) recordCitations(toolName string, raw string) {
	for _, citation := range citationsFromToolPayload(toolName, raw) {
		key := fmt.Sprintf("%s:%d-%d", citation.SourcePath, citation.StartLine, citation.EndLine)
		if _, seen := s.seenCitations[key]; seen {
			continue
		}
		s.seenCitations[key] = struct{}{}
		s.citations = append(s.citations, citation)
	}
}

func citationsFromToolPayload(toolName string, raw string) []retrieval.Citation {
	switch toolName {
	case "search_rag":
		var result ragtools.SearchResult
		if err := json.Unmarshal([]byte(raw), &result); err != nil {
			return nil
		}
		citations := make([]retrieval.Citation, 0, len(result.Matches))
		for _, match := range result.Matches {
			path := strings.TrimSpace(match.Path)
			if path == "" {
				path = strings.TrimSpace(result.Path)
			}
			if path == "" || match.StartLine < 1 || match.EndLine < match.StartLine {
				continue
			}
			citations = append(citations, retrieval.Citation{
				SourcePath: path,
				StartLine:  match.StartLine,
				EndLine:    match.EndLine,
			})
		}
		return citations
	case "search_file":
		var result files.SearchResult
		if err := json.Unmarshal([]byte(raw), &result); err != nil {
			return nil
		}
		citations := make([]retrieval.Citation, 0, len(result.Matches))
		for _, match := range result.Matches {
			path := strings.TrimSpace(match.Path)
			if path == "" {
				path = strings.TrimSpace(result.Path)
			}
			if path == "" || match.LineNumber < 1 {
				continue
			}
			citations = append(citations, retrieval.Citation{
				SourcePath: path,
				StartLine:  match.LineNumber,
				EndLine:    match.LineNumber,
			})
		}
		return citations
	case "read_lines":
		var result files.ReadLinesResult
		if err := json.Unmarshal([]byte(raw), &result); err != nil {
			return nil
		}
		path := strings.TrimSpace(result.Path)
		if path == "" || result.StartLine < 1 || result.EndLine < result.StartLine {
			return nil
		}
		return []retrieval.Citation{{
			SourcePath: path,
			StartLine:  result.StartLine,
			EndLine:    result.EndLine,
		}}
	case "read_around":
		var result files.ReadAroundResult
		if err := json.Unmarshal([]byte(raw), &result); err != nil {
			return nil
		}
		path := strings.TrimSpace(result.Path)
		if path == "" || result.StartLine < 1 || result.EndLine < result.StartLine {
			return nil
		}
		return []retrieval.Citation{{
			SourcePath: path,
			StartLine:  result.StartLine,
			EndLine:    result.EndLine,
		}}
	default:
		return nil
	}
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

func containsToolResponses(messages []openai.ChatCompletionMessage) bool {
	for _, message := range messages {
		if message.Role == "tool" {
			return true
		}
	}
	return false
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
