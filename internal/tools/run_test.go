package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/borro/ragcli/internal/aitools/files"
	"github.com/borro/ragcli/internal/input"
	"github.com/borro/ragcli/internal/llm"
	"github.com/borro/ragcli/internal/localize"
	ragruntime "github.com/borro/ragcli/internal/rag"
	"github.com/borro/ragcli/internal/verbose"
	"github.com/borro/ragcli/internal/verbose/testutil"
	openai "github.com/sashabaranov/go-openai"
)

func setRussianLocale(t *testing.T) {
	t.Helper()
	if err := localize.SetCurrent(localize.RU); err != nil {
		t.Fatalf("SetCurrent(ru) error = %v", err)
	}
}

func TestMain(m *testing.M) {
	if err := localize.SetCurrent(localize.RU); err != nil {
		panic(err)
	}
	os.Exit(m.Run())
}

type scriptedRequester struct {
	responses []*openai.ChatCompletionResponse
	errs      []error
	requests  []openai.ChatCompletionRequest
}

type testSource struct {
	kind         input.Kind
	inputPath    string
	snapshotPath string
	files        []input.File
}

type embeddingRequesterFunc func(context.Context, []string) ([][]float32, llm.EmbeddingMetrics, error)

func (f embeddingRequesterFunc) CreateEmbeddingsWithMetrics(ctx context.Context, inputs []string) ([][]float32, llm.EmbeddingMetrics, error) {
	return f(ctx, inputs)
}

func (s testSource) Kind() input.Kind {
	return s.kind
}

func (s testSource) InputPath() string {
	return s.inputPath
}

func (s testSource) DisplayName() string {
	if path := strings.TrimSpace(s.inputPath); path != "" {
		return path
	}
	if s.kind == input.KindStdin {
		return "stdin"
	}
	if len(s.files) > 0 && strings.TrimSpace(s.files[0].DisplayPath) != "" {
		return s.files[0].DisplayPath
	}
	return strings.TrimSpace(s.snapshotPath)
}

func (s testSource) SnapshotPath() string {
	return s.snapshotPath
}

func (s testSource) Open() (io.ReadCloser, error) {
	if strings.TrimSpace(s.snapshotPath) == "" {
		return nil, os.ErrNotExist
	}
	return os.Open(s.snapshotPath)
}

func (s testSource) BackingFiles() []input.File {
	return append([]input.File(nil), s.files...)
}

func (s testSource) FileCount() int {
	return len(s.files)
}

func (s testSource) IsMultiFile() bool {
	return s.kind == input.KindDirectory || len(s.files) > 1
}

func (s *scriptedRequester) SendRequestWithMetrics(_ context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, llm.RequestMetrics, error) {
	s.requests = append(s.requests, req)
	index := len(s.requests) - 1

	if index < len(s.errs) && s.errs[index] != nil {
		return nil, llm.RequestMetrics{}, s.errs[index]
	}

	if index >= len(s.responses) {
		return nil, llm.RequestMetrics{}, context.DeadlineExceeded
	}

	return s.responses[index], llm.RequestMetrics{
		Attempt:          1,
		MessageCount:     len(req.Messages),
		ToolCount:        len(req.Tools),
		PromptTokens:     10,
		CompletionTokens: 5,
		TotalTokens:      15,
		TokensPerSecond:  50,
	}, nil
}

func runTools(ctx context.Context, client llm.ChatRequester, filePath string, prompt string, plan *verbose.Plan) (string, error) {
	snapshotPath := strings.TrimSpace(filePath)
	if snapshotPath == "" || !pathExists(snapshotPath) {
		snapshotPath = writeAnonymousTempFile("placeholder\n")
	}

	source := testSource{
		kind:         input.KindFile,
		inputPath:    filePath,
		snapshotPath: snapshotPath,
		files: []input.File{{
			Path:        snapshotPath,
			DisplayPath: filepath.Base(filePath),
		}},
	}
	if filePath == "" {
		source.kind = input.KindStdin
		source.inputPath = ""
		source.files[0].DisplayPath = "stdin"
	}

	return runToolsWithSource(ctx, client, source, prompt, plan)
}

func runToolsWithSource(ctx context.Context, client llm.ChatRequester, source input.FileBackedSource, prompt string, plan *verbose.Plan) (string, error) {
	return Run(ctx, client, nil, source, Options{}, prompt, plan)
}

func pathExists(path string) bool {
	if strings.TrimSpace(path) == "" {
		return false
	}
	_, err := os.Stat(path)
	return err == nil
}

func fakeToolsEmbedder() llm.EmbeddingRequester {
	return embeddingRequesterFunc(func(_ context.Context, inputs []string) ([][]float32, llm.EmbeddingMetrics, error) {
		vectors := make([][]float32, 0, len(inputs))
		for _, input := range inputs {
			vectors = append(vectors, vectorFor(input))
		}
		return vectors, llm.EmbeddingMetrics{
			InputCount:   len(inputs),
			VectorCount:  len(vectors),
			PromptTokens: len(inputs),
			TotalTokens:  len(inputs),
		}, nil
	})
}

func vectorFor(input string) []float32 {
	lower := strings.ToLower(input)
	switch {
	case strings.Contains(lower, "retry"):
		return []float32{1, 0, 0}
	case strings.Contains(lower, "backoff"):
		return []float32{0.8, 0.1, 0}
	case strings.Contains(lower, "database"):
		return []float32{0, 1, 0}
	default:
		return []float32{0, 0, 0}
	}
}

func defaultRAGSearchOptions(t *testing.T) ragruntime.SearchOptions {
	t.Helper()
	return ragruntime.SearchOptions{
		TopK:           8,
		ChunkSize:      1000,
		ChunkOverlap:   200,
		IndexTTL:       time.Hour,
		IndexDir:       t.TempDir(),
		Rerank:         "heuristic",
		EmbeddingModel: "embed-v1",
	}
}

func runToolsWithOptions(ctx context.Context, client llm.ChatRequester, embedder llm.EmbeddingRequester, source input.FileBackedSource, opts Options, prompt string, plan *verbose.Plan) (string, error) {
	return Run(ctx, client, embedder, source, opts, prompt, plan)
}

func writeAnonymousTempFile(content string) string {
	tmpFile, err := os.CreateTemp("", "tools-test-*")
	if err != nil {
		panic(err)
	}
	if _, err := io.WriteString(tmpFile, content); err != nil {
		_ = tmpFile.Close()
		panic(err)
	}
	if err := tmpFile.Close(); err != nil {
		panic(err)
	}
	return tmpFile.Name()
}

func directorySource(path string, label string, files []input.File) testSource {
	return testSource{
		kind:         input.KindDirectory,
		inputPath:    path,
		snapshotPath: writeAnonymousTempFile(""),
		files:        append([]input.File(nil), files...),
	}
}

func fileSource(path string, label string) testSource {
	snapshotPath := strings.TrimSpace(path)
	if snapshotPath == "" || !pathExists(snapshotPath) {
		snapshotPath = writeAnonymousTempFile("")
	}
	return testSource{
		kind:         input.KindFile,
		inputPath:    path,
		snapshotPath: snapshotPath,
		files: []input.File{{
			Path:        snapshotPath,
			DisplayPath: filepath.Base(path),
		}},
	}
}

func TestRunTools_FinalAnswerWithoutTools(t *testing.T) {
	setRussianLocale(t)
	client := &scriptedRequester{
		responses: []*openai.ChatCompletionResponse{
			chatResponse(message("assistant", "Готовый ответ", nil)),
		},
	}

	result, err := runTools(context.Background(), client, "/tmp/file.txt", "Что в файле?", nil)
	if err != nil {
		t.Fatalf("RunTools() error = %v", err)
	}

	if result != "Готовый ответ" {
		t.Fatalf("RunTools() result = %q, want %q", result, "Готовый ответ")
	}
	if len(client.requests) != 1 {
		t.Fatalf("requests = %d, want 1", len(client.requests))
	}
	if len(client.requests[0].Tools) != 3 {
		t.Fatalf("tools sent = %d, want 3", len(client.requests[0].Tools))
	}
}

func TestRunTools_DirectoryInputAddsListFilesTool(t *testing.T) {
	setRussianLocale(t)
	client := &scriptedRequester{
		responses: []*openai.ChatCompletionResponse{
			chatResponse(message("assistant", "Готовый ответ", nil)),
		},
	}

	source := directorySource("/tmp/corpus", "/tmp/corpus", []input.File{
		{Path: writeTempFile(t, "alpha\n"), DisplayPath: "a.txt"},
		{Path: writeTempFile(t, "beta\n"), DisplayPath: "nested/b.txt"},
	})

	_, err := runToolsWithSource(context.Background(), client, source, "Что в директории?", nil)
	if err != nil {
		t.Fatalf("RunTools() error = %v", err)
	}

	if len(client.requests) != 1 {
		t.Fatalf("requests = %d, want 1", len(client.requests))
	}
	if len(client.requests[0].Tools) != 4 {
		t.Fatalf("tools sent = %d, want 4", len(client.requests[0].Tools))
	}
	if client.requests[0].Tools[0].Function == nil || client.requests[0].Tools[0].Function.Name != "list_files" {
		t.Fatalf("first tool = %#v, want list_files", client.requests[0].Tools[0].Function)
	}

	userPrompt := client.requests[0].Messages[1].Content
	if !strings.Contains(userPrompt, "Доступно файлов для анализа: 2.") {
		t.Fatalf("user prompt = %q, want file count", userPrompt)
	}
	if !strings.Contains(userPrompt, "сначала вызови list_files") {
		t.Fatalf("user prompt = %q, want list_files hint", userPrompt)
	}
}

func TestRunTools_StdinPromptExplainsSingleFileMode(t *testing.T) {
	setRussianLocale(t)
	client := &scriptedRequester{
		responses: []*openai.ChatCompletionResponse{
			chatResponse(message("assistant", "Готовый ответ", nil)),
		},
	}

	_, err := runTools(context.Background(), client, "", "Какой версии go-openai?", nil)
	if err != nil {
		t.Fatalf("RunTools() error = %v", err)
	}

	if len(client.requests) != 1 {
		t.Fatalf("requests = %d, want 1", len(client.requests))
	}
	if len(client.requests[0].Tools) != 3 {
		t.Fatalf("tools sent = %d, want 3", len(client.requests[0].Tools))
	}

	systemPrompt := client.requests[0].Messages[0].Content
	if !strings.Contains(systemPrompt, "Если подключён ровно один файл, list_files недоступен и не нужен") {
		t.Fatalf("system prompt = %q, want single-file instruction", systemPrompt)
	}

	userPrompt := client.requests[0].Messages[1].Content
	if !strings.Contains(userPrompt, "Подключён ровно один файл.") {
		t.Fatalf("user prompt = %q, want single-file hint", userPrompt)
	}
	if !strings.Contains(userPrompt, "Этот единственный файл пришёл через stdin") {
		t.Fatalf("user prompt = %q, want stdin hint", userPrompt)
	}
	if strings.Contains(userPrompt, "сначала вызови list_files") {
		t.Fatalf("user prompt = %q, must not suggest list_files for single-file stdin", userPrompt)
	}
}

func TestRunTools_VerboseEmitsSingleFinalCompletionLine(t *testing.T) {
	setRussianLocale(t)
	client := &scriptedRequester{
		responses: []*openai.ChatCompletionResponse{
			chatResponse(message("assistant", "Готовый ответ", nil)),
		},
	}
	var stderr bytes.Buffer

	result, err := runTools(context.Background(), client, "/tmp/file.txt", "Что в файле?", verbose.NewPlan(verbose.New(&stderr, true, false), "tools",
		verbose.StageDef{Key: "prepare", Label: "подготовка", Slots: 2},
		verbose.StageDef{Key: "init", Label: "инициализация", Slots: 2},
		verbose.StageDef{Key: "loop", Label: "LLM turn", Slots: 18},
		verbose.StageDef{Key: "final", Label: "финальный ответ", Slots: 2},
	))
	if err != nil {
		t.Fatalf("RunTools() error = %v", err)
	}
	if result != "Готовый ответ" {
		t.Fatalf("RunTools() result = %q, want %q", result, "Готовый ответ")
	}

	output := stderr.String()
	if strings.Count(output, "(100%)") != 1 {
		t.Fatalf("stderr = %q, want exactly one final completion line", output)
	}
	if !strings.Contains(output, "финальный ответ") {
		t.Fatalf("stderr = %q, want final answer stage", output)
	}
}

func TestRunTools_VerboseIncludesCompactToolCallLabel(t *testing.T) {
	setRussianLocale(t)
	client := &scriptedRequester{
		responses: []*openai.ChatCompletionResponse{
			chatResponse(message("assistant", "", []openai.ToolCall{
				toolCall("call-1", "search_file", `{"query":"alpha"}`),
			})),
			chatResponse(message("assistant", "alpha найдена", nil)),
		},
	}
	var stderr bytes.Buffer

	result, err := runTools(context.Background(), client, writeTempFile(t, "alpha\nbeta\n"), "Где alpha?", newVerbosePlan(&stderr, false))
	if err != nil {
		t.Fatalf("RunTools() error = %v", err)
	}
	if result != "alpha найдена" {
		t.Fatalf("RunTools() result = %q, want final answer", result)
	}

	output := stderr.String()
	if !strings.Contains(output, `search_file(query="alpha") (1/1): получил 1 новых строк`) {
		t.Fatalf("stderr = %q, want compact tool label in verbose progress", output)
	}
}

func TestRunTools_VerboseIncludesParametersOnToolError(t *testing.T) {
	setRussianLocale(t)
	client := &scriptedRequester{
		responses: []*openai.ChatCompletionResponse{
			chatResponse(message("assistant", "", []openai.ToolCall{
				toolCall("bad-call", "read_lines", `{"start_line":"bad","end_line":2}`),
			})),
			chatResponse(message("assistant", "Ошибка аргументов замечена", nil)),
		},
	}
	var stderr bytes.Buffer

	_, err := runTools(context.Background(), client, writeTempFile(t, "one\ntwo\n"), "test", newVerbosePlan(&stderr, false))
	if err != nil {
		t.Fatalf("RunTools() error = %v", err)
	}

	output := stderr.String()
	if !strings.Contains(output, `read_lines(args=`) {
		t.Fatalf("stderr = %q, want tool label with raw arguments on error", output)
	}
	if !strings.Contains(output, ": ошибка") {
		t.Fatalf("stderr = %q, want localized error status", output)
	}
}

func TestRunTools_DebugReporterKeepsVerboseSilent(t *testing.T) {
	setRussianLocale(t)
	client := &scriptedRequester{
		responses: []*openai.ChatCompletionResponse{
			chatResponse(message("assistant", "Готовый ответ", nil)),
		},
	}
	var stderr bytes.Buffer

	result, err := runTools(context.Background(), client, "/tmp/file.txt", "Что в файле?", newVerbosePlan(&stderr, true))
	if err != nil {
		t.Fatalf("RunTools() error = %v", err)
	}
	if result != "Готовый ответ" {
		t.Fatalf("RunTools() result = %q, want %q", result, "Готовый ответ")
	}
	if stderr.Len() != 0 {
		t.Fatalf("stderr = %q, want no verbose output when debug reporter is requested", stderr.String())
	}
}

func TestRunTools_ProgressFromPlanDoesNotRequirePrepareStage(t *testing.T) {
	setRussianLocale(t)
	client := &scriptedRequester{
		responses: []*openai.ChatCompletionResponse{
			chatResponse(message("assistant", "Готовый ответ", nil)),
		},
	}
	plan := planWithoutPrepare(nil)
	recorder := testutil.NewProgressRecorder(plan.Total())
	plan = planWithoutPrepare(recorder)

	result, err := runTools(context.Background(), client, "/tmp/file.txt", "Что в файле?", plan)
	if err != nil {
		t.Fatalf("RunTools() error = %v", err)
	}
	if result != "Готовый ответ" {
		t.Fatalf("RunTools() result = %q, want %q", result, "Готовый ответ")
	}

	_, currents, totals := recorder.Snapshot()
	if len(currents) == 0 {
		t.Fatal("progress recorder saw no updates")
	}
	if currents[len(currents)-1] != plan.Total() {
		t.Fatalf("final progress = %d, want %d", currents[len(currents)-1], plan.Total())
	}
	for _, total := range totals {
		if total != plan.Total() {
			t.Fatalf("reported total = %d, want %d", total, plan.Total())
		}
	}
}

func TestRunTools_EmptyFinalAnswerRetriesWithoutTools(t *testing.T) {
	setRussianLocale(t)
	client := &scriptedRequester{
		responses: []*openai.ChatCompletionResponse{
			chatResponse(message("assistant", "", nil)),
			chatResponse(message("assistant", "Итоговый ответ после retry", nil)),
		},
	}

	result, err := runTools(context.Background(), client, "/tmp/file.txt", "Что в файле?", nil)
	if err != nil {
		t.Fatalf("RunTools() error = %v", err)
	}
	if result != "Итоговый ответ после retry" {
		t.Fatalf("RunTools() result = %q, want retry answer", result)
	}
	if len(client.requests) != 2 {
		t.Fatalf("requests = %d, want 2", len(client.requests))
	}
	if client.requests[1].ToolChoice != "none" {
		t.Fatalf("ToolChoice = %#v, want none", client.requests[1].ToolChoice)
	}
	lastMessage := client.requests[1].Messages[len(client.requests[1].Messages)-1]
	if !strings.Contains(lastMessage.Content, "Сформулируй финальный ответ") {
		t.Fatalf("retry message = %q, want fallback instruction", lastMessage.Content)
	}
}

func TestRunTools_EmptyFinalAnswerRetryKeepsProgressMonotonic(t *testing.T) {
	setRussianLocale(t)
	client := &scriptedRequester{
		responses: []*openai.ChatCompletionResponse{
			chatResponse(message("assistant", "", nil)),
			chatResponse(message("assistant", "Итоговый ответ после retry", nil)),
		},
	}

	template := newToolsPlan(nil)
	recorder := testutil.NewProgressRecorder(template.Total())
	plan := newToolsPlan(recorder)

	result, err := runTools(context.Background(), client, "/tmp/file.txt", "Что в файле?", plan)
	if err != nil {
		t.Fatalf("RunTools() error = %v", err)
	}
	if result != "Итоговый ответ после retry" {
		t.Fatalf("RunTools() result = %q, want retry answer", result)
	}

	_, currents, _ := recorder.Snapshot()
	if len(currents) == 0 {
		t.Fatal("progress recorder saw no updates")
	}
	for i := 1; i < len(currents); i++ {
		if currents[i] < currents[i-1] {
			t.Fatalf("progress rolled back at step %d: %v", i, currents)
		}
	}
}

func TestRunTools_MultiStepToolLoop(t *testing.T) {
	setRussianLocale(t)
	client := &scriptedRequester{
		responses: []*openai.ChatCompletionResponse{
			chatResponse(message("assistant", "Сначала найду раздел.", []openai.ToolCall{
				toolCall("call-1", "search_file", `{"query":"архитектура"}`),
			})),
			chatResponse(message("assistant", "", []openai.ToolCall{
				toolCall("call-2", "read_lines", `{"start_line":10,"end_line":12}`),
			})),
			chatResponse(message("assistant", "Ответ по найденным строкам", nil)),
		},
	}

	filePath := writeTempFile(t, strings.Join([]string{
		"первая строка",
		"раздел архитектура",
		"ещё строка",
		"4",
		"5",
		"6",
		"7",
		"8",
		"9",
		"10: архитектура сервиса",
		"11: детали",
		"12: вывод",
	}, "\n"))

	result, err := runTools(context.Background(), client, filePath, "Что сказано про архитектуру?", nil)
	if err != nil {
		t.Fatalf("RunTools() error = %v", err)
	}

	if result != "Ответ по найденным строкам" {
		t.Fatalf("RunTools() result = %q, want %q", result, "Ответ по найденным строкам")
	}
	if len(client.requests) != 3 {
		t.Fatalf("requests = %d, want 3", len(client.requests))
	}
	assertToolMessage(t, client.requests[1].Messages, "call-1", "search_file")
	assertToolMessage(t, client.requests[2].Messages, "call-2", "read_lines")

	if got := client.requests[1].Messages[2].Content; !strings.Contains(got, "Сначала найду раздел.") {
		t.Fatalf("assistant plan content missing from history: %q", got)
	}

	var searchPayload map[string]any
	toolMsg := client.requests[1].Messages[len(client.requests[1].Messages)-1]
	if err := json.Unmarshal([]byte(toolMsg.Content), &searchPayload); err != nil {
		t.Fatalf("json.Unmarshal(search tool message) error = %v", err)
	}
	if searchPayload["novel_lines"] == nil || searchPayload["progress_hint"] == nil {
		t.Fatalf("tool payload = %#v, want orchestration metadata", searchPayload)
	}
}

func TestRunTools_FinalizesWithoutToolsAfterMaxTurns(t *testing.T) {
	responses := make([]*openai.ChatCompletionResponse, 0, toolsMaxTurns)
	for i := 0; i < toolsMaxTurns; i++ {
		responses = append(responses, chatResponse(message("assistant", "", []openai.ToolCall{
			toolCall("loop-call", "search_file", fmt.Sprintf(`{"query":"x","limit":1,"offset":%d}`, i)),
		})))
	}
	responses = append(responses, chatResponse(message("assistant", "Итог после принудительной финализации", nil)))

	client := &scriptedRequester{responses: responses}
	filePath := writeTempFile(t, strings.Join(buildNumberedLines("x", toolsMaxTurns+5), "\n")+"\n")

	result, err := runTools(context.Background(), client, filePath, "loop?", nil)
	if err != nil {
		t.Fatalf("RunTools() error = %v", err)
	}
	if result != "Итог после принудительной финализации" {
		t.Fatalf("RunTools() result = %q, want forced finalization answer", result)
	}
	if len(client.requests) != toolsMaxTurns+1 {
		t.Fatalf("requests = %d, want %d", len(client.requests), toolsMaxTurns+1)
	}
	lastReq := client.requests[len(client.requests)-1]
	if lastReq.ToolChoice != "none" {
		t.Fatalf("ToolChoice = %#v, want none", lastReq.ToolChoice)
	}
	lastMsg := lastReq.Messages[len(lastReq.Messages)-1]
	if !strings.Contains(lastMsg.Content, "Останови вызовы инструментов") {
		t.Fatalf("finalization message = %q, want stop-tools instruction", lastMsg.Content)
	}
}

func TestRunTools_ReturnsOrchestrationLimitWhenForcedFinalizationStillEmpty(t *testing.T) {
	responses := make([]*openai.ChatCompletionResponse, 0, toolsMaxTurns+2)
	for i := 0; i < toolsMaxTurns; i++ {
		responses = append(responses, chatResponse(message("assistant", "", []openai.ToolCall{
			toolCall("loop-call", "search_file", fmt.Sprintf(`{"query":"x","limit":1,"offset":%d}`, i)),
		})))
	}
	responses = append(responses,
		chatResponse(message("assistant", "", nil)),
		chatResponse(message("assistant", "", nil)),
	)

	client := &scriptedRequester{responses: responses}
	filePath := writeTempFile(t, strings.Join(buildNumberedLines("x", toolsMaxTurns+5), "\n")+"\n")

	_, err := runTools(context.Background(), client, filePath, "loop?", nil)
	var orchErr orchestrationError
	if !errorsAsOrchestration(err, &orchErr) || orchErr.Code != "orchestration_limit" {
		t.Fatalf("RunTools() error = %v, want orchestration_limit", err)
	}
}

func TestRunTools_ToolErrorJSONDoesNotBreakLoop(t *testing.T) {
	setRussianLocale(t)
	client := &scriptedRequester{
		responses: []*openai.ChatCompletionResponse{
			chatResponse(message("assistant", "", []openai.ToolCall{
				toolCall("bad-call", "read_lines", `{"start_line":"bad","end_line":2}`),
			})),
			chatResponse(message("assistant", "Не удалось прочитать строки, аргументы были некорректны", nil)),
		},
	}

	filePath := writeTempFile(t, "one\ntwo\n")
	result, err := runTools(context.Background(), client, filePath, "test", nil)
	if err != nil {
		t.Fatalf("RunTools() error = %v", err)
	}
	if !strings.Contains(result, "некоррект") {
		t.Fatalf("RunTools() result = %q, want tool-error-based answer", result)
	}

	lastMessages := client.requests[1].Messages
	toolMsg := lastMessages[len(lastMessages)-1]
	var payload map[string]any
	if err := json.Unmarshal([]byte(toolMsg.Content), &payload); err != nil {
		t.Fatalf("json.Unmarshal(tool message) error = %v", err)
	}
	if payload["code"] == "" || payload["message"] == "" {
		t.Fatalf("tool error payload = %#v, want code/message fields", payload)
	}
}

func TestRunTools_DuplicateSearchUsesCacheAndFinalizesWithoutTools(t *testing.T) {
	client := &scriptedRequester{
		responses: []*openai.ChatCompletionResponse{
			chatResponse(message("assistant", "", []openai.ToolCall{
				toolCall("call-1", "search_file", `{"query":"alpha"}`),
			})),
			chatResponse(message("assistant", "", []openai.ToolCall{
				toolCall("call-2", "search_file", `{"query":"alpha"}`),
			})),
			chatResponse(message("assistant", "", []openai.ToolCall{
				toolCall("call-3", "search_file", `{"query":"alpha"}`),
			})),
			chatResponse(message("assistant", "alpha найдена в первой строке", nil)),
		},
	}

	filePath := writeTempFile(t, "alpha\nbeta\n")
	result, err := runTools(context.Background(), client, filePath, "find alpha", nil)
	if err != nil {
		t.Fatalf("RunTools() error = %v", err)
	}
	if result != "alpha найдена в первой строке" {
		t.Fatalf("RunTools() result = %q, want fallback answer", result)
	}
	if len(client.requests) != 4 {
		t.Fatalf("requests = %d, want 4", len(client.requests))
	}
	if client.requests[3].ToolChoice != "none" {
		t.Fatalf("ToolChoice = %#v, want none", client.requests[3].ToolChoice)
	}
	lastMessage := client.requests[3].Messages[len(client.requests[3].Messages)-1]
	if !strings.Contains(lastMessage.Content, "Останови вызовы инструментов") {
		t.Fatalf("fallback message = %q, want stop-tools instruction", lastMessage.Content)
	}
}

func TestRunTools_DuplicateReadLinesUsesCacheAndFinalizesWithoutTools(t *testing.T) {
	client := &scriptedRequester{
		responses: []*openai.ChatCompletionResponse{
			chatResponse(message("assistant", "", []openai.ToolCall{
				toolCall("call-1", "read_lines", `{"start_line":1,"end_line":2}`),
			})),
			chatResponse(message("assistant", "", []openai.ToolCall{
				toolCall("call-2", "read_lines", `{"start_line":1,"end_line":2}`),
			})),
			chatResponse(message("assistant", "", []openai.ToolCall{
				toolCall("call-3", "read_lines", `{"start_line":1,"end_line":2}`),
			})),
			chatResponse(message("assistant", "В строках 1-2: one, two", nil)),
		},
	}

	filePath := writeTempFile(t, "one\ntwo\nthree\n")
	result, err := runTools(context.Background(), client, filePath, "read", nil)
	if err != nil {
		t.Fatalf("RunTools() error = %v", err)
	}
	if result != "В строках 1-2: one, two" {
		t.Fatalf("RunTools() result = %q, want fallback answer", result)
	}
}

func TestRunTools_NoProgressFinalizesWithoutTools(t *testing.T) {
	client := &scriptedRequester{
		responses: []*openai.ChatCompletionResponse{
			chatResponse(message("assistant", "", []openai.ToolCall{
				toolCall("call-1", "read_lines", `{"start_line":1,"end_line":3}`),
			})),
			chatResponse(message("assistant", "", []openai.ToolCall{
				toolCall("call-2", "read_around", `{"line":2,"before":1,"after":1}`),
			})),
			chatResponse(message("assistant", "", []openai.ToolCall{
				toolCall("call-3", "read_around", `{"line":2,"before":1,"after":1}`),
			})),
			chatResponse(message("assistant", "Повторное чтение не добавило новых строк", nil)),
		},
	}

	filePath := writeTempFile(t, "one\ntwo\nthree\nfour\n")
	result, err := runTools(context.Background(), client, filePath, "read same area", nil)
	if err != nil {
		t.Fatalf("RunTools() error = %v", err)
	}
	if result != "Повторное чтение не добавило новых строк" {
		t.Fatalf("RunTools() result = %q, want fallback answer", result)
	}
}

func TestToolLoopState_DuplicateSearchMarksCachedPayload(t *testing.T) {
	filePath := writeTempFile(t, "alpha\nbeta\n")
	state, err := newToolLoopState(context.Background(), fileSource(filePath, filePath), nil, Options{}, verbose.Meter{})
	if err != nil {
		t.Fatalf("newToolLoopState() error = %v", err)
	}
	first, _, err := state.executeToolCall(context.Background(), toolCall("call-1", "search_file", `{"query":"alpha"}`))
	if err != nil {
		t.Fatalf("executeToolCall(first) error = %v", err)
	}
	second, meta, err := state.executeToolCall(context.Background(), toolCall("call-2", "search_file", `{"query":"alpha"}`))
	if err != nil {
		t.Fatalf("executeToolCall(second) error = %v", err)
	}

	if first == second {
		t.Fatalf("expected duplicate payload to be annotated differently")
	}
	if !meta.Cached || !meta.Duplicate || meta.NovelLines != 0 {
		t.Fatalf("meta = %+v, want cached duplicate with no novel lines", meta)
	}

	var payload map[string]any
	if err := json.Unmarshal([]byte(second), &payload); err != nil {
		t.Fatalf("json.Unmarshal(tool message) error = %v", err)
	}
	if payload["cached"] != true || payload["duplicate"] != true {
		t.Fatalf("tool payload = %#v, want cached duplicate response", payload)
	}
}

func TestToolLoopState_StdinAliasPathWorksForSingleFileReader(t *testing.T) {
	filePath := writeTempFile(t, "github.com/sashabaranov/go-openai v1.41.2\n")
	source := testSource{
		kind:         input.KindStdin,
		inputPath:    "",
		snapshotPath: filePath,
		files: []input.File{{
			Path:        filePath,
			DisplayPath: "stdin",
		}},
	}
	state, err := newToolLoopState(context.Background(), source, nil, Options{}, verbose.Meter{})
	if err != nil {
		t.Fatalf("newToolLoopState() error = %v", err)
	}
	result, meta, err := state.executeToolCall(context.Background(), toolCall("call-stdin", "search_file", `{"path":"stdin","query":"go-openai"}`))
	if err != nil {
		t.Fatalf("executeToolCall(stdin alias) error = %v", err)
	}
	if meta.NovelLines == 0 {
		t.Fatalf("meta = %+v, want novel lines from stdin alias path", meta)
	}
	if !strings.Contains(result, "v1.41.2") {
		t.Fatalf("result = %q, want version from stdin alias path", result)
	}
}

func TestToolLoopState_ReadAroundSeenLinesMarksNoProgress(t *testing.T) {
	filePath := writeTempFile(t, "one\ntwo\nthree\nfour\n")
	state, err := newToolLoopState(context.Background(), fileSource(filePath, filePath), nil, Options{}, verbose.Meter{})
	if err != nil {
		t.Fatalf("newToolLoopState() error = %v", err)
	}
	_, firstMeta, err := state.executeToolCall(context.Background(), toolCall("call-1", "read_lines", `{"start_line":1,"end_line":3}`))
	if err != nil {
		t.Fatalf("executeToolCall(first) error = %v", err)
	}
	_, secondMeta, err := state.executeToolCall(context.Background(), toolCall("call-2", "read_around", `{"line":2,"before":1,"after":1}`))
	if err != nil {
		t.Fatalf("executeToolCall(second) error = %v", err)
	}

	if firstMeta.NovelLines == 0 {
		t.Fatalf("firstMeta = %+v, want novel lines", firstMeta)
	}
	if !secondMeta.AlreadySeen || secondMeta.NovelLines != 0 {
		t.Fatalf("secondMeta = %+v, want already seen with no novel lines", secondMeta)
	}
}

func TestToolLoopState_MultiFilePathsKeepSeenLinesSeparate(t *testing.T) {
	firstPath := writeTempFile(t, "alpha\n")
	secondPath := writeTempFile(t, "alpha\n")
	state, err := newToolLoopState(context.Background(), directorySource("/tmp/corpus", "corpus", []input.File{
		{Path: firstPath, DisplayPath: "a.txt"},
		{Path: secondPath, DisplayPath: "b.txt"},
	}), nil, Options{}, verbose.Meter{})
	if err != nil {
		t.Fatalf("newToolLoopState() error = %v", err)
	}

	_, firstMeta, err := state.executeToolCall(context.Background(), toolCall("call-1", "read_lines", `{"path":"a.txt","start_line":1,"end_line":1}`))
	if err != nil {
		t.Fatalf("executeToolCall(first) error = %v", err)
	}
	_, secondMeta, err := state.executeToolCall(context.Background(), toolCall("call-2", "read_lines", `{"path":"b.txt","start_line":1,"end_line":1}`))
	if err != nil {
		t.Fatalf("executeToolCall(second) error = %v", err)
	}

	if firstMeta.NovelLines != 1 {
		t.Fatalf("firstMeta = %+v, want one novel line", firstMeta)
	}
	if secondMeta.NovelLines != 1 || secondMeta.AlreadySeen {
		t.Fatalf("secondMeta = %+v, want separate line tracking per path", secondMeta)
	}
}

func TestToolLoopState_ListFilesPaginationCountsAsProgress(t *testing.T) {
	state, err := newToolLoopState(context.Background(), directorySource("/tmp/corpus", "corpus", []input.File{
		{Path: writeTempFile(t, "alpha\n"), DisplayPath: "a.txt"},
		{Path: writeTempFile(t, "beta\n"), DisplayPath: "b.txt"},
	}), nil, Options{}, verbose.Meter{})
	if err != nil {
		t.Fatalf("newToolLoopState() error = %v", err)
	}

	_, firstMeta, err := state.executeToolCall(context.Background(), toolCall("call-1", "list_files", `{"limit":1,"offset":0}`))
	if err != nil {
		t.Fatalf("executeToolCall(first) error = %v", err)
	}
	_, secondMeta, err := state.executeToolCall(context.Background(), toolCall("call-2", "list_files", `{"limit":1,"offset":1}`))
	if err != nil {
		t.Fatalf("executeToolCall(second) error = %v", err)
	}

	if firstMeta.NovelLines != 1 || firstMeta.AlreadySeen {
		t.Fatalf("firstMeta = %+v, want one novel file entry", firstMeta)
	}
	if secondMeta.NovelLines != 1 || secondMeta.AlreadySeen {
		t.Fatalf("secondMeta = %+v, want second page to count as progress", secondMeta)
	}
}

func TestRunTools_SystemPromptMentionsAgentPolicy(t *testing.T) {
	setRussianLocale(t)
	client := &scriptedRequester{
		responses: []*openai.ChatCompletionResponse{
			chatResponse(message("assistant", "ok", nil)),
		},
	}

	_, err := runTools(context.Background(), client, "/tmp/file.txt", "Что в файле?", nil)
	if err != nil {
		t.Fatalf("RunTools() error = %v", err)
	}

	systemPrompt := client.requests[0].Messages[0].Content
	if !strings.Contains(systemPrompt, "Сначала сформулируй короткий план") {
		t.Fatalf("system prompt = %q, want planning instruction", systemPrompt)
	}
	if !strings.Contains(systemPrompt, "Не повторяй один и тот же вызов") {
		t.Fatalf("system prompt = %q, want duplicate guardrail", systemPrompt)
	}
	if !strings.Contains(systemPrompt, "нельзя просить пользователя повторно прислать файл") {
		t.Fatalf("system prompt = %q, want no re-upload instruction", systemPrompt)
	}
	if !strings.Contains(systemPrompt, "инструменты не были использованы") {
		t.Fatalf("system prompt = %q, want backend limitation guidance", systemPrompt)
	}
}

func TestRunTools_UserPromptMentionsToolsFileContext(t *testing.T) {
	setRussianLocale(t)
	client := &scriptedRequester{
		responses: []*openai.ChatCompletionResponse{
			chatResponse(message("assistant", "ok", nil)),
		},
	}

	filePath := "/tmp/rest2push.log"
	_, err := runTools(context.Background(), client, filePath, "Есть ли тут явные критические ошибки?", nil)
	if err != nil {
		t.Fatalf("RunTools() error = %v", err)
	}

	userPrompt := client.requests[0].Messages[1].Content
	if !strings.Contains(userPrompt, "Входной путь уже доступен через инструменты") {
		t.Fatalf("user prompt = %q, want tools path context", userPrompt)
	}
	if !strings.Contains(userPrompt, `Путь для анализа: "/tmp/rest2push.log".`) {
		t.Fatalf("user prompt = %q, want path name", userPrompt)
	}
	if !strings.Contains(userPrompt, `Вопрос: "Есть ли тут явные критические ошибки?"`) {
		t.Fatalf("user prompt = %q, want original question", userPrompt)
	}
}

func TestRunTools_RAGAddsSearchToolAndPromptHints(t *testing.T) {
	setRussianLocale(t)
	client := &scriptedRequester{
		responses: []*openai.ChatCompletionResponse{
			chatResponse(message("assistant", "ok", nil)),
		},
	}

	filePath := writeTempFile(t, "retry policy enabled\nbackoff is exponential\n")
	source := fileSource(filePath, filePath)
	_, err := runToolsWithOptions(context.Background(), client, fakeToolsEmbedder(), source, Options{
		EnableRAG: true,
		RAG:       defaultRAGSearchOptions(t),
	}, "Что сказано про retry policy?", nil)
	if err != nil {
		t.Fatalf("RunTools() error = %v", err)
	}

	if len(client.requests) != 1 {
		t.Fatalf("requests = %d, want 1", len(client.requests))
	}
	if len(client.requests[0].Tools) != 4 {
		t.Fatalf("tools sent = %d, want 4", len(client.requests[0].Tools))
	}

	foundSearchRAG := false
	for _, tool := range client.requests[0].Tools {
		if tool.Function != nil && tool.Function.Name == "search_rag" {
			foundSearchRAG = true
		}
	}
	if !foundSearchRAG {
		t.Fatalf("tool definitions = %#v, want search_rag", client.requests[0].Tools)
	}

	systemPrompt := client.requests[0].Messages[0].Content
	if !strings.Contains(systemPrompt, "search_rag") {
		t.Fatalf("system prompt = %q, want search_rag guidance", systemPrompt)
	}
	userPrompt := client.requests[0].Messages[1].Content
	if !strings.Contains(userPrompt, "Доступен semantic retrieval") {
		t.Fatalf("user prompt = %q, want rag hint", userPrompt)
	}
}

func TestRunTools_RAGIndexFailureStopsBeforeFirstLLMRequest(t *testing.T) {
	client := &scriptedRequester{}
	source := fileSource(writeTempFile(t, "retry policy\n"), "/tmp/retries.txt")
	badEmbedder := embeddingRequesterFunc(func(_ context.Context, _ []string) ([][]float32, llm.EmbeddingMetrics, error) {
		return nil, llm.EmbeddingMetrics{}, errors.New("embed failed")
	})

	_, err := runToolsWithOptions(context.Background(), client, badEmbedder, source, Options{
		EnableRAG: true,
		RAG:       defaultRAGSearchOptions(t),
	}, "Что сказано про retry policy?", nil)
	if err == nil || !strings.Contains(err.Error(), "embed failed") {
		t.Fatalf("RunTools() error = %v, want embed failure", err)
	}
	if len(client.requests) != 0 {
		t.Fatalf("requests = %d, want no chat requests before preindexing succeeds", len(client.requests))
	}
}

func TestRunTools_SearchRAGToolLoop(t *testing.T) {
	setRussianLocale(t)
	client := &scriptedRequester{
		responses: []*openai.ChatCompletionResponse{
			chatResponse(message("assistant", "Сначала найду релевантный chunk.", []openai.ToolCall{
				toolCall("call-1", "search_rag", `{"query":"retry policy"}`),
			})),
			chatResponse(message("assistant", "Retry policy включён.", nil)),
		},
	}

	filePath := writeTempFile(t, "retry policy enabled\nbackoff is exponential\n")
	source := fileSource(filePath, filePath)
	result, err := runToolsWithOptions(context.Background(), client, fakeToolsEmbedder(), source, Options{
		EnableRAG: true,
		RAG:       defaultRAGSearchOptions(t),
	}, "Что сказано про retry policy?", nil)
	if err != nil {
		t.Fatalf("RunTools() error = %v", err)
	}
	if result != "Retry policy включён." {
		t.Fatalf("RunTools() result = %q, want final answer", result)
	}
	assertToolMessage(t, client.requests[1].Messages, "call-1", "search_rag")

	toolMsg := client.requests[1].Messages[len(client.requests[1].Messages)-1]
	var payload map[string]any
	if err := json.Unmarshal([]byte(toolMsg.Content), &payload); err != nil {
		t.Fatalf("json.Unmarshal(tool message) error = %v", err)
	}
	if payload["match_count"] == nil || payload["novel_lines"] == nil {
		t.Fatalf("tool payload = %#v, want search result and orchestration metadata", payload)
	}
}

func TestToolLoopState_SearchRAGAndReadLinesTrackCoverageSeparately(t *testing.T) {
	filePath := writeTempFile(t, "retry policy enabled\nbackoff is exponential\n")
	state, err := newToolLoopState(context.Background(), fileSource(filePath, filePath), fakeToolsEmbedder(), Options{
		EnableRAG: true,
		RAG:       defaultRAGSearchOptions(t),
	}, verbose.Meter{})
	if err != nil {
		t.Fatalf("newToolLoopState() error = %v", err)
	}

	_, firstMeta, err := state.executeToolCall(context.Background(), toolCall("call-1", "search_rag", `{"query":"retry policy"}`))
	if err != nil {
		t.Fatalf("executeToolCall(search_rag) error = %v", err)
	}
	_, secondMeta, err := state.executeToolCall(context.Background(), toolCall("call-2", "read_lines", `{"start_line":1,"end_line":2}`))
	if err != nil {
		t.Fatalf("executeToolCall(read_lines) error = %v", err)
	}

	if firstMeta.NovelLines == 0 {
		t.Fatalf("firstMeta = %+v, want semantic hit to count as progress", firstMeta)
	}
	if secondMeta.AlreadySeen || secondMeta.NovelLines == 0 {
		t.Fatalf("secondMeta = %+v, want file-domain coverage to stay separate from rag coverage", secondMeta)
	}
}

func TestToolLoopState_ReadLinesAndSearchRAGTrackCoverageSeparately(t *testing.T) {
	filePath := writeTempFile(t, "retry policy enabled\nbackoff is exponential\n")
	state, err := newToolLoopState(context.Background(), fileSource(filePath, filePath), fakeToolsEmbedder(), Options{
		EnableRAG: true,
		RAG:       defaultRAGSearchOptions(t),
	}, verbose.Meter{})
	if err != nil {
		t.Fatalf("newToolLoopState() error = %v", err)
	}

	_, firstMeta, err := state.executeToolCall(context.Background(), toolCall("call-1", "read_lines", `{"start_line":1,"end_line":2}`))
	if err != nil {
		t.Fatalf("executeToolCall(read_lines) error = %v", err)
	}
	_, secondMeta, err := state.executeToolCall(context.Background(), toolCall("call-2", "search_rag", `{"query":"retry policy"}`))
	if err != nil {
		t.Fatalf("executeToolCall(search_rag) error = %v", err)
	}

	if firstMeta.NovelLines == 0 {
		t.Fatalf("firstMeta = %+v, want file-domain hit to count as progress", firstMeta)
	}
	if secondMeta.AlreadySeen || secondMeta.NovelLines == 0 {
		t.Fatalf("secondMeta = %+v, want rag-domain coverage to stay separate from file coverage", secondMeta)
	}
}

func TestCitationsFromToolPayload_SearchFileAddsLineMatches(t *testing.T) {
	raw, err := files.MarshalJSON(files.SearchResult{
		Path: "fallback.txt",
		Matches: []files.SearchMatch{
			{Path: "a.txt", LineNumber: 1, Content: "retry policy enabled"},
			{LineNumber: 7, Content: "backoff is exponential"},
		},
	})
	if err != nil {
		t.Fatalf("MarshalJSON() error = %v", err)
	}

	citations := citationsFromToolPayload("search_file", raw)
	if len(citations) != 2 {
		t.Fatalf("len(citations) = %d, want 2 (%#v)", len(citations), citations)
	}
	if citations[0].SourcePath != "a.txt" || citations[0].StartLine != 1 || citations[0].EndLine != 1 {
		t.Fatalf("citations[0] = %#v, want a.txt:1", citations[0])
	}
	if citations[1].SourcePath != "fallback.txt" || citations[1].StartLine != 7 || citations[1].EndLine != 7 {
		t.Fatalf("citations[1] = %#v, want fallback.txt:7", citations[1])
	}
}

func TestParseTextToolCalls_ParsesPlanAndArguments(t *testing.T) {
	content := "## План поиска: дочитать локальный контекст.\n<tool_call> <function=read_around> <parameter=Path> a.txt <parameter=line> 7 <parameter=before> 2 <parameter=after> 3 </tool_call>"

	sanitized, calls, detected, err := parseTextToolCalls(content, 3)
	if err != nil {
		t.Fatalf("parseTextToolCalls() error = %v", err)
	}
	if !detected {
		t.Fatal("parseTextToolCalls() detected = false, want true")
	}
	if sanitized != "## План поиска: дочитать локальный контекст." {
		t.Fatalf("sanitized = %q, want plan without markup", sanitized)
	}
	if len(calls) != 1 {
		t.Fatalf("len(calls) = %d, want 1", len(calls))
	}
	if calls[0].ID != "text-tool-call-3-1" || calls[0].Function.Name != "read_around" {
		t.Fatalf("call = %#v, want parsed synthetic read_around", calls[0])
	}

	var args map[string]any
	if err := json.Unmarshal([]byte(calls[0].Function.Arguments), &args); err != nil {
		t.Fatalf("json.Unmarshal(arguments) error = %v", err)
	}
	if args["path"] != "a.txt" || args["line"] != float64(7) || args["before"] != float64(2) || args["after"] != float64(3) {
		t.Fatalf("arguments = %#v, want normalized path/line/before/after", args)
	}
}

func TestRunTools_ExecutesTextualToolCallsFallback(t *testing.T) {
	filePath := writeTempFile(t, "alpha\nbeta\n")
	client := &scriptedRequester{
		responses: []*openai.ChatCompletionResponse{
			chatResponse(message("assistant", "Сначала найду совпадения.\n<tool_call> <function=search_file> <parameter=query> alpha </tool_call>", nil)),
			chatResponse(message("assistant", "Нашёл совпадение на первой строке.", nil)),
		},
	}

	result, err := runTools(context.Background(), client, filePath, "Где упоминается alpha?", nil)
	if err != nil {
		t.Fatalf("RunTools() error = %v", err)
	}

	if result != "Нашёл совпадение на первой строке." {
		t.Fatalf("result = %q, want final answer after parsed textual tool call", result)
	}
	if len(client.requests) != 2 {
		t.Fatalf("requests = %d, want 2", len(client.requests))
	}
	assertToolMessage(t, client.requests[1].Messages, "text-tool-call-1-1", "search_file")
}

func TestRunTools_RetriesMalformedTextualToolCalls(t *testing.T) {
	filePath := writeTempFile(t, "alpha\nbeta\n")
	client := &scriptedRequester{
		responses: []*openai.ChatCompletionResponse{
			chatResponse(message("assistant", "## План поиска.\n<tool_call> <function=read_around> <parameter=Path>\na.txt", nil)),
			chatResponse(message("assistant", "Не удалось использовать инструменты, поэтому надёжно ответить нельзя.", nil)),
		},
	}

	result, err := runTools(context.Background(), client, filePath, "Что здесь важно?", nil)
	if err != nil {
		t.Fatalf("RunTools() error = %v", err)
	}

	if strings.Contains(result, "<tool_call>") {
		t.Fatalf("result = %q, want textual tool-call markup to be filtered out", result)
	}
	if result != "Не удалось использовать инструменты, поэтому надёжно ответить нельзя." {
		t.Fatalf("result = %q, want retry final answer", result)
	}
	if len(client.requests) != 2 {
		t.Fatalf("requests = %d, want 2", len(client.requests))
	}
	lastMessage := client.requests[1].Messages[len(client.requests[1].Messages)-1]
	if lastMessage.Role != "user" || lastMessage.Content != localize.T("tools.prompt.text_tool_call_retry") {
		t.Fatalf("last retry message = %#v, want textual-tool-call retry prompt", lastMessage)
	}
}

func TestRunTools_WarnsWhenFirstAnswerSkipsTools(t *testing.T) {
	client := &scriptedRequester{
		responses: []*openai.ChatCompletionResponse{
			chatResponse(message("assistant", "Не вижу файла, пришлите его.", nil)),
		},
	}

	original := slog.Default()
	var logs bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelWarn}))
	slog.SetDefault(logger)
	t.Cleanup(func() {
		slog.SetDefault(original)
	})

	result, err := runTools(context.Background(), client, "/tmp/rest2push.log", "Есть ли тут явные критические ошибки?", nil)
	if err != nil {
		t.Fatalf("RunTools() error = %v", err)
	}
	if result != "Не вижу файла, пришлите его." {
		t.Fatalf("RunTools() result = %q, want direct answer preserved", result)
	}

	output := logs.String()
	if !strings.Contains(output, "tools backend returned a direct answer without using file tools") {
		t.Fatalf("logs = %q, want warning about skipped tools", output)
	}
	if !strings.Contains(output, "file_path=/tmp/rest2push.log") {
		t.Fatalf("logs = %q, want file path in warning", output)
	}
}

func assertToolMessage(t *testing.T, messages []openai.ChatCompletionMessage, toolCallID string, name string) {
	t.Helper()

	foundAssistant := false
	foundTool := false
	for _, msg := range messages {
		if len(msg.ToolCalls) > 0 {
			for _, call := range msg.ToolCalls {
				if call.ID == toolCallID {
					foundAssistant = true
				}
			}
		}
		if msg.Role == "tool" && msg.ToolCallID == toolCallID && msg.Name == name {
			foundTool = true
		}
	}

	if !foundAssistant {
		t.Fatalf("assistant tool call %q not preserved in history", toolCallID)
	}
	if !foundTool {
		t.Fatalf("tool response %q/%q not found in history", toolCallID, name)
	}
}

func chatResponse(msg openai.ChatCompletionMessage) *openai.ChatCompletionResponse {
	return &openai.ChatCompletionResponse{
		Choices: []openai.ChatCompletionChoice{
			{
				Message: msg,
			},
		},
	}
}

func message(role string, content string, toolCalls []openai.ToolCall) openai.ChatCompletionMessage {
	return openai.ChatCompletionMessage{
		Role:      role,
		Content:   content,
		ToolCalls: toolCalls,
	}
}

func toolCall(id string, name string, arguments string) openai.ToolCall {
	return openai.ToolCall{
		ID:   id,
		Type: "function",
		Function: openai.FunctionCall{
			Name:      name,
			Arguments: arguments,
		},
	}
}

func buildNumberedLines(prefix string, count int) []string {
	lines := make([]string, 0, count)
	for i := 1; i <= count; i++ {
		lines = append(lines, fmt.Sprintf("%s%d", prefix, i))
	}
	return lines
}

func writeTempFile(t *testing.T, content string) string {
	t.Helper()

	path := t.TempDir() + "/input.txt"
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}
	return path
}

func planWithoutPrepare(reporter verbose.Reporter) *verbose.Plan {
	return verbose.NewPlan(reporter, "tools",
		verbose.StageDef{Key: "init", Label: "init", Slots: 2},
		verbose.StageDef{Key: "loop", Label: "loop", Slots: 18},
		verbose.StageDef{Key: "final", Label: "final", Slots: 2},
	)
}

func newVerbosePlan(stderr *bytes.Buffer, debug bool) *verbose.Plan {
	return newToolsPlan(verbose.New(stderr, true, debug))
}

func newToolsPlan(reporter verbose.Reporter) *verbose.Plan {
	return verbose.NewPlan(reporter, "tools",
		verbose.StageDef{Key: "prepare", Label: "подготовка", Slots: 2},
		verbose.StageDef{Key: "init", Label: "инициализация", Slots: 2},
		verbose.StageDef{Key: "loop", Label: "LLM turn", Slots: 18},
		verbose.StageDef{Key: "final", Label: "финальный ответ", Slots: 2},
	)
}
