package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"testing"

	"github.com/borro/ragcli/internal/llm"
	"github.com/borro/ragcli/internal/verbose"
	openai "github.com/sashabaranov/go-openai"
)

type scriptedRequester struct {
	responses []*openai.ChatCompletionResponse
	errs      []error
	requests  []openai.ChatCompletionRequest
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

func TestRunTools_FinalAnswerWithoutTools(t *testing.T) {
	client := &scriptedRequester{
		responses: []*openai.ChatCompletionResponse{
			chatResponse(message("assistant", "Готовый ответ", nil)),
		},
	}

	result, err := Run(context.Background(), client, "/tmp/file.txt", "Что в файле?", nil)
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

func TestRunTools_VerboseEmitsSingleFinalCompletionLine(t *testing.T) {
	client := &scriptedRequester{
		responses: []*openai.ChatCompletionResponse{
			chatResponse(message("assistant", "Готовый ответ", nil)),
		},
	}
	var stderr bytes.Buffer

	result, err := Run(context.Background(), client, "/tmp/file.txt", "Что в файле?", verbose.NewPlan(verbose.New(&stderr, true, false), "tools",
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

func TestRunTools_EmptyFinalAnswerRetriesWithoutTools(t *testing.T) {
	client := &scriptedRequester{
		responses: []*openai.ChatCompletionResponse{
			chatResponse(message("assistant", "", nil)),
			chatResponse(message("assistant", "Итоговый ответ после retry", nil)),
		},
	}

	result, err := Run(context.Background(), client, "/tmp/file.txt", "Что в файле?", nil)
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

func TestRunTools_MultiStepToolLoop(t *testing.T) {
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

	result, err := Run(context.Background(), client, filePath, "Что сказано про архитектуру?", nil)
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

func TestRunTools_StopsAfterMaxTurns(t *testing.T) {
	responses := make([]*openai.ChatCompletionResponse, 0, toolsMaxTurns)
	for i := 0; i < toolsMaxTurns; i++ {
		responses = append(responses, chatResponse(message("assistant", "", []openai.ToolCall{
			toolCall("loop-call", "search_file", fmt.Sprintf(`{"query":"x","limit":1,"offset":%d}`, i)),
		})))
	}

	client := &scriptedRequester{responses: responses}
	filePath := writeTempFile(t, "x1\nx2\nx3\nx4\nx5\nx6\nx7\nx8\nx9\nx10\n")

	_, err := Run(context.Background(), client, filePath, "loop?", nil)
	var orchErr orchestrationError
	if !errorsAsOrchestration(err, &orchErr) || orchErr.Code != "orchestration_limit" {
		t.Fatalf("RunTools() error = %v, want orchestration_limit", err)
	}
}

func TestRunTools_ToolErrorJSONDoesNotBreakLoop(t *testing.T) {
	client := &scriptedRequester{
		responses: []*openai.ChatCompletionResponse{
			chatResponse(message("assistant", "", []openai.ToolCall{
				toolCall("bad-call", "read_lines", `{"start_line":"bad","end_line":2}`),
			})),
			chatResponse(message("assistant", "Не удалось прочитать строки, аргументы были некорректны", nil)),
		},
	}

	filePath := writeTempFile(t, "one\ntwo\n")
	result, err := Run(context.Background(), client, filePath, "test", nil)
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
	result, err := Run(context.Background(), client, filePath, "find alpha", nil)
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
	result, err := Run(context.Background(), client, filePath, "read", nil)
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
	result, err := Run(context.Background(), client, filePath, "read same area", nil)
	if err != nil {
		t.Fatalf("RunTools() error = %v", err)
	}
	if result != "Повторное чтение не добавило новых строк" {
		t.Fatalf("RunTools() result = %q, want fallback answer", result)
	}
}

func TestToolLoopState_DuplicateSearchMarksCachedPayload(t *testing.T) {
	state := newToolLoopState(writeTempFile(t, "alpha\nbeta\n"))

	first, _, err := state.executeToolCall(toolCall("call-1", "search_file", `{"query":"alpha"}`))
	if err != nil {
		t.Fatalf("executeToolCall(first) error = %v", err)
	}
	second, meta, err := state.executeToolCall(toolCall("call-2", "search_file", `{"query":"alpha"}`))
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

func TestToolLoopState_ReadAroundSeenLinesMarksNoProgress(t *testing.T) {
	state := newToolLoopState(writeTempFile(t, "one\ntwo\nthree\nfour\n"))

	_, firstMeta, err := state.executeToolCall(toolCall("call-1", "read_lines", `{"start_line":1,"end_line":3}`))
	if err != nil {
		t.Fatalf("executeToolCall(first) error = %v", err)
	}
	_, secondMeta, err := state.executeToolCall(toolCall("call-2", "read_around", `{"line":2,"before":1,"after":1}`))
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

func TestRunTools_SystemPromptMentionsAgentPolicy(t *testing.T) {
	client := &scriptedRequester{
		responses: []*openai.ChatCompletionResponse{
			chatResponse(message("assistant", "ok", nil)),
		},
	}

	_, err := Run(context.Background(), client, "/tmp/file.txt", "Что в файле?", nil)
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
	client := &scriptedRequester{
		responses: []*openai.ChatCompletionResponse{
			chatResponse(message("assistant", "ok", nil)),
		},
	}

	filePath := "/tmp/rest2push.log"
	_, err := Run(context.Background(), client, filePath, "Есть ли тут явные критические ошибки?", nil)
	if err != nil {
		t.Fatalf("RunTools() error = %v", err)
	}

	userPrompt := client.requests[0].Messages[1].Content
	if !strings.Contains(userPrompt, "Файл уже доступен через инструменты") {
		t.Fatalf("user prompt = %q, want tools file context", userPrompt)
	}
	if !strings.Contains(userPrompt, `Имя файла для анализа: "rest2push.log".`) {
		t.Fatalf("user prompt = %q, want file name", userPrompt)
	}
	if !strings.Contains(userPrompt, `Вопрос: "Есть ли тут явные критические ошибки?"`) {
		t.Fatalf("user prompt = %q, want original question", userPrompt)
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

	result, err := Run(context.Background(), client, "/tmp/rest2push.log", "Есть ли тут явные критические ошибки?", nil)
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

func writeTempFile(t *testing.T, content string) string {
	t.Helper()

	path := t.TempDir() + "/input.txt"
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}
	return path
}
