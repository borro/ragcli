package processor

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/borro/ragcli/internal/llm"
	openai "github.com/sashabaranov/go-openai"
)

type scriptedRequester struct {
	responses []*llm.ChatCompletionResponse
	errs      []error
	requests  []llm.ChatCompletionRequest
}

func (s *scriptedRequester) SendRequest(_ context.Context, req llm.ChatCompletionRequest) (*llm.ChatCompletionResponse, error) {
	s.requests = append(s.requests, req)
	index := len(s.requests) - 1

	if index < len(s.errs) && s.errs[index] != nil {
		return nil, s.errs[index]
	}

	if index >= len(s.responses) {
		return nil, context.DeadlineExceeded
	}

	return s.responses[index], nil
}

func TestRunTools_FinalAnswerWithoutTools(t *testing.T) {
	client := &scriptedRequester{
		responses: []*llm.ChatCompletionResponse{
			chatResponse(message("assistant", "Готовый ответ", nil)),
		},
	}

	result, err := RunTools(context.Background(), client, "/tmp/file.txt", "Что в файле?")
	if err != nil {
		t.Fatalf("RunTools() error = %v", err)
	}

	if result != "Готовый ответ" {
		t.Fatalf("RunTools() result = %q, want %q", result, "Готовый ответ")
	}
	if len(client.requests) != 1 {
		t.Fatalf("requests = %d, want 1", len(client.requests))
	}
	if len(client.requests[0].Tools) != 2 {
		t.Fatalf("tools sent = %d, want 2", len(client.requests[0].Tools))
	}
}

func TestRunTools_MultiStepToolLoop(t *testing.T) {
	client := &scriptedRequester{
		responses: []*llm.ChatCompletionResponse{
			chatResponse(message("assistant", "", []openai.ToolCall{
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

	result, err := RunTools(context.Background(), client, filePath, "Что сказано про архитектуру?")
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
}

func TestRunTools_StopsAfterMaxTurns(t *testing.T) {
	responses := make([]*llm.ChatCompletionResponse, 0, toolsMaxTurns)
	for i := 0; i < toolsMaxTurns; i++ {
		responses = append(responses, chatResponse(message("assistant", "", []openai.ToolCall{
			toolCall("loop-call", "search_file", `{"query":"x"}`),
		})))
	}

	client := &scriptedRequester{responses: responses}
	filePath := writeTempFile(t, "x\n")

	_, err := RunTools(context.Background(), client, filePath, "loop?")
	if err == nil || !strings.Contains(err.Error(), "exceeded") {
		t.Fatalf("RunTools() error = %v, want max-turns error", err)
	}
}

func TestRunTools_ToolErrorJSONDoesNotBreakLoop(t *testing.T) {
	client := &scriptedRequester{
		responses: []*llm.ChatCompletionResponse{
			chatResponse(message("assistant", "", []openai.ToolCall{
				toolCall("bad-call", "read_lines", `{"start_line":"bad","end_line":2}`),
			})),
			chatResponse(message("assistant", "Не удалось прочитать строки, аргументы были некорректны", nil)),
		},
	}

	filePath := writeTempFile(t, "one\ntwo\n")
	result, err := RunTools(context.Background(), client, filePath, "test")
	if err != nil {
		t.Fatalf("RunTools() error = %v", err)
	}
	if !strings.Contains(result, "некоррект") {
		t.Fatalf("RunTools() result = %q, want tool-error-based answer", result)
	}

	lastMessages := client.requests[1].Messages
	toolMsg := lastMessages[len(lastMessages)-1]
	var payload map[string]string
	if err := json.Unmarshal([]byte(toolMsg.Content), &payload); err != nil {
		t.Fatalf("json.Unmarshal(tool message) error = %v", err)
	}
	if payload["error"] == "" {
		t.Fatalf("tool error payload = %#v, want error field", payload)
	}
}

func assertToolMessage(t *testing.T, messages []llm.Message, toolCallID string, name string) {
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

func chatResponse(msg llm.Message) *llm.ChatCompletionResponse {
	return &llm.ChatCompletionResponse{
		Choices: []openai.ChatCompletionChoice{
			{
				Message: msg,
			},
		},
	}
}

func message(role string, content string, toolCalls []openai.ToolCall) llm.Message {
	return llm.Message{
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
