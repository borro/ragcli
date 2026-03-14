package hybrid

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/borro/ragcli/internal/input"
	"github.com/borro/ragcli/internal/llm"
	"github.com/borro/ragcli/internal/localize"
	ragruntime "github.com/borro/ragcli/internal/rag"
	openai "github.com/sashabaranov/go-openai"
)

type scriptedRequester struct {
	responses []*openai.ChatCompletionResponse
	requests  []openai.ChatCompletionRequest
}

type embeddingRequesterFunc func(context.Context, []string) ([][]float32, llm.EmbeddingMetrics, error)

func (f embeddingRequesterFunc) CreateEmbeddingsWithMetrics(ctx context.Context, inputs []string) ([][]float32, llm.EmbeddingMetrics, error) {
	return f(ctx, inputs)
}

func TestMain(m *testing.M) {
	if err := localize.SetCurrent(localize.EN); err != nil {
		panic(err)
	}
	os.Exit(m.Run())
}

func (s *scriptedRequester) SendRequestWithMetrics(_ context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, llm.RequestMetrics, error) {
	s.requests = append(s.requests, req)
	index := len(s.requests) - 1
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
	}, nil
}

func fakeEmbedder() llm.EmbeddingRequester {
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

func defaultSearchOptions(t *testing.T) Options {
	t.Helper()
	return Options{
		Search: ragruntime.SearchOptions{
			TopK:           8,
			ChunkSize:      1000,
			ChunkOverlap:   200,
			IndexTTL:       time.Hour,
			IndexDir:       t.TempDir(),
			Rerank:         "heuristic",
			EmbeddingModel: "embed-v1",
		},
	}
}

func openSource(t *testing.T, path string) input.Source {
	t.Helper()
	handle, err := input.Open(path, nil)
	if err != nil {
		t.Fatalf("input.Open(%q) error = %v", path, err)
	}
	t.Cleanup(func() {
		_ = handle.Close()
	})
	return handle.Source()
}

func message(role string, content string, toolCalls []openai.ToolCall) openai.ChatCompletionMessage {
	return openai.ChatCompletionMessage{
		Role:      role,
		Content:   content,
		ToolCalls: toolCalls,
	}
}

func chatResponse(msg openai.ChatCompletionMessage) *openai.ChatCompletionResponse {
	return &openai.ChatCompletionResponse{
		Choices: []openai.ChatCompletionChoice{{Message: msg}},
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

func TestRunHybrid_PreloadsSearchAndVerificationBeforeFirstLLMRequest(t *testing.T) {
	filePath := filepath.Join(t.TempDir(), "retries.txt")
	if err := os.WriteFile(filePath, []byte("retry policy enabled\nbackoff is exponential\n"), 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	client := &scriptedRequester{
		responses: []*openai.ChatCompletionResponse{
			chatResponse(message(openai.ChatMessageRoleAssistant, "Retry policy is enabled.", nil)),
		},
	}

	result, err := Run(context.Background(), client, fakeEmbedder(), openSource(t, filePath), defaultSearchOptions(t), "What does it say about retry policy?", nil)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	if len(client.requests) != 1 {
		t.Fatalf("requests = %d, want 1", len(client.requests))
	}

	firstRequest := client.requests[0]
	if len(firstRequest.Messages) < 6 {
		t.Fatalf("message count = %d, want preloaded search and verification history", len(firstRequest.Messages))
	}

	searchCalls := collectToolCalls(firstRequest.Messages, "search_rag")
	if len(searchCalls) < 2 {
		t.Fatalf("searchCalls = %#v, want multiple fused seed search calls", searchCalls)
	}
	for _, call := range searchCalls {
		var args struct {
			Limit int `json:"limit"`
		}
		if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
			t.Fatalf("json.Unmarshal(search args) error = %v", err)
		}
		if args.Limit != 8 {
			t.Fatalf("search limit = %d, want TopK=8", args.Limit)
		}
	}

	if len(collectToolCalls(firstRequest.Messages, "read_lines")) == 0 && len(collectToolCalls(firstRequest.Messages, "read_around")) == 0 {
		t.Fatalf("messages = %#v, want synthetic verification reads in preloaded history", firstRequest.Messages)
	}
	if !strings.Contains(result, "Retry policy is enabled.") {
		t.Fatalf("result = %q, want final answer", result)
	}
	if !strings.Contains(result, "Sources:\n\nVerified:\n\n- "+filePath+":\n  1-2") {
		t.Fatalf("result = %q, want verified sources from synthetic reads", result)
	}
	if strings.Contains(result, "\n\nRetrieved:\n") {
		t.Fatalf("result = %q, overlapping retrieved evidence should be removed after verification", result)
	}
}

func TestRunHybrid_WeakSeedRetriesDirectAnswerWithoutToolUse(t *testing.T) {
	filePath := filepath.Join(t.TempDir(), "notes.txt")
	if err := os.WriteFile(filePath, []byte("retry policy enabled\nbackoff is exponential\n"), 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	client := &scriptedRequester{
		responses: []*openai.ChatCompletionResponse{
			chatResponse(message(openai.ChatMessageRoleAssistant, "Looks fine to me.", nil)),
			chatResponse(message(openai.ChatMessageRoleAssistant, "I verified the available evidence and still need a better query.", nil)),
		},
	}

	result, err := Run(context.Background(), client, fakeEmbedder(), openSource(t, filePath), defaultSearchOptions(t), "What does it say about networking?", nil)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	if !strings.Contains(result, "I verified the available evidence and still need a better query.") {
		t.Fatalf("result = %q, want retry-driven final answer", result)
	}
	if !strings.Contains(result, "Sources:\n\nVerified:\n\n- "+filePath+":\n  1-2") {
		t.Fatalf("result = %q, want preverified seed evidence to remain visible", result)
	}
	if len(client.requests) != 2 {
		t.Fatalf("requests = %d, want first-turn retry", len(client.requests))
	}
	lastMessage := client.requests[1].Messages[len(client.requests[1].Messages)-1]
	if lastMessage.Role != openai.ChatMessageRoleUser || lastMessage.Content != localize.T("tools.prompt.first_turn_tool_retry") {
		t.Fatalf("last retry message = %#v, want first-turn verification retry prompt", lastMessage)
	}
}

func TestRunHybrid_ToolLoopSplitsVerifiedSourcesAndIgnoresListFiles(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "a.txt"), []byte("retry policy enabled\nbackoff is exponential\n"), 0o600); err != nil {
		t.Fatalf("WriteFile(a.txt) error = %v", err)
	}

	client := &scriptedRequester{
		responses: []*openai.ChatCompletionResponse{
			chatResponse(message(openai.ChatMessageRoleAssistant, "Let me inspect the corpus.", []openai.ToolCall{
				toolCall("call-1", "list_files", `{"limit":10}`),
			})),
			chatResponse(message(openai.ChatMessageRoleAssistant, "", []openai.ToolCall{
				toolCall("call-2", "search_file", `{"path":"a.txt","query":"retry","limit":5}`),
			})),
			chatResponse(message(openai.ChatMessageRoleAssistant, "", []openai.ToolCall{
				toolCall("call-3", "read_lines", `{"path":"a.txt","start_line":1,"end_line":2}`),
			})),
			chatResponse(message(openai.ChatMessageRoleAssistant, "Retry policy is enabled and uses exponential backoff.", nil)),
		},
	}

	result, err := Run(context.Background(), client, fakeEmbedder(), openSource(t, dir), defaultSearchOptions(t), "What does it say about retry policy?", nil)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	if len(client.requests) != 4 {
		t.Fatalf("requests = %d, want 4", len(client.requests))
	}
	firstRequest := client.requests[0]
	if len(collectToolCalls(firstRequest.Messages, "search_rag")) == 0 {
		t.Fatalf("first request messages = %#v, want seeded search history", firstRequest.Messages)
	}
	if !strings.Contains(result, "Retry policy is enabled and uses exponential backoff.") {
		t.Fatalf("result = %q, want final answer", result)
	}
	if !strings.Contains(result, "Sources:\n\nVerified:\n\n- a.txt:\n  1-2") {
		t.Fatalf("result = %q, want verified source line range", result)
	}
	if strings.Contains(result, "\n\nRetrieved:\n") {
		t.Fatalf("result = %q, retrieved lines should be subtracted when verified lines overlap", result)
	}
	if strings.Contains(result, "\n- a.txt\n") {
		t.Fatalf("result = %q, list_files output must not become a source citation", result)
	}
}

func collectToolCalls(messages []openai.ChatCompletionMessage, name string) []openai.ToolCall {
	calls := make([]openai.ToolCall, 0, len(messages))
	for _, message := range messages {
		for _, call := range message.ToolCalls {
			if call.Function.Name == name {
				calls = append(calls, call)
			}
		}
	}
	return calls
}
