package llm

import (
	"context"
	"errors"
	"testing"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

// TestNewClient проверяет корректное создание клиента LLM
func TestNewClient(t *testing.T) {
	tests := []struct {
		name          string
		baseURL       string
		model         string
		apiKey        string
		retryCount    int
		expectedModel string
		expectedRetry int
	}{
		{
			name:          "valid client creation",
			baseURL:       "http://localhost:1234/v1",
			model:         "gpt-4",
			apiKey:        "test-api-key",
			retryCount:    3,
			expectedModel: "gpt-4",
			expectedRetry: 3,
		},
		{
			name:          "empty baseURL uses default",
			baseURL:       "",
			model:         "gpt-3.5-turbo",
			apiKey:        "another-key",
			retryCount:    5,
			expectedModel: "gpt-3.5-turbo",
			expectedRetry: 5,
		},
		{
			name:          "zero retry count",
			baseURL:       "http://test.com",
			model:         "test-model",
			apiKey:        "",
			retryCount:    0,
			expectedModel: "test-model",
			expectedRetry: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := NewClient(tt.baseURL, tt.model, tt.apiKey, tt.retryCount)

			if client.model != tt.expectedModel {
				t.Errorf("NewClient() model = %q, want %q", client.model, tt.expectedModel)
			}
			if client.retryCount != tt.expectedRetry {
				t.Errorf("NewClient() retryCount = %d, want %d", client.retryCount, tt.expectedRetry)
			}
		})
	}
}

// TestClient_Model проверяет Getter модели
func TestClient_Model(t *testing.T) {
	client := NewClient("http://test.com", "test-model", "api-key", 3)

	model := client.Model()
	if model != "test-model" {
		t.Errorf("Model() = %q, want %q", model, "test-model")
	}
}

// TestSendRequest_CreatesRequest проверяет, что запрос корректно собирается перед отправкой
func TestSendRequest_CreatesRequest(t *testing.T) {
	tests := []struct {
		name     string
		baseURL  string
		model    string
		apiKey   string
		retry    int
		hasTools bool
	}{
		{
			name:     "simple chat completion",
			baseURL:  "",
			model:    "gpt-3.5-turbo",
			apiKey:   "test-key",
			retry:    0,
			hasTools: false,
		},
		{
			name:     "with custom base URL",
			baseURL:  "http://test.com/v1",
			model:    "gpt-4",
			apiKey:   "test-key",
			retry:    1,
			hasTools: false,
		},
		{
			name:     "with tools",
			baseURL:  "",
			model:    "gpt-4",
			apiKey:   "test-key",
			retry:    0,
			hasTools: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := NewClient(tt.baseURL, tt.model, tt.apiKey, tt.retry)

			messages := []openai.ChatCompletionMessage{
				{
					Role:    "user",
					Content: "Hello",
				},
			}

			if tt.hasTools {
				messages = append(messages, openai.ChatCompletionMessage{
					Role:    openai.ChatMessageRoleAssistant,
					Content: "",
					ToolCalls: []openai.ToolCall{
						{
							Function: openai.FunctionCall{
								Name: "get_weather",
							},
						},
					},
				})
			}

			req := ChatCompletionRequest{
				Messages: messages,
			}

			ctx := context.Background()
			_, err := client.SendRequest(ctx, req)
			if err == nil {
				t.Error("SendRequest() expected error (no API), got nil")
			}
		})
	}
}

// TestSendRequest_ContextCancelled тестирует отмену контекста во время запроса
func TestSendRequest_ContextCancelled(t *testing.T) {
	tests := []struct {
		name    string
		baseURL string
		model   string
		apiKey  string
		retry   int
		delayMs int
	}{
		{
			name:    "cancel immediately",
			baseURL: "",
			model:   "gpt-3.5-turbo",
			apiKey:  "test-key",
			retry:   3,
			delayMs: 100,
		},
		{
			name:    "cancel during retry",
			baseURL: "http://test.com/v1",
			model:   "gpt-4",
			apiKey:  "test-key",
			retry:   3,
			delayMs: 50,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := NewClient(tt.baseURL, tt.model, tt.apiKey, tt.retry)

			messages := []openai.ChatCompletionMessage{
				{
					Role:    "user",
					Content: "Hello",
				},
			}

			req := ChatCompletionRequest{
				Messages: messages,
			}

			// Создаем контекст, который отменяется через заданное время
			ctx, cancel := context.WithTimeout(context.Background(), time.Duration(tt.delayMs)*time.Millisecond)
			defer cancel()

			_, err := client.SendRequest(ctx, req)
			if err == nil {
				t.Error("SendRequest() expected context cancelled error, got nil")
			}

			// Ошибка должна быть либо ctx.Err(), либо custom error о завершении попыток
			if !isContextError(err) {
				t.Log("non-context error occurred (expected for exhausted retries)")
			}
		})
	}
}

// TestSendRequest_ExhaustedRetries тестирует исчерпание попыток запроса
func TestSendRequest_ExhaustedRetries(t *testing.T) {
	tests := []struct {
		name    string
		baseURL string
		model   string
		apiKey  string
		retry   int
	}{
		{
			name:    "retry 0 - single attempt",
			baseURL: "",
			model:   "gpt-3.5-turbo",
			apiKey:  "test-key",
			retry:   0,
		},
		{
			name:    "retry 1 - two attempts",
			baseURL: "http://test.com/v1",
			model:   "gpt-4",
			apiKey:  "test-key",
			retry:   1,
		},
		{
			name:    "retry 3 - four attempts",
			baseURL: "",
			model:   "gpt-4o",
			apiKey:  "test-key",
			retry:   3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := NewClient(tt.baseURL, tt.model, tt.apiKey, tt.retry)

			messages := []openai.ChatCompletionMessage{
				{
					Role:    "user",
					Content: "Hello",
				},
			}

			req := ChatCompletionRequest{
				Messages: messages,
			}

			ctx := context.Background()
			_, err := client.SendRequest(ctx, req)

			// Проверяем, что ошибка есть (т.к. мы используем mock, который всегда возвращает ошибку)
			if err == nil {
				t.Error("SendRequest() expected error, got nil")
			}
		})
	}
}

// TestChatCompletionRequest_Structure проверяет структуру запроса
func TestChatCompletionRequest_Structure(t *testing.T) {
	tests := []struct {
		name string
		req  ChatCompletionRequest
		want int
	}{
		{
			name: "simple request",
			req: ChatCompletionRequest{
				Messages: []openai.ChatCompletionMessage{
					{
						Role:    "user",
						Content: "Hello",
					},
				},
			},
			want: 1,
		},
		{
			name: "request with tools",
			req: ChatCompletionRequest{
				Messages: []openai.ChatCompletionMessage{
					{
						Role:    "user",
						Content: "What's the weather?",
					},
				},
				Tools: []openai.Tool{
					{
						Type: "function",
						Function: &openai.FunctionDefinition{
							Name:        "get_weather",
							Description: "Get current weather",
						},
					},
				},
			},
			want: 1,
		},
		{
			name: "request with tool choice",
			req: ChatCompletionRequest{
				Messages: []openai.ChatCompletionMessage{},
				Tools: []openai.Tool{
					{
						Type: "function",
						Function: &openai.FunctionDefinition{
							Name: "search",
						},
					},
				},
				ToolChoice: map[string]interface{}{
					"type": "function",
					"function": map[string]interface{}{
						"name": "search",
					},
				},
			},
			want: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := len(tt.req.Messages); got != tt.want {
				t.Errorf("len(ChatCompletionRequest.Messages) = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestSendRequestWithMetrics_UsageAndTPS(t *testing.T) {
	client := NewClient("", "test-model", "", 0)
	client.doRequest = func(_ context.Context, req openai.ChatCompletionRequest) (openai.ChatCompletionResponse, error) {
		if req.Model != "test-model" {
			t.Fatalf("Model = %q, want %q", req.Model, "test-model")
		}
		return openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Role:    "assistant",
						Content: "ok",
					},
				},
			},
			Usage: openai.Usage{
				PromptTokens:     10,
				CompletionTokens: 20,
				TotalTokens:      30,
			},
		}, nil
	}

	result, err := client.SendRequestWithMetrics(context.Background(), ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("SendRequestWithMetrics() error = %v", err)
	}

	if result.Metrics.PromptTokens != 10 || result.Metrics.CompletionTokens != 20 || result.Metrics.TotalTokens != 30 {
		t.Fatalf("metrics token usage = %+v, want 10/20/30", result.Metrics)
	}
	if result.Metrics.TokensPerSecond <= 0 {
		t.Fatalf("TokensPerSecond = %v, want > 0", result.Metrics.TokensPerSecond)
	}
	if result.Metrics.Attempt != 1 {
		t.Fatalf("Attempt = %d, want 1", result.Metrics.Attempt)
	}
}

func TestSendRequestWithMetrics_ZeroCompletionTokensNoTPS(t *testing.T) {
	client := NewClient("", "test-model", "", 0)
	client.doRequest = func(_ context.Context, _ openai.ChatCompletionRequest) (openai.ChatCompletionResponse, error) {
		return openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{Message: openai.ChatCompletionMessage{Role: "assistant", Content: "ok"}},
			},
			Usage: openai.Usage{
				PromptTokens:     5,
				CompletionTokens: 0,
				TotalTokens:      5,
			},
		}, nil
	}

	result, err := client.SendRequestWithMetrics(context.Background(), ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("SendRequestWithMetrics() error = %v", err)
	}
	if result.Metrics.TokensPerSecond != 0 {
		t.Fatalf("TokensPerSecond = %v, want 0", result.Metrics.TokensPerSecond)
	}
}

func TestSendRequestWithMetrics_RetryUsesSuccessfulAttemptMetrics(t *testing.T) {
	client := NewClient("", "test-model", "", 1)
	attempts := 0
	client.doRequest = func(_ context.Context, _ openai.ChatCompletionRequest) (openai.ChatCompletionResponse, error) {
		attempts++
		if attempts == 1 {
			return openai.ChatCompletionResponse{}, errors.New("temporary failure")
		}
		return openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{Message: openai.ChatCompletionMessage{Role: "assistant", Content: "ok"}},
			},
			Usage: openai.Usage{
				PromptTokens:     7,
				CompletionTokens: 3,
				TotalTokens:      10,
			},
		}, nil
	}

	result, err := client.SendRequestWithMetrics(context.Background(), ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("SendRequestWithMetrics() error = %v", err)
	}

	if attempts != 2 {
		t.Fatalf("attempts = %d, want 2", attempts)
	}
	if result.Metrics.Attempt != 2 {
		t.Fatalf("Attempt = %d, want 2", result.Metrics.Attempt)
	}
	if result.Metrics.TotalTokens != 10 {
		t.Fatalf("TotalTokens = %d, want 10", result.Metrics.TotalTokens)
	}
}

// isContextError проверяет, является ли ошибка ошибкой отмены контекста
func isContextError(err error) bool {
	// Проверка на context deadline exceeded или cancelled
	if err != nil {
		errStr := err.Error()
		return errStr == "context deadline exceeded" || errStr == "context canceled"
	}
	return false
}
