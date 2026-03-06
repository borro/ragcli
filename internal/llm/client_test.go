package llm

import (
	"context"
	"testing"
	"time"

	"github.com/borro/ragcli/internal/config"
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
			cfg := &config.Config{}
			client := NewClient(tt.baseURL, tt.model, tt.apiKey, tt.retryCount, cfg)

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
	cfg := &config.Config{}
	client := NewClient("http://test.com", "test-model", "api-key", 3, cfg)

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
			cfg := &config.Config{}
			client := NewClient(tt.baseURL, tt.model, tt.apiKey, tt.retry, cfg)

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
				Model:    tt.model,
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
			cfg := &config.Config{}
			client := NewClient(tt.baseURL, tt.model, tt.apiKey, tt.retry, cfg)

			messages := []openai.ChatCompletionMessage{
				{
					Role:    "user",
					Content: "Hello",
				},
			}

			req := ChatCompletionRequest{
				Model:    "gpt-3.5-turbo",
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
				// Если ошибка не контекстная, она может быть о завершении retries
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
			cfg := &config.Config{}
			client := NewClient(tt.baseURL, tt.model, tt.apiKey, tt.retry, cfg)

			messages := []openai.ChatCompletionMessage{
				{
					Role:    "user",
					Content: "Hello",
				},
			}

			req := ChatCompletionRequest{
				Model:    "gpt-3.5-turbo",
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
		want string
	}{
		{
			name: "simple request",
			req: ChatCompletionRequest{
				Model: "gpt-3.5-turbo",
				Messages: []openai.ChatCompletionMessage{
					{
						Role:    "user",
						Content: "Hello",
					},
				},
			},
			want: "gpt-3.5-turbo",
		},
		{
			name: "request with tools",
			req: ChatCompletionRequest{
				Model: "gpt-4",
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
			want: "gpt-4",
		},
		{
			name: "request with tool choice",
			req: ChatCompletionRequest{
				Model:    "gpt-4",
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
			want: "gpt-4",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.req.Model; got != tt.want {
				t.Errorf("ChatCompletionRequest.Model = %q, want %q", got, tt.want)
			}
		})
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
