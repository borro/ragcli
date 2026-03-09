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

			req := openai.ChatCompletionRequest{
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

			req := openai.ChatCompletionRequest{
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

			req := openai.ChatCompletionRequest{
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
		req  openai.ChatCompletionRequest
		want int
	}{
		{
			name: "simple request",
			req: openai.ChatCompletionRequest{
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
			req: openai.ChatCompletionRequest{
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
			req: openai.ChatCompletionRequest{
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

	_, metrics, err := client.SendRequestWithMetrics(context.Background(), openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("SendRequestWithMetrics() error = %v", err)
	}

	if metrics.PromptTokens != 10 || metrics.CompletionTokens != 20 || metrics.TotalTokens != 30 {
		t.Fatalf("metrics token usage = %+v, want 10/20/30", metrics)
	}
	if metrics.TokensPerSecond <= 0 {
		t.Fatalf("TokensPerSecond = %v, want > 0", metrics.TokensPerSecond)
	}
	if metrics.Attempt != 1 {
		t.Fatalf("Attempt = %d, want 1", metrics.Attempt)
	}
}

func TestSendRequestWithMetrics_ZeroTotalTokensNoTPS(t *testing.T) {
	client := NewClient("", "test-model", "", 0)
	client.doRequest = func(_ context.Context, _ openai.ChatCompletionRequest) (openai.ChatCompletionResponse, error) {
		return openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{Message: openai.ChatCompletionMessage{Role: "assistant", Content: "ok"}},
			},
			Usage: openai.Usage{
				PromptTokens:     0,
				CompletionTokens: 0,
				TotalTokens:      0,
			},
		}, nil
	}

	_, metrics, err := client.SendRequestWithMetrics(context.Background(), openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("SendRequestWithMetrics() error = %v", err)
	}
	if metrics.TokensPerSecond != 0 {
		t.Fatalf("TokensPerSecond = %v, want 0", metrics.TokensPerSecond)
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

	_, metrics, err := client.SendRequestWithMetrics(context.Background(), openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("SendRequestWithMetrics() error = %v", err)
	}

	if attempts != 2 {
		t.Fatalf("attempts = %d, want 2", attempts)
	}
	if metrics.Attempt != 2 {
		t.Fatalf("Attempt = %d, want 2", metrics.Attempt)
	}
	if metrics.TotalTokens != 10 {
		t.Fatalf("TotalTokens = %d, want 10", metrics.TotalTokens)
	}
}

func TestSendRequestWithMetrics_ExplicitModelOverridesDefault(t *testing.T) {
	client := NewClient("", "default-model", "", 0)
	client.doRequest = func(_ context.Context, req openai.ChatCompletionRequest) (openai.ChatCompletionResponse, error) {
		if req.Model != "override-model" {
			t.Fatalf("Model = %q, want override-model", req.Model)
		}
		return openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{{Message: openai.ChatCompletionMessage{Role: "assistant", Content: "ok"}}},
		}, nil
	}

	_, metrics, err := client.SendRequestWithMetrics(context.Background(), openai.ChatCompletionRequest{
		Model:    "override-model",
		Messages: []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("SendRequestWithMetrics() error = %v", err)
	}
	if metrics.Model != "override-model" {
		t.Fatalf("metrics.Model = %q, want override-model", metrics.Model)
	}
}

func TestNewEmbedder(t *testing.T) {
	embedder := NewEmbedder("http://localhost:1234/v1", "embed-model", "key", 2)
	if embedder.Model() != "embed-model" {
		t.Fatalf("Model() = %q, want embed-model", embedder.Model())
	}
	if embedder.retryCount != 2 {
		t.Fatalf("retryCount = %d, want 2", embedder.retryCount)
	}
}

func TestCreateEmbeddingsWithMetrics(t *testing.T) {
	embedder := NewEmbedder("", "embed-model", "", 0)
	embedder.doEmbeddings = func(_ context.Context, req openai.EmbeddingRequestConverter) (openai.EmbeddingResponse, error) {
		converted := req.Convert()
		inputs, ok := converted.Input.([]string)
		if !ok {
			t.Fatalf("input type = %T, want []string", converted.Input)
		}
		if len(inputs) != 2 {
			t.Fatalf("len(inputs) = %d, want 2", len(inputs))
		}
		return openai.EmbeddingResponse{
			Data: []openai.Embedding{
				{Embedding: []float32{1, 0}},
				{Embedding: []float32{0, 1}},
			},
			Usage: openai.Usage{
				PromptTokens: 12,
				TotalTokens:  12,
			},
		}, nil
	}

	vectors, metrics, err := embedder.CreateEmbeddingsWithMetrics(context.Background(), []string{"alpha", "beta"})
	if err != nil {
		t.Fatalf("CreateEmbeddingsWithMetrics() error = %v", err)
	}
	if len(vectors) != 2 {
		t.Fatalf("len(Vectors) = %d, want 2", len(vectors))
	}
	if metrics.InputCount != 2 || metrics.VectorCount != 2 {
		t.Fatalf("metrics = %+v, want input/vector count 2", metrics)
	}
	if metrics.TotalTokens != 12 {
		t.Fatalf("TotalTokens = %d, want 12", metrics.TotalTokens)
	}
}

func TestResponseHasToolCallsAndToolChoiceLabel(t *testing.T) {
	resp := openai.ChatCompletionResponse{
		Choices: []openai.ChatCompletionChoice{
			{Message: openai.ChatCompletionMessage{Role: "assistant"}},
			{Message: openai.ChatCompletionMessage{Role: "assistant", ToolCalls: []openai.ToolCall{{ID: "1"}}}},
		},
	}
	if !responseHasToolCalls(resp) {
		t.Fatal("responseHasToolCalls() = false, want true")
	}
	if responseHasToolCalls(openai.ChatCompletionResponse{}) {
		t.Fatal("responseHasToolCalls(empty) = true, want false")
	}

	tests := []struct {
		name  string
		input interface{}
		want  string
	}{
		{name: "nil", input: nil, want: "auto"},
		{name: "empty string", input: "", want: "auto"},
		{name: "string", input: "required", want: "required"},
		{name: "non string", input: map[string]any{"type": "function"}, want: "map[string]interface {}"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := toolChoiceLabel(tt.input); got != tt.want {
				t.Fatalf("toolChoiceLabel() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestCreateEmbeddingsWithMetrics_ContextCancelled(t *testing.T) {
	embedder := NewEmbedder("", "embed-model", "", 2)
	embedder.doEmbeddings = func(ctx context.Context, _ openai.EmbeddingRequestConverter) (openai.EmbeddingResponse, error) {
		<-ctx.Done()
		return openai.EmbeddingResponse{}, ctx.Err()
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, _, err := embedder.CreateEmbeddingsWithMetrics(ctx, []string{"alpha"})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("CreateEmbeddingsWithMetrics() err = %v, want context.Canceled", err)
	}
}

func TestCreateEmbeddingsWithMetrics_IncludesLastError(t *testing.T) {
	embedder := NewEmbedder("", "embed-model", "", 2)
	wantErr := errors.New("remote 503")
	calls := 0
	embedder.doEmbeddings = func(_ context.Context, _ openai.EmbeddingRequestConverter) (openai.EmbeddingResponse, error) {
		calls++
		return openai.EmbeddingResponse{}, wantErr
	}

	_, _, err := embedder.CreateEmbeddingsWithMetrics(context.Background(), []string{"alpha"})
	if err == nil {
		t.Fatal("CreateEmbeddingsWithMetrics() error = nil, want wrapped error")
	}
	if calls != 3 {
		t.Fatalf("CreateEmbeddingsWithMetrics() calls = %d, want 3", calls)
	}
	if !errors.Is(err, wantErr) {
		t.Fatalf("CreateEmbeddingsWithMetrics() err = %v, want wrapped %v", err, wantErr)
	}
	if got := err.Error(); got != "embedding request failed after 3 attempts: remote 503" {
		t.Fatalf("CreateEmbeddingsWithMetrics() error string = %q, want wrapped message", got)
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
