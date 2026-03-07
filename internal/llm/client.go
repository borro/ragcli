package llm

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

// Client представляет клиент для взаимодействия с OpenAI API через go-openai
type Client struct {
	client     *openai.Client
	model      string
	retryCount int
	doRequest  func(ctx context.Context, req openai.ChatCompletionRequest) (openai.ChatCompletionResponse, error)
}

// Embedder представляет клиент для embeddings API.
type Embedder struct {
	client       *openai.Client
	model        string
	retryCount   int
	doEmbeddings func(ctx context.Context, req openai.EmbeddingRequestConverter) (openai.EmbeddingResponse, error)
}

// Message представляет сообщение в чате
type Message = openai.ChatCompletionMessage

// ChatCompletionRequest представляет запрос к chat completion API
type ChatCompletionRequest struct {
	Messages   []openai.ChatCompletionMessage `json:"messages"`
	Tools      []Tool                         `json:"tools,omitempty"`
	ToolChoice interface{}                    `json:"tool_choice,omitempty"`
}

// FunctionDefinition represents function definition for tool calling
type FunctionDefinition = openai.FunctionDefinition

// Tool представляет собой определение инструмента
type Tool = openai.Tool

// ChatCompletionResponse представляет ответ от API
type ChatCompletionResponse = openai.ChatCompletionResponse

// ToolCall represents a tool call from the model
type ToolCall = openai.ToolCall

// RequestMetrics содержит метаданные и наблюдаемость по одному запросу к LLM.
type RequestMetrics struct {
	Model            string
	Attempt          int
	MessageCount     int
	ToolCount        int
	ToolChoice       string
	Duration         time.Duration
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
	TokensPerSecond  float64
	ChoiceCount      int
	HasToolCalls     bool
}

// RequestResult содержит ответ LLM и собранные метрики запроса.
type RequestResult struct {
	Response *ChatCompletionResponse
	Metrics  RequestMetrics
}

// EmbeddingMetrics содержит метрики одного embeddings запроса.
type EmbeddingMetrics struct {
	Model        string
	Attempt      int
	InputCount   int
	VectorCount  int
	Duration     time.Duration
	PromptTokens int
	TotalTokens  int
}

// EmbeddingResult содержит векторы и метрики embeddings запроса.
type EmbeddingResult struct {
	Vectors [][]float32
	Metrics EmbeddingMetrics
}

// NewClient создаёт новый LLM клиент, используя go-openai библиотеку
func NewClient(baseURL, model, apiKey string, retryCount int) *Client {
	config := openai.DefaultConfig(apiKey)
	if baseURL != "" {
		config.BaseURL = baseURL
	}

	client := &Client{
		client:     openai.NewClientWithConfig(config),
		model:      model,
		retryCount: retryCount,
	}
	client.doRequest = func(ctx context.Context, req openai.ChatCompletionRequest) (openai.ChatCompletionResponse, error) {
		return client.client.CreateChatCompletion(ctx, req)
	}
	return client
}

// NewEmbedder создаёт отдельный embeddings клиент.
func NewEmbedder(baseURL, model, apiKey string, retryCount int) *Embedder {
	config := openai.DefaultConfig(apiKey)
	if baseURL != "" {
		config.BaseURL = baseURL
	}

	embedder := &Embedder{
		client:     openai.NewClientWithConfig(config),
		model:      model,
		retryCount: retryCount,
	}
	embedder.doEmbeddings = func(ctx context.Context, req openai.EmbeddingRequestConverter) (openai.EmbeddingResponse, error) {
		return embedder.client.CreateEmbeddings(ctx, req)
	}
	return embedder
}

// SendRequest отправляет запрос к LLM API с механизмом retry и exponential backoff
func (c *Client) SendRequest(ctx context.Context, req ChatCompletionRequest) (*ChatCompletionResponse, error) {
	result, err := c.SendRequestWithMetrics(ctx, req)
	if err != nil {
		return nil, err
	}

	return result.Response, nil
}

// SendRequestWithMetrics отправляет запрос к LLM API и возвращает ответ вместе с метриками.
func (c *Client) SendRequestWithMetrics(ctx context.Context, req ChatCompletionRequest) (*RequestResult, error) {
	toolChoice := toolChoiceLabel(req.ToolChoice)

	for attempt := 1; attempt <= c.retryCount+1; attempt++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		slog.Debug("llm request started",
			"model", c.model,
			"attempt", attempt,
			"message_count", len(req.Messages),
			"tool_count", len(req.Tools),
			"tool_choice", toolChoice,
		)

		startedAt := time.Now()
		resp, err := c.doRequestOnce(ctx, req)
		elapsed := time.Since(startedAt)
		if err == nil {
			metrics := buildRequestMetrics(c.model, attempt, req, resp, elapsed)
			slog.Debug("llm request finished",
				"model", metrics.Model,
				"attempt", metrics.Attempt,
				"message_count", metrics.MessageCount,
				"tool_count", metrics.ToolCount,
				"tool_choice", metrics.ToolChoice,
				"duration_ms", metrics.Duration.Milliseconds(),
				"choice_count", metrics.ChoiceCount,
				"has_tool_calls", metrics.HasToolCalls,
				"prompt_tokens", metrics.PromptTokens,
				"completion_tokens", metrics.CompletionTokens,
				"total_tokens", metrics.TotalTokens,
				"tokens_per_second", metrics.TokensPerSecond,
			)
			return &RequestResult{
				Response: &resp,
				Metrics:  metrics,
			}, nil
		}

		slog.Debug("llm request failed",
			"model", c.model,
			"attempt", attempt,
			"message_count", len(req.Messages),
			"tool_count", len(req.Tools),
			"tool_choice", toolChoice,
			"duration_ms", elapsed.Milliseconds(),
			"error", err,
		)

		// Проверяем тип ошибки и решаем стоит ли retry
		if attempt <= c.retryCount {
			delay := 1 << (attempt - 1) // exponential backoff: 1, 2, 4, 8 seconds
			slog.Debug("retrying llm request",
				"model", c.model,
				"failed_attempt", attempt,
				"next_attempt", attempt+1,
				"delay_seconds", delay,
				"reason", err,
			)
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(time.Duration(delay) * time.Second):
			}
		}
	}

	return nil, fmt.Errorf("request failed after %d attempts", c.retryCount+1)
}

// doRequestOnce выполняет один HTTP запрос к API через go-openai.
func (c *Client) doRequestOnce(ctx context.Context, req ChatCompletionRequest) (openai.ChatCompletionResponse, error) {
	openaiReq := openai.ChatCompletionRequest{
		Model:    c.model,
		Messages: req.Messages,
		Tools:    req.Tools,
	}

	if req.ToolChoice != nil {
		openaiReq.ToolChoice = req.ToolChoice
	}

	return c.doRequest(ctx, openaiReq)
}

// Model возвращает имя модели
func (c *Client) Model() string {
	return c.model
}

func buildRequestMetrics(model string, attempt int, req ChatCompletionRequest, resp openai.ChatCompletionResponse, duration time.Duration) RequestMetrics {
	metrics := RequestMetrics{
		Model:            model,
		Attempt:          attempt,
		MessageCount:     len(req.Messages),
		ToolCount:        len(req.Tools),
		ToolChoice:       toolChoiceLabel(req.ToolChoice),
		Duration:         duration,
		PromptTokens:     resp.Usage.PromptTokens,
		CompletionTokens: resp.Usage.CompletionTokens,
		TotalTokens:      resp.Usage.TotalTokens,
		ChoiceCount:      len(resp.Choices),
		HasToolCalls:     responseHasToolCalls(resp),
	}

	if metrics.CompletionTokens > 0 && duration > 0 {
		metrics.TokensPerSecond = float64(metrics.CompletionTokens) / duration.Seconds()
	}

	return metrics
}

func responseHasToolCalls(resp openai.ChatCompletionResponse) bool {
	for _, choice := range resp.Choices {
		if len(choice.Message.ToolCalls) > 0 {
			return true
		}
	}
	return false
}

func toolChoiceLabel(toolChoice interface{}) string {
	if toolChoice == nil {
		return "auto"
	}

	switch value := toolChoice.(type) {
	case string:
		if value == "" {
			return "auto"
		}
		return value
	default:
		return fmt.Sprintf("%T", toolChoice)
	}
}

// CreateEmbeddingsWithMetrics отправляет embeddings запрос и возвращает метрики.
func (e *Embedder) CreateEmbeddingsWithMetrics(ctx context.Context, inputs []string) (*EmbeddingResult, error) {
	req := openai.EmbeddingRequestStrings{
		Input: []string{},
		Model: openai.EmbeddingModel(e.model),
	}
	if len(inputs) > 0 {
		req.Input = inputs
	}

	for attempt := 1; attempt <= e.retryCount+1; attempt++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		slog.Debug("embedding request started",
			"model", e.model,
			"attempt", attempt,
			"input_count", len(req.Input),
		)

		startedAt := time.Now()
		resp, err := e.doEmbeddings(ctx, req)
		elapsed := time.Since(startedAt)
		if err == nil {
			vectors := make([][]float32, 0, len(resp.Data))
			for _, item := range resp.Data {
				vectors = append(vectors, item.Embedding)
			}
			metrics := EmbeddingMetrics{
				Model:        e.model,
				Attempt:      attempt,
				InputCount:   len(req.Input),
				VectorCount:  len(vectors),
				Duration:     elapsed,
				PromptTokens: resp.Usage.PromptTokens,
				TotalTokens:  resp.Usage.TotalTokens,
			}
			slog.Debug("embedding request finished",
				"model", metrics.Model,
				"attempt", metrics.Attempt,
				"input_count", metrics.InputCount,
				"vector_count", metrics.VectorCount,
				"duration_ms", metrics.Duration.Milliseconds(),
				"prompt_tokens", metrics.PromptTokens,
				"total_tokens", metrics.TotalTokens,
			)
			return &EmbeddingResult{Vectors: vectors, Metrics: metrics}, nil
		}

		slog.Debug("embedding request failed",
			"model", e.model,
			"attempt", attempt,
			"input_count", len(req.Input),
			"duration_ms", elapsed.Milliseconds(),
			"error", err,
		)

		if attempt <= e.retryCount {
			delay := 1 << (attempt - 1)
			slog.Debug("retrying embedding request",
				"model", e.model,
				"failed_attempt", attempt,
				"next_attempt", attempt+1,
				"delay_seconds", delay,
				"reason", err,
			)
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(time.Duration(delay) * time.Second):
			}
		}
	}

	return nil, fmt.Errorf("embedding request failed after %d attempts", e.retryCount+1)
}

// Model возвращает имя embeddings модели.
func (e *Embedder) Model() string {
	return e.model
}
