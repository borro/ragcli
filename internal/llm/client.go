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

// ChatRequester describes chat completion clients used by application packages.
type ChatRequester interface {
	SendRequestWithMetrics(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, RequestMetrics, error)
}

// Embedder представляет клиент для embeddings API.
type Embedder struct {
	client       *openai.Client
	model        string
	retryCount   int
	doEmbeddings func(ctx context.Context, req openai.EmbeddingRequestConverter) (openai.EmbeddingResponse, error)
}

// EmbeddingRequester describes embeddings clients used by application packages.
type EmbeddingRequester interface {
	CreateEmbeddingsWithMetrics(ctx context.Context, inputs []string) ([][]float32, EmbeddingMetrics, error)
}

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
func (c *Client) SendRequest(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
	resp, _, err := c.SendRequestWithMetrics(ctx, req)
	if err != nil {
		return nil, err
	}

	return resp, nil
}

// SendRequestWithMetrics отправляет запрос к LLM API и возвращает ответ вместе с метриками.
func (c *Client) SendRequestWithMetrics(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, RequestMetrics, error) {
	toolChoice := toolChoiceLabel(req.ToolChoice)

	for attempt := 1; attempt <= c.retryCount+1; attempt++ {
		select {
		case <-ctx.Done():
			return nil, RequestMetrics{}, ctx.Err()
		default:
		}

		slog.Debug("llm request started",
			"model", requestModel(c.model, req.Model),
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
				"duration", float64(metrics.Duration.Round(time.Millisecond))/float64(time.Second),
				"choice_count", metrics.ChoiceCount,
				"has_tool_calls", metrics.HasToolCalls,
				"prompt_tokens", metrics.PromptTokens,
				"completion_tokens", metrics.CompletionTokens,
				"total_tokens", metrics.TotalTokens,
				"tokens_per_second", metrics.TokensPerSecond,
			)
			return &resp, metrics, nil
		}

		slog.Debug("llm request failed",
			"model", requestModel(c.model, req.Model),
			"attempt", attempt,
			"message_count", len(req.Messages),
			"tool_count", len(req.Tools),
			"tool_choice", toolChoice,
			"duration", float64(elapsed.Round(time.Millisecond))/float64(time.Second),
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
				return nil, RequestMetrics{}, ctx.Err()
			case <-time.After(time.Duration(delay) * time.Second):
			}
		}
	}

	return nil, RequestMetrics{}, fmt.Errorf("request failed after %d attempts", c.retryCount+1)
}

// doRequestOnce выполняет один HTTP запрос к API через go-openai.
func (c *Client) doRequestOnce(ctx context.Context, req openai.ChatCompletionRequest) (openai.ChatCompletionResponse, error) {
	if req.Model == "" {
		req.Model = c.model
	}
	return c.doRequest(ctx, req)
}

// Model возвращает имя модели
func (c *Client) Model() string {
	return c.model
}

func buildRequestMetrics(defaultModel string, attempt int, req openai.ChatCompletionRequest, resp openai.ChatCompletionResponse, duration time.Duration) RequestMetrics {
	metrics := RequestMetrics{
		Model:            requestModel(defaultModel, req.Model),
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

func requestModel(defaultModel string, reqModel string) string {
	if reqModel != "" {
		return reqModel
	}
	return defaultModel
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
func (e *Embedder) CreateEmbeddingsWithMetrics(ctx context.Context, inputs []string) ([][]float32, EmbeddingMetrics, error) {
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
			return nil, EmbeddingMetrics{}, ctx.Err()
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
				"duration", float64(metrics.Duration.Round(time.Millisecond))/float64(time.Second),
				"prompt_tokens", metrics.PromptTokens,
				"total_tokens", metrics.TotalTokens,
			)
			return vectors, metrics, nil
		}

		slog.Debug("embedding request failed",
			"model", e.model,
			"attempt", attempt,
			"input_count", len(req.Input),
			"duration", float64(elapsed.Round(time.Millisecond))/float64(time.Second),
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
				return nil, EmbeddingMetrics{}, ctx.Err()
			case <-time.After(time.Duration(delay) * time.Second):
			}
		}
	}

	return nil, EmbeddingMetrics{}, fmt.Errorf("embedding request failed after %d attempts", e.retryCount+1)
}

// Model возвращает имя embeddings модели.
func (e *Embedder) Model() string {
	return e.model
}
