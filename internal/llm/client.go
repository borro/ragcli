package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

// Client представляет клиент для взаимодействия с OpenAI API через go-openai
type Client struct {
	client     *openai.Client
	model      string
	retryCount int
	baseURL    string
	apiKey     string
	doRequest  func(ctx context.Context, req openai.ChatCompletionRequest) (openai.ChatCompletionResponse, error)
	doHTTP     func(req *http.Request) (*http.Response, error)

	contextLengthMu       sync.Mutex
	contextLengthResolved bool
	contextLengthValue    int
	contextLengthSource   string
	contextLengthErr      error
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

const autoContextProbePromptTokens = 320000

var (
	reNContext            = regexp.MustCompile(`(?i)n_ctx\s*\((\d+)\)`)
	reMaximumContextLen   = regexp.MustCompile(`(?i)maximum context length[^0-9]*(\d+)`)
	reMaxContextLen       = regexp.MustCompile(`(?i)max(?:imum)? context length[^0-9]*(\d+)`)
	reGenericContextLimit = regexp.MustCompile(`(?i)context length[^0-9]*(\d+)`)
	errModelNotLoaded     = fmt.Errorf("matched model has no loaded instances")
)

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
		baseURL:    config.BaseURL,
		apiKey:     apiKey,
	}
	client.doRequest = func(ctx context.Context, req openai.ChatCompletionRequest) (openai.ChatCompletionResponse, error) {
		return client.client.CreateChatCompletion(ctx, req)
	}
	client.doHTTP = config.HTTPClient.Do
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

// ResolveAutoContextLength attempts to detect model context length.
// Source can be "lmstudio_api" or "probe_from_error".
func (c *Client) ResolveAutoContextLength(ctx context.Context) (int, string, error) {
	c.contextLengthMu.Lock()
	if c.contextLengthResolved {
		value := c.contextLengthValue
		source := c.contextLengthSource
		err := c.contextLengthErr
		c.contextLengthMu.Unlock()
		slog.Debug("auto context length cache hit",
			"model", c.model,
			"base_url", c.baseURL,
			"value", value,
			"source", source,
			"has_error", err != nil,
		)
		return value, source, err
	}
	c.contextLengthMu.Unlock()

	slog.Debug("auto context length resolve started",
		"model", c.model,
		"base_url", c.baseURL,
	)
	value, source, err := c.resolveAutoContextLengthNoCache(ctx)

	c.contextLengthMu.Lock()
	c.contextLengthResolved = true
	c.contextLengthValue = value
	c.contextLengthSource = source
	c.contextLengthErr = err
	c.contextLengthMu.Unlock()

	slog.Debug("auto context length resolve finished",
		"model", c.model,
		"base_url", c.baseURL,
		"value", value,
		"source", source,
		"error", err,
	)
	return value, source, err
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

	if metrics.TotalTokens > 0 && duration > 0 {
		metrics.TokensPerSecond = float64(metrics.TotalTokens) / duration.Seconds()
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

func (c *Client) resolveAutoContextLengthNoCache(ctx context.Context) (int, string, error) {
	modelsURLs := lmStudioModelsURLs(c.baseURL)
	for _, modelsURL := range modelsURLs {
		slog.Debug("auto context length trying lmstudio models endpoint",
			"model", c.model,
			"url", modelsURL,
		)
		length, err := c.resolveContextLengthFromLMStudioModels(ctx, modelsURL)
		if err == nil {
			slog.Debug("auto context length resolved from lmstudio models endpoint",
				"model", c.model,
				"url", modelsURL,
				"value", length,
			)
			return length, "lmstudio_api", nil
		}
		if errors.Is(err, errModelNotLoaded) {
			slog.Debug("auto context length model matched but not loaded, sending warmup request",
				"model", c.model,
				"url", modelsURL,
			)
			if warmupErr := c.warmupModel(ctx); warmupErr == nil {
				retryLength, retryErr := c.resolveContextLengthFromLMStudioModels(ctx, modelsURL)
				if retryErr == nil {
					slog.Debug("auto context length resolved from lmstudio models endpoint after warmup",
						"model", c.model,
						"url", modelsURL,
						"value", retryLength,
					)
					return retryLength, "lmstudio_api", nil
				}
				slog.Debug("auto context length warmup retry failed",
					"model", c.model,
					"url", modelsURL,
					"error", retryErr,
				)
			} else {
				slog.Debug("auto context length warmup request failed",
					"model", c.model,
					"url", modelsURL,
					"error", warmupErr,
				)
			}
		}
		slog.Debug("auto context length lmstudio models endpoint failed",
			"model", c.model,
			"url", modelsURL,
			"error", err,
		)
	}

	slog.Debug("auto context length trying probe from error", "model", c.model)
	length, err := c.resolveContextLengthFromProbe(ctx)
	if err == nil {
		slog.Debug("auto context length resolved from probe", "model", c.model, "value", length)
		return length, "probe_from_error", nil
	}
	slog.Debug("auto context length probe failed", "model", c.model, "error", err)

	return 0, "", fmt.Errorf("failed to resolve model context length")
}

func (c *Client) resolveContextLengthFromLMStudioModels(ctx context.Context, modelsURL string) (int, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, modelsURL, nil)
	if err != nil {
		return 0, err
	}
	req.Header.Set("Accept", "application/json")
	if strings.TrimSpace(c.apiKey) != "" {
		req.Header.Set("Authorization", "Bearer "+strings.TrimSpace(c.apiKey))
	}

	resp, err := c.doHTTP(req)
	if err != nil {
		return 0, err
	}
	defer func() {
		if closeErr := resp.Body.Close(); closeErr != nil {
			slog.Debug("auto context length models response body close failed",
				"model", c.model,
				"url", modelsURL,
				"error", closeErr,
			)
		}
	}()

	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return 0, fmt.Errorf("models endpoint status %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, err
	}

	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		return 0, err
	}

	modelsRaw, field, ok := extractModelsArray(payload)
	if !ok || len(modelsRaw) == 0 {
		return 0, fmt.Errorf("models payload does not contain models list")
	}
	slog.Debug("auto context length parsed models payload",
		"model", c.model,
		"url", modelsURL,
		"models_field", field,
		"models_count", len(modelsRaw),
	)

	modelMatched := false
	modelHasLoadedInstances := false
	for _, raw := range modelsRaw {
		modelObj, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if !lmStudioModelMatches(modelObj, c.model) {
			continue
		}
		modelMatched = true
		if loaded, ok := modelObj["loaded_instances"].([]any); ok && len(loaded) > 0 {
			modelHasLoadedInstances = true
		}
		if length, ok := extractLMStudioContextLength(modelObj, c.model); ok {
			slog.Debug("auto context length extracted from models payload",
				"model", c.model,
				"url", modelsURL,
				"value", length,
			)
			return length, nil
		}
	}
	if modelMatched && !modelHasLoadedInstances {
		return 0, errModelNotLoaded
	}

	return 0, fmt.Errorf("context length not found in models response")
}

func (c *Client) resolveContextLengthFromProbe(ctx context.Context) (int, error) {
	slog.Debug("auto context length probe request started", "model", c.model, "probe_prompt_tokens", autoContextProbePromptTokens)
	req := openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: strings.Repeat("x ", autoContextProbePromptTokens),
			},
		},
		MaxCompletionTokens: 1,
	}

	_, err := c.doRequestOnce(ctx, req)
	if err == nil {
		slog.Debug("auto context length probe did not fail as expected", "model", c.model)
		return 0, fmt.Errorf("probe request unexpectedly succeeded")
	}

	if value, ok := extractContextLengthFromError(err.Error()); ok {
		slog.Debug("auto context length probe parsed value from error", "model", c.model, "value", value)
		return value, nil
	}

	slog.Debug("auto context length probe error unparsable", "model", c.model, "error", err)
	return 0, err
}

func lmStudioModelsURLs(baseURL string) []string {
	parsed, err := url.Parse(strings.TrimSpace(baseURL))
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return nil
	}

	path := strings.TrimSuffix(parsed.Path, "/")
	var candidates []string

	if strings.HasSuffix(path, "/v1") {
		prefix := strings.TrimSuffix(path, "/v1")
		if prefix == "" {
			candidates = append(candidates, "/api/v1/models")
		} else {
			candidates = append(candidates, prefix+"/api/v1/models")
		}
	}
	candidates = append(candidates, "/api/v1/models")

	seen := make(map[string]struct{}, len(candidates))
	urls := make([]string, 0, len(candidates))
	for _, candidatePath := range candidates {
		if candidatePath == "" {
			continue
		}
		clone := *parsed
		clone.Path = candidatePath
		clone.RawPath = ""
		clone.RawQuery = ""
		clone.Fragment = ""
		value := clone.String()
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		urls = append(urls, value)
	}
	return urls
}

func lmStudioModelMatches(model map[string]any, target string) bool {
	target = strings.TrimSpace(target)
	if target == "" {
		return false
	}
	if key, ok := model["key"].(string); ok && strings.TrimSpace(key) == target {
		return true
	}
	if id, ok := model["id"].(string); ok && strings.TrimSpace(id) == target {
		return true
	}
	loaded, ok := model["loaded_instances"].([]any)
	if !ok {
		return false
	}
	for _, raw := range loaded {
		instance, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if id, ok := instance["id"].(string); ok && strings.TrimSpace(id) == target {
			return true
		}
	}
	return false
}

func extractModelsArray(payload map[string]any) ([]any, string, bool) {
	if models, ok := payload["data"].([]any); ok {
		return models, "data", true
	}
	if models, ok := payload["models"].([]any); ok {
		return models, "models", true
	}
	return nil, "", false
}

func extractLMStudioContextLength(model map[string]any, targetModel string) (int, bool) {
	loaded, ok := model["loaded_instances"].([]any)
	if ok {
		for _, raw := range loaded {
			instance, ok := raw.(map[string]any)
			if !ok {
				continue
			}
			if id, ok := instance["id"].(string); ok && strings.TrimSpace(id) == targetModel {
				if length, ok := extractContextLengthFromLoadedInstance(instance); ok {
					return length, true
				}
			}
		}
		for _, raw := range loaded {
			instance, ok := raw.(map[string]any)
			if !ok {
				continue
			}
			if length, ok := extractContextLengthFromLoadedInstance(instance); ok {
				return length, true
			}
		}
		if len(loaded) == 0 {
			return 0, false
		}
	}

	if length, ok := asPositiveInt(model["max_context_length"]); ok {
		return length, true
	}
	if length, ok := asPositiveInt(model["context_length"]); ok {
		return length, true
	}
	return 0, false
}

func (c *Client) warmupModel(ctx context.Context) error {
	req := openai.ChatCompletionRequest{
		Model: c.model,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: "ping",
			},
		},
		MaxCompletionTokens: 1,
	}
	_, err := c.doRequestOnce(ctx, req)
	return err
}

func extractContextLengthFromLoadedInstance(instance map[string]any) (int, bool) {
	config, ok := instance["config"].(map[string]any)
	if ok {
		if length, ok := asPositiveInt(config["context_length"]); ok {
			return length, true
		}
		if length, ok := asPositiveInt(config["max_context_length"]); ok {
			return length, true
		}
	}
	if length, ok := asPositiveInt(instance["context_length"]); ok {
		return length, true
	}
	if length, ok := asPositiveInt(instance["max_context_length"]); ok {
		return length, true
	}
	return 0, false
}

func asPositiveInt(value any) (int, bool) {
	switch v := value.(type) {
	case int:
		if v > 0 {
			return v, true
		}
	case int64:
		if v > 0 {
			return int(v), true
		}
	case float64:
		if v > 0 {
			return int(v), true
		}
	case string:
		parsed, err := strconv.Atoi(strings.TrimSpace(v))
		if err == nil && parsed > 0 {
			return parsed, true
		}
	case json.Number:
		parsed, err := strconv.Atoi(v.String())
		if err == nil && parsed > 0 {
			return parsed, true
		}
	}
	return 0, false
}

func extractContextLengthFromError(message string) (int, bool) {
	patterns := []*regexp.Regexp{
		reNContext,
		reMaximumContextLen,
		reMaxContextLen,
		reGenericContextLimit,
	}
	for _, pattern := range patterns {
		matches := pattern.FindStringSubmatch(message)
		if len(matches) < 2 {
			continue
		}
		value, err := strconv.Atoi(matches[1])
		if err != nil || value <= 0 {
			continue
		}
		return value, true
	}
	return 0, false
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

	var lastErr error
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
		lastErr = err

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

	if lastErr != nil {
		return nil, EmbeddingMetrics{}, fmt.Errorf("embedding request failed after %d attempts: %w", e.retryCount+1, lastErr)
	}
	return nil, EmbeddingMetrics{}, fmt.Errorf("embedding request failed after %d attempts", e.retryCount+1)
}

// Model возвращает имя embeddings модели.
func (e *Embedder) Model() string {
	return e.model
}
