package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"

	openai "github.com/sashabaranov/go-openai"
)

const autoContextProbePromptTokens = 320000

var (
	reNContext            = regexp.MustCompile(`(?i)n_ctx\s*\((\d+)\)`)
	reMaximumContextLen   = regexp.MustCompile(`(?i)maximum context length[^0-9]*(\d+)`)
	reMaxContextLen       = regexp.MustCompile(`(?i)max(?:imum)? context length[^0-9]*(\d+)`)
	reGenericContextLimit = regexp.MustCompile(`(?i)context length[^0-9]*(\d+)`)
	errModelNotLoaded     = errors.New("matched model has no loaded instances")
)

type lmStudioModelsPayload struct {
	Models []lmStudioModel `json:"models"`
}

type lmStudioModel struct {
	Key             string                   `json:"key"`
	LoadedInstances []lmStudioLoadedInstance `json:"loaded_instances"`
}

type lmStudioLoadedInstance struct {
	Config lmStudioLoadedInstanceConfig `json:"config"`
}

type lmStudioLoadedInstanceConfig struct {
	ContextLength int `json:"context_length"`
}

// ResolveAutoContextLength attempts to detect model context length.
func (c *Client) ResolveAutoContextLength(ctx context.Context) (int, error) {
	slog.Debug("auto context length resolve started",
		"model", c.model,
		"base_url", c.baseURL,
	)
	value, err := c.resolveAutoContextLengthNoCache(ctx)
	slog.Debug("auto context length resolve finished",
		"model", c.model,
		"base_url", c.baseURL,
		"value", value,
		"error", err,
	)
	return value, err
}

func (c *Client) resolveAutoContextLengthNoCache(ctx context.Context) (int, error) {
	var lastErr error
	for _, modelsURL := range lmStudioModelsURLs(c.baseURL) {
		slog.Debug("auto context length trying lmstudio models endpoint",
			"model", c.model,
			"url", modelsURL,
		)

		length, err := c.resolveContextLengthFromModelsURL(ctx, modelsURL)
		if err == nil {
			return length, nil
		}

		slog.Debug("auto context length lmstudio models endpoint failed",
			"model", c.model,
			"url", modelsURL,
			"error", err,
		)
		lastErr = err
	}

	slog.Debug("auto context length falling back to probe from error",
		"model", c.model,
		"last_error", lastErr,
	)
	length, err := c.resolveContextLengthFromProbe(ctx)
	if err == nil {
		return length, nil
	}

	slog.Debug("auto context length probe failed",
		"model", c.model,
		"error", err,
	)
	lastErr = err

	if lastErr == nil {
		lastErr = fmt.Errorf("context length not detected")
	}
	return 0, fmt.Errorf("failed to resolve model context length: %w", lastErr)
}

func (c *Client) resolveContextLengthFromModelsURL(ctx context.Context, modelsURL string) (int, error) {
	length, err := c.resolveContextLengthFromLMStudioModels(ctx, modelsURL)
	if err == nil {
		return length, nil
	}
	if !errors.Is(err, errModelNotLoaded) {
		return 0, err
	}

	slog.Debug("auto context length model matched but not loaded, sending warmup request",
		"model", c.model,
		"url", modelsURL,
	)
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
	if _, warmupErr := c.doRequestOnce(ctx, req); warmupErr != nil {
		slog.Debug("auto context length warmup request failed",
			"model", c.model,
			"url", modelsURL,
			"error", warmupErr,
		)
		return 0, warmupErr
	}

	retryLength, retryErr := c.resolveContextLengthFromLMStudioModels(ctx, modelsURL)
	if retryErr == nil {
		return retryLength, nil
	}

	slog.Debug("auto context length warmup retry failed",
		"model", c.model,
		"url", modelsURL,
		"error", retryErr,
	)
	return 0, retryErr
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

	resp, err := c.httpClient.Do(req)
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

	var payload lmStudioModelsPayload
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return 0, err
	}

	if len(payload.Models) == 0 {
		return 0, fmt.Errorf("models payload does not contain models list")
	}

	targetModel := strings.TrimSpace(c.model)
	for _, model := range payload.Models {
		if strings.TrimSpace(model.Key) != targetModel {
			continue
		}
		if len(model.LoadedInstances) == 0 {
			return 0, errModelNotLoaded
		}
		if length, ok := extractLMStudioContextLength(model); ok {
			return length, nil
		}
		return 0, fmt.Errorf("context length not found in models response")
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

func extractLMStudioContextLength(model lmStudioModel) (int, bool) {
	for _, instance := range model.LoadedInstances {
		if instance.Config.ContextLength > 0 {
			return instance.Config.ContextLength, true
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
