package llm

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

// SendRequest отправляет запрос к LLM API с механизмом retry и exponential backoff.
func (c *Client) SendRequest(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
	resp, _, err := c.SendRequestWithMetrics(ctx, req)
	if err != nil {
		return nil, err
	}

	return resp, nil
}

// SendRequestWithMetrics отправляет запрос к LLM API и возвращает ответ вместе с метриками.
func (c *Client) SendRequestWithMetrics(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, RequestMetrics, error) {
	model := requestModel(c.model, req.Model)
	toolChoice := toolChoiceLabel(req.ToolChoice)

	var metrics RequestMetrics
	resp, err := runWithRetry(
		ctx,
		c.retryCount+1,
		c.sleep,
		func(callCtx context.Context) (openai.ChatCompletionResponse, error) {
			return c.doRequestOnce(callCtx, req)
		},
		retryHooks[openai.ChatCompletionResponse]{
			OnStart: func(attempt int) {
				slog.Debug("llm request started",
					"model", model,
					"attempt", attempt,
					"message_count", len(req.Messages),
					"tool_count", len(req.Tools),
					"tool_choice", toolChoice,
				)
			},
			OnSuccess: func(attempt int, resp openai.ChatCompletionResponse, duration time.Duration) {
				metrics = buildRequestMetrics(c.model, attempt, req, resp, duration)
				slog.Debug("llm request finished",
					"model", metrics.Model,
					"attempt", metrics.Attempt,
					"message_count", metrics.MessageCount,
					"tool_count", metrics.ToolCount,
					"tool_choice", metrics.ToolChoice,
					"duration", durationSeconds(metrics.Duration),
					"choice_count", metrics.ChoiceCount,
					"has_tool_calls", metrics.HasToolCalls,
					"prompt_tokens", metrics.PromptTokens,
					"completion_tokens", metrics.CompletionTokens,
					"total_tokens", metrics.TotalTokens,
					"tokens_per_second", metrics.TokensPerSecond,
				)
			},
			OnFailure: func(attempt int, err error, duration time.Duration) {
				slog.Debug("llm request failed",
					"model", model,
					"attempt", attempt,
					"message_count", len(req.Messages),
					"tool_count", len(req.Tools),
					"tool_choice", toolChoice,
					"duration", durationSeconds(duration),
					"error", err,
				)
			},
			OnRetry: func(attempt int, err error, delay time.Duration) {
				slog.Debug("retrying llm request",
					"model", model,
					"failed_attempt", attempt,
					"next_attempt", attempt+1,
					"delay_seconds", int(delay/time.Second),
					"reason", err,
				)
			},
		},
	)
	if err != nil {
		return nil, RequestMetrics{}, err
	}

	return &resp, metrics, nil
}

// doRequestOnce выполняет один HTTP запрос к API через go-openai.
func (c *Client) doRequestOnce(ctx context.Context, req openai.ChatCompletionRequest) (openai.ChatCompletionResponse, error) {
	if req.Model == "" {
		req.Model = c.model
	}
	return c.client.CreateChatCompletion(ctx, req)
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

func toolChoiceLabel(toolChoice any) string {
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

func durationSeconds(duration time.Duration) float64 {
	return float64(duration.Round(time.Millisecond)) / float64(time.Second)
}
