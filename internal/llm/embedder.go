package llm

import (
	"context"
	"log/slog"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

// CreateEmbeddingsWithMetrics отправляет embeddings запрос и возвращает метрики.
func (e *Embedder) CreateEmbeddingsWithMetrics(ctx context.Context, inputs []string) ([][]float32, EmbeddingMetrics, error) {
	req := openai.EmbeddingRequestStrings{
		Input: []string{},
		Model: openai.EmbeddingModel(e.model),
	}
	if len(inputs) > 0 {
		req.Input = inputs
	}

	var metrics EmbeddingMetrics
	resp, err := runWithRetry(
		ctx,
		e.retryCount+1,
		e.sleep,
		func(callCtx context.Context) (openai.EmbeddingResponse, error) {
			return e.client.CreateEmbeddings(callCtx, req)
		},
		retryHooks[openai.EmbeddingResponse]{
			OnStart: func(attempt int) {
				slog.Debug("embedding request started",
					"model", e.model,
					"attempt", attempt,
					"input_count", len(req.Input),
				)
			},
			OnSuccess: func(attempt int, resp openai.EmbeddingResponse, duration time.Duration) {
				metrics = buildEmbeddingMetrics(e.model, attempt, len(req.Input), len(resp.Data), resp, duration)
				slog.Debug("embedding request finished",
					"model", metrics.Model,
					"attempt", metrics.Attempt,
					"input_count", metrics.InputCount,
					"vector_count", metrics.VectorCount,
					"duration", durationSeconds(metrics.Duration),
					"prompt_tokens", metrics.PromptTokens,
					"total_tokens", metrics.TotalTokens,
				)
			},
			OnFailure: func(attempt int, err error, duration time.Duration) {
				slog.Debug("embedding request failed",
					"model", e.model,
					"attempt", attempt,
					"input_count", len(req.Input),
					"duration", durationSeconds(duration),
					"error", err,
				)
			},
			OnRetry: func(attempt int, err error, delay time.Duration) {
				slog.Debug("retrying embedding request",
					"model", e.model,
					"failed_attempt", attempt,
					"next_attempt", attempt+1,
					"delay_seconds", int(delay/time.Second),
					"reason", err,
				)
			},
		},
	)
	if err != nil {
		return nil, EmbeddingMetrics{}, err
	}

	return embeddingsFromResponse(resp), metrics, nil
}

func buildEmbeddingMetrics(model string, attempt int, inputCount int, vectorCount int, resp openai.EmbeddingResponse, duration time.Duration) EmbeddingMetrics {
	return EmbeddingMetrics{
		Model:        model,
		Attempt:      attempt,
		InputCount:   inputCount,
		VectorCount:  vectorCount,
		Duration:     duration,
		PromptTokens: resp.Usage.PromptTokens,
		TotalTokens:  resp.Usage.TotalTokens,
	}
}

func embeddingsFromResponse(resp openai.EmbeddingResponse) [][]float32 {
	vectors := make([][]float32, 0, len(resp.Data))
	for _, item := range resp.Data {
		vectors = append(vectors, item.Embedding)
	}
	return vectors
}
