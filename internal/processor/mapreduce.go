package processor

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/borro/ragcli/internal/config"
	"github.com/borro/ragcli/internal/llm"
)

const separator = "\n\n---\n\n"

type llmCallContext struct {
	Stage      string
	ChunkIndex int
	ChunkCount int
	ChunkBytes int
	BatchIndex int
	BatchCount int
	InputItems int
}

type pipelineStats struct {
	LLMCalls         int
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
	MapSuccesses     int
	MapSkips         int
	MapErrors        int
}

//
// PROMPTS
//

const mapPrompt = `
Извлеки факты из текста, которые помогают ответить на вопрос.

Правила:
- каждый факт отдельной строкой
- максимум 5 фактов
- кратко
- только релевантная информация

Если информации нет — ответь SKIP
`

const reducePrompt = `
Объедини факты.

Правила:
- удаляй дубликаты
- объединяй похожие факты
- максимум 10 фактов
- каждый факт отдельной строкой
- не добавляй новую информацию
`

const finalPrompt = `
Сформулируй ответ на вопрос используя только факты.
Если информации недостаточно — скажи об этом.
`

const critiquePrompt = `
Ты проверяешь ответ.

Проверь:
- опирается ли ответ на факты
- есть ли неподтвержденные утверждения
- есть ли противоречия

Формат ответа:

SUPPORTED: yes/no
ISSUES:
- ...
`

const refinePrompt = `
Исправь ответ на основе замечаний.

Правила:
- используй только факты
- исправь ошибки
- не добавляй новую информацию
`

//
// LLM helper
//

func callLLM(ctx context.Context, client *llm.Client, messages []llm.Message, callCtx llmCallContext, stats *pipelineStats) (string, error) {
	req := llm.ChatCompletionRequest{
		Messages: messages,
	}

	result, err := client.SendRequestWithMetrics(ctx, req)
	if err != nil {
		return "", err
	}

	if stats != nil {
		stats.LLMCalls++
		stats.PromptTokens += result.Metrics.PromptTokens
		stats.CompletionTokens += result.Metrics.CompletionTokens
		stats.TotalTokens += result.Metrics.TotalTokens
	}

	logArgs := []any{"stage", callCtx.Stage}
	switch callCtx.Stage {
	case "map_chunk":
		logArgs = append(logArgs,
			"chunk_index", callCtx.ChunkIndex,
			"chunk_count", callCtx.ChunkCount,
			"chunk_bytes", callCtx.ChunkBytes,
		)
	case "reduce_batch":
		logArgs = append(logArgs,
			"batch_index", callCtx.BatchIndex,
			"batch_count", callCtx.BatchCount,
			"input_items", callCtx.InputItems,
		)
	}
	slog.Debug("map-reduce llm call completed", logArgs...)

	resp := result.Response
	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("empty response")
	}

	return strings.TrimSpace(resp.Choices[0].Message.Content), nil
}

//
// message builders
//

func mapMessages(question, chunk string) []llm.Message {
	return []llm.Message{
		{Role: "system", Content: mapPrompt},
		{
			Role:    "user",
			Content: fmt.Sprintf("Вопрос:\n%s\n\nТекст:\n```\n%s\n```", question, chunk),
		},
	}
}

func reduceMessages(question, facts string) []llm.Message {
	return []llm.Message{
		{Role: "system", Content: reducePrompt},
		{
			Role:    "user",
			Content: fmt.Sprintf("Вопрос:\n%s\n\nФакты:\n```\n%s\n```", question, facts),
		},
	}
}

func finalMessages(question, facts string) []llm.Message {
	return []llm.Message{
		{Role: "system", Content: finalPrompt},
		{
			Role:    "user",
			Content: fmt.Sprintf("Вопрос:\n%s\n\nФакты:\n```\n%s\n```", question, facts),
		},
	}
}

func critiqueMessages(question, facts, answer string) []llm.Message {
	return []llm.Message{
		{Role: "system", Content: critiquePrompt},
		{
			Role: "user",
			Content: fmt.Sprintf(
				"Вопрос:\n%s\n\nФакты:\n```\n%s\n```\n\nОтвет:\n%s",
				question,
				facts,
				answer,
			),
		},
	}
}

func refineMessages(question, facts, answer, critique string) []llm.Message {
	return []llm.Message{
		{Role: "system", Content: refinePrompt},
		{
			Role: "user",
			Content: fmt.Sprintf(
				"Вопрос:\n%s\n\nФакты:\n```\n%s\n```\n\nОтвет:\n%s\n\nЗамечания:\n%s",
				question,
				facts,
				answer,
				critique,
			),
		},
	}
}

//
// chunking
//

func SplitByLines(r io.Reader, maxLen int) ([]string, error) {

	scanner := bufio.NewScanner(r)

	var chunks []string
	var buf strings.Builder

	for scanner.Scan() {

		line := scanner.Text()

		nextLen := buf.Len() + len(line)
		if buf.Len() > 0 {
			nextLen++
		}

		if nextLen > maxLen {
			if buf.Len() > 0 {
				chunks = append(chunks, buf.String())
				buf.Reset()
			} else {
				// Если строка длиннее maxLen, сохраняем ее как отдельный чанк.
				chunks = append(chunks, line)
				continue
			}
		}

		if buf.Len() > 0 {
			buf.WriteByte('\n')
		}

		buf.WriteString(line)
	}

	if buf.Len() > 0 {
		chunks = append(chunks, buf.String())
	}

	return chunks, scanner.Err()
}

//
// MAP
//

func runMapParallel(ctx context.Context, client *llm.Client, chunks []string, question string, workers int, stats *pipelineStats) ([]string, error) {
	slog.Debug("map phase started", "chunk_count", len(chunks), "workers", workers)

	type mapJob struct {
		index int
		chunk string
	}
	type mapResult struct {
		output  string
		skipped bool
		err     error
	}

	jobs := make(chan mapJob)
	results := make(chan string)
	resultMeta := make(chan mapResult, len(chunks))

	var wg sync.WaitGroup

	for i := 0; i < workers; i++ {

		wg.Add(1)

		go func() {

			defer wg.Done()

			for job := range jobs {

				res, err := callLLM(ctx, client, mapMessages(question, job.chunk), llmCallContext{
					Stage:      "map_chunk",
					ChunkIndex: job.index + 1,
					ChunkCount: len(chunks),
					ChunkBytes: len(job.chunk),
				}, stats)
				if err != nil {
					slog.Debug("map chunk failed", "chunk_index", job.index+1, "chunk_count", len(chunks), "error", err)
					resultMeta <- mapResult{err: err}
					continue
				}

				if res != "" && res != "SKIP" {
					results <- res
					resultMeta <- mapResult{output: res}
				} else {
					resultMeta <- mapResult{skipped: true}
				}
			}

		}()
	}

	go func() {

		for i, c := range chunks {
			jobs <- mapJob{index: i, chunk: c}
		}

		close(jobs)

		wg.Wait()

		close(results)

	}()

	var collected []string

	for r := range results {
		collected = append(collected, r)
	}

	for i := 0; i < len(chunks); i++ {
		result := <-resultMeta
		switch {
		case result.err != nil:
			stats.MapErrors++
		case result.skipped:
			stats.MapSkips++
		default:
			stats.MapSuccesses++
		}
	}

	slog.Debug("map phase finished",
		"chunk_count", len(chunks),
		"workers", workers,
		"results_count", len(collected),
		"skip_count", stats.MapSkips,
		"error_count", stats.MapErrors,
	)

	return collected, nil
}

//
// batching
//

func batchStrings(items []string, size int) [][]string {

	var batches [][]string

	for size < len(items) {

		items, batches = items[size:], append(batches, items[:size])
	}

	batches = append(batches, items)

	return batches
}

//
// REDUCE
//

func fanInReduce(ctx context.Context, client *llm.Client, results []string, question string, cfg *config.Config, stats *pipelineStats) (string, error) {
	reduceBatchSize := cfg.Concurrency
	if reduceBatchSize < 2 {
		reduceBatchSize = 2
	}

	iteration := 0
	for len(results) > 1 {
		iteration++

		batches := batchStrings(results, reduceBatchSize)
		slog.Debug("reduce iteration started",
			"iteration", iteration,
			"input_items", len(results),
			"batch_size", reduceBatchSize,
			"batch_count", len(batches),
		)

		var next []string

		for batchIndex, batch := range batches {

			input := strings.Join(batch, separator)

			summary, err := callLLM(ctx, client, reduceMessages(question, input), llmCallContext{
				Stage:      "reduce_batch",
				BatchIndex: batchIndex + 1,
				BatchCount: len(batches),
				InputItems: len(batch),
			}, stats)
			if err != nil {
				return "", err
			}

			next = append(next, summary)
		}

		results = next
		slog.Debug("reduce iteration finished",
			"iteration", iteration,
			"output_items", len(results),
		)
	}

	return results[0], nil
}

//
// SELF REFINEMENT
//

func selfRefine(ctx context.Context, client *llm.Client, question, facts, answer string, stats *pipelineStats) (string, error) {
	slog.Debug("self-refine critique started")

	critique, err := callLLM(ctx, client, critiqueMessages(question, facts, answer), llmCallContext{Stage: "self_refine_critique"}, stats)
	if err != nil {
		return "", err
	}

	if strings.Contains(critique, "NONE") {
		slog.Debug("self-refine skipped", "reason", "critique_none")
		return answer, nil
	}

	slog.Debug("self-refine finalize started")
	return callLLM(ctx, client, refineMessages(question, facts, answer, critique), llmCallContext{Stage: "self_refine_finalize"}, stats)
}

//
// PIPELINE
//

func RunSRMR(ctx context.Context, client *llm.Client, input io.Reader, cfg *config.Config, question string) (string, error) {
	startedAt := time.Now()
	inputMode := "stdin"
	if cfg.InputPath != "" {
		inputMode = "file"
	}
	stats := &pipelineStats{}

	slog.Debug("map-reduce pipeline started",
		"input_mode", inputMode,
		"chunk_length", cfg.ChunkLength,
		"concurrency", cfg.Concurrency,
	)

	chunks, err := SplitByLines(input, cfg.ChunkLength)
	if err != nil {
		return "", err
	}
	slog.Debug("split into chunks", "chunk_count", len(chunks))

	if len(chunks) == 0 {
		slog.Debug("map-reduce pipeline finished", "reason", "empty_input", "duration_ms", time.Since(startedAt).Milliseconds())
		return "Файл пустой", nil
	}

	mapResults, err := runMapParallel(ctx, client, chunks, question, cfg.Concurrency, stats)
	if err != nil {
		return "", err
	}

	if len(mapResults) == 0 {
		slog.Debug("map-reduce pipeline finished",
			"reason", "no_map_results",
			"duration_ms", time.Since(startedAt).Milliseconds(),
			"llm_calls", stats.LLMCalls,
			"prompt_tokens", stats.PromptTokens,
			"completion_tokens", stats.CompletionTokens,
			"total_tokens", stats.TotalTokens,
			"map_successes", stats.MapSuccesses,
			"map_skips", stats.MapSkips,
			"map_errors", stats.MapErrors,
		)
		return "Нет информации для ответа", nil
	}

	slog.Debug("reduce phase started", "input_items", len(mapResults))
	facts, err := fanInReduce(ctx, client, mapResults, question, cfg, stats)
	if err != nil {
		return "", err
	}

	slog.Debug("final answer generation started")
	answer, err := callLLM(ctx, client, finalMessages(question, facts), llmCallContext{Stage: "final_answer"}, stats)
	if err != nil {
		return "", err
	}

	result, err := selfRefine(ctx, client, question, facts, answer, stats)
	if err != nil {
		return "", err
	}

	slog.Debug("map-reduce pipeline finished",
		"reason", "success",
		"duration_ms", time.Since(startedAt).Milliseconds(),
		"llm_calls", stats.LLMCalls,
		"prompt_tokens", stats.PromptTokens,
		"completion_tokens", stats.CompletionTokens,
		"total_tokens", stats.TotalTokens,
		"map_successes", stats.MapSuccesses,
		"map_skips", stats.MapSkips,
		"map_errors", stats.MapErrors,
	)
	return result, nil
}
