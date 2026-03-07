package processor

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/borro/ragcli/internal/config"
	"github.com/borro/ragcli/internal/llm"
	"golang.org/x/sync/errgroup"
)

// RunMapExecute выполняет фазу map: обрабатывает один чанк через LLM
func RunMapExecute(ctx context.Context, client *llm.Client, chunk string, prompt string) (string, error) {
	messages := []llm.Message{
		{Role: "system", Content: "Извлеки факты для ответа на вопрос. Если нет полезного, верни 'SKIP'."},
		{Role: "user", Content: fmt.Sprintf("Вопрос:\n%s\nТекст:\n```\n%s\n```", prompt, chunk)},
	}

	req := llm.ChatCompletionRequest{
		Model:    client.Model(),
		Messages: messages,
	}

	resp, err := client.SendRequest(ctx, req)
	if err != nil {
		return "", fmt.Errorf("failed to send request: %w", err)
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("no choices in response")
	}

	result := strings.TrimSpace(resp.Choices[0].Message.Content)
	return result, nil
}

// SplitByLines разбивает содержимое на чанки по размеру, не разрывая строки
func SplitByLines(reader io.Reader, maxLength int) ([]string, error) {
	var chunks []string
	scanner := bufio.NewScanner(reader)

	for scanner.Scan() {
		line := scanner.Text()

		if len(chunks) == 0 || len(chunks[len(chunks)-1])+len(line)+1 <= maxLength {
			if len(chunks) > 0 && chunks[len(chunks)-1] != "" {
				chunks[len(chunks)-1] += "\n" + line
			} else {
				chunks = append(chunks, line)
			}
		} else {
			chunks = append(chunks, line)
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("failed to scan input: %w", err)
	}

	return chunks, nil
}

// truncateString обрезает строку до максимальной длины
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen]
}

// RunMapReduce выполняет map-reduce обработку.
// prompt - вопрос пользователя для которого нужно найти ответ в файле.
func RunMapReduce(ctx context.Context, client *llm.Client, input io.Reader, cfg *config.Config, prompt string) (string, error) {
	config.Log.Info("starting map-reduce processing",
		"chunk_length", cfg.ChunkLength,
		"concurrency", cfg.Concurrency,
		"prompt", prompt)

	// Читаем и разбиваем на чанки
	chunks, err := SplitByLines(input, cfg.ChunkLength)
	if err != nil {
		return "", fmt.Errorf("failed to split input: %w", err)
	}

	config.Log.Info("split into chunks", "count", len(chunks))

	if len(chunks) == 0 {
		return "Файл пустой или не содержит текста", nil
	}

	// Используем errgroup для организации worker pool
	g, mapCtx := errgroup.WithContext(ctx)
	g.SetLimit(cfg.Concurrency)

	mapResults := make(chan string, len(chunks))

	for _, chunk := range chunks {
		chunkCopy := chunk // копия для горутины
		g.Go(func() error {
			result, err := RunMapExecute(mapCtx, client, chunkCopy, prompt)
			if err != nil {
				return fmt.Errorf("chunk processing failed: %w", err)
			}
			mapResults <- result
			return nil
		})
	}

	// Закрываем канал после завершения всех горутин
	go func() {
		//nolint:errcheck // g.Wait() не может вернуть ошибку при использовании errgroup.WithContext
		g.Wait()
		close(mapResults)
	}()

	// Собираем результаты
	var filteredResults []string
	for r := range mapResults {
		if r != "SKIP" && r != "" {
			filteredResults = append(filteredResults, r)
			config.Log.Debug("got map result", "result", truncateString(r, 100))
		}
	}

	// Проверяем ошибки
	if err := g.Wait(); err != nil {
		return "", fmt.Errorf("map phase failed: %w", err)
	}

	config.Log.Info("map phase completed", "results_count", len(filteredResults))

	// Reduce фаза: собираем все результаты и генерируем финальный ответ с предотвращением переполнения контекста
	// Используем исходный контекст ctx, а не mapCtx (который был отменен после завершения g.Wait())
	return RunMapReduceIterative(ctx, client, filteredResults, prompt, cfg)
}

// RunReduce выполняет reduce фазу
func RunReduce(ctx context.Context, client *llm.Client, results []string, prompt string) (string, error) {
	if len(results) == 0 {
		return "Нет информации в файле для ответа на этот вопрос", nil
	}

	// Объединяем все результаты
	reduceInput := strings.Join(results, "\n\n---\n\n")

	messages := []llm.Message{
		{Role: "system", Content: "Сформулируй полный и понятный ответ на вопрос, используя предоставленные факты. Если некоторые части неполные или противоречивые, укажи на это."},
		{Role: "user", Content: fmt.Sprintf("Вопрос:\n%s\n\nОбобщённая информация из файла:\n```\n%s\n```", prompt, reduceInput)},
	}

	req := llm.ChatCompletionRequest{
		Model:    client.Model(),
		Messages: messages,
	}

	resp, err := client.SendRequest(ctx, req)
	if err != nil {
		return "", fmt.Errorf("failed to send reduce request: %w", err)
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("no choices in reduce response")
	}

	return strings.TrimSpace(resp.Choices[0].Message.Content), nil
}

// RunMapReduceIterative выполняет итеративную обработку map-reduce с предотвращением переполнения контекста.
// Если результат reduce превышает cfg.ChunkLength, функция рекурсивно делит результаты на части,
// запускает дополнительную фазу map для каждой части и агрегирует их снова до тех пор, пока
// результат не поместится в окно длины ChunkLength.
func RunMapReduceIterative(ctx context.Context, client *llm.Client, results []string, prompt string, cfg *config.Config) (string, error) {
	return runMapReduceIterativeWithDepth(ctx, client, results, prompt, cfg, 0)
}

const maxRecursionDepth = 10

// runMapReduceIterativeWithDepth - внутренняя функция с подсчётом глубины рекурсии
func runMapReduceIterativeWithDepth(ctx context.Context, client *llm.Client, results []string, prompt string, cfg *config.Config, depth int) (string, error) {
	config.Log.Info("map-reduce iteration started", "results_count", len(results), "depth", depth)

	// Защита от бесконечной рекурсии
	if depth > maxRecursionDepth {
		return "", fmt.Errorf("max recursion depth (%d) exceeded, result still too large: %d bytes (max allowed: %d)", 
			maxRecursionDepth, len(results), cfg.ChunkLength)
	}

	// Выполняем reduce фазу для агрегации всех результатов
	reducedResult, err := RunReduce(ctx, client, results, prompt)
	if err != nil {
		return "", fmt.Errorf("reduce failed: %w", err)
	}

	config.Log.Debug("reduced result size", "size_bytes", len(reducedResult))

	// Проверяем размер результата в байтах
	if len(reducedResult) <= cfg.ChunkLength {
		config.Log.Info("map-reduce iteration completed successfully", "result_size", len(reducedResult), "max_allowed", cfg.ChunkLength, "depth", depth)
		return reducedResult, nil
	}

	// Защита: если результатов меньше 2, нельзя разделить дальше - возвращаем текущий результат
	if len(results) < 2 {
		config.Log.Warn("cannot split further, returning result despite exceeding chunk length", 
			"result_size", len(reducedResult), "max_allowed", cfg.ChunkLength)
		return reducedResult, nil
	}

	config.Log.Warn("chunk too large, splitting and retrying map phase", "current_size", len(reducedResult), "max_allowed", cfg.ChunkLength, "depth", depth)

	// Делим результаты пополам и рекурсивно обрабатываем каждую часть
	mid := len(results) / 2
	part1Results := results[:mid]
	part2Results := results[mid:]

	config.Log.Info("splitting results for recursive map phase", "part1_count", len(part1Results), "part2_count", len(part2Results))

	result1, err1 := runMapReduceIterativeWithDepth(ctx, client, part1Results, prompt, cfg, depth+1)
	if err1 != nil {
		return "", fmt.Errorf("map-reduce iteration failed for part 1: %w", err1)
	}

	result2, err2 := runMapReduceIterativeWithDepth(ctx, client, part2Results, prompt, cfg, depth+1)
	if err2 != nil {
		return "", fmt.Errorf("map-reduce iteration failed for part 2: %w", err2)
	}

	// Агрегируем результаты рекурсивных вызовов и запускаем reduce снова
	combinedResults := []string{result1, result2}
	config.Log.Info("aggregating recursive results and running reduce again", "combined_count", len(combinedResults))

	return runMapReduceIterativeWithDepth(ctx, client, combinedResults, prompt, cfg, depth+1)
}
