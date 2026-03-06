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

	// Reduce фаза: собираем все результаты и генерируем финальный ответ
	// Используем исходный контекст ctx, а не mapCtx (который был отменен после завершения g.Wait())
	return RunReduce(ctx, client, filteredResults, prompt)
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
