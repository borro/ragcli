package processor

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"strings"
	"sync"

	"github.com/borro/ragcli/internal/config"
	"github.com/borro/ragcli/internal/llm"
)

const separator = "\n\n---\n\n"

//
// PROMPTS
//

const mapSystemPrompt = `
Ты извлекаешь факты из текста для ответа на вопрос.

Правила:
- извлекай только факты, относящиеся к вопросу
- каждый факт — отдельной строкой
- максимум 5 фактов
- кратко
- игнорируй нерелевантный текст

Формат:
- факт
- факт

Если нет информации — SKIP
`

const reduceCompressPrompt = `
Ты объединяешь факты из разных частей документа.

Правила:
- удаляй повторяющиеся факты
- объединяй похожие факты
- максимум 10 фактов
- каждый факт отдельной строкой
- не добавляй новую информацию
`

const finalReducePrompt = `
Сформулируй финальный ответ на вопрос, используя только факты.

Правила:
- не добавляй новую информацию
- объединяй факты в связный ответ
- если информации недостаточно — скажи об этом
- если есть противоречия — укажи
`

//
// LLM helper
//

func callLLM(ctx context.Context, client *llm.Client, messages []llm.Message) (string, error) {
	req := llm.ChatCompletionRequest{
		Model:    client.Model(),
		Messages: messages,
	}

	resp, err := client.SendRequest(ctx, req)
	if err != nil {
		return "", err
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("empty response")
	}

	return strings.TrimSpace(resp.Choices[0].Message.Content), nil
}

//
// message builders
//

func mapMessages(prompt, chunk string) []llm.Message {
	return []llm.Message{
		{Role: "system", Content: mapSystemPrompt},
		{
			Role: "user",
			Content: fmt.Sprintf(
				"Вопрос:\n%s\n\nТекст:\n```\n%s\n```",
				prompt,
				chunk,
			),
		},
	}
}

func reduceCompressMessages(prompt, facts string) []llm.Message {
	return []llm.Message{
		{Role: "system", Content: reduceCompressPrompt},
		{
			Role: "user",
			Content: fmt.Sprintf(
				"Вопрос:\n%s\n\nФакты:\n```\n%s\n```",
				prompt,
				facts,
			),
		},
	}
}

func finalReduceMessages(prompt, facts string) []llm.Message {
	return []llm.Message{
		{Role: "system", Content: finalReducePrompt},
		{
			Role: "user",
			Content: fmt.Sprintf(
				"Вопрос:\n%s\n\nФакты:\n```\n%s\n```",
				prompt,
				facts,
			),
		},
	}
}

//
// Chunking
//

func SplitByLines(r io.Reader, maxLen int) ([]string, error) {
	scanner := bufio.NewScanner(r)

	var (
		chunks []string
		buf    strings.Builder
	)

	for scanner.Scan() {
		line := scanner.Text()
		if buf.Len()+len(line)+1 > maxLen {
			chunks = append(chunks, buf.String())
			buf.Reset()
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

func runMapParallel(ctx context.Context, client *llm.Client, chunks []string, prompt string, workers int) ([]string, error) {
	jobs := make(chan string)
	results := make(chan string)

	var wg sync.WaitGroup

	for i := 0; i < workers; i++ {
		wg.Add(1)

		go func() {
			defer wg.Done()

			for chunk := range jobs {
				res, err := callLLM(ctx, client, mapMessages(prompt, chunk))
				if err != nil {
					continue
				}

				if res != "" && res != "SKIP" {
					results <- res
				}
			}

		}()
	}

	go func() {
		for _, c := range chunks {
			jobs <- c
		}

		close(jobs)

		wg.Wait()

		close(results)

	}()

	var collected []string

	for r := range results {
		collected = append(collected, r)
	}

	return collected, nil
}

//
// BATCH HELPER
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
// REDUCE (fan-in)
//

func fanInReduce(ctx context.Context, client *llm.Client, results []string, prompt string, cfg *config.Config) (string, error) {
	for len(results) > 1 {
		batches := batchStrings(results, cfg.ChunkLength)

		var next []string
		for _, batch := range batches {
			input := strings.Join(batch, separator)

			summary, err := callLLM(ctx, client, reduceCompressMessages(prompt, input))
			if err != nil {
				return "", err
			}

			next = append(next, summary)
		}

		results = next
	}

	return callLLM(ctx, client, finalReduceMessages(prompt, results[0]))
}

//
// PIPELINE
//

func RunMapReduce(ctx context.Context, client *llm.Client, input io.Reader, cfg *config.Config, prompt string) (string, error) {

	chunks, err := SplitByLines(input, cfg.ChunkLength)
	if err != nil {
		return "", err
	}

	if len(chunks) == 0 {
		return "Файл пустой", nil
	}

	mapResults, err := runMapParallel(ctx, client, chunks, prompt, cfg.Concurrency)
	if err != nil {
		return "", err
	}

	if len(mapResults) == 0 {
		return "Нет информации для ответа", nil
	}

	return fanInReduce(ctx, client, mapResults, prompt, cfg)
}
