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

func callLLM(ctx context.Context, client *llm.Client, messages []llm.Message) (string, error) {

	req := llm.ChatCompletionRequest{
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

func runMapParallel(ctx context.Context, client *llm.Client, chunks []string, question string, workers int) ([]string, error) {

	jobs := make(chan string)
	results := make(chan string)

	var wg sync.WaitGroup

	for i := 0; i < workers; i++ {

		wg.Add(1)

		go func() {

			defer wg.Done()

			for chunk := range jobs {

				res, err := callLLM(ctx, client, mapMessages(question, chunk))
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

func fanInReduce(ctx context.Context, client *llm.Client, results []string, question string, cfg *config.Config) (string, error) {

	reduceBatchSize := cfg.Concurrency
	if reduceBatchSize < 2 {
		reduceBatchSize = 2
	}

	for len(results) > 1 {

		batches := batchStrings(results, reduceBatchSize)

		var next []string

		for _, batch := range batches {

			input := strings.Join(batch, separator)

			summary, err := callLLM(ctx, client, reduceMessages(question, input))
			if err != nil {
				return "", err
			}

			next = append(next, summary)
		}

		results = next
	}

	return results[0], nil
}

//
// SELF REFINEMENT
//

func selfRefine(ctx context.Context, client *llm.Client, question, facts, answer string) (string, error) {

	critique, err := callLLM(ctx, client, critiqueMessages(question, facts, answer))
	if err != nil {
		return "", err
	}

	if strings.Contains(critique, "NONE") {
		return answer, nil
	}

	return callLLM(ctx, client, refineMessages(question, facts, answer, critique))
}

//
// PIPELINE
//

func RunSRMR(ctx context.Context, client *llm.Client, input io.Reader, cfg *config.Config, question string) (string, error) {

	chunks, err := SplitByLines(input, cfg.ChunkLength)
	if err != nil {
		return "", err
	}

	if len(chunks) == 0 {
		return "Файл пустой", nil
	}

	mapResults, err := runMapParallel(ctx, client, chunks, question, cfg.Concurrency)
	if err != nil {
		return "", err
	}

	if len(mapResults) == 0 {
		return "Нет информации для ответа", nil
	}

	facts, err := fanInReduce(ctx, client, mapResults, question, cfg)
	if err != nil {
		return "", err
	}

	answer, err := callLLM(ctx, client, finalMessages(question, facts))
	if err != nil {
		return "", err
	}

	return selfRefine(ctx, client, question, facts, answer)
}
