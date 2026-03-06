package processor

import (
	"context"
	"fmt"

	"github.com/borro/ragcli/internal/config"
	"github.com/borro/ragcli/internal/llm"
)

// RunRAG выполняет обработку в режиме RAG (Agentic Tool Calling)
// prompt - вопрос пользователя для которого нужно найти ответ в файле.
func RunRAG(ctx context.Context, client *llm.Client, filePath string, cfg *config.Config, prompt string) (string, error) {
	config.Log.Info("starting RAG processing", "file_path", filePath, "prompt", prompt)

	// Создаём список инструментов
	tools := []llm.Tool{
		{
			Type: "function",
			Function: &llm.FunctionDefinition{
				Name:        "search_file",
				Description: "Искать подстроку в файле и вернуть номера строк с совпадениями. Аргументы: query (string) - строка для поиска.",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"query": map[string]any{
							"type":        "string",
							"description": "Строка для поиска в файле",
						},
					},
					"required": []string{"query"},
				},
			},
		},
		{
			Type: "function",
			Function: &llm.FunctionDefinition{
				Name:        "read_lines",
				Description: "Прочитать диапазон строк из файла. Аргументы: start_line (int), end_line (int) - номера строк.",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"start_line": map[string]any{
							"type":        "integer",
							"description": "Начальная строка (начиная с 1)",
						},
						"end_line": map[string]any{
							"type":        "integer",
							"description": "Конечная строка",
						},
					},
					"required": []string{"start_line", "end_line"},
				},
			},
		},
	}

	// Системный промпт для агента
	messages := []llm.Message{
		{Role: "system", Content: fmt.Sprintf("Ты — ИИ-агент. Пользователь задал вопрос по большому файлу. Его вопрос: \"%s\". Используй доступные инструменты (search_file и read_lines), чтобы исследовать файл и найти ответ. Не пытайся угадать, делай запросы к файлу", prompt)},
	}

	req := llm.ChatCompletionRequest{
		Model:      client.Model(),
		Messages:   messages,
		Tools:      tools,
		ToolChoice: "auto",
	}

	maxTurns := 10 // Максимум циклов общения с моделью
	for turn := 1; turn <= maxTurns; turn++ {
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		default:
		}

		config.Log.Debug("sending request to LLM", "turn", turn)

		resp, err := client.SendRequest(ctx, req)
		if err != nil {
			return "", fmt.Errorf("failed to send request: %w", err)
		}

		if len(resp.Choices) == 0 {
			return "", fmt.Errorf("no choices in response")
		}

		choice := resp.Choices[0]
		messages = append(messages, choice.Message)

		config.Log.Debug("got response from LLM", "role", choice.Message.Role)

		// Проверяем, есть ли tool calls
		if len(choice.Message.ToolCalls) > 0 {
			config.Log.Info("received tool calls from model", "count", len(choice.Message.ToolCalls))

			// Выполняем tool calls
			results, err := ExecuteToolCalls(ctx, client, choice.Message.ToolCalls, filePath)
			if err != nil {
				return "", fmt.Errorf("failed to execute tool calls: %w", err)
			}

			// Добавляем результаты в историю диалога
			for _, result := range results {
				messages = append(messages, llm.Message{
					Role:       "tool",
					Name:       choice.Message.ToolCalls[0].Function.Name,
					Content:    result,
					ToolCallID: choice.Message.ToolCalls[0].ID,
				})
			}

			continue
		}

		// Если нет tool calls, значит модель сгенерировала финальный ответ
		config.Log.Info("received final answer from model")
		return choice.Message.Content, nil
	}

	return "", fmt.Errorf("max turns reached without final answer")
}
