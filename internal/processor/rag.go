package processor

import (
	"context"
	"fmt"
	"strings"

	"github.com/borro/ragcli/internal/config"
	"github.com/borro/ragcli/internal/llm"
)

// toolsConfig содержит инструменты и системный промпт для RAG.
type toolsConfig struct {
	tools         []llm.Tool
	systemMessage llm.Message
	userQuestion  llm.Message
}

// NewToolsConfig создаёт конфигурацию инструментов для RAG.
func NewToolsConfig(prompt string) toolsConfig {
	// Определяем инструменты
	tools := []llm.Tool{
		{
			Type: "function",
			Function: &llm.FunctionDefinition{
				Name:        "search_file",
				Description: "Искать подстроку в файле и вернуть номера строк с совпадениями. Аргументы: query (string) - строка для поиска. ИСПОЛЬЗУЙ ЭТОТ ИНСТРУМЕНТ ТОЛЬКО ОДИН РАЗ!",
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
	systemMessage := llm.Message{
		Role: "system",
		Content: `Ты — ИИ-агент. Твоя цель - ответить на вопрос пользователя, исследуя предоставленный файл. Используй доступные инструменты (search_file и read_lines) для поиска информации в файле. Если ты не можешь найти ответ, скажи об этом. Не пытайся угадать, делай запросы к файлу.

ВАЖНО:
- Используй инструменты РАЗ, если это необходимо для поиска ответа.
- После получения результатов инструмента ПРЯМО СРАЗУ сформулируй ответ на вопрос пользователя.
- НЕ вызывай один и тот же инструмент повторно.
- Когда получил результаты поиска - немедленно ответь на вопрос, используя найденную информацию.
- Если поиск ничего не нашёл - сразу скажи об этом, не пытайся искать ещё раз.
- Всегда давай ответ на основе всех доступной информации.`,
	}

	userQuestion := llm.Message{
		Role:    "user",
		Content: fmt.Sprintf("Вопрос: \"%s\"", prompt),
	}

	return toolsConfig{
		tools:         tools,
		systemMessage: systemMessage,
		userQuestion:  userQuestion,
	}
}

// RunRAG выполняет обработку в режиме RAG (Agentic Tool Calling).
// filePath должен указывать на файл с данными; stdin уже нормализуется в temp file на уровне CLI.
func RunRAG(ctx context.Context, client *llm.Client, filePath, prompt string) (string, error) {
	config.Log.Info("starting RAG processing", "file_path", filePath, "prompt", prompt)

	// Создаём конфигурацию инструментов
	ragConfig := NewToolsConfig(prompt)
	messages := make([]llm.Message, 0, 2)
	messages = append(messages, ragConfig.systemMessage, ragConfig.userQuestion)

	// Настройки для цикла общения
	maxTurns := 6 // Максимум циклов общения с моделью (нужен 1 ход для ответа после tool call)
	toolsToSend := ragConfig.tools
	var toolChoiceToSend interface{} = "auto"

	// Первый запрос всегда отправляем с инструментами
	for turn := 1; turn <= maxTurns; turn++ {
		// На последнем ходу отключаем инструменты — модель должна дать финальный ответ
		if turn == maxTurns {
			toolsToSend = nil
			toolChoiceToSend = nil
		}

		req := llm.ChatCompletionRequest{
			Messages:   messages,
			Tools:      toolsToSend,
			ToolChoice: toolChoiceToSend,
		}

		select {
		case <-ctx.Done():
			return "", ctx.Err()
		default:
		}

		config.Log.Debug("sending request to LLM", "turn", turn, "no_tools", toolsToSend == nil)

		resp, err := client.SendRequest(ctx, req)
		if err != nil {
			return "", fmt.Errorf("failed to send request: %w", err)
		}

		if len(resp.Choices) == 0 {
			return "", fmt.Errorf("no choices in response")
		}

		choice := resp.Choices[0]
		messages = append(messages, choice.Message)

		config.Log.Debug("got response from LLM", "role", choice.Message.Role, "content", choice.Message.Content)

		// Проверяем, есть ли tool calls
		if len(choice.Message.ToolCalls) > 0 {
			config.Log.Info("received tool calls from model", "count", len(choice.Message.ToolCalls))

			// Выполняем tool calls
			results, err := ExecuteToolCalls(ctx, choice.Message.ToolCalls, filePath)
			if err != nil {
				return "", fmt.Errorf("failed to execute tool calls: %w", err)
			}

			// Добавляем результаты в историю диалога (в правильном порядке согласно спецификации)
			config.Log.Debug("tool calls results", "results", results)
			for i, result := range results {
				messages = append(messages, llm.Message{
					Role:       "tool",
					Name:       choice.Message.ToolCalls[i].Function.Name,
					Content:    result,
					ToolCallID: choice.Message.ToolCalls[i].ID,
				})
			}

			firstResult := firstToolResult(results)
			// Добавляем assistant message с результатами tool calls, чтобы модель использовала их
			toolResponse := fmt.Sprintf("Я выполнил поиск. Вот результаты:\n\n%s\n\nТеперь ответь на вопрос пользователя, используя найденную информацию.", firstResult)
			messages = append(messages, llm.Message{
				Role:    "assistant",
				Content: toolResponse,
			})

			// Если это был последний ход, не продолжаем цикл — возвращаем результаты tool calls
			if turn == maxTurns {
				config.Log.Info("this was the last turn, returning tool results as answer")
				return strings.TrimSpace(firstResult), nil
			}

			continue
		}

		// Если нет tool calls, значит модель сгенерировала финальный ответ
		config.Log.Info("received final answer from model")
		return strings.TrimSpace(choice.Message.Content), nil
	}

	return "", fmt.Errorf("max turns reached without final answer")
}

func firstToolResult(results []string) string {
	if len(results) == 0 {
		return ""
	}

	return results[0]
}
