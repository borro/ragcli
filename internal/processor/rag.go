package processor

import (
	"context"
	"fmt"

	"github.com/borro/ragcli/internal/config"
	"github.com/borro/ragcli/internal/llm"
)

const ragMaxTurns = 8

type chatRequester interface {
	SendRequest(ctx context.Context, req llm.ChatCompletionRequest) (*llm.ChatCompletionResponse, error)
}

// toolsConfig содержит инструменты и системный промпт для tool-calling режима.
type toolsConfig struct {
	tools         []llm.Tool
	systemMessage llm.Message
	userQuestion  llm.Message
}

// NewToolsConfig создаёт конфигурацию инструментов для tool-calling режима.
func NewToolsConfig(prompt string) toolsConfig {
	// Определяем инструменты
	tools := []llm.Tool{
		{
			Type: "function",
			Function: &llm.FunctionDefinition{
				Name:        "search_file",
				Description: "Искать по файлу. query сначала интерпретируется как regex; если regex невалиден, используется case-insensitive substring search. Возвращает JSON с query, mode, match_count и matches[].",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"query": map[string]any{
							"type":        "string",
							"description": "Regex или обычная строка для поиска по файлу",
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
				Description: "Прочитать диапазон строк из файла. Возвращает JSON с start_line, end_line, line_count и lines[].",
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
		Content: `Ты — ИИ-агент по исследованию большого файла.
Отвечай на вопрос пользователя только на основе информации, найденной через инструменты search_file и read_lines.
Файл целиком недоступен, поэтому при необходимости делай несколько последовательных вызовов инструментов.
Если search_file вернул совпадения, используй read_lines, чтобы прочитать нужный диапазон и собрать контекст.
Если информации недостаточно, честно скажи об этом.
Не выдумывай факты и не утверждай то, чего нет в результатах инструментов.
Результаты инструментов приходят в JSON. Внимательно используй поля match_count, matches, line_count и lines.`,
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

// RunTools выполняет обработку в режиме tools (Agentic Tool Calling).
// filePath должен указывать на файл с данными; stdin уже нормализуется в temp file на уровне CLI.
func RunTools(ctx context.Context, client chatRequester, filePath, prompt string) (string, error) {
	config.Log.Info("starting tools processing", "file_path", filePath, "prompt", prompt)

	ragConfig := NewToolsConfig(prompt)
	messages := make([]llm.Message, 0, 2)
	messages = append(messages, ragConfig.systemMessage, ragConfig.userQuestion)
	var toolChoiceToSend interface{} = "auto"

	for turn := 1; turn <= ragMaxTurns; turn++ {
		req := llm.ChatCompletionRequest{
			Messages:   messages,
			Tools:      ragConfig.tools,
			ToolChoice: toolChoiceToSend,
		}

		select {
		case <-ctx.Done():
			return "", ctx.Err()
		default:
		}

		config.Log.Debug("sending tools request to LLM", "turn", turn)

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

		if len(choice.Message.ToolCalls) > 0 {
			config.Log.Info("received tool calls from model", "count", len(choice.Message.ToolCalls))

			results, err := ExecuteToolCalls(ctx, choice.Message.ToolCalls, filePath)
			if err != nil {
				return "", fmt.Errorf("failed to execute tool calls: %w", err)
			}

			config.Log.Debug("tool calls results", "results", results)
			for i, result := range results {
				messages = append(messages, llm.Message{
					Role:       "tool",
					Name:       choice.Message.ToolCalls[i].Function.Name,
					Content:    result,
					ToolCallID: choice.Message.ToolCalls[i].ID,
				})
			}

			continue
		}

		config.Log.Info("received final answer from model")
		if choice.Message.Content == "" {
			return "", fmt.Errorf("received final response without content")
		}
		return choice.Message.Content, nil
	}

	return "", fmt.Errorf("tools orchestration exceeded %d turns without final answer", ragMaxTurns)
}
