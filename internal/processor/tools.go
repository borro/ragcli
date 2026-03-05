package processor

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/borro/ragcli/internal/config"
	"github.com/borro/ragcli/internal/llm"
)

// ToolCallResult представляет результат вызова инструмента
type ToolCallResult struct {
	Name   string `json:"name"`
	Result string `json:"result"`
	Error  string `json:"error,omitempty"`
}

// ExecuteToolCalls обрабатывает вызовы инструментов от LLM и возвращает результаты
func ExecuteToolCalls(ctx context.Context, client *llm.Client, toolCalls []llm.ToolCall, filePath string) ([]string, error) {
	var results []string

	for _, call := range toolCalls {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		config.Log.Debug("executing tool call", "name", call.Function.Name)

		result, err := executeTool(ctx, client.Model(), call, filePath)
		if err != nil {
			results = append(results, fmt.Sprintf(`{"error": %q}`, err.Error()))
			continue
		}
		results = append(results, result)
	}

	return results, nil
}

// executeTool выполняет конкретный инструмент
func executeTool(ctx context.Context, model string, call llm.ToolCall, filePath string) (string, error) {
	switch call.Function.Name {
	case "search_file":
		var params struct {
			Query string `json:"query"`
		}
		if err := json.Unmarshal([]byte(call.Function.Arguments), &params); err != nil {
			return "", fmt.Errorf("invalid arguments for search_file: %w", err)
		}

		results, err := SearchFile(filePath, params.Query)
		if err != nil {
			return "", err
		}
		return results, nil

	case "read_lines":
		var params struct {
			StartLine int `json:"start_line"`
			EndLine   int `json:"end_line"`
		}
		if err := json.Unmarshal([]byte(call.Function.Arguments), &params); err != nil {
			return "", fmt.Errorf("invalid arguments for read_lines: %w", err)
		}

		results, err := ReadLines(filePath, params.StartLine, params.EndLine)
		if err != nil {
			return "", err
		}
		return results, nil

	default:
		return "", fmt.Errorf("unknown tool: %s", call.Function.Name)
	}
}

// SearchFile ищет подстроку или regex в файле, возвращает номера строк с совпадениями
func SearchFile(filePath, query string) (string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return "", fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	var results []string
	scanner := bufio.NewScanner(file)
	lineNum := 1

	for scanner.Scan() {
		line := scanner.Text()
		if strings.Contains(strings.ToLower(line), strings.ToLower(query)) {
			results = append(results, fmt.Sprintf("Line %d: %s", lineNum, line))
		}
		lineNum++
	}

	if err := scanner.Err(); err != nil {
		return "", fmt.Errorf("failed to scan file: %w", err)
	}

	if len(results) == 0 {
		return "Нет совпадений", nil
	}

	return strings.Join(results, "\n"), nil
}

// ReadLines читает диапазон строк из файла
func ReadLines(filePath string, startLine, endLine int) (string, error) {
	if startLine < 1 {
		startLine = 1
	}

	file, err := os.Open(filePath)
	if err != nil {
		return "", fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	var results []string
	scanner := bufio.NewScanner(file)
	lineNum := 1

	for scanner.Scan() && lineNum <= endLine {
		if lineNum >= startLine {
			results = append(results, fmt.Sprintf("%d: %s", lineNum, scanner.Text()))
		}
		lineNum++
	}

	if err := scanner.Err(); err != nil {
		return "", fmt.Errorf("failed to scan file: %w", err)
	}

	if len(results) == 0 {
		return "Нет строк в указанном диапазоне", nil
	}

	return strings.Join(results, "\n"), nil
}
