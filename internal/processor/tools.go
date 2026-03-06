package processor

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/borro/ragcli/internal/llm"
)

// ToolCallResult представляет результат вызова инструмента.
type ToolCallResult struct {
	Name   string `json:"name"`
	Result string `json:"result"`
	Error  string `json:"error,omitempty"`
}

// FileReader интерфейс для доступа к содержимому (файл или stdin).
type FileReader interface {
	Search(query string) (string, error)
	ReadLines(start, end int) (string, error)
}

// fileReader реализует FileReader для файлов.
type fileReader struct {
	path string
}

// SearchFile ищет подстроку в файле, возвращает номера строк с совпадениями.
func (fr *fileReader) Search(query string) (string, error) {
	file, err := os.Open(fr.path)
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

// ReadLines читает диапазон строк из файла.
func (fr *fileReader) ReadLines(start, end int) (string, error) {
	if start < 1 {
		start = 1
	}

	file, err := os.Open(fr.path)
	if err != nil {
		return "", fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	var results []string
	scanner := bufio.NewScanner(file)
	lineNum := 1

	for scanner.Scan() && lineNum <= end {
		if lineNum >= start {
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

// NewFileReader создаёт reader для файла.
func NewFileReader(path string) FileReader {
	return &fileReader{path: path}
}

// SearchFile ищет подстроку в файле, возвращает номера строк с совпадениями.
// Если path пустой, данные читаются из stdin.
func SearchFile(path, query string) (string, error) {
	if path == "" {
		return StdinSearch(query)
	}
	fr := NewFileReader(path)
	return fr.Search(query)
}

// ReadLines читает диапазон строк из файла.
// Если path пустой, данные читаются из stdin.
func ReadLines(path string, startLine, endLine int) (string, error) {
	if startLine < 1 {
		startLine = 1
	}

	if path == "" {
		return StdinReadLines(startLine, endLine)
	}
	fr := NewFileReader(path)
	return fr.ReadLines(startLine, endLine)
}

// StdinSearch читает данные из stdin и ищет подстроку.
func StdinSearch(query string) (string, error) {
	var results []string
	lineNum := 1

	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.Contains(strings.ToLower(line), strings.ToLower(query)) {
			results = append(results, fmt.Sprintf("Line %d: %s", lineNum, line))
		}
		lineNum++
	}

	if err := scanner.Err(); err != nil {
		return "", fmt.Errorf("failed to scan stdin: %w", err)
	}

	if len(results) == 0 {
		return "Нет совпадений в stdin", nil
	}

	return strings.Join(results, "\n"), nil
}

// StdinReadLines читает диапазон строк из stdin.
func StdinReadLines(startLine, endLine int) (string, error) {
	var results []string
	lineNum := 1

	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() && lineNum <= endLine {
		if lineNum >= startLine {
			results = append(results, fmt.Sprintf("%d: %s", lineNum, scanner.Text()))
		}
		lineNum++
	}

	if err := scanner.Err(); err != nil {
		return "", fmt.Errorf("failed to scan stdin: %w", err)
	}

	if len(results) == 0 {
		return "Нет строк в stdin в указанном диапазоне", nil
	}

	return strings.Join(results, "\n"), nil
}

// executeTool выполняет конкретный инструмент через FileReader.
func executeTool(ctx context.Context, model string, call llm.ToolCall, fr FileReader) (string, error) {
	switch call.Function.Name {
	case "search_file":
		var params struct {
			Query string `json:"query"`
		}
		if err := json.Unmarshal([]byte(call.Function.Arguments), &params); err != nil {
			return "", fmt.Errorf("invalid arguments for search_file: %w", err)
		}

		return fr.Search(params.Query)

	case "read_lines":
		var params struct {
			StartLine int `json:"start_line"`
			EndLine   int `json:"end_line"`
		}
		if err := json.Unmarshal([]byte(call.Function.Arguments), &params); err != nil {
			return "", fmt.Errorf("invalid arguments for read_lines: %w", err)
		}

		return fr.ReadLines(params.StartLine, params.EndLine)

	default:
		return "", fmt.Errorf("unknown tool: %s", call.Function.Name)
	}
}

// ExecuteToolCalls обрабатывает вызовы инструментов от LLM и возвращает результаты.
func ExecuteToolCalls(ctx context.Context, client *llm.Client, toolCalls []llm.ToolCall, path string) ([]string, error) {
	fr := NewFileReader(path)

	var results []string
	for _, call := range toolCalls {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		result, err := executeTool(ctx, client.Model(), call, fr)
		if err != nil {
			results = append(results, fmt.Sprintf(`{"error": %q}`, err.Error()))
			continue
		}
		results = append(results, result)
	}

	return results, nil
}
