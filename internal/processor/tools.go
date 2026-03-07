package processor

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/borro/ragcli/internal/config"
	"github.com/borro/ragcli/internal/llm"
)

type lineReader interface {
	Search(query string) (string, error)
	ReadLines(start, end int) (string, error)
}

// fileReader реализует доступ к строкам файла.
type fileReader struct {
	path string
}

func (fr *fileReader) Search(query string) (string, error) {
	return searchInReader(
		fr.path,
		"file",
		query,
		"Нет совпадений",
		func() (io.ReadCloser, error) {
			return os.Open(fr.path)
		},
	)
}

// ReadLines читает диапазон строк из файла.
func (fr *fileReader) ReadLines(start, end int) (string, error) {
	return readLinesFromReader(
		fr.path,
		"file",
		start,
		end,
		"Нет строк в указанном диапазоне",
		func() (io.ReadCloser, error) {
			return os.Open(fr.path)
		},
	)
}

func newFileReader(path string) *fileReader {
	return &fileReader{path: path}
}

func searchInReader(
	sourceName string,
	sourceKind string,
	query string,
	noMatchesMessage string,
	open func() (io.ReadCloser, error),
) (string, error) {
	reader, err := open()
	if err != nil {
		return "", fmt.Errorf("failed to open %s: %w", sourceKind, err)
	}
	defer func() {
		if cerr := reader.Close(); cerr != nil {
			config.Log.Warn("failed to close source in Search", "source", sourceName, "error", cerr)
		}
	}()

	var results []string
	scanner := bufio.NewScanner(reader)
	lineNum := 1
	lowerQuery := strings.ToLower(query)

	for scanner.Scan() {
		line := scanner.Text()
		if strings.Contains(strings.ToLower(line), lowerQuery) {
			results = append(results, fmt.Sprintf("Line %d: %s", lineNum, line))
		}
		lineNum++
	}

	if err := scanner.Err(); err != nil {
		return "", fmt.Errorf("failed to scan %s: %w", sourceKind, err)
	}

	if len(results) == 0 {
		return noMatchesMessage, nil
	}

	return strings.Join(results, "\n"), nil
}

func readLinesFromReader(
	sourceName string,
	sourceKind string,
	start int,
	end int,
	emptyRangeMessage string,
	open func() (io.ReadCloser, error),
) (string, error) {
	if start < 1 {
		start = 1
	}

	reader, err := open()
	if err != nil {
		return "", fmt.Errorf("failed to open %s: %w", sourceKind, err)
	}
	defer func() {
		if cerr := reader.Close(); cerr != nil {
			config.Log.Warn("failed to close source in ReadLines", "source", sourceName, "error", cerr)
		}
	}()

	var results []string
	scanner := bufio.NewScanner(reader)
	lineNum := 1

	for scanner.Scan() && lineNum <= end {
		if lineNum >= start {
			results = append(results, fmt.Sprintf("%d: %s", lineNum, scanner.Text()))
		}
		lineNum++
	}

	if err := scanner.Err(); err != nil {
		return "", fmt.Errorf("failed to scan %s: %w", sourceKind, err)
	}

	if len(results) == 0 {
		return emptyRangeMessage, nil
	}

	return strings.Join(results, "\n"), nil
}

// SearchFile ищет подстроку в файле, возвращает номера строк с совпадениями.
// Если path пустой, данные читаются из stdin.
func SearchFile(path, query string) (string, error) {
	if path == "" {
		return searchInReader(
			"stdin",
			"stdin",
			query,
			"Нет совпадений в stdin",
			func() (io.ReadCloser, error) {
				return io.NopCloser(os.Stdin), nil
			},
		)
	}
	fr := newFileReader(path)
	return fr.Search(query)
}

// ReadLines читает диапазон строк из файла.
// Если path пустой, данные читаются из stdin.
func ReadLines(path string, startLine, endLine int) (string, error) {
	if startLine < 1 {
		startLine = 1
	}

	if path == "" {
		return readLinesFromReader(
			"stdin",
			"stdin",
			startLine,
			endLine,
			"Нет строк в stdin в указанном диапазоне",
			func() (io.ReadCloser, error) {
				return io.NopCloser(os.Stdin), nil
			},
		)
	}
	fr := newFileReader(path)
	return fr.ReadLines(startLine, endLine)
}

// StdinSearch читает данные из stdin и ищет подстроку.
func StdinSearch(query string) (string, error) {
	return SearchFile("", query)
}

// StdinReadLines читает диапазон строк из stdin.
func StdinReadLines(startLine, endLine int) (string, error) {
	return ReadLines("", startLine, endLine)
}

// executeTool выполняет конкретный инструмент через file-backed reader.
func executeTool(call llm.ToolCall, reader lineReader) (string, error) {
	switch call.Function.Name {
	case "search_file":
		var params struct {
			Query string `json:"query"`
		}
		if err := json.Unmarshal([]byte(call.Function.Arguments), &params); err != nil {
			return "", fmt.Errorf("invalid arguments for search_file: %w", err)
		}

		return reader.Search(params.Query)

	case "read_lines":
		var params struct {
			StartLine int `json:"start_line"`
			EndLine   int `json:"end_line"`
		}
		if err := json.Unmarshal([]byte(call.Function.Arguments), &params); err != nil {
			return "", fmt.Errorf("invalid arguments for read_lines: %w", err)
		}

		return reader.ReadLines(params.StartLine, params.EndLine)

	default:
		return "", fmt.Errorf("unknown tool: %s", call.Function.Name)
	}
}

// ExecuteToolCalls обрабатывает вызовы инструментов от LLM и возвращает результаты.
func ExecuteToolCalls(ctx context.Context, toolCalls []llm.ToolCall, path string) ([]string, error) {
	config.Log.Debug("ExecuteToolCalls called", "path_length", len(path), "path_is_empty", path == "", "tool_calls_count", len(toolCalls))

	reader := newFileReader(path)

	var results []string
	for _, call := range toolCalls {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		result, err := executeTool(call, reader)
		if err != nil {
			config.Log.Error("executeTool error", "tool_name", call.Function.Name, "error", err)
			results = append(results, fmt.Sprintf(`{"error": %q}`, err.Error()))
			continue
		}
		results = append(results, result)
	}

	return results, nil
}
