package processor

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"regexp"
	"strings"

	"github.com/borro/ragcli/internal/config"
	"github.com/borro/ragcli/internal/llm"
)

type SearchMatch struct {
	LineNumber int    `json:"line_number"`
	Content    string `json:"content"`
}

type SearchResult struct {
	Query      string        `json:"query"`
	Mode       string        `json:"mode"`
	MatchCount int           `json:"match_count"`
	Matches    []SearchMatch `json:"matches"`
	Error      string        `json:"error,omitempty"`
}

type LineSlice struct {
	LineNumber int    `json:"line_number"`
	Content    string `json:"content"`
}

type ReadLinesResult struct {
	StartLine int         `json:"start_line"`
	EndLine   int         `json:"end_line"`
	LineCount int         `json:"line_count"`
	Lines     []LineSlice `json:"lines"`
	Error     string      `json:"error,omitempty"`
}

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
		func() (io.ReadCloser, error) {
			return os.Open(fr.path)
		},
	)
}

func newFileReader(path string) *fileReader {
	return &fileReader{path: path}
}

func marshalJSON(v any) (string, error) {
	data, err := json.Marshal(v)
	if err != nil {
		return "", fmt.Errorf("failed to marshal tool result: %w", err)
	}

	return string(data), nil
}

func readAllLines(reader io.Reader) ([]string, error) {
	bufReader := bufio.NewReader(reader)
	lines := make([]string, 0, 128)

	for {
		line, err := bufReader.ReadString('\n')
		if err != nil && err != io.EOF {
			return nil, err
		}

		if len(line) > 0 {
			lines = append(lines, strings.TrimSuffix(line, "\n"))
		}

		if err == io.EOF {
			break
		}
	}

	return lines, nil
}

func searchInReader(
	sourceName string,
	sourceKind string,
	query string,
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

	lines, err := readAllLines(reader)
	if err != nil {
		return "", fmt.Errorf("failed to scan %s: %w", sourceKind, err)
	}

	result := SearchResult{
		Query:   query,
		Mode:    "substring",
		Matches: []SearchMatch{},
	}

	re, regexErr := regexp.Compile("(?i)" + query)
	if regexErr == nil {
		result.Mode = "regex"
		for idx, line := range lines {
			if re.MatchString(line) {
				result.Matches = append(result.Matches, SearchMatch{
					LineNumber: idx + 1,
					Content:    line,
				})
			}
		}
	} else {
		lowerQuery := strings.ToLower(query)
		for idx, line := range lines {
			if strings.Contains(strings.ToLower(line), lowerQuery) {
				result.Matches = append(result.Matches, SearchMatch{
					LineNumber: idx + 1,
					Content:    line,
				})
			}
		}
	}

	result.MatchCount = len(result.Matches)
	return marshalJSON(result)
}

func readLinesFromReader(
	sourceName string,
	sourceKind string,
	start int,
	end int,
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

	lines, err := readAllLines(reader)
	if err != nil {
		return "", fmt.Errorf("failed to scan %s: %w", sourceKind, err)
	}

	result := ReadLinesResult{
		StartLine: start,
		EndLine:   end,
		Lines:     []LineSlice{},
	}

	if end < start {
		return marshalJSON(result)
	}

	for idx, line := range lines {
		lineNumber := idx + 1
		if lineNumber < start {
			continue
		}
		if lineNumber > end {
			break
		}
		result.Lines = append(result.Lines, LineSlice{
			LineNumber: lineNumber,
			Content:    line,
		})
	}

	result.LineCount = len(result.Lines)
	return marshalJSON(result)
}

// SearchFile ищет подстроку в файле, возвращает номера строк с совпадениями.
// Если path пустой, данные читаются из stdin.
func SearchFile(path, query string) (string, error) {
	if path == "" {
		return searchInReader(
			"stdin",
			"stdin",
			query,
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
			toolErr, marshalErr := marshalJSON(map[string]string{"error": err.Error()})
			if marshalErr != nil {
				return nil, marshalErr
			}
			results = append(results, toolErr)
			continue
		}
		results = append(results, result)
	}

	return results, nil
}
