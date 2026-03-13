package filetools

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"os"
	"strings"
	"testing"

	openai "github.com/sashabaranov/go-openai"
)

func TestStdinSearch(t *testing.T) {
	tests := []struct {
		name          string
		input         string
		params        SearchParams
		expectedMode  string
		expectedLines []int
		expectError   bool
	}{
		{
			name: "found match (case insensitive literal)",
			input: `первая строка
вторая строка с текстом
третья строка`,
			params: SearchParams{
				Query: "ТЕКСТ",
			},
			expectedMode:  "literal",
			expectedLines: []int{2},
		},
		{
			name: "token overlap fallback",
			input: `retry behaviour is described here
another line
policy for retries and backoff`,
			params: SearchParams{
				Query: "retry policy",
			},
			expectedMode:  "token_overlap",
			expectedLines: []int{1, 3},
		},
		{
			name: "no match",
			input: `первая строка
вторая строка`,
			params: SearchParams{
				Query: "нет в файле",
			},
			expectedMode:  "token_overlap",
			expectedLines: []int{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			withStdin(t, tt.input, func() {
				result, err := SearchFileWithParams("", tt.params)

				if tt.expectError {
					if err == nil {
						t.Fatal("SearchFileWithParams() expected error, got nil")
					}
					return
				}

				if err != nil {
					t.Fatalf("SearchFileWithParams() error = %v", err)
				}

				assertSearchResult(t, result, tt.params.Query, tt.params.Mode, tt.expectedMode, tt.expectedLines)
			})
		})
	}
}

func TestStdinReadLines(t *testing.T) {
	tests := []struct {
		name          string
		input         string
		startLine     int
		endLine       int
		expectedLines []int
		expectError   bool
	}{
		{
			name: "valid range",
			input: `первая строка
вторая строка
третья строка
четвёртая строка`,
			startLine:     2,
			endLine:       3,
			expectedLines: []int{2, 3},
		},
		{
			name:          "out of bounds end",
			input:         "первая строка\nвторая строка\n",
			startLine:     100,
			endLine:       200,
			expectedLines: []int{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			withStdin(t, tt.input, func() {
				result, err := StdinReadLines(tt.startLine, tt.endLine)

				if tt.expectError {
					if err == nil {
						t.Fatal("StdinReadLines() expected error, got nil")
					}
					return
				}

				if err != nil {
					t.Fatalf("StdinReadLines() error = %v", err)
				}

				assertReadLinesResult(t, result, max(tt.startLine, 1), tt.endLine, tt.expectedLines)
			})
		})
	}
}

func TestStdinReadAround(t *testing.T) {
	withStdin(t, "one\ntwo\nthree\nfour\n", func() {
		result, err := StdinReadAround(2, 1, 1)
		if err != nil {
			t.Fatalf("StdinReadAround() error = %v", err)
		}

		var parsed ReadAroundResult
		if err := json.Unmarshal([]byte(result), &parsed); err != nil {
			t.Fatalf("json.Unmarshal() error = %v", err)
		}

		if parsed.StartLine != 1 || parsed.EndLine != 3 {
			t.Fatalf("range = %d..%d, want 1..3", parsed.StartLine, parsed.EndLine)
		}
	})
}

func TestThinWrappersAndDefaults(t *testing.T) {
	filePath := writeTempFile(t, "alpha\nbeta\n")

	searchResult, err := SearchFile(filePath, "alpha")
	if err != nil {
		t.Fatalf("SearchFile() error = %v", err)
	}
	assertSearchResult(t, searchResult, "alpha", "", "literal", []int{1})

	withStdin(t, "gamma\ndelta\n", func() {
		result, err := StdinSearch("gamma")
		if err != nil {
			t.Fatalf("StdinSearch() error = %v", err)
		}
		assertSearchResult(t, result, "gamma", "", "literal", []int{1})
	})
}

func TestSearchFile(t *testing.T) {
	filePath := writeTempFile(t, `первая строка
вторая строка с текстом
третья строка
четвёртая строка с текстом
пятая строка
`)

	tests := []struct {
		name          string
		filePath      string
		params        SearchParams
		expectedMode  string
		expectedLines []int
		expectError   bool
		errorCode     string
	}{
		{
			name:     "found match using regex",
			filePath: filePath,
			params: SearchParams{
				Query: "текстом$",
				Mode:  "regex",
			},
			expectedMode:  "regex",
			expectedLines: []int{2, 4},
		},
		{
			name:     "literal query with regex chars stays literal in auto mode",
			filePath: filePath,
			params: SearchParams{
				Query: "[текст",
			},
			expectedMode:  "token_overlap",
			expectedLines: []int{},
		},
		{
			name:     "invalid regex returns structured error",
			filePath: filePath,
			params: SearchParams{
				Query: "[текст",
				Mode:  "regex",
			},
			expectError: true,
			errorCode:   "invalid_arguments",
		},
		{
			name:        "non-existent file",
			filePath:    "/non/existent/file.txt",
			params:      SearchParams{Query: "test"},
			expectError: true,
		},
		{
			name:     "paged results expose next_offset",
			filePath: filePath,
			params: SearchParams{
				Query: "строка",
				Limit: 2,
			},
			expectedMode:  "literal",
			expectedLines: []int{1, 2},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := SearchFileWithParams(tt.filePath, tt.params)

			if tt.expectError {
				if err == nil {
					t.Fatal("SearchFileWithParams() expected error, got nil")
				}

				if tt.errorCode != "" {
					var toolErr ToolError
					if !errorsAs(err, &toolErr) {
						t.Fatalf("expected ToolError, got %T", err)
					}
					if toolErr.Code != tt.errorCode {
						t.Fatalf("Code = %q, want %q", toolErr.Code, tt.errorCode)
					}
				}
				return
			}

			if err != nil {
				t.Fatalf("SearchFileWithParams() error = %v", err)
			}

			assertSearchResult(t, result, tt.params.Query, tt.params.Mode, tt.expectedMode, tt.expectedLines)

			if tt.name == "paged results expose next_offset" {
				var parsed SearchResult
				if err := json.Unmarshal([]byte(result), &parsed); err != nil {
					t.Fatalf("json.Unmarshal() error = %v", err)
				}
				if !parsed.HasMore || parsed.NextOffset != 2 || parsed.TotalMatches != 5 {
					t.Fatalf("pagination = %+v, want has_more=true next_offset=2 total_matches=5", parsed)
				}
			}
		})
	}
}

func TestSearchFile_ContextLines(t *testing.T) {
	filePath := writeTempFile(t, "zero\none match\ntwo\n")

	result, err := SearchFileWithParams(filePath, SearchParams{
		Query:        "match",
		ContextLines: 1,
	})
	if err != nil {
		t.Fatalf("SearchFileWithParams() error = %v", err)
	}

	var parsed SearchResult
	if err := json.Unmarshal([]byte(result), &parsed); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}

	if len(parsed.Matches) != 1 {
		t.Fatalf("len(Matches) = %d, want 1", len(parsed.Matches))
	}
	if len(parsed.Matches[0].ContextBefore) != 1 || len(parsed.Matches[0].ContextAfter) != 1 {
		t.Fatalf("context lengths = before:%d after:%d, want 1/1", len(parsed.Matches[0].ContextBefore), len(parsed.Matches[0].ContextAfter))
	}
}

func TestReadLines(t *testing.T) {
	filePath := writeTempFile(t, `первая строка
вторая строка
третья строка
четвёртая строка
пятая строка
шестая строка
седьмая строка
восьмая строка
девятая строка
десятая строка
`)

	tests := []struct {
		name          string
		filePath      string
		startLine     int
		endLine       int
		expectedLines []int
		expectError   bool
	}{
		{
			name:          "valid range",
			filePath:      filePath,
			startLine:     3,
			endLine:       5,
			expectedLines: []int{3, 4, 5},
		},
		{
			name:          "out of bounds start",
			filePath:      filePath,
			startLine:     100,
			endLine:       200,
			expectedLines: []int{},
		},
		{
			name:        "invalid file path",
			filePath:    "/non/existent/file.txt",
			startLine:   1,
			endLine:     5,
			expectError: true,
		},
		{
			name:          "negative start (should be clamped to 1)",
			filePath:      filePath,
			startLine:     -5,
			endLine:       5,
			expectedLines: []int{1, 2, 3, 4, 5},
		},
		{
			name:          "empty range when end before start",
			filePath:      filePath,
			startLine:     6,
			endLine:       5,
			expectedLines: []int{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := ReadLines(tt.filePath, tt.startLine, tt.endLine)

			if tt.expectError {
				if err == nil {
					t.Fatal("ReadLines() expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Fatalf("ReadLines() error = %v", err)
			}

			assertReadLinesResult(t, result, max(tt.startLine, 1), tt.endLine, tt.expectedLines)
		})
	}
}

func TestReadLines_LongLine(t *testing.T) {
	longLine := strings.Repeat("a", 200000)
	filePath := writeTempFile(t, longLine+"\nshort line\n")

	result, err := ReadLines(filePath, 1, 1)
	if err != nil {
		t.Fatalf("ReadLines() error = %v", err)
	}

	var parsed ReadLinesResult
	if err := json.Unmarshal([]byte(result), &parsed); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}

	if parsed.LineCount != 1 {
		t.Fatalf("LineCount = %d, want 1", parsed.LineCount)
	}
	if got := parsed.Lines[0].Content; got != longLine {
		t.Fatalf("long line length = %d, want %d", len(got), len(longLine))
	}
}

func TestReadAround_Bounds(t *testing.T) {
	filePath := writeTempFile(t, "1\n2\n3\n")

	result, err := ReadAround(filePath, 1, 5, 2)
	if err != nil {
		t.Fatalf("ReadAround() error = %v", err)
	}

	var parsed ReadAroundResult
	if err := json.Unmarshal([]byte(result), &parsed); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}

	if parsed.StartLine != 1 || parsed.EndLine != 3 || parsed.LineCount != 3 {
		t.Fatalf("result = %+v, want start=1 end=3 count=3", parsed)
	}
}

func TestReadAround_InvalidLine(t *testing.T) {
	filePath := writeTempFile(t, "1\n2\n")

	_, err := ReadAround(filePath, 0, 1, 1)
	if err == nil {
		t.Fatal("ReadAround() expected error, got nil")
	}

	var toolErr ToolError
	if !errorsAs(err, &toolErr) {
		t.Fatalf("expected ToolError, got %T", err)
	}
	if toolErr.Code != "invalid_arguments" {
		t.Fatalf("Code = %q, want invalid_arguments", toolErr.Code)
	}
}

func TestReadAround_DefaultsForNegativeWindow(t *testing.T) {
	filePath := writeTempFile(t, "1\n2\n3\n4\n5\n")

	result, err := ReadAround(filePath, 3, -1, -1)
	if err != nil {
		t.Fatalf("ReadAround() error = %v", err)
	}

	var parsed ReadAroundResult
	if err := json.Unmarshal([]byte(result), &parsed); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if parsed.Before != DefaultReadAroundBefore || parsed.After != DefaultReadAroundAfter {
		t.Fatalf("window = %d/%d, want defaults %d/%d", parsed.Before, parsed.After, DefaultReadAroundBefore, DefaultReadAroundAfter)
	}
}

func TestCachedLineReader_LoadsOnceAcrossTools(t *testing.T) {
	filePath := writeTempFile(t, "alpha\nbeta\nalpha beta\n")
	reader := NewFileReader(filePath)

	if _, err := ExecuteTool(toolCall("1", "search_file", `{"query":"alpha","limit":1}`), reader); err != nil {
		t.Fatalf("executeTool(search_file) error = %v", err)
	}
	if _, err := ExecuteTool(toolCall("2", "read_lines", `{"start_line":1,"end_line":2}`), reader); err != nil {
		t.Fatalf("executeTool(read_lines) error = %v", err)
	}
	if _, err := ExecuteTool(toolCall("3", "read_around", `{"line":2,"before":1,"after":1}`), reader); err != nil {
		t.Fatalf("executeTool(read_around) error = %v", err)
	}

	if reader.loadCount != 1 {
		t.Fatalf("loadCount = %d, want 1", reader.loadCount)
	}
}

func TestListFiles_DefaultsAndPagination(t *testing.T) {
	reader := NewCorpusReader([]CorpusFile{
		{Path: writeTempFile(t, "alpha\n"), DisplayPath: "a.txt"},
		{Path: writeTempFile(t, "beta\n"), DisplayPath: "nested/b.txt"},
		{Path: writeTempFile(t, "gamma\n"), DisplayPath: "z.log"},
	})

	raw, err := ExecuteTool(toolCall("1", "list_files", `{"limit":2}`), reader)
	if err != nil {
		t.Fatalf("ExecuteTool(list_files) error = %v", err)
	}

	var result ListFilesResult
	if err := json.Unmarshal([]byte(raw), &result); err != nil {
		t.Fatalf("json.Unmarshal(list_files) error = %v", err)
	}
	if result.FileCount != 2 || result.TotalFiles != 3 || !result.HasMore || result.NextOffset != 2 {
		t.Fatalf("result = %+v, want first paged file listing", result)
	}
	if got := strings.Join(result.Files, ","); got != "a.txt,nested/b.txt" {
		t.Fatalf("files = %q, want first page", got)
	}

	secondRaw, err := ExecuteTool(toolCall("2", "list_files", `{"limit":2,"offset":2}`), reader)
	if err != nil {
		t.Fatalf("ExecuteTool(list_files page 2) error = %v", err)
	}
	secondResult := ListFilesResult{}
	if err := json.Unmarshal([]byte(secondRaw), &secondResult); err != nil {
		t.Fatalf("json.Unmarshal(list_files page 2) error = %v", err)
	}
	if secondResult.FileCount != 1 || secondResult.TotalFiles != 3 || secondResult.HasMore || secondResult.NextOffset != 0 {
		t.Fatalf("result = %+v, want second paged file listing", secondResult)
	}
	if got := strings.Join(secondResult.Files, ","); got != "z.log" {
		t.Fatalf("files = %q, want second page", got)
	}
}

func TestSummarizeToolArguments_NormalizesSearchParams(t *testing.T) {
	summary := SummarizeToolArguments(toolCall("1", "search_file", `{"query":"alpha","limit":50,"offset":-1}`))

	if summary["mode"] != "auto" {
		t.Fatalf("mode = %#v, want auto", summary["mode"])
	}
	if summary["limit"] != maxSearchLimit {
		t.Fatalf("limit = %#v, want %d", summary["limit"], maxSearchLimit)
	}
	if summary["offset"] != 0 {
		t.Fatalf("offset = %#v, want 0", summary["offset"])
	}
}

func TestSummarizeToolArguments_OtherToolsAndErrors(t *testing.T) {
	listSummary := SummarizeToolArguments(toolCall("0", "list_files", `{"limit":500,"offset":-1}`))
	if listSummary["limit"] != maxListFilesLimit {
		t.Fatalf("limit = %#v, want %d", listSummary["limit"], maxListFilesLimit)
	}
	if listSummary["offset"] != 0 {
		t.Fatalf("offset = %#v, want 0", listSummary["offset"])
	}

	readLinesSummary := SummarizeToolArguments(toolCall("1", "read_lines", `{"start_line":3,"end_line":7}`))
	if readLinesSummary["start_line"] != float64(3) && readLinesSummary["start_line"] != 3 {
		t.Fatalf("start_line = %#v, want 3", readLinesSummary["start_line"])
	}

	readAroundSummary := SummarizeToolArguments(toolCall("2", "read_around", `{"line":10,"before":2,"after":4}`))
	if readAroundSummary["line"] != float64(10) && readAroundSummary["line"] != 10 {
		t.Fatalf("line = %#v, want 10", readAroundSummary["line"])
	}

	if summary := SummarizeToolArguments(toolCall("3", "search_file", "")); summary != nil {
		t.Fatalf("summary = %#v, want nil for empty args", summary)
	}

	decodeErr := SummarizeToolArguments(toolCall("4", "read_lines", `{"start_line":`))
	if _, ok := decodeErr["decode_error"]; !ok {
		t.Fatalf("summary = %#v, want decode_error", decodeErr)
	}

	unknown := SummarizeToolArguments(toolCall("5", "custom", `{"x":1}`))
	if unknown["raw_arguments"] != `{"x":1}` {
		t.Fatalf("raw_arguments = %#v, want original payload", unknown["raw_arguments"])
	}
}

func TestSummarizeToolResult_SearchPagination(t *testing.T) {
	raw, err := MarshalJSON(SearchResult{
		Mode:         "literal",
		MatchCount:   2,
		TotalMatches: 5,
		HasMore:      true,
		NextOffset:   2,
	})
	if err != nil {
		t.Fatalf("marshalJSON() error = %v", err)
	}

	summary := SummarizeToolResult("search_file", raw)
	if summary["mode"] != "literal" {
		t.Fatalf("mode = %#v, want literal", summary["mode"])
	}
	if summary["match_count"] != float64(2) && summary["match_count"] != 2 {
		t.Fatalf("match_count = %#v, want 2", summary["match_count"])
	}
	if summary["next_offset"] != float64(2) && summary["next_offset"] != 2 {
		t.Fatalf("next_offset = %#v, want 2", summary["next_offset"])
	}
}

func TestSummarizeToolResult_ListFilesPagination(t *testing.T) {
	raw, err := MarshalJSON(ListFilesResult{
		FileCount:  2,
		TotalFiles: 5,
		HasMore:    true,
		NextOffset: 2,
	})
	if err != nil {
		t.Fatalf("marshalJSON(ListFilesResult) error = %v", err)
	}

	summary := SummarizeToolResult("list_files", raw)
	if summary["file_count"] != float64(2) && summary["file_count"] != 2 {
		t.Fatalf("file_count = %#v, want 2", summary["file_count"])
	}
	if summary["total_files"] != float64(5) && summary["total_files"] != 5 {
		t.Fatalf("total_files = %#v, want 5", summary["total_files"])
	}
	if summary["next_offset"] != float64(2) && summary["next_offset"] != 2 {
		t.Fatalf("next_offset = %#v, want 2", summary["next_offset"])
	}
}

func TestSummarizeToolResult_ReadRanges(t *testing.T) {
	readLinesRaw, err := MarshalJSON(ReadLinesResult{
		StartLine: 10,
		EndLine:   12,
		LineCount: 3,
	})
	if err != nil {
		t.Fatalf("marshalJSON(ReadLinesResult) error = %v", err)
	}
	readLinesSummary := SummarizeToolResult("read_lines", readLinesRaw)
	if readLinesSummary["start_line"] != float64(10) && readLinesSummary["start_line"] != 10 {
		t.Fatalf("start_line = %#v, want 10", readLinesSummary["start_line"])
	}
	if readLinesSummary["line_count"] != float64(3) && readLinesSummary["line_count"] != 3 {
		t.Fatalf("line_count = %#v, want 3", readLinesSummary["line_count"])
	}

	readAroundRaw, err := MarshalJSON(ReadAroundResult{
		Line:      11,
		StartLine: 10,
		EndLine:   12,
		LineCount: 3,
	})
	if err != nil {
		t.Fatalf("marshalJSON(ReadAroundResult) error = %v", err)
	}
	readAroundSummary := SummarizeToolResult("read_around", readAroundRaw)
	if readAroundSummary["line"] != float64(11) && readAroundSummary["line"] != 11 {
		t.Fatalf("line = %#v, want 11", readAroundSummary["line"])
	}
	if readAroundSummary["end_line"] != float64(12) && readAroundSummary["end_line"] != 12 {
		t.Fatalf("end_line = %#v, want 12", readAroundSummary["end_line"])
	}
}

func TestSummarizeToolResult_DecodeErrorAndUnknownTool(t *testing.T) {
	decodeErr := SummarizeToolResult("search_file", "{")
	if _, ok := decodeErr["decode_error"]; !ok {
		t.Fatalf("summary = %#v, want decode_error", decodeErr)
	}
	if summary := SummarizeToolResult("custom", "{}"); summary != nil {
		t.Fatalf("summary = %#v, want nil", summary)
	}
}

func TestExecuteTool_ErrorPaths(t *testing.T) {
	reader := NewFileReader(writeTempFile(t, "alpha\nbeta\n"))

	tests := []struct {
		name     string
		call     openai.ToolCall
		wantCode string
	}{
		{
			name:     "invalid list_files args",
			call:     toolCall("0", "list_files", `{"limit":`),
			wantCode: "invalid_arguments",
		},
		{
			name:     "invalid search args",
			call:     toolCall("1", "search_file", `{"query":`),
			wantCode: "invalid_arguments",
		},
		{
			name:     "invalid read_lines args",
			call:     toolCall("2", "read_lines", `{"start_line":`),
			wantCode: "invalid_arguments",
		},
		{
			name:     "invalid read_around args",
			call:     toolCall("3", "read_around", `{"line":`),
			wantCode: "invalid_arguments",
		},
		{
			name:     "unknown tool",
			call:     toolCall("4", "boom", `{}`),
			wantCode: "unknown_tool",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ExecuteTool(tt.call, reader)
			if err == nil {
				t.Fatal("ExecuteTool() error = nil, want error")
			}
			toolErr := AsToolError(err)
			if toolErr.Code != tt.wantCode {
				t.Fatalf("Code = %q, want %q", toolErr.Code, tt.wantCode)
			}
		})
	}
}

func TestExecuteToolCalls_ContextDoneAndToolErrorJSON(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := ExecuteToolCalls(ctx, []openai.ToolCall{toolCall("1", "search_file", `{"query":"x"}`)}, writeTempFile(t, "x\n"))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}

	results, err := ExecuteToolCalls(context.Background(), []openai.ToolCall{
		toolCall("2", "search_file", `{"query":""}`),
	}, writeTempFile(t, "alpha\n"))
	if err != nil {
		t.Fatalf("ExecuteToolCalls() error = %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("len(results) = %d, want 1", len(results))
	}
	var toolErr ToolError
	if err := json.Unmarshal([]byte(results[0]), &toolErr); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if toolErr.Code != "invalid_arguments" {
		t.Fatalf("Code = %q, want invalid_arguments", toolErr.Code)
	}
}

func TestMultiFileReader_SearchAcrossCorpusAndReadByPath(t *testing.T) {
	firstPath := writeTempFile(t, "alpha\nbeta\n")
	secondPath := writeTempFile(t, "gamma\nalpha in nested\n")
	reader := NewCorpusReader([]CorpusFile{
		{Path: firstPath, DisplayPath: "a.txt"},
		{Path: secondPath, DisplayPath: "nested/b.txt"},
	})

	raw, err := ExecuteTool(toolCall("1", "search_file", `{"query":"alpha"}`), reader)
	if err != nil {
		t.Fatalf("ExecuteTool(search_file) error = %v", err)
	}

	var result SearchResult
	if err := json.Unmarshal([]byte(raw), &result); err != nil {
		t.Fatalf("json.Unmarshal(search) error = %v", err)
	}
	if len(result.Matches) != 2 {
		t.Fatalf("len(Matches) = %d, want 2", len(result.Matches))
	}
	if result.Matches[0].Path != "a.txt" {
		t.Fatalf("Matches[0].Path = %q, want a.txt", result.Matches[0].Path)
	}
	if result.Matches[1].Path != "nested/b.txt" {
		t.Fatalf("Matches[1].Path = %q, want nested/b.txt", result.Matches[1].Path)
	}

	linesRaw, err := ExecuteTool(toolCall("2", "read_lines", `{"path":"nested/b.txt","start_line":2,"end_line":2}`), reader)
	if err != nil {
		t.Fatalf("ExecuteTool(read_lines) error = %v", err)
	}
	var lines ReadLinesResult
	if err := json.Unmarshal([]byte(linesRaw), &lines); err != nil {
		t.Fatalf("json.Unmarshal(read_lines) error = %v", err)
	}
	if lines.Path != "nested/b.txt" {
		t.Fatalf("ReadLinesResult.Path = %q, want nested/b.txt", lines.Path)
	}
	if len(lines.Lines) != 1 || lines.Lines[0].Path != "nested/b.txt" || lines.Lines[0].LineNumber != 2 {
		t.Fatalf("read_lines result = %+v, want path-aware line 2", lines)
	}
}

func TestMultiFileReader_RequiresPathForReadTools(t *testing.T) {
	path := writeTempFile(t, "alpha\n")
	reader := NewCorpusReader([]CorpusFile{
		{Path: path, DisplayPath: "a.txt"},
		{Path: writeTempFile(t, "beta\n"), DisplayPath: "b.txt"},
	})

	_, err := ExecuteTool(toolCall("1", "read_lines", `{"start_line":1,"end_line":1}`), reader)
	if err == nil {
		t.Fatal("ExecuteTool(read_lines) error = nil, want error")
	}
	if got := AsToolError(err).Code; got != "invalid_arguments" {
		t.Fatalf("Code = %q, want invalid_arguments", got)
	}
}

func TestToolErrorHelpersAndLogging(t *testing.T) {
	baseErr := errors.New("boom")
	if got := AsToolError(baseErr); got.Code != "tool_execution_failed" || got.Message != "boom" {
		t.Fatalf("AsToolError(baseErr) = %+v, want fallback tool error", got)
	}

	toolErr := NewToolError("invalid_arguments", "bad input", false, map[string]any{"line": 1})
	if toolErr.Error() != "bad input" {
		t.Fatalf("toolErr.Error() = %q, want bad input", toolErr.Error())
	}
	if got := AsToolError(toolErr); got.Code != "invalid_arguments" {
		t.Fatalf("AsToolError(toolErr).Code = %q, want invalid_arguments", got.Code)
	}

	var buf bytes.Buffer
	original := slog.Default()
	logger := slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug}))
	slog.SetDefault(logger)
	t.Cleanup(func() {
		slog.SetDefault(original)
	})

	call := toolCall("1", "search_file", `{"query":"alpha"}`)
	LogToolCallStarted(call, map[string]any{"query": "alpha"})
	LogToolCallFinished(call, 1500000000, "ok", map[string]any{"match_count": 1})
	LogToolCallError(call, 200000000, toolErr)

	logs := buf.String()
	for _, want := range []string{"tool call started", "tool call finished", "tool call failed", "tool_name=search_file"} {
		if !strings.Contains(logs, want) {
			t.Fatalf("logs = %q, want substring %q", logs, want)
		}
	}
}

func assertSearchResult(t *testing.T, raw string, query string, requestedMode string, expectedMode string, expectedLines []int) {
	t.Helper()

	var result SearchResult
	if err := json.Unmarshal([]byte(raw), &result); err != nil {
		t.Fatalf("json.Unmarshal() error = %v, raw=%s", err, raw)
	}

	wantRequested := requestedMode
	if wantRequested == "" {
		wantRequested = "auto"
	}
	if result.Query != query {
		t.Fatalf("Query = %q, want %q", result.Query, query)
	}
	if result.RequestedMode != wantRequested {
		t.Fatalf("RequestedMode = %q, want %q", result.RequestedMode, wantRequested)
	}
	if result.Mode != expectedMode {
		t.Fatalf("Mode = %q, want %q", result.Mode, expectedMode)
	}
	if result.MatchCount != len(expectedLines) {
		t.Fatalf("MatchCount = %d, want %d", result.MatchCount, len(expectedLines))
	}
	if len(result.Matches) != len(expectedLines) {
		t.Fatalf("len(Matches) = %d, want %d", len(result.Matches), len(expectedLines))
	}

	for i, line := range expectedLines {
		if result.Matches[i].LineNumber != line {
			t.Fatalf("Matches[%d].LineNumber = %d, want %d", i, result.Matches[i].LineNumber, line)
		}
	}
}

func toolCall(id string, name string, arguments string) openai.ToolCall {
	return openai.ToolCall{
		ID:   id,
		Type: "function",
		Function: openai.FunctionCall{
			Name:      name,
			Arguments: arguments,
		},
	}
}

func assertReadLinesResult(t *testing.T, raw string, start int, end int, expectedLines []int) {
	t.Helper()

	var result ReadLinesResult
	if err := json.Unmarshal([]byte(raw), &result); err != nil {
		t.Fatalf("json.Unmarshal() error = %v, raw=%s", err, raw)
	}

	if result.StartLine != start {
		t.Fatalf("StartLine = %d, want %d", result.StartLine, start)
	}
	if result.EndLine != end {
		t.Fatalf("EndLine = %d, want %d", result.EndLine, end)
	}
	if result.LineCount != len(expectedLines) {
		t.Fatalf("LineCount = %d, want %d", result.LineCount, len(expectedLines))
	}
	if len(result.Lines) != len(expectedLines) {
		t.Fatalf("len(Lines) = %d, want %d", len(result.Lines), len(expectedLines))
	}

	for i, line := range expectedLines {
		if result.Lines[i].LineNumber != line {
			t.Fatalf("Lines[%d].LineNumber = %d, want %d", i, result.Lines[i].LineNumber, line)
		}
	}
}

func writeTempFile(t *testing.T, content string) string {
	t.Helper()

	tmpFile, err := os.CreateTemp("", "processor-tools-*.txt")
	if err != nil {
		t.Fatalf("failed to create temp file: %v", err)
	}

	if _, err := tmpFile.WriteString(content); err != nil {
		t.Fatalf("failed to write temp file: %v", err)
	}

	if err := tmpFile.Close(); err != nil {
		t.Fatalf("failed to close temp file: %v", err)
	}

	t.Cleanup(func() {
		if err := os.Remove(tmpFile.Name()); err != nil {
			t.Logf("failed to remove temp file: %v", err)
		}
	})

	return tmpFile.Name()
}

func withStdin(t *testing.T, input string, fn func()) {
	t.Helper()

	originalStdin := os.Stdin
	reader, writer, err := os.Pipe()
	if err != nil {
		t.Fatalf("failed to create pipe: %v", err)
	}

	if _, err := writer.WriteString(input); err != nil {
		t.Fatalf("failed to write stdin fixture: %v", err)
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("failed to close stdin writer: %v", err)
	}

	os.Stdin = reader
	t.Cleanup(func() {
		os.Stdin = originalStdin
		if err := reader.Close(); err != nil {
			t.Logf("failed to close stdin reader: %v", err)
		}
	})

	fn()
}
