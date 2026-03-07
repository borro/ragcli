package processor

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
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

func TestCachedLineReader_LoadsOnceAcrossTools(t *testing.T) {
	filePath := writeTempFile(t, "alpha\nbeta\nalpha beta\n")
	reader := newFileReader(filePath)

	if _, err := executeTool(toolCall("1", "search_file", `{"query":"alpha","limit":1}`), reader); err != nil {
		t.Fatalf("executeTool(search_file) error = %v", err)
	}
	if _, err := executeTool(toolCall("2", "read_lines", `{"start_line":1,"end_line":2}`), reader); err != nil {
		t.Fatalf("executeTool(read_lines) error = %v", err)
	}
	if _, err := executeTool(toolCall("3", "read_around", `{"line":2,"before":1,"after":1}`), reader); err != nil {
		t.Fatalf("executeTool(read_around) error = %v", err)
	}

	if reader.loadCount != 1 {
		t.Fatalf("loadCount = %d, want 1", reader.loadCount)
	}
}

func TestSummarizeToolArguments_NormalizesSearchParams(t *testing.T) {
	summary := summarizeToolArguments(toolCall("1", "search_file", `{"query":"alpha","limit":50,"offset":-1}`))

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

func TestSummarizeToolResult_SearchPagination(t *testing.T) {
	raw, err := marshalJSON(SearchResult{
		Mode:         "literal",
		MatchCount:   2,
		TotalMatches: 5,
		HasMore:      true,
		NextOffset:   2,
	})
	if err != nil {
		t.Fatalf("marshalJSON() error = %v", err)
	}

	summary := summarizeToolResult("search_file", raw)
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

func TestSummarizeToolResult_ReadRanges(t *testing.T) {
	readLinesRaw, err := marshalJSON(ReadLinesResult{
		StartLine: 10,
		EndLine:   12,
		LineCount: 3,
	})
	if err != nil {
		t.Fatalf("marshalJSON(ReadLinesResult) error = %v", err)
	}
	readLinesSummary := summarizeToolResult("read_lines", readLinesRaw)
	if readLinesSummary["start_line"] != float64(10) && readLinesSummary["start_line"] != 10 {
		t.Fatalf("start_line = %#v, want 10", readLinesSummary["start_line"])
	}
	if readLinesSummary["line_count"] != float64(3) && readLinesSummary["line_count"] != 3 {
		t.Fatalf("line_count = %#v, want 3", readLinesSummary["line_count"])
	}

	readAroundRaw, err := marshalJSON(ReadAroundResult{
		Line:      11,
		StartLine: 10,
		EndLine:   12,
		LineCount: 3,
	})
	if err != nil {
		t.Fatalf("marshalJSON(ReadAroundResult) error = %v", err)
	}
	readAroundSummary := summarizeToolResult("read_around", readAroundRaw)
	if readAroundSummary["line"] != float64(11) && readAroundSummary["line"] != 11 {
		t.Fatalf("line = %#v, want 11", readAroundSummary["line"])
	}
	if readAroundSummary["end_line"] != float64(12) && readAroundSummary["end_line"] != 12 {
		t.Fatalf("end_line = %#v, want 12", readAroundSummary["end_line"])
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
