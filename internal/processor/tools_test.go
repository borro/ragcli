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
		query         string
		expectedMode  string
		expectedLines []int
		expectError   bool
	}{
		{
			name: "found match (case insensitive)",
			input: `первая строка
вторая строка с текстом
третья строка`,
			query:         "ТЕКСТ",
			expectedMode:  "regex",
			expectedLines: []int{2},
		},
		{
			name:          "no match",
			input:         "первая строка\nвторая строка\n",
			query:         "нет в файле",
			expectedMode:  "regex",
			expectedLines: []int{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			withStdin(t, tt.input, func() {
				result, err := StdinSearch(tt.query)

				if tt.expectError {
					if err == nil {
						t.Fatal("StdinSearch() expected error, got nil")
					}
					return
				}

				if err != nil {
					t.Fatalf("StdinSearch() error = %v", err)
				}

				assertSearchResult(t, result, tt.query, tt.expectedMode, tt.expectedLines)
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
		query         string
		expectedMode  string
		expectedLines []int
		expectError   bool
	}{
		{
			name:          "found match using regex",
			filePath:      filePath,
			query:         "текстом$",
			expectedMode:  "regex",
			expectedLines: []int{2, 4},
		},
		{
			name:          "no match",
			filePath:      filePath,
			query:         "нет в файле",
			expectedMode:  "regex",
			expectedLines: []int{},
		},
		{
			name:          "invalid regex falls back to substring",
			filePath:      filePath,
			query:         "[текст",
			expectedMode:  "substring",
			expectedLines: []int{},
		},
		{
			name:        "non-existent file",
			filePath:    "/non/existent/file.txt",
			query:       "test",
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := SearchFile(tt.filePath, tt.query)

			if tt.expectError {
				if err == nil {
					t.Fatal("SearchFile() expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Fatalf("SearchFile() error = %v", err)
			}

			assertSearchResult(t, result, tt.query, tt.expectedMode, tt.expectedLines)
		})
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

func assertSearchResult(t *testing.T, raw string, query string, expectedMode string, expectedLines []int) {
	t.Helper()

	var result SearchResult
	if err := json.Unmarshal([]byte(raw), &result); err != nil {
		t.Fatalf("json.Unmarshal() error = %v, raw=%s", err, raw)
	}

	if result.Query != query {
		t.Fatalf("Query = %q, want %q", result.Query, query)
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

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
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
