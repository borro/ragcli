package files

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"os"
	"strings"
	"testing"

	"github.com/borro/ragcli/internal/aitools"
	openai "github.com/sashabaranov/go-openai"
)

func payloadAs[T any](t *testing.T, execution aitools.Execution) T {
	t.Helper()
	payload, ok := execution.Result.Payload.(T)
	if !ok {
		t.Fatalf("payload type = %T, want %T", execution.Result.Payload, *new(T))
	}
	return payload
}

func payloadJSON(t *testing.T, execution aitools.Execution) string {
	t.Helper()
	raw, err := aitools.MarshalJSON(execution.Result.Payload)
	if err != nil {
		t.Fatalf("MarshalJSON(payload) error = %v", err)
	}
	return raw
}

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
				result, err := executeSearch("", tt.params)

				if tt.expectError {
					if err == nil {
						t.Fatal("executeSearch() expected error, got nil")
					}
					return
				}

				if err != nil {
					t.Fatalf("executeSearch() error = %v", err)
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
				result, err := executeReadLines("", tt.startLine, tt.endLine)

				if tt.expectError {
					if err == nil {
						t.Fatal("executeReadLines() expected error, got nil")
					}
					return
				}

				if err != nil {
					t.Fatalf("executeReadLines() error = %v", err)
				}

				assertReadLinesResult(t, result, max(tt.startLine, 1), tt.endLine, tt.expectedLines)
			})
		})
	}
}

func TestStdinReadAround(t *testing.T) {
	withStdin(t, "one\ntwo\nthree\nfour\n", func() {
		result, err := executeReadAround("", 2, 1, 1)
		if err != nil {
			t.Fatalf("executeReadAround() error = %v", err)
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

func TestSingleFileReadersAndDefaults(t *testing.T) {
	filePath := writeTempFile(t, "alpha\nbeta\n")

	searchResult, err := executeSearch(filePath, SearchParams{Query: "alpha"})
	if err != nil {
		t.Fatalf("executeSearch() error = %v", err)
	}
	assertSearchResult(t, searchResult, "alpha", "", "literal", []int{1})

	withStdin(t, "gamma\ndelta\n", func() {
		result, err := executeSearch("", SearchParams{Query: "gamma"})
		if err != nil {
			t.Fatalf("executeSearch() error = %v", err)
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
			result, err := executeSearch(tt.filePath, tt.params)

			if tt.expectError {
				if err == nil {
					t.Fatal("executeSearch() expected error, got nil")
				}

				if tt.errorCode != "" {
					var toolErr ToolError
					if !errors.As(err, &toolErr) {
						t.Fatalf("expected ToolError, got %T", err)
					}
					if toolErr.Code != tt.errorCode {
						t.Fatalf("Code = %q, want %q", toolErr.Code, tt.errorCode)
					}
				}
				return
			}

			if err != nil {
				t.Fatalf("executeSearch() error = %v", err)
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

	result, err := executeSearch(filePath, SearchParams{
		Query:        "match",
		ContextLines: 1,
	})
	if err != nil {
		t.Fatalf("executeSearch() error = %v", err)
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
			result, err := executeReadLines(tt.filePath, tt.startLine, tt.endLine)

			if tt.expectError {
				if err == nil {
					t.Fatal("executeReadLines() expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Fatalf("executeReadLines() error = %v", err)
			}

			assertReadLinesResult(t, result, max(tt.startLine, 1), tt.endLine, tt.expectedLines)
		})
	}
}

func TestReadLines_LongLine(t *testing.T) {
	longLine := strings.Repeat("a", 200000)
	filePath := writeTempFile(t, longLine+"\nshort line\n")

	result, err := executeReadLines(filePath, 1, 1)
	if err != nil {
		t.Fatalf("executeReadLines() error = %v", err)
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

	result, err := executeReadAround(filePath, 1, 5, 2)
	if err != nil {
		t.Fatalf("executeReadAround() error = %v", err)
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

	_, err := executeReadAround(filePath, 0, 1, 1)
	if err == nil {
		t.Fatal("executeReadAround() expected error, got nil")
	}

	var toolErr ToolError
	if !errors.As(err, &toolErr) {
		t.Fatalf("expected ToolError, got %T", err)
	}
	if toolErr.Code != "invalid_arguments" {
		t.Fatalf("Code = %q, want invalid_arguments", toolErr.Code)
	}
}

func TestReadAround_DefaultsForNegativeWindow(t *testing.T) {
	filePath := writeTempFile(t, "1\n2\n3\n4\n5\n")

	result, err := executeReadAround(filePath, 3, -1, -1)
	if err != nil {
		t.Fatalf("executeReadAround() error = %v", err)
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

	execution, err := ExecuteTool(toolCall("1", "list_files", `{"limit":2}`), reader)
	if err != nil {
		t.Fatalf("ExecuteTool(list_files) error = %v", err)
	}
	result := payloadAs[ListFilesResult](t, execution)
	if result.FileCount != 2 || result.TotalFiles != 3 || !result.HasMore || result.NextOffset != 2 {
		t.Fatalf("result = %+v, want first paged file listing", result)
	}
	if got := strings.Join(result.Files, ","); got != "a.txt,nested/b.txt" {
		t.Fatalf("files = %q, want first page", got)
	}

	secondExecution, err := ExecuteTool(toolCall("2", "list_files", `{"limit":2,"offset":2}`), reader)
	if err != nil {
		t.Fatalf("ExecuteTool(list_files page 2) error = %v", err)
	}
	secondResult := payloadAs[ListFilesResult](t, secondExecution)
	if secondResult.FileCount != 1 || secondResult.TotalFiles != 3 || secondResult.HasMore || secondResult.NextOffset != 0 {
		t.Fatalf("result = %+v, want second paged file listing", secondResult)
	}
	if got := strings.Join(secondResult.Files, ","); got != "z.log" {
		t.Fatalf("files = %q, want second page", got)
	}
}

func TestSearchAutoAcrossCorpusPrefersLiteralMatches(t *testing.T) {
	reader := NewCorpusReader([]CorpusFile{
		{Path: writeTempFile(t, "retry only\n"), DisplayPath: "a.txt"},
		{Path: writeTempFile(t, "retry policy\n"), DisplayPath: "b.txt"},
	})

	execution, err := ExecuteTool(toolCall("1", "search_file", `{"query":"retry policy"}`), reader)
	if err != nil {
		t.Fatalf("ExecuteTool(search_file) error = %v", err)
	}
	result := payloadAs[SearchResult](t, execution)
	if result.Mode != "literal" {
		t.Fatalf("Mode = %q, want literal", result.Mode)
	}
	if len(result.Matches) != 1 || result.Matches[0].Path != "b.txt" {
		t.Fatalf("Matches = %+v, want only literal match from b.txt", result.Matches)
	}
}

func TestSearchAutoAcrossCorpusSortsTokenOverlapGlobally(t *testing.T) {
	reader := NewCorpusReader([]CorpusFile{
		{Path: writeTempFile(t, "retry policy\n"), DisplayPath: "a.txt"},
		{Path: writeTempFile(t, "retry backoff policy\n"), DisplayPath: "b.txt"},
	})

	execution, err := ExecuteTool(toolCall("1", "search_file", `{"query":"retry policy backoff"}`), reader)
	if err != nil {
		t.Fatalf("ExecuteTool(search_file) error = %v", err)
	}
	result := payloadAs[SearchResult](t, execution)
	if result.Mode != "token_overlap" {
		t.Fatalf("Mode = %q, want token_overlap", result.Mode)
	}
	if len(result.Matches) < 2 {
		t.Fatalf("Matches = %+v, want at least two overlap matches", result.Matches)
	}
	if result.Matches[0].Path != "b.txt" {
		t.Fatalf("first match = %+v, want best-scored b.txt result first", result.Matches[0])
	}
}

func TestToolDefinitions_SingleAndMultiFile(t *testing.T) {
	singleFile := NewTools(nil, ToolOptions{MultiFile: false})
	if len(singleFile) != 3 {
		t.Fatalf("len(singleFile) = %d, want 3", len(singleFile))
	}
	singleDefinitions := toolDefinitions(singleFile)
	if toolByName(singleDefinitions, "list_files") != nil {
		t.Fatal("single-file tool set unexpectedly contains list_files")
	}

	readLinesSingle := toolByName(singleDefinitions, "read_lines")
	if readLinesSingle == nil {
		t.Fatal("single-file tool set missing read_lines")
	}
	if required := requiredFields(t, *readLinesSingle); len(required) != 2 || !containsString(required, "start_line") || !containsString(required, "end_line") {
		t.Fatalf("single-file read_lines required = %#v, want start_line/end_line", required)
	}

	multiFile := NewTools(nil, ToolOptions{MultiFile: true})
	if len(multiFile) != 4 {
		t.Fatalf("len(multiFile) = %d, want 4", len(multiFile))
	}
	for i, want := range []string{"list_files", "search_file", "read_lines", "read_around"} {
		if multiFile[i].Name() != want {
			t.Fatalf("tools[%d].Name() = %q, want %q", i, multiFile[i].Name(), want)
		}
	}

	multiDefinitions := toolDefinitions(multiFile)
	listFiles := toolByName(multiDefinitions, "list_files")
	if listFiles == nil {
		t.Fatal("multi-file tool set missing list_files")
	}
	if desc := listFiles.Function.Description; strings.TrimSpace(desc) == "" {
		t.Fatal("list_files description must not be empty")
	}

	readLinesMulti := toolByName(multiDefinitions, "read_lines")
	if readLinesMulti == nil {
		t.Fatal("multi-file tool set missing read_lines")
	}
	if required := requiredFields(t, *readLinesMulti); len(required) != 3 || !containsString(required, "path") || !containsString(required, "start_line") || !containsString(required, "end_line") {
		t.Fatalf("multi-file read_lines required = %#v, want path/start_line/end_line", required)
	}

	readAroundMulti := toolByName(multiDefinitions, "read_around")
	if readAroundMulti == nil {
		t.Fatal("multi-file tool set missing read_around")
	}
	if required := requiredFields(t, *readAroundMulti); len(required) != 2 || !containsString(required, "path") || !containsString(required, "line") {
		t.Fatalf("multi-file read_around required = %#v, want path/line", required)
	}
}

func TestConcreteTools_ExposeSharedInterface(t *testing.T) {
	reader := NewFileReader(writeTempFile(t, "alpha\nbeta\n"))
	tools := []aitools.Tool{
		newListFilesTool(reader),
		newSearchFileTool(reader),
		newReadLinesTool(reader, false),
		newReadAroundTool(reader, false),
	}

	for _, tool := range tools {
		definition := tool.ToolDefinition()
		if definition.Function == nil {
			t.Fatalf("%T.ToolDefinition() returned nil Function", tool)
		}
		if definition.Function.Name != tool.Name() {
			t.Fatalf("%T.ToolDefinition().Function.Name = %q, want %q", tool, definition.Function.Name, tool.Name())
		}
	}
}

func TestConcreteTool_RejectsWrongToolName(t *testing.T) {
	tool := newSearchFileTool(NewFileReader(writeTempFile(t, "alpha\n")))

	_, err := tool.Execute(context.Background(), toolCall("1", "read_lines", `{"start_line":1,"end_line":1}`))
	if err == nil {
		t.Fatal("Execute() error = nil, want unknown_tool")
	}
	if got := AsToolError(err).Code; got != "unknown_tool" {
		t.Fatalf("Code = %q, want unknown_tool", got)
	}
}

func TestConcreteTool_CachesDuplicateCalls(t *testing.T) {
	tool := newSearchFileTool(NewFileReader(writeTempFile(t, "alpha\nbeta\n")))

	first, err := tool.Execute(context.Background(), toolCall("1", "search_file", `{"query":"alpha"}`))
	if err != nil {
		t.Fatalf("Execute(first) error = %v", err)
	}
	second, err := tool.Execute(context.Background(), toolCall("2", "search_file", `{"query":"alpha"}`))
	if err != nil {
		t.Fatalf("Execute(second) error = %v", err)
	}

	if first.Cached || first.Duplicate {
		t.Fatalf("first result = %+v, want fresh execution", first)
	}
	if !second.Cached || !second.Duplicate {
		t.Fatalf("second result = %+v, want cached duplicate", second)
	}
	if payloadJSON(t, second) != payloadJSON(t, first) {
		t.Fatalf("cached payload changed:\nfirst=%q\nsecond=%q", payloadJSON(t, first), payloadJSON(t, second))
	}
}

func TestNewTools_BuildsOrderedDefinitionsAndSearchContract(t *testing.T) {
	reader := NewCorpusReader([]CorpusFile{
		{Path: writeTempFile(t, "alpha\n"), DisplayPath: "a.txt"},
		{Path: writeTempFile(t, "beta\n"), DisplayPath: "nested/b.txt"},
	})
	tools := NewTools(reader, ToolOptions{MultiFile: true})

	definitions := toolDefinitions(tools)
	if len(definitions) != 4 {
		t.Fatalf("len(definitions) = %d, want 4", len(definitions))
	}
	if definitions[0].Function == nil || definitions[0].Function.Name != "list_files" {
		t.Fatalf("definitions[0] = %#v, want list_files", definitions[0].Function)
	}
	if tools[1].Name() != "search_file" {
		t.Fatalf("tools[1].Name() = %q, want search_file", tools[1].Name())
	}

	searchTool := toolInterfaceByName(t, tools, "search_file")
	result, err := searchTool.Execute(context.Background(), toolCall("1", "search_file", `{"query":"alpha"}`))
	if err != nil {
		t.Fatalf("searchTool.Execute() error = %v", err)
	}
	if result.Result.Payload == nil || len(result.Result.ProgressKeys) == 0 {
		t.Fatalf("result = %+v, want payload and progress keys", result)
	}
	payload := payloadAs[SearchResult](t, result)
	if payload.MatchCount != 1 || len(payload.Matches) != 1 {
		t.Fatalf("payload = %+v, want one match", payload)
	}
	if payload.Matches[0].Path != "a.txt" || payload.Matches[0].LineNumber != 1 {
		t.Fatalf("first match = %+v, want a.txt:1", payload.Matches[0])
	}
}

func TestConcreteTools_ExposeProgressKeysAndHints(t *testing.T) {
	reader := NewCorpusReader([]CorpusFile{
		{Path: writeTempFile(t, "alpha\n"), DisplayPath: "a.txt"},
		{Path: writeTempFile(t, "alpha\n"), DisplayPath: "nested/b.txt"},
	})

	listTool := newListFilesTool(reader)
	listResult, err := listTool.Execute(context.Background(), toolCall("1", "list_files", `{"limit":1,"offset":0}`))
	if err != nil {
		t.Fatalf("listTool.Execute() error = %v", err)
	}
	if len(listResult.Result.ProgressKeys) != 1 || listResult.Result.ProgressKeys[0] != "files:file:a.txt" {
		t.Fatalf("listResult.ProgressKeys = %#v, want files:file:a.txt", listResult.Result.ProgressKeys)
	}
	if got := listResult.Result.Hints[aitools.HintSuggestedNextOffset]; got != 1 {
		t.Fatalf("listResult.Hints[%q] = %#v, want 1", HintSuggestedNextOffset, got)
	}

	searchTool := newSearchFileTool(reader)
	searchResult, err := searchTool.Execute(context.Background(), toolCall("2", "search_file", `{"query":"alpha","limit":1}`))
	if err != nil {
		t.Fatalf("searchTool.Execute() error = %v", err)
	}
	if len(searchResult.Result.ProgressKeys) == 0 {
		t.Fatalf("searchResult.ProgressKeys = %#v, want non-empty", searchResult.Result.ProgressKeys)
	}
	if got := searchResult.Result.Hints[aitools.HintSuggestedNextOffset]; got != 1 {
		t.Fatalf("searchResult.Hints[%q] = %#v, want 1", HintSuggestedNextOffset, got)
	}
}

func TestReadLinesRejectsMissingRequiredArguments(t *testing.T) {
	_, err := ExecuteTool(toolCall("1", "read_lines", `{}`), NewFileReader(writeTempFile(t, "alpha\n")))
	if err == nil {
		t.Fatal("ExecuteTool(read_lines) error = nil, want invalid_arguments")
	}
	if got := AsToolError(err).Code; got != "invalid_arguments" {
		t.Fatalf("Code = %q, want invalid_arguments", got)
	}
}

func TestReadAllLinesNormalizesCRLFInput(t *testing.T) {
	filePath := writeTempFile(t, "retry policy\r\nnext line\r\n")

	raw, err := executeSearch(filePath, SearchParams{Query: "policy$", Mode: "regex"})
	if err != nil {
		t.Fatalf("executeSearch() error = %v", err)
	}
	assertSearchResult(t, raw, "policy$", "regex", "regex", []int{1})

	linesRaw, err := executeReadLines(filePath, 1, 1)
	if err != nil {
		t.Fatalf("executeReadLines() error = %v", err)
	}
	var result ReadLinesResult
	if err := json.Unmarshal([]byte(linesRaw), &result); err != nil {
		t.Fatalf("json.Unmarshal(read_lines) error = %v", err)
	}
	if len(result.Lines) != 1 || result.Lines[0].Content != "retry policy" {
		t.Fatalf("Lines = %+v, want CRLF-trimmed content", result.Lines)
	}
}

func TestNewCorpusReaderDisambiguatesDuplicateDisplayPaths(t *testing.T) {
	reader := NewCorpusReader([]CorpusFile{
		{Path: writeTempFile(t, "alpha\n"), DisplayPath: "a.txt"},
		{Path: writeTempFile(t, "beta\n"), DisplayPath: "a.txt"},
	})

	listExec, err := ExecuteTool(toolCall("1", "list_files", `{}`), reader)
	if err != nil {
		t.Fatalf("ExecuteTool(list_files) error = %v", err)
	}
	listResult := payloadAs[ListFilesResult](t, listExec)
	if got := strings.Join(listResult.Files, ","); got != "a.txt,a.txt#2" {
		t.Fatalf("files = %q, want disambiguated duplicate paths", got)
	}

	readExec, err := ExecuteTool(toolCall("2", "read_lines", `{"path":"a.txt#2","start_line":1,"end_line":1}`), reader)
	if err != nil {
		t.Fatalf("ExecuteTool(read_lines) error = %v", err)
	}
	readResult := payloadAs[ReadLinesResult](t, readExec)
	if len(readResult.Lines) != 1 || readResult.Lines[0].Content != "beta" {
		t.Fatalf("readResult = %+v, want second file contents", readResult)
	}
}

func TestCanonicalToolSignature_NormalizesArguments(t *testing.T) {
	searchA, err := canonicalToolSignature(toolCall("1", "search_file", `{"query":"alpha","limit":50,"offset":-1}`))
	if err != nil {
		t.Fatalf("canonicalToolSignature(searchA) error = %v", err)
	}
	searchB, err := canonicalToolSignature(toolCall("2", "search_file", `{"query":"alpha","mode":"auto","limit":20,"offset":0}`))
	if err != nil {
		t.Fatalf("canonicalToolSignature(searchB) error = %v", err)
	}
	if searchA != searchB {
		t.Fatalf("search signatures differ:\nA=%q\nB=%q", searchA, searchB)
	}

	readAroundA, err := canonicalToolSignature(toolCall("3", "read_around", `{"path":"a.txt","line":9,"before":-1,"after":-1}`))
	if err != nil {
		t.Fatalf("canonicalToolSignature(readAroundA) error = %v", err)
	}
	readAroundB, err := canonicalToolSignature(toolCall("4", "read_around", `{"path":"a.txt","line":9,"before":3,"after":3}`))
	if err != nil {
		t.Fatalf("canonicalToolSignature(readAroundB) error = %v", err)
	}
	if readAroundA != readAroundB {
		t.Fatalf("read_around signatures differ:\nA=%q\nB=%q", readAroundA, readAroundB)
	}

	readLinesA, err := canonicalToolSignature(toolCall("5", "read_lines", `{"path":"a.txt","start_line":0,"end_line":5}`))
	if err != nil {
		t.Fatalf("canonicalToolSignature(readLinesA) error = %v", err)
	}
	readLinesB, err := canonicalToolSignature(toolCall("6", "read_lines", `{"path":"a.txt","start_line":1,"end_line":5}`))
	if err != nil {
		t.Fatalf("canonicalToolSignature(readLinesB) error = %v", err)
	}
	if readLinesA != readLinesB {
		t.Fatalf("read_lines signatures differ:\nA=%q\nB=%q", readLinesA, readLinesB)
	}
}

func TestCanonicalToolSignature_InvalidArguments(t *testing.T) {
	_, err := canonicalToolSignature(toolCall("1", "list_files", `{"limit":`))
	if err == nil {
		t.Fatal("canonicalToolSignature() error = nil, want invalid_arguments")
	}
	if got := AsToolError(err).Code; got != "invalid_arguments" {
		t.Fatalf("Code = %q, want invalid_arguments", got)
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

	normalizedReadLines := SummarizeToolArguments(toolCall("2a", "read_lines", `{"start_line":0,"end_line":5}`))
	if normalizedReadLines["start_line"] != 1 {
		t.Fatalf("start_line = %#v, want normalized value 1", normalizedReadLines["start_line"])
	}

	normalizedReadAround := SummarizeToolArguments(toolCall("2b", "read_around", `{"line":10,"before":-1,"after":-1}`))
	if normalizedReadAround["before"] != DefaultReadAroundBefore || normalizedReadAround["after"] != DefaultReadAroundAfter {
		t.Fatalf("window = %#v, want normalized defaults %d/%d", normalizedReadAround, DefaultReadAroundBefore, DefaultReadAroundAfter)
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

func TestConcreteTools_DescribeCall_UsesCompactVerboseLabel(t *testing.T) {
	tools := NewTools(nil, ToolOptions{MultiFile: true})

	tests := []struct {
		name      string
		call      openai.ToolCall
		wantLabel string
		wantArgs  map[string]any
	}{
		{
			name:      "list_files hides default offset",
			call:      toolCall("1", "list_files", `{"limit":500,"offset":0}`),
			wantLabel: "list_files(limit=200)",
			wantArgs: map[string]any{
				"limit":  maxListFilesLimit,
				"offset": 0,
			},
		},
		{
			name:      "search_file keeps key params",
			call:      toolCall("2", "search_file", `{"query":"alpha","path":"a.go","mode":"regex","offset":2}`),
			wantLabel: `search_file(query="alpha", path="a.go", mode="regex", offset=2)`,
			wantArgs: map[string]any{
				"query":  "alpha",
				"path":   "a.go",
				"mode":   "regex",
				"limit":  defaultSearchLimit,
				"offset": 2,
			},
		},
		{
			name:      "read_lines uses range",
			call:      toolCall("3", "read_lines", `{"path":"a.go","start_line":3,"end_line":7}`),
			wantLabel: `read_lines(path="a.go", start_line=3, end_line=7)`,
			wantArgs: map[string]any{
				"path":       "a.go",
				"start_line": 3,
				"end_line":   7,
			},
		},
		{
			name:      "read_around hides default before",
			call:      toolCall("4", "read_around", `{"path":"a.go","line":10,"before":3,"after":1}`),
			wantLabel: `read_around(path="a.go", line=10, after=1)`,
			wantArgs: map[string]any{
				"path":   "a.go",
				"line":   10,
				"before": 3,
				"after":  1,
			},
		},
		{
			name:      "read_around normalizes negative window",
			call:      toolCall("5", "read_around", `{"path":"a.go","line":10,"before":-1,"after":-1}`),
			wantLabel: `read_around(path="a.go", line=10)`,
			wantArgs: map[string]any{
				"path":   "a.go",
				"line":   10,
				"before": DefaultReadAroundBefore,
				"after":  DefaultReadAroundAfter,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tool := toolInterfaceByName(t, tools, tt.call.Function.Name)
			desc := tool.DescribeCall(tt.call)
			if desc.VerboseLabel != tt.wantLabel {
				t.Fatalf("VerboseLabel = %q, want %q", desc.VerboseLabel, tt.wantLabel)
			}
			for key, want := range tt.wantArgs {
				if got := desc.Arguments[key]; got != want {
					t.Fatalf("Arguments[%q] = %#v, want %#v", key, got, want)
				}
			}
		})
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

func TestExtractLineKeys_ByTool(t *testing.T) {
	listFilesRaw, err := MarshalJSON(ListFilesResult{
		Files: []string{"a.txt", "nested/b.txt"},
	})
	if err != nil {
		t.Fatalf("MarshalJSON(ListFilesResult) error = %v", err)
	}
	listKeys, err := extractLineKeys("list_files", listFilesRaw)
	if err != nil {
		t.Fatalf("extractLineKeys(list_files) error = %v", err)
	}
	for _, want := range []string{"files:file:a.txt", "files:file:nested/b.txt"} {
		if _, ok := listKeys[want]; !ok {
			t.Fatalf("list_keys missing %q: %#v", want, listKeys)
		}
	}

	searchRaw, err := MarshalJSON(SearchResult{
		Path: "a.txt",
		Matches: []SearchMatch{
			{
				LineNumber: 10,
				ContextBefore: []LineSlice{
					{LineNumber: 9},
				},
				ContextAfter: []LineSlice{
					{LineNumber: 11, Path: "other.txt"},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("MarshalJSON(SearchResult) error = %v", err)
	}
	searchKeys, err := extractLineKeys("search_file", searchRaw)
	if err != nil {
		t.Fatalf("extractLineKeys(search_file) error = %v", err)
	}
	for _, want := range []string{"files:a.txt:9", "files:a.txt:10", "files:other.txt:11"} {
		if _, ok := searchKeys[want]; !ok {
			t.Fatalf("search_keys missing %q: %#v", want, searchKeys)
		}
	}

	readLinesRaw, err := MarshalJSON(ReadLinesResult{
		Path: "nested/b.txt",
		Lines: []LineSlice{
			{LineNumber: 4},
		},
	})
	if err != nil {
		t.Fatalf("MarshalJSON(ReadLinesResult) error = %v", err)
	}
	readLinesKeys, err := extractLineKeys("read_lines", readLinesRaw)
	if err != nil {
		t.Fatalf("extractLineKeys(read_lines) error = %v", err)
	}
	if _, ok := readLinesKeys["files:nested/b.txt:4"]; !ok {
		t.Fatalf("read_lines keys = %#v, want files:nested/b.txt:4", readLinesKeys)
	}

	readAroundRaw, err := MarshalJSON(ReadAroundResult{
		Path: "nested/c.txt",
		Lines: []LineSlice{
			{LineNumber: 7},
		},
	})
	if err != nil {
		t.Fatalf("MarshalJSON(ReadAroundResult) error = %v", err)
	}
	readAroundKeys, err := extractLineKeys("read_around", readAroundRaw)
	if err != nil {
		t.Fatalf("extractLineKeys(read_around) error = %v", err)
	}
	if _, ok := readAroundKeys["files:nested/c.txt:7"]; !ok {
		t.Fatalf("read_around keys = %#v, want files:nested/c.txt:7", readAroundKeys)
	}
}

func TestSuggestedNextOffset_Pagination(t *testing.T) {
	listFilesRaw, err := MarshalJSON(ListFilesResult{
		HasMore:    true,
		NextOffset: 5,
	})
	if err != nil {
		t.Fatalf("MarshalJSON(ListFilesResult) error = %v", err)
	}
	if got := suggestedNextOffset("list_files", listFilesRaw); got != 5 {
		t.Fatalf("suggestedNextOffset(list_files) = %d, want 5", got)
	}

	searchRaw, err := MarshalJSON(SearchResult{
		HasMore:    false,
		NextOffset: 9,
	})
	if err != nil {
		t.Fatalf("MarshalJSON(SearchResult) error = %v", err)
	}
	if got := suggestedNextOffset("search_file", searchRaw); got != 0 {
		t.Fatalf("suggestedNextOffset(search_file) = %d, want 0", got)
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

func TestExecuteTool_MarshalToolErrorJSON(t *testing.T) {
	reader := NewFileReader(writeTempFile(t, "alpha\n"))

	_, err := ExecuteTool(toolCall("2", "search_file", `{"query":""}`), reader)
	if err == nil {
		t.Fatal("ExecuteTool() error = nil, want invalid_arguments")
	}

	raw, marshalErr := MarshalJSON(AsToolError(err))
	if marshalErr != nil {
		t.Fatalf("MarshalJSON() error = %v", marshalErr)
	}

	var toolErr ToolError
	if err := json.Unmarshal([]byte(raw), &toolErr); err != nil {
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

	searchExec, err := ExecuteTool(toolCall("1", "search_file", `{"query":"alpha"}`), reader)
	if err != nil {
		t.Fatalf("ExecuteTool(search_file) error = %v", err)
	}
	result := payloadAs[SearchResult](t, searchExec)
	if len(result.Matches) != 2 {
		t.Fatalf("len(Matches) = %d, want 2", len(result.Matches))
	}
	if result.Matches[0].Path != "a.txt" {
		t.Fatalf("Matches[0].Path = %q, want a.txt", result.Matches[0].Path)
	}
	if result.Matches[1].Path != "nested/b.txt" {
		t.Fatalf("Matches[1].Path = %q, want nested/b.txt", result.Matches[1].Path)
	}

	linesExec, err := ExecuteTool(toolCall("2", "read_lines", `{"path":"nested/b.txt","start_line":2,"end_line":2}`), reader)
	if err != nil {
		t.Fatalf("ExecuteTool(read_lines) error = %v", err)
	}
	lines := payloadAs[ReadLinesResult](t, linesExec)
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
	aitools.LogToolCallStarted(call, map[string]any{"query": "alpha"})
	aitools.LogToolCallFinished(call, 1500000000, "ok", map[string]any{"match_count": 1})
	aitools.LogToolCallError(call, 200000000, map[string]any{"query": "alpha"}, toolErr)

	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	if len(lines) != 3 {
		t.Fatalf("log lines = %d, want 3\nlogs=%q", len(lines), buf.String())
	}
	for _, want := range []string{"tool call started", "tool call finished", "tool call failed", "tool_name=search_file"} {
		if !strings.Contains(buf.String(), want) {
			t.Fatalf("logs = %q, want substring %q", buf.String(), want)
		}
	}
	if !strings.Contains(lines[0], "arguments.query=alpha") {
		t.Fatalf("started log = %q, want structured arguments", lines[0])
	}
	if strings.Contains(lines[1], "arguments.query=alpha") {
		t.Fatalf("finished log = %q, want no duplicated arguments", lines[1])
	}
	if !strings.Contains(lines[2], "arguments.query=alpha") {
		t.Fatalf("failed log = %q, want structured arguments", lines[2])
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

func executeSearch(path string, params SearchParams) (string, error) {
	return newSingleFileReader(path).Search(params)
}

func executeReadLines(path string, startLine int, endLine int) (string, error) {
	return newSingleFileReader(path).ReadLines("", startLine, endLine)
}

func executeReadAround(path string, line int, before int, after int) (string, error) {
	return newSingleFileReader(path).ReadAround("", line, before, after)
}

func newSingleFileReader(path string) *cachedLineReader {
	if path == "" {
		return NewStdinReader()
	}
	return NewFileReader(path)
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

func toolByName(tools []openai.Tool, name string) *openai.Tool {
	for i := range tools {
		if tools[i].Function != nil && tools[i].Function.Name == name {
			return &tools[i]
		}
	}
	return nil
}

func toolDefinitions(tools []aitools.Tool) []openai.Tool {
	definitions := make([]openai.Tool, 0, len(tools))
	for _, tool := range tools {
		definitions = append(definitions, tool.ToolDefinition())
	}
	return definitions
}

func toolInterfaceByName(t *testing.T, tools []aitools.Tool, name string) aitools.Tool {
	t.Helper()

	for _, tool := range tools {
		if tool.Name() == name {
			return tool
		}
	}

	t.Fatalf("tool %q not found", name)
	return nil
}

func requiredFields(t *testing.T, tool openai.Tool) []string {
	t.Helper()

	params, ok := tool.Function.Parameters.(map[string]any)
	if !ok {
		t.Fatalf("Parameters type = %T, want map[string]any", tool.Function.Parameters)
	}

	rawRequired, ok := params["required"]
	if !ok {
		return nil
	}

	required, ok := rawRequired.([]string)
	if ok {
		return required
	}

	rawSlice, ok := rawRequired.([]any)
	if !ok {
		t.Fatalf("required type = %T, want []string or []any", rawRequired)
	}

	required = make([]string, 0, len(rawSlice))
	for _, value := range rawSlice {
		field, ok := value.(string)
		if !ok {
			t.Fatalf("required element type = %T, want string", value)
		}
		required = append(required, field)
	}
	return required
}

func containsString(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
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
