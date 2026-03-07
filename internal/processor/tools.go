package processor

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"regexp"
	"slices"
	"strings"
	"unicode/utf8"

	"github.com/borro/ragcli/internal/config"
	"github.com/borro/ragcli/internal/llm"
)

const (
	defaultSearchLimit      = 5
	maxSearchLimit          = 20
	defaultReadAroundBefore = 3
	defaultReadAroundAfter  = 3
)

type LineSlice struct {
	LineNumber int    `json:"line_number"`
	Content    string `json:"content"`
}

type SearchMatch struct {
	LineNumber    int         `json:"line_number"`
	Content       string      `json:"content"`
	Score         float64     `json:"score,omitempty"`
	Reason        string      `json:"reason,omitempty"`
	ContextBefore []LineSlice `json:"context_before,omitempty"`
	ContextAfter  []LineSlice `json:"context_after,omitempty"`
}

type SearchResult struct {
	Query         string        `json:"query"`
	RequestedMode string        `json:"requested_mode"`
	Mode          string        `json:"mode"`
	Limit         int           `json:"limit"`
	Offset        int           `json:"offset"`
	MatchCount    int           `json:"match_count"`
	TotalMatches  int           `json:"total_matches"`
	HasMore       bool          `json:"has_more"`
	NextOffset    int           `json:"next_offset,omitempty"`
	Matches       []SearchMatch `json:"matches"`
	Error         string        `json:"error,omitempty"`
}

type ReadLinesResult struct {
	StartLine int         `json:"start_line"`
	EndLine   int         `json:"end_line"`
	LineCount int         `json:"line_count"`
	Lines     []LineSlice `json:"lines"`
	Error     string      `json:"error,omitempty"`
}

type ReadAroundResult struct {
	Line      int         `json:"line"`
	Before    int         `json:"before"`
	After     int         `json:"after"`
	StartLine int         `json:"start_line"`
	EndLine   int         `json:"end_line"`
	LineCount int         `json:"line_count"`
	Lines     []LineSlice `json:"lines"`
	Error     string      `json:"error,omitempty"`
}

type ToolError struct {
	Code      string         `json:"code"`
	Message   string         `json:"message"`
	Retryable bool           `json:"retryable"`
	Details   map[string]any `json:"details,omitempty"`
}

type SearchParams struct {
	Query        string
	Mode         string
	Limit        int
	Offset       int
	ContextLines int
}

type lineReader interface {
	Search(params SearchParams) (string, error)
	ReadLines(start, end int) (string, error)
	ReadAround(line, before, after int) (string, error)
}

type cachedLineReader struct {
	sourceName string
	sourceKind string
	open       func() (io.ReadCloser, error)

	lines     []string
	loadErr   error
	loadCount int
}

func newFileReader(path string) *cachedLineReader {
	return &cachedLineReader{
		sourceName: path,
		sourceKind: "file",
		open: func() (io.ReadCloser, error) {
			return os.Open(path)
		},
	}
}

func newStdinReader() *cachedLineReader {
	return &cachedLineReader{
		sourceName: "stdin",
		sourceKind: "stdin",
		open: func() (io.ReadCloser, error) {
			return io.NopCloser(os.Stdin), nil
		},
	}
}

func (r *cachedLineReader) ensureLoaded() error {
	if r.lines != nil || r.loadErr != nil {
		return r.loadErr
	}

	reader, err := r.open()
	if err != nil {
		r.loadErr = fmt.Errorf("failed to open %s: %w", r.sourceKind, err)
		return r.loadErr
	}
	defer func() {
		if cerr := reader.Close(); cerr != nil {
			config.Log.Warn("failed to close source", "source", r.sourceName, "error", cerr)
		}
	}()

	lines, err := readAllLines(reader)
	if err != nil {
		r.loadErr = fmt.Errorf("failed to scan %s: %w", r.sourceKind, err)
		return r.loadErr
	}

	r.lines = lines
	r.loadCount++
	return nil
}

func (r *cachedLineReader) Search(params SearchParams) (string, error) {
	if err := r.ensureLoaded(); err != nil {
		return "", err
	}

	return searchInLines(r.lines, params)
}

func (r *cachedLineReader) ReadLines(start, end int) (string, error) {
	if err := r.ensureLoaded(); err != nil {
		return "", err
	}

	return readLinesFromLines(r.lines, start, end)
}

func (r *cachedLineReader) ReadAround(line, before, after int) (string, error) {
	if err := r.ensureLoaded(); err != nil {
		return "", err
	}

	return readAroundFromLines(r.lines, line, before, after)
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

func normalizeSearchParams(params SearchParams) SearchParams {
	params.Query = strings.TrimSpace(params.Query)

	if params.Mode == "" {
		params.Mode = "auto"
	}

	switch params.Mode {
	case "auto", "literal", "regex":
	default:
		params.Mode = "auto"
	}

	if params.Limit <= 0 {
		params.Limit = defaultSearchLimit
	}
	if params.Limit > maxSearchLimit {
		params.Limit = maxSearchLimit
	}
	if params.Offset < 0 {
		params.Offset = 0
	}
	if params.ContextLines < 0 {
		params.ContextLines = 0
	}

	return params
}

func searchInLines(lines []string, params SearchParams) (string, error) {
	params = normalizeSearchParams(params)

	if params.Query == "" {
		return "", newToolError("invalid_arguments", "query must not be empty", false, map[string]any{
			"tool":  "search_file",
			"field": "query",
		})
	}

	var matches []SearchMatch
	actualMode := params.Mode

	switch params.Mode {
	case "literal":
		matches = literalMatches(lines, params.Query, params.ContextLines)
	case "regex":
		regexMatches, err := regexMatches(lines, params.Query, params.ContextLines)
		if err != nil {
			return "", err
		}
		matches = regexMatches
	case "auto":
		matches = literalMatches(lines, params.Query, params.ContextLines)
		actualMode = "literal"
		if len(matches) == 0 {
			matches = tokenOverlapMatches(lines, params.Query, params.ContextLines)
			actualMode = "token_overlap"
		}
	default:
		matches = literalMatches(lines, params.Query, params.ContextLines)
		actualMode = "literal"
	}

	totalMatches := len(matches)
	start := min(params.Offset, totalMatches)
	end := min(start+params.Limit, totalMatches)
	pagedMatches := slices.Clone(matches[start:end])

	result := SearchResult{
		Query:         params.Query,
		RequestedMode: params.Mode,
		Mode:          actualMode,
		Limit:         params.Limit,
		Offset:        params.Offset,
		MatchCount:    len(pagedMatches),
		TotalMatches:  totalMatches,
		HasMore:       end < totalMatches,
		Matches:       pagedMatches,
	}
	if result.HasMore {
		result.NextOffset = end
	}

	return marshalJSON(result)
}

func literalMatches(lines []string, query string, contextLines int) []SearchMatch {
	lowerQuery := strings.ToLower(query)
	matches := make([]SearchMatch, 0)
	for idx, line := range lines {
		if strings.Contains(strings.ToLower(line), lowerQuery) {
			matches = append(matches, buildSearchMatch(lines, idx, line, 1, "literal_match", contextLines))
		}
	}
	return matches
}

func regexMatches(lines []string, query string, contextLines int) ([]SearchMatch, error) {
	re, err := regexp.Compile("(?i)" + query)
	if err != nil {
		return nil, newToolError("invalid_arguments", "invalid regex for search_file", false, map[string]any{
			"tool":  "search_file",
			"field": "query",
			"mode":  "regex",
			"error": err.Error(),
		})
	}

	matches := make([]SearchMatch, 0)
	for idx, line := range lines {
		if re.MatchString(line) {
			matches = append(matches, buildSearchMatch(lines, idx, line, 1, "regex_match", contextLines))
		}
	}

	return matches, nil
}

func tokenOverlapMatches(lines []string, query string, contextLines int) []SearchMatch {
	tokens := tokenizeQuery(query)
	if len(tokens) < 2 {
		return []SearchMatch{}
	}

	matches := make([]SearchMatch, 0)
	for idx, line := range lines {
		score, matchedTokens := overlapScore(line, tokens)
		if matchedTokens == 0 {
			continue
		}
		reason := fmt.Sprintf("token_overlap:%d/%d", matchedTokens, len(tokens))
		matches = append(matches, buildSearchMatch(lines, idx, line, score, reason, contextLines))
	}

	slices.SortStableFunc(matches, func(a, b SearchMatch) int {
		switch {
		case a.Score > b.Score:
			return -1
		case a.Score < b.Score:
			return 1
		case a.LineNumber < b.LineNumber:
			return -1
		case a.LineNumber > b.LineNumber:
			return 1
		default:
			return 0
		}
	})

	return matches
}

func tokenizeQuery(query string) []string {
	splitter := regexp.MustCompile(`[^\p{L}\p{N}]+`)
	rawTokens := splitter.Split(strings.ToLower(query), -1)
	uniq := make(map[string]struct{}, len(rawTokens))
	tokens := make([]string, 0, len(rawTokens))
	for _, token := range rawTokens {
		if utf8.RuneCountInString(token) < 2 {
			continue
		}
		if _, exists := uniq[token]; exists {
			continue
		}
		uniq[token] = struct{}{}
		tokens = append(tokens, token)
	}
	return tokens
}

func overlapScore(line string, tokens []string) (float64, int) {
	lowerLine := strings.ToLower(line)
	matchedTokens := 0
	for _, token := range tokens {
		if strings.Contains(lowerLine, token) {
			matchedTokens++
		}
	}
	if matchedTokens == 0 {
		return 0, 0
	}

	score := float64(matchedTokens) / float64(len(tokens))
	return math.Round(score*1000) / 1000, matchedTokens
}

func buildSearchMatch(lines []string, idx int, content string, score float64, reason string, contextLines int) SearchMatch {
	match := SearchMatch{
		LineNumber: idx + 1,
		Content:    content,
		Score:      score,
		Reason:     reason,
	}

	if contextLines > 0 {
		start := max(idx-contextLines, 0)
		if start < idx {
			match.ContextBefore = buildLineSlices(lines, start+1, idx)
		}

		end := min(idx+contextLines+1, len(lines))
		if idx+1 < end {
			match.ContextAfter = buildLineSlices(lines, idx+2, end)
		}
	}

	return match
}

func readLinesFromLines(lines []string, start int, end int) (string, error) {
	if start < 1 {
		start = 1
	}

	result := ReadLinesResult{
		StartLine: start,
		EndLine:   end,
		Lines:     []LineSlice{},
	}

	if end < start {
		return marshalJSON(result)
	}

	result.Lines = buildLineSlices(lines, start, end)
	result.LineCount = len(result.Lines)
	return marshalJSON(result)
}

func readAroundFromLines(lines []string, line, before, after int) (string, error) {
	if line < 1 {
		return "", newToolError("invalid_arguments", "line must be >= 1", false, map[string]any{
			"tool":  "read_around",
			"field": "line",
			"value": line,
		})
	}
	if before < 0 {
		before = defaultReadAroundBefore
	}
	if after < 0 {
		after = defaultReadAroundAfter
	}

	start := max(line-before, 1)
	end := min(line+after, len(lines))
	if len(lines) == 0 {
		start = line
		end = line - 1
	}

	result := ReadAroundResult{
		Line:      line,
		Before:    before,
		After:     after,
		StartLine: start,
		EndLine:   end,
		Lines:     []LineSlice{},
	}

	if start <= end {
		result.Lines = buildLineSlices(lines, start, end)
	}
	result.LineCount = len(result.Lines)

	return marshalJSON(result)
}

func buildLineSlices(lines []string, start int, end int) []LineSlice {
	if start < 1 {
		start = 1
	}
	if end > len(lines) {
		end = len(lines)
	}
	if end < start {
		return []LineSlice{}
	}

	result := make([]LineSlice, 0, end-start+1)
	for lineNumber := start; lineNumber <= end; lineNumber++ {
		result = append(result, LineSlice{
			LineNumber: lineNumber,
			Content:    lines[lineNumber-1],
		})
	}
	return result
}

func SearchFile(path, query string) (string, error) {
	return SearchFileWithParams(path, SearchParams{Query: query})
}

func SearchFileWithParams(path string, params SearchParams) (string, error) {
	var reader *cachedLineReader
	if path == "" {
		reader = newStdinReader()
	} else {
		reader = newFileReader(path)
	}
	return reader.Search(params)
}

func ReadLines(path string, startLine, endLine int) (string, error) {
	var reader *cachedLineReader
	if path == "" {
		reader = newStdinReader()
	} else {
		reader = newFileReader(path)
	}
	return reader.ReadLines(startLine, endLine)
}

func ReadAround(path string, line, before, after int) (string, error) {
	var reader *cachedLineReader
	if path == "" {
		reader = newStdinReader()
	} else {
		reader = newFileReader(path)
	}
	return reader.ReadAround(line, before, after)
}

func StdinSearch(query string) (string, error) {
	return SearchFile("", query)
}

func StdinReadLines(startLine, endLine int) (string, error) {
	return ReadLines("", startLine, endLine)
}

func StdinReadAround(line, before, after int) (string, error) {
	return ReadAround("", line, before, after)
}

func executeTool(call llm.ToolCall, reader lineReader) (string, error) {
	switch call.Function.Name {
	case "search_file":
		var params struct {
			Query        string `json:"query"`
			Mode         string `json:"mode"`
			Limit        int    `json:"limit"`
			Offset       int    `json:"offset"`
			ContextLines int    `json:"context_lines"`
		}
		if err := json.Unmarshal([]byte(call.Function.Arguments), &params); err != nil {
			return "", newToolError("invalid_arguments", "invalid arguments for search_file", false, map[string]any{
				"tool":  "search_file",
				"error": err.Error(),
			})
		}

		return reader.Search(SearchParams{
			Query:        params.Query,
			Mode:         params.Mode,
			Limit:        params.Limit,
			Offset:       params.Offset,
			ContextLines: params.ContextLines,
		})

	case "read_lines":
		var params struct {
			StartLine int `json:"start_line"`
			EndLine   int `json:"end_line"`
		}
		if err := json.Unmarshal([]byte(call.Function.Arguments), &params); err != nil {
			return "", newToolError("invalid_arguments", "invalid arguments for read_lines", false, map[string]any{
				"tool":  "read_lines",
				"error": err.Error(),
			})
		}

		return reader.ReadLines(params.StartLine, params.EndLine)

	case "read_around":
		var params struct {
			Line   int `json:"line"`
			Before int `json:"before"`
			After  int `json:"after"`
		}
		if err := json.Unmarshal([]byte(call.Function.Arguments), &params); err != nil {
			return "", newToolError("invalid_arguments", "invalid arguments for read_around", false, map[string]any{
				"tool":  "read_around",
				"error": err.Error(),
			})
		}

		return reader.ReadAround(params.Line, params.Before, params.After)

	default:
		return "", newToolError("unknown_tool", "unknown tool requested", false, map[string]any{
			"tool": call.Function.Name,
		})
	}
}

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
			toolErr, marshalErr := marshalJSON(asToolError(err))
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

func asToolError(err error) ToolError {
	var toolErr ToolError
	if ok := errorsAs(err, &toolErr); ok {
		return toolErr
	}

	return ToolError{
		Code:      "tool_execution_failed",
		Message:   err.Error(),
		Retryable: false,
	}
}

func newToolError(code, message string, retryable bool, details map[string]any) error {
	return ToolError{
		Code:      code,
		Message:   message,
		Retryable: retryable,
		Details:   details,
	}
}

func (e ToolError) Error() string {
	return e.Message
}

func errorsAs(err error, target *ToolError) bool {
	return errors.As(err, target)
}
