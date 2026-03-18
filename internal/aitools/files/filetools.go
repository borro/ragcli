package files

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"unicode/utf8"

	"github.com/borro/ragcli/internal/aitools"
	openai "github.com/sashabaranov/go-openai"
)

const (
	defaultSearchLimit      = 5
	maxSearchLimit          = 20
	defaultListFilesLimit   = 50
	maxListFilesLimit       = 200
	DefaultReadAroundBefore = 3
	DefaultReadAroundAfter  = 3
)

type LineSlice struct {
	LineNumber int    `json:"line_number"`
	Content    string `json:"content"`
	Path       string `json:"path,omitempty"`
}

type SearchMatch struct {
	LineNumber    int         `json:"line_number"`
	Content       string      `json:"content"`
	Path          string      `json:"path,omitempty"`
	Score         float64     `json:"score,omitempty"`
	Reason        string      `json:"reason,omitempty"`
	ContextBefore []LineSlice `json:"context_before,omitempty"`
	ContextAfter  []LineSlice `json:"context_after,omitempty"`
}

type SearchResult struct {
	Query         string        `json:"query"`
	Path          string        `json:"path,omitempty"`
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
	Path      string      `json:"path,omitempty"`
	StartLine int         `json:"start_line"`
	EndLine   int         `json:"end_line"`
	LineCount int         `json:"line_count"`
	Lines     []LineSlice `json:"lines"`
	Error     string      `json:"error,omitempty"`
}

type ReadAroundResult struct {
	Path      string      `json:"path,omitempty"`
	Line      int         `json:"line"`
	Before    int         `json:"before"`
	After     int         `json:"after"`
	StartLine int         `json:"start_line"`
	EndLine   int         `json:"end_line"`
	LineCount int         `json:"line_count"`
	Lines     []LineSlice `json:"lines"`
	Error     string      `json:"error,omitempty"`
}

type ListFilesResult struct {
	Limit      int      `json:"limit"`
	Offset     int      `json:"offset"`
	FileCount  int      `json:"file_count"`
	TotalFiles int      `json:"total_files"`
	HasMore    bool     `json:"has_more"`
	NextOffset int      `json:"next_offset,omitempty"`
	Files      []string `json:"files"`
	Error      string   `json:"error,omitempty"`
}

type ToolError = aitools.ToolError

type ListFilesParams struct {
	Limit  int
	Offset int
}

type SearchParams struct {
	Path         string
	Query        string
	Mode         string
	Limit        int
	Offset       int
	ContextLines int
}

type LineReader interface {
	ListFiles(params ListFilesParams) (string, error)
	Search(params SearchParams) (string, error)
	ReadLines(path string, start, end int) (string, error)
	ReadAround(path string, line, before, after int) (string, error)
}

type cachedLineReader struct {
	sourceName  string
	displayPath string
	sourceKind  string
	open        func() (io.ReadCloser, error)

	lines     []string
	loadErr   error
	loadCount int
}

type CorpusFile struct {
	Path        string
	DisplayPath string
}

func AsToolError(err error) ToolError {
	return aitools.AsToolError(err)
}

func NewToolError(code, message string, retryable bool, details map[string]any) error {
	return aitools.NewToolError(code, message, retryable, details)
}

type multiFileReader struct {
	files     []CorpusFile
	byPath    map[string]*cachedLineReader
	ordered   []*cachedLineReader
	loadCount int
}

func NewFileReader(path string) *cachedLineReader {
	return NewNamedFileReader(path, filepath.Base(path))
}

func NewNamedFileReader(path string, displayPath string) *cachedLineReader {
	trimmedDisplayPath := strings.TrimSpace(displayPath)
	if trimmedDisplayPath == "" {
		trimmedDisplayPath = filepath.Base(path)
	}
	return &cachedLineReader{
		sourceName:  path,
		displayPath: trimmedDisplayPath,
		sourceKind:  "file",
		open: func() (io.ReadCloser, error) {
			return os.Open(path)
		},
	}
}

func NewStdinReader() *cachedLineReader {
	return &cachedLineReader{
		sourceName:  "stdin",
		displayPath: "stdin",
		sourceKind:  "stdin",
		open: func() (io.ReadCloser, error) {
			return io.NopCloser(os.Stdin), nil
		},
	}
}

func NewCorpusReader(files []CorpusFile) LineReader {
	normalized := make([]CorpusFile, 0, len(files))
	byPath := make(map[string]*cachedLineReader, len(files))
	ordered := make([]*cachedLineReader, 0, len(files))
	usedPaths := make(map[string]int, len(files))
	for _, file := range files {
		displayPath := strings.TrimSpace(file.DisplayPath)
		if displayPath == "" {
			displayPath = filepath.Base(strings.TrimSpace(file.Path))
		}
		displayPath = uniqueCorpusDisplayPath(displayPath, usedPaths)
		normalized = append(normalized, CorpusFile{
			Path:        file.Path,
			DisplayPath: displayPath,
		})
		reader := &cachedLineReader{
			sourceName:  file.Path,
			displayPath: displayPath,
			sourceKind:  "file",
			open: func(path string) func() (io.ReadCloser, error) {
				return func() (io.ReadCloser, error) {
					return os.Open(path)
				}
			}(file.Path),
		}
		byPath[displayPath] = reader
		ordered = append(ordered, reader)
	}
	return &multiFileReader{
		files:   normalized,
		byPath:  byPath,
		ordered: ordered,
	}
}

func uniqueCorpusDisplayPath(displayPath string, used map[string]int) string {
	if used[displayPath] == 0 {
		used[displayPath] = 1
		return displayPath
	}

	base := displayPath
	for suffix := used[base] + 1; ; suffix++ {
		candidate := fmt.Sprintf("%s#%d", base, suffix)
		if used[candidate] != 0 {
			continue
		}
		used[base] = suffix
		used[candidate] = 1
		return candidate
	}
}

func (r *cachedLineReader) ListFiles(params ListFilesParams) (string, error) {
	return listFilesResult([]string{r.displayPath}, params)
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
			slog.Warn("failed to close source", "source", r.sourceName, "error", cerr)
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

func (r *cachedLineReader) matchesPath(path string) bool {
	trimmed := strings.TrimSpace(path)
	if trimmed == "" {
		return true
	}
	return trimmed == r.displayPath || trimmed == r.sourceName || trimmed == filepath.Base(r.sourceName)
}

func (r *multiFileReader) ListFiles(params ListFilesParams) (string, error) {
	paths := make([]string, 0, len(r.files))
	for _, file := range r.files {
		paths = append(paths, file.DisplayPath)
	}
	return listFilesResult(paths, params)
}

func (r *multiFileReader) Search(params SearchParams) (string, error) {
	params = NormalizeSearchParams(params)
	if strings.TrimSpace(params.Path) != "" {
		reader, err := r.readerForPath(params.Path, "search_file", false)
		if err != nil {
			return "", err
		}
		raw, err := reader.Search(params)
		if err != nil {
			return "", err
		}
		r.refreshLoadCount()
		return raw, nil
	}

	for _, reader := range r.ordered {
		if err := reader.ensureLoaded(); err != nil {
			return "", err
		}
	}
	r.refreshLoadCount()

	allMatches, actualMode, err := searchMatchesAcrossReaders(r.ordered, params)
	if err != nil {
		return "", err
	}

	totalMatches := len(allMatches)
	start := min(params.Offset, totalMatches)
	end := min(start+params.Limit, totalMatches)
	pagedMatches := slices.Clone(allMatches[start:end])

	result := SearchResult{
		Query:         params.Query,
		Path:          params.Path,
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

	return MarshalJSON(result)
}

func searchMatchesAcrossReaders(readers []*cachedLineReader, params SearchParams) ([]SearchMatch, string, error) {
	params = NormalizeSearchParams(params)

	switch params.Mode {
	case "literal":
		matches := make([]SearchMatch, 0, 32)
		for _, reader := range readers {
			matches = append(matches, literalMatches(reader.lines, reader.displayPath, params.Query, params.ContextLines)...)
		}
		return matches, "literal", nil
	case "regex":
		matches := make([]SearchMatch, 0, 32)
		for _, reader := range readers {
			fileMatches, err := regexMatches(reader.lines, reader.displayPath, params.Query, params.ContextLines)
			if err != nil {
				return nil, "", err
			}
			matches = append(matches, fileMatches...)
		}
		return matches, "regex", nil
	case "auto":
		literal := make([]SearchMatch, 0, 32)
		for _, reader := range readers {
			literal = append(literal, literalMatches(reader.lines, reader.displayPath, params.Query, params.ContextLines)...)
		}
		if len(literal) > 0 {
			return literal, "literal", nil
		}

		overlap := make([]SearchMatch, 0, 32)
		for _, reader := range readers {
			overlap = append(overlap, tokenOverlapMatches(reader.lines, reader.displayPath, params.Query, params.ContextLines)...)
		}
		sortTokenOverlapMatches(overlap)
		return overlap, "token_overlap", nil
	default:
		return searchMatchesAcrossReaders(readers, SearchParams{
			Query:        params.Query,
			Mode:         "auto",
			Limit:        params.Limit,
			Offset:       params.Offset,
			ContextLines: params.ContextLines,
		})
	}
}

func (r *multiFileReader) ReadLines(path string, start, end int) (string, error) {
	reader, err := r.readerForPath(path, "read_lines", true)
	if err != nil {
		return "", err
	}
	raw, err := reader.ReadLines(path, start, end)
	if err != nil {
		return "", err
	}
	r.refreshLoadCount()
	return raw, nil
}

func (r *multiFileReader) ReadAround(path string, line, before, after int) (string, error) {
	reader, err := r.readerForPath(path, "read_around", true)
	if err != nil {
		return "", err
	}
	raw, err := reader.ReadAround(path, line, before, after)
	if err != nil {
		return "", err
	}
	r.refreshLoadCount()
	return raw, nil
}

func (r *multiFileReader) readerForPath(path string, tool string, required bool) (*cachedLineReader, error) {
	trimmed := strings.TrimSpace(path)
	if trimmed == "" {
		if required {
			return nil, NewToolError("invalid_arguments", "path is required when multiple files are attached", false, map[string]any{
				"tool":  tool,
				"field": "path",
			})
		}
		return nil, nil
	}
	reader, ok := r.byPath[trimmed]
	if !ok {
		return nil, NewToolError("invalid_arguments", "path is not part of the attached corpus", false, map[string]any{
			"tool": tool,
			"path": trimmed,
		})
	}
	return reader, nil
}

func (r *multiFileReader) refreshLoadCount() {
	total := 0
	for _, reader := range r.ordered {
		total += reader.loadCount
	}
	r.loadCount = total
}

func (r *cachedLineReader) Search(params SearchParams) (string, error) {
	if err := r.ensureLoaded(); err != nil {
		return "", err
	}

	if !r.matchesPath(params.Path) {
		return "", NewToolError("invalid_arguments", "path does not match the available file", false, map[string]any{
			"tool": "search_file",
			"path": params.Path,
		})
	}

	return searchInLines(r.lines, r.displayPath, params)
}

func (r *cachedLineReader) ReadLines(path string, start, end int) (string, error) {
	if err := r.ensureLoaded(); err != nil {
		return "", err
	}
	if !r.matchesPath(path) {
		return "", NewToolError("invalid_arguments", "path does not match the available file", false, map[string]any{
			"tool": "read_lines",
			"path": path,
		})
	}

	return readLinesFromLines(r.lines, r.displayPath, start, end)
}

func (r *cachedLineReader) ReadAround(path string, line, before, after int) (string, error) {
	if err := r.ensureLoaded(); err != nil {
		return "", err
	}
	if !r.matchesPath(path) {
		return "", NewToolError("invalid_arguments", "path does not match the available file", false, map[string]any{
			"tool": "read_around",
			"path": path,
		})
	}

	return readAroundFromLines(r.lines, r.displayPath, line, before, after)
}

func MarshalJSON(v any) (string, error) {
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
			lines = append(lines, strings.TrimRight(line, "\r\n"))
		}

		if err == io.EOF {
			break
		}
	}

	return lines, nil
}

func NormalizeSearchParams(params SearchParams) SearchParams {
	params.Path = strings.TrimSpace(params.Path)
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

func NormalizeListFilesParams(params ListFilesParams) ListFilesParams {
	if params.Limit <= 0 {
		params.Limit = defaultListFilesLimit
	}
	if params.Limit > maxListFilesLimit {
		params.Limit = maxListFilesLimit
	}
	if params.Offset < 0 {
		params.Offset = 0
	}
	return params
}

func listFilesResult(paths []string, params ListFilesParams) (string, error) {
	params = NormalizeListFilesParams(params)

	totalFiles := len(paths)
	start := min(params.Offset, totalFiles)
	end := min(start+params.Limit, totalFiles)

	result := ListFilesResult{
		Limit:      params.Limit,
		Offset:     params.Offset,
		FileCount:  end - start,
		TotalFiles: totalFiles,
		HasMore:    end < totalFiles,
		Files:      slices.Clone(paths[start:end]),
	}
	if result.HasMore {
		result.NextOffset = end
	}

	return MarshalJSON(result)
}

func searchInLines(lines []string, displayPath string, params SearchParams) (string, error) {
	params = NormalizeSearchParams(params)
	matches, actualMode, err := searchMatches(lines, displayPath, params)
	if err != nil {
		return "", err
	}

	totalMatches := len(matches)
	start := min(params.Offset, totalMatches)
	end := min(start+params.Limit, totalMatches)
	pagedMatches := slices.Clone(matches[start:end])

	result := SearchResult{
		Query:         params.Query,
		Path:          displayPath,
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

	return MarshalJSON(result)
}

func searchMatches(lines []string, displayPath string, params SearchParams) ([]SearchMatch, string, error) {
	params = NormalizeSearchParams(params)

	if params.Query == "" {
		return nil, "", NewToolError("invalid_arguments", "query must not be empty", false, map[string]any{
			"tool":  "search_file",
			"field": "query",
		})
	}

	var matches []SearchMatch
	actualMode := params.Mode

	switch params.Mode {
	case "literal":
		matches = literalMatches(lines, displayPath, params.Query, params.ContextLines)
	case "regex":
		regexMatches, err := regexMatches(lines, displayPath, params.Query, params.ContextLines)
		if err != nil {
			return nil, "", err
		}
		matches = regexMatches
	case "auto":
		matches = literalMatches(lines, displayPath, params.Query, params.ContextLines)
		actualMode = "literal"
		if len(matches) == 0 {
			matches = tokenOverlapMatches(lines, displayPath, params.Query, params.ContextLines)
			actualMode = "token_overlap"
		}
	default:
		matches = literalMatches(lines, displayPath, params.Query, params.ContextLines)
		actualMode = "literal"
	}

	return matches, actualMode, nil
}

func literalMatches(lines []string, displayPath string, query string, contextLines int) []SearchMatch {
	lowerQuery := strings.ToLower(query)
	matches := make([]SearchMatch, 0)
	for idx, line := range lines {
		if strings.Contains(strings.ToLower(line), lowerQuery) {
			matches = append(matches, buildSearchMatch(lines, displayPath, idx, line, 1, "literal_match", contextLines))
		}
	}
	return matches
}

func regexMatches(lines []string, displayPath string, query string, contextLines int) ([]SearchMatch, error) {
	re, err := regexp.Compile("(?i)" + query)
	if err != nil {
		return nil, NewToolError("invalid_arguments", "invalid regex for search_file", false, map[string]any{
			"tool":  "search_file",
			"field": "query",
			"mode":  "regex",
			"error": err.Error(),
		})
	}

	matches := make([]SearchMatch, 0)
	for idx, line := range lines {
		if re.MatchString(line) {
			matches = append(matches, buildSearchMatch(lines, displayPath, idx, line, 1, "regex_match", contextLines))
		}
	}

	return matches, nil
}

func tokenOverlapMatches(lines []string, displayPath string, query string, contextLines int) []SearchMatch {
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
		matches = append(matches, buildSearchMatch(lines, displayPath, idx, line, score, reason, contextLines))
	}

	sortTokenOverlapMatches(matches)

	return matches
}

func sortTokenOverlapMatches(matches []SearchMatch) {
	slices.SortStableFunc(matches, func(a, b SearchMatch) int {
		switch {
		case a.Score > b.Score:
			return -1
		case a.Score < b.Score:
			return 1
		case a.Path < b.Path:
			return -1
		case a.Path > b.Path:
			return 1
		case a.LineNumber < b.LineNumber:
			return -1
		case a.LineNumber > b.LineNumber:
			return 1
		default:
			return 0
		}
	})
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

func buildSearchMatch(lines []string, displayPath string, idx int, content string, score float64, reason string, contextLines int) SearchMatch {
	match := SearchMatch{
		LineNumber: idx + 1,
		Content:    content,
		Path:       displayPath,
		Score:      score,
		Reason:     reason,
	}

	if contextLines > 0 {
		start := max(idx-contextLines, 0)
		if start < idx {
			match.ContextBefore = buildLineSlices(lines, displayPath, start+1, idx)
		}

		end := min(idx+contextLines+1, len(lines))
		if idx+1 < end {
			match.ContextAfter = buildLineSlices(lines, displayPath, idx+2, end)
		}
	}

	return match
}

func readLinesFromLines(lines []string, displayPath string, start int, end int) (string, error) {
	if start < 1 {
		start = 1
	}

	result := ReadLinesResult{
		Path:      displayPath,
		StartLine: start,
		EndLine:   end,
		Lines:     []LineSlice{},
	}

	if end < start {
		return MarshalJSON(result)
	}

	result.Lines = buildLineSlices(lines, displayPath, start, end)
	result.LineCount = len(result.Lines)
	return MarshalJSON(result)
}

func readAroundFromLines(lines []string, displayPath string, line, before, after int) (string, error) {
	if line < 1 {
		return "", NewToolError("invalid_arguments", "line must be >= 1", false, map[string]any{
			"tool":  "read_around",
			"field": "line",
			"value": line,
		})
	}
	if before < 0 {
		before = DefaultReadAroundBefore
	}
	if after < 0 {
		after = DefaultReadAroundAfter
	}

	start := max(line-before, 1)
	end := min(line+after, len(lines))
	if len(lines) == 0 {
		start = line
		end = line - 1
	}

	result := ReadAroundResult{
		Path:      displayPath,
		Line:      line,
		Before:    before,
		After:     after,
		StartLine: start,
		EndLine:   end,
		Lines:     []LineSlice{},
	}

	if start <= end {
		result.Lines = buildLineSlices(lines, displayPath, start, end)
	}
	result.LineCount = len(result.Lines)

	return MarshalJSON(result)
}

func buildLineSlices(lines []string, displayPath string, start int, end int) []LineSlice {
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
			Path:       displayPath,
		})
	}
	return result
}

func ExecuteTool(call openai.ToolCall, reader LineReader) (aitools.Execution, error) {
	raw, err := executeToolRaw(call, reader)
	if err != nil {
		return aitools.Execution{}, err
	}
	result, err := buildToolResult(call.Function.Name, raw)
	if err != nil {
		return aitools.Execution{}, err
	}
	return aitools.Execution{Result: result}, nil
}

func executeToolRaw(call openai.ToolCall, reader LineReader) (string, error) {
	args, err := decodeToolArgs(call)
	if err != nil {
		return "", err
	}
	return args.execute(reader)
}

func SummarizeToolArguments(call openai.ToolCall) map[string]any {
	if strings.TrimSpace(call.Function.Arguments) == "" {
		return nil
	}
	if !isKnownToolName(call.Function.Name) {
		return map[string]any{
			"raw_arguments": call.Function.Arguments,
		}
	}
	args, err := decodeToolArgs(call)
	if err != nil {
		return map[string]any{"decode_error": AsToolError(err).Message}
	}
	return args.summary()
}

func compactToolCallLabel(call openai.ToolCall) string {
	label := strings.TrimSpace(call.Function.Name)
	if label == "" {
		label = "tool"
	}

	raw := strings.TrimSpace(call.Function.Arguments)
	if raw == "" {
		return label
	}
	args, err := decodeToolArgs(call)
	if err != nil {
		return compactRawToolCallLabel(label, raw)
	}
	return args.compactLabel(label)
}

func isKnownToolName(name string) bool {
	switch name {
	case "list_files", "search_file", "read_lines", "read_around":
		return true
	default:
		return false
	}
}

func formatToolCallLabel(label string, parts []string) string {
	if len(parts) == 0 {
		return label
	}
	return fmt.Sprintf("%s(%s)", label, strings.Join(parts, ", "))
}

func compactRawToolCallLabel(label string, raw string) string {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return label
	}
	if len(trimmed) > 80 {
		trimmed = trimmed[:77] + "..."
	}
	return fmt.Sprintf("%s(args=%q)", label, trimmed)
}

func formatStringToolParam(name string, value string) string {
	return fmt.Sprintf("%s=%q", name, value)
}

func formatIntToolParam(name string, value int) string {
	return fmt.Sprintf("%s=%d", name, value)
}

func SummarizeToolResult(toolName string, raw string) map[string]any {
	switch toolName {
	case "list_files":
		var result ListFilesResult
		if err := json.Unmarshal([]byte(raw), &result); err != nil {
			return map[string]any{"decode_error": err.Error()}
		}
		summary := map[string]any{
			"file_count":  result.FileCount,
			"total_files": result.TotalFiles,
			"has_more":    result.HasMore,
		}
		if result.NextOffset > 0 {
			summary["next_offset"] = result.NextOffset
		}
		return summary
	case "search_file":
		var result SearchResult
		if err := json.Unmarshal([]byte(raw), &result); err != nil {
			return map[string]any{"decode_error": err.Error()}
		}
		summary := map[string]any{
			"mode":          result.Mode,
			"match_count":   result.MatchCount,
			"total_matches": result.TotalMatches,
			"has_more":      result.HasMore,
		}
		if strings.TrimSpace(result.Path) != "" {
			summary["path"] = strings.TrimSpace(result.Path)
		}
		if result.NextOffset > 0 {
			summary["next_offset"] = result.NextOffset
		}
		return summary
	case "read_lines":
		var result ReadLinesResult
		if err := json.Unmarshal([]byte(raw), &result); err != nil {
			return map[string]any{"decode_error": err.Error()}
		}
		summary := map[string]any{
			"start_line": result.StartLine,
			"end_line":   result.EndLine,
			"line_count": result.LineCount,
		}
		if strings.TrimSpace(result.Path) != "" {
			summary["path"] = strings.TrimSpace(result.Path)
		}
		return summary
	case "read_around":
		var result ReadAroundResult
		if err := json.Unmarshal([]byte(raw), &result); err != nil {
			return map[string]any{"decode_error": err.Error()}
		}
		summary := map[string]any{
			"line":       result.Line,
			"start_line": result.StartLine,
			"end_line":   result.EndLine,
			"line_count": result.LineCount,
		}
		if strings.TrimSpace(result.Path) != "" {
			summary["path"] = strings.TrimSpace(result.Path)
		}
		return summary
	default:
		return nil
	}
}
