package files

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/borro/ragcli/internal/aitools"
	"github.com/borro/ragcli/internal/localize"
	openai "github.com/sashabaranov/go-openai"
)

const HintSuggestedNextOffset = "suggested_next_offset"

type ToolOptions struct {
	MultiFile bool
}

type baseTool struct {
	reader LineReader
	cache  map[string]aitools.ExecuteResult
}

func NewTools(reader LineReader, opts ToolOptions) []aitools.Tool {
	tools := make([]aitools.Tool, 0, 4)
	if opts.MultiFile {
		tools = append(tools, newListFilesTool(reader))
	}
	tools = append(tools,
		newSearchFileTool(reader),
		newReadLinesTool(reader, opts.MultiFile),
		newReadAroundTool(reader, opts.MultiFile),
	)
	return tools
}

func newBaseTool(reader LineReader) baseTool {
	return baseTool{
		reader: reader,
		cache:  make(map[string]aitools.ExecuteResult),
	}
}

func canonicalToolSignature(call openai.ToolCall) (string, error) {
	switch call.Function.Name {
	case "list_files":
		var params struct {
			Limit  int `json:"limit"`
			Offset int `json:"offset"`
		}
		if strings.TrimSpace(call.Function.Arguments) != "" {
			if err := json.Unmarshal([]byte(call.Function.Arguments), &params); err != nil {
				return "", NewToolError("invalid_arguments", "invalid arguments for list_files", false, map[string]any{
					"tool":  "list_files",
					"error": err.Error(),
				})
			}
		}
		normalized := NormalizeListFilesParams(ListFilesParams{
			Limit:  params.Limit,
			Offset: params.Offset,
		})
		body, err := MarshalJSON(normalized)
		if err != nil {
			return "", err
		}
		return call.Function.Name + ":" + body, nil
	case "search_file":
		var params struct {
			Path         string `json:"path"`
			Query        string `json:"query"`
			Mode         string `json:"mode"`
			Limit        int    `json:"limit"`
			Offset       int    `json:"offset"`
			ContextLines int    `json:"context_lines"`
		}
		if err := json.Unmarshal([]byte(call.Function.Arguments), &params); err != nil {
			return "", NewToolError("invalid_arguments", "invalid arguments for search_file", false, map[string]any{
				"tool":  "search_file",
				"error": err.Error(),
			})
		}
		normalized := NormalizeSearchParams(SearchParams{
			Path:         params.Path,
			Query:        params.Query,
			Mode:         params.Mode,
			Limit:        params.Limit,
			Offset:       params.Offset,
			ContextLines: params.ContextLines,
		})
		body, err := MarshalJSON(normalized)
		if err != nil {
			return "", err
		}
		return call.Function.Name + ":" + body, nil
	case "read_lines":
		var params struct {
			Path      string `json:"path"`
			StartLine int    `json:"start_line"`
			EndLine   int    `json:"end_line"`
		}
		if err := json.Unmarshal([]byte(call.Function.Arguments), &params); err != nil {
			return "", NewToolError("invalid_arguments", "invalid arguments for read_lines", false, map[string]any{
				"tool":  "read_lines",
				"error": err.Error(),
			})
		}
		if params.StartLine < 1 {
			params.StartLine = 1
		}
		params.Path = strings.TrimSpace(params.Path)
		body, err := MarshalJSON(params)
		if err != nil {
			return "", err
		}
		return call.Function.Name + ":" + body, nil
	case "read_around":
		var params struct {
			Path   string `json:"path"`
			Line   int    `json:"line"`
			Before int    `json:"before"`
			After  int    `json:"after"`
		}
		if err := json.Unmarshal([]byte(call.Function.Arguments), &params); err != nil {
			return "", NewToolError("invalid_arguments", "invalid arguments for read_around", false, map[string]any{
				"tool":  "read_around",
				"error": err.Error(),
			})
		}
		if params.Before < 0 {
			params.Before = DefaultReadAroundBefore
		}
		if params.After < 0 {
			params.After = DefaultReadAroundAfter
		}
		params.Path = strings.TrimSpace(params.Path)
		body, err := MarshalJSON(params)
		if err != nil {
			return "", err
		}
		return call.Function.Name + ":" + body, nil
	default:
		return "", NewToolError("unknown_tool", "unknown tool requested", false, map[string]any{
			"tool": call.Function.Name,
		})
	}
}

func (b *baseTool) execute(ctx context.Context, name string, call openai.ToolCall) (aitools.ExecuteResult, error) {
	select {
	case <-ctx.Done():
		return aitools.ExecuteResult{}, ctx.Err()
	default:
	}

	if call.Function.Name != name {
		err := NewToolError("unknown_tool", "unknown tool requested", false, map[string]any{
			"tool": call.Function.Name,
		})
		aitools.LogToolCallError(call, 0, err)
		return aitools.ExecuteResult{}, err
	}

	aitools.LogToolCallStarted(call, SummarizeToolArguments(call))

	signature, err := canonicalToolSignature(call)
	if err != nil {
		aitools.LogToolCallError(call, 0, err)
		return aitools.ExecuteResult{}, err
	}

	if cached, ok := b.cache[signature]; ok {
		result := cloneExecuteResult(cached)
		result.Cached = true
		result.Duplicate = true

		summary := cloneSummary(result.Summary)
		if summary == nil {
			summary = map[string]any{}
		}
		summary["cached"] = true
		summary["duplicate"] = true
		aitools.LogToolCallFinished(call, 0, "cached", summary)
		return result, nil
	}

	startedAt := time.Now()
	raw, err := executeToolRaw(call, b.reader)
	if err != nil {
		aitools.LogToolCallError(call, time.Since(startedAt), err)
		return aitools.ExecuteResult{}, err
	}

	keys, err := progressKeys(call.Function.Name, raw)
	if err != nil {
		aitools.LogToolCallError(call, time.Since(startedAt), err)
		return aitools.ExecuteResult{}, err
	}

	result := aitools.ExecuteResult{
		Payload:      raw,
		Summary:      cloneSummary(SummarizeToolResult(call.Function.Name, raw)),
		ProgressKeys: cloneStrings(keys),
		Hints:        cloneHints(toolHints(call.Function.Name, raw)),
	}
	b.cache[signature] = cloneExecuteResult(result)

	aitools.LogToolCallFinished(call, time.Since(startedAt), "ok", cloneSummary(result.Summary))
	return cloneExecuteResult(result), nil
}

type listFilesTool struct {
	baseTool
}

func newListFilesTool(reader LineReader) *listFilesTool {
	return &listFilesTool{baseTool: newBaseTool(reader)}
}

func (t *listFilesTool) Name() string {
	return "list_files"
}

func (t *listFilesTool) ToolDefinition() openai.Tool {
	return openai.Tool{
		Type: "function",
		Function: &openai.FunctionDefinition{
			Name:        t.Name(),
			Description: localize.T("tools.description.list_files"),
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"limit": map[string]any{
						"type":        "integer",
						"description": localize.T("tools.param.limit"),
					},
					"offset": map[string]any{
						"type":        "integer",
						"description": localize.T("tools.param.offset"),
					},
				},
			},
		},
	}
}

func (t *listFilesTool) Execute(ctx context.Context, call openai.ToolCall) (aitools.ExecuteResult, error) {
	return t.execute(ctx, t.Name(), call)
}

type searchFileTool struct {
	baseTool
}

func newSearchFileTool(reader LineReader) *searchFileTool {
	return &searchFileTool{baseTool: newBaseTool(reader)}
}

func (t *searchFileTool) Name() string {
	return "search_file"
}

func (t *searchFileTool) ToolDefinition() openai.Tool {
	return openai.Tool{
		Type: "function",
		Function: &openai.FunctionDefinition{
			Name:        t.Name(),
			Description: localize.T("tools.description.search_file"),
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path": map[string]any{
						"type":        "string",
						"description": localize.T("tools.param.path"),
					},
					"query": map[string]any{
						"type":        "string",
						"description": localize.T("tools.param.query"),
					},
					"mode": map[string]any{
						"type":        "string",
						"enum":        []string{"auto", "literal", "regex"},
						"description": localize.T("tools.param.mode"),
					},
					"limit": map[string]any{
						"type":        "integer",
						"description": localize.T("tools.param.limit"),
					},
					"offset": map[string]any{
						"type":        "integer",
						"description": localize.T("tools.param.offset"),
					},
					"context_lines": map[string]any{
						"type":        "integer",
						"description": localize.T("tools.param.context_lines"),
					},
				},
				"required": []string{"query"},
			},
		},
	}
}

func (t *searchFileTool) Execute(ctx context.Context, call openai.ToolCall) (aitools.ExecuteResult, error) {
	return t.execute(ctx, t.Name(), call)
}

type readLinesTool struct {
	baseTool
	multiFile bool
}

func newReadLinesTool(reader LineReader, multiFile bool) *readLinesTool {
	return &readLinesTool{
		baseTool:  newBaseTool(reader),
		multiFile: multiFile,
	}
}

func (t *readLinesTool) Name() string {
	return "read_lines"
}

func (t *readLinesTool) ToolDefinition() openai.Tool {
	required := []string{"start_line", "end_line"}
	if t.multiFile {
		required = []string{"path", "start_line", "end_line"}
	}

	return openai.Tool{
		Type: "function",
		Function: &openai.FunctionDefinition{
			Name:        t.Name(),
			Description: localize.T("tools.description.read_lines"),
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path": map[string]any{
						"type":        "string",
						"description": localize.T("tools.param.path"),
					},
					"start_line": map[string]any{
						"type":        "integer",
						"description": localize.T("tools.param.start_line"),
					},
					"end_line": map[string]any{
						"type":        "integer",
						"description": localize.T("tools.param.end_line"),
					},
				},
				"required": required,
			},
		},
	}
}

func (t *readLinesTool) Execute(ctx context.Context, call openai.ToolCall) (aitools.ExecuteResult, error) {
	return t.execute(ctx, t.Name(), call)
}

type readAroundTool struct {
	baseTool
	multiFile bool
}

func newReadAroundTool(reader LineReader, multiFile bool) *readAroundTool {
	return &readAroundTool{
		baseTool:  newBaseTool(reader),
		multiFile: multiFile,
	}
}

func (t *readAroundTool) Name() string {
	return "read_around"
}

func (t *readAroundTool) ToolDefinition() openai.Tool {
	required := []string{"line"}
	if t.multiFile {
		required = []string{"path", "line"}
	}

	return openai.Tool{
		Type: "function",
		Function: &openai.FunctionDefinition{
			Name:        t.Name(),
			Description: localize.T("tools.description.read_around"),
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path": map[string]any{
						"type":        "string",
						"description": localize.T("tools.param.path"),
					},
					"line": map[string]any{
						"type":        "integer",
						"description": localize.T("tools.param.line"),
					},
					"before": map[string]any{
						"type":        "integer",
						"description": localize.T("tools.param.before"),
					},
					"after": map[string]any{
						"type":        "integer",
						"description": localize.T("tools.param.after"),
					},
				},
				"required": required,
			},
		},
	}
}

func (t *readAroundTool) Execute(ctx context.Context, call openai.ToolCall) (aitools.ExecuteResult, error) {
	return t.execute(ctx, t.Name(), call)
}

func progressKeys(toolName, raw string) ([]string, error) {
	keys, err := extractLineKeys(toolName, raw)
	if err != nil {
		return nil, err
	}

	result := make([]string, 0, len(keys))
	for key := range keys {
		result = append(result, key)
	}
	sort.Strings(result)
	return result, nil
}

func extractLineKeys(toolName, raw string) (map[string]struct{}, error) {
	keys := make(map[string]struct{})

	switch toolName {
	case "list_files":
		var result ListFilesResult
		if err := json.Unmarshal([]byte(raw), &result); err != nil {
			return nil, fmt.Errorf("failed to decode list_files result: %w", err)
		}
		for _, path := range result.Files {
			keys["file:"+strings.TrimSpace(path)] = struct{}{}
		}
	case "search_file":
		var result SearchResult
		if err := json.Unmarshal([]byte(raw), &result); err != nil {
			return nil, fmt.Errorf("failed to decode search result: %w", err)
		}
		for _, match := range result.Matches {
			keys[lineKey(match.Path, result.Path, match.LineNumber)] = struct{}{}
			for _, line := range match.ContextBefore {
				keys[lineKey(line.Path, result.Path, line.LineNumber)] = struct{}{}
			}
			for _, line := range match.ContextAfter {
				keys[lineKey(line.Path, result.Path, line.LineNumber)] = struct{}{}
			}
		}
	case "read_lines":
		var result ReadLinesResult
		if err := json.Unmarshal([]byte(raw), &result); err != nil {
			return nil, fmt.Errorf("failed to decode read_lines result: %w", err)
		}
		for _, line := range result.Lines {
			keys[lineKey(line.Path, result.Path, line.LineNumber)] = struct{}{}
		}
	case "read_around":
		var result ReadAroundResult
		if err := json.Unmarshal([]byte(raw), &result); err != nil {
			return nil, fmt.Errorf("failed to decode read_around result: %w", err)
		}
		for _, line := range result.Lines {
			keys[lineKey(line.Path, result.Path, line.LineNumber)] = struct{}{}
		}
	default:
		return nil, fmt.Errorf("unsupported tool for line extraction: %s", toolName)
	}

	return keys, nil
}

func lineKey(path string, fallbackPath string, lineNumber int) string {
	trimmed := strings.TrimSpace(path)
	if trimmed == "" {
		trimmed = strings.TrimSpace(fallbackPath)
	}
	if lineNumber <= 0 {
		return "file:" + trimmed
	}
	return fmt.Sprintf("%s:%d", trimmed, lineNumber)
}

func suggestedNextOffset(toolName, raw string) int {
	switch toolName {
	case "list_files":
		var result ListFilesResult
		if err := json.Unmarshal([]byte(raw), &result); err != nil {
			return 0
		}
		if result.HasMore {
			return result.NextOffset
		}
	case "search_file":
		var result SearchResult
		if err := json.Unmarshal([]byte(raw), &result); err != nil {
			return 0
		}
		if result.HasMore {
			return result.NextOffset
		}
	}
	return 0
}

func toolHints(toolName, raw string) map[string]any {
	if next := suggestedNextOffset(toolName, raw); next > 0 {
		return map[string]any{HintSuggestedNextOffset: next}
	}
	return nil
}

func cloneExecuteResult(result aitools.ExecuteResult) aitools.ExecuteResult {
	return aitools.ExecuteResult{
		Payload:      result.Payload,
		Summary:      cloneSummary(result.Summary),
		Cached:       result.Cached,
		Duplicate:    result.Duplicate,
		ProgressKeys: cloneStrings(result.ProgressKeys),
		Hints:        cloneHints(result.Hints),
	}
}

func cloneSummary(summary map[string]any) map[string]any {
	if summary == nil {
		return nil
	}
	cloned := make(map[string]any, len(summary))
	for key, value := range summary {
		cloned[key] = value
	}
	return cloned
}

func cloneStrings(values []string) []string {
	if values == nil {
		return nil
	}
	return append([]string(nil), values...)
}

func cloneHints(hints map[string]any) map[string]any {
	if hints == nil {
		return nil
	}
	cloned := make(map[string]any, len(hints))
	for key, value := range hints {
		cloned[key] = value
	}
	return cloned
}
