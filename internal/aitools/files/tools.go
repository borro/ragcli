package files

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

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
	cache  map[string]aitools.Execution
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
		cache:  make(map[string]aitools.Execution),
	}
}

func (b baseTool) clone() baseTool {
	cloned := baseTool{
		reader: b.reader,
		cache:  make(map[string]aitools.Execution, len(b.cache)),
	}
	for key, value := range b.cache {
		cloned.cache[key] = aitools.CloneExecution(value)
	}
	return cloned
}

func (b baseTool) DescribeCall(call openai.ToolCall) aitools.CallDescription {
	return aitools.CallDescription{
		Arguments:    aitools.CloneSummary(SummarizeToolArguments(call)),
		VerboseLabel: compactToolCallLabel(call),
	}
}

func canonicalToolSignature(call openai.ToolCall) (string, error) {
	args, err := decodeToolArgs(call)
	if err != nil {
		return "", err
	}
	return args.canonicalSignature(call.Function.Name)
}

func (b *baseTool) execute(ctx context.Context, name string, call openai.ToolCall) (aitools.Execution, error) {
	return aitools.ExecuteCachedTool(ctx, call, aitools.CachedExecutionSpec{
		Name:               name,
		Cache:              b.cache,
		SummarizeArguments: SummarizeToolArguments,
		CanonicalSignature: canonicalToolSignature,
		Execute: func(ctx context.Context, call openai.ToolCall) (aitools.Execution, error) {
			raw, err := executeToolRaw(call, b.reader)
			if err != nil {
				return aitools.Execution{}, err
			}
			result, err := buildToolResult(call.Function.Name, raw)
			if err != nil {
				return aitools.Execution{}, err
			}
			return aitools.Execution{Result: result}, nil
		},
	})
}

func buildToolResult(toolName string, raw string) (aitools.ToolResult, error) {
	payload, err := decodeToolPayload(toolName, raw)
	if err != nil {
		return aitools.ToolResult{}, err
	}
	keys, err := progressKeys(toolName, raw)
	if err != nil {
		return aitools.ToolResult{}, err
	}
	return aitools.ToolResult{
		Payload:      payload,
		Summary:      aitools.CloneSummary(SummarizeToolResult(toolName, raw)),
		ProgressKeys: aitools.CloneStrings(keys),
		Hints:        aitools.CloneHints(toolHints(toolName, raw)),
		Evidence:     evidenceFromToolPayload(toolName, raw),
	}, nil
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

func (t *listFilesTool) Execute(ctx context.Context, call openai.ToolCall) (aitools.Execution, error) {
	return t.execute(ctx, t.Name(), call)
}

func (t *listFilesTool) CloneTool() aitools.Tool {
	if t == nil {
		return (*listFilesTool)(nil)
	}
	return &listFilesTool{baseTool: t.clone()}
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

func (t *searchFileTool) Execute(ctx context.Context, call openai.ToolCall) (aitools.Execution, error) {
	return t.execute(ctx, t.Name(), call)
}

func (t *searchFileTool) CloneTool() aitools.Tool {
	if t == nil {
		return (*searchFileTool)(nil)
	}
	return &searchFileTool{baseTool: t.clone()}
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

func (t *readLinesTool) Execute(ctx context.Context, call openai.ToolCall) (aitools.Execution, error) {
	return t.execute(ctx, t.Name(), call)
}

func (t *readLinesTool) CloneTool() aitools.Tool {
	if t == nil {
		return (*readLinesTool)(nil)
	}
	return &readLinesTool{
		baseTool:  t.clone(),
		multiFile: t.multiFile,
	}
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

func (t *readAroundTool) Execute(ctx context.Context, call openai.ToolCall) (aitools.Execution, error) {
	return t.execute(ctx, t.Name(), call)
}

func (t *readAroundTool) CloneTool() aitools.Tool {
	if t == nil {
		return (*readAroundTool)(nil)
	}
	return &readAroundTool{
		baseTool:  t.clone(),
		multiFile: t.multiFile,
	}
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
			keys["files:file:"+strings.TrimSpace(path)] = struct{}{}
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
		return "files:file:" + trimmed
	}
	return fmt.Sprintf("files:%s:%d", trimmed, lineNumber)
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
		return map[string]any{aitools.HintSuggestedNextOffset: next}
	}
	return nil
}

func decodeToolPayload(toolName string, raw string) (any, error) {
	switch toolName {
	case "list_files":
		var result ListFilesResult
		if err := json.Unmarshal([]byte(raw), &result); err != nil {
			return nil, fmt.Errorf("failed to decode list_files result: %w", err)
		}
		return result, nil
	case "search_file":
		var result SearchResult
		if err := json.Unmarshal([]byte(raw), &result); err != nil {
			return nil, fmt.Errorf("failed to decode search_file result: %w", err)
		}
		return result, nil
	case "read_lines":
		var result ReadLinesResult
		if err := json.Unmarshal([]byte(raw), &result); err != nil {
			return nil, fmt.Errorf("failed to decode read_lines result: %w", err)
		}
		return result, nil
	case "read_around":
		var result ReadAroundResult
		if err := json.Unmarshal([]byte(raw), &result); err != nil {
			return nil, fmt.Errorf("failed to decode read_around result: %w", err)
		}
		return result, nil
	default:
		return nil, fmt.Errorf("unsupported tool payload: %s", toolName)
	}
}

func evidenceFromToolPayload(toolName string, raw string) []aitools.Evidence {
	switch toolName {
	case "search_file":
		var result SearchResult
		if err := json.Unmarshal([]byte(raw), &result); err != nil {
			return nil
		}
		evidence := make([]aitools.Evidence, 0, len(result.Matches))
		for _, match := range result.Matches {
			path := strings.TrimSpace(match.Path)
			if path == "" {
				path = strings.TrimSpace(result.Path)
			}
			if path == "" || match.LineNumber < 1 {
				continue
			}
			evidence = append(evidence, aitools.Evidence{
				Kind:       aitools.EvidenceVerified,
				SourcePath: path,
				StartLine:  match.LineNumber,
				EndLine:    match.LineNumber,
			})
		}
		return evidence
	case "read_lines":
		var result ReadLinesResult
		if err := json.Unmarshal([]byte(raw), &result); err != nil {
			return nil
		}
		path := strings.TrimSpace(result.Path)
		if path == "" || result.StartLine < 1 || result.EndLine < result.StartLine {
			return nil
		}
		return []aitools.Evidence{{
			Kind:       aitools.EvidenceVerified,
			SourcePath: path,
			StartLine:  result.StartLine,
			EndLine:    result.EndLine,
		}}
	case "read_around":
		var result ReadAroundResult
		if err := json.Unmarshal([]byte(raw), &result); err != nil {
			return nil
		}
		path := strings.TrimSpace(result.Path)
		if path == "" || result.StartLine < 1 || result.EndLine < result.StartLine {
			return nil
		}
		return []aitools.Evidence{{
			Kind:       aitools.EvidenceVerified,
			SourcePath: path,
			StartLine:  result.StartLine,
			EndLine:    result.EndLine,
		}}
	default:
		return nil
	}
}
