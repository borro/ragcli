package files

import (
	"encoding/json"
	"fmt"
	"strings"

	openai "github.com/sashabaranov/go-openai"
)

type toolArgs interface {
	canonicalSignature(toolName string) (string, error)
	execute(reader LineReader) (string, error)
	summary() map[string]any
	compactLabel(toolName string) string
}

type listFilesArgs struct {
	ListFilesParams
}

type searchFileArgs struct {
	SearchParams
}

type readLinesArgs struct {
	Path      string `json:"path,omitempty"`
	StartLine int    `json:"start_line"`
	EndLine   int    `json:"end_line"`
}

type readAroundArgs struct {
	Path   string `json:"path,omitempty"`
	Line   int    `json:"line"`
	Before int    `json:"before"`
	After  int    `json:"after"`
}

func decodeToolArgs(call openai.ToolCall) (toolArgs, error) {
	switch call.Function.Name {
	case "list_files":
		params, err := decodeJSONArguments[ListFilesParams](call, true)
		if err != nil {
			return nil, err
		}
		return listFilesArgs{ListFilesParams: NormalizeListFilesParams(params)}, nil
	case "search_file":
		params, err := decodeJSONArguments[SearchParams](call, false)
		if err != nil {
			return nil, err
		}
		return searchFileArgs{SearchParams: NormalizeSearchParams(params)}, nil
	case "read_lines":
		if err := ensureRequiredArguments(call, "start_line", "end_line"); err != nil {
			return nil, err
		}
		params, err := decodeJSONArguments[readLinesArgs](call, false)
		if err != nil {
			return nil, err
		}
		params.Path = strings.TrimSpace(params.Path)
		params.StartLine = max(params.StartLine, 1)
		return params, nil
	case "read_around":
		params, err := decodeJSONArguments[readAroundArgs](call, false)
		if err != nil {
			return nil, err
		}
		params.Path = strings.TrimSpace(params.Path)
		if params.Before < 0 {
			params.Before = DefaultReadAroundBefore
		}
		if params.After < 0 {
			params.After = DefaultReadAroundAfter
		}
		return params, nil
	default:
		return nil, NewToolError("unknown_tool", "unknown tool requested", false, map[string]any{
			"tool": call.Function.Name,
		})
	}
}

func ensureRequiredArguments(call openai.ToolCall, required ...string) error {
	raw := strings.TrimSpace(call.Function.Arguments)
	if raw == "" {
		return NewToolError("invalid_arguments", fmt.Sprintf("invalid arguments for %s", call.Function.Name), false, map[string]any{
			"tool": call.Function.Name,
		})
	}

	var payload map[string]json.RawMessage
	if err := json.Unmarshal([]byte(raw), &payload); err != nil {
		return NewToolError("invalid_arguments", fmt.Sprintf("invalid arguments for %s", call.Function.Name), false, map[string]any{
			"tool":  call.Function.Name,
			"error": err.Error(),
		})
	}
	for _, field := range required {
		if _, ok := payload[field]; ok {
			continue
		}
		return NewToolError("invalid_arguments", fmt.Sprintf("missing required argument %s for %s", field, call.Function.Name), false, map[string]any{
			"tool":  call.Function.Name,
			"field": field,
		})
	}
	return nil
}

func decodeJSONArguments[T any](call openai.ToolCall, allowEmpty bool) (T, error) {
	var params T
	raw := strings.TrimSpace(call.Function.Arguments)
	if raw == "" && allowEmpty {
		return params, nil
	}
	if err := json.Unmarshal([]byte(call.Function.Arguments), &params); err != nil {
		return params, NewToolError("invalid_arguments", fmt.Sprintf("invalid arguments for %s", call.Function.Name), false, map[string]any{
			"tool":  call.Function.Name,
			"error": err.Error(),
		})
	}
	return params, nil
}

func (a listFilesArgs) canonicalSignature(toolName string) (string, error) {
	body, err := MarshalJSON(a.ListFilesParams)
	if err != nil {
		return "", err
	}
	return toolName + ":" + body, nil
}

func (a listFilesArgs) execute(reader LineReader) (string, error) {
	return reader.ListFiles(a.ListFilesParams)
}

func (a listFilesArgs) summary() map[string]any {
	return map[string]any{
		"limit":  a.Limit,
		"offset": a.Offset,
	}
}

func (a listFilesArgs) compactLabel(toolName string) string {
	parts := make([]string, 0, 2)
	if a.Limit != defaultListFilesLimit {
		parts = append(parts, formatIntToolParam("limit", a.Limit))
	}
	if a.Offset != 0 {
		parts = append(parts, formatIntToolParam("offset", a.Offset))
	}
	return formatToolCallLabel(toolName, parts)
}

func (a searchFileArgs) canonicalSignature(toolName string) (string, error) {
	body, err := MarshalJSON(a.SearchParams)
	if err != nil {
		return "", err
	}
	return toolName + ":" + body, nil
}

func (a searchFileArgs) execute(reader LineReader) (string, error) {
	return reader.Search(a.SearchParams)
}

func (a searchFileArgs) summary() map[string]any {
	summary := map[string]any{
		"query":         a.Query,
		"mode":          a.Mode,
		"limit":         a.Limit,
		"offset":        a.Offset,
		"context_lines": a.ContextLines,
	}
	if a.Path != "" {
		summary["path"] = a.Path
	}
	return summary
}

func (a searchFileArgs) compactLabel(toolName string) string {
	parts := make([]string, 0, 6)
	if a.Query != "" {
		parts = append(parts, formatStringToolParam("query", a.Query))
	}
	if a.Path != "" {
		parts = append(parts, formatStringToolParam("path", a.Path))
	}
	if a.Mode != "auto" {
		parts = append(parts, formatStringToolParam("mode", a.Mode))
	}
	if a.Limit != defaultSearchLimit {
		parts = append(parts, formatIntToolParam("limit", a.Limit))
	}
	if a.Offset != 0 {
		parts = append(parts, formatIntToolParam("offset", a.Offset))
	}
	if a.ContextLines != 0 {
		parts = append(parts, formatIntToolParam("context_lines", a.ContextLines))
	}
	return formatToolCallLabel(toolName, parts)
}

func (a readLinesArgs) canonicalSignature(toolName string) (string, error) {
	body, err := MarshalJSON(a)
	if err != nil {
		return "", err
	}
	return toolName + ":" + body, nil
}

func (a readLinesArgs) execute(reader LineReader) (string, error) {
	return reader.ReadLines(a.Path, a.StartLine, a.EndLine)
}

func (a readLinesArgs) summary() map[string]any {
	summary := map[string]any{
		"start_line": a.StartLine,
		"end_line":   a.EndLine,
	}
	if a.Path != "" {
		summary["path"] = a.Path
	}
	return summary
}

func (a readLinesArgs) compactLabel(toolName string) string {
	parts := make([]string, 0, 3)
	if a.Path != "" {
		parts = append(parts, formatStringToolParam("path", a.Path))
	}
	parts = append(parts, formatIntToolParam("start_line", a.StartLine))
	if a.EndLine != 0 {
		parts = append(parts, formatIntToolParam("end_line", a.EndLine))
	}
	return formatToolCallLabel(toolName, parts)
}

func (a readAroundArgs) canonicalSignature(toolName string) (string, error) {
	body, err := MarshalJSON(a)
	if err != nil {
		return "", err
	}
	return toolName + ":" + body, nil
}

func (a readAroundArgs) execute(reader LineReader) (string, error) {
	return reader.ReadAround(a.Path, a.Line, a.Before, a.After)
}

func (a readAroundArgs) summary() map[string]any {
	summary := map[string]any{
		"line":   a.Line,
		"before": a.Before,
		"after":  a.After,
	}
	if a.Path != "" {
		summary["path"] = a.Path
	}
	return summary
}

func (a readAroundArgs) compactLabel(toolName string) string {
	parts := make([]string, 0, 4)
	if a.Path != "" {
		parts = append(parts, formatStringToolParam("path", a.Path))
	}
	if a.Line != 0 {
		parts = append(parts, formatIntToolParam("line", a.Line))
	}
	if a.Before != 0 && a.Before != DefaultReadAroundBefore {
		parts = append(parts, formatIntToolParam("before", a.Before))
	}
	if a.After != 0 && a.After != DefaultReadAroundAfter {
		parts = append(parts, formatIntToolParam("after", a.After))
	}
	return formatToolCallLabel(toolName, parts)
}
