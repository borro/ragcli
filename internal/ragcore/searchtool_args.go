package ragcore

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/borro/ragcli/internal/aitools"
	openai "github.com/sashabaranov/go-openai"
)

type searchToolArgs struct {
	SearchParams
}

func parseSearchToolArgs(call openai.ToolCall) (searchToolArgs, error) {
	var params SearchParams
	if err := json.Unmarshal([]byte(call.Function.Arguments), &params); err != nil {
		return searchToolArgs{}, aitools.NewToolError("invalid_arguments", "invalid arguments for search_rag", false, map[string]any{
			"tool":  "search_rag",
			"error": err.Error(),
		})
	}
	return searchToolArgs{SearchParams: NormalizeSearchParams(params)}, nil
}

func (a searchToolArgs) canonicalSignature(toolName string) (string, error) {
	body, err := MarshalJSON(a.SearchParams)
	if err != nil {
		return "", err
	}
	return toolName + ":" + body, nil
}

func (a searchToolArgs) summary() map[string]any {
	summary := map[string]any{
		"query":  a.Query,
		"limit":  a.Limit,
		"offset": a.Offset,
	}
	if a.Path != "" {
		summary["path"] = a.Path
	}
	return summary
}

func (a searchToolArgs) compactLabel(toolName string) string {
	parts := make([]string, 0, 4)
	if a.Query != "" {
		parts = append(parts, fmt.Sprintf("query=%q", a.Query))
	}
	if a.Path != "" {
		parts = append(parts, fmt.Sprintf("path=%q", a.Path))
	}
	if a.Limit != defaultSearchLimit {
		parts = append(parts, fmt.Sprintf("limit=%d", a.Limit))
	}
	if a.Offset != 0 {
		parts = append(parts, fmt.Sprintf("offset=%d", a.Offset))
	}
	if len(parts) == 0 {
		return toolName
	}
	return fmt.Sprintf("%s(%s)", toolName, strings.Join(parts, ", "))
}

func compactRawToolCallLabel(toolName string, raw string) string {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return toolName
	}
	if len(trimmed) > 80 {
		trimmed = trimmed[:77] + "..."
	}
	return fmt.Sprintf("%s(args=%q)", toolName, trimmed)
}
