package aitools

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"strings"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

type CallDescription struct {
	Arguments    map[string]any
	VerboseLabel string
}

type Tool interface {
	Name() string
	ToolDefinition() openai.Tool
	DescribeCall(call openai.ToolCall) CallDescription
	Execute(ctx context.Context, call openai.ToolCall) (Execution, error)
}

type CloneableTool interface {
	Tool
	CloneTool() Tool
}

type Registry interface {
	ToolDefinitions() []openai.Tool
	DescribeCall(call openai.ToolCall) CallDescription
	Execute(ctx context.Context, call openai.ToolCall) (Execution, error)
}

type ToolError struct {
	Code      string         `json:"code"`
	Message   string         `json:"message"`
	Retryable bool           `json:"retryable"`
	Details   map[string]any `json:"details,omitempty"`
}

type toolRegistry struct {
	ordered []Tool
	byName  map[string]Tool
}

func NewRegistry(tools ...Tool) Registry {
	ordered := make([]Tool, 0, len(tools))
	byName := make(map[string]Tool, len(tools))
	for _, tool := range tools {
		if tool == nil {
			continue
		}
		name := tool.Name()
		if _, exists := byName[name]; exists {
			continue
		}
		ordered = append(ordered, tool)
		byName[name] = tool
	}
	return &toolRegistry{
		ordered: ordered,
		byName:  byName,
	}
}

func (r *toolRegistry) ToolDefinitions() []openai.Tool {
	definitions := make([]openai.Tool, 0, len(r.ordered))
	for _, tool := range r.ordered {
		definitions = append(definitions, tool.ToolDefinition())
	}
	return definitions
}

func (r *toolRegistry) DescribeCall(call openai.ToolCall) CallDescription {
	tool, ok := r.byName[call.Function.Name]
	if !ok {
		return fallbackCallDescription(call)
	}
	desc := tool.DescribeCall(call)
	if strings.TrimSpace(desc.VerboseLabel) == "" {
		desc.VerboseLabel = fallbackCallDescription(call).VerboseLabel
	}
	if desc.Arguments == nil {
		desc.Arguments = fallbackCallDescription(call).Arguments
	}
	return desc
}

func (r *toolRegistry) Execute(ctx context.Context, call openai.ToolCall) (Execution, error) {
	tool, ok := r.byName[call.Function.Name]
	if !ok {
		err := NewToolError("unknown_tool", "unknown tool requested", false, map[string]any{
			"tool": call.Function.Name,
		})
		LogToolCallError(call, 0, fallbackCallDescription(call).Arguments, err)
		return Execution{}, err
	}
	return tool.Execute(ctx, call)
}

func AsToolError(err error) ToolError {
	if toolErr, ok := errors.AsType[ToolError](err); ok {
		return toolErr
	}

	return ToolError{
		Code:      "tool_execution_failed",
		Message:   err.Error(),
		Retryable: false,
	}
}

func NewToolError(code, message string, retryable bool, details map[string]any) error {
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

func LogToolCallStarted(call openai.ToolCall, args map[string]any) {
	logArgs := []any{
		"tool_name", call.Function.Name,
		"tool_call_id", call.ID,
	}
	logArgs = appendSummaryGroup(logArgs, "arguments", args)
	slog.Debug("tool call started", logArgs...)
}

func LogToolCallFinished(call openai.ToolCall, duration time.Duration, status string, summary map[string]any) {
	logArgs := make([]any, 0, 8+2*len(summary))
	logArgs = append(logArgs,
		"tool_name", call.Function.Name,
		"tool_call_id", call.ID,
		"status", status,
		"duration", float64(duration.Round(time.Millisecond))/float64(time.Second),
	)
	for key, value := range summary {
		logArgs = append(logArgs, key, value)
	}
	slog.Debug("tool call finished", logArgs...)
}

func LogToolCallError(call openai.ToolCall, duration time.Duration, args map[string]any, err error) {
	toolErr := AsToolError(err)
	logArgs := []any{
		"tool_name", call.Function.Name,
		"tool_call_id", call.ID,
		"duration", float64(duration.Round(time.Millisecond)) / float64(time.Second),
		"error_code", toolErr.Code,
		"retryable", toolErr.Retryable,
		"error", toolErr.Message,
	}
	logArgs = appendSummaryGroup(logArgs, "arguments", args)
	if len(toolErr.Details) > 0 {
		logArgs = append(logArgs, "details", toolErr.Details)
	}
	slog.Debug("tool call failed", logArgs...)
}

func fallbackCallDescription(call openai.ToolCall) CallDescription {
	label := strings.TrimSpace(call.Function.Name)
	if label == "" {
		label = "tool"
	}

	raw := strings.TrimSpace(call.Function.Arguments)
	if raw == "" {
		return CallDescription{VerboseLabel: label}
	}

	return CallDescription{
		Arguments: map[string]any{
			"raw_arguments": raw,
		},
		VerboseLabel: labelWithRawArguments(label, raw),
	}
}

func labelWithRawArguments(label string, raw string) string {
	trimmedLabel := strings.TrimSpace(label)
	if trimmedLabel == "" {
		trimmedLabel = "tool"
	}
	trimmedRaw := strings.TrimSpace(raw)
	if trimmedRaw == "" {
		return trimmedLabel
	}
	if len(trimmedRaw) > 80 {
		trimmedRaw = trimmedRaw[:77] + "..."
	}
	return fmt.Sprintf("%s(args=%q)", trimmedLabel, trimmedRaw)
}

func appendSummaryGroup(logArgs []any, key string, summary map[string]any) []any {
	if len(summary) == 0 {
		return logArgs
	}

	keys := make([]string, 0, len(summary))
	for name := range summary {
		keys = append(keys, name)
	}
	sort.Strings(keys)

	groupArgs := make([]any, 0, len(keys))
	for _, name := range keys {
		groupArgs = append(groupArgs, slog.Any(name, summary[name]))
	}

	return append(logArgs, slog.Group(key, groupArgs...))
}
