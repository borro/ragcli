package aitools

import (
	"context"
	"errors"
	"log/slog"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

type Tool interface {
	Name() string
	ToolDefinition() openai.Tool
	Execute(ctx context.Context, call openai.ToolCall) (ExecuteResult, error)
}

type ExecuteResult struct {
	Payload      string
	Summary      map[string]any
	Cached       bool
	Duplicate    bool
	ProgressKeys []string
	Hints        map[string]any
}

type Registry interface {
	ToolDefinitions() []openai.Tool
	Execute(ctx context.Context, call openai.ToolCall) (ExecuteResult, error)
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

func (r *toolRegistry) Execute(ctx context.Context, call openai.ToolCall) (ExecuteResult, error) {
	tool, ok := r.byName[call.Function.Name]
	if !ok {
		return ExecuteResult{}, NewToolError("unknown_tool", "unknown tool requested", false, map[string]any{
			"tool": call.Function.Name,
		})
	}
	return tool.Execute(ctx, call)
}

func AsToolError(err error) ToolError {
	var toolErr ToolError
	if errors.As(err, &toolErr) {
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
	if args != nil {
		logArgs = append(logArgs, "arguments", args)
	}
	slog.Debug("tool call started", logArgs...)
}

func LogToolCallFinished(call openai.ToolCall, duration time.Duration, status string, summary map[string]any) {
	logArgs := []any{
		"tool_name", call.Function.Name,
		"tool_call_id", call.ID,
		"status", status,
		"duration", float64(duration.Round(time.Millisecond)) / float64(time.Second),
	}
	for key, value := range summary {
		logArgs = append(logArgs, key, value)
	}
	slog.Debug("tool call finished", logArgs...)
}

func LogToolCallError(call openai.ToolCall, duration time.Duration, err error) {
	toolErr := AsToolError(err)
	logArgs := []any{
		"tool_name", call.Function.Name,
		"tool_call_id", call.ID,
		"duration", float64(duration.Round(time.Millisecond)) / float64(time.Second),
		"error_code", toolErr.Code,
		"retryable", toolErr.Retryable,
		"error", toolErr.Message,
	}
	if len(toolErr.Details) > 0 {
		logArgs = append(logArgs, "details", toolErr.Details)
	}
	slog.Debug("tool call failed", logArgs...)
}
