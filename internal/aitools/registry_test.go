package aitools

import (
	"context"
	"fmt"
	"testing"

	openai "github.com/sashabaranov/go-openai"
)

type fakeTool struct {
	name    string
	payload string
	cache   map[string]ExecuteResult
}

func (t *fakeTool) Name() string {
	return t.name
}

func (t *fakeTool) ToolDefinition() openai.Tool {
	return openai.Tool{
		Type: "function",
		Function: &openai.FunctionDefinition{
			Name: t.name,
		},
	}
}

func (t *fakeTool) DescribeCall(call openai.ToolCall) CallDescription {
	return CallDescription{
		Arguments: map[string]any{
			"raw_arguments": call.Function.Arguments,
		},
		VerboseLabel: fmt.Sprintf("%s(args=%q)", t.name, call.Function.Arguments),
	}
}

func (t *fakeTool) Execute(ctx context.Context, call openai.ToolCall) (ExecuteResult, error) {
	select {
	case <-ctx.Done():
		return ExecuteResult{}, ctx.Err()
	default:
	}

	if call.Function.Name != t.name {
		return ExecuteResult{}, NewToolError("unknown_tool", "unknown tool requested", false, map[string]any{
			"tool": call.Function.Name,
		})
	}

	if t.cache == nil {
		t.cache = make(map[string]ExecuteResult)
	}
	if cached, ok := t.cache[call.Function.Arguments]; ok {
		cached.Cached = true
		cached.Duplicate = true
		return cached, nil
	}

	result := ExecuteResult{
		Payload:      t.payload,
		ProgressKeys: []string{t.name + ":" + call.Function.Arguments},
	}
	t.cache[call.Function.Arguments] = result
	return result, nil
}

func TestRegistry_BuildsDefinitionsInStableOrder(t *testing.T) {
	registry := NewRegistry(
		&fakeTool{name: "search_file"},
		&fakeTool{name: "read_lines"},
		&fakeTool{name: "search_file"},
	)

	definitions := registry.ToolDefinitions()
	if len(definitions) != 2 {
		t.Fatalf("len(definitions) = %d, want 2", len(definitions))
	}
	if definitions[0].Function == nil || definitions[0].Function.Name != "search_file" {
		t.Fatalf("definitions[0] = %#v, want search_file", definitions[0].Function)
	}
	if definitions[1].Function == nil || definitions[1].Function.Name != "read_lines" {
		t.Fatalf("definitions[1] = %#v, want read_lines", definitions[1].Function)
	}
}

func TestRegistry_DispatchesAndPreservesToolLevelCache(t *testing.T) {
	registry := NewRegistry(&fakeTool{name: "search_file", payload: `{"ok":true}`})
	call := openai.ToolCall{
		ID:   "1",
		Type: "function",
		Function: openai.FunctionCall{
			Name:      "search_file",
			Arguments: `{"query":"alpha"}`,
		},
	}

	first, err := registry.Execute(context.Background(), call)
	if err != nil {
		t.Fatalf("registry.Execute(first) error = %v", err)
	}
	second, err := registry.Execute(context.Background(), call)
	if err != nil {
		t.Fatalf("registry.Execute(second) error = %v", err)
	}

	if first.Cached || first.Duplicate {
		t.Fatalf("first result = %+v, want fresh execution", first)
	}
	if !second.Cached || !second.Duplicate {
		t.Fatalf("second result = %+v, want cached duplicate", second)
	}
}

func TestRegistry_UnknownToolReturnsToolError(t *testing.T) {
	registry := NewRegistry(&fakeTool{name: "search_file"})

	_, err := registry.Execute(context.Background(), openai.ToolCall{
		ID:   "1",
		Type: "function",
		Function: openai.FunctionCall{
			Name:      "boom",
			Arguments: `{}`,
		},
	})
	if err == nil {
		t.Fatal("registry.Execute() error = nil, want unknown_tool")
	}
	if got := AsToolError(err).Code; got != "unknown_tool" {
		t.Fatalf("Code = %q, want unknown_tool", got)
	}
}

func TestRegistry_DescribeCallFallsBackForUnknownTool(t *testing.T) {
	registry := NewRegistry(&fakeTool{name: "search_file"})

	desc := registry.DescribeCall(openai.ToolCall{
		ID:   "1",
		Type: "function",
		Function: openai.FunctionCall{
			Name:      "boom",
			Arguments: `{"query":"alpha"}`,
		},
	})

	if desc.VerboseLabel != `boom(args="{\"query\":\"alpha\"}")` {
		t.Fatalf("VerboseLabel = %q, want fallback with raw args", desc.VerboseLabel)
	}
	if got := desc.Arguments["raw_arguments"]; got != `{"query":"alpha"}` {
		t.Fatalf("raw_arguments = %#v, want original raw arguments", got)
	}
}
