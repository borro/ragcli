package aitools

import (
	"context"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

type cacheBypassKey struct{}

type CachedExecutionSpec struct {
	Name               string
	Cache              map[string]Execution
	SummarizeArguments func(openai.ToolCall) map[string]any
	CanonicalSignature func(openai.ToolCall) (string, error)
	Execute            func(context.Context, openai.ToolCall) (Execution, error)
}

func WithoutToolCache(ctx context.Context) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	return context.WithValue(ctx, cacheBypassKey{}, true)
}

func ExecuteCachedTool(ctx context.Context, call openai.ToolCall, spec CachedExecutionSpec) (Execution, error) {
	select {
	case <-ctx.Done():
		return Execution{}, ctx.Err()
	default:
	}

	args := maybeSummarizeArguments(spec.SummarizeArguments, call)

	if call.Function.Name != spec.Name {
		err := NewToolError("unknown_tool", "unknown tool requested", false, map[string]any{
			"tool": call.Function.Name,
		})
		LogToolCallError(call, 0, args, err)
		return Execution{}, err
	}

	LogToolCallStarted(call, args)

	signature, err := spec.CanonicalSignature(call)
	if err != nil {
		LogToolCallError(call, 0, args, err)
		return Execution{}, err
	}

	bypassCache := ctx != nil && ctx.Value(cacheBypassKey{}) == true
	if !bypassCache {
		if cached, ok := spec.Cache[signature]; ok {
			result := CloneExecution(cached)
			result.Cached = true
			result.Duplicate = true

			summary := CloneSummary(result.Result.Summary)
			if summary == nil {
				summary = map[string]any{}
			}
			summary["cached"] = true
			summary["duplicate"] = true
			LogToolCallFinished(call, 0, "cached", summary)
			return result, nil
		}
	}

	startedAt := time.Now()
	result, err := spec.Execute(ctx, call)
	if err != nil {
		LogToolCallError(call, time.Since(startedAt), args, err)
		return Execution{}, err
	}

	if !bypassCache {
		spec.Cache[signature] = CloneExecution(result)
	}
	LogToolCallFinished(call, time.Since(startedAt), "ok", CloneSummary(result.Result.Summary))
	return CloneExecution(result), nil
}

func CloneSummary(summary map[string]any) map[string]any {
	if summary == nil {
		return nil
	}
	cloned := make(map[string]any, len(summary))
	for key, value := range summary {
		cloned[key] = value
	}
	return cloned
}

func CloneStrings(values []string) []string {
	if values == nil {
		return nil
	}
	return append([]string(nil), values...)
}

func CloneHints(hints map[string]any) map[string]any {
	if hints == nil {
		return nil
	}
	cloned := make(map[string]any, len(hints))
	for key, value := range hints {
		cloned[key] = value
	}
	return cloned
}

func maybeSummarizeArguments(fn func(openai.ToolCall) map[string]any, call openai.ToolCall) map[string]any {
	if fn == nil {
		return nil
	}
	return fn(call)
}
