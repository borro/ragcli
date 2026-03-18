package seed

import (
	"context"
	"fmt"
	"strings"

	"github.com/borro/ragcli/internal/aitools"
	ragtools "github.com/borro/ragcli/internal/aitools/rag"
	"github.com/borro/ragcli/internal/ragcore"
	openai "github.com/sashabaranov/go-openai"
)

const (
	maxSeedSearchLimit        = 20
	maxSeedVerificationReads  = 4
	maxVerifiedReadLines      = 40
	verifiedReadAroundContext = 20
)

const (
	seedSearchToolCallIDPrefix = "hybrid-seed-search-rag"
	seedReadToolCallIDPrefix   = "hybrid-seed-read"
)

type SyntheticToolExecutor interface {
	ExecuteSyntheticToolCall(ctx context.Context, call openai.ToolCall) (string, error)
}

type SeedBundle struct {
	History []openai.ChatCompletionMessage
	Hits    []ragcore.FusedSeedHit
	Weak    bool
}

type syntheticToolResult struct {
	call   openai.ToolCall
	result string
}

func BuildSeedBundle(ctx context.Context, executor SyntheticToolExecutor, prompt string, searchOpts ragcore.SearchOptions) (SeedBundle, error) {
	queries := ragcore.SeedQueries(prompt)
	searchResults, fused, err := executeSeedSearches(ctx, executor, queries, searchOpts)
	if err != nil {
		return SeedBundle{}, err
	}

	verificationResults, err := executeSeedVerificationReads(ctx, executor, fused)
	if err != nil {
		return SeedBundle{}, err
	}

	history := make([]openai.ChatCompletionMessage, 0, len(searchResults)+len(verificationResults)+2)
	history = append(history, syntheticHistory(searchResults)...)
	history = append(history, syntheticHistory(verificationResults)...)

	return SeedBundle{
		History: history,
		Hits:    fused,
		Weak:    ragcore.FusedSearchTooWeak(fused),
	}, nil
}

func executeSeedSearches(ctx context.Context, executor SyntheticToolExecutor, queries []string, searchOpts ragcore.SearchOptions) ([]syntheticToolResult, []ragcore.FusedSeedHit, error) {
	searchLimit := min(searchOpts.TopK, maxSeedSearchLimit)
	results := make([]syntheticToolResult, 0, len(queries))
	matches := make([]ragcore.FusedSeedMatch, 0, len(queries)*searchLimit)

	for index, query := range queries {
		call, err := seedSearchToolCall(query, searchLimit, index+1)
		if err != nil {
			return nil, nil, err
		}
		result, err := executor.ExecuteSyntheticToolCall(ctx, call)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to execute hybrid seed search: %w", err)
		}
		results = append(results, syntheticToolResult{call: call, result: result})

		var payload ragtools.SearchResult
		if _, err := aitools.DecodeToolResult(result, &payload); err != nil {
			return nil, nil, fmt.Errorf("failed to decode hybrid seed search result: %w", err)
		}
		for rank, match := range payload.Matches {
			path := strings.TrimSpace(match.Path)
			if path == "" {
				path = strings.TrimSpace(payload.Path)
			}
			if path == "" {
				continue
			}
			matches = append(matches, ragcore.FusedSeedMatch{
				Path:       path,
				ChunkID:    match.ChunkID,
				StartLine:  match.StartLine,
				EndLine:    match.EndLine,
				Text:       match.Text,
				Score:      match.Score,
				Similarity: match.Similarity,
				Overlap:    match.Overlap,
				Rank:       rank,
			})
		}
	}

	return results, ragcore.FuseSeedMatches(matches, searchOpts.TopK), nil
}

func executeSeedVerificationReads(ctx context.Context, executor SyntheticToolExecutor, hits []ragcore.FusedSeedHit) ([]syntheticToolResult, error) {
	selected := selectVerificationHits(hits, maxSeedVerificationReads)
	results := make([]syntheticToolResult, 0, len(selected))
	for index, hit := range selected {
		call, err := verificationToolCall(hit, index+1)
		if err != nil {
			return nil, err
		}
		result, err := executor.ExecuteSyntheticToolCall(ctx, call)
		if err != nil {
			return nil, fmt.Errorf("failed to execute hybrid seed verification read: %w", err)
		}
		results = append(results, syntheticToolResult{call: call, result: result})
	}
	return results, nil
}

func selectVerificationHits(hits []ragcore.FusedSeedHit, limit int) []ragcore.FusedSeedHit {
	if len(hits) == 0 || limit <= 0 {
		return nil
	}

	selected := make([]ragcore.FusedSeedHit, 0, min(limit, len(hits)))
	selectedKeys := make(map[string]struct{}, limit)
	usedPaths := make(map[string]struct{}, limit)

	appendHit := func(hit ragcore.FusedSeedHit) bool {
		key := fmt.Sprintf("%s:%d", hit.Path, hit.ChunkID)
		if _, ok := selectedKeys[key]; ok {
			return false
		}
		if overlapsSelected(hit, selected) {
			return false
		}
		selected = append(selected, hit)
		selectedKeys[key] = struct{}{}
		return true
	}

	for _, hit := range hits {
		if len(selected) >= limit {
			return selected
		}
		if _, used := usedPaths[hit.Path]; used {
			continue
		}
		if appendHit(hit) {
			usedPaths[hit.Path] = struct{}{}
		}
	}

	for _, hit := range hits {
		if len(selected) >= limit {
			break
		}
		appendHit(hit)
	}

	return selected
}

func overlapsSelected(hit ragcore.FusedSeedHit, selected []ragcore.FusedSeedHit) bool {
	for _, item := range selected {
		if item.Path != hit.Path {
			continue
		}
		if hit.StartLine <= item.EndLine && item.StartLine <= hit.EndLine {
			return true
		}
	}
	return false
}

func verificationToolCall(hit ragcore.FusedSeedHit, index int) (openai.ToolCall, error) {
	path := strings.TrimSpace(hit.Path)
	if path == "" {
		return openai.ToolCall{}, fmt.Errorf("hybrid verification hit path is required")
	}

	rangeLen := hit.EndLine - hit.StartLine + 1
	call := openai.ToolCall{
		ID:   fmt.Sprintf("%s-%d", seedReadToolCallIDPrefix, index),
		Type: "function",
	}
	if rangeLen <= maxVerifiedReadLines {
		arguments, err := aitools.MarshalJSON(map[string]any{
			"path":       path,
			"start_line": hit.StartLine,
			"end_line":   hit.EndLine,
		})
		if err != nil {
			return openai.ToolCall{}, fmt.Errorf("failed to encode hybrid seed read_lines arguments: %w", err)
		}
		call.Function = openai.FunctionCall{
			Name:      "read_lines",
			Arguments: arguments,
		}
		return call, nil
	}

	midpoint := hit.StartLine + (rangeLen / 2)
	arguments, err := aitools.MarshalJSON(map[string]any{
		"path":   path,
		"line":   midpoint,
		"before": verifiedReadAroundContext,
		"after":  verifiedReadAroundContext,
	})
	if err != nil {
		return openai.ToolCall{}, fmt.Errorf("failed to encode hybrid seed read_around arguments: %w", err)
	}
	call.Function = openai.FunctionCall{
		Name:      "read_around",
		Arguments: arguments,
	}
	return call, nil
}

func syntheticHistory(results []syntheticToolResult) []openai.ChatCompletionMessage {
	if len(results) == 0 {
		return nil
	}

	toolCalls := make([]openai.ToolCall, 0, len(results))
	history := make([]openai.ChatCompletionMessage, 0, len(results)+1)
	for _, result := range results {
		toolCalls = append(toolCalls, result.call)
	}
	history = append(history, openai.ChatCompletionMessage{
		Role:      openai.ChatMessageRoleAssistant,
		ToolCalls: toolCalls,
	})
	for _, result := range results {
		history = append(history, openai.ChatCompletionMessage{
			Role:       openai.ChatMessageRoleTool,
			Name:       result.call.Function.Name,
			Content:    result.result,
			ToolCallID: result.call.ID,
		})
	}
	return history
}

func seedSearchToolCall(query string, limit int, index int) (openai.ToolCall, error) {
	arguments, err := aitools.MarshalJSON(map[string]any{
		"query": query,
		"limit": limit,
	})
	if err != nil {
		return openai.ToolCall{}, fmt.Errorf("failed to encode hybrid seed search arguments: %w", err)
	}

	return openai.ToolCall{
		ID:   fmt.Sprintf("%s-%d", seedSearchToolCallIDPrefix, index),
		Type: "function",
		Function: openai.FunctionCall{
			Name:      "search_rag",
			Arguments: arguments,
		},
	}, nil
}
