package ragtools

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/borro/ragcli/internal/aitools"
	"github.com/borro/ragcli/internal/llm"
	"github.com/borro/ragcli/internal/localize"
	ragruntime "github.com/borro/ragcli/internal/rag"
	openai "github.com/sashabaranov/go-openai"
)

const (
	defaultSearchLimit      = 5
	maxSearchLimit          = 20
	HintSuggestedNextOffset = "suggested_next_offset"
)

type SearchParams struct {
	Path   string
	Query  string
	Limit  int
	Offset int
}

type SearchMatch struct {
	Path      string  `json:"path,omitempty"`
	StartLine int     `json:"start_line"`
	EndLine   int     `json:"end_line"`
	Score     float64 `json:"score"`
	Text      string  `json:"text"`
}

type SearchResult struct {
	Query        string        `json:"query"`
	Path         string        `json:"path,omitempty"`
	Limit        int           `json:"limit"`
	Offset       int           `json:"offset"`
	MatchCount   int           `json:"match_count"`
	TotalMatches int           `json:"total_matches"`
	HasMore      bool          `json:"has_more"`
	NextOffset   int           `json:"next_offset,omitempty"`
	Matches      []SearchMatch `json:"matches"`
	Error        string        `json:"error,omitempty"`
}

type searchRAGTool struct {
	searcher *ragruntime.PreparedSearch
	embedder llm.EmbeddingRequester
	cache    map[string]aitools.ExecuteResult
}

func NewTool(searcher *ragruntime.PreparedSearch, embedder llm.EmbeddingRequester) aitools.Tool {
	return &searchRAGTool{
		searcher: searcher,
		embedder: embedder,
		cache:    make(map[string]aitools.ExecuteResult),
	}
}

func NormalizeSearchParams(params SearchParams) SearchParams {
	params.Path = strings.TrimSpace(params.Path)
	params.Query = strings.TrimSpace(params.Query)
	if params.Limit <= 0 {
		params.Limit = defaultSearchLimit
	}
	if params.Limit > maxSearchLimit {
		params.Limit = maxSearchLimit
	}
	if params.Offset < 0 {
		params.Offset = 0
	}
	return params
}

func MarshalJSON(v any) (string, error) {
	data, err := json.Marshal(v)
	if err != nil {
		return "", fmt.Errorf("failed to marshal tool result: %w", err)
	}
	return string(data), nil
}

func (t *searchRAGTool) Name() string {
	return "search_rag"
}

func (t *searchRAGTool) ToolDefinition() openai.Tool {
	return openai.Tool{
		Type: "function",
		Function: &openai.FunctionDefinition{
			Name:        t.Name(),
			Description: localize.T("tools.description.search_rag"),
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path": map[string]any{
						"type":        "string",
						"description": localize.T("tools.rag.param.path"),
					},
					"query": map[string]any{
						"type":        "string",
						"description": localize.T("tools.rag.param.query"),
					},
					"limit": map[string]any{
						"type":        "integer",
						"description": localize.T("tools.rag.param.limit"),
					},
					"offset": map[string]any{
						"type":        "integer",
						"description": localize.T("tools.rag.param.offset"),
					},
				},
				"required": []string{"query"},
			},
		},
	}
}

func (t *searchRAGTool) Execute(ctx context.Context, call openai.ToolCall) (aitools.ExecuteResult, error) {
	var params SearchParams
	return aitools.ExecuteCachedTool(ctx, call, aitools.CachedExecutionSpec{
		Name:               t.Name(),
		Cache:              t.cache,
		SummarizeArguments: SummarizeToolArguments,
		CanonicalSignature: func(call openai.ToolCall) (string, error) {
			signature, normalized, err := canonicalToolSignature(call)
			if err == nil {
				params = normalized
			}
			return signature, err
		},
		Execute: func(ctx context.Context, call openai.ToolCall) (aitools.ExecuteResult, error) {
			raw, err := t.search(ctx, params)
			if err != nil {
				return aitools.ExecuteResult{}, err
			}
			return aitools.ExecuteResult{
				Payload:      raw,
				Summary:      aitools.CloneSummary(SummarizeToolResult(raw)),
				ProgressKeys: progressKeys(raw),
				Hints:        toolHints(raw),
			}, nil
		},
	})
}

func canonicalToolSignature(call openai.ToolCall) (string, SearchParams, error) {
	var params struct {
		Path   string `json:"path"`
		Query  string `json:"query"`
		Limit  int    `json:"limit"`
		Offset int    `json:"offset"`
	}
	if err := json.Unmarshal([]byte(call.Function.Arguments), &params); err != nil {
		return "", SearchParams{}, aitools.NewToolError("invalid_arguments", "invalid arguments for search_rag", false, map[string]any{
			"tool":  "search_rag",
			"error": err.Error(),
		})
	}

	normalized := NormalizeSearchParams(SearchParams{
		Path:   params.Path,
		Query:  params.Query,
		Limit:  params.Limit,
		Offset: params.Offset,
	})
	body, err := MarshalJSON(normalized)
	if err != nil {
		return "", SearchParams{}, err
	}
	return call.Function.Name + ":" + body, normalized, nil
}

func (t *searchRAGTool) search(ctx context.Context, params SearchParams) (string, error) {
	if t.searcher == nil || t.searcher.Index == nil {
		return "", aitools.NewToolError("tool_not_ready", "rag index is not prepared", false, map[string]any{
			"tool": "search_rag",
		})
	}
	if t.embedder == nil {
		return "", aitools.NewToolError("tool_not_ready", "embedding client is not configured", false, map[string]any{
			"tool": "search_rag",
		})
	}
	if params.Query == "" {
		return "", aitools.NewToolError("invalid_arguments", "query must not be empty", false, map[string]any{
			"tool":  "search_rag",
			"field": "query",
		})
	}
	if params.Path != "" && !pathExists(t.searcher.Index, params.Path) {
		return "", aitools.NewToolError("invalid_arguments", "path is not part of the attached corpus", false, map[string]any{
			"tool":  "search_rag",
			"field": "path",
			"value": params.Path,
		})
	}

	hits, _, err := t.searcher.Search(ctx, t.embedder, params.Query, params.Path)
	if err != nil {
		return "", fmt.Errorf("failed to search rag index: %w", err)
	}

	totalMatches := len(hits)
	if params.Offset > totalMatches {
		params.Offset = totalMatches
	}
	end := min(params.Offset+params.Limit, totalMatches)
	page := hits[params.Offset:end]

	result := SearchResult{
		Query:        params.Query,
		Path:         params.Path,
		Limit:        params.Limit,
		Offset:       params.Offset,
		MatchCount:   len(page),
		TotalMatches: totalMatches,
		HasMore:      end < totalMatches,
		Matches:      make([]SearchMatch, 0, len(page)),
	}
	if result.HasMore {
		result.NextOffset = end
	}
	for _, hit := range page {
		result.Matches = append(result.Matches, SearchMatch{
			Path:      hit.Chunk.SourcePath,
			StartLine: hit.Chunk.StartLine,
			EndLine:   hit.Chunk.EndLine,
			Score:     hit.Score,
			Text:      hit.Chunk.Text,
		})
	}

	return MarshalJSON(result)
}

func pathExists(index *ragruntime.Index, path string) bool {
	target := strings.TrimSpace(path)
	for _, chunk := range index.Chunks {
		if chunk.SourcePath == target {
			return true
		}
	}
	return false
}

func SummarizeToolArguments(call openai.ToolCall) map[string]any {
	if strings.TrimSpace(call.Function.Arguments) == "" {
		return nil
	}

	var params struct {
		Path   string `json:"path"`
		Query  string `json:"query"`
		Limit  int    `json:"limit"`
		Offset int    `json:"offset"`
	}
	if err := json.Unmarshal([]byte(call.Function.Arguments), &params); err != nil {
		return map[string]any{"decode_error": err.Error()}
	}
	normalized := NormalizeSearchParams(SearchParams{
		Path:   params.Path,
		Query:  params.Query,
		Limit:  params.Limit,
		Offset: params.Offset,
	})

	summary := map[string]any{
		"query":  normalized.Query,
		"limit":  normalized.Limit,
		"offset": normalized.Offset,
	}
	if normalized.Path != "" {
		summary["path"] = normalized.Path
	}
	return summary
}

func SummarizeToolResult(raw string) map[string]any {
	var result SearchResult
	if err := json.Unmarshal([]byte(raw), &result); err != nil {
		return map[string]any{"decode_error": err.Error()}
	}

	summary := map[string]any{
		"match_count":   result.MatchCount,
		"total_matches": result.TotalMatches,
		"has_more":      result.HasMore,
	}
	if result.Path != "" {
		summary["path"] = result.Path
	}
	if result.NextOffset > 0 {
		summary["next_offset"] = result.NextOffset
	}
	return summary
}

func progressKeys(raw string) []string {
	var result SearchResult
	if err := json.Unmarshal([]byte(raw), &result); err != nil {
		return nil
	}

	keys := make([]string, 0, len(result.Matches)*4)
	for _, match := range result.Matches {
		path := strings.TrimSpace(match.Path)
		if path == "" {
			path = strings.TrimSpace(result.Path)
		}
		for line := match.StartLine; line <= match.EndLine; line++ {
			keys = append(keys, fmt.Sprintf("%s:%d", path, line))
		}
	}
	return keys
}

func toolHints(raw string) map[string]any {
	var result SearchResult
	if err := json.Unmarshal([]byte(raw), &result); err != nil {
		return nil
	}
	if result.HasMore {
		return map[string]any{HintSuggestedNextOffset: result.NextOffset}
	}
	return nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
