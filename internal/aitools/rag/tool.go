package rag

import (
	"context"
	"strconv"
	"strings"

	"github.com/borro/ragcli/internal/aitools"
	"github.com/borro/ragcli/internal/llm"
	"github.com/borro/ragcli/internal/localize"
	"github.com/borro/ragcli/internal/ragcore"
	openai "github.com/sashabaranov/go-openai"
)

const (
	defaultSearchLimit = 5
	maxSearchLimit     = 20
)

type SearchParams struct {
	Path   string
	Query  string
	Limit  int
	Offset int
}

type SearchMatch struct {
	ChunkID    int     `json:"chunk_id,omitempty"`
	Path       string  `json:"path,omitempty"`
	StartLine  int     `json:"start_line"`
	EndLine    int     `json:"end_line"`
	Similarity float64 `json:"similarity,omitempty"`
	Overlap    float64 `json:"overlap,omitempty"`
	Score      float64 `json:"score"`
	Text       string  `json:"text"`
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
}

type searchTool struct {
	searcher *ragcore.PreparedSearch
	embedder llm.EmbeddingRequester
	cache    map[string]aitools.Execution
}

func NewSearchTool(searcher *ragcore.PreparedSearch, embedder llm.EmbeddingRequester) aitools.Tool {
	return &searchTool{
		searcher: searcher,
		embedder: embedder,
		cache:    make(map[string]aitools.Execution),
	}
}

func (t *searchTool) CloneTool() aitools.Tool {
	if t == nil {
		return (*searchTool)(nil)
	}
	cloned := &searchTool{
		searcher: t.searcher,
		embedder: t.embedder,
		cache:    make(map[string]aitools.Execution, len(t.cache)),
	}
	for key, value := range t.cache {
		cloned.cache[key] = aitools.CloneExecution(value)
	}
	return cloned
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

func (t *searchTool) Name() string {
	return "search_rag"
}

func (t *searchTool) ToolDefinition() openai.Tool {
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

func (t *searchTool) DescribeCall(call openai.ToolCall) aitools.CallDescription {
	return aitools.CallDescription{
		Arguments:    aitools.CloneSummary(SummarizeToolArguments(call)),
		VerboseLabel: compactToolCallLabel(call),
	}
}

func (t *searchTool) Execute(ctx context.Context, call openai.ToolCall) (aitools.Execution, error) {
	var params searchToolArgs
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
		Execute: func(ctx context.Context, call openai.ToolCall) (aitools.Execution, error) {
			result, err := t.search(ctx, params.SearchParams)
			if err != nil {
				return aitools.Execution{}, err
			}
			return aitools.Execution{Result: result}, nil
		},
	})
}

func canonicalToolSignature(call openai.ToolCall) (string, searchToolArgs, error) {
	params, err := parseSearchToolArgs(call)
	if err != nil {
		return "", searchToolArgs{}, err
	}
	signature, err := params.canonicalSignature(call.Function.Name)
	if err != nil {
		return "", searchToolArgs{}, err
	}
	return signature, params, nil
}

func (t *searchTool) search(ctx context.Context, params SearchParams) (aitools.ToolResult, error) {
	if t.searcher == nil || t.searcher.Index == nil {
		return aitools.ToolResult{}, aitools.NewToolError("tool_not_ready", "rag index is not prepared", false, map[string]any{
			"tool": "search_rag",
		})
	}
	if t.embedder == nil {
		return aitools.ToolResult{}, aitools.NewToolError("tool_not_ready", "embedding client is not configured", false, map[string]any{
			"tool": "search_rag",
		})
	}
	if params.Query == "" {
		return aitools.ToolResult{}, aitools.NewToolError("invalid_arguments", "query must not be empty", false, map[string]any{
			"tool":  "search_rag",
			"field": "query",
		})
	}
	if params.Path != "" && !pathExists(t.searcher.Index, params.Path) {
		return aitools.ToolResult{}, aitools.NewToolError("invalid_arguments", "path is not part of the attached corpus", false, map[string]any{
			"tool":  "search_rag",
			"field": "path",
			"value": params.Path,
		})
	}

	hits, _, err := t.searcher.Search(ctx, t.embedder, params.Query, params.Path)
	if err != nil {
		return aitools.ToolResult{}, aitools.NewToolError("tool_execution_failed", "failed to search rag index", false, map[string]any{
			"tool":  "search_rag",
			"error": err.Error(),
		})
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
			ChunkID:    hit.Chunk.ChunkID,
			Path:       hit.Chunk.SourcePath,
			StartLine:  hit.Chunk.StartLine,
			EndLine:    hit.Chunk.EndLine,
			Similarity: hit.Similarity,
			Overlap:    hit.Overlap,
			Score:      hit.Score,
			Text:       hit.Chunk.Text,
		})
	}

	return aitools.ToolResult{
		Payload:      result,
		Summary:      summarizeSearchResult(result),
		ProgressKeys: progressKeys(result),
		Hints:        toolHints(result),
		Evidence:     evidence(result),
	}, nil
}

func pathExists(index *ragcore.Index, path string) bool {
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
	params, err := parseSearchToolArgs(call)
	if err != nil {
		return map[string]any{"decode_error": aitools.AsToolError(err).Message}
	}
	return params.summary()
}

func compactToolCallLabel(call openai.ToolCall) string {
	label := strings.TrimSpace(call.Function.Name)
	if label == "" {
		label = "tool"
	}

	raw := strings.TrimSpace(call.Function.Arguments)
	if raw == "" {
		return label
	}
	params, err := parseSearchToolArgs(call)
	if err != nil {
		return compactRawToolCallLabel(label, raw)
	}
	return params.compactLabel(label)
}

func summarizeSearchResult(result SearchResult) map[string]any {
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

func progressKeys(result SearchResult) []string {
	keys := make([]string, 0, len(result.Matches)*4)
	for _, match := range result.Matches {
		path := strings.TrimSpace(match.Path)
		if path == "" {
			path = strings.TrimSpace(result.Path)
		}
		for line := match.StartLine; line <= match.EndLine; line++ {
			keys = append(keys, "rag:"+path+":"+strconv.Itoa(line))
		}
	}
	return keys
}

func toolHints(result SearchResult) map[string]any {
	if result.HasMore {
		return map[string]any{aitools.HintSuggestedNextOffset: result.NextOffset}
	}
	return nil
}

func evidence(result SearchResult) []aitools.Evidence {
	evidence := make([]aitools.Evidence, 0, len(result.Matches))
	for _, match := range result.Matches {
		path := strings.TrimSpace(match.Path)
		if path == "" {
			path = strings.TrimSpace(result.Path)
		}
		if path == "" || match.StartLine < 1 || match.EndLine < match.StartLine {
			continue
		}
		evidence = append(evidence, aitools.Evidence{
			Kind:       aitools.EvidenceRetrieved,
			SourcePath: path,
			StartLine:  match.StartLine,
			EndLine:    match.EndLine,
		})
	}
	return evidence
}
