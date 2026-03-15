package ragcore

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	filetools "github.com/borro/ragcli/internal/aitools/files"
	"github.com/borro/ragcli/internal/retrieval"
	openai "github.com/sashabaranov/go-openai"
)

const (
	maxSeedQueries             = 5
	maxSeedSearchLimit         = 20
	maxSeedVerificationReads   = 4
	maxVerifiedReadLines       = 40
	verifiedReadAroundContext  = 20
	seedFusionReciprocalOffset = 60
)

const (
	seedSearchToolCallIDPrefix = "hybrid-seed-search-rag"
	seedReadToolCallIDPrefix   = "hybrid-seed-read"
)

var seedStopwords = map[string]struct{}{
	"a": {}, "an": {}, "and": {}, "are": {}, "as": {}, "at": {}, "be": {}, "by": {}, "for": {}, "from": {},
	"how": {}, "if": {}, "in": {}, "is": {}, "it": {}, "of": {}, "on": {}, "or": {}, "that": {}, "the": {},
	"to": {}, "what": {}, "when": {}, "where": {}, "which": {}, "with": {}, "about": {}, "есть": {}, "или": {},
	"как": {}, "какая": {}, "какие": {}, "какой": {}, "когда": {}, "где": {}, "для": {}, "что": {}, "это": {},
	"про": {}, "по": {}, "из": {}, "на": {}, "и": {}, "в": {}, "во": {}, "не": {}, "ли": {}, "а": {},
}

type SyntheticToolExecutor interface {
	ExecuteSyntheticToolCall(ctx context.Context, call openai.ToolCall) (string, error)
}

type SeedBundle struct {
	History []openai.ChatCompletionMessage
	Hits    []FusedSeedHit
	Weak    bool
}

type fusedSeedCandidate struct {
	Path       string
	ChunkID    int
	StartLine  int
	EndLine    int
	Text       string
	Fusion     float64
	Score      float64
	Similarity float64
	Overlap    float64
}

type syntheticToolResult struct {
	call   openai.ToolCall
	result string
}

type seedMatch struct {
	Path       string
	ChunkID    int
	StartLine  int
	EndLine    int
	Text       string
	Score      float64
	Similarity float64
	Overlap    float64
}

func BuildSeedBundle(ctx context.Context, executor SyntheticToolExecutor, prompt string, searchOpts SearchOptions) (SeedBundle, error) {
	queries := seedQueries(prompt)
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
		Weak:    FusedSearchTooWeak(fused),
	}, nil
}

func seedQueries(prompt string) []string {
	exact := strings.TrimSpace(prompt)
	normalized := normalizeSeedPrompt(prompt)
	contentTokens := contentTokens(prompt)
	contentOnly := strings.Join(contentTokens, " ")
	firstTokens := strings.Join(contentTokens[:min(8, len(contentTokens))], " ")
	longestTokens := strings.Join(longestTokens(contentTokens, 5), " ")

	candidates := []string{exact, normalized, contentOnly, firstTokens, longestTokens}
	queries := make([]string, 0, maxSeedQueries)
	seen := make(map[string]struct{}, len(candidates))
	for _, candidate := range candidates {
		query := strings.TrimSpace(candidate)
		if query == "" {
			continue
		}
		if _, ok := seen[query]; ok {
			continue
		}
		seen[query] = struct{}{}
		queries = append(queries, query)
		if len(queries) >= maxSeedQueries {
			break
		}
	}
	return queries
}

func normalizeSeedPrompt(prompt string) string {
	fields := strings.FieldsFunc(strings.ToLower(prompt), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r)
	})
	return strings.Join(fields, " ")
}

func contentTokens(prompt string) []string {
	tokens := retrieval.Tokenize(prompt)
	filtered := make([]string, 0, len(tokens))
	for _, token := range tokens {
		if _, stop := seedStopwords[token]; stop {
			continue
		}
		filtered = append(filtered, token)
	}
	return filtered
}

func longestTokens(tokens []string, limit int) []string {
	if len(tokens) == 0 || limit <= 0 {
		return nil
	}

	sorted := append([]string(nil), tokens...)
	sort.SliceStable(sorted, func(i, j int) bool {
		left := utf8.RuneCountInString(sorted[i])
		right := utf8.RuneCountInString(sorted[j])
		if left != right {
			return left > right
		}
		return sorted[i] < sorted[j]
	})
	return sorted[:min(limit, len(sorted))]
}

func executeSeedSearches(ctx context.Context, executor SyntheticToolExecutor, queries []string, searchOpts SearchOptions) ([]syntheticToolResult, []FusedSeedHit, error) {
	searchLimit := min(searchOpts.TopK, maxSeedSearchLimit)
	results := make([]syntheticToolResult, 0, len(queries))
	fused := make(map[string]*fusedSeedCandidate, len(queries)*searchLimit)

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

		var payload SearchResult
		if err := decodeSearchPayload(result, &payload); err != nil {
			return nil, nil, err
		}
		for rank, match := range payload.Matches {
			path := strings.TrimSpace(match.Path)
			if path == "" {
				path = strings.TrimSpace(payload.Path)
			}
			if path == "" {
				continue
			}
			addFusedCandidate(fused, seedMatch{
				Path:       path,
				ChunkID:    match.ChunkID,
				StartLine:  match.StartLine,
				EndLine:    match.EndLine,
				Text:       match.Text,
				Score:      match.Score,
				Similarity: match.Similarity,
				Overlap:    match.Overlap,
			}, rank)
		}
	}

	return results, truncateFusedHits(sortFusedHits(fused), searchOpts.TopK), nil
}

func executeSeedVerificationReads(ctx context.Context, executor SyntheticToolExecutor, hits []FusedSeedHit) ([]syntheticToolResult, error) {
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

func decodeSearchPayload(raw string, payload *SearchResult) error {
	if err := json.Unmarshal([]byte(raw), payload); err != nil {
		return fmt.Errorf("failed to decode hybrid seed search result: %w", err)
	}
	return nil
}

func addFusedCandidate(fused map[string]*fusedSeedCandidate, match seedMatch, rank int) {
	path := strings.TrimSpace(match.Path)
	if path == "" {
		return
	}
	key := fmt.Sprintf("%s:%d", path, match.ChunkID)
	entry := fused[key]
	if entry == nil {
		entry = &fusedSeedCandidate{
			Path:       path,
			ChunkID:    match.ChunkID,
			StartLine:  match.StartLine,
			EndLine:    match.EndLine,
			Text:       match.Text,
			Score:      match.Score,
			Similarity: match.Similarity,
			Overlap:    match.Overlap,
		}
		fused[key] = entry
	}
	entry.Fusion += 1.0 / (seedFusionReciprocalOffset + float64(rank+1))
	if match.Score > entry.Score {
		entry.Score = match.Score
	}
	if match.Similarity > entry.Similarity {
		entry.Similarity = match.Similarity
	}
	if match.Overlap > entry.Overlap {
		entry.Overlap = match.Overlap
	}
	if entry.StartLine == 0 || (match.StartLine > 0 && match.StartLine < entry.StartLine) {
		entry.StartLine = match.StartLine
	}
	if match.EndLine > entry.EndLine {
		entry.EndLine = match.EndLine
	}
	if entry.Text == "" && match.Text != "" {
		entry.Text = match.Text
	}
}

func sortFusedHits(candidates map[string]*fusedSeedCandidate) []FusedSeedHit {
	hits := make([]FusedSeedHit, 0, len(candidates))
	for _, candidate := range candidates {
		hits = append(hits, FusedSeedHit{
			Path:       candidate.Path,
			ChunkID:    candidate.ChunkID,
			StartLine:  candidate.StartLine,
			EndLine:    candidate.EndLine,
			Score:      candidate.Fusion,
			Similarity: candidate.Similarity,
			Overlap:    candidate.Overlap,
			Text:       candidate.Text,
		})
	}

	sort.SliceStable(hits, func(i, j int) bool {
		switch {
		case hits[i].Score > hits[j].Score:
			return true
		case hits[i].Score < hits[j].Score:
			return false
		case hits[i].Similarity > hits[j].Similarity:
			return true
		case hits[i].Similarity < hits[j].Similarity:
			return false
		case hits[i].Overlap > hits[j].Overlap:
			return true
		case hits[i].Overlap < hits[j].Overlap:
			return false
		case hits[i].Path != hits[j].Path:
			return hits[i].Path < hits[j].Path
		default:
			return hits[i].ChunkID < hits[j].ChunkID
		}
	})

	return hits
}

func truncateFusedHits(hits []FusedSeedHit, topK int) []FusedSeedHit {
	if topK > 0 && len(hits) > topK {
		return append([]FusedSeedHit(nil), hits[:topK]...)
	}
	return append([]FusedSeedHit(nil), hits...)
}

func selectVerificationHits(hits []FusedSeedHit, limit int) []FusedSeedHit {
	if len(hits) == 0 || limit <= 0 {
		return nil
	}

	selected := make([]FusedSeedHit, 0, min(limit, len(hits)))
	selectedKeys := make(map[string]struct{}, limit)
	usedPaths := make(map[string]struct{}, limit)

	appendHit := func(hit FusedSeedHit) bool {
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

func overlapsSelected(hit FusedSeedHit, selected []FusedSeedHit) bool {
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

func verificationToolCall(hit FusedSeedHit, index int) (openai.ToolCall, error) {
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
		arguments, err := filetools.MarshalJSON(map[string]any{
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
	arguments, err := filetools.MarshalJSON(map[string]any{
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
	arguments, err := MarshalJSON(map[string]any{
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
