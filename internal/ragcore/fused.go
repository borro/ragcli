package ragcore

import (
	"sort"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/borro/ragcli/internal/retrieval"
)

const (
	maxSeedQueries             = 5
	seedFusionReciprocalOffset = 60
)

var seedStopwords = map[string]struct{}{
	"a": {}, "an": {}, "and": {}, "are": {}, "as": {}, "at": {}, "be": {}, "by": {}, "for": {}, "from": {},
	"how": {}, "if": {}, "in": {}, "is": {}, "it": {}, "of": {}, "on": {}, "or": {}, "that": {}, "the": {},
	"to": {}, "what": {}, "when": {}, "where": {}, "which": {}, "with": {}, "about": {}, "есть": {}, "или": {},
	"как": {}, "какая": {}, "какие": {}, "какой": {}, "когда": {}, "где": {}, "для": {}, "что": {}, "это": {},
	"про": {}, "по": {}, "из": {}, "на": {}, "и": {}, "в": {}, "во": {}, "не": {}, "ли": {}, "а": {},
}

type FusedSeedMatch struct {
	Path       string
	ChunkID    int
	StartLine  int
	EndLine    int
	Text       string
	Score      float64
	Similarity float64
	Overlap    float64
	Rank       int
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

func SeedQueries(prompt string) []string {
	exact := strings.TrimSpace(prompt)
	normalized := normalizeSeedPrompt(prompt)
	contentTokens := seedContentTokens(prompt)
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

func FuseSeedMatches(matches []FusedSeedMatch, topK int) []FusedSeedHit {
	fused := make(map[string]*fusedSeedCandidate, len(matches))
	for _, match := range matches {
		path := strings.TrimSpace(match.Path)
		if path == "" {
			continue
		}
		key := path + ":" + strconv.Itoa(match.ChunkID)
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
		entry.Fusion += 1.0 / (seedFusionReciprocalOffset + float64(match.Rank+1))
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

	hits := make([]FusedSeedHit, 0, len(fused))
	for _, candidate := range fused {
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

	if topK > 0 && len(hits) > topK {
		return append([]FusedSeedHit(nil), hits[:topK]...)
	}
	return append([]FusedSeedHit(nil), hits...)
}

func normalizeSeedPrompt(prompt string) string {
	fields := strings.FieldsFunc(strings.ToLower(prompt), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r)
	})
	return strings.Join(fields, " ")
}

func seedContentTokens(prompt string) []string {
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
