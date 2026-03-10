package retrieval

import (
	"context"
	"fmt"
	"math"
	"strings"
	"unicode"

	"github.com/borro/ragcli/internal/llm"
)

func NormalizeEmbeddingInput(raw string) string {
	return strings.Join(strings.Fields(raw), " ")
}

func EmbedQuery(ctx context.Context, embedder llm.EmbeddingRequester, question string, label string) ([]float32, llm.EmbeddingMetrics, error) {
	vectors, metrics, err := embedder.CreateEmbeddingsWithMetrics(ctx, []string{NormalizeEmbeddingInput(question)})
	if err != nil {
		return nil, llm.EmbeddingMetrics{}, fmt.Errorf("failed to embed %s: %w", label, err)
	}
	if len(vectors) != 1 {
		return nil, llm.EmbeddingMetrics{}, fmt.Errorf("%s embeddings count = %d, want 1", label, len(vectors))
	}
	return vectors[0], metrics, nil
}

func EmbedTexts(ctx context.Context, embedder llm.EmbeddingRequester, texts []string) ([][]float32, llm.EmbeddingMetrics, error) {
	inputs := make([]string, 0, len(texts))
	for _, text := range texts {
		inputs = append(inputs, NormalizeEmbeddingInput(text))
	}

	vectors, metrics, err := embedder.CreateEmbeddingsWithMetrics(ctx, inputs)
	if err != nil {
		return nil, llm.EmbeddingMetrics{}, err
	}
	if len(vectors) != len(texts) {
		return nil, llm.EmbeddingMetrics{}, fmt.Errorf("embedding count mismatch: got %d, want %d", len(vectors), len(texts))
	}
	return vectors, metrics, nil
}

func CosineSimilarity(a, b []float32) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}

	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i] * b[i])
		normA += float64(a[i] * a[i])
		normB += float64(b[i] * b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func TokenOverlapRatio(query, text string) float64 {
	queryTokens := Tokenize(query)
	if len(queryTokens) == 0 {
		return 0
	}
	textLower := strings.ToLower(text)
	matched := 0
	for _, token := range queryTokens {
		if strings.Contains(textLower, token) {
			matched++
		}
	}
	return float64(matched) / float64(len(queryTokens))
}

func Tokenize(input string) []string {
	normalized := strings.ToLower(input)
	fields := strings.FieldsFunc(normalized, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r)
	})
	tokens := make([]string, 0, len(fields))
	seen := make(map[string]struct{}, len(fields))
	for _, field := range fields {
		token := strings.TrimSpace(field)
		if len([]rune(token)) < 2 {
			continue
		}
		if _, ok := seen[token]; ok {
			continue
		}
		seen[token] = struct{}{}
		tokens = append(tokens, token)
	}
	return tokens
}
