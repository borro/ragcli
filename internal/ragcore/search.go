package ragcore

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/borro/ragcli/internal/input"
	"github.com/borro/ragcli/internal/llm"
	"github.com/borro/ragcli/internal/retrieval"
	"github.com/borro/ragcli/internal/verbose"
)

type SearchOptions struct {
	TopK           int
	ChunkSize      int
	ChunkOverlap   int
	IndexTTL       time.Duration
	IndexDir       string
	Rerank         string
	EmbeddingModel string
}

type SearchStats struct {
	EmbeddingCalls  int
	EmbeddingTokens int
	CacheHit        bool
	ChunkCount      int
}

type FusedSearchStats struct {
	EmbeddingCalls  int
	EmbeddingTokens int
}

type SearchHit struct {
	Chunk      Chunk
	Similarity float64
	Overlap    float64
	Score      float64
}

type FusedSeedHit struct {
	Path       string
	ChunkID    int
	StartLine  int
	EndLine    int
	Score      float64
	Similarity float64
	Overlap    float64
	Text       string
}

type PreparedSearch struct {
	Index   *Index
	Options SearchOptions
}

func SearchOptionsFromRunOptions(opts RunOptions) SearchOptions {
	return SearchOptions{
		TopK:           opts.TopK,
		ChunkSize:      opts.ChunkSize,
		ChunkOverlap:   opts.ChunkOverlap,
		IndexTTL:       opts.IndexTTL,
		IndexDir:       opts.IndexDir,
		Rerank:         opts.Rerank,
		EmbeddingModel: opts.EmbeddingModel,
	}
}

func PrepareSearch(ctx context.Context, embedder llm.EmbeddingRequester, source input.FileBackedSource, opts SearchOptions, meter verbose.Meter) (*PreparedSearch, SearchStats, error) {
	stats := pipelineStats{}
	index, err := buildOrLoadIndex(ctx, embedder, source, RunOptions{
		TopK:           opts.TopK,
		FinalK:         1,
		ChunkSize:      opts.ChunkSize,
		ChunkOverlap:   opts.ChunkOverlap,
		IndexTTL:       opts.IndexTTL,
		IndexDir:       opts.IndexDir,
		Rerank:         opts.Rerank,
		EmbeddingModel: opts.EmbeddingModel,
	}, &stats, meter)
	if err != nil {
		return nil, SearchStats{}, err
	}

	return &PreparedSearch{
			Index:   index,
			Options: opts,
		}, SearchStats{
			EmbeddingCalls:  stats.EmbeddingCalls,
			EmbeddingTokens: stats.EmbeddingTokens,
			CacheHit:        stats.CacheHit,
			ChunkCount:      len(index.Chunks),
		}, nil
}

func (p *PreparedSearch) Search(ctx context.Context, embedder llm.EmbeddingRequester, question string, path string) ([]SearchHit, llm.EmbeddingMetrics, error) {
	queryEmbedding, metrics, err := embedQuery(ctx, embedder, question)
	if err != nil {
		return nil, llm.EmbeddingMetrics{}, err
	}

	ranked := retrieveSearchHits(p.Index, queryEmbedding, question, p.Options, path)
	return ranked, metrics, nil
}

func (p *PreparedSearch) FusedSearch(ctx context.Context, embedder llm.EmbeddingRequester, prompt string, path string) ([]FusedSeedHit, FusedSearchStats, error) {
	return p.fusedSearch(ctx, embedder, prompt, path, nil)
}

func (p *PreparedSearch) FusedSearchWithHook(ctx context.Context, embedder llm.EmbeddingRequester, prompt string, path string, onFirstQueryEmbedded func()) ([]FusedSeedHit, FusedSearchStats, error) {
	return p.fusedSearch(ctx, embedder, prompt, path, onFirstQueryEmbedded)
}

func (p *PreparedSearch) fusedSearch(ctx context.Context, embedder llm.EmbeddingRequester, prompt string, path string, onFirstQueryEmbedded func()) ([]FusedSeedHit, FusedSearchStats, error) {
	queries := seedQueries(prompt)
	fused := make(map[string]*fusedSeedCandidate, len(queries)*p.Options.TopK)
	stats := FusedSearchStats{}
	firstQueryReady := false
	filter := strings.TrimSpace(path)

	for _, query := range queries {
		hits, metrics, err := p.Search(ctx, embedder, query, filter)
		if err != nil {
			return nil, FusedSearchStats{}, fmt.Errorf("failed to execute fused rag search: %w", err)
		}
		stats.EmbeddingCalls++
		stats.EmbeddingTokens += metrics.TotalTokens
		if !firstQueryReady && onFirstQueryEmbedded != nil {
			onFirstQueryEmbedded()
			firstQueryReady = true
		}
		for rank, hit := range hits {
			addFusedCandidate(fused, seedMatch{
				Path:       hit.Chunk.SourcePath,
				ChunkID:    hit.Chunk.ChunkID,
				StartLine:  hit.Chunk.StartLine,
				EndLine:    hit.Chunk.EndLine,
				Text:       hit.Chunk.Text,
				Score:      hit.Score,
				Similarity: hit.Similarity,
				Overlap:    hit.Overlap,
			}, rank)
		}
	}

	return truncateFusedHits(sortFusedHits(fused), p.Options.TopK), stats, nil
}

func FusedSearchTooWeak(hits []FusedSeedHit) bool {
	if len(hits) == 0 {
		return true
	}
	return retrieval.SemanticSearchTooWeak(hits[0].Similarity, hits[0].Overlap)
}
