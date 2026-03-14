package rag

import (
	"context"
	"time"

	"github.com/borro/ragcli/internal/input"
	"github.com/borro/ragcli/internal/llm"
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

type SearchHit struct {
	Chunk      Chunk
	Similarity float64
	Overlap    float64
	Score      float64
}

type PreparedSearch struct {
	Index   *Index
	Options SearchOptions
}

func SearchOptionsFromRunOptions(opts Options) SearchOptions {
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
	index, err := buildOrLoadIndex(ctx, embedder, source, Options{
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
