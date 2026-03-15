package ragcore

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/borro/ragcli/internal/input"
	"github.com/borro/ragcli/internal/llm"
	"github.com/borro/ragcli/internal/localize"
	"github.com/borro/ragcli/internal/retrieval"
	"github.com/borro/ragcli/internal/verbose"
)

type RunOptions struct {
	TopK           int
	FinalK         int
	ChunkSize      int
	ChunkOverlap   int
	IndexTTL       time.Duration
	IndexDir       string
	Rerank         string
	EmbeddingModel string
}

const (
	indexSchemaVersion = 2
	chunksFilename     = "chunks.jsonl"
	embeddingsFilename = "embeddings.jsonl"
	manifestFilename   = "manifest.json"
)

type Manifest = retrieval.Manifest

type Chunk struct {
	DocID      string `json:"doc_id"`
	SourcePath string `json:"source_path"`
	ChunkID    int    `json:"chunk_id"`
	StartLine  int    `json:"start_line"`
	EndLine    int    `json:"end_line"`
	ByteOffset int    `json:"byte_offset"`
	Text       string `json:"text"`
}

type Index struct {
	Manifest   Manifest
	Chunks     []Chunk
	Embeddings [][]float32
}

type pipelineStats struct {
	EmbeddingCalls      int
	EmbeddingTokens     int
	CacheHit            bool
	ChunkCount          int
	RetrievedK          int
	AnswerContextChunks int
}

type candidate struct {
	Chunk      Chunk
	Similarity float64
	Overlap    float64
	Score      float64
}

func buildOrLoadIndex(ctx context.Context, embedder llm.EmbeddingRequester, source input.FileBackedSource, opts RunOptions, stats *pipelineStats, indexMeter verbose.Meter) (*Index, error) {
	if err := os.MkdirAll(opts.IndexDir, 0o700); err != nil {
		return nil, fmt.Errorf("failed to create rag index dir: %w", err)
	}
	retrieval.CleanupExpiredIndexes(opts.IndexDir, opts.IndexTTL)

	if source.IsMultiFile() {
		return buildOrLoadIndexFromFiles(ctx, embedder, source, opts, stats, indexMeter)
	}

	sourcePath := normalizeSourcePath(source)
	reader, err := source.Open()
	if err != nil {
		return nil, err
	}
	defer func() {
		_ = reader.Close()
	}()

	spooled, err := retrieval.SpoolSource(reader, "ragcli-rag-source-*", []string{
		sourcePath,
		fmt.Sprintf("%d", indexSchemaVersion),
		fmt.Sprintf("%d", opts.ChunkSize),
		fmt.Sprintf("%d", opts.ChunkOverlap),
		opts.EmbeddingModel,
	}, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to spool source: %w", err)
	}
	defer func() {
		_ = os.Remove(spooled.TempPath)
	}()
	indexMeter.Note(localize.T("progress.rag.read_done", localize.Data{"Bytes": spooled.Bytes}))
	indexDir := filepath.Join(opts.IndexDir, spooled.Hash)

	if cached, err := loadIndex(indexDir, opts.IndexTTL); err == nil {
		stats.CacheHit = true
		indexMeter.Done(localize.T("progress.rag.cache_hit"))
		return cached, nil
	}

	sourceFile, err := os.Open(spooled.TempPath)
	if err != nil {
		return nil, fmt.Errorf("failed to reopen spooled source: %w", err)
	}
	chunks, err := chunkTextReader(sourceFile, sourcePath, spooled.Hash, opts.ChunkSize, opts.ChunkOverlap)
	_ = sourceFile.Close()
	if err != nil {
		return nil, fmt.Errorf("failed to chunk source: %w", err)
	}
	if len(chunks) == 0 {
		return nil, errors.New("input did not produce retrieval chunks")
	}
	indexMeter.Note(localize.T("progress.rag.index_build", localize.Data{"Count": len(chunks)}))
	embeddings, calls, tokens, err := embedChunks(ctx, embedder, chunks, indexMeter)
	if err != nil {
		return nil, err
	}
	stats.EmbeddingCalls += calls
	stats.EmbeddingTokens += tokens

	index := &Index{
		Manifest: Manifest{
			SchemaVersion:  indexSchemaVersion,
			CreatedAt:      time.Now().UTC(),
			InputHash:      spooled.Hash,
			SourcePath:     sourcePath,
			EmbeddingModel: opts.EmbeddingModel,
			ChunkSize:      opts.ChunkSize,
			ChunkOverlap:   opts.ChunkOverlap,
			ItemCount:      len(chunks),
		},
		Chunks:     chunks,
		Embeddings: embeddings,
	}

	if err := persistIndex(indexDir, index); err != nil {
		if cached, loadErr := loadIndex(indexDir, opts.IndexTTL); loadErr == nil {
			stats.CacheHit = true
			indexMeter.Done(localize.T("progress.rag.index_race_cached"))
			return cached, nil
		}
		return nil, err
	}
	indexMeter.Done(localize.T("progress.rag.index_saved"))

	return index, nil
}

func buildOrLoadIndexFromFiles(ctx context.Context, embedder llm.EmbeddingRequester, source input.FileBackedSource, opts RunOptions, stats *pipelineStats, indexMeter verbose.Meter) (*Index, error) {
	sourcePath := normalizeSourcePath(source)
	files := source.BackingFiles()
	inputHash, err := hashSourceFiles(files, []string{
		sourcePath,
		fmt.Sprintf("%d", indexSchemaVersion),
		fmt.Sprintf("%d", opts.ChunkSize),
		fmt.Sprintf("%d", opts.ChunkOverlap),
		opts.EmbeddingModel,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to hash source files: %w", err)
	}
	indexDir := filepath.Join(opts.IndexDir, inputHash)

	if cached, err := loadIndex(indexDir, opts.IndexTTL); err == nil {
		stats.CacheHit = true
		indexMeter.Done(localize.T("progress.rag.cache_hit"))
		return cached, nil
	}

	chunks, err := chunkSourceFiles(files, inputHash, opts.ChunkSize, opts.ChunkOverlap)
	if err != nil {
		return nil, fmt.Errorf("failed to chunk source files: %w", err)
	}
	if len(chunks) == 0 {
		return nil, errors.New("input did not produce retrieval chunks")
	}
	indexMeter.Note(localize.T("progress.rag.index_build", localize.Data{"Count": len(chunks)}))
	embeddings, calls, tokens, err := embedChunks(ctx, embedder, chunks, indexMeter)
	if err != nil {
		return nil, err
	}
	stats.EmbeddingCalls += calls
	stats.EmbeddingTokens += tokens

	index := &Index{
		Manifest: Manifest{
			SchemaVersion:  indexSchemaVersion,
			CreatedAt:      time.Now().UTC(),
			InputHash:      inputHash,
			SourcePath:     sourcePath,
			EmbeddingModel: opts.EmbeddingModel,
			ChunkSize:      opts.ChunkSize,
			ChunkOverlap:   opts.ChunkOverlap,
			ItemCount:      len(chunks),
		},
		Chunks:     chunks,
		Embeddings: embeddings,
	}

	if err := persistIndex(indexDir, index); err != nil {
		if cached, loadErr := loadIndex(indexDir, opts.IndexTTL); loadErr == nil {
			stats.CacheHit = true
			indexMeter.Done(localize.T("progress.rag.index_race_cached"))
			return cached, nil
		}
		return nil, err
	}
	indexMeter.Done(localize.T("progress.rag.index_saved"))
	return index, nil
}

func hashSourceFiles(files []input.File, hashParts []string) (string, error) {
	hasher := sha256.New()
	for _, part := range hashParts {
		_, _ = fmt.Fprintf(hasher, "|%s", part)
	}
	for _, file := range files {
		_, _ = fmt.Fprintf(hasher, "|%s|", file.DisplayPath)
		reader, err := os.Open(file.Path)
		if err != nil {
			return "", err
		}
		_, copyErr := io.Copy(hasher, reader)
		closeErr := reader.Close()
		if copyErr != nil {
			return "", copyErr
		}
		if closeErr != nil {
			return "", closeErr
		}
	}
	return hex.EncodeToString(hasher.Sum(nil)), nil
}

func chunkSourceFiles(files []input.File, docID string, chunkSize, overlap int) ([]Chunk, error) {
	chunks := make([]Chunk, 0, len(files)*2)
	for _, file := range files {
		reader, err := os.Open(file.Path)
		if err != nil {
			return nil, err
		}
		fileChunks, chunkErr := chunkTextReader(reader, file.DisplayPath, docID, chunkSize, overlap)
		closeErr := reader.Close()
		if chunkErr != nil {
			return nil, chunkErr
		}
		if closeErr != nil {
			return nil, closeErr
		}
		chunks = append(chunks, fileChunks...)
	}
	return chunks, nil
}

func normalizeSourcePath(source input.SourceMeta) string {
	if displayName := strings.TrimSpace(source.DisplayName()); displayName != "" {
		return displayName
	}
	if inputPath := source.InputPath(); inputPath != "" {
		return inputPath
	}
	return "stdin"
}

func loadIndex(indexDir string, ttl time.Duration) (*Index, error) {
	manifest, chunks, embeddings, err := retrieval.LoadIndex[Chunk](indexDir, ttl, indexSchemaVersion, retrieval.Filenames{
		Manifest:   manifestFilename,
		Items:      chunksFilename,
		Embeddings: embeddingsFilename,
	})
	if err != nil {
		return nil, retrieval.WrapIndexError("rag", err, "chunks and embeddings count mismatch")
	}
	return &Index{
		Manifest:   manifest,
		Chunks:     chunks,
		Embeddings: embeddings,
	}, nil
}

func persistIndex(indexDir string, index *Index) error {
	return retrieval.PersistIndex(indexDir, index.Manifest, index.Chunks, index.Embeddings, retrieval.Filenames{
		Manifest:   manifestFilename,
		Items:      chunksFilename,
		Embeddings: embeddingsFilename,
	})
}

func chunkText(content []byte, sourcePath string, chunkSize, overlap int) []Chunk {
	chunks, err := chunkTextReader(bytes.NewReader(content), sourcePath, sourcePath, chunkSize, overlap)
	if err != nil {
		return nil
	}
	return chunks
}

func chunkTextReader(reader io.Reader, sourcePath, docID string, chunkSize, overlap int) ([]Chunk, error) {
	type lineInfo struct {
		text      string
		byteStart int
		byteLen   int
		lineNo    int
	}

	window := make([]lineInfo, 0, 32)
	chunks := make([]Chunk, 0, 32)
	windowBytes := 0

	emitChunk := func(lines []lineInfo) {
		if len(lines) == 0 {
			return
		}
		textParts := make([]string, 0, len(lines))
		for _, line := range lines {
			textParts = append(textParts, line.text)
		}
		chunks = append(chunks, Chunk{
			DocID:      docID,
			SourcePath: sourcePath,
			ChunkID:    len(chunks),
			StartLine:  lines[0].lineNo,
			EndLine:    lines[len(lines)-1].lineNo,
			ByteOffset: lines[0].byteStart,
			Text:       strings.Join(textParts, "\n"),
		})
	}

	retainOverlap := func(lines []lineInfo) []lineInfo {
		if overlap <= 0 || len(lines) == 0 {
			windowBytes = 0
			return nil
		}
		backBytes := 0
		start := len(lines)
		for i := len(lines) - 1; i >= 0; i-- {
			backBytes += lines[i].byteLen
			start = i
			if backBytes >= overlap {
				break
			}
		}
		if start <= 0 {
			windowBytes = 0
			return nil
		}
		retained := append([]lineInfo(nil), lines[start:]...)
		windowBytes = 0
		for _, line := range retained {
			windowBytes += line.byteLen
		}
		return retained
	}

	_, err := retrieval.WalkLines(reader, func(rawLine, text string, lineNo, byteStart, _ int) error {
		line := lineInfo{
			text:      text,
			byteStart: byteStart,
			byteLen:   len(rawLine),
			lineNo:    lineNo,
		}

		for len(window) > 0 && windowBytes+line.byteLen > chunkSize {
			emitChunk(window)
			window = retainOverlap(window)
		}

		window = append(window, line)
		windowBytes += line.byteLen
		return nil
	})
	if err != nil {
		return nil, err
	}

	if len(window) > 0 {
		emitChunk(window)
	}
	return chunks, nil
}

func embedChunks(ctx context.Context, embedder llm.EmbeddingRequester, chunks []Chunk, indexMeter verbose.Meter) ([][]float32, int, int, error) {
	const batchSize = 32
	vectors := make([][]float32, 0, len(chunks))
	calls := 0
	totalTokens := 0
	totalBatches := (len(chunks) + batchSize - 1) / batchSize

	for start := 0; start < len(chunks); start += batchSize {
		end := min(start+batchSize, len(chunks))
		inputs := make([]string, 0, end-start)
		for _, chunk := range chunks[start:end] {
			inputs = append(inputs, chunk.Text)
		}
		indexMeter.Note(localize.T("progress.rag.embed_batch", localize.Data{"Index": calls + 1, "Total": totalBatches}))

		vectorsBatch, metrics, err := retrieval.EmbedTexts(ctx, embedder, inputs)
		if err != nil {
			return nil, calls, totalTokens, fmt.Errorf("failed to embed chunks: %w", err)
		}
		calls++
		totalTokens += metrics.TotalTokens
		vectors = append(vectors, vectorsBatch...)
		indexMeter.Report(calls, totalBatches, fmt.Sprintf("готов embeddings batch %d/%d", calls, totalBatches))
	}

	if len(vectors) != len(chunks) {
		return nil, calls, totalTokens, fmt.Errorf("embedding count mismatch: got %d, want %d", len(vectors), len(chunks))
	}
	return vectors, calls, totalTokens, nil
}

func embedQuery(ctx context.Context, embedder llm.EmbeddingRequester, question string) ([]float32, llm.EmbeddingMetrics, error) {
	return retrieval.EmbedQuery(ctx, embedder, question, "query")
}

func retrieveSearchHits(index *Index, query []float32, question string, opts SearchOptions, pathFilter string) []SearchHit {
	ranked := retrieveCandidates(index, query, question, opts, pathFilter)
	hits := make([]SearchHit, 0, len(ranked))
	for _, hit := range ranked {
		hits = append(hits, SearchHit(hit))
	}
	return hits
}

func retrieveCandidates(index *Index, query []float32, question string, opts SearchOptions, pathFilter string) []candidate {
	ranked := make([]candidate, 0, len(index.Chunks))
	filter := strings.TrimSpace(pathFilter)
	for i, chunk := range index.Chunks {
		if filter != "" && chunk.SourcePath != filter {
			continue
		}
		similarity := retrieval.CosineSimilarity(query, index.Embeddings[i])
		ranked = append(ranked, candidate{
			Chunk:      chunk,
			Similarity: similarity,
			Overlap:    retrieval.TokenOverlapRatio(question, chunk.Text),
			Score:      similarity,
		})
	}

	slices.SortStableFunc(ranked, func(a, b candidate) int {
		switch {
		case a.Score > b.Score:
			return -1
		case a.Score < b.Score:
			return 1
		default:
			return cmpChunkOrder(a.Chunk, b.Chunk)
		}
	})
	if len(ranked) > opts.TopK {
		ranked = slices.Clone(ranked[:opts.TopK])
	}

	switch opts.Rerank {
	case "off":
		return ranked
	case "model":
		fallthrough
	default:
		for i := range ranked {
			ranked[i].Score = heuristicScore(question, ranked[i])
		}
		slices.SortStableFunc(ranked, func(a, b candidate) int {
			switch {
			case a.Score > b.Score:
				return -1
			case a.Score < b.Score:
				return 1
			default:
				return cmpChunkOrder(a.Chunk, b.Chunk)
			}
		})
		return ranked
	}
}

func heuristicScore(question string, c candidate) float64 {
	score := c.Similarity + 0.25*c.Overlap
	lowerQuestion := strings.ToLower(strings.TrimSpace(question))
	if lowerQuestion != "" && strings.Contains(strings.ToLower(c.Chunk.Text), lowerQuestion) {
		score += 0.15
	}
	if c.Chunk.StartLine == 1 {
		score += 0.01
	}
	return score
}

func cmpChunkOrder(a, b Chunk) int {
	switch {
	case a.SourcePath < b.SourcePath:
		return -1
	case a.SourcePath > b.SourcePath:
		return 1
	case a.ChunkID < b.ChunkID:
		return -1
	case a.ChunkID > b.ChunkID:
		return 1
	default:
		return 0
	}
}
