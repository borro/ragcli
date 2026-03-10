package rag

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"
	"unicode"

	"github.com/borro/ragcli/internal/llm"
	"github.com/borro/ragcli/internal/localize"
	"github.com/borro/ragcli/internal/verbose"
	openai "github.com/sashabaranov/go-openai"
)

type Options struct {
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
	indexSchemaVersion = 1
	chunksFilename     = "chunks.jsonl"
	embeddingsFilename = "embeddings.jsonl"
	manifestFilename   = "manifest.json"
)

func ragAntiInjectionPolicy() string {
	return localize.T("rag.prompt.shared_policy")
}

type Source struct {
	Path        string
	DisplayName string
	Reader      io.Reader
}

type Manifest struct {
	SchemaVersion  int       `json:"schema_version"`
	CreatedAt      time.Time `json:"created_at"`
	InputHash      string    `json:"input_hash"`
	DocID          string    `json:"doc_id"`
	SourcePath     string    `json:"source_path"`
	EmbeddingModel string    `json:"embedding_model"`
	ChunkSize      int       `json:"chunk_size"`
	ChunkOverlap   int       `json:"chunk_overlap"`
	ChunkCount     int       `json:"chunk_count"`
}

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

func Run(ctx context.Context, chat llm.ChatRequester, embedder llm.EmbeddingRequester, source Source, opts Options, question string, plan *verbose.Plan) (string, error) {
	startedAt := time.Now()
	stats := pipelineStats{}
	if plan == nil {
		plan = verbose.NewPlan(nil, localize.T("progress.mode.rag"),
			verbose.StageDef{Key: "prepare", Label: localize.T("progress.stage.prepare"), Slots: 2},
			verbose.StageDef{Key: "index", Label: localize.T("progress.stage.index"), Slots: 8},
			verbose.StageDef{Key: "retrieval", Label: localize.T("progress.stage.retrieval"), Slots: 7},
			verbose.StageDef{Key: "final", Label: localize.T("progress.stage.final"), Slots: 7},
		)
	}
	indexMeter := plan.Stage("index")
	retrievalMeter := plan.Stage("retrieval")
	finalMeter := plan.Stage("final")

	if source.Reader == nil {
		return "", errors.New("rag source reader is required")
	}

	indexMeter.Start(localize.T("progress.rag.read_input"))
	index, err := buildOrLoadIndex(ctx, embedder, source, opts, &stats, indexMeter)
	if err != nil {
		return "", err
	}
	stats.ChunkCount = len(index.Chunks)

	retrievalMeter.Start(localize.T("progress.rag.embed_query"))
	queryEmbedding, metrics, err := embedQuery(ctx, embedder, question)
	if err != nil {
		return "", err
	}
	stats.EmbeddingCalls += 1
	stats.EmbeddingTokens += metrics.TotalTokens
	retrievalMeter.Note(localize.T("progress.rag.query_embedding_ready"))

	retrievalMeter.Note(localize.T("progress.rag.search_relevant"))
	ranked := retrieve(index, queryEmbedding, question, opts)
	retrievalMeter.Done(localize.T("progress.rag.retrieval_done", localize.Data{"Candidates": len(ranked), "Evidence": min(len(ranked), opts.FinalK)}))
	if len(ranked) == 0 || retrievalTooWeak(ranked) {
		slog.Debug("rag retrieval insufficient",
			"duration", float64(time.Since(startedAt).Round(time.Millisecond))/float64(time.Second),
			"chunk_count", stats.ChunkCount,
			"embedding_calls", stats.EmbeddingCalls,
			"embedding_tokens", stats.EmbeddingTokens,
			"cache_hit", stats.CacheHit,
			"retrieved_k", 0,
		)
		return localize.T("rag.answer.insufficient"), nil
	}
	stats.RetrievedK = len(ranked)

	evidence := selectEvidence(ranked, opts.FinalK)
	stats.AnswerContextChunks = len(evidence)

	finalMeter.Start(localize.T("progress.rag.final_start"))
	finalMeter.Note(localize.T("progress.rag.final_send", localize.Data{"Count": len(evidence)}))
	answer, chatMetrics, err := synthesizeAnswer(ctx, chat, question, evidence)
	if err != nil {
		return "", err
	}
	finalMeter.Note(localize.T("progress.rag.final_citations"))

	slog.Debug("rag processing finished",
		"duration", float64(time.Since(startedAt).Round(time.Millisecond))/float64(time.Second),
		"chunk_count", stats.ChunkCount,
		"embedding_calls", stats.EmbeddingCalls,
		"embedding_tokens", stats.EmbeddingTokens,
		"cache_hit", stats.CacheHit,
		"retrieved_k", stats.RetrievedK,
		"answer_context_chunks", stats.AnswerContextChunks,
		"prompt_tokens", chatMetrics.PromptTokens,
		"completion_tokens", chatMetrics.CompletionTokens,
		"total_tokens", chatMetrics.TotalTokens,
	)

	finalMeter.Done(localize.T("progress.rag.final_done"))
	return appendSources(answer, evidence), nil
}

func buildOrLoadIndex(ctx context.Context, embedder llm.EmbeddingRequester, source Source, opts Options, stats *pipelineStats, indexMeter verbose.Meter) (*Index, error) {
	if err := os.MkdirAll(opts.IndexDir, 0o700); err != nil {
		return nil, fmt.Errorf("failed to create rag index dir: %w", err)
	}
	cleanupExpiredIndexes(opts.IndexDir, opts.IndexTTL)

	sourcePath := normalizeSourcePath(source)
	tempSourcePath, sourceBytes, hash, err := spoolSourceToTemp(source.Reader, sourcePath, opts)
	if err != nil {
		return nil, fmt.Errorf("failed to spool source: %w", err)
	}
	defer func() {
		_ = os.Remove(tempSourcePath)
	}()
	indexMeter.Note(localize.T("progress.rag.read_done", localize.Data{"Bytes": sourceBytes}))
	indexDir := filepath.Join(opts.IndexDir, hash)

	if cached, err := loadIndex(indexDir, opts.IndexTTL); err == nil {
		stats.CacheHit = true
		indexMeter.Done(localize.T("progress.rag.cache_hit"))
		return cached, nil
	}

	sourceFile, err := os.Open(tempSourcePath)
	if err != nil {
		return nil, fmt.Errorf("failed to reopen spooled source: %w", err)
	}
	chunks, err := chunkTextReader(sourceFile, sourcePath, hash, opts.ChunkSize, opts.ChunkOverlap)
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
			InputHash:      hash,
			DocID:          hash,
			SourcePath:     sourcePath,
			EmbeddingModel: opts.EmbeddingModel,
			ChunkSize:      opts.ChunkSize,
			ChunkOverlap:   opts.ChunkOverlap,
			ChunkCount:     len(chunks),
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

func normalizeSourcePath(source Source) string {
	if strings.TrimSpace(source.DisplayName) != "" {
		return strings.TrimSpace(source.DisplayName)
	}
	if strings.TrimSpace(source.Path) != "" {
		return strings.TrimSpace(source.Path)
	}
	return "stdin"
}

func walkRawLines(reader io.Reader, onLine func(rawLine string, lineNo, byteStart, byteEnd int) error) (int, error) {
	buffered := bufio.NewReader(reader)
	byteOffset := 0
	lineNo := 1

	for {
		rawLine, err := buffered.ReadString('\n')
		if err != nil && !errors.Is(err, io.EOF) {
			return byteOffset, err
		}
		if rawLine == "" && errors.Is(err, io.EOF) {
			break
		}

		nextOffset := byteOffset + len(rawLine)
		if err := onLine(rawLine, lineNo, byteOffset, nextOffset); err != nil {
			return byteOffset, err
		}
		byteOffset = nextOffset
		lineNo++

		if errors.Is(err, io.EOF) {
			break
		}
	}
	return byteOffset, nil
}

func spoolSourceToTemp(reader io.Reader, sourcePath string, opts Options) (string, int, string, error) {
	tempFile, err := os.CreateTemp("", "ragcli-rag-source-*")
	if err != nil {
		return "", 0, "", err
	}
	defer func() {
		_ = tempFile.Close()
	}()

	hasher := sha256.New()
	written, err := walkRawLines(reader, func(rawLine string, _ int, _, _ int) error {
		if _, writeErr := io.WriteString(tempFile, rawLine); writeErr != nil {
			return writeErr
		}
		if _, writeErr := io.WriteString(hasher, rawLine); writeErr != nil {
			return writeErr
		}
		return nil
	})
	if err != nil {
		_ = os.Remove(tempFile.Name())
		return "", 0, "", err
	}
	writeHashMetadata(hasher, sourcePath, opts)
	return tempFile.Name(), written, hex.EncodeToString(hasher.Sum(nil)), nil
}

func writeHashMetadata(w io.Writer, sourcePath string, opts Options) {
	_, _ = fmt.Fprintf(w, "|%s|%d|%d|%d|%s",
		sourcePath,
		indexSchemaVersion,
		opts.ChunkSize,
		opts.ChunkOverlap,
		opts.EmbeddingModel,
	)
}

func cleanupExpiredIndexes(baseDir string, ttl time.Duration) {
	entries, err := os.ReadDir(baseDir)
	if err != nil {
		return
	}
	now := time.Now()
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			continue
		}
		if now.Sub(info.ModTime()) <= ttl {
			continue
		}
		_ = os.RemoveAll(filepath.Join(baseDir, entry.Name()))
	}
}

func loadIndex(indexDir string, ttl time.Duration) (*Index, error) {
	manifestPath := filepath.Join(indexDir, manifestFilename)
	manifestData, err := os.ReadFile(manifestPath)
	if err != nil {
		return nil, err
	}

	var manifest Manifest
	if err := json.Unmarshal(manifestData, &manifest); err != nil {
		return nil, fmt.Errorf("failed to decode manifest: %w", err)
	}
	if manifest.SchemaVersion != indexSchemaVersion {
		return nil, errors.New("rag index schema mismatch")
	}
	if time.Since(manifest.CreatedAt) > ttl {
		return nil, errors.New("rag index ttl expired")
	}

	chunks, err := readChunks(filepath.Join(indexDir, chunksFilename))
	if err != nil {
		return nil, err
	}
	embeddings, err := readEmbeddings(filepath.Join(indexDir, embeddingsFilename))
	if err != nil {
		return nil, err
	}
	if len(chunks) != len(embeddings) {
		return nil, errors.New("rag index corrupted: chunks and embeddings count mismatch")
	}

	return &Index{
		Manifest:   manifest,
		Chunks:     chunks,
		Embeddings: embeddings,
	}, nil
}

func writeJSONFile(path string, value any, createErr, encodeErr, closeErr string) error {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		return fmt.Errorf("%s: %w", createErr, err)
	}
	if err := json.NewEncoder(file).Encode(value); err != nil {
		_ = file.Close()
		return fmt.Errorf("%s: %w", encodeErr, err)
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("%s: %w", closeErr, err)
	}
	return nil
}

func writeJSONLines[T any](path string, values []T, createErr, encodeErr, closeErr string) error {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		return fmt.Errorf("%s: %w", createErr, err)
	}

	encoder := json.NewEncoder(file)
	for _, value := range values {
		if err := encoder.Encode(value); err != nil {
			_ = file.Close()
			return fmt.Errorf("%s: %w", encodeErr, err)
		}
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("%s: %w", closeErr, err)
	}
	return nil
}

func readJSONLines[T any](path string, maxTokenSize int, openErr, decodeErr, scanErr string) ([]T, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", openErr, err)
	}
	defer func() {
		_ = file.Close()
	}()

	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 0, 64*1024), maxTokenSize)
	values := make([]T, 0, 32)
	for scanner.Scan() {
		var value T
		if err := json.Unmarshal(scanner.Bytes(), &value); err != nil {
			return nil, fmt.Errorf("%s: %w", decodeErr, err)
		}
		values = append(values, value)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("%s: %w", scanErr, err)
	}
	return values, nil
}

func persistIndex(indexDir string, index *Index) error {
	parentDir := filepath.Dir(indexDir)
	if err := os.MkdirAll(parentDir, 0o700); err != nil {
		return fmt.Errorf("failed to create parent index dir: %w", err)
	}

	tempDir, err := os.MkdirTemp(parentDir, "."+filepath.Base(indexDir)+".tmp-*")
	if err != nil {
		return fmt.Errorf("failed to create temp index dir: %w", err)
	}
	defer func() {
		_ = os.RemoveAll(tempDir)
	}()

	if err := os.Chmod(tempDir, 0o700); err != nil {
		return fmt.Errorf("failed to secure temp index dir: %w", err)
	}

	if _, err := os.Stat(indexDir); err == nil {
		return nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("failed to stat index dir: %w", err)
	}

	if err := writeJSONLines(filepath.Join(tempDir, chunksFilename), index.Chunks,
		"failed to create chunks file",
		"failed to write chunk metadata",
		"failed to close chunks file",
	); err != nil {
		return err
	}
	if err := writeJSONLines(filepath.Join(tempDir, embeddingsFilename), index.Embeddings,
		"failed to create embeddings file",
		"failed to write embeddings",
		"failed to close embeddings file",
	); err != nil {
		return err
	}
	if err := writeJSONFile(filepath.Join(tempDir, manifestFilename), index.Manifest,
		"failed to create manifest file",
		"failed to write manifest",
		"failed to close manifest file",
	); err != nil {
		return err
	}

	if err := os.Rename(tempDir, indexDir); err != nil {
		if errors.Is(err, os.ErrExist) {
			return nil
		}
		return fmt.Errorf("failed to publish index dir: %w", err)
	}

	return nil
}

func readChunks(path string) ([]Chunk, error) {
	return readJSONLines[Chunk](path, 2*1024*1024,
		"failed to open chunks file",
		"failed to decode chunk metadata",
		"failed to scan chunks file",
	)
}

func readEmbeddings(path string) ([][]float32, error) {
	return readJSONLines[[]float32](path, 8*1024*1024,
		"failed to open embeddings file",
		"failed to decode embedding vector",
		"failed to scan embeddings file",
	)
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

	_, err := walkRawLines(reader, func(rawLine string, lineNo, byteStart, _ int) error {
		line := lineInfo{
			text:      strings.TrimSuffix(rawLine, "\n"),
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
			inputs = append(inputs, normalizeEmbeddingInput(chunk.Text))
		}
		indexMeter.Note(localize.T("progress.rag.embed_batch", localize.Data{"Index": calls + 1, "Total": totalBatches}))

		vectorsBatch, metrics, err := embedder.CreateEmbeddingsWithMetrics(ctx, inputs)
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
	vectors, metrics, err := embedder.CreateEmbeddingsWithMetrics(ctx, []string{normalizeEmbeddingInput(question)})
	if err != nil {
		return nil, llm.EmbeddingMetrics{}, fmt.Errorf("failed to embed query: %w", err)
	}
	if len(vectors) != 1 {
		return nil, llm.EmbeddingMetrics{}, fmt.Errorf("query embeddings count = %d, want 1", len(vectors))
	}
	return vectors[0], metrics, nil
}

func retrieve(index *Index, query []float32, question string, opts Options) []candidate {
	ranked := make([]candidate, 0, len(index.Chunks))
	for i, chunk := range index.Chunks {
		similarity := cosineSimilarity(query, index.Embeddings[i])
		ranked = append(ranked, candidate{
			Chunk:      chunk,
			Similarity: similarity,
			Overlap:    tokenOverlapRatio(question, chunk.Text),
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

func retrievalTooWeak(candidates []candidate) bool {
	best := candidates[0]
	return best.Similarity < 0.18 && best.Overlap < 0.1
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

func selectEvidence(ranked []candidate, finalK int) []candidate {
	selected := make([]candidate, 0, min(finalK, len(ranked)))
	for i, cand := range ranked {
		if len(selected) >= finalK {
			break
		}
		if isAdjacentToSelected(cand, selected) && len(ranked)-i > (finalK-len(selected)) {
			continue
		}
		selected = append(selected, cand)
	}
	if len(selected) < finalK {
		for _, cand := range ranked {
			if len(selected) >= finalK {
				break
			}
			if containsChunk(selected, cand.Chunk) {
				continue
			}
			selected = append(selected, cand)
		}
	}
	return selected
}

func containsChunk(selected []candidate, chunk Chunk) bool {
	for _, item := range selected {
		if item.Chunk.SourcePath == chunk.SourcePath && item.Chunk.ChunkID == chunk.ChunkID {
			return true
		}
	}
	return false
}

func isAdjacentToSelected(cand candidate, selected []candidate) bool {
	for _, item := range selected {
		if item.Chunk.SourcePath != cand.Chunk.SourcePath {
			continue
		}
		if abs(item.Chunk.ChunkID-cand.Chunk.ChunkID) <= 1 {
			return true
		}
	}
	return false
}

func synthesizeAnswer(ctx context.Context, chat llm.ChatRequester, question string, evidence []candidate) (string, llm.RequestMetrics, error) {
	contextBlocks := make([]string, 0, len(evidence))
	for i, cand := range evidence {
		contextBlocks = append(contextBlocks, fmt.Sprintf(
			"[source %d] %s:%d-%d\n%s",
			i+1,
			cand.Chunk.SourcePath,
			cand.Chunk.StartLine,
			cand.Chunk.EndLine,
			sanitizeUntrustedBlock(cand.Chunk.Text),
		))
	}

	req := openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    "system",
				Content: localize.T("rag.prompt.answer_system", localize.Data{"Policy": ragAntiInjectionPolicy()}),
			},
			{
				Role:    "user",
				Content: localize.T("rag.prompt.answer_user", localize.Data{"Question": question, "Sources": strings.Join(contextBlocks, "\n\n---\n\n")}),
			},
		},
	}

	resp, metrics, err := chat.SendRequestWithMetrics(ctx, req)
	if err != nil {
		return "", llm.RequestMetrics{}, fmt.Errorf("failed to synthesize rag answer: %w", err)
	}
	if resp == nil || len(resp.Choices) == 0 {
		return "", llm.RequestMetrics{}, errors.New("rag answer response has no choices")
	}
	answer := strings.TrimSpace(resp.Choices[0].Message.Content)
	if answer == "" {
		return "", llm.RequestMetrics{}, errors.New("rag answer response is empty")
	}
	return answer, metrics, nil
}

func appendSources(answer string, evidence []candidate) string {
	seen := make(map[string]struct{}, len(evidence))
	lines := make([]string, 0, len(evidence))
	for _, cand := range evidence {
		citation := fmt.Sprintf("%s:%d-%d", cand.Chunk.SourcePath, cand.Chunk.StartLine, cand.Chunk.EndLine)
		if _, exists := seen[citation]; exists {
			continue
		}
		seen[citation] = struct{}{}
		lines = append(lines, "- "+citation)
	}
	if len(lines) == 0 {
		lines = append(lines, "- none")
	}
	return strings.TrimSpace(answer) + "\n\nSources:\n" + strings.Join(lines, "\n")
}

func normalizeEmbeddingInput(raw string) string {
	return strings.Join(strings.Fields(raw), " ")
}

func cosineSimilarity(a, b []float32) float64 {
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

func tokenOverlapRatio(query, text string) float64 {
	queryTokens := tokenize(query)
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

func tokenize(input string) []string {
	raw := strings.FieldsFunc(strings.ToLower(input), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r)
	})
	seen := make(map[string]struct{}, len(raw))
	tokens := make([]string, 0, len(raw))
	for _, token := range raw {
		if len([]rune(token)) < 2 {
			continue
		}
		if _, exists := seen[token]; exists {
			continue
		}
		seen[token] = struct{}{}
		tokens = append(tokens, token)
	}
	return tokens
}

func sanitizeUntrustedBlock(raw string) string {
	lines := strings.Split(raw, "\n")
	safeLines := lines[:0]
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		lower := strings.ToLower(trimmed)
		if strings.HasPrefix(lower, "system:") || strings.HasPrefix(lower, "assistant:") || strings.HasPrefix(lower, "user:") {
			continue
		}
		if strings.Contains(lower, "ignore previous instructions") {
			continue
		}
		safeLines = append(safeLines, line)
	}
	return strings.TrimSpace(strings.Join(safeLines, "\n"))
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
