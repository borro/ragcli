package rag

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/borro/ragcli/internal/config"
	"github.com/borro/ragcli/internal/llm"
	openai "github.com/sashabaranov/go-openai"
)

type fakeEmbedder struct {
	calls       int
	totalTokens int
}

func (f *fakeEmbedder) CreateEmbeddingsWithMetrics(_ context.Context, inputs []string) (*llm.EmbeddingResult, error) {
	f.calls++
	vectors := make([][]float32, 0, len(inputs))
	for _, input := range inputs {
		vectors = append(vectors, vectorFor(input))
	}
	f.totalTokens += len(inputs) * 7
	return &llm.EmbeddingResult{
		Vectors: vectors,
		Metrics: llm.EmbeddingMetrics{
			Attempt:      1,
			InputCount:   len(inputs),
			VectorCount:  len(vectors),
			TotalTokens:  len(inputs) * 7,
			PromptTokens: len(inputs) * 7,
		},
	}, nil
}

type fakeChat struct {
	requests []llm.ChatCompletionRequest
	answer   string
}

func (f *fakeChat) SendRequestWithMetrics(_ context.Context, req llm.ChatCompletionRequest) (*llm.RequestResult, error) {
	f.requests = append(f.requests, req)
	return &llm.RequestResult{
		Response: &llm.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: llm.Message{
						Role:    "assistant",
						Content: f.answer,
					},
				},
			},
		},
		Metrics: llm.RequestMetrics{
			Attempt:          1,
			PromptTokens:     11,
			CompletionTokens: 5,
			TotalTokens:      16,
		},
	}, nil
}

func TestChunkTextPreservesLineRanges(t *testing.T) {
	content := []byte(strings.Join([]string{
		"line 1",
		"line 2 with more bytes",
		"line 3",
		"line 4",
		"line 5",
	}, "\n"))

	chunks := chunkText(content, "sample.txt", 32, 8)
	if len(chunks) < 2 {
		t.Fatalf("len(chunks) = %d, want >= 2", len(chunks))
	}
	if chunks[0].StartLine != 1 || chunks[0].EndLine < 2 {
		t.Fatalf("first chunk lines = %d-%d, want start at 1 and include line 2", chunks[0].StartLine, chunks[0].EndLine)
	}
	if chunks[1].StartLine <= chunks[0].StartLine {
		t.Fatalf("second chunk start = %d, want > first start %d", chunks[1].StartLine, chunks[0].StartLine)
	}
	if chunks[1].SourcePath != "sample.txt" {
		t.Fatalf("SourcePath = %q, want sample.txt", chunks[1].SourcePath)
	}
}

func TestChunkTextIgnoresTrailingNewlinePhantomLine(t *testing.T) {
	chunks := chunkText([]byte("one\ntwo\n"), "sample.txt", 64, 0)
	if len(chunks) != 1 {
		t.Fatalf("len(chunks) = %d, want 1", len(chunks))
	}
	if chunks[0].EndLine != 2 {
		t.Fatalf("EndLine = %d, want 2", chunks[0].EndLine)
	}
	if strings.HasSuffix(chunks[0].Text, "\n") {
		t.Fatalf("chunk text = %q, want no synthetic trailing empty line", chunks[0].Text)
	}
}

func TestBuildOrLoadIndexUsesCacheAndInvalidatesByModel(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{
		EmbeddingModel:  "embed-v1",
		RAGChunkSize:    40,
		RAGChunkOverlap: 8,
		RAGIndexTTL:     time.Hour,
		RAGIndexDir:     tempDir,
	}
	embedder := &fakeEmbedder{}
	source := Source{
		Path:        "/tmp/source.txt",
		DisplayName: "source.txt",
		Reader:      strings.NewReader("alpha\nretry policy is enabled\nbeta\n"),
	}

	stats := &pipelineStats{}
	_, err := buildOrLoadIndex(context.Background(), embedder, source, cfg, stats)
	if err != nil {
		t.Fatalf("buildOrLoadIndex() error = %v", err)
	}
	if embedder.calls == 0 {
		t.Fatal("expected embedding calls during initial build")
	}
	if stats.CacheHit {
		t.Fatal("CacheHit = true on initial build, want false")
	}

	stats = &pipelineStats{}
	_, err = buildOrLoadIndex(context.Background(), embedder, Source{
		Path:        "/tmp/source.txt",
		DisplayName: "source.txt",
		Reader:      strings.NewReader("alpha\nretry policy is enabled\nbeta\n"),
	}, cfg, stats)
	if err != nil {
		t.Fatalf("buildOrLoadIndex(cache) error = %v", err)
	}
	if !stats.CacheHit {
		t.Fatal("CacheHit = false, want true on repeated build")
	}
	callsAfterCache := embedder.calls

	cfg.EmbeddingModel = "embed-v2"
	stats = &pipelineStats{}
	_, err = buildOrLoadIndex(context.Background(), embedder, Source{
		Path:        "/tmp/source.txt",
		DisplayName: "source.txt",
		Reader:      strings.NewReader("alpha\nretry policy is enabled\nbeta\n"),
	}, cfg, stats)
	if err != nil {
		t.Fatalf("buildOrLoadIndex(model change) error = %v", err)
	}
	if embedder.calls <= callsAfterCache {
		t.Fatal("expected cache invalidation after embedding model change")
	}
}

func TestRunReturnsAnswerWithCitations(t *testing.T) {
	cfg := &config.Config{
		EmbeddingModel:  "embed",
		RAGTopK:         3,
		RAGFinalK:       2,
		RAGChunkSize:    55,
		RAGChunkOverlap: 10,
		RAGIndexTTL:     time.Hour,
		RAGIndexDir:     t.TempDir(),
		RAGRerank:       "heuristic",
	}
	embedder := &fakeEmbedder{}
	chat := &fakeChat{answer: "Retry policy is enabled for failed requests [source 1]."}
	content := strings.Join([]string{
		"overview",
		"retry policy is enabled for failed requests",
		"backoff starts at 1s",
		"database settings are described elsewhere",
		"totally unrelated section about metrics",
	}, "\n")

	answer, err := Run(context.Background(), chat, embedder, Source{
		Path:        "/tmp/retries.txt",
		DisplayName: "retries.txt",
		Reader:      strings.NewReader(content),
	}, cfg, "What is the retry policy?")
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if !strings.Contains(answer, "Sources:\n- retries.txt:") {
		t.Fatalf("answer = %q, want appended sources block", answer)
	}
	if len(chat.requests) != 1 {
		t.Fatalf("chat requests = %d, want 1", len(chat.requests))
	}
	userPrompt := chat.requests[0].Messages[1].Content
	if !strings.Contains(userPrompt, "retry policy is enabled") {
		t.Fatalf("user prompt = %q, want evidence chunk", userPrompt)
	}
	if strings.Contains(userPrompt, "totally unrelated section") {
		t.Fatalf("user prompt unexpectedly contains unrelated chunk: %q", userPrompt)
	}
}

func TestRunReturnsHonestInsufficientAnswer(t *testing.T) {
	cfg := &config.Config{
		EmbeddingModel:  "embed",
		RAGTopK:         3,
		RAGFinalK:       2,
		RAGChunkSize:    80,
		RAGChunkOverlap: 10,
		RAGIndexTTL:     time.Hour,
		RAGIndexDir:     t.TempDir(),
		RAGRerank:       "heuristic",
	}
	answer, err := Run(context.Background(), &fakeChat{answer: "should not be used"}, &fakeEmbedder{}, Source{
		DisplayName: "stdin",
		Reader:      strings.NewReader("alpha beta gamma\ndelta epsilon\n"),
	}, cfg, "payments and invoices")
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if !strings.Contains(answer, "Недостаточно данных") {
		t.Fatalf("answer = %q, want insufficient-data response", answer)
	}
	if !strings.Contains(answer, "Sources:\n- none") {
		t.Fatalf("answer = %q, want empty sources block", answer)
	}
}

func TestPersistedIndexFilesExist(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{
		EmbeddingModel:  "embed",
		RAGTopK:         2,
		RAGFinalK:       1,
		RAGChunkSize:    40,
		RAGChunkOverlap: 8,
		RAGIndexTTL:     time.Hour,
		RAGIndexDir:     tempDir,
		RAGRerank:       "off",
	}
	source := Source{
		Path:        "/tmp/source.txt",
		DisplayName: "source.txt",
		Reader:      strings.NewReader("one\ntwo\nthree\nfour\n"),
	}

	index, err := buildOrLoadIndex(context.Background(), &fakeEmbedder{}, source, cfg, &pipelineStats{})
	if err != nil {
		t.Fatalf("buildOrLoadIndex() error = %v", err)
	}

	hash := hashInput([]byte("one\ntwo\nthree\nfour\n"), "source.txt", cfg)
	indexDir := filepath.Join(tempDir, hash)
	for _, name := range []string{manifestFilename, chunksFilename, embeddingsFilename} {
		if _, err := os.Stat(filepath.Join(indexDir, name)); err != nil {
			t.Fatalf("os.Stat(%s) error = %v", name, err)
		}
	}
	if index.Manifest.ChunkCount == 0 {
		t.Fatal("manifest chunk count = 0, want persisted chunks")
	}
}

func TestPersistIndexUsesPrivatePermissions(t *testing.T) {
	tempDir := t.TempDir()
	indexDir := filepath.Join(tempDir, "index")
	index := &Index{
		Manifest: Manifest{
			SchemaVersion:  indexSchemaVersion,
			CreatedAt:      time.Now().UTC(),
			InputHash:      "hash",
			DocID:          "doc",
			SourcePath:     "source.txt",
			EmbeddingModel: "embed",
			ChunkSize:      100,
			ChunkOverlap:   10,
			ChunkCount:     1,
		},
		Chunks:     []Chunk{{DocID: "doc", SourcePath: "source.txt", ChunkID: 0, StartLine: 1, EndLine: 1, Text: "alpha"}},
		Embeddings: [][]float32{{1, 0, 0}},
	}

	if err := persistIndex(indexDir, index); err != nil {
		t.Fatalf("persistIndex() error = %v", err)
	}

	dirInfo, err := os.Stat(indexDir)
	if err != nil {
		t.Fatalf("os.Stat(indexDir) error = %v", err)
	}
	if dirInfo.Mode().Perm()&0o077 != 0 {
		t.Fatalf("index dir perms = %o, want private perms", dirInfo.Mode().Perm())
	}

	for _, name := range []string{manifestFilename, chunksFilename, embeddingsFilename} {
		info, err := os.Stat(filepath.Join(indexDir, name))
		if err != nil {
			t.Fatalf("os.Stat(%s) error = %v", name, err)
		}
		if info.Mode().Perm()&0o077 != 0 {
			t.Fatalf("%s perms = %o, want private perms", name, info.Mode().Perm())
		}
	}
}

func TestBuildOrLoadIndexConcurrentWritersSharePublishedIndex(t *testing.T) {
	cfg := &config.Config{
		EmbeddingModel:  "embed",
		RAGTopK:         2,
		RAGFinalK:       1,
		RAGChunkSize:    40,
		RAGChunkOverlap: 8,
		RAGIndexTTL:     time.Hour,
		RAGIndexDir:     t.TempDir(),
		RAGRerank:       "off",
	}

	embedder := &fakeEmbedder{}
	content := "one\nretry policy\nthree\n"
	var wg sync.WaitGroup
	errs := make(chan error, 2)

	for range 2 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := buildOrLoadIndex(context.Background(), embedder, Source{
				Path:        "/tmp/source.txt",
				DisplayName: "source.txt",
				Reader:      strings.NewReader(content),
			}, cfg, &pipelineStats{})
			errs <- err
		}()
	}

	wg.Wait()
	close(errs)

	for err := range errs {
		if err != nil {
			t.Fatalf("buildOrLoadIndex() concurrent error = %v", err)
		}
	}

	hash := hashInput([]byte(content), "source.txt", cfg)
	indexDir := filepath.Join(cfg.RAGIndexDir, hash)
	for _, name := range []string{manifestFilename, chunksFilename, embeddingsFilename} {
		if _, err := os.Stat(filepath.Join(indexDir, name)); err != nil {
			t.Fatalf("expected published index file %s: %v", name, err)
		}
	}
}

func vectorFor(input string) []float32 {
	lower := strings.ToLower(input)
	switch {
	case strings.Contains(lower, "retry"):
		return []float32{1, 0, 0}
	case strings.Contains(lower, "backoff"):
		return []float32{0.8, 0.1, 0}
	case strings.Contains(lower, "database"):
		return []float32{0, 1, 0}
	case strings.Contains(lower, "metrics"):
		return []float32{0, 0, 1}
	default:
		return []float32{0, 0, 0}
	}
}
