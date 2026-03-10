package rag

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/borro/ragcli/internal/llm"
	"github.com/borro/ragcli/internal/localize"
	"github.com/borro/ragcli/internal/testutil"
	"github.com/borro/ragcli/internal/verbose"
	openai "github.com/sashabaranov/go-openai"
)

func setRussianLocale(t *testing.T) {
	t.Helper()
	if err := localize.SetCurrent(localize.RU); err != nil {
		t.Fatalf("SetCurrent(ru) error = %v", err)
	}
}

func TestMain(m *testing.M) {
	if err := localize.SetCurrent(localize.RU); err != nil {
		panic(err)
	}
	os.Exit(m.Run())
}

type fakeEmbedder struct {
	mu          sync.Mutex
	calls       int
	totalTokens int
}

func (f *fakeEmbedder) CreateEmbeddingsWithMetrics(_ context.Context, inputs []string) ([][]float32, llm.EmbeddingMetrics, error) {
	f.mu.Lock()
	f.calls++
	f.totalTokens += len(inputs) * 7
	f.mu.Unlock()

	vectors := make([][]float32, 0, len(inputs))
	for _, input := range inputs {
		vectors = append(vectors, vectorFor(input))
	}
	return vectors, llm.EmbeddingMetrics{
		Attempt:      1,
		InputCount:   len(inputs),
		VectorCount:  len(vectors),
		TotalTokens:  len(inputs) * 7,
		PromptTokens: len(inputs) * 7,
	}, nil
}

type fakeChat struct {
	mu       sync.Mutex
	requests []openai.ChatCompletionRequest
	answer   string
}

func (f *fakeChat) SendRequestWithMetrics(_ context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, llm.RequestMetrics, error) {
	f.mu.Lock()
	f.requests = append(f.requests, req)
	answer := f.answer
	f.mu.Unlock()

	return &openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Role:    "assistant",
						Content: answer,
					},
				},
			},
		}, llm.RequestMetrics{
			Attempt:          1,
			PromptTokens:     11,
			CompletionTokens: 5,
			TotalTokens:      16,
		}, nil
}

type gatedEmbedder struct {
	started chan<- string
	release <-chan struct{}
}

func (g *gatedEmbedder) CreateEmbeddingsWithMetrics(_ context.Context, inputs []string) ([][]float32, llm.EmbeddingMetrics, error) {
	select {
	case g.started <- "embedding":
	default:
	}
	<-g.release

	vectors := make([][]float32, 0, len(inputs))
	for _, input := range inputs {
		vectors = append(vectors, vectorFor(input))
	}
	return vectors, llm.EmbeddingMetrics{
		Attempt:      1,
		InputCount:   len(inputs),
		VectorCount:  len(vectors),
		TotalTokens:  len(inputs) * 7,
		PromptTokens: len(inputs) * 7,
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
	cfg := Options{
		EmbeddingModel: "embed-v1",
		ChunkSize:      40,
		ChunkOverlap:   8,
		IndexTTL:       time.Hour,
		IndexDir:       tempDir,
	}
	embedder := &fakeEmbedder{}
	source := Source{
		Path:        "/tmp/source.txt",
		DisplayName: "source.txt",
		Reader:      strings.NewReader("alpha\nretry policy is enabled\nbeta\n"),
	}

	stats := &pipelineStats{}
	_, err := buildOrLoadIndex(context.Background(), embedder, source, cfg, stats, verbose.Meter{})
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
	}, cfg, stats, verbose.Meter{})
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
	}, cfg, stats, verbose.Meter{})
	if err != nil {
		t.Fatalf("buildOrLoadIndex(model change) error = %v", err)
	}
	if embedder.calls <= callsAfterCache {
		t.Fatal("expected cache invalidation after embedding model change")
	}
}

func TestRunReturnsAnswerWithCitations(t *testing.T) {
	cfg := Options{
		EmbeddingModel: "embed",
		TopK:           3,
		FinalK:         2,
		ChunkSize:      55,
		ChunkOverlap:   10,
		IndexTTL:       time.Hour,
		IndexDir:       t.TempDir(),
		Rerank:         "heuristic",
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
	}, cfg, "What is the retry policy?", nil)
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
	setRussianLocale(t)
	cfg := Options{
		EmbeddingModel: "embed",
		TopK:           3,
		FinalK:         2,
		ChunkSize:      80,
		ChunkOverlap:   10,
		IndexTTL:       time.Hour,
		IndexDir:       t.TempDir(),
		Rerank:         "heuristic",
	}
	answer, err := Run(context.Background(), &fakeChat{answer: "should not be used"}, &fakeEmbedder{}, Source{
		DisplayName: "stdin",
		Reader:      strings.NewReader("alpha beta gamma\ndelta epsilon\n"),
	}, cfg, "payments and invoices", nil)
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

func TestRunVerboseProgressIsIsolatedPerRun(t *testing.T) {
	content := strings.Join([]string{
		"overview",
		"retry policy is enabled for failed requests",
		"backoff starts at 1s",
		"database settings are described elsewhere",
	}, "\n")

	started := make(chan string, 8)
	release := make(chan struct{})
	runOnce := func(t *testing.T, index int, recorder *testutil.ProgressRecorder) <-chan error {
		t.Helper()

		plan := verbose.NewPlan(recorder, "rag",
			verbose.StageDef{Key: "prepare", Label: "подготовка", Slots: 2},
			verbose.StageDef{Key: "index", Label: "индекс", Slots: 8},
			verbose.StageDef{Key: "retrieval", Label: "retrieval", Slots: 7},
			verbose.StageDef{Key: "final", Label: "финальный ответ", Slots: 7},
		)
		errCh := make(chan error, 1)
		go func() {
			_, err := Run(context.Background(), &fakeChat{answer: "Retry policy is enabled [source 1]."}, &gatedEmbedder{
				started: started,
				release: release,
			}, Source{
				Path:        "/tmp/retries.txt",
				DisplayName: "retries.txt",
				Reader:      strings.NewReader(content),
			}, Options{
				EmbeddingModel: "embed",
				TopK:           3,
				FinalK:         2,
				ChunkSize:      55,
				ChunkOverlap:   10,
				IndexTTL:       time.Hour,
				IndexDir:       filepath.Join(t.TempDir(), "idx", string(rune('a'+index))),
				Rerank:         "heuristic",
			}, "What is the retry policy?", plan)
			errCh <- err
		}()
		return errCh
	}

	expectedTotal := verbose.NewPlan(nil, "rag",
		verbose.StageDef{Key: "prepare", Label: "подготовка", Slots: 2},
		verbose.StageDef{Key: "index", Label: "индекс", Slots: 8},
		verbose.StageDef{Key: "retrieval", Label: "retrieval", Slots: 7},
		verbose.StageDef{Key: "final", Label: "финальный ответ", Slots: 7},
	).Total()
	first := testutil.NewProgressRecorder(expectedTotal)
	second := testutil.NewProgressRecorder(expectedTotal)
	errFirst := runOnce(t, 0, first)
	errSecond := runOnce(t, 1, second)

	for i := 0; i < 2; i++ {
		select {
		case <-started:
		case <-time.After(2 * time.Second):
			t.Fatal("timed out waiting for both runs to start embeddings")
		}
	}
	close(release)

	if err := <-errFirst; err != nil {
		t.Fatalf("first Run() error = %v", err)
	}
	if err := <-errSecond; err != nil {
		t.Fatalf("second Run() error = %v", err)
	}

	for _, tc := range []struct {
		name     string
		recorder *testutil.ProgressRecorder
	}{
		{name: "first", recorder: first},
		{name: "second", recorder: second},
	} {
		progressCalls, _, totals := tc.recorder.Snapshot()
		if progressCalls == 0 {
			t.Fatalf("%s recorder saw no progress updates", tc.name)
		}
		for _, total := range totals {
			if total != expectedTotal {
				t.Fatalf("%s recorder total = %d, want %d", tc.name, total, expectedTotal)
			}
		}
	}
}

func TestPersistedIndexFilesExist(t *testing.T) {
	tempDir := t.TempDir()
	cfg := Options{
		EmbeddingModel: "embed",
		TopK:           2,
		FinalK:         1,
		ChunkSize:      40,
		ChunkOverlap:   8,
		IndexTTL:       time.Hour,
		IndexDir:       tempDir,
		Rerank:         "off",
	}
	source := Source{
		Path:        "/tmp/source.txt",
		DisplayName: "source.txt",
		Reader:      strings.NewReader("one\ntwo\nthree\nfour\n"),
	}

	index, err := buildOrLoadIndex(context.Background(), &fakeEmbedder{}, source, cfg, &pipelineStats{}, verbose.Meter{})
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
	cfg := Options{
		EmbeddingModel: "embed",
		TopK:           2,
		FinalK:         1,
		ChunkSize:      40,
		ChunkOverlap:   8,
		IndexTTL:       time.Hour,
		IndexDir:       t.TempDir(),
		Rerank:         "off",
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
			}, cfg, &pipelineStats{}, verbose.Meter{})
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
	indexDir := filepath.Join(cfg.IndexDir, hash)
	for _, name := range []string{manifestFilename, chunksFilename, embeddingsFilename} {
		if _, err := os.Stat(filepath.Join(indexDir, name)); err != nil {
			t.Fatalf("expected published index file %s: %v", name, err)
		}
	}
}

func TestNormalizeSourcePath(t *testing.T) {
	if got := normalizeSourcePath(Source{DisplayName: "stdin"}); got != "stdin" {
		t.Fatalf("normalizeSourcePath(stdin) = %q, want stdin", got)
	}
	if got := normalizeSourcePath(Source{Path: "/tmp/a.txt", DisplayName: "a.txt"}); got != "a.txt" {
		t.Fatalf("normalizeSourcePath(file) = %q, want a.txt", got)
	}
}

func TestEmbedQueryAndHelpers(t *testing.T) {
	vector, metrics, err := embedQuery(context.Background(), &fakeEmbedder{}, " retry\n policy ")
	if err != nil {
		t.Fatalf("embedQuery() error = %v", err)
	}
	if len(vector) == 0 {
		t.Fatal("embedQuery() vector empty, want non-empty")
	}
	if metrics.InputCount != 1 {
		t.Fatalf("metrics.InputCount = %d, want 1", metrics.InputCount)
	}
	if got := normalizeEmbeddingInput(" a \n b\tc "); got != "a b c" {
		t.Fatalf("normalizeEmbeddingInput() = %q, want %q", got, "a b c")
	}
}

func TestEmbedQueryErrors(t *testing.T) {
	embedderErr := embeddingRequesterFunc(func(_ context.Context, _ []string) ([][]float32, llm.EmbeddingMetrics, error) {
		return nil, llm.EmbeddingMetrics{}, errors.New("embed failed")
	})
	_, _, err := embedQuery(context.Background(), embedderErr, "question")
	if err == nil || !strings.Contains(err.Error(), "failed to embed query") {
		t.Fatalf("embedQuery() error = %v, want wrapped embed error", err)
	}

	badCount := embeddingRequesterFunc(func(_ context.Context, _ []string) ([][]float32, llm.EmbeddingMetrics, error) {
		return [][]float32{{1}, {2}}, llm.EmbeddingMetrics{}, nil
	})
	_, _, err = embedQuery(context.Background(), badCount, "question")
	if err == nil || !strings.Contains(err.Error(), "query embeddings count = 2, want 1") {
		t.Fatalf("embedQuery() error = %v, want count mismatch", err)
	}
}

func TestSelectionHelpersAndAppendSources(t *testing.T) {
	a0 := candidate{Chunk: Chunk{SourcePath: "a.txt", ChunkID: 0, StartLine: 1, EndLine: 2}, Score: 0.9}
	a1 := candidate{Chunk: Chunk{SourcePath: "a.txt", ChunkID: 1, StartLine: 3, EndLine: 4}, Score: 0.8}
	a3 := candidate{Chunk: Chunk{SourcePath: "a.txt", ChunkID: 3, StartLine: 9, EndLine: 10}, Score: 0.7}
	b0 := candidate{Chunk: Chunk{SourcePath: "b.txt", ChunkID: 0, StartLine: 1, EndLine: 1}, Score: 0.6}

	selected := selectEvidence([]candidate{a0, a1, a3}, 2)
	if len(selected) != 2 {
		t.Fatalf("len(selected) = %d, want 2", len(selected))
	}
	if selected[0].Chunk.ChunkID != 0 || selected[1].Chunk.ChunkID != 3 {
		t.Fatalf("selected chunk IDs = %d,%d, want 0,3", selected[0].Chunk.ChunkID, selected[1].Chunk.ChunkID)
	}

	fallback := selectEvidence([]candidate{a0, a1}, 2)
	if len(fallback) != 2 {
		t.Fatalf("len(fallback) = %d, want 2", len(fallback))
	}
	if fallback[1].Chunk.ChunkID != 1 {
		t.Fatalf("fallback second chunk = %d, want 1", fallback[1].Chunk.ChunkID)
	}

	if !containsChunk(selected, a0.Chunk) {
		t.Fatal("containsChunk() = false, want true")
	}
	if containsChunk(selected, b0.Chunk) {
		t.Fatal("containsChunk() = true, want false")
	}
	if got := cmpChunkOrder(a0.Chunk, b0.Chunk); got >= 0 {
		t.Fatalf("cmpChunkOrder(a,b) = %d, want < 0", got)
	}
	if got := cmpChunkOrder(a1.Chunk, a0.Chunk); got <= 0 {
		t.Fatalf("cmpChunkOrder(a1,a0) = %d, want > 0", got)
	}

	answer := appendSources("Answer", []candidate{a0, a0, b0})
	if strings.Count(answer, "- a.txt:1-2") != 1 {
		t.Fatalf("appendSources() = %q, want deduped citation", answer)
	}
	if !strings.Contains(answer, "- b.txt:1-1") {
		t.Fatalf("appendSources() = %q, want second source", answer)
	}
	if got := appendSources("Answer", nil); !strings.Contains(got, "Sources:\n- none") {
		t.Fatalf("appendSources(nil) = %q, want none source", got)
	}
}

type embeddingRequesterFunc func(context.Context, []string) ([][]float32, llm.EmbeddingMetrics, error)

func (f embeddingRequesterFunc) CreateEmbeddingsWithMetrics(ctx context.Context, inputs []string) ([][]float32, llm.EmbeddingMetrics, error) {
	return f(ctx, inputs)
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
