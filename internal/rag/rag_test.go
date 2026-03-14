package rag

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/borro/ragcli/internal/input"
	"github.com/borro/ragcli/internal/llm"
	"github.com/borro/ragcli/internal/localize"
	"github.com/borro/ragcli/internal/retrieval"
	"github.com/borro/ragcli/internal/verbose"
	"github.com/borro/ragcli/internal/verbose/testutil"
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

type testSource struct {
	kind         input.Kind
	inputPath    string
	snapshotPath string
	files        []input.File
}

func (s testSource) Kind() input.Kind {
	return s.kind
}

func (s testSource) InputPath() string {
	return s.inputPath
}

func (s testSource) DisplayName() string {
	if path := strings.TrimSpace(s.inputPath); path != "" {
		return path
	}
	if s.kind == input.KindStdin {
		return "stdin"
	}
	if len(s.files) > 0 && strings.TrimSpace(s.files[0].DisplayPath) != "" {
		return s.files[0].DisplayPath
	}
	return strings.TrimSpace(s.snapshotPath)
}

func (s testSource) SnapshotPath() string {
	return s.snapshotPath
}

func (s testSource) Open() (io.ReadCloser, error) {
	if strings.TrimSpace(s.snapshotPath) == "" {
		return nil, os.ErrNotExist
	}
	return os.Open(s.snapshotPath)
}

func (s testSource) BackingFiles() []input.File {
	return append([]input.File(nil), s.files...)
}

func (s testSource) FileCount() int {
	return len(s.files)
}

func (s testSource) IsMultiFile() bool {
	return s.kind == input.KindDirectory || len(s.files) > 1
}

func fileSource(reader io.Reader, path string, label string) testSource {
	snapshotPath := ""
	files := []input.File(nil)
	if reader != nil {
		snapshotPath = writeSourceTempFile(reader, filepath.Base(path))
		files = []input.File{{
			Path:        snapshotPath,
			DisplayPath: filepath.Base(path),
		}}
	}
	return testSource{
		kind:         input.KindFile,
		inputPath:    path,
		snapshotPath: snapshotPath,
		files:        files,
	}
}

func stdinSource(reader io.Reader) testSource {
	snapshotPath := ""
	files := []input.File(nil)
	if reader != nil {
		snapshotPath = writeSourceTempFile(reader, "stdin.txt")
		files = []input.File{{
			Path:        snapshotPath,
			DisplayPath: "stdin",
		}}
	}
	return testSource{
		kind:         input.KindStdin,
		snapshotPath: snapshotPath,
		files:        files,
	}
}

func directorySource(label string, path string, files []input.File) testSource {
	return testSource{
		kind:      input.KindDirectory,
		inputPath: path,
		files:     append([]input.File(nil), files...),
	}
}

func writeSourceTempFile(reader io.Reader, name string) string {
	tmpFile, err := os.CreateTemp("", "rag-test-*"+filepath.Ext(name))
	if err != nil {
		panic(err)
	}
	if _, err := io.Copy(tmpFile, reader); err != nil {
		_ = tmpFile.Close()
		panic(err)
	}
	if err := tmpFile.Close(); err != nil {
		panic(err)
	}
	return tmpFile.Name()
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

func runRAG(ctx context.Context, chat llm.ChatRequester, embedder llm.EmbeddingRequester, source input.FileBackedSource, opts Options, question string, plan *verbose.Plan) (string, error) {
	return Run(ctx, chat, embedder, source, opts, question, plan)
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
	source := fileSource(strings.NewReader("alpha\nretry policy is enabled\nbeta\n"), "/tmp/source.txt", "source.txt")

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
	_, err = buildOrLoadIndex(context.Background(), embedder, fileSource(strings.NewReader("alpha\nretry policy is enabled\nbeta\n"), "/tmp/source.txt", "source.txt"), cfg, stats, verbose.Meter{})
	if err != nil {
		t.Fatalf("buildOrLoadIndex(cache) error = %v", err)
	}
	if !stats.CacheHit {
		t.Fatal("CacheHit = false, want true on repeated build")
	}
	callsAfterCache := embedder.calls

	cfg.EmbeddingModel = "embed-v2"
	stats = &pipelineStats{}
	_, err = buildOrLoadIndex(context.Background(), embedder, fileSource(strings.NewReader("alpha\nretry policy is enabled\nbeta\n"), "/tmp/source.txt", "source.txt"), cfg, stats, verbose.Meter{})
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

	answer, err := runRAG(context.Background(), chat, embedder, fileSource(strings.NewReader(content), "/tmp/retries.txt", "retries.txt"), cfg, "What is the retry policy?", nil)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if !strings.Contains(answer, "Sources:\n\n- /tmp/retries.txt:\n  ") {
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
	answer, err := runRAG(context.Background(), &fakeChat{answer: "should not be used"}, &fakeEmbedder{}, stdinSource(strings.NewReader("alpha beta gamma\ndelta epsilon\n")), cfg, "payments and invoices", nil)
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
			_, err := runRAG(context.Background(), &fakeChat{answer: "Retry policy is enabled [source 1]."}, &gatedEmbedder{
				started: started,
				release: release,
			}, fileSource(strings.NewReader(content), "/tmp/retries.txt", "retries.txt"), Options{
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

func TestRunProgressFromPlanDoesNotRequirePrepareStage(t *testing.T) {
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
	content := strings.Join([]string{
		"overview",
		"retry policy is enabled for failed requests",
		"backoff starts at 1s",
	}, "\n")
	plan := planWithoutPrepare(nil)
	recorder := testutil.NewProgressRecorder(plan.Total())
	plan = planWithoutPrepare(recorder)

	answer, err := runRAG(context.Background(), &fakeChat{answer: "Retry policy is enabled [source 1]."}, &fakeEmbedder{}, fileSource(strings.NewReader(content), "/tmp/retries.txt", "retries.txt"), cfg, "What is the retry policy?", plan)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if !strings.Contains(answer, "Retry policy is enabled") {
		t.Fatalf("answer = %q, want synthesized answer", answer)
	}

	_, currents, totals := recorder.Snapshot()
	if len(currents) == 0 {
		t.Fatal("progress recorder saw no updates")
	}
	if currents[len(currents)-1] != plan.Total() {
		t.Fatalf("final progress = %d, want %d", currents[len(currents)-1], plan.Total())
	}
	for _, total := range totals {
		if total != plan.Total() {
			t.Fatalf("reported total = %d, want %d", total, plan.Total())
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
	source := fileSource(strings.NewReader("one\ntwo\nthree\nfour\n"), "/tmp/source.txt", "source.txt")

	index, err := buildOrLoadIndex(context.Background(), &fakeEmbedder{}, source, cfg, &pipelineStats{}, verbose.Meter{})
	if err != nil {
		t.Fatalf("buildOrLoadIndex() error = %v", err)
	}

	indexDir := filepath.Join(tempDir, index.Manifest.InputHash)
	for _, name := range []string{manifestFilename, chunksFilename, embeddingsFilename} {
		if _, err := os.Stat(filepath.Join(indexDir, name)); err != nil {
			t.Fatalf("os.Stat(%s) error = %v", name, err)
		}
	}
	if index.Manifest.ItemCount == 0 {
		t.Fatal("manifest chunk count = 0, want persisted chunks")
	}
	for _, chunk := range index.Chunks {
		if chunk.DocID != index.Manifest.InputHash {
			t.Fatalf("chunk.DocID = %q, want manifest InputHash %q", chunk.DocID, index.Manifest.InputHash)
		}
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
			SourcePath:     "source.txt",
			EmbeddingModel: "embed",
			ChunkSize:      100,
			ChunkOverlap:   10,
			ItemCount:      1,
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

func TestLoadIndexRejectsOldSchemaVersion(t *testing.T) {
	indexDir := filepath.Join(t.TempDir(), "index")
	index := &Index{
		Manifest: Manifest{
			SchemaVersion:  indexSchemaVersion - 1,
			CreatedAt:      time.Now().UTC(),
			InputHash:      "hash",
			SourcePath:     "source.txt",
			EmbeddingModel: "embed",
			ChunkSize:      100,
			ChunkOverlap:   10,
			ItemCount:      1,
		},
		Chunks:     []Chunk{{DocID: "hash", SourcePath: "source.txt", ChunkID: 0, StartLine: 1, EndLine: 1, Text: "alpha"}},
		Embeddings: [][]float32{{1, 0, 0}},
	}

	if err := persistIndex(indexDir, index); err != nil {
		t.Fatalf("persistIndex() error = %v", err)
	}

	_, err := loadIndex(indexDir, time.Hour)
	if err == nil || !strings.Contains(err.Error(), "rag index schema mismatch") {
		t.Fatalf("loadIndex() error = %v, want rag schema mismatch", err)
	}
	if !errors.Is(err, retrieval.ErrIndexSchemaMismatch) {
		t.Fatalf("errors.Is(schema mismatch) = false for %v", err)
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
			_, err := buildOrLoadIndex(context.Background(), embedder, fileSource(strings.NewReader(content), "/tmp/source.txt", "source.txt"), cfg, &pipelineStats{}, verbose.Meter{})
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

	index, err := buildOrLoadIndex(context.Background(), embedder, fileSource(strings.NewReader(content), "/tmp/source.txt", "source.txt"), cfg, &pipelineStats{}, verbose.Meter{})
	if err != nil {
		t.Fatalf("buildOrLoadIndex() verification error = %v", err)
	}

	indexDir := filepath.Join(cfg.IndexDir, index.Manifest.InputHash)
	for _, name := range []string{manifestFilename, chunksFilename, embeddingsFilename} {
		if _, err := os.Stat(filepath.Join(indexDir, name)); err != nil {
			t.Fatalf("expected published index file %s: %v", name, err)
		}
	}
}

func TestNormalizeSourcePath(t *testing.T) {
	if got := normalizeSourcePath(stdinSource(nil)); got != "stdin" {
		t.Fatalf("normalizeSourcePath(stdin) = %q, want stdin", got)
	}
	if got := normalizeSourcePath(fileSource(nil, "/tmp/a.txt", "a.txt")); got != "/tmp/a.txt" {
		t.Fatalf("normalizeSourcePath(file) = %q, want /tmp/a.txt", got)
	}
}

func TestBuildOrLoadIndexFromMultipleFilesKeepsRelativePaths(t *testing.T) {
	cfg := Options{
		EmbeddingModel: "embed",
		TopK:           3,
		FinalK:         2,
		ChunkSize:      64,
		ChunkOverlap:   8,
		IndexTTL:       time.Hour,
		IndexDir:       t.TempDir(),
		Rerank:         "off",
	}
	firstPath := filepath.Join(t.TempDir(), "a.txt")
	if err := os.WriteFile(firstPath, []byte("retry policy\n"), 0o600); err != nil {
		t.Fatalf("WriteFile(a.txt) error = %v", err)
	}
	secondDir := t.TempDir()
	secondPath := filepath.Join(secondDir, "nested-b.txt")
	if err := os.WriteFile(secondPath, []byte("metrics line\n"), 0o600); err != nil {
		t.Fatalf("WriteFile(nested-b.txt) error = %v", err)
	}

	index, err := buildOrLoadIndex(context.Background(), &fakeEmbedder{}, directorySource("corpus", "/tmp/corpus", []input.File{
		{Path: firstPath, DisplayPath: "a.txt"},
		{Path: secondPath, DisplayPath: "nested/b.txt"},
	}), cfg, &pipelineStats{}, verbose.Meter{})
	if err != nil {
		t.Fatalf("buildOrLoadIndex() error = %v", err)
	}
	if len(index.Chunks) < 2 {
		t.Fatalf("len(Chunks) = %d, want >= 2", len(index.Chunks))
	}
	seen := make(map[string]struct{})
	for _, chunk := range index.Chunks {
		seen[chunk.SourcePath] = struct{}{}
	}
	for _, want := range []string{"a.txt", "nested/b.txt"} {
		if _, ok := seen[want]; !ok {
			t.Fatalf("chunks missing source path %q: %+v", want, index.Chunks)
		}
	}
}

func TestEmbedQuery(t *testing.T) {
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
	if strings.Count(answer, "- a.txt:\n  1-2") != 1 {
		t.Fatalf("appendSources() = %q, want deduped citation", answer)
	}
	if !strings.Contains(answer, "- b.txt:\n  1") {
		t.Fatalf("appendSources() = %q, want second source", answer)
	}
	if got := appendSources("Answer", nil); !strings.Contains(got, "Sources:\n\n- none") {
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

func planWithoutPrepare(reporter verbose.Reporter) *verbose.Plan {
	return verbose.NewPlan(reporter, "rag",
		verbose.StageDef{Key: "index", Label: "index", Slots: 8},
		verbose.StageDef{Key: "retrieval", Label: "retrieval", Slots: 7},
		verbose.StageDef{Key: "final", Label: "final", Slots: 7},
	)
}
