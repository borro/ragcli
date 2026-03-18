package ragcore

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
	"github.com/borro/ragcli/internal/input/testsource"
	"github.com/borro/ragcli/internal/llm"
	"github.com/borro/ragcli/internal/localize"
	"github.com/borro/ragcli/internal/retrieval"
	"github.com/borro/ragcli/internal/verbose"
)

type Options = RunOptions

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
	return testsource.DisplayName(s.kind, s.inputPath, s.snapshotPath, s.files)
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
		snapshotPath = testsource.WriteTempFile(reader, filepath.Base(path))
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

func directorySource(label string, path string, files []input.File) testSource {
	return testSource{
		kind:      input.KindDirectory,
		inputPath: path,
		files:     append([]input.File(nil), files...),
	}
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

func TestChunkTextSplitsOversizedSingleLine(t *testing.T) {
	line := strings.Repeat("a", 25)
	chunks := chunkText([]byte(line+"\n"), "sample.txt", 10, 0)
	if len(chunks) != 3 {
		t.Fatalf("len(chunks) = %d, want 3", len(chunks))
	}
	for i, chunk := range chunks {
		if len(chunk.Text) > 10 {
			t.Fatalf("chunk[%d].Text len = %d, want <= 10", i, len(chunk.Text))
		}
		if chunk.StartLine != 1 || chunk.EndLine != 1 {
			t.Fatalf("chunk[%d] lines = %d-%d, want 1-1", i, chunk.StartLine, chunk.EndLine)
		}
	}
	if got := chunks[0].Text + chunks[1].Text + chunks[2].Text; got != line {
		t.Fatalf("reconstructed line = %q, want %q", got, line)
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

func TestBuildOrLoadIndexRepairsCorruptedPublishedIndex(t *testing.T) {
	cfg := Options{
		EmbeddingModel: "embed",
		ChunkSize:      40,
		ChunkOverlap:   8,
		IndexTTL:       time.Hour,
		IndexDir:       t.TempDir(),
	}
	source := fileSource(strings.NewReader("alpha\nretry policy\nbeta\n"), "/tmp/source.txt", "source.txt")
	embedder := &fakeEmbedder{}

	index, err := buildOrLoadIndex(context.Background(), embedder, source, cfg, &pipelineStats{}, verbose.Meter{})
	if err != nil {
		t.Fatalf("buildOrLoadIndex(initial) error = %v", err)
	}

	indexDir := filepath.Join(cfg.IndexDir, index.Manifest.InputHash)
	if err := os.WriteFile(filepath.Join(indexDir, embeddingsFilename), []byte("{bad json\n"), 0o600); err != nil {
		t.Fatalf("WriteFile(embeddings) error = %v", err)
	}

	rebuilt, err := buildOrLoadIndex(context.Background(), embedder, source, cfg, &pipelineStats{}, verbose.Meter{})
	if err != nil {
		t.Fatalf("buildOrLoadIndex(repair) error = %v", err)
	}
	if embedder.calls < 2 {
		t.Fatalf("embedder.calls = %d, want rebuild after corruption", embedder.calls)
	}
	if rebuilt.Manifest.InputHash != index.Manifest.InputHash {
		t.Fatalf("rebuilt hash = %q, want %q", rebuilt.Manifest.InputHash, index.Manifest.InputHash)
	}
	if _, err := loadIndex(indexDir, time.Hour); err != nil {
		t.Fatalf("loadIndex(repaired) error = %v", err)
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

type sourceMetaStub struct {
	kind        input.Kind
	displayName string
	inputPath   string
}

func (s sourceMetaStub) Kind() input.Kind {
	return s.kind
}

func (s sourceMetaStub) InputPath() string {
	return s.inputPath
}

func (s sourceMetaStub) DisplayName() string {
	return s.displayName
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

func TestBuildOrLoadIndexFromFilesUsesCache(t *testing.T) {
	cfg := Options{
		EmbeddingModel: "embed",
		ChunkSize:      64,
		ChunkOverlap:   8,
		IndexTTL:       time.Hour,
		IndexDir:       t.TempDir(),
	}
	firstPath := testsource.WriteTempFile(strings.NewReader("retry policy enabled\n"), "a.txt")
	secondPath := testsource.WriteTempFile(strings.NewReader("database settings\n"), "b.txt")
	source := directorySource("corpus", "/tmp/corpus", []input.File{
		{Path: firstPath, DisplayPath: "a.txt"},
		{Path: secondPath, DisplayPath: "b.txt"},
	})
	embedder := &fakeEmbedder{}

	stats := &pipelineStats{}
	_, err := buildOrLoadIndexFromFiles(context.Background(), embedder, source, cfg, stats, verbose.Meter{})
	if err != nil {
		t.Fatalf("buildOrLoadIndexFromFiles() error = %v", err)
	}
	if stats.CacheHit {
		t.Fatal("CacheHit = true on initial build, want false")
	}
	callsAfterFirstBuild := embedder.calls

	stats = &pipelineStats{}
	_, err = buildOrLoadIndexFromFiles(context.Background(), embedder, source, cfg, stats, verbose.Meter{})
	if err != nil {
		t.Fatalf("buildOrLoadIndexFromFiles(cache) error = %v", err)
	}
	if !stats.CacheHit {
		t.Fatal("CacheHit = false on repeated build, want true")
	}
	if embedder.calls != callsAfterFirstBuild {
		t.Fatalf("embedder.calls = %d after cache hit, want %d", embedder.calls, callsAfterFirstBuild)
	}
}

func TestBuildOrLoadIndexFromFilesRejectsEmptyCorpus(t *testing.T) {
	cfg := Options{
		EmbeddingModel: "embed",
		ChunkSize:      64,
		ChunkOverlap:   8,
		IndexTTL:       time.Hour,
		IndexDir:       t.TempDir(),
	}

	_, err := buildOrLoadIndexFromFiles(context.Background(), &fakeEmbedder{}, directorySource("empty", "/tmp/empty", nil), cfg, &pipelineStats{}, verbose.Meter{})
	if err == nil || !strings.Contains(err.Error(), "input did not produce retrieval chunks") {
		t.Fatalf("buildOrLoadIndexFromFiles() error = %v, want empty corpus rejection", err)
	}
}

func TestNormalizeSourcePath(t *testing.T) {
	tests := []struct {
		name   string
		source input.SourceMeta
		want   string
	}{
		{
			name:   "display name wins",
			source: sourceMetaStub{kind: input.KindFile, displayName: "visible.txt", inputPath: "/tmp/hidden.txt"},
			want:   "visible.txt",
		},
		{
			name:   "input path fallback",
			source: sourceMetaStub{kind: input.KindFile, displayName: "   ", inputPath: "/tmp/a.txt"},
			want:   "/tmp/a.txt",
		},
		{
			name:   "stdin fallback",
			source: sourceMetaStub{kind: input.KindStdin},
			want:   "stdin",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := normalizeSourcePath(tt.source); got != tt.want {
				t.Fatalf("normalizeSourcePath() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestCmpChunkOrder(t *testing.T) {
	tests := []struct {
		name string
		a    Chunk
		b    Chunk
		want int
	}{
		{
			name: "path sorts first",
			a:    Chunk{SourcePath: "a.txt", ChunkID: 1},
			b:    Chunk{SourcePath: "b.txt", ChunkID: 0},
			want: -1,
		},
		{
			name: "chunk id breaks ties",
			a:    Chunk{SourcePath: "same.txt", ChunkID: 1},
			b:    Chunk{SourcePath: "same.txt", ChunkID: 2},
			want: -1,
		},
		{
			name: "equal chunks compare equal",
			a:    Chunk{SourcePath: "same.txt", ChunkID: 2},
			b:    Chunk{SourcePath: "same.txt", ChunkID: 2},
			want: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := cmpChunkOrder(tt.a, tt.b); got != tt.want {
				t.Fatalf("cmpChunkOrder() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestPreparedSearchFusedSearch(t *testing.T) {
	source := directorySource("corpus", "/tmp/corpus", []input.File{
		{Path: testsource.WriteTempFile(strings.NewReader("retry policy enabled\nbackoff is exponential\n"), "a.txt"), DisplayPath: "a.txt"},
		{Path: testsource.WriteTempFile(strings.NewReader("database connections stable\n"), "b.txt"), DisplayPath: "b.txt"},
	})
	searcher, stats, err := PrepareSearch(context.Background(), &fakeEmbedder{}, source, SearchOptions{
		TopK:           3,
		ChunkSize:      256,
		ChunkOverlap:   32,
		IndexTTL:       time.Hour,
		IndexDir:       t.TempDir(),
		Rerank:         "heuristic",
		EmbeddingModel: "embed",
	}, verbose.Meter{})
	if err != nil {
		t.Fatalf("PrepareSearch() error = %v", err)
	}
	if stats.ChunkCount == 0 {
		t.Fatal("ChunkCount = 0, want prepared index")
	}

	hits, fusedStats, err := searcher.FusedSearch(context.Background(), &fakeEmbedder{}, "What does it say about retry policy and backoff?", "")
	if err != nil {
		t.Fatalf("FusedSearch() error = %v", err)
	}
	if len(hits) == 0 {
		t.Fatal("len(hits) = 0, want fused matches")
	}
	if hits[0].Path != "a.txt" {
		t.Fatalf("top hit path = %q, want a.txt", hits[0].Path)
	}
	if fusedStats.EmbeddingCalls == 0 || fusedStats.EmbeddingTokens == 0 {
		t.Fatalf("fused stats = %+v, want query embedding work", fusedStats)
	}
}

func TestFusedSearchTooWeak(t *testing.T) {
	tests := []struct {
		name string
		hits []FusedSeedHit
		want bool
	}{
		{
			name: "empty hits are weak",
			want: true,
		},
		{
			name: "low similarity and overlap are weak",
			hits: []FusedSeedHit{{Similarity: 0.10, Overlap: 0.05}},
			want: true,
		},
		{
			name: "strong similarity is enough",
			hits: []FusedSeedHit{{Similarity: 0.20, Overlap: 0.00}},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := FusedSearchTooWeak(tt.hits); got != tt.want {
				t.Fatalf("FusedSearchTooWeak() = %v, want %v", got, tt.want)
			}
		})
	}
}
