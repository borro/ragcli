package retrieval

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/borro/ragcli/internal/llm"
)

type embeddingRequesterFunc func(context.Context, []string) ([][]float32, llm.EmbeddingMetrics, error)

func (f embeddingRequesterFunc) CreateEmbeddingsWithMetrics(ctx context.Context, inputs []string) ([][]float32, llm.EmbeddingMetrics, error) {
	return f(ctx, inputs)
}

func TestWalkLinesTracksOffsets(t *testing.T) {
	var got []struct {
		raw       string
		text      string
		lineNo    int
		byteStart int
		byteEnd   int
	}

	written, err := WalkLines(strings.NewReader("alpha\nbeta"), func(raw, text string, lineNo, byteStart, byteEnd int) error {
		got = append(got, struct {
			raw       string
			text      string
			lineNo    int
			byteStart int
			byteEnd   int
		}{raw, text, lineNo, byteStart, byteEnd})
		return nil
	})
	if err != nil {
		t.Fatalf("WalkLines() error = %v", err)
	}
	if written != len("alpha\nbeta") {
		t.Fatalf("written = %d, want %d", written, len("alpha\nbeta"))
	}
	if len(got) != 2 {
		t.Fatalf("len(got) = %d, want 2", len(got))
	}
	if got[0].text != "alpha" || got[0].lineNo != 1 || got[0].byteStart != 0 || got[0].byteEnd != 6 {
		t.Fatalf("first line = %+v, want alpha line 1 offsets 0..6", got[0])
	}
	if got[1].text != "beta" || got[1].lineNo != 2 || got[1].byteStart != 6 || got[1].byteEnd != 10 {
		t.Fatalf("second line = %+v, want beta line 2 offsets 6..10", got[1])
	}
}

func TestSpoolSourceProducesStableHashAndBytes(t *testing.T) {
	first, err := SpoolSource(strings.NewReader("one\ntwo\n"), "retrieval-test-*", []string{"source.txt", "1"}, nil)
	if err != nil {
		t.Fatalf("SpoolSource(first) error = %v", err)
	}
	defer func() { _ = os.Remove(first.TempPath) }()

	second, err := SpoolSource(strings.NewReader("one\ntwo\n"), "retrieval-test-*", []string{"source.txt", "1"}, nil)
	if err != nil {
		t.Fatalf("SpoolSource(second) error = %v", err)
	}
	defer func() { _ = os.Remove(second.TempPath) }()

	if first.Bytes != len("one\ntwo\n") {
		t.Fatalf("Bytes = %d, want %d", first.Bytes, len("one\ntwo\n"))
	}
	if first.Hash != second.Hash {
		t.Fatalf("Hash mismatch: %q != %q", first.Hash, second.Hash)
	}
}

func TestPersistAndLoadIndexRoundTrip(t *testing.T) {
	const schemaVersion = 2

	type item struct {
		ID   int    `json:"id"`
		Text string `json:"text"`
	}

	indexDir := filepath.Join(t.TempDir(), "index")
	manifest := Manifest{
		SchemaVersion:  schemaVersion,
		CreatedAt:      time.Now().UTC(),
		InputHash:      "hash",
		SourcePath:     "source.txt",
		EmbeddingModel: "embed",
		ChunkSize:      128,
		ChunkOverlap:   16,
		ItemCount:      1,
	}

	err := PersistIndex(indexDir, manifest, []item{{ID: 1, Text: "alpha"}}, [][]float32{{1, 0, 0}}, Filenames{
		Manifest:   "manifest.json",
		Items:      "items.jsonl",
		Embeddings: "embeddings.jsonl",
	})
	if err != nil {
		t.Fatalf("PersistIndex() error = %v", err)
	}

	loadedManifest, items, embeddings, err := LoadIndex[item](indexDir, time.Hour, schemaVersion, Filenames{
		Manifest:   "manifest.json",
		Items:      "items.jsonl",
		Embeddings: "embeddings.jsonl",
	})
	if err != nil {
		t.Fatalf("LoadIndex() error = %v", err)
	}
	if loadedManifest.InputHash != manifest.InputHash {
		t.Fatalf("loadedManifest.InputHash = %q, want %q", loadedManifest.InputHash, manifest.InputHash)
	}
	if len(items) != 1 || items[0].Text != "alpha" {
		t.Fatalf("items = %+v, want one alpha item", items)
	}
	if len(embeddings) != 1 || len(embeddings[0]) != 3 {
		t.Fatalf("embeddings = %+v, want one vector", embeddings)
	}
}

func TestLoadIndexRejectsSchemaAndTTL(t *testing.T) {
	const schemaVersion = 2

	indexDir := filepath.Join(t.TempDir(), "index")
	manifest := Manifest{
		SchemaVersion:  schemaVersion,
		CreatedAt:      time.Now().UTC().Add(-2 * time.Hour),
		InputHash:      "hash",
		SourcePath:     "source.txt",
		EmbeddingModel: "embed",
		ChunkSize:      128,
		ChunkOverlap:   16,
		ItemCount:      1,
	}

	err := PersistIndex(indexDir, manifest, []string{"alpha"}, [][]float32{{1, 0, 0}}, Filenames{
		Manifest:   "manifest.json",
		Items:      "items.jsonl",
		Embeddings: "embeddings.jsonl",
	})
	if err != nil {
		t.Fatalf("PersistIndex() error = %v", err)
	}

	_, _, _, err = LoadIndex[string](indexDir, time.Hour, schemaVersion+1, Filenames{
		Manifest:   "manifest.json",
		Items:      "items.jsonl",
		Embeddings: "embeddings.jsonl",
	})
	if err == nil || !strings.Contains(err.Error(), "index schema mismatch") {
		t.Fatalf("LoadIndex(schema) error = %v, want schema mismatch", err)
	}

	_, _, _, err = LoadIndex[string](indexDir, time.Minute, schemaVersion, Filenames{
		Manifest:   "manifest.json",
		Items:      "items.jsonl",
		Embeddings: "embeddings.jsonl",
	})
	if err == nil || !strings.Contains(err.Error(), "index ttl expired") {
		t.Fatalf("LoadIndex(ttl) error = %v, want ttl expired", err)
	}
}

func TestLoadIndexReportsContextualReadErrors(t *testing.T) {
	const schemaVersion = 2

	filenames := Filenames{
		Manifest:   "manifest.json",
		Items:      "items.jsonl",
		Embeddings: "embeddings.jsonl",
	}

	t.Run("manifest read", func(t *testing.T) {
		_, _, _, err := LoadIndex[string](t.TempDir(), time.Hour, schemaVersion, filenames)
		if err == nil || !strings.Contains(err.Error(), "failed to read manifest file") {
			t.Fatalf("LoadIndex(manifest read) error = %v, want manifest read context", err)
		}
	})

	t.Run("manifest decode", func(t *testing.T) {
		indexDir := t.TempDir()
		writeTestFile(t, filepath.Join(indexDir, filenames.Manifest), "{bad json")

		_, _, _, err := LoadIndex[string](indexDir, time.Hour, schemaVersion, filenames)
		if err == nil || !strings.Contains(err.Error(), "failed to decode manifest") {
			t.Fatalf("LoadIndex(manifest decode) error = %v, want manifest decode context", err)
		}
	})

	t.Run("items decode", func(t *testing.T) {
		indexDir := t.TempDir()
		writeManifest(t, filepath.Join(indexDir, filenames.Manifest), Manifest{
			SchemaVersion:  schemaVersion,
			CreatedAt:      time.Now().UTC(),
			InputHash:      "hash",
			SourcePath:     "source.txt",
			EmbeddingModel: "embed",
			ChunkSize:      128,
			ChunkOverlap:   16,
			ItemCount:      1,
		})
		writeTestFile(t, filepath.Join(indexDir, filenames.Items), "{bad json\n")
		writeTestFile(t, filepath.Join(indexDir, filenames.Embeddings), "[1,0,0]\n")

		_, _, _, err := LoadIndex[string](indexDir, time.Hour, schemaVersion, filenames)
		if err == nil || !strings.Contains(err.Error(), "failed to decode items entry") {
			t.Fatalf("LoadIndex(items decode) error = %v, want items decode context", err)
		}
	})

	t.Run("embeddings decode", func(t *testing.T) {
		indexDir := t.TempDir()
		writeManifest(t, filepath.Join(indexDir, filenames.Manifest), Manifest{
			SchemaVersion:  schemaVersion,
			CreatedAt:      time.Now().UTC(),
			InputHash:      "hash",
			SourcePath:     "source.txt",
			EmbeddingModel: "embed",
			ChunkSize:      128,
			ChunkOverlap:   16,
			ItemCount:      1,
		})
		writeTestFile(t, filepath.Join(indexDir, filenames.Items), "\"alpha\"\n")
		writeTestFile(t, filepath.Join(indexDir, filenames.Embeddings), "{bad json\n")

		_, _, _, err := LoadIndex[string](indexDir, time.Hour, schemaVersion, filenames)
		if err == nil || !strings.Contains(err.Error(), "failed to decode embeddings entry") {
			t.Fatalf("LoadIndex(embeddings decode) error = %v, want embeddings decode context", err)
		}
	})
}

func TestPersistIndexReportsContextualWriteErrors(t *testing.T) {
	manifest := Manifest{
		SchemaVersion:  2,
		CreatedAt:      time.Now().UTC(),
		InputHash:      "hash",
		SourcePath:     "source.txt",
		EmbeddingModel: "embed",
		ChunkSize:      128,
		ChunkOverlap:   16,
		ItemCount:      1,
	}

	testCases := []struct {
		name      string
		filenames Filenames
		want      string
	}{
		{
			name: "items create",
			filenames: Filenames{
				Manifest:   "manifest.json",
				Items:      ".",
				Embeddings: "embeddings.jsonl",
			},
			want: "failed to create items file",
		},
		{
			name: "embeddings create",
			filenames: Filenames{
				Manifest:   "manifest.json",
				Items:      "items.jsonl",
				Embeddings: ".",
			},
			want: "failed to create embeddings file",
		},
		{
			name: "manifest create",
			filenames: Filenames{
				Manifest:   ".",
				Items:      "items.jsonl",
				Embeddings: "embeddings.jsonl",
			},
			want: "failed to create manifest file",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			indexDir := filepath.Join(t.TempDir(), "index")
			err := PersistIndex(indexDir, manifest, []string{"alpha"}, [][]float32{{1, 0, 0}}, tc.filenames)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("PersistIndex() error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestEmbedQuery(t *testing.T) {
	vector, metrics, err := EmbedQuery(context.Background(), embeddingRequesterFunc(func(_ context.Context, inputs []string) ([][]float32, llm.EmbeddingMetrics, error) {
		if len(inputs) != 1 || inputs[0] != "retry policy" {
			t.Fatalf("inputs = %#v, want normalized single input", inputs)
		}
		return [][]float32{{1, 2}}, llm.EmbeddingMetrics{InputCount: 1}, nil
	}), " retry\npolicy ", "query")
	if err != nil {
		t.Fatalf("EmbedQuery() error = %v", err)
	}
	if len(vector) != 2 || metrics.InputCount != 1 {
		t.Fatalf("vector/metrics = %#v %#v, want one vector and InputCount 1", vector, metrics)
	}
}

func TestEmbedQueryErrors(t *testing.T) {
	_, _, err := EmbedQuery(context.Background(), embeddingRequesterFunc(func(_ context.Context, _ []string) ([][]float32, llm.EmbeddingMetrics, error) {
		return nil, llm.EmbeddingMetrics{}, errors.New("boom")
	}), "question", "query")
	if err == nil || !strings.Contains(err.Error(), "failed to embed query") {
		t.Fatalf("EmbedQuery(error) = %v, want wrapped error", err)
	}

	_, _, err = EmbedQuery(context.Background(), embeddingRequesterFunc(func(_ context.Context, _ []string) ([][]float32, llm.EmbeddingMetrics, error) {
		return [][]float32{{1}, {2}}, llm.EmbeddingMetrics{}, nil
	}), "question", "query")
	if err == nil || !strings.Contains(err.Error(), "query embeddings count = 2, want 1") {
		t.Fatalf("EmbedQuery(count) = %v, want count mismatch", err)
	}
}

func TestEmbedTextsNormalizesInputs(t *testing.T) {
	vectors, metrics, err := EmbedTexts(context.Background(), embeddingRequesterFunc(func(_ context.Context, inputs []string) ([][]float32, llm.EmbeddingMetrics, error) {
		if len(inputs) != 2 || inputs[0] != "retry policy" || inputs[1] != "backoff rules" {
			t.Fatalf("inputs = %#v, want normalized inputs", inputs)
		}
		return [][]float32{{1}, {2}}, llm.EmbeddingMetrics{InputCount: 2}, nil
	}), []string{" retry\npolicy ", "backoff\t rules"})
	if err != nil {
		t.Fatalf("EmbedTexts() error = %v", err)
	}
	if len(vectors) != 2 || metrics.InputCount != 2 {
		t.Fatalf("vectors/metrics = %#v %#v, want two vectors and InputCount 2", vectors, metrics)
	}
}

func TestEmbedTextsRejectsCountMismatch(t *testing.T) {
	_, _, err := EmbedTexts(context.Background(), embeddingRequesterFunc(func(_ context.Context, _ []string) ([][]float32, llm.EmbeddingMetrics, error) {
		return [][]float32{{1}}, llm.EmbeddingMetrics{}, nil
	}), []string{"alpha", "beta"})
	if err == nil || !strings.Contains(err.Error(), "embedding count mismatch: got 1, want 2") {
		t.Fatalf("EmbedTexts() error = %v, want count mismatch", err)
	}
}

func TestTokenizeSkipsShortAndDuplicateTokens(t *testing.T) {
	got := Tokenize("A a alpha, beta beta x")
	want := []string{"alpha", "beta"}
	if len(got) != len(want) {
		t.Fatalf("Tokenize() len = %d, want %d (%v)", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Tokenize()[%d] = %q, want %q (%v)", i, got[i], want[i], got)
		}
	}
}

func TestTokenizePreservesUnicodeNumberTokens(t *testing.T) {
	got := Tokenize("x² phaseⅣ")
	want := []string{"x²", "phaseⅳ"}
	if len(got) != len(want) {
		t.Fatalf("Tokenize() len = %d, want %d (%v)", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Tokenize()[%d] = %q, want %q (%v)", i, got[i], want[i], got)
		}
	}
}

func TestTokenOverlapRatioMatchesSubstrings(t *testing.T) {
	got := TokenOverlapRatio("auth retry", "Authentication retry is enabled")
	if got != 1 {
		t.Fatalf("TokenOverlapRatio() = %v, want 1", got)
	}
}

func TestTokenOverlapRatioMatchesUnicodeNumberTokens(t *testing.T) {
	got := TokenOverlapRatio("x²", "Formula includes x² in the output")
	if got != 1 {
		t.Fatalf("TokenOverlapRatio() = %v, want 1", got)
	}
}

func TestAppendSourcesFormatsGroupedSortedRanges(t *testing.T) {
	got := AppendSources("Answer", []Citation{
		{SourcePath: "b.txt", StartLine: 3, EndLine: 4},
		{SourcePath: "a.txt", StartLine: 1, EndLine: 2},
		{SourcePath: "a.txt", StartLine: 1, EndLine: 2},
		{SourcePath: "a.txt", StartLine: 2, EndLine: 5},
		{SourcePath: "a.txt", StartLine: 7, EndLine: 7},
		{SourcePath: "a.txt", StartLine: 9, EndLine: 10},
		{SourcePath: "a.txt", StartLine: 11, EndLine: 11},
	})

	want := "Answer\n\nSources:\n\n- a.txt:\n  1-5, 7, 9-10, 11\n- b.txt:\n  3-4"
	if got != want {
		t.Fatalf("AppendSources() = %q, want %q", got, want)
	}
}

func TestAppendEvidenceSourcesSplitsVerifiedAndRetrieved(t *testing.T) {
	got := AppendEvidenceSources("Answer", []Evidence{
		{Kind: EvidenceRetrieved, SourcePath: "a.txt", StartLine: 1, EndLine: 5},
		{Kind: EvidenceVerified, SourcePath: "a.txt", StartLine: 2, EndLine: 3},
		{Kind: EvidenceRetrieved, SourcePath: "b.txt", StartLine: 8, EndLine: 9},
		{Kind: EvidenceVerified, SourcePath: "c.txt", StartLine: 4, EndLine: 4},
	})

	want := "Answer\n\nSources:\n\nVerified:\n\n- a.txt:\n  2-3\n- c.txt:\n  4\n\nRetrieved:\n\n- a.txt:\n  1, 4-5\n- b.txt:\n  8-9"
	if got != want {
		t.Fatalf("AppendEvidenceSources() = %q, want %q", got, want)
	}
}

func writeManifest(t *testing.T, path string, manifest Manifest) {
	t.Helper()

	data, err := json.Marshal(manifest)
	if err != nil {
		t.Fatalf("json.Marshal(manifest) error = %v", err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("os.WriteFile(%s) error = %v", path, err)
	}
}

func writeTestFile(t *testing.T, path, content string) {
	t.Helper()

	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("os.WriteFile(%s) error = %v", path, err)
	}
}
