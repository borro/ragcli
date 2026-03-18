package rag_test

import (
	"context"
	"errors"
	"io"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/borro/ragcli/internal/aitools"
	ragtools "github.com/borro/ragcli/internal/aitools/rag"
	"github.com/borro/ragcli/internal/input"
	"github.com/borro/ragcli/internal/llm"
	"github.com/borro/ragcli/internal/localize"
	"github.com/borro/ragcli/internal/ragcore"
	"github.com/borro/ragcli/internal/verbose"
	openai "github.com/sashabaranov/go-openai"
)

type testSource struct {
	kind         input.Kind
	inputPath    string
	snapshotPath string
	files        []input.File
}

func (s testSource) Kind() input.Kind             { return s.kind }
func (s testSource) InputPath() string            { return s.inputPath }
func (s testSource) SnapshotPath() string         { return s.snapshotPath }
func (s testSource) DisplayName() string          { return s.inputPath }
func (s testSource) BackingFiles() []input.File   { return append([]input.File(nil), s.files...) }
func (s testSource) FileCount() int               { return len(s.files) }
func (s testSource) IsMultiFile() bool            { return s.kind == input.KindDirectory || len(s.files) > 1 }
func (s testSource) Open() (io.ReadCloser, error) { return os.Open(s.snapshotPath) }

type embeddingRequesterFunc func(context.Context, []string) ([][]float32, llm.EmbeddingMetrics, error)

func (f embeddingRequesterFunc) CreateEmbeddingsWithMetrics(ctx context.Context, inputs []string) ([][]float32, llm.EmbeddingMetrics, error) {
	return f(ctx, inputs)
}

func TestMain(m *testing.M) {
	if err := localize.SetCurrent(localize.EN); err != nil {
		panic(err)
	}
	os.Exit(m.Run())
}

func TestToolDefinitionRequiresQuery(t *testing.T) {
	tool := ragtools.NewSearchTool(&ragcore.PreparedSearch{}, embeddingRequesterFunc(func(context.Context, []string) ([][]float32, llm.EmbeddingMetrics, error) {
		return nil, llm.EmbeddingMetrics{}, nil
	}))

	def := tool.ToolDefinition()
	if def.Function == nil || def.Function.Name != "search_rag" {
		t.Fatalf("definition = %#v, want search_rag", def.Function)
	}

	params, ok := def.Function.Parameters.(map[string]any)
	if !ok {
		t.Fatalf("parameters = %#v, want map[string]any", def.Function.Parameters)
	}
	required, ok := params["required"].([]string)
	if !ok || len(required) != 1 || required[0] != "query" {
		t.Fatalf("required = %#v, want only query", params["required"])
	}
}

func TestSearchRAGExecuteReturnsTypedPaginationAndHints(t *testing.T) {
	tool := newPreparedTool(t)

	first, err := tool.Execute(context.Background(), toolCall("1", "search_rag", `{"query":"retry","limit":1}`))
	if err != nil {
		t.Fatalf("Execute(first) error = %v", err)
	}
	if next := first.Result.Hints[aitools.HintSuggestedNextOffset]; next != 1 {
		t.Fatalf("next offset = %#v, want 1", next)
	}
	if len(first.Result.ProgressKeys) == 0 || !strings.HasPrefix(first.Result.ProgressKeys[0], "rag:") {
		t.Fatalf("ProgressKeys = %#v, want rag-prefixed keys", first.Result.ProgressKeys)
	}

	payload, ok := first.Result.Payload.(ragtools.SearchResult)
	if !ok {
		t.Fatalf("payload type = %T, want rag.SearchResult", first.Result.Payload)
	}
	if payload.MatchCount != 1 || payload.TotalMatches < 2 || !payload.HasMore {
		t.Fatalf("payload = %+v, want paginated matches", payload)
	}
	if payload.NextOffset != 1 {
		t.Fatalf("NextOffset = %d, want 1", payload.NextOffset)
	}

	second, err := tool.Execute(context.Background(), toolCall("2", "search_rag", `{"query":"retry","limit":1,"offset":1}`))
	if err != nil {
		t.Fatalf("Execute(second) error = %v", err)
	}
	secondPayload := second.Result.Payload.(ragtools.SearchResult)
	if secondPayload.Offset != 1 || secondPayload.MatchCount != 1 {
		t.Fatalf("second payload = %+v, want second page", secondPayload)
	}
}

func TestSearchRAGExecuteSupportsPathFilterAndCache(t *testing.T) {
	tool := newPreparedTool(t)

	result, err := tool.Execute(context.Background(), toolCall("1", "search_rag", `{"path":"c.txt","query":"retry"}`))
	if err != nil {
		t.Fatalf("Execute(path filter) error = %v", err)
	}
	payload := result.Result.Payload.(ragtools.SearchResult)
	if payload.Path != "c.txt" || payload.MatchCount != 1 {
		t.Fatalf("payload = %+v, want one match in c.txt", payload)
	}

	first, err := tool.Execute(context.Background(), toolCall("2", "search_rag", `{"query":"database"}`))
	if err != nil {
		t.Fatalf("Execute(first) error = %v", err)
	}
	second, err := tool.Execute(context.Background(), toolCall("3", "search_rag", `{"query":"database"}`))
	if err != nil {
		t.Fatalf("Execute(second) error = %v", err)
	}
	if first.Cached || first.Duplicate {
		t.Fatalf("first = %+v, want fresh execution", first)
	}
	if !second.Cached || !second.Duplicate {
		t.Fatalf("second = %+v, want cached duplicate", second)
	}
}

func TestSearchRAGDescribeCallAndErrors(t *testing.T) {
	tool := newPreparedTool(t)

	desc := tool.DescribeCall(toolCall("1", "search_rag", `{"query":"retry","path":"c.txt","limit":1,"offset":2}`))
	if desc.VerboseLabel != `search_rag(query="retry", path="c.txt", limit=1, offset=2)` {
		t.Fatalf("VerboseLabel = %q, want compact label", desc.VerboseLabel)
	}

	if _, err := tool.Execute(context.Background(), toolCall("2", "search_rag", `{"path":"missing.txt","query":"retry"}`)); err == nil {
		t.Fatal("Execute(missing path) error = nil, want invalid_arguments")
	} else if code := aitools.AsToolError(err).Code; code != "invalid_arguments" {
		t.Fatalf("Code = %q, want invalid_arguments", code)
	}
}

func TestSearchRAGExecuteReportsEmbedderErrors(t *testing.T) {
	searcher, err := preparedSearch(t, embeddingRequesterFunc(func(_ context.Context, inputs []string) ([][]float32, llm.EmbeddingMetrics, error) {
		vectors := make([][]float32, 0, len(inputs))
		for _, input := range inputs {
			vectors = append(vectors, vectorFor(input))
		}
		return vectors, llm.EmbeddingMetrics{InputCount: len(inputs), VectorCount: len(vectors)}, nil
	}))
	if err != nil {
		t.Fatalf("preparedSearch() error = %v", err)
	}
	tool := ragtools.NewSearchTool(searcher, embeddingRequesterFunc(func(_ context.Context, _ []string) ([][]float32, llm.EmbeddingMetrics, error) {
		return nil, llm.EmbeddingMetrics{}, errors.New("embed failed")
	}))

	_, err = tool.Execute(context.Background(), toolCall("1", "search_rag", `{"query":"retry"}`))
	if err == nil || !strings.Contains(err.Error(), "failed to search rag index") {
		t.Fatalf("Execute() error = %v, want wrapped embed failure", err)
	}
}

func newPreparedTool(t *testing.T) aitools.Tool {
	t.Helper()

	embedder := embeddingRequesterFunc(func(_ context.Context, inputs []string) ([][]float32, llm.EmbeddingMetrics, error) {
		vectors := make([][]float32, 0, len(inputs))
		for _, input := range inputs {
			vectors = append(vectors, vectorFor(input))
		}
		return vectors, llm.EmbeddingMetrics{
			InputCount:   len(inputs),
			VectorCount:  len(vectors),
			PromptTokens: len(inputs),
			TotalTokens:  len(inputs),
		}, nil
	})

	searcher, err := preparedSearch(t, embedder)
	if err != nil {
		t.Fatalf("preparedSearch() error = %v", err)
	}

	return ragtools.NewSearchTool(searcher, embedder)
}

func preparedSearch(t *testing.T, embedder llm.EmbeddingRequester) (*ragcore.PreparedSearch, error) {
	t.Helper()

	source := corpusSource(t)
	searcher, _, err := ragcore.PrepareSearch(context.Background(), embedder, source, ragcore.SearchOptions{
		TopK:           8,
		ChunkSize:      1000,
		ChunkOverlap:   200,
		IndexTTL:       time.Hour,
		IndexDir:       t.TempDir(),
		Rerank:         "heuristic",
		EmbeddingModel: "embed-v1",
	}, verbose.Meter{})
	return searcher, err
}

func corpusSource(t *testing.T) testSource {
	t.Helper()

	files := []input.File{
		{Path: writeTempFile(t, "retry policy enabled\n"), DisplayPath: "a.txt"},
		{Path: writeTempFile(t, "database connections stable\n"), DisplayPath: "b.txt"},
		{Path: writeTempFile(t, "retry happens with backoff\n"), DisplayPath: "c.txt"},
	}
	return testSource{
		kind:         input.KindDirectory,
		inputPath:    "/tmp/corpus",
		snapshotPath: writeTempFile(t, ""),
		files:        files,
	}
}

func toolCall(id string, name string, arguments string) openai.ToolCall {
	return openai.ToolCall{
		ID:   id,
		Type: "function",
		Function: openai.FunctionCall{
			Name:      name,
			Arguments: arguments,
		},
	}
}

func writeTempFile(t *testing.T, content string) string {
	t.Helper()

	file, err := os.CreateTemp(t.TempDir(), "rag-*.txt")
	if err != nil {
		t.Fatalf("CreateTemp() error = %v", err)
	}
	if _, err := file.WriteString(content); err != nil {
		_ = file.Close()
		t.Fatalf("WriteString() error = %v", err)
	}
	if err := file.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	return file.Name()
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
	default:
		return []float32{0, 0, 0}
	}
}
