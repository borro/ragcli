package rag

import (
	"context"
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
	"github.com/borro/ragcli/internal/ragcore"
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

func fileSource(reader io.Reader, path string) testSource {
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

	answer, err := Run(context.Background(), chat, embedder, fileSource(strings.NewReader(content), "/tmp/retries.txt"), cfg, "What is the retry policy?", nil)
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
	answer, err := Run(context.Background(), &fakeChat{answer: "should not be used"}, &fakeEmbedder{}, stdinSource(strings.NewReader("alpha beta gamma\ndelta epsilon\n")), cfg, "payments and invoices", nil)
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
			}, fileSource(strings.NewReader(content), "/tmp/retries.txt"), Options{
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

	answer, err := Run(context.Background(), &fakeChat{answer: "Retry policy is enabled [source 1]."}, &fakeEmbedder{}, fileSource(strings.NewReader(content), "/tmp/retries.txt"), cfg, "What is the retry policy?", plan)
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

func TestSelectionHelpersAndAppendSources(t *testing.T) {
	a0 := ragcore.FusedSeedHit{Path: "a.txt", ChunkID: 0, StartLine: 1, EndLine: 2, Score: 0.9}
	a1 := ragcore.FusedSeedHit{Path: "a.txt", ChunkID: 1, StartLine: 3, EndLine: 4, Score: 0.8}
	a3 := ragcore.FusedSeedHit{Path: "a.txt", ChunkID: 3, StartLine: 9, EndLine: 10, Score: 0.7}
	b0 := ragcore.FusedSeedHit{Path: "b.txt", ChunkID: 0, StartLine: 1, EndLine: 1, Score: 0.6}

	selected := selectEvidence([]ragcore.FusedSeedHit{a0, a1, a3}, 2)
	if len(selected) != 2 {
		t.Fatalf("len(selected) = %d, want 2", len(selected))
	}
	if selected[0].ChunkID != 0 || selected[1].ChunkID != 3 {
		t.Fatalf("selected chunk IDs = %d,%d, want 0,3", selected[0].ChunkID, selected[1].ChunkID)
	}

	fallback := selectEvidence([]ragcore.FusedSeedHit{a0, a1}, 2)
	if len(fallback) != 2 {
		t.Fatalf("len(fallback) = %d, want 2", len(fallback))
	}
	if fallback[1].ChunkID != 1 {
		t.Fatalf("fallback second chunk = %d, want 1", fallback[1].ChunkID)
	}

	if !containsHit(selected, a0) {
		t.Fatal("containsHit() = false, want true")
	}
	if containsHit(selected, b0) {
		t.Fatal("containsHit() = true, want false")
	}
	if got := appendSources("Answer", []ragcore.FusedSeedHit{a0, a0, b0}); strings.Count(got, "- a.txt:\n  1-2") != 1 {
		t.Fatalf("appendSources() = %q, want deduped citation", got)
	}
	if got := appendSources("Answer", []ragcore.FusedSeedHit{a0, a0, b0}); !strings.Contains(got, "- b.txt:\n  1") {
		t.Fatalf("appendSources() = %q, want second source", got)
	}
	if got := appendSources("Answer", nil); !strings.Contains(got, "Sources:\n\n- none") {
		t.Fatalf("appendSources(nil) = %q, want none source", got)
	}
}
