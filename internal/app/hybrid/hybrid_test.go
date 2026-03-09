package hybrid

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/borro/ragcli/internal/llm"
	openai "github.com/sashabaranov/go-openai"
)

type fakeEmbedder struct {
	err error
}

func (f *fakeEmbedder) CreateEmbeddingsWithMetrics(_ context.Context, inputs []string) ([][]float32, llm.EmbeddingMetrics, error) {
	if f.err != nil {
		return nil, llm.EmbeddingMetrics{}, f.err
	}
	vectors := make([][]float32, 0, len(inputs))
	for _, input := range inputs {
		vectors = append(vectors, vectorFor(input))
	}
	return vectors, llm.EmbeddingMetrics{InputCount: len(inputs), VectorCount: len(vectors)}, nil
}

type scriptedChat struct {
	responses []string
	requests  []openai.ChatCompletionRequest
}

func (s *scriptedChat) SendRequestWithMetrics(_ context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, llm.RequestMetrics, error) {
	s.requests = append(s.requests, req)
	idx := len(s.requests) - 1
	if idx >= len(s.responses) {
		return nil, llm.RequestMetrics{}, context.DeadlineExceeded
	}
	return &openai.ChatCompletionResponse{
		Choices: []openai.ChatCompletionChoice{{
			Message: openai.ChatCompletionMessage{Role: "assistant", Content: s.responses[idx]},
		}},
	}, llm.RequestMetrics{}, nil
}

func TestProfileDocument(t *testing.T) {
	if got := profileDocument("README.md", "# H1\n\n## H2\ntext"); got != ProfileMarkdown {
		t.Fatalf("profileDocument(markdown) = %q, want %q", got, ProfileMarkdown)
	}
	if got := profileDocument("app.log", "2026-03-09 10:00:00 ERROR fail\n2026-03-09 10:00:01 INFO retry\n2026-03-09 10:00:02 WARN done"); got != ProfileLogs {
		t.Fatalf("profileDocument(logs) = %q, want %q", got, ProfileLogs)
	}
	if got := profileDocument("notes.txt", "plain text\n\nsecond paragraph"); got != ProfilePlain {
		t.Fatalf("profileDocument(plain) = %q, want %q", got, ProfilePlain)
	}
}

func TestSegmentDocumentMarkdownKeepsSections(t *testing.T) {
	content := []byte("# Intro\nline 1\nline 2\n\n## Details\nline 3\nline 4\n")
	_, meso := segmentDocument(content, ProfileMarkdown, Options{ChunkSize: 80, ReadWindow: 2})
	if len(meso) < 2 {
		t.Fatalf("len(meso) = %d, want >= 2", len(meso))
	}
	if meso[0].StartLine != 1 {
		t.Fatalf("meso[0].StartLine = %d, want 1", meso[0].StartLine)
	}
	if meso[1].StartLine <= meso[0].StartLine {
		t.Fatalf("meso[1].StartLine = %d, want > %d", meso[1].StartLine, meso[0].StartLine)
	}
}

func TestSegmentDocumentPlainMergesParagraphsIntoLargerSections(t *testing.T) {
	content := []byte(strings.Join([]string{
		"Paragraph one line one.",
		"Paragraph one line two.",
		"",
		"Paragraph two line one.",
		"Paragraph two line two.",
		"",
		"Paragraph three line one.",
		"Paragraph three line two.",
	}, "\n"))
	_, meso := segmentDocument(content, ProfilePlain, Options{ChunkSize: 120, ReadWindow: 2})
	if len(meso) >= 3 {
		t.Fatalf("len(meso) = %d, want merged sections < 3", len(meso))
	}
	if len(meso) == 0 {
		t.Fatal("len(meso) = 0, want merged section")
	}
	if !strings.Contains(meso[0].Text, "Paragraph two line one.") {
		t.Fatalf("meso[0].Text = %q, want merged paragraph content", meso[0].Text)
	}
}

func TestRunHybridFallsBackToMapWhenEmbeddingsFail(t *testing.T) {
	chat := &scriptedChat{
		responses: []string{
			"fact from fallback map",
			"fallback answer",
			"SUPPORTED: yes\nISSUES:\nNONE",
		},
	}

	answer, err := Run(context.Background(), chat, &fakeEmbedder{err: errors.New("embed fail")}, Source{
		Path:        "",
		DisplayName: "stdin",
		Reader:      strings.NewReader("retry policy is enabled\n"),
	}, Options{
		TopK:           4,
		FinalK:         2,
		MapK:           2,
		ReadWindow:     2,
		Fallback:       "map",
		ChunkSize:      1000,
		ChunkOverlap:   100,
		IndexTTL:       time.Hour,
		IndexDir:       t.TempDir(),
		EmbeddingModel: "embed",
	}, "What is the retry policy?")
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if answer != "fallback answer" {
		t.Fatalf("Run() answer = %q, want fallback answer", answer)
	}
}

func TestRunHybridCoverageTriggersExtraRegion(t *testing.T) {
	content := strings.Join([]string{
		"Retry policy",
		"requests use exponential backoff",
		"",
		"Operational notes",
		"service metrics are emitted every minute",
		"",
		"Maintenance",
		"weekly backup runs every Sunday",
		"",
		"Release process",
		"rollback checklist is required before deploy",
	}, "\n")

	chat := &scriptedChat{
		responses: []string{
			"requests use exponential backoff",
			"COVERED: no\nGAPS:\n- release process\nCONFLICTS:\n- none\nFOLLOW_UP_QUERY: release process rollback checklist",
			"rollback checklist is required before deploy",
			"Retry uses exponential backoff [source 1]. Release requires a rollback checklist [source 2].",
		},
	}

	answer, err := Run(context.Background(), chat, &fakeEmbedder{}, Source{
		Path:        writeTempFile(t, content),
		DisplayName: "sample.txt",
		Reader:      strings.NewReader(content),
	}, Options{
		TopK:           1,
		FinalK:         1,
		MapK:           1,
		ReadWindow:     1,
		Fallback:       "rag-only",
		ChunkSize:      1000,
		ChunkOverlap:   100,
		IndexTTL:       time.Hour,
		IndexDir:       t.TempDir(),
		EmbeddingModel: "embed",
	}, "What is the retry policy and what is required before release?")
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if !strings.Contains(answer, "rollback checklist") {
		t.Fatalf("answer = %q, want extra-region fact", answer)
	}
	if !strings.Contains(answer, "Sources:\n- sample.txt:") {
		t.Fatalf("answer = %q, want sources block", answer)
	}
}

func TestParseCoverageReport(t *testing.T) {
	report := parseCoverageReport("COVERED: no\nGAPS:\n- release process\nCONFLICTS:\n- outdated section\nFOLLOW_UP_QUERY: release process rollback checklist")
	if report.Covered {
		t.Fatal("Covered = true, want false")
	}
	if report.FollowUpQuery != "release process rollback checklist" {
		t.Fatalf("FollowUpQuery = %q, want parsed query", report.FollowUpQuery)
	}
	if len(report.Gaps) != 1 || report.Gaps[0] != "release process" {
		t.Fatalf("Gaps = %#v, want parsed gap", report.Gaps)
	}
	if len(report.Conflicts) != 1 || report.Conflicts[0] != "outdated section" {
		t.Fatalf("Conflicts = %#v, want parsed conflict", report.Conflicts)
	}
}

func TestDeriveFollowUpQueryFromCoverageGaps(t *testing.T) {
	query := deriveFollowUpQuery("What changed in retries and release?", CoverageReport{
		Covered:   false,
		Gaps:      []string{"release process", "rollback checklist"},
		Conflicts: []string{"none"},
	})
	if query != "release process rollback checklist" {
		t.Fatalf("deriveFollowUpQuery() = %q, want gap-based query", query)
	}
}

func TestTrimSemanticHitsDropsWeakTail(t *testing.T) {
	hits := []RetrievedHit{
		{SegmentID: 1, Score: 0.91},
		{SegmentID: 2, Score: 0.72},
		{SegmentID: 3, Score: 0.51},
		{SegmentID: 4, Score: 0.18},
		{SegmentID: 5, Score: 0.05},
	}
	trimmed := trimSemanticHits(hits, 1)
	if len(trimmed) >= len(hits) {
		t.Fatalf("len(trimmed) = %d, want fewer than %d", len(trimmed), len(hits))
	}
	if trimmed[len(trimmed)-1].Score < 0.18 {
		t.Fatalf("last trimmed score = %f, want >= 0.18", trimmed[len(trimmed)-1].Score)
	}
}

func vectorFor(input string) []float32 {
	lower := strings.ToLower(input)
	return []float32{
		feature(lower, "retry"),
		feature(lower, "backoff"),
		feature(lower, "release"),
		feature(lower, "rollback"),
		feature(lower, "error"),
	}
}

func feature(input, token string) float32 {
	if strings.Contains(input, token) {
		return 1
	}
	return 0
}

func writeTempFile(t *testing.T, content string) string {
	t.Helper()
	path := t.TempDir() + "/sample.txt"
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("os.WriteFile() error = %v", err)
	}
	return path
}
