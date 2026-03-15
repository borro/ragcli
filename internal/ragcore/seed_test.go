package ragcore

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	openai "github.com/sashabaranov/go-openai"
)

type scriptedSyntheticExecutor struct {
	t               *testing.T
	searchPayload   string
	readLinesRaw    string
	readAroundRaw   string
	failToolName    string
	failErr         error
	calls           []openai.ToolCall
	searchCallCount int
}

func (s *scriptedSyntheticExecutor) ExecuteSyntheticToolCall(_ context.Context, call openai.ToolCall) (string, error) {
	s.calls = append(s.calls, call)
	if s.failToolName != "" && call.Function.Name == s.failToolName {
		return "", s.failErr
	}

	switch call.Function.Name {
	case "search_rag":
		s.searchCallCount++
		var params SearchParams
		if err := json.Unmarshal([]byte(call.Function.Arguments), &params); err != nil {
			s.t.Fatalf("json.Unmarshal(search args) error = %v", err)
		}
		if params.Query == "" {
			s.t.Fatal("seed search query = empty, want generated query")
		}
		return s.searchPayload, nil
	case "read_lines":
		return s.readLinesRaw, nil
	case "read_around":
		return s.readAroundRaw, nil
	default:
		s.t.Fatalf("unexpected synthetic tool call %q", call.Function.Name)
		return "", nil
	}
}

func TestBuildSeedBundleBuildsSyntheticHistory(t *testing.T) {
	searchPayload, err := MarshalJSON(SearchResult{
		Path:         "fallback.txt",
		MatchCount:   3,
		TotalMatches: 3,
		Matches: []SearchMatch{
			{Path: "a.txt", ChunkID: 1, StartLine: 1, EndLine: 2, Text: "retry policy enabled", Score: 0.9, Similarity: 0.9, Overlap: 0.5},
			{Path: "", ChunkID: 2, StartLine: 10, EndLine: 60, Text: "database config", Score: 0.6, Similarity: 0.3, Overlap: 0.2},
			{Path: "a.txt", ChunkID: 3, StartLine: 2, EndLine: 3, Text: "retry details", Score: 0.7, Similarity: 0.85, Overlap: 0.4},
		},
	})
	if err != nil {
		t.Fatalf("MarshalJSON(search payload) error = %v", err)
	}

	executor := &scriptedSyntheticExecutor{
		t:             t,
		searchPayload: searchPayload,
		readLinesRaw:  `{"path":"a.txt","start_line":1,"end_line":2}`,
		readAroundRaw: `{"path":"fallback.txt","line":35,"before":20,"after":20}`,
	}

	bundle, err := BuildSeedBundle(context.Background(), executor, "What does it say about retry policy and database config?", SearchOptions{TopK: 3})
	if err != nil {
		t.Fatalf("BuildSeedBundle() error = %v", err)
	}
	if executor.searchCallCount < 2 {
		t.Fatalf("searchCallCount = %d, want multiple fused seed searches", executor.searchCallCount)
	}
	if bundle.Weak {
		t.Fatalf("bundle.Weak = true, want strong fused hit: %+v", bundle.Hits)
	}
	if len(bundle.Hits) != 3 {
		t.Fatalf("len(bundle.Hits) = %d, want 3", len(bundle.Hits))
	}
	if len(bundle.History) != executor.searchCallCount+4 {
		t.Fatalf("len(bundle.History) = %d, want %d", len(bundle.History), executor.searchCallCount+4)
	}
	if bundle.History[0].Role != openai.ChatMessageRoleAssistant {
		t.Fatalf("first history role = %q, want assistant", bundle.History[0].Role)
	}
	if countToolCalls(executor.calls, "read_lines") != 1 {
		t.Fatalf("read_lines calls = %d, want 1", countToolCalls(executor.calls, "read_lines"))
	}
	if countToolCalls(executor.calls, "read_around") != 1 {
		t.Fatalf("read_around calls = %d, want 1", countToolCalls(executor.calls, "read_around"))
	}
	if !containsHitPath(bundle.Hits, "fallback.txt") {
		t.Fatalf("bundle.Hits = %+v, want payload path fallback in fused hits", bundle.Hits)
	}
}

func TestExecuteSeedSearchesRejectsInvalidJSON(t *testing.T) {
	executor := &scriptedSyntheticExecutor{
		t:             t,
		searchPayload: "{not-json",
	}

	_, _, err := executeSeedSearches(context.Background(), executor, []string{"retry"}, SearchOptions{TopK: 3})
	if err == nil || !strings.Contains(err.Error(), "failed to decode hybrid seed search result") {
		t.Fatalf("executeSeedSearches() error = %v, want decode failure", err)
	}
}

func TestExecuteSeedVerificationReadsPropagatesExecutorError(t *testing.T) {
	executor := &scriptedSyntheticExecutor{
		t:            t,
		failToolName: "read_lines",
		failErr:      errors.New("boom"),
	}

	_, err := executeSeedVerificationReads(context.Background(), executor, []FusedSeedHit{{Path: "a.txt", StartLine: 1, EndLine: 2}})
	if err == nil || !strings.Contains(err.Error(), "failed to execute hybrid seed verification read: boom") {
		t.Fatalf("executeSeedVerificationReads() error = %v, want wrapped executor failure", err)
	}
}

func TestSortFusedHitsUsesDeterministicTieBreakers(t *testing.T) {
	hits := sortFusedHits(map[string]*fusedSeedCandidate{
		"c:1": {Path: "c.txt", ChunkID: 1, Fusion: 0.5, Similarity: 0.8, Overlap: 0.0},
		"d:1": {Path: "d.txt", ChunkID: 1, Fusion: 0.5, Similarity: 0.7, Overlap: 0.2},
		"a:1": {Path: "a.txt", ChunkID: 1, Fusion: 0.5, Similarity: 0.7, Overlap: 0.1},
		"a:2": {Path: "a.txt", ChunkID: 2, Fusion: 0.5, Similarity: 0.7, Overlap: 0.1},
		"b:1": {Path: "b.txt", ChunkID: 1, Fusion: 0.5, Similarity: 0.7, Overlap: 0.1},
	})

	got := make([]string, 0, len(hits))
	for _, hit := range hits {
		got = append(got, hit.Path+":"+string(rune('0'+hit.ChunkID)))
	}
	want := []string{"c.txt:1", "d.txt:1", "a.txt:1", "a.txt:2", "b.txt:1"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("sortFusedHits() = %v, want %v", got, want)
	}
}

func TestOverlapsSelectedAndVerificationToolCall(t *testing.T) {
	if !overlapsSelected(FusedSeedHit{Path: "a.txt", StartLine: 2, EndLine: 4}, []FusedSeedHit{{Path: "a.txt", StartLine: 1, EndLine: 3}}) {
		t.Fatal("overlapsSelected() = false, want overlap in same file")
	}
	if overlapsSelected(FusedSeedHit{Path: "b.txt", StartLine: 2, EndLine: 4}, []FusedSeedHit{{Path: "a.txt", StartLine: 1, EndLine: 3}}) {
		t.Fatal("overlapsSelected() = true, want false for different file")
	}

	_, err := verificationToolCall(FusedSeedHit{}, 1)
	if err == nil || !strings.Contains(err.Error(), "path is required") {
		t.Fatalf("verificationToolCall(empty) error = %v, want path validation", err)
	}

	lineCall, err := verificationToolCall(FusedSeedHit{Path: "a.txt", StartLine: 1, EndLine: 2}, 1)
	if err != nil {
		t.Fatalf("verificationToolCall(read_lines) error = %v", err)
	}
	if lineCall.Function.Name != "read_lines" {
		t.Fatalf("lineCall.Function.Name = %q, want read_lines", lineCall.Function.Name)
	}

	aroundCall, err := verificationToolCall(FusedSeedHit{Path: "b.txt", StartLine: 10, EndLine: 60}, 2)
	if err != nil {
		t.Fatalf("verificationToolCall(read_around) error = %v", err)
	}
	if aroundCall.Function.Name != "read_around" {
		t.Fatalf("aroundCall.Function.Name = %q, want read_around", aroundCall.Function.Name)
	}
}

func countToolCalls(calls []openai.ToolCall, name string) int {
	total := 0
	for _, call := range calls {
		if call.Function.Name == name {
			total++
		}
	}
	return total
}

func containsHitPath(hits []FusedSeedHit, want string) bool {
	for _, hit := range hits {
		if hit.Path == want {
			return true
		}
	}
	return false
}
