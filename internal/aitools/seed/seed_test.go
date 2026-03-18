package seed

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/borro/ragcli/internal/aitools"
	ragtools "github.com/borro/ragcli/internal/aitools/rag"
	"github.com/borro/ragcli/internal/ragcore"
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
		var params ragtools.SearchParams
		if err := json.Unmarshal([]byte(call.Function.Arguments), &params); err != nil {
			s.t.Fatalf("json.Unmarshal(search args) error = %v", err)
		}
		if strings.TrimSpace(params.Query) == "" {
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
	searchPayload, err := aitools.EncodeToolResult(aitools.Execution{
		Result: aitools.ToolResult{
			Payload: ragtools.SearchResult{
				Path:         "fallback.txt",
				MatchCount:   3,
				TotalMatches: 3,
				Matches: []ragtools.SearchMatch{
					{Path: "a.txt", ChunkID: 1, StartLine: 1, EndLine: 2, Text: "retry policy enabled", Score: 0.9, Similarity: 0.9, Overlap: 0.5},
					{Path: "", ChunkID: 2, StartLine: 10, EndLine: 60, Text: "database config", Score: 0.6, Similarity: 0.3, Overlap: 0.2},
					{Path: "a.txt", ChunkID: 3, StartLine: 2, EndLine: 3, Text: "retry details", Score: 0.7, Similarity: 0.85, Overlap: 0.4},
				},
			},
		},
	}, aitools.ToolResponseMeta{NovelLines: 3})
	if err != nil {
		t.Fatalf("EncodeToolResult(search payload) error = %v", err)
	}

	executor := &scriptedSyntheticExecutor{
		t:             t,
		searchPayload: searchPayload,
		readLinesRaw:  `{"result":{"path":"a.txt","start_line":1,"end_line":2},"meta":{"novel_lines":2}}`,
		readAroundRaw: `{"result":{"path":"fallback.txt","line":35,"before":20,"after":20},"meta":{"novel_lines":40}}`,
	}

	bundle, err := BuildSeedBundle(context.Background(), executor, "What does it say about retry policy and database config?", ragcore.SearchOptions{TopK: 3})
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
	if countToolCalls(executor.calls, "read_lines") != 1 || countToolCalls(executor.calls, "read_around") != 1 {
		t.Fatalf("verification calls = %#v, want one read_lines and one read_around", executor.calls)
	}
	if !containsHitPath(bundle.Hits, "fallback.txt") {
		t.Fatalf("bundle.Hits = %+v, want payload path fallback in fused hits", bundle.Hits)
	}
}

func TestExecuteSeedSearchesRejectsInvalidEnvelope(t *testing.T) {
	executor := &scriptedSyntheticExecutor{
		t:             t,
		searchPayload: "{not-json",
	}

	_, _, err := executeSeedSearches(context.Background(), executor, []string{"retry"}, ragcore.SearchOptions{TopK: 3})
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

	_, err := executeSeedVerificationReads(context.Background(), executor, []ragcore.FusedSeedHit{{Path: "a.txt", StartLine: 1, EndLine: 2}})
	if err == nil || !strings.Contains(err.Error(), "failed to execute hybrid seed verification read: boom") {
		t.Fatalf("executeSeedVerificationReads() error = %v, want wrapped executor failure", err)
	}
}

func TestOverlapsSelectedAndVerificationToolCall(t *testing.T) {
	if !overlapsSelected(ragcore.FusedSeedHit{Path: "a.txt", StartLine: 2, EndLine: 4}, []ragcore.FusedSeedHit{{Path: "a.txt", StartLine: 1, EndLine: 3}}) {
		t.Fatal("overlapsSelected() = false, want overlap in same file")
	}
	if overlapsSelected(ragcore.FusedSeedHit{Path: "b.txt", StartLine: 2, EndLine: 4}, []ragcore.FusedSeedHit{{Path: "a.txt", StartLine: 1, EndLine: 3}}) {
		t.Fatal("overlapsSelected() = true, want false for different file")
	}

	_, err := verificationToolCall(ragcore.FusedSeedHit{}, 1)
	if err == nil || !strings.Contains(err.Error(), "path is required") {
		t.Fatalf("verificationToolCall(empty) error = %v, want path validation", err)
	}

	lineCall, err := verificationToolCall(ragcore.FusedSeedHit{Path: "a.txt", StartLine: 1, EndLine: 2}, 1)
	if err != nil || lineCall.Function.Name != "read_lines" {
		t.Fatalf("verificationToolCall(read_lines) = %#v, %v", lineCall.Function, err)
	}

	aroundCall, err := verificationToolCall(ragcore.FusedSeedHit{Path: "b.txt", StartLine: 10, EndLine: 60}, 2)
	if err != nil || aroundCall.Function.Name != "read_around" {
		t.Fatalf("verificationToolCall(read_around) = %#v, %v", aroundCall.Function, err)
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

func containsHitPath(hits []ragcore.FusedSeedHit, want string) bool {
	for _, hit := range hits {
		if hit.Path == want {
			return true
		}
	}
	return false
}
