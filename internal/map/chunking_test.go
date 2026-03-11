package mapmode

import (
	"context"
	"strings"
	"testing"
)

func TestApproxTokenCount(t *testing.T) {
	tests := []struct {
		input string
		want  int
	}{
		{input: "", want: 0},
		{input: "a", want: 1},
		{input: "abc", want: 1},
		{input: "abcd", want: 2},
		{input: "привет", want: 2},
	}

	for _, tt := range tests {
		if got := approxTokenCount(tt.input); got != tt.want {
			t.Fatalf("approxTokenCount(%q) = %d, want %d", tt.input, got, tt.want)
		}
	}
}

func TestEffectiveApproxTokenBudget(t *testing.T) {
	tests := []struct {
		input int
		want  int
	}{
		{input: 1, want: 1},
		{input: 5, want: 3},
		{input: 10, want: 6},
		{input: 30000, want: 18000},
	}

	for _, tt := range tests {
		if got := effectiveMapApproxTokenBudget(tt.input); got != tt.want {
			t.Fatalf("effectiveMapApproxTokenBudget(%d) = %d, want %d", tt.input, got, tt.want)
		}
	}
}

func TestSplitByApproxTokens(t *testing.T) {
	tests := []struct {
		name           string
		question       string
		input          string
		maxTokens      int
		expectedChunks []string
	}{
		{
			name:           "empty input",
			question:       "Что в файле?",
			input:          "",
			maxTokens:      estimateMapRequestTokens("Что в файле?", "") + 10,
			expectedChunks: []string{},
		},
		{
			name:      "multiple ascii lines fit into one chunk",
			question:  "Что в файле?",
			input:     "abc\ndef",
			maxTokens: (estimateMapRequestTokens("Что в файле?", "abc\ndef") * mapBudgetDenominator / mapBudgetNumerator) + 1,
			expectedChunks: []string{
				"abc\ndef",
			},
		},
		{
			name:      "new line makes next chunk overflow",
			question:  "Что в файле?",
			input:     "abcdef\nghi",
			maxTokens: (estimateMapRequestTokens("Что в файле?", "abcdef") * mapBudgetDenominator / mapBudgetNumerator) + 1,
			expectedChunks: []string{
				"abcdef",
				"ghi",
			},
		},
		{
			name:      "unicode respects runes not bytes",
			question:  "Что в файле?",
			input:     "привет\nмир",
			maxTokens: (estimateMapRequestTokens("Что в файле?", "привет") * mapBudgetDenominator / mapBudgetNumerator) + 1,
			expectedChunks: []string{
				"привет",
				"мир",
			},
		},
		{
			name:      "oversized line splits by rune boundaries",
			question:  "Что в файле?",
			input:     "abcdefghij",
			maxTokens: (estimateMapRequestTokens("Что в файле?", "abcdefghi") * mapBudgetDenominator / mapBudgetNumerator) + 1,
			expectedChunks: []string{
				"abcdefghi",
				"j",
			},
		},
		{
			name:      "exact boundary",
			question:  "Что в файле?",
			input:     "abcdef",
			maxTokens: (estimateMapRequestTokens("Что в файле?", "abcdef") * mapBudgetDenominator / mapBudgetNumerator) + 1,
			expectedChunks: []string{
				"abcdef",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			chunks, err := SplitByApproxTokens(context.Background(), strings.NewReader(tt.input), tt.question, tt.maxTokens)
			if err != nil {
				t.Fatalf("SplitByApproxTokens() error = %v", err)
			}
			if len(chunks) != len(tt.expectedChunks) {
				t.Fatalf("SplitByApproxTokens() returned %d chunks, want %d", len(chunks), len(tt.expectedChunks))
			}

			for i := range chunks {
				if chunks[i].Text != tt.expectedChunks[i] {
					t.Fatalf("chunk[%d] = %q, want %q", i, chunks[i].Text, tt.expectedChunks[i])
				}
				if chunks[i].TokenCount > tt.maxTokens {
					t.Fatalf("chunk[%d] token count = %d, want <= %d", i, chunks[i].TokenCount, tt.maxTokens)
				}
				if chunks[i].RequestTokens > tt.maxTokens {
					t.Fatalf("chunk[%d] request tokens = %d, want <= %d", i, chunks[i].RequestTokens, tt.maxTokens)
				}
			}
		})
	}
}

func TestSplitByApproxTokens_LargeInput(t *testing.T) {
	var builder strings.Builder
	for i := 0; i < 1000; i++ {
		builder.WriteString("line\n")
	}

	chunks, err := SplitByApproxTokens(context.Background(), strings.NewReader(builder.String()), "Что в файле?", (estimateMapRequestTokens("Что в файле?", "line\nline\nline")*mapBudgetDenominator/mapBudgetNumerator)+1)
	if err != nil {
		t.Fatalf("SplitByApproxTokens() error = %v", err)
	}
	if len(chunks) == 0 {
		t.Fatal("expected non-empty chunks")
	}
}

func TestSplitByApproxTokens_ErrorsWhenPromptAlreadyExceedsLimit(t *testing.T) {
	_, err := SplitByApproxTokens(context.Background(), strings.NewReader("a"), strings.Repeat("очень длинный вопрос ", 40), 1)
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestSplitByApproxTokens_RespectsContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := SplitByApproxTokens(ctx, strings.NewReader("abc"), "Что в файле?", (estimateMapRequestTokens("Что в файле?", "abc")*mapBudgetDenominator/mapBudgetNumerator)+1)
	if err == nil {
		t.Fatal("expected context cancellation error")
	}
}
