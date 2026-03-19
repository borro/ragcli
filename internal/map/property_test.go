package mapmode

import (
	"context"
	"strings"
	"testing"
	"unicode/utf8"

	"pgregory.net/rapid"
)

func TestSplitOversizedApproxLine_PropertyPreservesTextAndBudgets(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		line := drawMapString(t, "line", 72, []rune{'a', 'B', 'ж', 'Я', '🙂', '1', ' ', '\t', '-', '_'})
		maxChunkTokens := rapid.IntRange(1, 12).Draw(t, "maxChunkTokens")
		requestOverhead := rapid.IntRange(0, 80).Draw(t, "requestOverhead")

		chunks, err := splitOversizedApproxLine(context.Background(), line, maxChunkTokens, requestOverhead)
		if err != nil {
			t.Fatalf("splitOversizedApproxLine() error = %v", err)
		}

		if line == "" {
			if chunks != nil {
				t.Fatalf("chunks = %v, want nil for empty input", chunks)
			}
			return
		}
		if len(chunks) == 0 {
			t.Fatal("expected at least one chunk for non-empty input")
		}

		var rebuilt strings.Builder
		for i, chunk := range chunks {
			if chunk.Text == "" {
				t.Fatalf("chunk[%d] is empty", i)
			}
			if !utf8.ValidString(chunk.Text) {
				t.Fatalf("chunk[%d] is not valid UTF-8: %q", i, chunk.Text)
			}
			if chunk.TokenCount != approxTokenCount(chunk.Text) {
				t.Fatalf("chunk[%d].TokenCount = %d, want %d", i, chunk.TokenCount, approxTokenCount(chunk.Text))
			}
			if chunk.TokenCount > maxChunkTokens {
				t.Fatalf("chunk[%d].TokenCount = %d, want <= %d", i, chunk.TokenCount, maxChunkTokens)
			}
			if chunk.RequestTokens != requestOverhead+chunk.TokenCount {
				t.Fatalf("chunk[%d].RequestTokens = %d, want %d", i, chunk.RequestTokens, requestOverhead+chunk.TokenCount)
			}
			if chunk.ByteCount != len(chunk.Text) {
				t.Fatalf("chunk[%d].ByteCount = %d, want %d", i, chunk.ByteCount, len(chunk.Text))
			}

			rebuilt.WriteString(chunk.Text)
		}

		if rebuilt.String() != line {
			t.Fatalf("rebuilt line = %q, want %q", rebuilt.String(), line)
		}
	})
}

func drawMapString(t *rapid.T, label string, maxLen int, alphabet []rune) string {
	t.Helper()

	length := rapid.IntRange(0, maxLen).Draw(t, label+".len")
	runes := make([]rune, 0, length)
	for i := 0; i < length; i++ {
		idx := rapid.IntRange(0, len(alphabet)-1).Draw(t, label+".rune")
		runes = append(runes, alphabet[idx])
	}
	return string(runes)
}
