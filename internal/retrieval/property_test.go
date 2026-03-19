package retrieval

import (
	"math"
	"slices"
	"strings"
	"testing"
	"unicode"
	"unicode/utf8"

	"github.com/borro/ragcli/internal/testutil"
	"pgregory.net/rapid"
)

func TestWalkLines_PropertyPreservesOffsetsAndContent(t *testing.T) {
	testutil.RapidCheck(t, func(t *rapid.T) {
		input := drawRetrievalString(t, "input", 48, []rune{'a', 'B', 'ж', 'Я', '🙂', '1', ' ', '\t', '\n', '\r', '-', '_'})

		type lineRecord struct {
			raw       string
			text      string
			lineNo    int
			byteStart int
			byteEnd   int
		}

		var records []lineRecord
		written, err := WalkLines(strings.NewReader(input), func(rawLine, text string, lineNo, byteStart, byteEnd int) error {
			records = append(records, lineRecord{
				raw:       rawLine,
				text:      text,
				lineNo:    lineNo,
				byteStart: byteStart,
				byteEnd:   byteEnd,
			})
			return nil
		})
		if err != nil {
			t.Fatalf("WalkLines() error = %v", err)
		}
		if written != len(input) {
			t.Fatalf("written = %d, want %d", written, len(input))
		}

		var rebuilt strings.Builder
		byteOffset := 0
		for i, record := range records {
			if record.lineNo != i+1 {
				t.Fatalf("lineNo[%d] = %d, want %d", i, record.lineNo, i+1)
			}
			if record.byteStart != byteOffset {
				t.Fatalf("byteStart[%d] = %d, want %d", i, record.byteStart, byteOffset)
			}
			if record.byteEnd != record.byteStart+len(record.raw) {
				t.Fatalf("byteEnd[%d] = %d, want %d", i, record.byteEnd, record.byteStart+len(record.raw))
			}
			if record.text != trimTrailingNewline(record.raw) {
				t.Fatalf("text[%d] = %q, want %q", i, record.text, trimTrailingNewline(record.raw))
			}

			rebuilt.WriteString(record.raw)
			byteOffset = record.byteEnd
		}

		if rebuilt.String() != input {
			t.Fatalf("rebuilt input = %q, want %q", rebuilt.String(), input)
		}
		if byteOffset != written {
			t.Fatalf("final byte offset = %d, want %d", byteOffset, written)
		}
	})
}

func TestMergeRanges_PropertyPreservesUnionAndNormalizesOrder(t *testing.T) {
	testutil.RapidCheck(t, func(t *rapid.T) {
		ranges := drawLineRanges(t, "ranges", 12, -5, 20)
		merged := mergeRanges(ranges)

		if !sameCoveredPoints(coveredPoints(ranges), coveredPoints(merged)) {
			t.Fatalf("mergeRanges() changed coverage: input=%v merged=%v", ranges, merged)
		}
		if !slices.Equal(merged, mergeRanges(merged)) {
			t.Fatalf("mergeRanges() is not idempotent: %v", merged)
		}

		for i, span := range merged {
			if span.start > span.end {
				t.Fatalf("merged[%d] = %+v, want normalized range", i, span)
			}
			if i == 0 {
				continue
			}

			prev := merged[i-1]
			if prev.start > span.start {
				t.Fatalf("merged ranges are not sorted: prev=%+v current=%+v", prev, span)
			}
			if prev.end >= span.start {
				t.Fatalf("merged ranges still overlap: prev=%+v current=%+v", prev, span)
			}
		}
	})
}

func TestSubtractRanges_PropertyMatchesSetDifference(t *testing.T) {
	testutil.RapidCheck(t, func(t *rapid.T) {
		base := mergeRanges(drawLineRanges(t, "base", 12, -5, 20))
		subtractor := drawLineRange(t, "subtractor", -5, 20)

		got := subtractRanges(base, subtractor)
		want := subtractCoveredPoints(coveredPoints(base), subtractor)
		if !sameCoveredPoints(want, coveredPoints(got)) {
			t.Fatalf("subtractRanges() changed set difference: base=%v subtractor=%+v got=%v", base, subtractor, got)
		}

		for i, span := range got {
			if span.start > span.end {
				t.Fatalf("got[%d] = %+v, want normalized range", i, span)
			}
			if i == 0 {
				continue
			}

			prev := got[i-1]
			if prev.start > span.start {
				t.Fatalf("subtractRanges() returned unsorted ranges: prev=%+v current=%+v", prev, span)
			}
			if prev.end >= span.start {
				t.Fatalf("subtractRanges() returned overlapping ranges: prev=%+v current=%+v", prev, span)
			}
		}
	})
}

func TestTokenize_PropertyNormalizesDistinctLowercaseTokens(t *testing.T) {
	testutil.RapidCheck(t, func(t *rapid.T) {
		input := drawRetrievalString(t, "input", 48, []rune{'A', 'a', 'B', 'b', 'Ж', 'ж', 'X', 'x', '2', '4', ' ', '\t', '\n', ',', '.', '-', '_', '/', ':'})
		tokens := Tokenize(input)
		lowerInput := strings.ToLower(input)
		seen := make(map[string]struct{}, len(tokens))

		for i, token := range tokens {
			if token != strings.ToLower(token) {
				t.Fatalf("tokens[%d] = %q, want lowercase", i, token)
			}
			if utf8.RuneCountInString(token) < 2 {
				t.Fatalf("tokens[%d] = %q, want length >= 2", i, token)
			}
			if _, exists := seen[token]; exists {
				t.Fatalf("tokens[%d] = %q, want distinct tokens (%v)", i, token, tokens)
			}
			seen[token] = struct{}{}

			for _, r := range token {
				if !unicode.IsLetter(r) && !unicode.IsNumber(r) {
					t.Fatalf("tokens[%d] = %q contains non-token rune %q", i, token, r)
				}
			}
			if !strings.Contains(lowerInput, token) {
				t.Fatalf("tokens[%d] = %q is not present in normalized input %q", i, token, lowerInput)
			}
		}
	})
}

func TestTokenOverlapRatio_PropertyMatchesMatchedTokenFraction(t *testing.T) {
	testutil.RapidCheck(t, func(t *rapid.T) {
		query := drawRetrievalString(t, "query", 36, []rune{'A', 'a', 'B', 'b', 'Ж', 'ж', 'X', 'x', '2', '4', ' ', '\t', '\n', ',', '.', '-', '_', '/', ':'})
		text := drawRetrievalString(t, "text", 36, []rune{'A', 'a', 'B', 'b', 'Ж', 'ж', 'X', 'x', '2', '4', ' ', '\t', '\n', ',', '.', '-', '_', '/', ':'})

		got := TokenOverlapRatio(query, text)
		if got < 0 || got > 1 {
			t.Fatalf("TokenOverlapRatio() = %v, want value in [0,1]", got)
		}

		queryTokens := Tokenize(query)
		if len(queryTokens) == 0 {
			if got != 0 {
				t.Fatalf("TokenOverlapRatio() = %v, want 0 for empty token set", got)
			}
			return
		}

		matched := 0
		textLower := strings.ToLower(text)
		for _, token := range queryTokens {
			if strings.Contains(textLower, token) {
				matched++
			}
		}

		want := float64(matched) / float64(len(queryTokens))
		if math.Abs(got-want) > 1e-12 {
			t.Fatalf("TokenOverlapRatio() = %v, want %v for query=%q text=%q", got, want, query, text)
		}
	})
}

func drawRetrievalString(t *rapid.T, label string, maxLen int, alphabet []rune) string {
	t.Helper()

	length := rapid.IntRange(0, maxLen).Draw(t, label+".len")
	runes := make([]rune, 0, length)
	for i := 0; i < length; i++ {
		idx := rapid.IntRange(0, len(alphabet)-1).Draw(t, label+".rune")
		runes = append(runes, alphabet[idx])
	}
	return string(runes)
}

func drawLineRanges(t *rapid.T, label string, maxCount int, minValue int, maxValue int) []lineRange {
	t.Helper()

	count := rapid.IntRange(0, maxCount).Draw(t, label+".count")
	ranges := make([]lineRange, 0, count)
	for i := 0; i < count; i++ {
		ranges = append(ranges, drawLineRange(t, label, minValue, maxValue))
	}
	return ranges
}

func drawLineRange(t *rapid.T, label string, minValue int, maxValue int) lineRange {
	t.Helper()

	start := rapid.IntRange(minValue, maxValue).Draw(t, label+".start")
	end := rapid.IntRange(minValue, maxValue).Draw(t, label+".end")
	if start > end {
		start, end = end, start
	}
	return lineRange{start: start, end: end}
}

func coveredPoints(ranges []lineRange) map[int]struct{} {
	points := make(map[int]struct{})
	for _, span := range ranges {
		for point := span.start; point <= span.end; point++ {
			points[point] = struct{}{}
		}
	}
	return points
}

func subtractCoveredPoints(points map[int]struct{}, subtractor lineRange) map[int]struct{} {
	remaining := make(map[int]struct{}, len(points))
	for point := range points {
		if point >= subtractor.start && point <= subtractor.end {
			continue
		}
		remaining[point] = struct{}{}
	}
	return remaining
}

func sameCoveredPoints(left map[int]struct{}, right map[int]struct{}) bool {
	if len(left) != len(right) {
		return false
	}
	for point := range left {
		if _, ok := right[point]; !ok {
			return false
		}
	}
	return true
}
