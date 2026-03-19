package files

import (
	"encoding/json"
	"fmt"
	"testing"

	"pgregory.net/rapid"
)

func TestBuildLineSlices_PropertyClampsAndPreservesOrder(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		lines := drawFileToolLines(t, "lines", 12, 24)
		displayPath := drawFileToolPath(t, "path", 24)
		start := rapid.IntRange(-5, 20).Draw(t, "start")
		end := rapid.IntRange(-5, 20).Draw(t, "end")

		got := buildLineSlices(lines, displayPath, start, end)
		want := expectedLineSlices(lines, displayPath, start, end)
		if len(got) != len(want) {
			t.Fatalf("len(buildLineSlices()) = %d, want %d", len(got), len(want))
		}

		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("buildLineSlices()[%d] = %+v, want %+v", i, got[i], want[i])
			}
			if i > 0 && got[i-1].LineNumber >= got[i].LineNumber {
				t.Fatalf("line numbers are not strictly increasing: prev=%+v current=%+v", got[i-1], got[i])
			}
		}
	})
}

func TestReadLinesFromLines_PropertyReturnsConsistentJSON(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		lines := drawFileToolLines(t, "lines", 12, 24)
		displayPath := drawFileToolPath(t, "path", 24)
		start := rapid.IntRange(-5, 20).Draw(t, "start")
		end := rapid.IntRange(-5, 20).Draw(t, "end")

		raw, err := readLinesFromLines(lines, displayPath, start, end)
		if err != nil {
			t.Fatalf("readLinesFromLines() error = %v", err)
		}

		var result ReadLinesResult
		if err := json.Unmarshal([]byte(raw), &result); err != nil {
			t.Fatalf("json.Unmarshal() error = %v, raw=%s", err, raw)
		}

		wantStart := max(start, 1)
		wantLines := expectedLineSlices(lines, displayPath, start, end)
		if result.Path != displayPath {
			t.Fatalf("Path = %q, want %q", result.Path, displayPath)
		}
		if result.StartLine != wantStart {
			t.Fatalf("StartLine = %d, want %d", result.StartLine, wantStart)
		}
		if result.EndLine != end {
			t.Fatalf("EndLine = %d, want %d", result.EndLine, end)
		}
		if result.LineCount != len(result.Lines) {
			t.Fatalf("LineCount = %d, want len(Lines)=%d", result.LineCount, len(result.Lines))
		}
		if len(result.Lines) != len(wantLines) {
			t.Fatalf("len(Lines) = %d, want %d", len(result.Lines), len(wantLines))
		}
		for i := range wantLines {
			if result.Lines[i] != wantLines[i] {
				t.Fatalf("Lines[%d] = %+v, want %+v", i, result.Lines[i], wantLines[i])
			}
		}
	})
}

func TestReadAroundFromLines_PropertyReturnsClampedWindow(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		lines := drawFileToolLines(t, "lines", 12, 24)
		displayPath := drawFileToolPath(t, "path", 24)
		line := rapid.IntRange(-2, 20).Draw(t, "line")
		before := rapid.IntRange(-2, 8).Draw(t, "before")
		after := rapid.IntRange(-2, 8).Draw(t, "after")

		raw, err := readAroundFromLines(lines, displayPath, line, before, after)
		if line < 1 {
			if err == nil {
				t.Fatalf("readAroundFromLines() error = nil, want invalid_arguments for line=%d", line)
			}
			toolErr := AsToolError(err)
			if toolErr.Code != "invalid_arguments" {
				t.Fatalf("error code = %q, want invalid_arguments", toolErr.Code)
			}
			return
		}
		if err != nil {
			t.Fatalf("readAroundFromLines() error = %v", err)
		}

		var result ReadAroundResult
		if err := json.Unmarshal([]byte(raw), &result); err != nil {
			t.Fatalf("json.Unmarshal() error = %v, raw=%s", err, raw)
		}

		wantBefore := before
		if wantBefore < 0 {
			wantBefore = DefaultReadAroundBefore
		}
		wantAfter := after
		if wantAfter < 0 {
			wantAfter = DefaultReadAroundAfter
		}
		wantStart, wantEnd, wantLines := expectedReadAround(lines, displayPath, line, wantBefore, wantAfter)

		if result.Path != displayPath {
			t.Fatalf("Path = %q, want %q", result.Path, displayPath)
		}
		if result.Line != line {
			t.Fatalf("Line = %d, want %d", result.Line, line)
		}
		if result.Before != wantBefore || result.After != wantAfter {
			t.Fatalf("window = %d/%d, want %d/%d", result.Before, result.After, wantBefore, wantAfter)
		}
		if result.StartLine != wantStart || result.EndLine != wantEnd {
			t.Fatalf("range = %d..%d, want %d..%d", result.StartLine, result.EndLine, wantStart, wantEnd)
		}
		if result.LineCount != len(result.Lines) {
			t.Fatalf("LineCount = %d, want len(Lines)=%d", result.LineCount, len(result.Lines))
		}
		if len(result.Lines) != len(wantLines) {
			t.Fatalf("len(Lines) = %d, want %d", len(result.Lines), len(wantLines))
		}
		for i := range wantLines {
			if result.Lines[i] != wantLines[i] {
				t.Fatalf("Lines[%d] = %+v, want %+v", i, result.Lines[i], wantLines[i])
			}
		}
	})
}

func TestSortTokenOverlapMatches_PropertyOrdersAndPreservesEqualKeyOrder(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		count := rapid.IntRange(0, 20).Draw(t, "count")
		matches := make([]SearchMatch, 0, count)
		originalIndex := make(map[string]int, count)
		for i := 0; i < count; i++ {
			id := fmt.Sprintf("match-%02d", i)
			matches = append(matches, SearchMatch{
				Content:    id,
				Path:       drawFileToolPath(t, "path", 12),
				LineNumber: rapid.IntRange(1, 30).Draw(t, "line"),
				Score:      float64(rapid.IntRange(0, 1000).Draw(t, "score")) / 1000,
			})
			originalIndex[id] = i
		}

		sortTokenOverlapMatches(matches)
		for i := 1; i < len(matches); i++ {
			prev := matches[i-1]
			current := matches[i]

			switch {
			case prev.Score < current.Score:
				t.Fatalf("score order violated: prev=%+v current=%+v", prev, current)
			case prev.Score > current.Score:
				continue
			}

			switch {
			case prev.Path > current.Path:
				t.Fatalf("path order violated: prev=%+v current=%+v", prev, current)
			case prev.Path < current.Path:
				continue
			}

			switch {
			case prev.LineNumber > current.LineNumber:
				t.Fatalf("line order violated: prev=%+v current=%+v", prev, current)
			case prev.LineNumber < current.LineNumber:
				continue
			}

			if originalIndex[prev.Content] > originalIndex[current.Content] {
				t.Fatalf("stable order violated for equal keys: prev=%+v current=%+v", prev, current)
			}
		}
	})
}

func drawFileToolLines(t *rapid.T, label string, maxCount int, maxLen int) []string {
	t.Helper()

	count := rapid.IntRange(0, maxCount).Draw(t, label+".count")
	lines := make([]string, 0, count)
	for i := 0; i < count; i++ {
		lines = append(lines, drawFileToolString(t, label+".line", maxLen, []rune{'a', 'B', 'ж', 'Я', '1', ' ', '\t', '-', '_', '/', '.'}))
	}
	return lines
}

func drawFileToolPath(t *rapid.T, label string, maxLen int) string {
	t.Helper()
	return drawFileToolString(t, label, maxLen, []rune{'a', 'B', 'ж', 'Я', '1', '-', '_', '/', '.'})
}

func drawFileToolString(t *rapid.T, label string, maxLen int, alphabet []rune) string {
	t.Helper()

	length := rapid.IntRange(0, maxLen).Draw(t, label+".len")
	runes := make([]rune, 0, length)
	for i := 0; i < length; i++ {
		idx := rapid.IntRange(0, len(alphabet)-1).Draw(t, label+".rune")
		runes = append(runes, alphabet[idx])
	}
	return string(runes)
}

func expectedLineSlices(lines []string, displayPath string, start int, end int) []LineSlice {
	if start < 1 {
		start = 1
	}
	if end > len(lines) {
		end = len(lines)
	}
	if end < start {
		return []LineSlice{}
	}

	result := make([]LineSlice, 0, end-start+1)
	for lineNumber := start; lineNumber <= end; lineNumber++ {
		result = append(result, LineSlice{
			LineNumber: lineNumber,
			Content:    lines[lineNumber-1],
			Path:       displayPath,
		})
	}
	return result
}

func expectedReadAround(lines []string, displayPath string, line int, before int, after int) (int, int, []LineSlice) {
	start := max(line-before, 1)
	end := min(line+after, len(lines))
	if len(lines) == 0 {
		return line, line - 1, []LineSlice{}
	}
	return start, end, expectedLineSlices(lines, displayPath, start, end)
}
