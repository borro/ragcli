package retrieval

import (
	"fmt"
	"sort"
	"strings"
)

type Citation struct {
	SourcePath string
	StartLine  int
	EndLine    int
}

func AppendSources(answer string, citations []Citation) string {
	lines := formatSources(citations)
	if len(lines) == 0 {
		lines = append(lines, "- none")
	}
	return strings.TrimSpace(answer) + "\n\nSources:\n" + strings.Join(lines, "\n")
}

type lineRange struct {
	start int
	end   int
}

func formatSources(citations []Citation) []string {
	byPath := make(map[string][]lineRange, len(citations))
	for _, citation := range citations {
		path := strings.TrimSpace(citation.SourcePath)
		if path == "" {
			continue
		}

		start := citation.StartLine
		end := citation.EndLine
		if start < 1 {
			start = 1
		}
		if end < start {
			end = start
		}

		byPath[path] = append(byPath[path], lineRange{start: start, end: end})
	}
	if len(byPath) == 0 {
		return nil
	}

	paths := make([]string, 0, len(byPath))
	for path := range byPath {
		paths = append(paths, path)
	}
	sort.Strings(paths)

	lines := make([]string, 0, len(paths))
	for _, path := range paths {
		ranges := mergeRanges(byPath[path])
		formatted := make([]string, 0, len(ranges))
		for _, span := range ranges {
			if span.start == span.end {
				formatted = append(formatted, fmt.Sprintf("%d", span.start))
				continue
			}
			formatted = append(formatted, fmt.Sprintf("%d-%d", span.start, span.end))
		}
		lines = append(lines, "- "+path+": "+strings.Join(formatted, ", "))
	}
	return lines
}

func mergeRanges(ranges []lineRange) []lineRange {
	if len(ranges) == 0 {
		return nil
	}

	sorted := append([]lineRange(nil), ranges...)
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i].start != sorted[j].start {
			return sorted[i].start < sorted[j].start
		}
		return sorted[i].end < sorted[j].end
	})

	merged := make([]lineRange, 0, len(sorted))
	current := sorted[0]
	for _, next := range sorted[1:] {
		if next.start <= current.end {
			if next.end > current.end {
				current.end = next.end
			}
			continue
		}
		merged = append(merged, current)
		current = next
	}
	merged = append(merged, current)
	return merged
}
