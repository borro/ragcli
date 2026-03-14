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

type EvidenceKind string

const (
	EvidenceRetrieved EvidenceKind = "retrieved"
	EvidenceVerified  EvidenceKind = "verified"
)

type Evidence struct {
	Kind       EvidenceKind
	SourcePath string
	StartLine  int
	EndLine    int
}

func AppendSources(answer string, citations []Citation) string {
	lines := formatSources(citations)
	if len(lines) == 0 {
		lines = append(lines, "- none")
	}
	return strings.TrimSpace(answer) + "\n\nSources:\n\n" + strings.Join(lines, "\n")
}

func AppendEvidenceSources(answer string, evidence []Evidence) string {
	verified := evidenceByKind(evidence, EvidenceVerified)
	retrieved := evidenceByKind(evidence, EvidenceRetrieved)

	verifiedLines := formatLineRanges(verified)
	retrievedLines := formatLineRanges(subtractEvidenceRanges(retrieved, verified))

	lines := make([]string, 0, len(verifiedLines)+len(retrievedLines)+2)
	if len(verifiedLines) > 0 {
		lines = append(lines, "Verified:")
		lines = append(lines, "")
		lines = append(lines, verifiedLines...)
	}
	if len(retrievedLines) > 0 {
		if len(lines) > 0 {
			lines = append(lines, "")
		}
		lines = append(lines, "Retrieved:")
		lines = append(lines, "")
		lines = append(lines, retrievedLines...)
	}
	if len(lines) == 0 {
		lines = append(lines, "- none")
	}

	return strings.TrimSpace(answer) + "\n\nSources:\n\n" + strings.Join(lines, "\n")
}

type lineRange struct {
	start int
	end   int
}

func formatSources(citations []Citation) []string {
	byPath := make(map[string][]lineRange, len(citations))
	for _, citation := range citations {
		appendLineRange(byPath, citation.SourcePath, citation.StartLine, citation.EndLine)
	}
	return formatLineRanges(byPath)
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

func evidenceByKind(evidence []Evidence, kind EvidenceKind) map[string][]lineRange {
	byPath := make(map[string][]lineRange, len(evidence))
	for _, item := range evidence {
		if item.Kind != kind {
			continue
		}
		appendLineRange(byPath, item.SourcePath, item.StartLine, item.EndLine)
	}
	return byPath
}

func appendLineRange(byPath map[string][]lineRange, sourcePath string, startLine int, endLine int) {
	path := strings.TrimSpace(sourcePath)
	if path == "" {
		return
	}

	start := startLine
	end := endLine
	if start < 1 {
		start = 1
	}
	if end < start {
		end = start
	}

	byPath[path] = append(byPath[path], lineRange{start: start, end: end})
}

func formatLineRanges(byPath map[string][]lineRange) []string {
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
		lines = append(lines, "- "+path+":")
		lines = append(lines, "  "+strings.Join(formatted, ", "))
	}
	return lines
}

func subtractEvidenceRanges(retrieved map[string][]lineRange, verified map[string][]lineRange) map[string][]lineRange {
	if len(retrieved) == 0 {
		return nil
	}

	result := make(map[string][]lineRange, len(retrieved))
	for path, ranges := range retrieved {
		remaining := mergeRanges(ranges)
		subtractors := mergeRanges(verified[path])
		for _, subtractor := range subtractors {
			remaining = subtractRanges(remaining, subtractor)
			if len(remaining) == 0 {
				break
			}
		}
		if len(remaining) == 0 {
			continue
		}
		result[path] = remaining
	}

	return result
}

func subtractRanges(ranges []lineRange, subtractor lineRange) []lineRange {
	if len(ranges) == 0 {
		return nil
	}

	result := make([]lineRange, 0, len(ranges))
	for _, current := range ranges {
		if subtractor.end < current.start || subtractor.start > current.end {
			result = append(result, current)
			continue
		}
		if subtractor.start > current.start {
			result = append(result, lineRange{start: current.start, end: subtractor.start - 1})
		}
		if subtractor.end < current.end {
			result = append(result, lineRange{start: subtractor.end + 1, end: current.end})
		}
	}
	return result
}
