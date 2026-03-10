package retrieval

import (
	"fmt"
	"strings"
)

type Citation struct {
	SourcePath string
	StartLine  int
	EndLine    int
}

func AppendSources(answer string, citations []Citation) string {
	seen := make(map[string]struct{}, len(citations))
	lines := make([]string, 0, len(citations))
	for _, citation := range citations {
		value := fmt.Sprintf("%s:%d-%d", citation.SourcePath, citation.StartLine, citation.EndLine)
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		lines = append(lines, "- "+value)
	}
	if len(lines) == 0 {
		lines = append(lines, "- none")
	}
	return strings.TrimSpace(answer) + "\n\nSources:\n" + strings.Join(lines, "\n")
}
