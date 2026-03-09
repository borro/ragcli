package app

import (
	"fmt"
	"io"
	"strings"

	"github.com/charmbracelet/glamour"
	"golang.org/x/term"
)

var isTerminalWriter = func(writer io.Writer) bool {
	fdWriter, ok := writer.(interface{ Fd() uintptr })
	if !ok {
		return false
	}
	return term.IsTerminal(int(fdWriter.Fd()))
}

func formatOutput(result string, stdout io.Writer, raw bool) (string, error) {
	trimmed := strings.TrimSpace(result)
	if raw || !isTerminalWriter(stdout) || trimmed == "" {
		return trimmed, nil
	}

	rendered, err := glamour.Render(trimmed, "dark")
	if err != nil {
		return "", fmt.Errorf("render markdown: %w", err)
	}
	return strings.TrimSpace(rendered), nil
}
