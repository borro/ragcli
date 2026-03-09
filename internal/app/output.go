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

var terminalWidth = func(writer io.Writer) int {
	fdWriter, ok := writer.(interface{ Fd() uintptr })
	if !ok {
		return 0
	}

	width, _, err := term.GetSize(int(fdWriter.Fd()))
	if err != nil || width <= 0 {
		return 0
	}
	return width
}

func formatOutput(result string, stdout io.Writer, raw bool) (string, error) {
	trimmed := strings.TrimSpace(result)
	if raw || !isTerminalWriter(stdout) || trimmed == "" {
		return trimmed, nil
	}

	options := []glamour.TermRendererOption{
		glamour.WithStandardStyle("dark"),
	}
	if width := terminalWidth(stdout); width > 0 {
		options = append(options, glamour.WithWordWrap(width))
	}

	renderer, err := glamour.NewTermRenderer(options...)
	if err != nil {
		return "", fmt.Errorf("create renderer: %w", err)
	}

	rendered, err := renderer.Render(trimmed)
	if err != nil {
		return "", fmt.Errorf("render markdown: %w", err)
	}
	return strings.TrimSpace(rendered), nil
}
