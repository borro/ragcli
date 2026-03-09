package app

import (
	"bytes"
	"io"
	"strings"
	"testing"
)

func TestFormatOutputRawSkipsRendering(t *testing.T) {
	t.Cleanup(setTerminalWriterForTest(false))

	got, err := formatOutput("# Title", &bytes.Buffer{}, true)
	if err != nil {
		t.Fatalf("formatOutput() error = %v", err)
	}
	if got != "# Title" {
		t.Fatalf("formatOutput() = %q, want raw markdown", got)
	}
}

func TestFormatOutputNonTTYSkipsRendering(t *testing.T) {
	t.Cleanup(setTerminalWriterForTest(false))

	got, err := formatOutput("# Title", &bytes.Buffer{}, false)
	if err != nil {
		t.Fatalf("formatOutput() error = %v", err)
	}
	if got != "# Title" {
		t.Fatalf("formatOutput() = %q, want raw markdown on non-tty", got)
	}
}

func TestFormatOutputTTYRendersMarkdown(t *testing.T) {
	t.Cleanup(setTerminalWriterForTest(true))
	t.Cleanup(setTerminalWidthForTest(120))

	got, err := formatOutput("# Title", &bytes.Buffer{}, false)
	if err != nil {
		t.Fatalf("formatOutput() error = %v", err)
	}
	if strings.TrimSpace(got) == "# Title" {
		t.Fatalf("formatOutput() = %q, want rendered output", got)
	}
	if !strings.Contains(got, "Title") {
		t.Fatalf("formatOutput() = %q, want rendered text content", got)
	}
}

func TestFormatOutputTTYUsesTerminalWidth(t *testing.T) {
	t.Cleanup(setTerminalWriterForTest(true))
	t.Cleanup(setTerminalWidthForTest(120))

	input := strings.Repeat("word ", 18)
	got, err := formatOutput(input, &bytes.Buffer{}, false)
	if err != nil {
		t.Fatalf("formatOutput() error = %v", err)
	}
	if strings.Contains(got, "\n") {
		t.Fatalf("formatOutput() = %q, want single line when terminal is wide enough", got)
	}
}

func setTerminalWriterForTest(value bool) func() {
	prev := isTerminalWriter
	isTerminalWriter = func(_ io.Writer) bool {
		return value
	}
	return func() {
		isTerminalWriter = prev
	}
}

func setTerminalWidthForTest(value int) func() {
	prev := terminalWidth
	terminalWidth = func(_ io.Writer) int {
		return value
	}
	return func() {
		terminalWidth = prev
	}
}
