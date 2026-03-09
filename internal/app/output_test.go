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

func setTerminalWriterForTest(value bool) func() {
	prev := isTerminalWriter
	isTerminalWriter = func(_ io.Writer) bool {
		return value
	}
	return func() {
		isTerminalWriter = prev
	}
}
