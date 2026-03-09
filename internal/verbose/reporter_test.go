package verbose

import (
	"bytes"
	"strings"
	"testing"
)

func TestNewReturnsStateReporterWhenEnabled(t *testing.T) {
	reporter := New(&bytes.Buffer{}, true, false)
	if _, ok := reporter.(*stateReporter); !ok {
		t.Fatalf("New() = %T, want *stateReporter", reporter)
	}
}

func TestNewDisablesVerboseWhenDebugEnabled(t *testing.T) {
	reporter := New(&bytes.Buffer{}, true, true)
	if _, ok := reporter.(nopReporter); !ok {
		t.Fatalf("New() = %T, want nopReporter", reporter)
	}
}

func TestRenderLineUsesSingleLineFormat(t *testing.T) {
	view := renderLine("rag", "retrieval", "строю индекс", 3, 5)
	if strings.Contains(view, "\n") {
		t.Fatalf("renderLine() returned multiline output: %q", view)
	}
	want := "rag (60%): retrieval (строю индекс)"
	if view != want {
		t.Fatalf("renderLine() = %q, want %q", view, want)
	}
}

func TestRenderLineAlignsPercentField(t *testing.T) {
	view := renderLine("rag", "retrieval", "строю индекс", 1, 100)
	want := "rag  (1%): retrieval (строю индекс)"
	if view != want {
		t.Fatalf("renderLine() = %q, want %q", view, want)
	}
}

func TestStateReporterSkipsDuplicateLines(t *testing.T) {
	var buf bytes.Buffer
	reporter := New(&buf, true, false)

	reporter.ModeStarted("rag", 10)
	reporter.StageChanged("retrieval", "строю индекс", 1)
	reporter.StageChanged("retrieval", "строю индекс", 1)

	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	if len(lines) != 1 {
		t.Fatalf("lines = %d, want 1; output = %q", len(lines), buf.String())
	}
}
