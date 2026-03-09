package logging

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"testing"
)

func TestConfigureEnablesDebugLevelWhenDebugEnabled(t *testing.T) {
	original := slog.Default()
	t.Cleanup(func() {
		slog.SetDefault(original)
	})

	Configure(&bytes.Buffer{}, true)
	if !slog.Default().Enabled(context.TODO(), slog.LevelDebug) {
		t.Fatal("default logger should be enabled for debug after Configure(true)")
	}
}

func TestConfigureDisablesDebugLevelByDefault(t *testing.T) {
	original := slog.Default()
	t.Cleanup(func() {
		slog.SetDefault(original)
	})

	Configure(&bytes.Buffer{}, false)
	if slog.Default().Enabled(context.TODO(), slog.LevelDebug) {
		t.Fatal("default logger should not enable debug")
	}
}

func TestConfigureDiscardsErrorsWhenDebugDisabled(t *testing.T) {
	original := slog.Default()
	t.Cleanup(func() {
		slog.SetDefault(original)
	})

	var buf bytes.Buffer
	Configure(&buf, false)
	slog.Error("hidden error")

	if buf.Len() != 0 {
		t.Fatalf("buffer = %q, want no log output when debug is disabled", buf.String())
	}
}

func TestLoggerFormatsSourceRelativeToRepoRoot(t *testing.T) {
	var buf bytes.Buffer
	logger := newTestLogger(&buf, true)

	logger.Debug("test message")

	output := buf.String()
	if !strings.Contains(output, "source=internal/logging/logger_test.go:") {
		t.Fatalf("output = %q, want repo-relative source path", output)
	}
	if strings.Contains(output, "/home/") || strings.Contains(output, "\\") {
		t.Fatalf("output = %q, want shortened source path", output)
	}
}

func newTestLogger(buf *bytes.Buffer, debug bool) *slog.Logger {
	level := slog.LevelError
	if debug {
		level = slog.LevelDebug
	}

	handler := slog.NewTextHandler(buf, &slog.HandlerOptions{
		Level:       level,
		AddSource:   true,
		ReplaceAttr: replaceSourceAttr,
	})
	return slog.New(handler)
}
