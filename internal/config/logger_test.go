package config

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"testing"
)

func TestDefaultLoggerConfig(t *testing.T) {
	cfg := DefaultLoggerConfig()

	if cfg.Level != slog.LevelError {
		t.Fatalf("DefaultLoggerConfig().Level = %v, want %v", cfg.Level, slog.LevelError)
	}
	if cfg.Writer == nil {
		t.Fatal("DefaultLoggerConfig().Writer = nil, want stderr writer")
	}
	if !cfg.AddSource {
		t.Fatal("DefaultLoggerConfig().AddSource = false, want true")
	}
}

func TestConfigureLoggerSetsDebugLevelWhenVerbose(t *testing.T) {
	original := slog.Default()
	t.Cleanup(func() {
		slog.SetDefault(original)
	})

	ConfigureLogger(true)
	if !slog.Default().Enabled(context.TODO(), slog.LevelDebug) {
		t.Fatal("default logger should be enabled for debug after ConfigureLogger(true)")
	}

	ConfigureLogger(false)
	if slog.Default().Enabled(context.TODO(), slog.LevelDebug) {
		t.Fatal("default logger should not be enabled for debug after ConfigureLogger(false)")
	}
}

func TestSetLoggerNilFallsBackToDefault(t *testing.T) {
	original := slog.Default()
	t.Cleanup(func() {
		slog.SetDefault(original)
	})

	SetLogger(nil)
	if slog.Default() == nil {
		t.Fatal("slog.Default() = nil, want default logger")
	}
	if slog.Default().Enabled(context.TODO(), slog.LevelDebug) {
		t.Fatal("default logger should not enable debug")
	}
}

func TestNewLoggerFormatsSourceRelativeToRepoRoot(t *testing.T) {
	var buf bytes.Buffer
	logger := NewLogger(LoggerConfig{
		Level:     slog.LevelDebug,
		Writer:    &buf,
		AddSource: true,
	})

	logger.Debug("test message")

	output := buf.String()
	if !strings.Contains(output, "source=internal/config/logger_test.go:") {
		t.Fatalf("output = %q, want repo-relative source path", output)
	}
	if strings.Contains(output, "/home/") || strings.Contains(output, "\\") {
		t.Fatalf("output = %q, want shortened source path", output)
	}
}
