package logging

import (
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

var projectRootPath string

type LoggerConfig struct {
	Level     slog.Level
	Writer    io.Writer
	AddSource bool
}

func InitProjectRoot(root string) {
	if strings.TrimSpace(root) == "" {
		root = defaultProjectRoot()
	}
	projectRootPath = filepath.Clean(root)
}

func DefaultLoggerConfig() LoggerConfig {
	return LoggerConfig{
		Level:     slog.LevelError,
		Writer:    os.Stderr,
		AddSource: true,
	}
}

func NewLogger(cfg LoggerConfig) *slog.Logger {
	writer := cfg.Writer
	if writer == nil {
		writer = os.Stderr
	}

	handler := slog.NewTextHandler(writer, &slog.HandlerOptions{
		Level:       cfg.Level,
		AddSource:   cfg.AddSource,
		ReplaceAttr: replaceSourceAttr,
	})
	return slog.New(handler)
}

func SetLogger(logger *slog.Logger) {
	if logger == nil {
		logger = NewLogger(DefaultLoggerConfig())
	}
	slog.SetDefault(logger)
}

func ConfigureLogger(verbose bool) {
	cfg := DefaultLoggerConfig()
	if verbose {
		cfg.Level = slog.LevelDebug
	}
	SetLogger(NewLogger(cfg))
}

func replaceSourceAttr(_ []string, attr slog.Attr) slog.Attr {
	if attr.Key != slog.SourceKey {
		return attr
	}

	source, ok := attr.Value.Any().(*slog.Source)
	if !ok || source == nil {
		return attr
	}

	shortFile := shortenSourcePath(source.File)
	if shortFile == "" {
		return attr
	}

	return slog.Any(attr.Key, &slog.Source{
		File:     shortFile,
		Line:     source.Line,
		Function: source.Function,
	})
}

func shortenSourcePath(file string) string {
	if file == "" {
		return file
	}

	cleanFile := filepath.Clean(file)
	if projectRootPath != "" {
		if rel, err := filepath.Rel(projectRootPath, cleanFile); err == nil && !strings.HasPrefix(rel, "..") {
			return filepath.ToSlash(rel)
		}
	}

	return filepath.Base(cleanFile)
}

func defaultProjectRoot() string {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		return ""
	}

	return filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
}
