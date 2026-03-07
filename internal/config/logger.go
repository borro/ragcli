package config

import (
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

var projectRootPath string

// LoggerConfig описывает настройки runtime-логгера.
type LoggerConfig struct {
	Level     slog.Level
	Writer    io.Writer
	AddSource bool
}

func init() {
	initProjectRoot()
	ConfigureLogger(false)
}

// DefaultLoggerConfig возвращает стандартные настройки логгера приложения.
func DefaultLoggerConfig() LoggerConfig {
	return LoggerConfig{
		Level:     slog.LevelError,
		Writer:    os.Stderr,
		AddSource: true,
	}
}

// NewLogger создаёт slog.Logger на основе переданной конфигурации.
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

// SetLogger централизованно обновляет глобальный logger.
func SetLogger(logger *slog.Logger) {
	if logger == nil {
		logger = NewLogger(DefaultLoggerConfig())
	}
	slog.SetDefault(logger)
}

// ConfigureLogger применяет runtime-настройки уровня логирования.
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

func initProjectRoot() {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		return
	}

	projectRootPath = filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
}
