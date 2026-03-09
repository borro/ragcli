package logging

import (
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
)

var (
	projectRootPath string
	projectRootOnce sync.Once
)

func Configure(debug bool) {
	slog.SetDefault(newLogger(os.Stderr, debug))
}

func newLogger(writer io.Writer, debug bool) *slog.Logger {
	level := slog.LevelError
	if debug {
		level = slog.LevelDebug
	}

	handler := slog.NewTextHandler(writer, &slog.HandlerOptions{
		Level:       level,
		AddSource:   true,
		ReplaceAttr: replaceSourceAttr,
	})
	return slog.New(handler)
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
	if root := projectRoot(); root != "" {
		if rel, err := filepath.Rel(root, cleanFile); err == nil && !strings.HasPrefix(rel, "..") {
			return filepath.ToSlash(rel)
		}
	}

	return filepath.Base(cleanFile)
}

func projectRoot() string {
	projectRootOnce.Do(func() {
		projectRootPath = defaultProjectRoot()
	})
	return projectRootPath
}

func defaultProjectRoot() string {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		return ""
	}

	return filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
}
