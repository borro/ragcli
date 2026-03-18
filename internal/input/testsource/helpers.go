package testsource

import (
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/borro/ragcli/internal/input"
)

// DisplayName mirrors input.Source display-name selection for test doubles.
func DisplayName(kind input.Kind, inputPath string, snapshotPath string, files []input.File) string {
	if path := strings.TrimSpace(inputPath); path != "" {
		return path
	}
	if kind == input.KindStdin {
		return "stdin"
	}
	if len(files) > 0 && strings.TrimSpace(files[0].DisplayPath) != "" {
		return files[0].DisplayPath
	}
	return strings.TrimSpace(snapshotPath)
}

// WriteTempFile materializes reader content into a temp file and returns its path.
func WriteTempFile(reader io.Reader, name string) string {
	tmpFile, err := os.CreateTemp("", "rag-test-*"+filepath.Ext(name))
	if err != nil {
		panic(err)
	}
	if _, err := io.Copy(tmpFile, reader); err != nil {
		_ = tmpFile.Close()
		panic(err)
	}
	if err := tmpFile.Close(); err != nil {
		panic(err)
	}
	return tmpFile.Name()
}
