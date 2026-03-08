package input

import (
	"bytes"
	"os"
	"testing"
)

func TestOpenFile(t *testing.T) {
	filePath := t.TempDir() + "/sample.txt"
	if err := os.WriteFile(filePath, []byte("hello"), 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	handle, err := Open(filePath, nil)
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	t.Cleanup(func() {
		_ = handle.Close()
	})

	if handle.Path != filePath {
		t.Fatalf("Path = %q, want %q", handle.Path, filePath)
	}
	if handle.DisplayName != filePath {
		t.Fatalf("DisplayName = %q, want %q", handle.DisplayName, filePath)
	}
}

func TestOpenStdinMaterializesTempFile(t *testing.T) {
	handle, err := Open("", bytes.NewBufferString("alpha\nbeta\n"))
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}

	if handle.DisplayName != "stdin" {
		t.Fatalf("DisplayName = %q, want stdin", handle.DisplayName)
	}
	if handle.Path == "" {
		t.Fatal("Path = empty, want temp file path")
	}
	if _, err := os.Stat(handle.Path); err != nil {
		t.Fatalf("temp path stat error = %v", err)
	}

	tempPath := handle.Path
	if err := handle.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if _, err := os.Stat(tempPath); !os.IsNotExist(err) {
		t.Fatalf("temp file still exists after Close(), err = %v", err)
	}
}
