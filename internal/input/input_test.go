package input

import (
	"bytes"
	"errors"
	"io"
	"os"
	"strings"
	"testing"

	"github.com/borro/ragcli/internal/localize"
)

type failingReader struct {
	err error
}

func (r failingReader) Read(_ []byte) (int, error) {
	return 0, r.err
}

type fakeTempFile struct {
	bytes.Buffer
	name     string
	closeErr error
}

func (f *fakeTempFile) Close() error {
	return f.closeErr
}

func (f *fakeTempFile) Name() string {
	return f.name
}

func stubInputDeps(t *testing.T) {
	t.Helper()

	originalOpenFile := openFile
	originalCreateTemp := createTemp
	originalDefaultStdin := defaultStdin
	t.Cleanup(func() {
		openFile = originalOpenFile
		createTemp = originalCreateTemp
		defaultStdin = originalDefaultStdin
	})
}

func initLocale(t *testing.T) {
	t.Helper()
	if err := localize.SetCurrent(localize.EN); err != nil {
		t.Fatalf("SetCurrent(en) error = %v", err)
	}
}

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

	content, err := io.ReadAll(handle.Reader)
	if err != nil {
		t.Fatalf("ReadAll() error = %v", err)
	}
	if string(content) != "hello" {
		t.Fatalf("content = %q, want hello", string(content))
	}
	if handle.Path != filePath {
		t.Fatalf("Path = %q, want %q", handle.Path, filePath)
	}
	if handle.DisplayName != filePath {
		t.Fatalf("DisplayName = %q, want %q", handle.DisplayName, filePath)
	}

	source := handle.Source()
	if source.Reader == nil {
		t.Fatal("Source().Reader = nil, want file reader")
	}
	if source.Path != filePath {
		t.Fatalf("Source().Path = %q, want %q", source.Path, filePath)
	}
	if source.DisplayName != filePath {
		t.Fatalf("Source().DisplayName = %q, want %q", source.DisplayName, filePath)
	}
}

func TestOpenFileMissing(t *testing.T) {
	initLocale(t)
	_, err := Open(t.TempDir()+"/missing.txt", nil)
	if err == nil {
		t.Fatal("Open() error = nil, want error")
	}
	if !strings.Contains(err.Error(), "failed to open file") {
		t.Fatalf("error = %q, want open file context", err)
	}
}

func TestOpenStdinMaterializesTempFile(t *testing.T) {
	handle, err := Open("", bytes.NewBufferString("alpha\nbeta\n"))
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}

	content, err := io.ReadAll(handle.Reader)
	if err != nil {
		t.Fatalf("ReadAll() error = %v", err)
	}
	if string(content) != "alpha\nbeta\n" {
		t.Fatalf("content = %q, want exact stdin content", string(content))
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

	source := handle.Source()
	if source.Reader == nil {
		t.Fatal("Source().Reader = nil, want stdin temp reader")
	}
	if source.Path != handle.Path {
		t.Fatalf("Source().Path = %q, want %q", source.Path, handle.Path)
	}
	if source.DisplayName != "stdin" {
		t.Fatalf("Source().DisplayName = %q, want stdin", source.DisplayName)
	}

	tempPath := handle.Path
	if err := handle.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if _, err := os.Stat(tempPath); !os.IsNotExist(err) {
		t.Fatalf("temp file still exists after Close(), err = %v", err)
	}
}

func TestOpenUsesDefaultStdinWhenNil(t *testing.T) {
	stubInputDeps(t)
	defaultStdin = func() io.Reader {
		return bytes.NewBufferString("from default stdin")
	}

	handle, err := Open("", nil)
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	t.Cleanup(func() {
		_ = handle.Close()
	})

	content, err := io.ReadAll(handle.Reader)
	if err != nil {
		t.Fatalf("ReadAll() error = %v", err)
	}
	if string(content) != "from default stdin" {
		t.Fatalf("content = %q, want default stdin content", string(content))
	}
}

func TestOpenCreateTempError(t *testing.T) {
	initLocale(t)
	stubInputDeps(t)
	wantErr := errors.New("create temp failed")
	createTemp = func(_, _ string) (tempFile, error) {
		return nil, wantErr
	}

	_, err := Open("", bytes.NewBufferString("data"))
	if err == nil {
		t.Fatal("Open() error = nil, want error")
	}
	if !strings.Contains(err.Error(), "failed to create temp file") {
		t.Fatalf("error = %q, want create temp context", err)
	}
	if !strings.Contains(err.Error(), wantErr.Error()) {
		t.Fatalf("error = %q, want wrapped cause %q", err, wantErr)
	}
}

func TestOpenReadStdinError(t *testing.T) {
	initLocale(t)
	stubInputDeps(t)
	wantErr := errors.New("stdin read failed")

	_, err := Open("", failingReader{err: wantErr})
	if err == nil {
		t.Fatal("Open() error = nil, want error")
	}
	if !strings.Contains(err.Error(), "failed to read stdin") {
		t.Fatalf("error = %q, want read stdin context", err)
	}
	if !strings.Contains(err.Error(), wantErr.Error()) {
		t.Fatalf("error = %q, want wrapped cause %q", err, wantErr)
	}
}

func TestOpenCloseTempFileError(t *testing.T) {
	initLocale(t)
	stubInputDeps(t)
	wantErr := errors.New("close temp failed")
	createTemp = func(_, _ string) (tempFile, error) {
		return &fakeTempFile{name: t.TempDir() + "/stdin.txt", closeErr: wantErr}, nil
	}

	_, err := Open("", bytes.NewBufferString("data"))
	if err == nil {
		t.Fatal("Open() error = nil, want error")
	}
	if !strings.Contains(err.Error(), "failed to close temp file") {
		t.Fatalf("error = %q, want close temp context", err)
	}
	if !strings.Contains(err.Error(), wantErr.Error()) {
		t.Fatalf("error = %q, want wrapped cause %q", err, wantErr)
	}
}

func TestOpenReopenTempFileError(t *testing.T) {
	initLocale(t)
	stubInputDeps(t)
	wantErr := errors.New("reopen failed")
	originalOpenFile := openFile
	openFile = func(path string) (readCloser, error) {
		if strings.Contains(path, "ragcli-stdin-") {
			return nil, wantErr
		}
		return originalOpenFile(path)
	}

	handle, err := Open("", bytes.NewBufferString("data"))
	if err == nil {
		if handle != nil {
			_ = handle.Close()
		}
		t.Fatal("Open() error = nil, want error")
	}
	if !strings.Contains(err.Error(), "failed to reopen temp file") {
		t.Fatalf("error = %q, want reopen context", err)
	}
	if !strings.Contains(err.Error(), wantErr.Error()) {
		t.Fatalf("error = %q, want wrapped cause %q", err, wantErr)
	}
}

func TestHandleCloseNilIsNoop(t *testing.T) {
	var handle *Handle
	if err := handle.Close(); err != nil {
		t.Fatalf("Close() error = %v, want nil", err)
	}
}

func TestHandleCloseWithoutCloserIsNoop(t *testing.T) {
	handle := &Handle{}
	if err := handle.Close(); err != nil {
		t.Fatalf("Close() error = %v, want nil", err)
	}
}

func TestHandleCloseReturnsCloseError(t *testing.T) {
	wantErr := errors.New("close failed")
	handle := &Handle{
		closeFn: func() error {
			return wantErr
		},
	}

	err := handle.Close()
	if !errors.Is(err, wantErr) {
		t.Fatalf("Close() error = %v, want %v", err, wantErr)
	}
}
