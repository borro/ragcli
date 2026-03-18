package input

import (
	"bytes"
	"errors"
	"io"
	"os"
	"path/filepath"
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
	writeErr error
}

func (f *fakeTempFile) Write(p []byte) (int, error) {
	if f.writeErr != nil {
		return 0, f.writeErr
	}
	return f.Buffer.Write(p)
}

func (f *fakeTempFile) WriteString(s string) (int, error) {
	if f.writeErr != nil {
		return 0, f.writeErr
	}
	return f.Buffer.WriteString(s)
}

func (f *fakeTempFile) Close() error {
	return f.closeErr
}

func (f *fakeTempFile) Name() string {
	if f.name != "" {
		return f.name
	}
	return "fake-temp.txt"
}

type fakeReadCloser struct {
	io.Reader
	closeErr error
}

func (r fakeReadCloser) Close() error {
	return r.closeErr
}

type trackingTempFile struct {
	data []byte
}

func (f *trackingTempFile) Write(p []byte) (int, error) {
	f.data = append(f.data, p...)
	return len(p), nil
}

func (f *trackingTempFile) Close() error {
	return nil
}

func (f *trackingTempFile) Name() string {
	return "tracking-temp.txt"
}

type writerToReadCloser struct {
	content      string
	usedWriterTo bool
}

func (r *writerToReadCloser) Read(_ []byte) (int, error) {
	return 0, errors.New("Read() must not be used when WriteTo() is available")
}

func (r *writerToReadCloser) WriteTo(w io.Writer) (int64, error) {
	r.usedWriterTo = true
	n, err := io.WriteString(w, r.content)
	return int64(n), err
}

func (r *writerToReadCloser) Close() error {
	return nil
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

func writeFile(t *testing.T, path string, content []byte) {
	t.Helper()
	if err := os.WriteFile(path, content, 0o600); err != nil {
		t.Fatalf("WriteFile(%s) error = %v", path, err)
	}
}

func assertErrorContains(t *testing.T, err error, wantParts ...string) {
	t.Helper()
	if err == nil {
		t.Fatal("error = nil, want error")
	}
	for _, want := range wantParts {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("error = %q, want substring %q", err, want)
		}
	}
}

func newTextDirectory(t *testing.T) string {
	t.Helper()

	root := t.TempDir()
	writeFile(t, filepath.Join(root, "a.txt"), []byte("alpha"))
	if err := os.Mkdir(filepath.Join(root, "nested"), 0o700); err != nil {
		t.Fatalf("Mkdir(nested) error = %v", err)
	}
	writeFile(t, filepath.Join(root, "nested", "b.md"), []byte("# beta\n"))
	if err := os.Mkdir(filepath.Join(root, ".hidden"), 0o700); err != nil {
		t.Fatalf("Mkdir(.hidden) error = %v", err)
	}
	writeFile(t, filepath.Join(root, ".hidden", "skip.txt"), []byte("skip\n"))
	writeFile(t, filepath.Join(root, "blob.bin"), []byte("%PDF-1.7\nbinary"))
	if err := os.Symlink(filepath.Join(root, "a.txt"), filepath.Join(root, "linked.txt")); err != nil {
		t.Logf("Symlink() skipped: %v", err)
	}

	return root
}

func readSource(t *testing.T, source Source) string {
	t.Helper()
	reader, err := source.Open()
	if err != nil {
		t.Fatalf("Source.Open() error = %v", err)
	}
	defer func() {
		_ = reader.Close()
	}()

	content, err := io.ReadAll(reader)
	if err != nil {
		t.Fatalf("ReadAll() error = %v", err)
	}
	return string(content)
}

func TestSourceDerivedProperties(t *testing.T) {
	tests := []struct {
		name         string
		source       Source
		wantSnapshot string
		wantDisplay  string
		wantMulti    bool
	}{
		{
			name:        "empty source",
			source:      Source{},
			wantDisplay: "",
		},
		{
			name: "stdin source",
			source: newSource(Descriptor{
				Kind: KindStdin,
			}, ""),
			wantDisplay: "stdin",
		},
		{
			name: "input path becomes snapshot",
			source: newSource(Descriptor{
				Kind: KindFile,
				Path: " /tmp/example.txt ",
			}, ""),
			wantSnapshot: "/tmp/example.txt",
			wantDisplay:  "/tmp/example.txt",
		},
		{
			name: "backing file display name fallback",
			source: newSource(Descriptor{
				Files: []File{{DisplayPath: "nested/example.md"}},
			}, ""),
			wantDisplay: "nested/example.md",
		},
		{
			name: "snapshot display name fallback",
			source: newSource(Descriptor{
				Files: []File{{DisplayPath: " "}},
			}, "/tmp/snapshot.txt"),
			wantSnapshot: "/tmp/snapshot.txt",
			wantDisplay:  "/tmp/snapshot.txt",
		},
		{
			name: "multiple files are multi file",
			source: newSource(Descriptor{
				Files: []File{
					{DisplayPath: "a.txt"},
					{DisplayPath: "b.txt"},
				},
			}, ""),
			wantDisplay: "a.txt",
			wantMulti:   true,
		},
		{
			name: "directory kind is multi file",
			source: newSource(Descriptor{
				Kind: KindDirectory,
			}, "/tmp/corpus.txt"),
			wantSnapshot: "/tmp/corpus.txt",
			wantDisplay:  "/tmp/corpus.txt",
			wantMulti:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.source.SnapshotPath(); got != tt.wantSnapshot {
				t.Fatalf("SnapshotPath() = %q, want %q", got, tt.wantSnapshot)
			}
			if got := tt.source.DisplayName(); got != tt.wantDisplay {
				t.Fatalf("DisplayName() = %q, want %q", got, tt.wantDisplay)
			}
			if got := tt.source.IsMultiFile(); got != tt.wantMulti {
				t.Fatalf("IsMultiFile() = %t, want %t", got, tt.wantMulti)
			}
		})
	}

	initLocale(t)
	_, err := (Source{}).Open()
	assertErrorContains(t, err, "snapshot path is empty")

	t.Run("open wraps openFile failure", func(t *testing.T) {
		stubInputDeps(t)
		wantErr := errors.New("open failed")
		openFile = func(string) (io.ReadCloser, error) {
			return nil, wantErr
		}

		_, err := newSource(Descriptor{Path: "/tmp/example.txt"}, "").Open()
		assertErrorContains(t, err, "failed to open file", wantErr.Error())
	})
}

func TestHandleSourceAndClose(t *testing.T) {
	t.Run("nil handle is safe", func(t *testing.T) {
		var handle *Handle
		got := handle.Source()
		if got.snapshotPath != "" || got.descriptor.Kind != "" || got.descriptor.Path != "" || len(got.descriptor.Files) != 0 {
			t.Fatalf("Source() = %#v, want zero Source", got)
		}
		if err := handle.Close(); err != nil {
			t.Fatalf("Close() error = %v, want nil", err)
		}
	})

	t.Run("source returns clone", func(t *testing.T) {
		handle := &Handle{
			source: newSource(Descriptor{
				Kind:  KindFile,
				Files: []File{{DisplayPath: "original.txt"}},
			}, "/tmp/original.txt"),
		}

		source := handle.Source()
		source.descriptor.Files[0].DisplayPath = "mutated.txt"

		if got := handle.Source().descriptor.Files[0].DisplayPath; got != "original.txt" {
			t.Fatalf("Source() did not clone files, got %q", got)
		}
	})

	t.Run("close without closer is noop", func(t *testing.T) {
		handle := &Handle{}
		if err := handle.Close(); err != nil {
			t.Fatalf("Close() error = %v, want nil", err)
		}
	})

	t.Run("close returns error", func(t *testing.T) {
		wantErr := errors.New("close failed")
		handle := &Handle{
			closeFn: func() error {
				return wantErr
			},
		}

		if err := handle.Close(); !errors.Is(err, wantErr) {
			t.Fatalf("Close() error = %v, want %v", err, wantErr)
		}
	})
}

func TestOpenFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sample.txt")
	writeFile(t, path, []byte("hello"))

	handle, err := Open(path, nil)
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	t.Cleanup(func() {
		_ = handle.Close()
	})

	source := handle.Source()
	if got := readSource(t, source); got != "hello" {
		t.Fatalf("content = %q, want hello", got)
	}
	if got := source.Kind(); got != KindFile {
		t.Fatalf("Kind() = %q, want %q", got, KindFile)
	}
	if got := source.InputPath(); got != path {
		t.Fatalf("InputPath() = %q, want %q", got, path)
	}
	if got := source.SnapshotPath(); got != path {
		t.Fatalf("SnapshotPath() = %q, want %q", got, path)
	}
	if got := source.DisplayName(); got != path {
		t.Fatalf("DisplayName() = %q, want %q", got, path)
	}
	if source.IsMultiFile() {
		t.Fatal("IsMultiFile() = true, want false")
	}
	files := source.BackingFiles()
	if len(files) != 1 {
		t.Fatalf("BackingFiles() len = %d, want 1", len(files))
	}
	if files[0].Path != path || files[0].DisplayPath != "sample.txt" {
		t.Fatalf("BackingFiles()[0] = %#v, want path=%q display=sample.txt", files[0], path)
	}
}

func TestOpenStdin(t *testing.T) {
	t.Run("materializes stdin and cleans snapshot", func(t *testing.T) {
		handle, err := Open("", bytes.NewBufferString("alpha\nbeta\n"))
		if err != nil {
			t.Fatalf("Open() error = %v", err)
		}

		source := handle.Source()
		if got := readSource(t, source); got != "alpha\nbeta\n" {
			t.Fatalf("content = %q, want exact stdin content", got)
		}
		if got := source.Kind(); got != KindStdin {
			t.Fatalf("Kind() = %q, want %q", got, KindStdin)
		}
		if got := source.DisplayName(); got != "stdin" {
			t.Fatalf("DisplayName() = %q, want stdin", got)
		}
		if got := source.InputPath(); got != "" {
			t.Fatalf("InputPath() = %q, want empty", got)
		}
		if source.SnapshotPath() == "" {
			t.Fatal("SnapshotPath() = empty, want temp file path")
		}
		if source.IsMultiFile() {
			t.Fatal("IsMultiFile() = true, want false")
		}

		tempPath := source.SnapshotPath()
		if _, err := os.Stat(tempPath); err != nil {
			t.Fatalf("Stat(%q) error = %v", tempPath, err)
		}
		if err := handle.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
		if _, err := os.Stat(tempPath); !os.IsNotExist(err) {
			t.Fatalf("temp file still exists after Close(), err = %v", err)
		}
	})

	t.Run("uses default stdin when reader is nil", func(t *testing.T) {
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

		if got := readSource(t, handle.Source()); got != "from default stdin" {
			t.Fatalf("content = %q, want default stdin content", got)
		}
	})
}

func TestOpenErrors(t *testing.T) {
	initLocale(t)

	tests := []struct {
		name      string
		call      func(t *testing.T) (*Handle, error)
		wantError string
		wantCause error
	}{
		{
			name: "missing file",
			call: func(t *testing.T) (*Handle, error) {
				t.Helper()
				return Open(filepath.Join(t.TempDir(), "missing.txt"), nil)
			},
			wantError: "failed to open file",
		},
		{
			name: "create temp",
			call: func(t *testing.T) (*Handle, error) {
				t.Helper()
				stubInputDeps(t)
				wantErr := errors.New("create temp failed")
				createTemp = func(_, _ string) (tempFile, error) {
					return nil, wantErr
				}
				return Open("", bytes.NewBufferString("data"))
			},
			wantError: "failed to create temp file",
			wantCause: errors.New("create temp failed"),
		},
		{
			name: "read stdin",
			call: func(t *testing.T) (*Handle, error) {
				t.Helper()
				return Open("", failingReader{err: errors.New("stdin read failed")})
			},
			wantError: "failed to read stdin",
			wantCause: errors.New("stdin read failed"),
		},
		{
			name: "close temp",
			call: func(t *testing.T) (*Handle, error) {
				t.Helper()
				stubInputDeps(t)
				wantErr := errors.New("close temp failed")
				createTemp = func(_, _ string) (tempFile, error) {
					return &fakeTempFile{
						name:     filepath.Join(t.TempDir(), "stdin.txt"),
						closeErr: wantErr,
					}, nil
				}
				return Open("", bytes.NewBufferString("data"))
			},
			wantError: "failed to close temp file",
			wantCause: errors.New("close temp failed"),
		},
		{
			name: "reopen temp",
			call: func(t *testing.T) (*Handle, error) {
				t.Helper()
				stubInputDeps(t)
				wantErr := errors.New("reopen failed")
				originalOpenFile := openFile
				openFile = func(path string) (io.ReadCloser, error) {
					if strings.Contains(path, "ragcli-stdin-") {
						return nil, wantErr
					}
					return originalOpenFile(path)
				}
				return Open("", bytes.NewBufferString("data"))
			},
			wantError: "failed to reopen temp file",
			wantCause: errors.New("reopen failed"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			handle, err := tt.call(t)
			if handle != nil {
				t.Cleanup(func() {
					_ = handle.Close()
				})
			}
			if tt.wantCause != nil {
				assertErrorContains(t, err, tt.wantError, tt.wantCause.Error())
				return
			}
			assertErrorContains(t, err, tt.wantError)
		})
	}
}

func TestOpenDirectory(t *testing.T) {
	root := newTextDirectory(t)

	handle, err := Open(root, nil)
	if err != nil {
		t.Fatalf("Open(directory) error = %v", err)
	}

	source := handle.Source()
	if got := source.Kind(); got != KindDirectory {
		t.Fatalf("Kind() = %q, want %q", got, KindDirectory)
	}
	if got := source.InputPath(); got != root {
		t.Fatalf("InputPath() = %q, want %q", got, root)
	}
	if got := source.DisplayName(); got != root {
		t.Fatalf("DisplayName() = %q, want %q", got, root)
	}
	if !source.IsMultiFile() {
		t.Fatal("IsMultiFile() = false, want true")
	}
	if source.SnapshotPath() == "" || source.SnapshotPath() == root {
		t.Fatalf("SnapshotPath() = %q, want temp corpus path", source.SnapshotPath())
	}

	files := source.BackingFiles()
	if len(files) != 2 {
		t.Fatalf("BackingFiles() len = %d, want 2", len(files))
	}
	if files[0].DisplayPath != "a.txt" || files[1].DisplayPath != "nested/b.md" {
		t.Fatalf("BackingFiles() = %#v, want sorted text files", files)
	}

	wantCorpus := "=== file: a.txt ===\nalpha\n\n=== file: nested/b.md ===\n# beta\n"
	if got := readSource(t, source); got != wantCorpus {
		t.Fatalf("materialized corpus = %q, want %q", got, wantCorpus)
	}

	tempPath := source.SnapshotPath()
	if err := handle.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if _, err := os.Stat(tempPath); !os.IsNotExist(err) {
		t.Fatalf("temp file still exists after Close(), err = %v", err)
	}
}

func TestOpenDirectoryErrors(t *testing.T) {
	initLocale(t)

	t.Run("no readable text files", func(t *testing.T) {
		root := t.TempDir()
		writeFile(t, filepath.Join(root, "blob.bin"), []byte("%PDF-1.7\nbinary"))
		if err := os.Mkdir(filepath.Join(root, ".hidden"), 0o700); err != nil {
			t.Fatalf("Mkdir(.hidden) error = %v", err)
		}
		writeFile(t, filepath.Join(root, ".hidden", "skip.txt"), []byte("skip\n"))

		_, err := Open(root, nil)
		if err == nil {
			t.Fatal("Open(directory) error = nil, want error")
		}
		if !strings.Contains(err.Error(), "no readable text files found in the directory") {
			t.Fatalf("error = %q, want no-files error", err)
		}
	})

	t.Run("create temp", func(t *testing.T) {
		stubInputDeps(t)
		root := newTextDirectory(t)
		wantErr := errors.New("create temp failed")
		createTemp = func(_, _ string) (tempFile, error) {
			return nil, wantErr
		}

		_, err := Open(root, nil)
		assertErrorContains(t, err, "failed to create temp file", wantErr.Error())
	})

	t.Run("close temp", func(t *testing.T) {
		stubInputDeps(t)
		root := newTextDirectory(t)
		wantErr := errors.New("close temp failed")
		createTemp = func(_, _ string) (tempFile, error) {
			return &fakeTempFile{
				name:     filepath.Join(t.TempDir(), "dir.txt"),
				closeErr: wantErr,
			}, nil
		}

		_, err := Open(root, nil)
		assertErrorContains(t, err, "failed to close temp file", wantErr.Error())
	})

	t.Run("reopen temp", func(t *testing.T) {
		stubInputDeps(t)
		root := newTextDirectory(t)
		wantErr := errors.New("reopen failed")
		originalOpenFile := openFile
		openFile = func(path string) (io.ReadCloser, error) {
			if strings.Contains(path, "ragcli-path-") {
				return nil, wantErr
			}
			return originalOpenFile(path)
		}

		_, err := Open(root, nil)
		assertErrorContains(t, err, "failed to reopen temp file", wantErr.Error())
	})

	t.Run("read dir", func(t *testing.T) {
		_, err := openDirectory(filepath.Join(t.TempDir(), "missing"))
		assertErrorContains(t, err, "failed to read input directory")
	})
}

func TestIsBinaryFile(t *testing.T) {
	t.Run("text mime", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "sample.txt")
		writeFile(t, path, []byte("plain text\nsecond line\n"))

		binary, err := isBinaryFile(path)
		if err != nil {
			t.Fatalf("isBinaryFile() error = %v", err)
		}
		if binary {
			t.Fatal("isBinaryFile() = true, want false")
		}
	})

	t.Run("binary mime", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "sample.bin")
		writeFile(t, path, []byte("%PDF-1.7\nbinary"))

		binary, err := isBinaryFile(path)
		if err != nil {
			t.Fatalf("isBinaryFile() error = %v", err)
		}
		if !binary {
			t.Fatal("isBinaryFile() = false, want true")
		}
	})

	t.Run("empty file is text", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "empty.txt")
		writeFile(t, path, nil)

		binary, err := isBinaryFile(path)
		if err != nil {
			t.Fatalf("isBinaryFile() error = %v", err)
		}
		if binary {
			t.Fatal("isBinaryFile() = true, want false")
		}
	})

	t.Run("read error", func(t *testing.T) {
		stubInputDeps(t)
		wantErr := errors.New("read failed")
		openFile = func(string) (io.ReadCloser, error) {
			return fakeReadCloser{Reader: failingReader{err: wantErr}}, nil
		}

		_, err := isBinaryFile("broken.txt")
		if !errors.Is(err, wantErr) {
			t.Fatalf("isBinaryFile() error = %v, want %v", err, wantErr)
		}
	})

	t.Run("open error", func(t *testing.T) {
		stubInputDeps(t)
		wantErr := errors.New("open failed")
		openFile = func(string) (io.ReadCloser, error) {
			return nil, wantErr
		}

		_, err := isBinaryFile("broken.txt")
		if !errors.Is(err, wantErr) {
			t.Fatalf("isBinaryFile() error = %v, want %v", err, wantErr)
		}
	})
}

func TestWriteDirectoryCorpusErrors(t *testing.T) {
	tests := []struct {
		name    string
		dst     tempFile
		open    func(string) (io.ReadCloser, error)
		wantErr error
	}{
		{
			name:    "destination write",
			dst:     &fakeTempFile{writeErr: errors.New("write failed")},
			open:    func(string) (io.ReadCloser, error) { return nil, errors.New("unexpected open") },
			wantErr: errors.New("write failed"),
		},
		{
			name:    "source open",
			dst:     &fakeTempFile{},
			open:    func(string) (io.ReadCloser, error) { return nil, errors.New("open failed") },
			wantErr: errors.New("open failed"),
		},
		{
			name: "source read",
			dst:  &fakeTempFile{},
			open: func(string) (io.ReadCloser, error) {
				return fakeReadCloser{Reader: failingReader{err: errors.New("read failed")}}, nil
			},
			wantErr: errors.New("read failed"),
		},
		{
			name: "source close",
			dst:  &fakeTempFile{},
			open: func(string) (io.ReadCloser, error) {
				return fakeReadCloser{
					Reader:   bytes.NewBufferString("ok"),
					closeErr: errors.New("close failed"),
				}, nil
			},
			wantErr: errors.New("close failed"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stubInputDeps(t)
			openFile = tt.open

			err := writeDirectoryCorpus(tt.dst, []File{{
				Path:        "sample.txt",
				DisplayPath: "sample.txt",
			}})
			if err == nil {
				t.Fatal("writeDirectoryCorpus() error = nil, want error")
			}
			if !strings.Contains(err.Error(), tt.wantErr.Error()) {
				t.Fatalf("writeDirectoryCorpus() error = %v, want %v", err, tt.wantErr)
			}
		})
	}
}

func TestWriteDirectoryCorpusStreamsFileContents(t *testing.T) {
	stubInputDeps(t)

	src := &writerToReadCloser{content: "alpha\nbeta\n"}
	openFile = func(string) (io.ReadCloser, error) {
		return src, nil
	}
	dst := &trackingTempFile{}

	if err := writeDirectoryCorpus(dst, []File{{
		Path:        "sample.txt",
		DisplayPath: "sample.txt",
	}}); err != nil {
		t.Fatalf("writeDirectoryCorpus() error = %v", err)
	}
	if !src.usedWriterTo {
		t.Fatal("writeDirectoryCorpus() did not stream file contents")
	}

	got := string(dst.data)
	want := "=== file: sample.txt ===\nalpha\nbeta\n"
	if got != want {
		t.Fatalf("corpus = %q, want %q", got, want)
	}
}
