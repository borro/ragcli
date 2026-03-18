package input

import (
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/borro/ragcli/internal/localize"
)

type tempFile interface {
	io.WriteCloser
	Name() string
}

type File struct {
	Path        string
	DisplayPath string
}

type Kind string

const (
	KindStdin     Kind = "stdin"
	KindFile      Kind = "file"
	KindDirectory Kind = "directory"
)

type Descriptor struct {
	Kind  Kind
	Path  string
	Files []File
}

type SourceMeta interface {
	Kind() Kind
	InputPath() string
	DisplayName() string
}

type SnapshotSource interface {
	SourceMeta
	SnapshotPath() string
	Open() (io.ReadCloser, error)
}

type FileBackedSource interface {
	SnapshotSource
	BackingFiles() []File
	FileCount() int
	IsMultiFile() bool
}

type Source struct {
	descriptor   Descriptor
	snapshotPath string
}

func (s Source) clone() Source {
	cloned := s
	cloned.descriptor.Files = append([]File(nil), s.descriptor.Files...)
	return cloned
}

func (s Source) InputPath() string {
	return strings.TrimSpace(s.descriptor.Path)
}

func (s Source) SnapshotPath() string {
	if path := strings.TrimSpace(s.snapshotPath); path != "" {
		return path
	}
	return s.InputPath()
}

func (s Source) Open() (io.ReadCloser, error) {
	path := s.SnapshotPath()
	if path == "" {
		return nil, fmt.Errorf("%s", localize.T("error.input.open_file", localize.Data{"Error": "snapshot path is empty"}))
	}
	file, err := openFile(path)
	if err != nil {
		return nil, fmt.Errorf("%s", localize.T("error.input.open_file", localize.Data{"Error": err}))
	}
	return file, nil
}

func (s Source) DisplayName() string {
	if s.descriptor.Kind == KindStdin {
		return "stdin"
	}
	if path := s.InputPath(); path != "" {
		return path
	}
	if s.FileCount() > 0 {
		if displayPath := strings.TrimSpace(s.descriptor.Files[0].DisplayPath); displayPath != "" {
			return displayPath
		}
	}
	if path := s.SnapshotPath(); path != "" {
		return path
	}
	return ""
}

func (s Source) BackingFiles() []File {
	return append([]File(nil), s.descriptor.Files...)
}

func (s Source) FileCount() int {
	return len(s.descriptor.Files)
}

func (s Source) Kind() Kind {
	return s.descriptor.Kind
}

func (s Source) IsMultiFile() bool {
	return s.Kind() == KindDirectory || s.FileCount() > 1
}

var (
	openFile     = func(path string) (io.ReadCloser, error) { return os.Open(path) }
	createTemp   = func(dir, pattern string) (tempFile, error) { return os.CreateTemp(dir, pattern) }
	defaultStdin = func() io.Reader { return os.Stdin }
)

type Handle struct {
	source  Source
	closeFn func() error
}

func (h *Handle) Source() Source {
	if h == nil {
		return Source{}
	}
	return h.source.clone()
}

func Open(path string, stdin io.Reader) (*Handle, error) {
	if path != "" {
		if info, err := os.Stat(path); err == nil && info.IsDir() {
			return openDirectory(path)
		}
		file, err := openFile(path)
		if err != nil {
			return nil, fmt.Errorf("%s", localize.T("error.input.open_file", localize.Data{"Error": err}))
		}
		if err := file.Close(); err != nil {
			return nil, fmt.Errorf("%s", localize.T("error.input.open_file", localize.Data{"Error": err}))
		}
		return &Handle{
			source: newSource(Descriptor{
				Kind: KindFile,
				Path: path,
				Files: []File{{
					Path:        path,
					DisplayPath: filepath.Base(path),
				}},
			}, path),
		}, nil
	}

	if stdin == nil {
		stdin = defaultStdin()
	}

	tmpFile, err := createTemp("", "ragcli-stdin-*.txt")
	if err != nil {
		return nil, fmt.Errorf("%s", localize.T("error.input.create_temp", localize.Data{"Error": err}))
	}

	if _, err := io.Copy(tmpFile, stdin); err != nil {
		_ = tmpFile.Close()
		_ = os.Remove(tmpFile.Name())
		return nil, fmt.Errorf("%s", localize.T("error.input.read_stdin", localize.Data{"Error": err}))
	}
	snapshotPath, err := finalizeTempSnapshot(tmpFile)
	if err != nil {
		return nil, err
	}

	return &Handle{
		source: newSource(Descriptor{
			Kind: KindStdin,
			Files: []File{{
				Path:        snapshotPath,
				DisplayPath: "stdin",
			}},
		}, snapshotPath),
		closeFn: func() error {
			return os.Remove(snapshotPath)
		},
	}, nil
}

func (h *Handle) Close() error {
	if h == nil || h.closeFn == nil {
		return nil
	}
	return h.closeFn()
}

func openDirectory(path string) (*Handle, error) {
	files, err := collectDirectoryFiles(path)
	if err != nil {
		return nil, fmt.Errorf("%s", localize.T("error.input.read_dir", localize.Data{"Error": err}))
	}
	if len(files) == 0 {
		return nil, fmt.Errorf("%s", localize.T("error.input.no_files"))
	}

	tmpFile, err := createTemp("", "ragcli-path-*.txt")
	if err != nil {
		return nil, fmt.Errorf("%s", localize.T("error.input.create_temp", localize.Data{"Error": err}))
	}
	if err := writeDirectoryCorpus(tmpFile, files); err != nil {
		_ = tmpFile.Close()
		_ = os.Remove(tmpFile.Name())
		return nil, fmt.Errorf("%s", localize.T("error.input.read_dir", localize.Data{"Error": err}))
	}
	snapshotPath, err := finalizeTempSnapshot(tmpFile)
	if err != nil {
		return nil, err
	}

	return &Handle{
		source: newSource(Descriptor{
			Kind:  KindDirectory,
			Path:  path,
			Files: files,
		}, snapshotPath),
		closeFn: func() error {
			return os.Remove(snapshotPath)
		},
	}, nil
}

func newSource(descriptor Descriptor, snapshotPath string) Source {
	cloned := descriptor
	cloned.Files = append([]File(nil), descriptor.Files...)
	return Source{
		descriptor:   cloned,
		snapshotPath: strings.TrimSpace(snapshotPath),
	}
}

func verifySnapshot(path string) error {
	file, err := openFile(path)
	if err != nil {
		return err
	}
	return file.Close()
}

func finalizeTempSnapshot(tmpFile tempFile) (string, error) {
	snapshotPath := tmpFile.Name()
	if err := tmpFile.Close(); err != nil {
		_ = os.Remove(snapshotPath)
		return "", fmt.Errorf("%s", localize.T("error.input.close_temp", localize.Data{"Error": err}))
	}
	if err := verifySnapshot(snapshotPath); err != nil {
		_ = os.Remove(snapshotPath)
		return "", fmt.Errorf("%s", localize.T("error.input.reopen_temp", localize.Data{"Error": err}))
	}
	return snapshotPath, nil
}

func collectDirectoryFiles(root string) ([]File, error) {
	collected := make([]File, 0, 16)
	if err := walkDirectory(root, root, &collected); err != nil {
		return nil, err
	}
	sort.SliceStable(collected, func(i, j int) bool {
		return collected[i].DisplayPath < collected[j].DisplayPath
	})
	return collected, nil
}

func walkDirectory(root string, dir string, collected *[]File) error {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return err
	}

	for _, entry := range entries {
		name := entry.Name()
		fullPath := filepath.Join(dir, name)
		if entry.Type()&os.ModeSymlink != 0 {
			continue
		}
		if entry.IsDir() {
			if strings.HasPrefix(name, ".") {
				continue
			}
			if err := walkDirectory(root, fullPath, collected); err != nil {
				return err
			}
			continue
		}

		info, err := entry.Info()
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() {
			continue
		}

		binary, err := isBinaryFile(fullPath)
		if err != nil {
			return err
		}
		if binary {
			continue
		}

		relPath, err := filepath.Rel(root, fullPath)
		if err != nil {
			return err
		}
		*collected = append(*collected, File{
			Path:        fullPath,
			DisplayPath: filepath.ToSlash(relPath),
		})
	}

	return nil
}

func isBinaryFile(path string) (bool, error) {
	file, err := openFile(path)
	if err != nil {
		return false, err
	}
	defer func() {
		_ = file.Close()
	}()

	buffer := make([]byte, 512)
	read, err := file.Read(buffer)
	if err != nil && !errors.Is(err, io.EOF) {
		return false, err
	}
	sample := buffer[:read]
	if len(sample) == 0 {
		return false, nil
	}

	contentType := http.DetectContentType(sample)
	mediaType, _, parseErr := mime.ParseMediaType(contentType)
	if parseErr != nil {
		mediaType = contentType
	}
	return !strings.HasPrefix(mediaType, "text/"), nil
}

func writeDirectoryCorpus(dst tempFile, files []File) error {
	for index, file := range files {
		if _, err := io.WriteString(dst, fmt.Sprintf("=== file: %s ===\n", file.DisplayPath)); err != nil {
			return err
		}

		src, err := openFile(file.Path)
		if err != nil {
			return err
		}

		hadTrailingNewline, copyErr := copyFileContents(dst, src)
		closeErr := src.Close()
		if copyErr != nil {
			return copyErr
		}
		if closeErr != nil {
			return closeErr
		}
		if !hadTrailingNewline {
			if _, err := io.WriteString(dst, "\n"); err != nil {
				return err
			}
		}
		if index < len(files)-1 {
			if _, err := io.WriteString(dst, "\n"); err != nil {
				return err
			}
		}
	}
	return nil
}

type trailingNewlineTracker struct {
	writer             io.Writer
	lastByte           byte
	hasWrittenAnyBytes bool
}

func (t *trailingNewlineTracker) Write(p []byte) (int, error) {
	if len(p) > 0 {
		t.lastByte = p[len(p)-1]
		t.hasWrittenAnyBytes = true
	}
	return t.writer.Write(p)
}

func (t *trailingNewlineTracker) endedWithNewline() bool {
	return t.hasWrittenAnyBytes && t.lastByte == '\n'
}

func copyFileContents(dst io.Writer, src io.Reader) (bool, error) {
	tracker := &trailingNewlineTracker{writer: dst}
	if _, err := io.Copy(tracker, src); err != nil {
		return false, err
	}
	return tracker.endedWithNewline(), nil
}
