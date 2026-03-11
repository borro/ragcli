package input

import (
	"fmt"
	"io"
	"os"

	"github.com/borro/ragcli/internal/localize"
)

type readCloser interface {
	io.Reader
	io.Closer
}

type tempFile interface {
	io.Writer
	io.Closer
	Name() string
}

type Source struct {
	Reader      io.Reader
	Path        string
	DisplayName string
}

var (
	openFile     = func(path string) (readCloser, error) { return os.Open(path) }
	createTemp   = func(dir, pattern string) (tempFile, error) { return os.CreateTemp(dir, pattern) }
	defaultStdin = func() io.Reader { return os.Stdin }
)

type Handle struct {
	Reader      io.Reader
	Path        string
	DisplayName string

	closeFn func() error
}

func (h *Handle) Source() Source {
	if h == nil {
		return Source{}
	}

	return Source{
		Reader:      h.Reader,
		Path:        h.Path,
		DisplayName: h.DisplayName,
	}
}

func Open(path string, stdin io.Reader) (*Handle, error) {
	if path != "" {
		file, err := openFile(path)
		if err != nil {
			return nil, fmt.Errorf("%s", localize.T("error.input.open_file", localize.Data{"Error": err}))
		}
		return &Handle{
			Reader:      file,
			Path:        path,
			DisplayName: path,
			closeFn:     file.Close,
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
	if err := tmpFile.Close(); err != nil {
		_ = os.Remove(tmpFile.Name())
		return nil, fmt.Errorf("%s", localize.T("error.input.close_temp", localize.Data{"Error": err}))
	}

	reader, err := openFile(tmpFile.Name())
	if err != nil {
		_ = os.Remove(tmpFile.Name())
		return nil, fmt.Errorf("%s", localize.T("error.input.reopen_temp", localize.Data{"Error": err}))
	}

	return &Handle{
		Reader:      reader,
		Path:        tmpFile.Name(),
		DisplayName: "stdin",
		closeFn: func() error {
			closeErr := reader.Close()
			removeErr := os.Remove(tmpFile.Name())
			if closeErr != nil {
				return closeErr
			}
			return removeErr
		},
	}, nil
}

func (h *Handle) Close() error {
	if h == nil || h.closeFn == nil {
		return nil
	}
	return h.closeFn()
}
