package input

import (
	"fmt"
	"io"
	"os"
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

func Open(path string, stdin io.Reader) (*Handle, error) {
	if path != "" {
		file, err := openFile(path)
		if err != nil {
			return nil, fmt.Errorf("failed to open file: %w", err)
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
		return nil, fmt.Errorf("failed to create temp file: %w", err)
	}

	if _, err := io.Copy(tmpFile, stdin); err != nil {
		_ = tmpFile.Close()
		_ = os.Remove(tmpFile.Name())
		return nil, fmt.Errorf("failed to read stdin: %w", err)
	}
	if err := tmpFile.Close(); err != nil {
		_ = os.Remove(tmpFile.Name())
		return nil, fmt.Errorf("failed to close temp file: %w", err)
	}

	reader, err := openFile(tmpFile.Name())
	if err != nil {
		_ = os.Remove(tmpFile.Name())
		return nil, fmt.Errorf("failed to reopen temp file: %w", err)
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
