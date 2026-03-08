package input

import (
	"fmt"
	"io"
	"os"
)

type Handle struct {
	Reader      io.Reader
	Path        string
	DisplayName string

	closeFn func() error
}

func Open(path string, stdin io.Reader) (*Handle, error) {
	if path != "" {
		file, err := os.Open(path)
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
		stdin = os.Stdin
	}

	tmpFile, err := os.CreateTemp("", "ragcli-stdin-*.txt")
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

	reader, err := os.Open(tmpFile.Name())
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
