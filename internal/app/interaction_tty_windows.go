//go:build windows

package app

import (
	"os"
)

func openInteractionTTY() (*interactionIO, error) {
	reader, err := openTTYFile("CONIN$", os.O_RDONLY)
	if err != nil {
		return nil, err
	}
	writer, err := openTTYFile("CONOUT$", os.O_WRONLY)
	if err != nil {
		_ = reader.Close()
		return nil, err
	}
	return &interactionIO{
		reader: reader,
		writer: writer,
		closeFn: func() error {
			readErr := reader.Close()
			writeErr := writer.Close()
			if readErr != nil {
				return readErr
			}
			return writeErr
		},
	}, nil
}
