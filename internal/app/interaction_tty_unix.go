//go:build !windows

package app

import (
	"os"
)

func openInteractionTTY() (*interactionIO, error) {
	file, err := openTTYFile("/dev/tty", os.O_RDWR)
	if err != nil {
		return nil, err
	}
	return &interactionIO{
		reader: file,
		writer: file,
		closeFn: func() error {
			return file.Close()
		},
	}, nil
}
