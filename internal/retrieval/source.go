package retrieval

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
)

type SpoolResult struct {
	TempPath string
	Bytes    int
	Hash     string
}

func WalkLines(reader io.Reader, onLine func(rawLine, text string, lineNo, byteStart, byteEnd int) error) (int, error) {
	buffered := bufio.NewReader(reader)
	byteOffset := 0
	lineNo := 1

	for {
		rawLine, err := buffered.ReadString('\n')
		if err != nil && err != io.EOF {
			return byteOffset, err
		}
		if rawLine == "" && err == io.EOF {
			break
		}

		nextOffset := byteOffset + len(rawLine)
		if callErr := onLine(rawLine, trimTrailingNewline(rawLine), lineNo, byteOffset, nextOffset); callErr != nil {
			return byteOffset, callErr
		}
		byteOffset = nextOffset
		lineNo++

		if err == io.EOF {
			break
		}
	}
	return byteOffset, nil
}

func SpoolSource(reader io.Reader, tempPattern string, hashParts []string, observe func(rawLine, text string, lineNo, byteStart, byteEnd int) error) (*SpoolResult, error) {
	tempFile, err := os.CreateTemp("", tempPattern)
	if err != nil {
		return nil, err
	}
	defer func() {
		_ = tempFile.Close()
	}()

	hasher := sha256.New()
	written, err := WalkLines(reader, func(rawLine, text string, lineNo, byteStart, byteEnd int) error {
		if _, writeErr := io.WriteString(tempFile, rawLine); writeErr != nil {
			return writeErr
		}
		if _, writeErr := io.WriteString(hasher, rawLine); writeErr != nil {
			return writeErr
		}
		if observe != nil {
			return observe(rawLine, text, lineNo, byteStart, byteEnd)
		}
		return nil
	})
	if err != nil {
		_ = os.Remove(tempFile.Name())
		return nil, err
	}
	writeHashMetadata(hasher, hashParts...)
	return &SpoolResult{
		TempPath: tempFile.Name(),
		Bytes:    written,
		Hash:     hex.EncodeToString(hasher.Sum(nil)),
	}, nil
}

func writeHashMetadata(w io.Writer, parts ...string) {
	for _, part := range parts {
		_, _ = fmt.Fprintf(w, "|%d:%s", len(part), part)
	}
}

func trimTrailingNewline(raw string) string {
	if len(raw) > 0 && raw[len(raw)-1] == '\n' {
		return raw[:len(raw)-1]
	}
	return raw
}
