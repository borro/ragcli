package app

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/borro/ragcli/internal/input"
)

func TestDefaultInteractionIOPrefersTTYWhenStdinIsNotTerminal(t *testing.T) {
	handle, err := input.Open(writeRuntimeTempFile(t, "alpha\n"), nil)
	if err != nil {
		t.Fatalf("input.Open() error = %v", err)
	}
	defer func() { _ = handle.Close() }()

	ttyReader := strings.NewReader("from tty\n")
	ttyWriter := &bytes.Buffer{}
	closed := false

	prev := openInteractionTTYFunc
	openInteractionTTYFunc = func() (*interactionIO, error) {
		return &interactionIO{
			reader: ttyReader,
			writer: ttyWriter,
			closeFn: func() error {
				closed = true
				return nil
			},
		}, nil
	}
	defer func() { openInteractionTTYFunc = prev }()

	ioState, err := defaultInteractionIO(handle.Source(), strings.NewReader("from stdin\n"), io.Discard)
	if err != nil {
		t.Fatalf("defaultInteractionIO() error = %v", err)
	}
	if ioState.reader != ttyReader {
		t.Fatalf("reader = %#v, want tty reader when stdin is not a terminal", ioState.reader)
	}
	if ioState.writer != ttyWriter {
		t.Fatalf("writer = %#v, want tty writer when stdin is not a terminal", ioState.writer)
	}
	if err := ioState.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if !closed {
		t.Fatal("Close() did not call tty closeFn")
	}
}

func TestReadInteractionLineWritesPromptAndTrimsLine(t *testing.T) {
	reader := bufio.NewReader(strings.NewReader("hello\r\n"))
	var writer bytes.Buffer

	line, err := readInteractionLine(context.Background(), reader, nil, &writer, "> ")
	if err != nil {
		t.Fatalf("readInteractionLine() error = %v", err)
	}
	if line != "hello" {
		t.Fatalf("line = %q, want %q", line, "hello")
	}
	if writer.String() != "> " {
		t.Fatalf("prompt output = %q, want %q", writer.String(), "> ")
	}
}

func TestNewInteractionLineReaderUsesBufferedReaderForNonTerminalIO(t *testing.T) {
	var writer bytes.Buffer
	lineReader := newInteractionLineReader(&interactionIO{
		reader: strings.NewReader("custom\n"),
		writer: &writer,
	})

	line, err := lineReader(context.Background(), &writer, "> ")
	if err != nil {
		t.Fatalf("lineReader() error = %v", err)
	}
	if line != "custom" {
		t.Fatalf("line = %q, want %q", line, "custom")
	}
	if writer.String() != "> " {
		t.Fatalf("writer = %q, want prompt output", writer.String())
	}
}

func TestReadInteractionLineReturnsOnContextCancel(t *testing.T) {
	pr, _ := io.Pipe()
	reader := bufio.NewReader(pr)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	done := make(chan struct{})
	go func() {
		time.Sleep(10 * time.Millisecond)
		cancel()
		close(done)
	}()

	_, err := readInteractionLine(ctx, reader, pr.Close, io.Discard, "> ")
	if err == nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("readInteractionLine() error = %v, want context canceled", err)
	}
	<-done
	_ = pr.Close()
}

func TestReadLineWithContextWaitsForCanceledReadToFinish(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	started := make(chan struct{})
	release := make(chan struct{})
	done := make(chan struct{})
	cancelCalls := 0

	go func() {
		<-started
		cancel()
	}()

	_, err := readLineWithContext(ctx, func() (string, error) {
		close(started)
		<-release
		close(done)
		return "", io.EOF
	}, func() error {
		cancelCalls++
		close(release)
		return nil
	})
	if err == nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("readLineWithContext() error = %v, want context canceled", err)
	}
	if cancelCalls != 1 {
		t.Fatalf("cancelCalls = %d, want 1", cancelCalls)
	}
	select {
	case <-done:
	default:
		t.Fatal("read goroutine did not finish before return")
	}
}

func TestInteractionReaderFDReturnsTerminalFD(t *testing.T) {
	file, err := os.Open(writeRuntimeTempFile(t, "alpha\n"))
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	defer func() { _ = file.Close() }()

	prevIsTerminalReader := isTerminalReader
	isTerminalReader = func(io.Reader) bool { return true }
	defer func() { isTerminalReader = prevIsTerminalReader }()

	got, ok := interactionReaderFD(&interactionIO{reader: file})
	if !ok {
		t.Fatal("interactionReaderFD() ok = false, want true")
	}
	if got != int(file.Fd()) {
		t.Fatalf("fd = %d, want %d", got, int(file.Fd()))
	}
}
