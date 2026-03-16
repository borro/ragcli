package app

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/borro/ragcli/internal/input"
	"github.com/borro/ragcli/internal/localize"
	"golang.org/x/term"
)

type interactionIO struct {
	reader  io.Reader
	writer  io.Writer
	closeFn func() error
}

type interactionReadResult struct {
	line string
	err  error
}

type interactionLineReader func(context.Context, io.Writer, string) (string, error)

var openInteractionTTYFunc = openInteractionTTY

func (ioState *interactionIO) Close() error {
	if ioState == nil || ioState.closeFn == nil {
		return nil
	}
	return ioState.closeFn()
}

func defaultInteractionIO(source input.Source, stdin io.Reader, fallbackWriter io.Writer) (*interactionIO, error) {
	if source.Kind() == input.KindStdin {
		tty, err := openInteractionTTYFunc()
		if err != nil {
			return nil, fmt.Errorf("%s", localize.T("error.interaction.tty_required"))
		}
		return tty, nil
	}

	if tty, err := openInteractionTTYFunc(); err == nil {
		if !isTerminalReader(stdin) {
			return tty, nil
		}
		return &interactionIO{
			reader:  stdin,
			writer:  tty.writer,
			closeFn: tty.closeFn,
		}, nil
	}

	return &interactionIO{
		reader: stdin,
		writer: fallbackWriter,
	}, nil
}

func (s *commandSession) runInteraction(interactive interactiveSession) error {
	ioState, err := s.runtime.openREPL(s.source, s.runtime.stdin, s.runtime.stderr)
	if err != nil {
		return err
	}
	if ioState == nil {
		return nil
	}
	defer func() { _ = ioState.Close() }()

	if _, err := fmt.Fprintln(ioState.writer, localize.T("interaction.banner")); err != nil {
		return err
	}

	lineReader := newInteractionLineReader(ioState)
	prompt := localize.T("interaction.prompt")

	for {
		line, readErr := lineReader(s.ctx, ioState.writer, prompt)
		if errors.Is(readErr, context.Canceled) {
			return nil
		}
		if readErr != nil && !errors.Is(readErr, io.EOF) {
			return fmt.Errorf("read interaction input: %w", readErr)
		}

		query := strings.TrimSpace(line)
		switch query {
		case "":
			if errors.Is(readErr, io.EOF) {
				return nil
			}
			continue
		case "/exit":
			return nil
		case "/reset":
			interactive.Reset()
			if _, err := fmt.Fprintln(ioState.writer, localize.T("interaction.reset")); err != nil {
				return err
			}
		default:
			answer, askErr := interactive.Ask(s.ctx, query, interactive.FollowUpPlan(s.reporter))
			if askErr != nil {
				handleRunError(s.runtime.stderr, askErr, s.inv.Common.Debug)
			} else if err := s.writeResult(answer); err != nil {
				return err
			}
		}

		if errors.Is(readErr, io.EOF) {
			return nil
		}
	}
}

func newInteractionLineReader(ioState *interactionIO) interactionLineReader {
	if readerFD, ok := interactionReaderFD(ioState); ok && isTerminalWriter(ioState.writer) {
		readWriter := interactionReadWriter{reader: ioState.reader, writer: ioState.writer}
		return func(ctx context.Context, writer io.Writer, prompt string) (string, error) {
			return readTerminalLine(ctx, readerFD, readWriter, prompt)
		}
	}

	reader := bufio.NewReader(ioState.reader)
	return func(ctx context.Context, writer io.Writer, prompt string) (string, error) {
		return readInteractionLine(ctx, reader, writer, prompt)
	}
}

type interactionReadWriter struct {
	reader io.Reader
	writer io.Writer
}

func (rw interactionReadWriter) Read(p []byte) (int, error) {
	return rw.reader.Read(p)
}

func (rw interactionReadWriter) Write(p []byte) (int, error) {
	return rw.writer.Write(p)
}

func interactionReaderFD(ioState *interactionIO) (int, bool) {
	if ioState == nil || !isTerminalReader(ioState.reader) {
		return 0, false
	}
	fdReader, ok := ioState.reader.(interface{ Fd() uintptr })
	if !ok {
		return 0, false
	}
	return int(fdReader.Fd()), true
}

func readTerminalLine(ctx context.Context, readerFD int, readWriter io.ReadWriter, prompt string) (string, error) {
	if readerFD < 0 || readWriter == nil {
		return "", io.EOF
	}

	oldState, err := term.MakeRaw(readerFD)
	if err != nil {
		return "", err
	}
	defer func() {
		_ = term.Restore(readerFD, oldState)
	}()

	terminal := term.NewTerminal(readWriter, prompt)
	terminal.SetBracketedPasteMode(false)

	if ctx == nil {
		return readTerminalLineFromEditor(terminal)
	}

	resultCh := make(chan interactionReadResult, 1)
	go func() {
		line, err := readTerminalLineFromEditor(terminal)
		resultCh <- interactionReadResult{line: line, err: err}
	}()

	select {
	case <-ctx.Done():
		return "", ctx.Err()
	case result := <-resultCh:
		return result.line, result.err
	}
}

func readTerminalLineFromEditor(terminal *term.Terminal) (string, error) {
	if terminal == nil {
		return "", io.EOF
	}
	line, err := terminal.ReadLine()
	if errors.Is(err, term.ErrPasteIndicator) {
		err = nil
	}
	return strings.TrimRight(line, "\r\n"), err
}

func readInteractionLine(ctx context.Context, reader *bufio.Reader, writer io.Writer, prompt string) (string, error) {
	if _, err := io.WriteString(writer, prompt); err != nil {
		return "", err
	}

	if ctx == nil {
		return readBufferedInteractionLine(reader)
	}

	resultCh := make(chan interactionReadResult, 1)
	go func() {
		line, err := readBufferedInteractionLine(reader)
		resultCh <- interactionReadResult{line: line, err: err}
	}()

	select {
	case <-ctx.Done():
		return "", ctx.Err()
	case result := <-resultCh:
		return result.line, result.err
	}
}

func readBufferedInteractionLine(reader *bufio.Reader) (string, error) {
	if reader == nil {
		return "", io.EOF
	}
	line, err := reader.ReadString('\n')
	return strings.TrimRight(line, "\r\n"), err
}

func openTTYFile(path string, flag int) (*os.File, error) {
	file, err := os.OpenFile(path, flag, 0)
	if err != nil {
		return nil, err
	}
	return file, nil
}
