package app

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/borro/ragcli/internal/localize"
	"github.com/borro/ragcli/internal/logging"
	"github.com/joho/godotenv"
)

func Run(args []string, stdout io.Writer, stderr io.Writer, stdin io.Reader, version string) int {
	startedAt := time.Now()
	if stderr == nil {
		stderr = io.Discard
	}

	envErr := godotenv.Load()
	adjustedArgs := applyPromptArgCompatibility(args)
	localeCode, _ := localize.Detect(adjustedArgs)
	if err := localize.SetCurrent(localeCode); err != nil {
		_, _ = fmt.Fprintln(stderr, err)
		return 1
	}
	debug := debugEnabled(adjustedArgs)
	logging.Configure(stderr, debug)

	runtime := newRuntime(stdout, stderr, stdin, version, len(args), envErr, startedAt)
	root := newCLI(cliConfig{
		stdout:  stdout,
		stderr:  stderr,
		version: version,
		execute: func(ctx context.Context, inv commandInvocation) error {
			return runtime.execute(ctx, inv)
		},
	})
	if err := root.Run(context.Background(), append([]string{root.Name}, adjustedArgs...)); err != nil {
		handleRunError(stderr, err, debug)
		return exitCode(err)
	}
	return 0
}

func roundDurationSeconds(duration time.Duration) float64 {
	return float64(duration.Round(time.Millisecond)) / float64(time.Second)
}

func handleRunError(stderr io.Writer, err error, debug bool) {
	if err == nil {
		return
	}

	if debug {
		slog.Error("cli execution failed", "stage", "run_cli", "error", err)
		return
	}

	_, _ = fmt.Fprintln(stderr, err)
}

func debugEnabled(args []string) bool {
	if value, found := debugFlagValue(args); found {
		return value
	}

	return debugEnvValue()
}

func debugFlagValue(args []string) (bool, bool) {
	value := false
	found := false

	for _, arg := range args {
		if arg == "--" {
			break
		}

		switch {
		case arg == "--debug" || arg == "-d":
			value = true
			found = true
		case strings.HasPrefix(arg, "--debug="):
			value = parseBool(strings.TrimPrefix(arg, "--debug="))
			found = true
		}
	}

	return value, found
}

func debugEnvValue() bool {
	raw, ok := os.LookupEnv("DEBUG")
	if !ok {
		return false
	}

	return parseBool(raw)
}

func parseBool(raw string) bool {
	value, err := strconv.ParseBool(strings.TrimSpace(raw))
	if err != nil {
		return false
	}

	return value
}

func exitCode(err error) int {
	type exitCoder interface {
		ExitCode() int
	}

	var codedErr exitCoder
	if errors.As(err, &codedErr) {
		return codedErr.ExitCode()
	}

	return 1
}
