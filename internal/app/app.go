package app

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/borro/ragcli/internal/app/hybrid"
	mapmode "github.com/borro/ragcli/internal/app/map"
	"github.com/borro/ragcli/internal/app/rag"
	"github.com/borro/ragcli/internal/app/tools"
	"github.com/borro/ragcli/internal/input"
	"github.com/borro/ragcli/internal/llm"
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
	debug := debugEnabled(adjustedArgs)
	logging.Configure(stderr, debug)

	root := newCLI(cliConfig{
		stdout:  stdout,
		stderr:  stderr,
		version: version,
		execute: func(ctx context.Context, cmd Command) error {
			return executeCommand(ctx, cmd, stdout, stderr, stdin, version, len(args), envErr, startedAt)
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

func applyPromptArgCompatibility(args []string) []string {
	if len(args) == 0 || !supportsImplicitPrompt(args[0]) {
		return args
	}

	subArgs := args[1:]
	for i := 0; i < len(subArgs); i++ {
		token := subArgs[i]
		if token == "--" {
			return args
		}
		if !strings.HasPrefix(token, "-") || token == "-" {
			result := make([]string, 0, len(args)+1)
			result = append(result, args[:i+1]...)
			result = append(result, "--")
			result = append(result, args[i+1:]...)
			return result
		}
		if commandFlagConsumesValue(args[0], token) && !strings.Contains(token, "=") && i+1 < len(subArgs) {
			i++
		}
	}

	return args
}

func supportsImplicitPrompt(subcommand string) bool {
	switch subcommand {
	case "map", "rag", "hybrid", "tools":
		return true
	default:
		return false
	}
}

func commandFlagConsumesValue(subcommand string, token string) bool {
	switch token {
	case "-f", "--file", "--api-url", "--model", "--api-key", "-r", "--retry":
		return true
	case "-c", "--concurrency", "-l", "--length":
		return subcommand == "map"
	case "--embedding-model", "--rag-top-k", "--rag-final-k", "--rag-chunk-size", "--rag-chunk-overlap", "--rag-index-ttl", "--rag-index-dir", "--rag-rerank":
		return subcommand == "rag" || subcommand == "hybrid"
	case "--hybrid-top-k", "--hybrid-final-k", "--hybrid-map-k", "--hybrid-read-window", "--hybrid-fallback":
		return subcommand == "hybrid"
	default:
		return false
	}
}

func executeCommand(ctx context.Context, cmd Command, stdout io.Writer, stderr io.Writer, stdin io.Reader, version string, argsCount int, envErr error, startedAt time.Time) error {
	logging.Configure(stderr, cmd.Common.Debug)
	slog.Info("application run started",
		"args_count", argsCount,
		"version", strings.TrimSpace(version),
		"version_supplied", strings.TrimSpace(version) != "",
	)
	slog.Debug("parsing cli arguments", "args_count", argsCount)
	if envErr != nil {
		slog.Warn("failed to load .env file, using system environment variables", "stage", "load_env", "error", envErr)
	} else {
		slog.Info("loaded environment from .env")
	}
	slog.Info("command parsed",
		"command", cmd.Name,
		"input_path_set", strings.TrimSpace(cmd.Common.InputPath) != "",
		"raw", cmd.Common.Raw,
		"debug", cmd.Common.Debug,
		"retry_count", cmd.LLM.RetryCount,
	)
	slog.Debug("command options prepared",
		"command", cmd.Name,
		"input_path", strings.TrimSpace(cmd.Common.InputPath),
		"has_prompt", strings.TrimSpace(cmd.Common.Prompt) != "",
		"api_url_set", strings.TrimSpace(cmd.LLM.APIURL) != "",
		"model", strings.TrimSpace(cmd.LLM.Model),
		"embedding_model", strings.TrimSpace(cmd.LLM.EmbeddingModel),
	)

	commandCtx, stop := signal.NotifyContext(ctx, syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	slog.Info("preparing input source", "command", cmd.Name, "input_path_set", strings.TrimSpace(cmd.Common.InputPath) != "")
	handle, err := input.Open(cmd.Common.InputPath, stdin)
	if err != nil {
		return err
	}
	defer func() {
		if err := handle.Close(); err != nil {
			slog.Warn("failed to cleanup input", "stage", "close_input", "input_source", handle.DisplayName, "error", err)
			return
		}
		slog.Debug("input cleanup finished", "stage", "close_input")
	}()
	slog.Info("input source ready",
		"input_source", handle.DisplayName,
		"has_path", strings.TrimSpace(handle.Path) != "",
	)
	slog.Debug("creating llm client",
		"command", cmd.Name,
		"api_url_set", strings.TrimSpace(cmd.LLM.APIURL) != "",
		"model", strings.TrimSpace(cmd.LLM.Model),
		"retry_count", cmd.LLM.RetryCount,
	)

	client := llm.NewClient(cmd.LLM.APIURL, cmd.LLM.Model, cmd.LLM.APIKey, cmd.LLM.RetryCount)

	var result string
	switch cmd.Name {
	case "map":
		slog.Info("starting map processing")
		result, err = mapmode.Run(commandCtx, client, handle.Reader, cmd.Map, cmd.Common.Prompt)
	case "tools":
		slog.Info("starting tools processing")
		result, err = tools.Run(commandCtx, client, handle.Path, cmd.Common.Prompt, cmd.Tools)
	case "hybrid":
		slog.Debug("creating embedding client",
			"embedding_model", strings.TrimSpace(cmd.LLM.EmbeddingModel),
			"retry_count", cmd.LLM.RetryCount,
		)
		embedder := llm.NewEmbedder(cmd.LLM.APIURL, cmd.LLM.EmbeddingModel, cmd.LLM.APIKey, cmd.LLM.RetryCount)
		slog.Info("starting hybrid processing")
		result, err = hybrid.Run(commandCtx, client, embedder, hybrid.Source{
			Path:        handle.Path,
			DisplayName: handle.DisplayName,
			Reader:      handle.Reader,
		}, cmd.Hybrid, cmd.Common.Prompt)
	case "rag":
		slog.Debug("creating embedding client",
			"embedding_model", strings.TrimSpace(cmd.LLM.EmbeddingModel),
			"retry_count", cmd.LLM.RetryCount,
		)
		embedder := llm.NewEmbedder(cmd.LLM.APIURL, cmd.LLM.EmbeddingModel, cmd.LLM.APIKey, cmd.LLM.RetryCount)
		slog.Info("starting rag processing")
		result, err = rag.Run(commandCtx, client, embedder, rag.Source{
			Path:        handle.Path,
			DisplayName: handle.DisplayName,
			Reader:      handle.Reader,
		}, cmd.RAG, cmd.Common.Prompt)
	default:
		err = fmt.Errorf("unknown subcommand: %s", cmd.Name)
	}
	if err != nil {
		return err
	}
	slog.Info("processing finished",
		"command", cmd.Name,
		"result_empty", strings.TrimSpace(result) == "",
	)

	formatted, err := formatOutput(result, stdout, cmd.Common.Raw)
	if err != nil {
		return err
	}

	slog.Debug("writing result to stdout", "command", cmd.Name)
	if _, err := fmt.Fprintln(stdout, formatted); err != nil {
		return err
	}
	slog.Info("application run finished", "command", cmd.Name, "exit_code", 0, "duration", roundDurationSeconds(time.Since(startedAt)))
	return nil
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
