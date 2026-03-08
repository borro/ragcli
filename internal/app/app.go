package app

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"os/signal"
	"strings"
	"syscall"
	"time"

	mapmode "github.com/borro/ragcli/internal/app/map"
	"github.com/borro/ragcli/internal/app/rag"
	"github.com/borro/ragcli/internal/app/tools"
	"github.com/borro/ragcli/internal/input"
	"github.com/borro/ragcli/internal/llm"
	"github.com/borro/ragcli/internal/logging"
	"github.com/joho/godotenv"
)

func Run(args []string, stdout io.Writer, stdin io.Reader, version string) int {
	startedAt := time.Now()
	logging.Configure(false)

	envErr := godotenv.Load()

	cmd, err := parseArgs(args)
	if err != nil {
		if usageErr, ok := err.(usageError); ok {
			if usageErr.message != "" {
				slog.Error(usageErr.message, "stage", "parse_args")
			}
			if usageErr.usage != "" {
				_, _ = fmt.Fprintln(stdout, usageErr.usage)
			}
			if usageErr.message == "" {
				return 0
			}
		} else {
			slog.Error("failed to parse CLI args", "stage", "parse_args", "error", err)
		}
		return 1
	}

	logging.Configure(cmd.Common.Verbose)
	slog.Info("application run started",
		"args_count", len(args),
		"version", strings.TrimSpace(version),
		"version_supplied", strings.TrimSpace(version) != "",
	)
	slog.Debug("parsing cli arguments", "args_count", len(args))
	if envErr != nil {
		slog.Warn("failed to load .env file, using system environment variables", "stage", "load_env", "error", envErr)
	} else {
		slog.Info("loaded environment from .env")
	}
	slog.Info("command parsed",
		"command", cmd.Name,
		"input_path_set", strings.TrimSpace(cmd.Common.InputPath) != "",
		"verbose", cmd.Common.Verbose,
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

	if cmd.Common.Version {
		slog.Info("writing version output")
		if _, err := fmt.Fprintln(stdout, version); err != nil {
			slog.Error("failed to write version", "stage", "write_version", "error", err)
			return 1
		}
		slog.Info("application run finished", "stage", "write_version", "exit_code", 0, "duration", roundDurationSeconds(time.Since(startedAt)))
		return 0
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	slog.Info("preparing input source", "command", cmd.Name, "input_path_set", strings.TrimSpace(cmd.Common.InputPath) != "")
	handle, err := input.Open(cmd.Common.InputPath, stdin)
	if err != nil {
		slog.Error("failed to prepare input", "stage", "open_input", "command", cmd.Name, "input_path", strings.TrimSpace(cmd.Common.InputPath), "error", err)
		return 1
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
		result, err = mapmode.Run(ctx, client, handle.Reader, cmd.Map, cmd.Common.Prompt)
	case "tools":
		slog.Info("starting tools processing")
		result, err = tools.Run(ctx, client, handle.Path, cmd.Common.Prompt, cmd.Tools)
	case "rag":
		slog.Debug("creating embedding client",
			"embedding_model", strings.TrimSpace(cmd.LLM.EmbeddingModel),
			"retry_count", cmd.LLM.RetryCount,
		)
		embedder := llm.NewEmbedder(cmd.LLM.APIURL, cmd.LLM.EmbeddingModel, cmd.LLM.APIKey, cmd.LLM.RetryCount)
		slog.Info("starting rag processing")
		result, err = rag.Run(ctx, client, embedder, rag.Source{
			Path:        handle.Path,
			DisplayName: handle.DisplayName,
			Reader:      handle.Reader,
		}, cmd.RAG, cmd.Common.Prompt)
	default:
		err = fmt.Errorf("unknown subcommand: %s", cmd.Name)
	}
	if err != nil {
		slog.Error("processing failed", "stage", "process", "command", cmd.Name, "input_source", handle.DisplayName, "error", err)
		return 1
	}
	slog.Info("processing finished",
		"command", cmd.Name,
		"result_empty", strings.TrimSpace(result) == "",
	)

	slog.Debug("writing result to stdout", "command", cmd.Name)
	if _, err := fmt.Fprintln(stdout, strings.TrimSpace(result)); err != nil {
		slog.Error("failed to write result", "stage", "write_result", "command", cmd.Name, "error", err)
		return 1
	}
	slog.Info("application run finished", "command", cmd.Name, "exit_code", 0, "duration", roundDurationSeconds(time.Since(startedAt)))
	return 0
}

func roundDurationSeconds(duration time.Duration) float64 {
	return float64(duration.Round(time.Millisecond)) / float64(time.Second)
}
