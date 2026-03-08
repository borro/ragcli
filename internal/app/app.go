package app

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"os/signal"
	"strings"
	"syscall"

	mapmode "github.com/borro/ragcli/internal/app/map"
	"github.com/borro/ragcli/internal/app/rag"
	"github.com/borro/ragcli/internal/app/tools"
	"github.com/borro/ragcli/internal/input"
	"github.com/borro/ragcli/internal/llm"
	"github.com/borro/ragcli/internal/logging"
	"github.com/joho/godotenv"
)

func Run(args []string, stdout io.Writer, stdin io.Reader, version string) int {
	logging.InitProjectRoot("")
	logging.ConfigureLogger(false)

	if err := godotenv.Load(); err != nil {
		slog.Warn("failed to load .env file, using system environment variables", "error", err)
	}

	cmd, err := parseArgs(args)
	if err != nil {
		if usageErr, ok := err.(usageError); ok {
			if usageErr.message != "" {
				slog.Error(usageErr.message)
			}
			if usageErr.usage != "" {
				_, _ = fmt.Fprintln(stdout, usageErr.usage)
			}
			if usageErr.message == "" {
				return 0
			}
		} else {
			slog.Error("failed to parse CLI args", "error", err)
		}
		return 1
	}

	logging.ConfigureLogger(cmd.Common.Verbose)

	if cmd.Common.Version {
		if _, err := fmt.Fprintln(stdout, version); err != nil {
			slog.Error("failed to write version", "error", err)
			return 1
		}
		return 0
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	handle, err := input.Open(cmd.Common.InputPath, stdin)
	if err != nil {
		slog.Error("failed to prepare input", "error", err)
		return 1
	}
	defer func() {
		if err := handle.Close(); err != nil {
			slog.Warn("failed to cleanup input", "error", err)
		}
	}()

	client := llm.NewClient(cmd.LLM.APIURL, cmd.LLM.Model, cmd.LLM.APIKey, cmd.LLM.RetryCount)

	var result string
	switch cmd.Name {
	case "map":
		result, err = mapmode.Run(ctx, client, handle.Reader, cmd.Map, cmd.Common.Prompt)
	case "tools":
		result, err = tools.Run(ctx, client, handle.Path, cmd.Common.Prompt, cmd.Tools)
	case "rag":
		embedder := llm.NewEmbedder(cmd.LLM.APIURL, cmd.LLM.EmbeddingModel, cmd.LLM.APIKey, cmd.LLM.RetryCount)
		result, err = rag.Run(ctx, client, embedder, rag.Source{
			Path:        handle.Path,
			DisplayName: handle.DisplayName,
			Reader:      handle.Reader,
		}, cmd.RAG, cmd.Common.Prompt)
	default:
		err = fmt.Errorf("unknown subcommand: %s", cmd.Name)
	}
	if err != nil {
		slog.Error("processing failed", "command", cmd.Name, "error", err)
		return 1
	}

	if _, err := fmt.Fprintln(stdout, strings.TrimSpace(result)); err != nil {
		slog.Error("failed to write result", "error", err)
		return 1
	}
	return 0
}
