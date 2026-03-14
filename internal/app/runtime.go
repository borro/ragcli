package app

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/borro/ragcli/internal/input"
	"github.com/borro/ragcli/internal/llm"
	"github.com/borro/ragcli/internal/localize"
	"github.com/borro/ragcli/internal/logging"
	mapmode "github.com/borro/ragcli/internal/map"
	"github.com/borro/ragcli/internal/rag"
	"github.com/borro/ragcli/internal/tools"
	"github.com/borro/ragcli/internal/verbose"
)

type chatClientFactory func(llm.Config) (llm.ChatAutoContextRequester, error)
type embedderFactory func(llm.Config) (llm.EmbeddingRequester, error)
type reporterFactory func(io.Writer, bool, bool) verbose.Reporter
type inputOpener func(string, io.Reader) (*input.Handle, error)
type outputFormatter func(string, io.Writer, bool) (string, error)
type runtimeCommandExecutor func(*commandSession) (string, error)

type appRuntime struct {
	stdout io.Writer
	stderr io.Writer
	stdin  io.Reader

	version   string
	argsCount int
	envErr    error
	startedAt time.Time

	newChatClient chatClientFactory
	newEmbedder   embedderFactory
	newReporter   reporterFactory
	openInput     inputOpener
	formatOutput  outputFormatter
}

type commandSession struct {
	runtime     *appRuntime
	ctx         context.Context
	inv         commandInvocation
	plan        *verbose.Plan
	proxyMode   string
	proxyTarget string
}

func newRuntime(stdout io.Writer, stderr io.Writer, stdin io.Reader, version string, argsCount int, envErr error, startedAt time.Time) *appRuntime {
	return &appRuntime{
		stdout:    stdout,
		stderr:    stderr,
		stdin:     stdin,
		version:   version,
		argsCount: argsCount,
		envErr:    envErr,
		startedAt: startedAt,
		newChatClient: func(cfg llm.Config) (llm.ChatAutoContextRequester, error) {
			return llm.NewClient(cfg)
		},
		newEmbedder: func(cfg llm.Config) (llm.EmbeddingRequester, error) {
			return llm.NewEmbedder(cfg)
		},
		newReporter:  verbose.New,
		openInput:    input.Open,
		formatOutput: formatOutput,
	}
}

func (r *appRuntime) execute(ctx context.Context, inv commandInvocation) error {
	logging.Configure(r.stderr, inv.Common.Debug)

	commandCtx, stop := newCommandContext(ctx)
	defer stop()

	reporter := r.newReporter(r.stderr, inv.Common.Verbose, inv.Common.Debug)
	session := &commandSession{
		runtime: r,
		ctx:     commandCtx,
		inv:     inv,
		plan:    NewPlan(inv.Name(), reporter),
	}
	session.proxyMode, session.proxyTarget = llmProxyLogFields(inv.LLM.ProxyURL, inv.LLM.NoProxy)
	session.logRunStarted()

	if inv.spec == nil || inv.spec.execute == nil {
		return unknownSubcommandError(inv.Name())
	}

	result, err := inv.spec.execute(session)
	if err != nil {
		return err
	}

	slog.Info("processing finished",
		"command", inv.Name(),
		"result_empty", strings.TrimSpace(result) == "",
	)

	if err := session.writeResult(result); err != nil {
		return err
	}

	slog.Info("application run finished",
		"command", inv.Name(),
		"exit_code", 0,
		"duration", roundDurationSeconds(time.Since(r.startedAt)),
	)
	return nil
}

func newCommandContext(ctx context.Context) (context.Context, context.CancelFunc) {
	return signal.NotifyContext(ctx, syscall.SIGINT, syscall.SIGTERM)
}

func (s *commandSession) logRunStarted() {
	slog.Info("application run started",
		"args_count", s.runtime.argsCount,
		"version", strings.TrimSpace(s.runtime.version),
		"version_supplied", strings.TrimSpace(s.runtime.version) != "",
	)
	slog.Debug("parsing cli arguments", "args_count", s.runtime.argsCount)
	if s.runtime.envErr != nil {
		slog.Warn("failed to load .env file, using system environment variables", "stage", "load_env", "error", s.runtime.envErr)
	} else {
		slog.Info("loaded environment from .env")
	}
	slog.Info("command parsed",
		"command", s.inv.Name(),
		"input_path_set", strings.TrimSpace(s.inv.Common.InputPath) != "",
		"raw", s.inv.Common.Raw,
		"debug", s.inv.Common.Debug,
		"retry_count", s.inv.LLM.RetryCount,
	)
	slog.Debug("command options prepared",
		"command", s.inv.Name(),
		"input_path", strings.TrimSpace(s.inv.Common.InputPath),
		"has_prompt", strings.TrimSpace(s.inv.Common.Prompt) != "",
		"api_url_set", strings.TrimSpace(s.inv.LLM.APIURL) != "",
		"model", strings.TrimSpace(s.inv.LLM.Model),
		"embedding_model", strings.TrimSpace(s.inv.LLM.EmbeddingModel),
		"proxy_mode", s.proxyMode,
		"proxy_target", s.proxyTarget,
	)
}

func (s *commandSession) withInput(meter verbose.Meter, fn func(input.Source) (string, error)) (string, error) {
	meter.Start(localize.T("progress.detail.prepare.open_input"))
	slog.Info("preparing input source", "command", s.inv.Name(), "input_path_set", strings.TrimSpace(s.inv.Common.InputPath) != "")

	handle, err := s.runtime.openInput(s.inv.Common.InputPath, s.runtime.stdin)
	if err != nil {
		return "", err
	}
	source := handle.Source()
	defer func() {
		if closeErr := handle.Close(); closeErr != nil {
			slog.Warn("failed to cleanup input", "stage", "close_input", "input_source", source.DisplayName(), "error", closeErr)
			return
		}
		slog.Debug("input cleanup finished", "stage", "close_input")
	}()

	slog.Info("input source ready",
		"input_source", source.DisplayName(),
		"has_path", source.InputPath() != "",
	)
	meter.Done(localize.T("progress.detail.prepare.input_ready"))
	return fn(source)
}

func (s *commandSession) chatConfig() llm.Config {
	return llm.Config{
		BaseURL:    s.inv.LLM.APIURL,
		Model:      s.inv.LLM.Model,
		APIKey:     s.inv.LLM.APIKey,
		RetryCount: s.inv.LLM.RetryCount,
		ProxyURL:   s.inv.LLM.ProxyURL,
		NoProxy:    s.inv.LLM.NoProxy,
	}
}

func (s *commandSession) embeddingConfig() llm.Config {
	cfg := s.chatConfig()
	cfg.Model = s.inv.LLM.EmbeddingModel
	return cfg
}

func (s *commandSession) newChatClient() (llm.ChatAutoContextRequester, error) {
	slog.Debug("creating llm client",
		"command", s.inv.Name(),
		"api_url_set", strings.TrimSpace(s.inv.LLM.APIURL) != "",
		"model", strings.TrimSpace(s.inv.LLM.Model),
		"retry_count", s.inv.LLM.RetryCount,
		"proxy_mode", s.proxyMode,
		"proxy_target", s.proxyTarget,
	)
	return s.runtime.newChatClient(s.chatConfig())
}

func (s *commandSession) newEmbeddingClient() (llm.EmbeddingRequester, error) {
	slog.Debug("creating embedding client",
		"embedding_model", strings.TrimSpace(s.inv.LLM.EmbeddingModel),
		"retry_count", s.inv.LLM.RetryCount,
		"proxy_mode", s.proxyMode,
		"proxy_target", s.proxyTarget,
	)
	return s.runtime.newEmbedder(s.embeddingConfig())
}

func (s *commandSession) withChatInput(meter verbose.Meter, fn func(input.Source, llm.ChatAutoContextRequester) (string, error)) (string, error) {
	return s.withInput(meter, func(source input.Source) (string, error) {
		client, err := s.newChatClient()
		if err != nil {
			return "", err
		}
		return fn(source, client)
	})
}

func (s *commandSession) withChatAndEmbeddingInput(
	meter verbose.Meter,
	fn func(input.Source, llm.ChatAutoContextRequester, llm.EmbeddingRequester) (string, error),
) (string, error) {
	return s.withChatInput(meter, func(source input.Source, client llm.ChatAutoContextRequester) (string, error) {
		embedder, err := s.newEmbeddingClient()
		if err != nil {
			return "", err
		}
		return fn(source, client, embedder)
	})
}

func (s *commandSession) writeResult(result string) error {
	formatted, err := s.runtime.formatOutput(result, s.runtime.stdout, s.inv.Common.Raw)
	if err != nil {
		return err
	}

	slog.Debug("writing result to stdout", "command", s.inv.Name())
	if _, err := fmt.Fprintln(s.runtime.stdout, formatted); err != nil {
		return err
	}
	return nil
}

func unknownSubcommandError(name string) error {
	return fmt.Errorf("%s", localize.T("error.app.unknown_subcommand", localize.Data{"Name": name}))
}

func executeMapCommand(session *commandSession) (string, error) {
	opts, err := payloadAs[mapmode.Options](session.inv.payload)
	if err != nil {
		return "", err
	}

	return session.withChatInput(session.plan.Stage("prepare"), func(source input.Source, client llm.ChatAutoContextRequester) (string, error) {
		slog.Info("starting map processing")
		return mapmode.Run(session.ctx, client, source, opts, session.inv.Common.Prompt, session.plan)
	})
}

func executeToolsCommand(session *commandSession) (string, error) {
	opts, err := payloadAs[tools.Options](session.inv.payload)
	if err != nil {
		return "", err
	}

	if !opts.EnableRAG {
		return session.withChatInput(session.plan.Stage("prepare"), func(source input.Source, client llm.ChatAutoContextRequester) (string, error) {
			slog.Info("starting tools processing")
			return tools.Run(session.ctx, client, nil, source, opts, session.inv.Common.Prompt, session.plan)
		})
	}

	return session.withChatAndEmbeddingInput(session.plan.Stage("prepare"), func(source input.Source, client llm.ChatAutoContextRequester, embedder llm.EmbeddingRequester) (string, error) {
		slog.Info("starting tools processing", "rag_enabled", true)
		return tools.Run(session.ctx, client, embedder, source, opts, session.inv.Common.Prompt, session.plan)
	})
}

func executeRAGCommand(session *commandSession) (string, error) {
	opts, err := payloadAs[rag.Options](session.inv.payload)
	if err != nil {
		return "", err
	}

	return session.withChatAndEmbeddingInput(session.plan.Stage("prepare"), func(source input.Source, client llm.ChatAutoContextRequester, embedder llm.EmbeddingRequester) (string, error) {
		slog.Info("starting rag processing")
		return rag.Run(session.ctx, client, embedder, source, opts, session.inv.Common.Prompt, session.plan)
	})
}

func payloadAs[T any](payload any) (T, error) {
	var zero T
	value, ok := payload.(T)
	if !ok {
		return zero, errors.New("invalid command payload")
	}
	return value, nil
}
