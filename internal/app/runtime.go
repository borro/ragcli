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

	"github.com/borro/ragcli/internal/hybrid"
	"github.com/borro/ragcli/internal/input"
	"github.com/borro/ragcli/internal/llm"
	"github.com/borro/ragcli/internal/localize"
	"github.com/borro/ragcli/internal/logging"
	mapmode "github.com/borro/ragcli/internal/map"
	"github.com/borro/ragcli/internal/rag"
	selfupdatepkg "github.com/borro/ragcli/internal/selfupdate"
	"github.com/borro/ragcli/internal/tools"
	"github.com/borro/ragcli/internal/verbose"
)

type chatClientFactory func(llm.Config) (llm.ChatAutoContextRequester, error)
type embedderFactory func(llm.Config) (llm.EmbeddingRequester, error)
type reporterFactory func(io.Writer, bool, bool) verbose.Reporter
type inputOpener func(string, io.Reader) (*input.Handle, error)
type outputFormatter func(string, io.Writer, bool) (string, error)
type interactionIOOpener func(input.Source, io.Reader, io.Writer) (*interactionIO, error)
type runtimeCommandExecutor func(*commandSession) (commandResult, error)
type selfUpdaterFactory func() (selfUpdater, error)

type selfUpdater interface {
	Run(context.Context, string, selfupdatepkg.Options, verbose.Meter) (selfupdatepkg.Result, error)
}

type interactiveSession interface {
	Ask(context.Context, string, *verbose.Plan) (string, error)
	FollowUpPlan(verbose.Reporter) *verbose.Plan
	Reset()
}

type commandResult struct {
	Output      string
	Interactive interactiveSession
}

type appRuntime struct {
	stdout io.Writer
	stderr io.Writer
	stdin  io.Reader

	version   string
	argsCount int
	envErr    error
	startedAt time.Time

	newChatClient  chatClientFactory
	newEmbedder    embedderFactory
	newReporter    reporterFactory
	openInput      inputOpener
	openREPL       interactionIOOpener
	formatOutput   outputFormatter
	newSelfUpdater selfUpdaterFactory
}

type commandSession struct {
	runtime     *appRuntime
	ctx         context.Context
	inv         commandInvocation
	reporter    verbose.Reporter
	plan        *verbose.Plan
	proxyMode   string
	proxyTarget string
	handle      *input.Handle
	source      input.Source
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
		openREPL:     defaultInteractionIO,
		formatOutput: formatOutput,
		newSelfUpdater: func() (selfUpdater, error) {
			return selfupdatepkg.New(selfupdatepkg.Config{})
		},
	}
}

func (r *appRuntime) execute(ctx context.Context, inv commandInvocation) error {
	logging.Configure(r.stderr, inv.Common.Debug)

	commandCtx, stop := newCommandContext(ctx)
	defer stop()

	reporter := r.newReporter(r.stderr, inv.Common.Verbose, inv.Common.Debug)
	session := &commandSession{
		runtime:  r,
		ctx:      commandCtx,
		inv:      inv,
		reporter: reporter,
		plan:     NewPlan(inv.Name(), reporter),
	}
	defer session.closeInput()
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
		"result_empty", strings.TrimSpace(result.Output) == "",
	)

	if err := session.writeResult(result.Output); err != nil {
		return err
	}
	if inv.Common.Interaction && result.Interactive != nil {
		if err := session.runInteraction(result.Interactive); err != nil {
			return err
		}
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
		"interaction", s.inv.Common.Interaction,
		"retry_count", s.inv.LLM.RetryCount,
	)
	slog.Debug("command options prepared",
		"command", s.inv.Name(),
		"input_path", strings.TrimSpace(s.inv.Common.InputPath),
		"has_prompt", strings.TrimSpace(s.inv.Common.Prompt) != "",
		"api_url_set", strings.TrimSpace(s.inv.LLM.APIURL) != "",
		"model", strings.TrimSpace(s.inv.LLM.Model),
		"embedding_model", strings.TrimSpace(s.inv.LLM.EmbeddingModel),
		"interaction", s.inv.Common.Interaction,
		"proxy_mode", s.proxyMode,
		"proxy_target", s.proxyTarget,
	)
}

func (s *commandSession) ensureInputOpened(meter verbose.Meter) (input.Source, error) {
	meter.Start(localize.T("progress.detail.prepare.open_input"))
	if s.handle != nil {
		meter.Done(localize.T("progress.detail.prepare.input_ready"))
		return s.source, nil
	}

	slog.Info("preparing input source", "command", s.inv.Name(), "input_path_set", strings.TrimSpace(s.inv.Common.InputPath) != "")
	handle, err := s.runtime.openInput(s.inv.Common.InputPath, s.runtime.stdin)
	if err != nil {
		return input.Source{}, err
	}
	s.handle = handle
	s.source = handle.Source()

	slog.Info("input source ready",
		"input_source", s.source.DisplayName(),
		"has_path", s.source.InputPath() != "",
	)
	meter.Done(localize.T("progress.detail.prepare.input_ready"))
	return s.source, nil
}

func (s *commandSession) closeInput() {
	if s.handle == nil {
		return
	}
	if closeErr := s.handle.Close(); closeErr != nil {
		slog.Warn("failed to cleanup input", "stage", "close_input", "input_source", s.source.DisplayName(), "error", closeErr)
		return
	}
	slog.Debug("input cleanup finished", "stage", "close_input")
	s.handle = nil
}

func (s *commandSession) withInput(meter verbose.Meter, fn func(input.Source) (commandResult, error)) (commandResult, error) {
	source, err := s.ensureInputOpened(meter)
	if err != nil {
		return commandResult{}, err
	}
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

func (s *commandSession) withChatInput(meter verbose.Meter, fn func(input.Source, llm.ChatAutoContextRequester) (commandResult, error)) (commandResult, error) {
	return s.withInput(meter, func(source input.Source) (commandResult, error) {
		client, err := s.newChatClient()
		if err != nil {
			return commandResult{}, err
		}
		return fn(source, client)
	})
}

func (s *commandSession) withChatAndEmbeddingInput(
	meter verbose.Meter,
	fn func(input.Source, llm.ChatAutoContextRequester, llm.EmbeddingRequester) (commandResult, error),
) (commandResult, error) {
	return s.withChatInput(meter, func(source input.Source, client llm.ChatAutoContextRequester) (commandResult, error) {
		embedder, err := s.newEmbeddingClient()
		if err != nil {
			return commandResult{}, err
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

func executeMapCommand(session *commandSession) (commandResult, error) {
	opts, err := payloadAs[mapmode.Options](session.inv.payload)
	if err != nil {
		return commandResult{}, err
	}

	return session.withChatInput(session.plan.Stage("prepare"), func(source input.Source, client llm.ChatAutoContextRequester) (commandResult, error) {
		slog.Info("starting map processing")
		if !session.inv.Common.Interaction {
			answer, err := mapmode.Run(session.ctx, client, source, opts, session.inv.Common.Prompt, session.plan)
			return commandResult{Output: answer}, err
		}
		conv, answer, err := mapmode.StartConversation(session.ctx, client, source, opts, session.inv.Common.Prompt, session.plan)
		if err != nil {
			return commandResult{}, err
		}
		return commandResult{Output: answer, Interactive: conv}, nil
	})
}

func executeToolsCommand(session *commandSession) (commandResult, error) {
	opts, err := payloadAs[tools.Options](session.inv.payload)
	if err != nil {
		return commandResult{}, err
	}

	if !opts.EnableRAG {
		return session.withChatInput(session.plan.Stage("prepare"), func(source input.Source, client llm.ChatAutoContextRequester) (commandResult, error) {
			slog.Info("starting tools processing")
			if !session.inv.Common.Interaction {
				answer, err := tools.Run(session.ctx, client, nil, source, opts, session.inv.Common.Prompt, session.plan)
				return commandResult{Output: answer}, err
			}
			conv, answer, err := tools.StartConversation(session.ctx, client, nil, source, opts, session.inv.Common.Prompt, session.plan)
			if err != nil {
				return commandResult{}, err
			}
			return commandResult{Output: answer, Interactive: conv}, nil
		})
	}

	return session.withChatAndEmbeddingInput(session.plan.Stage("prepare"), func(source input.Source, client llm.ChatAutoContextRequester, embedder llm.EmbeddingRequester) (commandResult, error) {
		slog.Info("starting tools processing", "rag_enabled", true)
		if !session.inv.Common.Interaction {
			answer, err := tools.Run(session.ctx, client, embedder, source, opts, session.inv.Common.Prompt, session.plan)
			return commandResult{Output: answer}, err
		}
		conv, answer, err := tools.StartConversation(session.ctx, client, embedder, source, opts, session.inv.Common.Prompt, session.plan)
		if err != nil {
			return commandResult{}, err
		}
		return commandResult{Output: answer, Interactive: conv}, nil
	})
}

func executeRAGCommand(session *commandSession) (commandResult, error) {
	opts, err := payloadAs[rag.Options](session.inv.payload)
	if err != nil {
		return commandResult{}, err
	}

	return session.withChatAndEmbeddingInput(session.plan.Stage("prepare"), func(source input.Source, client llm.ChatAutoContextRequester, embedder llm.EmbeddingRequester) (commandResult, error) {
		slog.Info("starting rag processing")
		if !session.inv.Common.Interaction {
			answer, err := rag.Run(session.ctx, client, embedder, source, opts, session.inv.Common.Prompt, session.plan)
			return commandResult{Output: answer}, err
		}
		conv, answer, err := rag.StartConversation(session.ctx, client, embedder, source, opts, session.inv.Common.Prompt, session.plan)
		if err != nil {
			return commandResult{}, err
		}
		return commandResult{Output: answer, Interactive: conv}, nil
	})
}

func executeHybridCommand(session *commandSession) (commandResult, error) {
	opts, err := payloadAs[hybrid.Options](session.inv.payload)
	if err != nil {
		return commandResult{}, err
	}

	return session.withChatAndEmbeddingInput(session.plan.Stage("prepare"), func(source input.Source, client llm.ChatAutoContextRequester, embedder llm.EmbeddingRequester) (commandResult, error) {
		slog.Info("starting hybrid processing")
		if !session.inv.Common.Interaction {
			answer, err := hybrid.Run(session.ctx, client, embedder, source, opts, session.inv.Common.Prompt, session.plan)
			return commandResult{Output: answer}, err
		}
		conv, answer, err := hybrid.StartConversation(session.ctx, client, embedder, source, opts, session.inv.Common.Prompt, session.plan)
		if err != nil {
			return commandResult{}, err
		}
		return commandResult{Output: answer, Interactive: conv}, nil
	})
}

func executeSelfUpdateCommand(session *commandSession) (commandResult, error) {
	opts, err := payloadAs[selfupdatepkg.Options](session.inv.payload)
	if err != nil {
		return commandResult{}, err
	}

	updater, err := session.runtime.newSelfUpdater()
	if err != nil {
		return commandResult{}, err
	}

	slog.Info("starting self-update", "check_only", opts.CheckOnly)
	result, err := updater.Run(session.ctx, session.runtime.version, opts, session.plan.Stage("update"))
	if err != nil {
		return commandResult{}, err
	}
	return commandResult{Output: result.Output}, nil
}

func payloadAs[T any](payload any) (T, error) {
	var zero T
	value, ok := payload.(T)
	if !ok {
		return zero, errors.New("invalid command payload")
	}
	return value, nil
}
