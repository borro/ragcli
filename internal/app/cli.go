package app

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/borro/ragcli/internal/app/hybrid"
	mapmode "github.com/borro/ragcli/internal/app/map"
	"github.com/borro/ragcli/internal/app/rag"
	"github.com/borro/ragcli/internal/app/tools"
	"github.com/borro/ragcli/internal/localize"
	"github.com/urfave/cli/v3"
)

type commandExecutor func(context.Context, Command) error
type commandBinder func(*cli.Command) (Command, error)

type cliConfig struct {
	stdout  io.Writer
	stderr  io.Writer
	version string
	execute commandExecutor
}

func newCLI(cfg cliConfig) *cli.Command {
	cli.VersionPrinter = printVersion
	cli.ShowCommandHelp = showCommandHelp
	cli.VersionFlag = &cli.BoolFlag{
		Name:        "version",
		Usage:       localize.T("cli.version.flag.usage"),
		HideDefault: true,
		Local:       true,
	}

	return &cli.Command{
		Name:                  "ragcli",
		Usage:                 localize.T("cli.root.usage"),
		Version:               strings.TrimSpace(cfg.version),
		Flags:                 globalFlags(),
		Action:                showRootHelp,
		OnUsageError:          showUsageHelp,
		Before:                validateGlobalLocale,
		Commands:              buildCLICommands(cfg),
		Writer:                cfg.stdout,
		ErrWriter:             cfg.stderr,
		ExitErrHandler:        suppressExitError,
		HideHelp:              false,
		EnableShellCompletion: true,
	}
}

func buildCLICommands(cfg cliConfig) []*cli.Command {
	commands := []*cli.Command{
		newPromptCLICommand(promptCommandConfig{
			name:        "map",
			usage:       localize.T("cli.command.map.usage"),
			description: mapCommandDescription(),
			flags: []cli.Flag{
				&cli.IntFlag{
					Name:    "concurrency",
					Aliases: []string{"c"},
					Usage:   localize.T("cli.flag.map.concurrency.usage"),
					Value:   1,
					Sources: cli.EnvVars("CONCURRENCY"),
				},
				&cli.IntFlag{
					Name:    "length",
					Aliases: []string{"l"},
					Usage:   localize.T("cli.flag.map.length.usage"),
					Value:   10000,
					Sources: cli.EnvVars("LENGTH"),
				},
			},
			bind:    bindMapCommand,
			execute: cfg.execute,
		}),
		newPromptCLICommand(promptCommandConfig{
			name:        "rag",
			usage:       localize.T("cli.command.rag.usage"),
			description: ragCommandDescription(),
			flags: []cli.Flag{
				&cli.StringFlag{
					Name:    "embedding-model",
					Usage:   localize.T("cli.flag.rag.embedding_model.usage"),
					Value:   defaultLLMOptions().EmbeddingModel,
					Sources: cli.EnvVars("EMBEDDING_MODEL"),
				},
				&cli.IntFlag{
					Name:    "rag-top-k",
					Usage:   localize.T("cli.flag.rag.top_k.usage"),
					Value:   8,
					Sources: cli.EnvVars("RAG_TOP_K"),
				},
				&cli.IntFlag{
					Name:    "rag-final-k",
					Usage:   localize.T("cli.flag.rag.final_k.usage"),
					Value:   4,
					Sources: cli.EnvVars("RAG_FINAL_K"),
				},
				&cli.IntFlag{
					Name:    "rag-chunk-size",
					Usage:   localize.T("cli.flag.rag.chunk_size.usage"),
					Value:   1800,
					Sources: cli.EnvVars("RAG_CHUNK_SIZE"),
				},
				&cli.IntFlag{
					Name:    "rag-chunk-overlap",
					Usage:   localize.T("cli.flag.rag.chunk_overlap.usage"),
					Value:   200,
					Sources: cli.EnvVars("RAG_CHUNK_OVERLAP"),
				},
				&cli.DurationFlag{
					Name:    "rag-index-ttl",
					Usage:   localize.T("cli.flag.rag.index_ttl.usage"),
					Value:   24 * time.Hour,
					Sources: cli.EnvVars("RAG_INDEX_TTL"),
				},
				&cli.StringFlag{
					Name:    "rag-index-dir",
					Usage:   localize.T("cli.flag.rag.index_dir.usage"),
					Value:   defaultRAGIndexDir(),
					Sources: cli.EnvVars("RAG_INDEX_DIR"),
				},
				&cli.StringFlag{
					Name:    "rag-rerank",
					Usage:   localize.T("cli.flag.rag.rerank.usage"),
					Value:   "heuristic",
					Sources: cli.EnvVars("RAG_RERANK"),
				},
			},
			bind:    bindRAGCommand,
			execute: cfg.execute,
		}),
		newPromptCLICommand(promptCommandConfig{
			name:        "hybrid",
			usage:       localize.T("cli.command.hybrid.usage"),
			description: hybridCommandDescription(),
			flags: []cli.Flag{
				&cli.StringFlag{
					Name:    "embedding-model",
					Usage:   localize.T("cli.flag.hybrid.embedding_model.usage"),
					Value:   defaultLLMOptions().EmbeddingModel,
					Sources: cli.EnvVars("EMBEDDING_MODEL"),
				},
				&cli.IntFlag{
					Name:    "rag-chunk-size",
					Usage:   localize.T("cli.flag.hybrid.chunk_size.usage"),
					Value:   1800,
					Sources: cli.EnvVars("RAG_CHUNK_SIZE"),
				},
				&cli.IntFlag{
					Name:    "rag-chunk-overlap",
					Usage:   localize.T("cli.flag.hybrid.chunk_overlap.usage"),
					Value:   200,
					Sources: cli.EnvVars("RAG_CHUNK_OVERLAP"),
				},
				&cli.DurationFlag{
					Name:    "rag-index-ttl",
					Usage:   localize.T("cli.flag.hybrid.index_ttl.usage"),
					Value:   24 * time.Hour,
					Sources: cli.EnvVars("RAG_INDEX_TTL"),
				},
				&cli.StringFlag{
					Name:    "rag-index-dir",
					Usage:   localize.T("cli.flag.hybrid.index_dir.usage"),
					Value:   defaultRAGIndexDir(),
					Sources: cli.EnvVars("RAG_INDEX_DIR"),
				},
				&cli.IntFlag{
					Name:    "hybrid-top-k",
					Usage:   localize.T("cli.flag.hybrid.top_k.usage"),
					Value:   8,
					Sources: cli.EnvVars("HYBRID_TOP_K"),
				},
				&cli.IntFlag{
					Name:    "hybrid-final-k",
					Usage:   localize.T("cli.flag.hybrid.final_k.usage"),
					Value:   4,
					Sources: cli.EnvVars("HYBRID_FINAL_K"),
				},
				&cli.IntFlag{
					Name:    "hybrid-map-k",
					Usage:   localize.T("cli.flag.hybrid.map_k.usage"),
					Value:   4,
					Sources: cli.EnvVars("HYBRID_MAP_K"),
				},
				&cli.IntFlag{
					Name:    "hybrid-read-window",
					Usage:   localize.T("cli.flag.hybrid.read_window.usage"),
					Value:   3,
					Sources: cli.EnvVars("HYBRID_READ_WINDOW"),
				},
				&cli.StringFlag{
					Name:    "hybrid-fallback",
					Usage:   localize.T("cli.flag.hybrid.fallback.usage"),
					Value:   "map",
					Sources: cli.EnvVars("HYBRID_FALLBACK"),
				},
			},
			bind:    bindHybridCommand,
			execute: cfg.execute,
		}),
		newPromptCLICommand(promptCommandConfig{
			name:        "tools",
			usage:       localize.T("cli.command.tools.usage"),
			description: toolsCommandDescription(),
			bind:        bindToolsCommand,
			execute:     cfg.execute,
		}),
		newVersionCLICommand(),
	}

	for _, cmd := range commands {
		cmd.OnUsageError = showUsageHelp
	}

	return commands
}

type promptCommandConfig struct {
	name        string
	usage       string
	usageText   string
	argsUsage   string
	description string
	flags       []cli.Flag
	bind        commandBinder
	execute     commandExecutor
}

func newPromptCLICommand(cfg promptCommandConfig) *cli.Command {
	return &cli.Command{
		Name:        cfg.name,
		Usage:       cfg.usage,
		UsageText:   cfg.usageText,
		ArgsUsage:   cfg.argsUsage,
		Description: cfg.description,
		Arguments:   promptArguments(),
		Flags:       cfg.flags,
		Action:      bindAndExecute(cfg.bind, cfg.execute),
	}
}

func promptArguments() []cli.Argument {
	return []cli.Argument{
		&cli.StringArgs{Name: "prompt", Min: 0, Max: -1, UsageText: "<prompt>"},
	}
}

func bindAndExecute(bind commandBinder, execute commandExecutor) cli.ActionFunc {
	return func(ctx context.Context, cmd *cli.Command) error {
		bound, err := bind(cmd)
		if err != nil {
			_ = cli.ShowSubcommandHelp(cmd)
			return err
		}
		return execute(ctx, bound)
	}
}

func showRootHelp(ctx context.Context, cmd *cli.Command) error {
	return cli.ShowRootCommandHelp(cmd)
}

func newVersionCLICommand() *cli.Command {
	return &cli.Command{
		Name:  "version",
		Usage: localize.T("cli.command.version.usage"),
		Action: func(_ context.Context, cmd *cli.Command) error {
			cli.ShowVersion(cmd.Root())
			return nil
		},
	}
}

func printVersion(cmd *cli.Command) {
	_, _ = fmt.Fprintln(cmd.Root().Writer, strings.TrimSpace(cmd.Version))
}

// Workaround for urfave/cli/v3: built-in "help <command>" treats leaf commands
// as subcommands because of the hidden help command and drops GLOBAL OPTIONS.
func showCommandHelp(ctx context.Context, cmd *cli.Command, commandName string) error {
	for _, subCmd := range cmd.Commands {
		if !subCmd.HasName(commandName) {
			continue
		}

		tmpl := subCmd.CustomHelpTemplate
		if tmpl == "" {
			if len(subCmd.VisibleCommands()) == 0 {
				tmpl = cli.CommandHelpTemplate
			} else {
				tmpl = cli.SubcommandHelpTemplate
			}
		}

		cli.HelpPrinter(cmd.Root().Writer, tmpl, subCmd)
		return nil
	}

	return cli.DefaultShowCommandHelp(ctx, cmd, commandName)
}

func globalFlags() []cli.Flag {
	return []cli.Flag{
		&cli.StringFlag{
			Name:    "file",
			Aliases: []string{"f"},
			Usage:   localize.T("cli.flag.file.usage"),
			Sources: cli.EnvVars("INPUT_FILE"),
		},
		&cli.StringFlag{
			Name:    "api-url",
			Usage:   localize.T("cli.flag.api_url.usage"),
			Value:   defaultLLMOptions().APIURL,
			Sources: cli.EnvVars("LLM_API_URL"),
		},
		&cli.StringFlag{
			Name:    "api-key",
			Usage:   localize.T("cli.flag.api_key.usage"),
			Sources: cli.EnvVars("OPENAI_API_KEY"),
		},
		&cli.StringFlag{
			Name:    "proxy-url",
			Usage:   localize.T("cli.flag.proxy_url.usage"),
			Sources: cli.EnvVars("LLM_PROXY_URL"),
		},
		&cli.BoolFlag{
			Name:    "no-proxy",
			Usage:   localize.T("cli.flag.no_proxy.usage"),
			Sources: cli.EnvVars("LLM_NO_PROXY"),
		},
		&cli.StringFlag{
			Name:    "model",
			Usage:   localize.T("cli.flag.model.usage"),
			Value:   defaultLLMOptions().Model,
			Sources: cli.EnvVars("LLM_MODEL"),
		},
		&cli.IntFlag{
			Name:    "retry",
			Aliases: []string{"r"},
			Usage:   localize.T("cli.flag.retry.usage"),
			Value:   defaultLLMOptions().RetryCount,
			Sources: cli.EnvVars("RETRY"),
		},
		&cli.BoolFlag{
			Name:    "raw",
			Usage:   localize.T("cli.flag.raw.usage"),
			Sources: cli.EnvVars("RAW"),
		},
		&cli.BoolFlag{
			Name:    "debug",
			Aliases: []string{"d"},
			Usage:   localize.T("cli.flag.debug.usage"),
			Sources: cli.EnvVars("DEBUG"),
		},
		&cli.BoolFlag{
			Name:    "verbose",
			Aliases: []string{"v"},
			Usage:   localize.T("cli.flag.verbose.usage"),
			Sources: cli.EnvVars("VERBOSE"),
		},
		&cli.StringFlag{
			Name:    "lang",
			Usage:   localize.T("cli.flag.lang.usage"),
			Sources: cli.EnvVars("RAGCLI_LANG"),
		},
	}
}

func showUsageHelp(ctx context.Context, cmd *cli.Command, err error, _ bool) error {
	if cmd.Root() == cmd {
		_ = cli.ShowRootCommandHelp(cmd)
		return err
	}

	_ = cli.ShowSubcommandHelp(cmd)
	return err
}

func suppressExitError(context.Context, *cli.Command, error) {}

func bindCommonOptions(cmd *cli.Command) CommonOptions {
	return CommonOptions{
		InputPath: cmd.String("file"),
		Prompt:    bindPrompt(cmd),
		Raw:       cmd.Bool("raw"),
		Debug:     cmd.Bool("debug"),
		Verbose:   cmd.Bool("verbose"),
	}
}

func bindLLMOptions(cmd *cli.Command) (LLMOptions, error) {
	ops := defaultLLMOptions()
	ops.APIURL = cmd.String("api-url")
	ops.Model = cmd.String("model")
	ops.APIKey = cmd.String("api-key")
	ops.ProxyURL = cmd.String("proxy-url")
	ops.NoProxy = cmd.Bool("no-proxy")
	ops.RetryCount = cmd.Int("retry")
	return ops, nil
}

func bindPrompt(cmd *cli.Command) string {
	return strings.TrimSpace(strings.Join(cmd.StringArgs("prompt"), " "))
}

func bindMapCommand(cmd *cli.Command) (Command, error) {
	bound, err := bindCommandBase(cmd, "map")
	if err != nil {
		return Command{}, err
	}
	bound.Map = normalizeMapOptions(mapmode.Options{
		ChunkLength:    cmd.Int("length"),
		LengthExplicit: cmd.IsSet("length"),
		Concurrency:    cmd.Int("concurrency"),
		InputPath:      bound.Common.InputPath,
	})

	if err := validatePrompt(bound.Common.Prompt); err != nil {
		return Command{}, err
	}
	return bound, nil
}

func normalizeMapOptions(options mapmode.Options) mapmode.Options {
	options.Concurrency = maxInt(options.Concurrency, 1)
	return options
}

func bindRAGCommand(cmd *cli.Command) (Command, error) {
	bound, err := bindCommandBase(cmd, "rag")
	if err != nil {
		return Command{}, err
	}
	bound.LLM.EmbeddingModel = normalizeEmbeddingModel(cmd.String("embedding-model"))
	bound.RAG = normalizeRAGOptions(rag.Options{
		TopK:         cmd.Int("rag-top-k"),
		FinalK:       cmd.Int("rag-final-k"),
		ChunkSize:    cmd.Int("rag-chunk-size"),
		ChunkOverlap: cmd.Int("rag-chunk-overlap"),
		IndexTTL:     cmd.Duration("rag-index-ttl"),
		IndexDir:     cmd.String("rag-index-dir"),
		Rerank:       cmd.String("rag-rerank"),
	})

	if err := validatePrompt(bound.Common.Prompt); err != nil {
		return Command{}, err
	}
	return bound, nil
}

func bindHybridCommand(cmd *cli.Command) (Command, error) {
	bound, err := bindCommandBase(cmd, "hybrid")
	if err != nil {
		return Command{}, err
	}
	bound.LLM.EmbeddingModel = normalizeEmbeddingModel(cmd.String("embedding-model"))
	bound.Hybrid = normalizeHybridOptions(hybrid.Options{
		TopK:           cmd.Int("hybrid-top-k"),
		FinalK:         cmd.Int("hybrid-final-k"),
		MapK:           cmd.Int("hybrid-map-k"),
		ReadWindow:     cmd.Int("hybrid-read-window"),
		Fallback:       cmd.String("hybrid-fallback"),
		ChunkSize:      cmd.Int("rag-chunk-size"),
		ChunkOverlap:   cmd.Int("rag-chunk-overlap"),
		IndexTTL:       cmd.Duration("rag-index-ttl"),
		IndexDir:       cmd.String("rag-index-dir"),
		EmbeddingModel: bound.LLM.EmbeddingModel,
	})

	if err := validatePrompt(bound.Common.Prompt); err != nil {
		return Command{}, err
	}
	return bound, nil
}

func normalizeEmbeddingModel(value string) string {
	if strings.TrimSpace(value) == "" {
		return "text-embedding-3-small"
	}
	return value
}

func normalizeRAGOptions(options rag.Options) rag.Options {
	options.TopK = maxInt(options.TopK, 1)
	options.FinalK = maxInt(options.FinalK, 1)
	if options.FinalK > options.TopK {
		options.FinalK = options.TopK
	}
	options.ChunkSize = maxInt(options.ChunkSize, 1000)
	options.ChunkOverlap = maxInt(options.ChunkOverlap, 0)
	if options.ChunkOverlap >= options.ChunkSize {
		options.ChunkOverlap = options.ChunkSize / 4
	}
	if strings.TrimSpace(options.IndexDir) == "" {
		options.IndexDir = defaultRAGIndexDir()
	}
	options.Rerank = normalizeRAGRerank(options.Rerank)
	return options
}

func normalizeHybridOptions(options hybrid.Options) hybrid.Options {
	options.TopK = maxInt(options.TopK, 1)
	options.FinalK = maxInt(options.FinalK, 1)
	if options.FinalK > options.TopK {
		options.FinalK = options.TopK
	}
	options.MapK = maxInt(options.MapK, 1)
	if options.MapK > options.TopK {
		options.MapK = options.TopK
	}
	options.ReadWindow = maxInt(options.ReadWindow, 1)
	options.ChunkSize = maxInt(options.ChunkSize, 1000)
	options.ChunkOverlap = maxInt(options.ChunkOverlap, 0)
	if options.ChunkOverlap >= options.ChunkSize {
		options.ChunkOverlap = options.ChunkSize / 4
	}
	if strings.TrimSpace(options.IndexDir) == "" {
		options.IndexDir = defaultRAGIndexDir()
	}
	options.EmbeddingModel = normalizeEmbeddingModel(options.EmbeddingModel)
	options.Fallback = normalizeHybridFallback(options.Fallback)
	return options
}

func bindToolsCommand(cmd *cli.Command) (Command, error) {
	bound, err := bindCommandBase(cmd, "tools")
	if err != nil {
		return Command{}, err
	}
	bound.Tools = tools.Options{}

	if err := validatePrompt(bound.Common.Prompt); err != nil {
		return Command{}, err
	}
	return bound, nil
}

func bindCommandBase(cmd *cli.Command, name string) (Command, error) {
	llmOptions, err := bindLLMOptions(cmd)
	if err != nil {
		return Command{}, err
	}

	return Command{
		Name:   name,
		Common: bindCommonOptions(cmd),
		LLM:    llmOptions,
	}, nil
}

func validatePrompt(prompt string) error {
	if strings.TrimSpace(prompt) == "" {
		return fmt.Errorf("%s", localize.T("error.prompt.required"))
	}
	return nil
}

func defaultLLMOptions() LLMOptions {
	return LLMOptions{
		APIURL:         "http://localhost:1234/v1",
		Model:          "local-model",
		EmbeddingModel: "text-embedding-nomic-embed-text-v1.5",
		APIKey:         "",
		RetryCount:     3,
	}
}

func normalizeRAGRerank(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "", "heuristic":
		return "heuristic"
	case "off":
		return "off"
	case "model":
		return "model"
	default:
		return "heuristic"
	}
}

func normalizeHybridFallback(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "", "map":
		return "map"
	case "rag-only":
		return "rag-only"
	case "fail":
		return "fail"
	default:
		return "map"
	}
}

func defaultRAGIndexDir() string {
	return os.TempDir() + string(os.PathSeparator) + "ragcli-index"
}

func maxInt(value int, min int) int {
	if value < min {
		return min
	}
	return value
}

func mapCommandDescription() string {
	return strings.TrimSpace(localize.T("cli.description.map"))
}

func ragCommandDescription() string {
	return strings.TrimSpace(localize.T("cli.description.rag"))
}

func toolsCommandDescription() string {
	return strings.TrimSpace(localize.T("cli.description.tools"))
}

func hybridCommandDescription() string {
	return strings.TrimSpace(localize.T("cli.description.hybrid"))
}

func validateGlobalLocale(ctx context.Context, cmd *cli.Command) (context.Context, error) {
	if !cmd.IsSet("lang") {
		return ctx, nil
	}
	if _, valid := localize.Normalize(cmd.String("lang")); valid {
		return ctx, nil
	}
	return ctx, fmt.Errorf("%s", localize.T("error.locale.unsupported", localize.Data{"Value": cmd.String("lang")}))
}
