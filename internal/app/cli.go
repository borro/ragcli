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
		Usage:       "print the version",
		HideDefault: true,
		Local:       true,
	}

	return &cli.Command{
		Name:                  "ragcli",
		Usage:                 "Work with local text through map, RAG, and tool-based retrieval",
		Version:               strings.TrimSpace(cfg.version),
		Flags:                 globalFlags(),
		Action:                showRootHelp,
		OnUsageError:          showUsageHelp,
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
			usage:       "Обработать большой текст чанками через map-reduce",
			description: mapCommandDescription(),
			flags: []cli.Flag{
				&cli.IntFlag{
					Name:    "concurrency",
					Aliases: []string{"c"},
					Usage:   "Количество параллельных запросов к chat-модели.",
					Value:   1,
					Sources: cli.EnvVars("CONCURRENCY"),
				},
				&cli.IntFlag{
					Name:    "length",
					Aliases: []string{"l"},
					Usage:   "Приблизительный лимит токенов на chunk; внутри используется консервативно.",
					Value:   10000,
					Sources: cli.EnvVars("LENGTH"),
				},
			},
			bind:    bindMapCommand,
			execute: cfg.execute,
		}),
		newPromptCLICommand(promptCommandConfig{
			name:        "rag",
			usage:       "Построить локальный retrieval index и ответить по evidence chunks",
			description: ragCommandDescription(),
			flags: []cli.Flag{
				&cli.StringFlag{
					Name:    "embedding-model",
					Usage:   "Embedding-модель для retrieval index.",
					Value:   defaultLLMOptions().EmbeddingModel,
					Sources: cli.EnvVars("EMBEDDING_MODEL"),
				},
				&cli.IntFlag{
					Name:    "rag-top-k",
					Usage:   "Сколько чанков взять после vector search.",
					Value:   8,
					Sources: cli.EnvVars("RAG_TOP_K"),
				},
				&cli.IntFlag{
					Name:    "rag-final-k",
					Usage:   "Сколько evidence chunks оставить перед финальным ответом.",
					Value:   4,
					Sources: cli.EnvVars("RAG_FINAL_K"),
				},
				&cli.IntFlag{
					Name:    "rag-chunk-size",
					Usage:   "Размер чанка для индекса.",
					Value:   1800,
					Sources: cli.EnvVars("RAG_CHUNK_SIZE"),
				},
				&cli.IntFlag{
					Name:    "rag-chunk-overlap",
					Usage:   "Перекрытие между соседними чанками.",
					Value:   200,
					Sources: cli.EnvVars("RAG_CHUNK_OVERLAP"),
				},
				&cli.DurationFlag{
					Name:    "rag-index-ttl",
					Usage:   "Время жизни локального индекса.",
					Value:   24 * time.Hour,
					Sources: cli.EnvVars("RAG_INDEX_TTL"),
				},
				&cli.StringFlag{
					Name:    "rag-index-dir",
					Usage:   "Каталог локального индекса.",
					Value:   defaultRAGIndexDir(),
					Sources: cli.EnvVars("RAG_INDEX_DIR"),
				},
				&cli.StringFlag{
					Name:    "rag-rerank",
					Usage:   "Стратегия rerank: heuristic, off, model.",
					Value:   "heuristic",
					Sources: cli.EnvVars("RAG_RERANK"),
				},
			},
			bind:    bindRAGCommand,
			execute: cfg.execute,
		}),
		newPromptCLICommand(promptCommandConfig{
			name:        "hybrid",
			usage:       "Комбинировать retrieval, локальное дочитывание и map-агрегацию",
			description: hybridCommandDescription(),
			flags: []cli.Flag{
				&cli.StringFlag{
					Name:    "embedding-model",
					Usage:   "Embedding-модель для semantic retrieval.",
					Value:   defaultLLMOptions().EmbeddingModel,
					Sources: cli.EnvVars("EMBEDDING_MODEL"),
				},
				&cli.IntFlag{
					Name:    "rag-chunk-size",
					Usage:   "Целевой размер meso-сегмента для semantic retrieval.",
					Value:   1800,
					Sources: cli.EnvVars("RAG_CHUNK_SIZE"),
				},
				&cli.IntFlag{
					Name:    "rag-chunk-overlap",
					Usage:   "Перекрытие соседних meso-сегментов.",
					Value:   200,
					Sources: cli.EnvVars("RAG_CHUNK_OVERLAP"),
				},
				&cli.DurationFlag{
					Name:    "rag-index-ttl",
					Usage:   "Время жизни локального semantic индекса.",
					Value:   24 * time.Hour,
					Sources: cli.EnvVars("RAG_INDEX_TTL"),
				},
				&cli.StringFlag{
					Name:    "rag-index-dir",
					Usage:   "Каталог локального semantic индекса.",
					Value:   defaultRAGIndexDir(),
					Sources: cli.EnvVars("RAG_INDEX_DIR"),
				},
				&cli.IntFlag{
					Name:    "hybrid-top-k",
					Usage:   "Сколько fusion-кандидатов брать после retrieval.",
					Value:   8,
					Sources: cli.EnvVars("HYBRID_TOP_K"),
				},
				&cli.IntFlag{
					Name:    "hybrid-final-k",
					Usage:   "Сколько регионов оставить для финального синтеза.",
					Value:   4,
					Sources: cli.EnvVars("HYBRID_FINAL_K"),
				},
				&cli.IntFlag{
					Name:    "hybrid-map-k",
					Usage:   "По скольким регионам запускать map-извлечение фактов.",
					Value:   4,
					Sources: cli.EnvVars("HYBRID_MAP_K"),
				},
				&cli.IntFlag{
					Name:    "hybrid-read-window",
					Usage:   "Сколько строк дочитывать вокруг сильного совпадения.",
					Value:   3,
					Sources: cli.EnvVars("HYBRID_READ_WINDOW"),
				},
				&cli.StringFlag{
					Name:    "hybrid-fallback",
					Usage:   "Fallback при слабом retrieval или недоступных embeddings: map, rag-only, fail.",
					Value:   "map",
					Sources: cli.EnvVars("HYBRID_FALLBACK"),
				},
			},
			bind:    bindHybridCommand,
			execute: cfg.execute,
		}),
		newPromptCLICommand(promptCommandConfig{
			name:        "tools",
			usage:       "Дать модели инструменты для точечного исследования файла",
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
		Usage: "Показать версию ragcli",
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
			Usage:   "Путь к входному файлу. Если не задан, команда читает stdin.",
			Sources: cli.EnvVars("INPUT_FILE"),
		},
		&cli.StringFlag{
			Name:    "api-url",
			Usage:   "URL OpenAI-compatible API.",
			Value:   defaultLLMOptions().APIURL,
			Sources: cli.EnvVars("LLM_API_URL"),
		},
		&cli.StringFlag{
			Name:    "api-key",
			Usage:   "API key для OpenAI-compatible API.",
			Sources: cli.EnvVars("OPENAI_API_KEY"),
		},
		&cli.StringFlag{
			Name:    "model",
			Usage:   "Chat-модель для map/tools/rag answer generation.",
			Value:   defaultLLMOptions().Model,
			Sources: cli.EnvVars("LLM_MODEL"),
		},
		&cli.IntFlag{
			Name:    "retry",
			Aliases: []string{"r"},
			Usage:   "Количество retry для LLM HTTP-клиента.",
			Value:   defaultLLMOptions().RetryCount,
			Sources: cli.EnvVars("RETRY"),
		},
		&cli.BoolFlag{
			Name:    "raw",
			Usage:   "Печатать финальный ответ как есть, без markdown-рендера в терминале.",
			Sources: cli.EnvVars("RAW"),
		},
		&cli.BoolFlag{
			Name:    "debug",
			Aliases: []string{"d"},
			Usage:   "Включить debug runtime-логи в stderr; без флага ошибки печатаются одной строкой.",
			Sources: cli.EnvVars("DEBUG"),
		},
		&cli.BoolFlag{
			Name:    "verbose",
			Aliases: []string{"v"},
			Usage:   "Показывать ход выполнения в stderr отдельными строками статуса.",
			Sources: cli.EnvVars("VERBOSE"),
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

func bindCommandBase(cmd *cli.Command, name string) Command {
	return Command{
		Name:   name,
		Common: bindCommonOptions(cmd),
		LLM:    bindLLMOptions(cmd),
	}
}

func bindCommonOptions(cmd *cli.Command) CommonOptions {
	return CommonOptions{
		InputPath: cmd.String("file"),
		Prompt:    bindPrompt(cmd),
		Raw:       cmd.Bool("raw"),
		Debug:     cmd.Bool("debug"),
		Verbose:   cmd.Bool("verbose"),
	}
}

func bindLLMOptions(cmd *cli.Command) LLMOptions {
	ops := defaultLLMOptions()
	ops.APIURL = cmd.String("api-url")
	ops.Model = cmd.String("model")
	ops.APIKey = cmd.String("api-key")
	ops.RetryCount = cmd.Int("retry")
	return ops
}

func bindPrompt(cmd *cli.Command) string {
	return strings.TrimSpace(strings.Join(cmd.StringArgs("prompt"), " "))
}

func bindMapCommand(cmd *cli.Command) (Command, error) {
	bound := bindCommandBase(cmd, "map")
	bound.Map = normalizeMapOptions(mapmode.Options{
		ChunkLength: cmd.Int("length"),
		Concurrency: cmd.Int("concurrency"),
		InputPath:   bound.Common.InputPath,
	})

	if err := validatePrompt(bound.Common.Prompt); err != nil {
		return Command{}, err
	}
	return bound, nil
}

func normalizeMapOptions(options mapmode.Options) mapmode.Options {
	options.Concurrency = maxInt(options.Concurrency, 1)
	options.ChunkLength = maxInt(options.ChunkLength, 1000)
	return options
}

func bindRAGCommand(cmd *cli.Command) (Command, error) {
	bound := bindCommandBase(cmd, "rag")
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
	bound := bindCommandBase(cmd, "hybrid")
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
	bound := bindCommandBase(cmd, "tools")
	bound.Tools = tools.Options{}

	if err := validatePrompt(bound.Common.Prompt); err != nil {
		return Command{}, err
	}
	return bound, nil
}

func validatePrompt(prompt string) error {
	if strings.TrimSpace(prompt) == "" {
		return fmt.Errorf("missing required argument: prompt")
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
	return strings.TrimSpace(`
Разбивает входной текст на чанки, отправляет их в chat-модель и затем агрегирует промежуточные ответы.

Examples:
  ragcli map --file document.txt -c 4 "Какие основные выводы?"
  ragcli map --file document.txt --length 8000 --api-url http://localhost:1234/v1 --model my-model "Какие основные выводы?"
  cat document.txt | ragcli map "Какие основные выводы из этого документа?"
`)
}

func ragCommandDescription() string {
	return strings.TrimSpace(`
Строит локальный retrieval index по входному файлу или stdin, подбирает релевантные evidence chunks и формирует ответ только по ним.

Examples:
  ragcli rag --file file.txt --model qwen2.5 --embedding-model text-embedding-3-small "Что сказано про retry policy?"
  cat spec.txt | ragcli rag --rag-top-k 10 "Какие ограничения описаны в документе?"
`)
}

func toolsCommandDescription() string {
	return strings.TrimSpace(`
Дает модели инструменты search_file, read_lines и read_around, чтобы она сама исследовала файл и нашла нужные строки.

Examples:
  ragcli tools --file file.txt "Какие ключевые моменты обсуждаются в разделе про архитектуру?"
  ragcli tools --file app.log --debug "Где в логах причины 5xx?"
`)
}

func hybridCommandDescription() string {
	return strings.TrimSpace(`
Комбинирует lexical и semantic retrieval, дочитывает точные строки локально и агрегирует факты только по найденным регионам.

Examples:
  ragcli hybrid --file file.txt "Что в документе сказано про retry policy и ограничения?"
  ragcli hybrid --file app.log --hybrid-fallback rag-only "Где причина 5xx и что происходит рядом?"
  cat handbook.md | ragcli hybrid --hybrid-top-k 10 --hybrid-map-k 5 "Какие шаги release process и rollback?"
`)
}
