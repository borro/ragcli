package app

import (
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/borro/ragcli/internal/hybrid"
	"github.com/borro/ragcli/internal/localize"
	mapmode "github.com/borro/ragcli/internal/map"
	"github.com/borro/ragcli/internal/rag"
	toolsmode "github.com/borro/ragcli/internal/tools"
	"github.com/borro/ragcli/internal/verbose"
	"github.com/urfave/cli/v3"
)

type flagSpec struct {
	names      []string
	takesValue bool
	build      func() cli.Flag
}

type commandSpec struct {
	name        string
	usage       string
	description string
	flagSpecs   []flagSpec
	template    templateSpec
	bind        func(*cli.Command, *commandSpec) (commandInvocation, error)
	execute     runtimeCommandExecutor
}

type templateSpec struct {
	modeKey string
	stages  []stageTemplateSpec
}

type stageTemplateSpec struct {
	key      string
	labelKey string
	slots    int
}

type Template struct {
	Mode   string
	Stages []verbose.StageDef
}

func TemplateFor(command string) Template {
	spec, ok := commandSpecByName(command)
	if !ok {
		return Template{
			Mode: command,
			Stages: []verbose.StageDef{
				{Key: "prepare", Label: localize.T("progress.stage.prepare"), Slots: 1},
			},
		}
	}

	return templateFromSpec(spec)
}

func NewPlan(command string, reporter verbose.Reporter) *verbose.Plan {
	template := TemplateFor(command)
	return verbose.NewPlan(reporter, template.Mode, template.Stages...)
}

func applyPromptArgCompatibility(args []string) []string {
	if len(args) == 0 {
		return args
	}

	spec, ok := commandSpecByName(args[0])
	if !ok {
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
		if promptFlagConsumesValue(spec, token) && !strings.Contains(token, "=") && i+1 < len(subArgs) {
			i++
		}
	}

	return args
}

func commandSpecs() []*commandSpec {
	return []*commandSpec{
		{
			name:        "map",
			usage:       localize.T("cli.command.map.usage"),
			description: mapCommandDescription(),
			flagSpecs: []flagSpec{
				{
					names:      []string{"--concurrency", "-c"},
					takesValue: true,
					build: func() cli.Flag {
						return &cli.IntFlag{
							Name:    "concurrency",
							Aliases: []string{"c"},
							Usage:   localize.T("cli.flag.map.concurrency.usage"),
							Value:   1,
							Sources: cli.EnvVars("CONCURRENCY"),
						}
					},
				},
				{
					names:      []string{"--length", "-l"},
					takesValue: true,
					build: func() cli.Flag {
						return &cli.IntFlag{
							Name:    "length",
							Aliases: []string{"l"},
							Usage:   localize.T("cli.flag.map.length.usage"),
							Value:   10000,
							Sources: cli.EnvVars("LENGTH"),
						}
					},
				},
			},
			template: progressTemplate("progress.mode.map",
				stageTemplate("prepare", "progress.stage.prepare", 2),
				stageTemplate("chunking", "progress.stage.chunking", 4),
				stageTemplate("chunks", "progress.stage.chunks", 9),
				stageTemplate("reduce", "progress.stage.reduce", 5),
				stageTemplate("verify", "progress.stage.verify", 2),
				stageTemplate("final", "progress.stage.final", 2),
			),
			bind:    bindMapInvocation,
			execute: executeMapCommand,
		},
		{
			name:        "rag",
			usage:       localize.T("cli.command.rag.usage"),
			description: ragCommandDescription(),
			flagSpecs:   ragFlagSpecs(true),
			template: progressTemplate("progress.mode.rag",
				stageTemplate("prepare", "progress.stage.prepare", 2),
				stageTemplate("index", "progress.stage.index", 8),
				stageTemplate("retrieval", "progress.stage.retrieval", 7),
				stageTemplate("final", "progress.stage.final", 7),
			),
			bind:    bindRAGInvocation,
			execute: executeRAGCommand,
		},
		{
			name:        "tools",
			usage:       localize.T("cli.command.tools.usage"),
			description: toolsCommandDescription(),
			flagSpecs:   append([]flagSpec{toolsRAGFlagSpec()}, ragFlagSpecs(false)...),
			template: progressTemplate("progress.mode.tools",
				stageTemplate("prepare", "progress.stage.prepare", 2),
				stageTemplate("index", "progress.stage.index", 8),
				stageTemplate("init", "progress.stage.init", 2),
				stageTemplate("loop", "progress.stage.loop", 10),
				stageTemplate("final", "progress.stage.final", 2),
			),
			bind:    bindToolsInvocation,
			execute: executeToolsCommand,
		},
		{
			name:        "hybrid",
			usage:       localize.T("cli.command.hybrid.usage"),
			description: hybridCommandDescription(),
			flagSpecs:   ragFlagSpecs(false),
			template: progressTemplate("progress.mode.hybrid",
				stageTemplate("prepare", "progress.stage.prepare", 2),
				stageTemplate("index", "progress.stage.index", 8),
				stageTemplate("retrieval", "progress.stage.retrieval", 4),
				stageTemplate("init", "progress.stage.init", 2),
				stageTemplate("loop", "progress.stage.loop", 10),
				stageTemplate("final", "progress.stage.final", 2),
			),
			bind:    bindHybridInvocation,
			execute: executeHybridCommand,
		},
	}
}

func commandSpecByName(name string) (*commandSpec, bool) {
	for _, spec := range commandSpecs() {
		if spec.name == name {
			return spec, true
		}
	}

	return nil, false
}

func templateFromSpec(spec *commandSpec) Template {
	if spec == nil {
		return Template{}
	}

	stages := make([]verbose.StageDef, 0, len(spec.template.stages))
	for _, stage := range spec.template.stages {
		stages = append(stages, verbose.StageDef{
			Key:   stage.key,
			Label: localize.T(stage.labelKey),
			Slots: stage.slots,
		})
	}

	return Template{
		Mode:   localize.T(spec.template.modeKey),
		Stages: stages,
	}
}

func progressTemplate(modeKey string, stages ...stageTemplateSpec) templateSpec {
	return templateSpec{
		modeKey: modeKey,
		stages:  stages,
	}
}

func stageTemplate(key string, labelKey string, slots int) stageTemplateSpec {
	return stageTemplateSpec{
		key:      key,
		labelKey: labelKey,
		slots:    slots,
	}
}

func globalFlagSpecs() []flagSpec {
	return []flagSpec{
		{
			names:      []string{"--path", "--file", "-f"},
			takesValue: true,
			build: func() cli.Flag {
				return &cli.StringFlag{
					Name:    "path",
					Aliases: []string{"file", "f"},
					Usage:   localize.T("cli.flag.path.usage"),
					Sources: cli.EnvVars("INPUT_PATH", "INPUT_FILE"),
				}
			},
		},
		{
			names:      []string{"--api-url"},
			takesValue: true,
			build: func() cli.Flag {
				return &cli.StringFlag{
					Name:    "api-url",
					Usage:   localize.T("cli.flag.api_url.usage"),
					Value:   defaultLLMOptions().APIURL,
					Sources: cli.EnvVars("LLM_API_URL"),
				}
			},
		},
		{
			names:      []string{"--api-key"},
			takesValue: true,
			build: func() cli.Flag {
				return &cli.StringFlag{
					Name:    "api-key",
					Usage:   localize.T("cli.flag.api_key.usage"),
					Sources: cli.EnvVars("OPENAI_API_KEY"),
				}
			},
		},
		{
			names:      []string{"--proxy-url"},
			takesValue: true,
			build: func() cli.Flag {
				return &cli.StringFlag{
					Name:    "proxy-url",
					Usage:   localize.T("cli.flag.proxy_url.usage"),
					Sources: cli.EnvVars("LLM_PROXY_URL"),
				}
			},
		},
		{
			names: []string{"--no-proxy"},
			build: func() cli.Flag {
				return &cli.BoolFlag{
					Name:    "no-proxy",
					Usage:   localize.T("cli.flag.no_proxy.usage"),
					Sources: cli.EnvVars("LLM_NO_PROXY"),
				}
			},
		},
		{
			names:      []string{"--model"},
			takesValue: true,
			build: func() cli.Flag {
				return &cli.StringFlag{
					Name:    "model",
					Usage:   localize.T("cli.flag.model.usage"),
					Value:   defaultLLMOptions().Model,
					Sources: cli.EnvVars("LLM_MODEL"),
				}
			},
		},
		{
			names:      []string{"--retry", "-r"},
			takesValue: true,
			build: func() cli.Flag {
				return &cli.IntFlag{
					Name:    "retry",
					Aliases: []string{"r"},
					Usage:   localize.T("cli.flag.retry.usage"),
					Value:   defaultLLMOptions().RetryCount,
					Sources: cli.EnvVars("RETRY"),
				}
			},
		},
		{
			names: []string{"--raw"},
			build: func() cli.Flag {
				return &cli.BoolFlag{
					Name:    "raw",
					Usage:   localize.T("cli.flag.raw.usage"),
					Sources: cli.EnvVars("RAW"),
				}
			},
		},
		{
			names: []string{"--debug", "-d"},
			build: func() cli.Flag {
				return &cli.BoolFlag{
					Name:    "debug",
					Aliases: []string{"d"},
					Usage:   localize.T("cli.flag.debug.usage"),
					Sources: cli.EnvVars("DEBUG"),
				}
			},
		},
		{
			names: []string{"--verbose", "-v"},
			build: func() cli.Flag {
				return &cli.BoolFlag{
					Name:    "verbose",
					Aliases: []string{"v"},
					Usage:   localize.T("cli.flag.verbose.usage"),
					Sources: cli.EnvVars("VERBOSE"),
				}
			},
		},
		{
			names:      []string{"--lang"},
			takesValue: true,
			build: func() cli.Flag {
				return &cli.StringFlag{
					Name:    "lang",
					Usage:   localize.T("cli.flag.lang.usage"),
					Sources: cli.EnvVars("RAGCLI_LANG"),
				}
			},
		},
	}
}

func promptFlagConsumesValue(spec *commandSpec, token string) bool {
	for _, flag := range globalFlagSpecs() {
		if flagMatches(flag, token) && flag.takesValue {
			return true
		}
	}
	for _, flag := range spec.flagSpecs {
		if flagMatches(flag, token) && flag.takesValue {
			return true
		}
	}
	return false
}

func buildFlags(specs []flagSpec) []cli.Flag {
	flags := make([]cli.Flag, 0, len(specs))
	for _, spec := range specs {
		flags = append(flags, spec.build())
	}
	return flags
}

func flagMatches(spec flagSpec, token string) bool {
	for _, name := range spec.names {
		if token == name {
			return true
		}
	}
	return false
}

func bindCommonOptions(cmd *cli.Command) CommonOptions {
	return CommonOptions{
		InputPath: cmd.String("path"),
		Prompt:    bindPrompt(cmd),
		Raw:       cmd.Bool("raw"),
		Debug:     cmd.Bool("debug"),
		Verbose:   cmd.Bool("verbose"),
	}
}

func bindLLMOptions(cmd *cli.Command) LLMOptions {
	opts := defaultLLMOptions()
	opts.APIURL = cmd.String("api-url")
	opts.Model = cmd.String("model")
	opts.APIKey = cmd.String("api-key")
	opts.ProxyURL = cmd.String("proxy-url")
	opts.NoProxy = cmd.Bool("no-proxy")
	opts.RetryCount = cmd.Int("retry")
	return opts
}

func bindPrompt(cmd *cli.Command) string {
	return strings.TrimSpace(strings.Join(cmd.StringArgs("prompt"), " "))
}

func bindMapInvocation(cmd *cli.Command, spec *commandSpec) (commandInvocation, error) {
	inv, err := bindCommandBase(cmd, spec)
	if err != nil {
		return commandInvocation{}, err
	}

	inv.payload = normalizeMapOptions(mapmode.Options{
		ChunkLength:    cmd.Int("length"),
		LengthExplicit: cmd.IsSet("length"),
		Concurrency:    cmd.Int("concurrency"),
	})
	return inv, nil
}

func bindRAGInvocation(cmd *cli.Command, spec *commandSpec) (commandInvocation, error) {
	inv, err := bindCommandBase(cmd, spec)
	if err != nil {
		return commandInvocation{}, err
	}

	inv.LLM.EmbeddingModel = normalizeEmbeddingModel(cmd.String("embedding-model"))
	inv.payload = normalizeRAGOptions(rag.Options{
		TopK:           cmd.Int("rag-top-k"),
		FinalK:         cmd.Int("rag-final-k"),
		ChunkSize:      cmd.Int("rag-chunk-size"),
		ChunkOverlap:   cmd.Int("rag-chunk-overlap"),
		IndexTTL:       cmd.Duration("rag-index-ttl"),
		IndexDir:       cmd.String("rag-index-dir"),
		Rerank:         cmd.String("rag-rerank"),
		EmbeddingModel: inv.LLM.EmbeddingModel,
	})
	return inv, nil
}

func bindToolsInvocation(cmd *cli.Command, spec *commandSpec) (commandInvocation, error) {
	inv, err := bindCommandBase(cmd, spec)
	if err != nil {
		return commandInvocation{}, err
	}

	inv.LLM.EmbeddingModel = normalizeEmbeddingModel(cmd.String("embedding-model"))
	inv.payload = normalizeToolsOptions(toolsmode.Options{
		EnableRAG: cmd.Bool("rag"),
		RAG: rag.SearchOptions{
			TopK:           cmd.Int("rag-top-k"),
			ChunkSize:      cmd.Int("rag-chunk-size"),
			ChunkOverlap:   cmd.Int("rag-chunk-overlap"),
			IndexTTL:       cmd.Duration("rag-index-ttl"),
			IndexDir:       cmd.String("rag-index-dir"),
			Rerank:         cmd.String("rag-rerank"),
			EmbeddingModel: inv.LLM.EmbeddingModel,
		},
	})
	return inv, nil
}

func bindHybridInvocation(cmd *cli.Command, spec *commandSpec) (commandInvocation, error) {
	inv, err := bindCommandBase(cmd, spec)
	if err != nil {
		return commandInvocation{}, err
	}

	inv.LLM.EmbeddingModel = normalizeEmbeddingModel(cmd.String("embedding-model"))
	inv.payload = normalizeHybridOptions(hybrid.Options{
		Search: rag.SearchOptions{
			TopK:           cmd.Int("rag-top-k"),
			ChunkSize:      cmd.Int("rag-chunk-size"),
			ChunkOverlap:   cmd.Int("rag-chunk-overlap"),
			IndexTTL:       cmd.Duration("rag-index-ttl"),
			IndexDir:       cmd.String("rag-index-dir"),
			Rerank:         cmd.String("rag-rerank"),
			EmbeddingModel: inv.LLM.EmbeddingModel,
		},
	})
	return inv, nil
}

func bindCommandBase(cmd *cli.Command, spec *commandSpec) (commandInvocation, error) {
	inv := commandInvocation{
		spec:   spec,
		Common: bindCommonOptions(cmd),
		LLM:    bindLLMOptions(cmd),
	}

	if err := validatePrompt(inv.Common.Prompt); err != nil {
		return commandInvocation{}, err
	}

	return inv, nil
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

func normalizeMapOptions(options mapmode.Options) mapmode.Options {
	options.Concurrency = maxInt(options.Concurrency, 1)
	return options
}

func normalizeEmbeddingModel(value string) string {
	if strings.TrimSpace(value) == "" {
		return "text-embedding-3-small"
	}
	return value
}

func normalizeRAGOptions(options rag.Options) rag.Options {
	shared := normalizeRAGSearchOptions(rag.SearchOptions{
		TopK:           options.TopK,
		ChunkSize:      options.ChunkSize,
		ChunkOverlap:   options.ChunkOverlap,
		IndexTTL:       options.IndexTTL,
		IndexDir:       options.IndexDir,
		Rerank:         options.Rerank,
		EmbeddingModel: options.EmbeddingModel,
	})
	options.TopK = shared.TopK
	options.ChunkSize = shared.ChunkSize
	options.ChunkOverlap = shared.ChunkOverlap
	options.IndexTTL = shared.IndexTTL
	options.IndexDir = shared.IndexDir
	options.Rerank = shared.Rerank
	options.EmbeddingModel = shared.EmbeddingModel
	options.FinalK = maxInt(options.FinalK, 1)
	if options.FinalK > options.TopK {
		options.FinalK = options.TopK
	}
	return options
}

func normalizeRAGSearchOptions(options rag.SearchOptions) rag.SearchOptions {
	options.TopK = maxInt(options.TopK, 1)
	options.ChunkSize = maxInt(options.ChunkSize, 1000)
	options.ChunkOverlap = maxInt(options.ChunkOverlap, 0)
	if options.ChunkOverlap >= options.ChunkSize {
		options.ChunkOverlap = options.ChunkSize / 4
	}
	if strings.TrimSpace(options.IndexDir) == "" {
		options.IndexDir = defaultRAGIndexDir()
	}
	options.EmbeddingModel = normalizeEmbeddingModel(options.EmbeddingModel)
	options.Rerank = normalizeRAGRerank(options.Rerank)
	return options
}

func normalizeToolsOptions(options toolsmode.Options) toolsmode.Options {
	options.RAG = normalizeRAGSearchOptions(options.RAG)
	return options
}

func normalizeHybridOptions(options hybrid.Options) hybrid.Options {
	options.Search = normalizeRAGSearchOptions(options.Search)
	return options
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

func toolsRAGFlagSpec() flagSpec {
	return flagSpec{
		names: []string{"--rag"},
		build: func() cli.Flag {
			return &cli.BoolFlag{
				Name:    "rag",
				Usage:   localize.T("cli.flag.tools.rag.usage"),
				Sources: cli.EnvVars("TOOLS_RAG"),
			}
		},
	}
}

func ragFlagSpecs(includeFinalK bool) []flagSpec {
	specs := []flagSpec{
		{
			names:      []string{"--embedding-model"},
			takesValue: true,
			build: func() cli.Flag {
				return &cli.StringFlag{
					Name:    "embedding-model",
					Usage:   localize.T("cli.flag.rag.embedding_model.usage"),
					Value:   defaultLLMOptions().EmbeddingModel,
					Sources: cli.EnvVars("EMBEDDING_MODEL"),
				}
			},
		},
		{
			names:      []string{"--rag-top-k"},
			takesValue: true,
			build: func() cli.Flag {
				return &cli.IntFlag{
					Name:    "rag-top-k",
					Usage:   localize.T("cli.flag.rag.top_k.usage"),
					Value:   8,
					Sources: cli.EnvVars("RAG_TOP_K"),
				}
			},
		},
	}
	if includeFinalK {
		specs = append(specs, flagSpec{
			names:      []string{"--rag-final-k"},
			takesValue: true,
			build: func() cli.Flag {
				return &cli.IntFlag{
					Name:    "rag-final-k",
					Usage:   localize.T("cli.flag.rag.final_k.usage"),
					Value:   4,
					Sources: cli.EnvVars("RAG_FINAL_K"),
				}
			},
		})
	}
	specs = append(specs,
		flagSpec{
			names:      []string{"--rag-chunk-size"},
			takesValue: true,
			build: func() cli.Flag {
				return &cli.IntFlag{
					Name:    "rag-chunk-size",
					Usage:   localize.T("cli.flag.rag.chunk_size.usage"),
					Value:   1800,
					Sources: cli.EnvVars("RAG_CHUNK_SIZE"),
				}
			},
		},
		flagSpec{
			names:      []string{"--rag-chunk-overlap"},
			takesValue: true,
			build: func() cli.Flag {
				return &cli.IntFlag{
					Name:    "rag-chunk-overlap",
					Usage:   localize.T("cli.flag.rag.chunk_overlap.usage"),
					Value:   200,
					Sources: cli.EnvVars("RAG_CHUNK_OVERLAP"),
				}
			},
		},
		flagSpec{
			names:      []string{"--rag-index-ttl"},
			takesValue: true,
			build: func() cli.Flag {
				return &cli.DurationFlag{
					Name:    "rag-index-ttl",
					Usage:   localize.T("cli.flag.rag.index_ttl.usage"),
					Value:   24 * time.Hour,
					Sources: cli.EnvVars("RAG_INDEX_TTL"),
				}
			},
		},
		flagSpec{
			names:      []string{"--rag-index-dir"},
			takesValue: true,
			build: func() cli.Flag {
				return &cli.StringFlag{
					Name:    "rag-index-dir",
					Usage:   localize.T("cli.flag.rag.index_dir.usage"),
					Value:   defaultRAGIndexDir(),
					Sources: cli.EnvVars("RAG_INDEX_DIR"),
				}
			},
		},
		flagSpec{
			names:      []string{"--rag-rerank"},
			takesValue: true,
			build: func() cli.Flag {
				return &cli.StringFlag{
					Name:    "rag-rerank",
					Usage:   localize.T("cli.flag.rag.rerank.usage"),
					Value:   "heuristic",
					Sources: cli.EnvVars("RAG_RERANK"),
				}
			},
		},
	)
	return specs
}
