package app

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"time"

	mapmode "github.com/borro/ragcli/internal/app/map"
	"github.com/borro/ragcli/internal/app/rag"
	"github.com/borro/ragcli/internal/app/tools"
)

type usageError struct {
	message string
	usage   string
}

func (e usageError) Error() string {
	return e.message
}

func parseArgs(args []string) (Command, error) {
	root := flag.NewFlagSet("ragcli", flag.ContinueOnError)
	root.SetOutput(io.Discard)
	version := root.Bool("version", false, "show version")
	if err := root.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return Command{}, usageError{usage: rootUsage()}
		}
		return Command{}, usageError{message: err.Error(), usage: rootUsage()}
	}

	if *version && len(root.Args()) == 0 {
		return Command{Common: CommonOptions{Version: true}}, nil
	}
	if len(root.Args()) == 0 {
		return Command{}, usageError{message: "missing subcommand", usage: rootUsage()}
	}

	subcommand := root.Args()[0]
	subArgs := root.Args()[1:]

	switch subcommand {
	case "map":
		return parseMapArgs(subArgs)
	case "rag":
		return parseRAGArgs(subArgs)
	case "tools":
		return parseToolsArgs(subArgs)
	default:
		return Command{}, usageError{
			message: fmt.Sprintf("unknown subcommand: %s", subcommand),
			usage:   rootUsage(),
		}
	}
}

func parseMapArgs(args []string) (Command, error) {
	var cmd Command
	cmd.Name = "map"
	cmd.Map = mapmode.Options{
		ChunkLength: 10000,
		Concurrency: 1,
	}
	cmd.LLM = defaultLLMOptions()

	fs := flag.NewFlagSet("ragcli map", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	addSharedFlags(fs, &cmd.Common, &cmd.LLM)
	fs.IntVar(&cmd.Map.Concurrency, "c", getIntEnv("CONCURRENCY", 1), "concurrency")
	fs.IntVar(&cmd.Map.Concurrency, "concurrency", getIntEnv("CONCURRENCY", 1), "concurrency")
	fs.IntVar(&cmd.Map.ChunkLength, "l", getIntEnv("LENGTH", 10000), "chunk length")
	fs.IntVar(&cmd.Map.ChunkLength, "length", getIntEnv("LENGTH", 10000), "chunk length")

	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return Command{}, usageError{usage: mapUsage()}
		}
		return Command{}, usageError{message: err.Error(), usage: mapUsage()}
	}

	cmd.Map.Concurrency = maxInt(cmd.Map.Concurrency, 1)
	cmd.Map.ChunkLength = maxInt(cmd.Map.ChunkLength, 1000)
	cmd.Map.InputPath = cmd.Common.InputPath
	cmd.Common.Prompt = joinPrompt(fs.Args())
	if cmd.Common.Prompt == "" {
		return Command{}, usageError{message: "missing required argument: prompt", usage: mapUsage()}
	}
	return cmd, nil
}

func parseRAGArgs(args []string) (Command, error) {
	var cmd Command
	cmd.Name = "rag"
	cmd.LLM = defaultLLMOptions()
	cmd.RAG = rag.Options{
		TopK:         getIntEnv("RAG_TOP_K", 8),
		FinalK:       getIntEnv("RAG_FINAL_K", 4),
		ChunkSize:    getIntEnv("RAG_CHUNK_SIZE", 1800),
		ChunkOverlap: getIntEnv("RAG_CHUNK_OVERLAP", 200),
		IndexTTL:     parsePositiveDurationOrDefault(getEnv("RAG_INDEX_TTL", "24h"), "", 24*time.Hour),
		IndexDir:     strings.TrimSpace(getEnv("RAG_INDEX_DIR", defaultRAGIndexDir())),
		Rerank:       normalizeRAGRerank(getEnv("RAG_RERANK", "heuristic")),
	}

	fs := flag.NewFlagSet("ragcli rag", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	addSharedFlags(fs, &cmd.Common, &cmd.LLM)
	fs.StringVar(&cmd.LLM.EmbeddingModel, "embedding-model", getEnv("EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5"), "embedding model")
	fs.IntVar(&cmd.RAG.TopK, "rag-top-k", cmd.RAG.TopK, "top k")
	fs.IntVar(&cmd.RAG.FinalK, "rag-final-k", cmd.RAG.FinalK, "final k")
	fs.IntVar(&cmd.RAG.ChunkSize, "rag-chunk-size", cmd.RAG.ChunkSize, "chunk size")
	fs.IntVar(&cmd.RAG.ChunkOverlap, "rag-chunk-overlap", cmd.RAG.ChunkOverlap, "chunk overlap")
	fs.DurationVar(&cmd.RAG.IndexTTL, "rag-index-ttl", cmd.RAG.IndexTTL, "index ttl")
	fs.StringVar(&cmd.RAG.IndexDir, "rag-index-dir", cmd.RAG.IndexDir, "index dir")
	fs.StringVar(&cmd.RAG.Rerank, "rag-rerank", cmd.RAG.Rerank, "rerank")

	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return Command{}, usageError{usage: ragUsage()}
		}
		return Command{}, usageError{message: err.Error(), usage: ragUsage()}
	}

	if strings.TrimSpace(cmd.LLM.EmbeddingModel) == "" {
		cmd.LLM.EmbeddingModel = "text-embedding-3-small"
	}
	cmd.RAG.TopK = maxInt(cmd.RAG.TopK, 1)
	cmd.RAG.FinalK = maxInt(cmd.RAG.FinalK, 1)
	if cmd.RAG.FinalK > cmd.RAG.TopK {
		cmd.RAG.FinalK = cmd.RAG.TopK
	}
	cmd.RAG.ChunkSize = maxInt(cmd.RAG.ChunkSize, 1000)
	cmd.RAG.ChunkOverlap = maxInt(cmd.RAG.ChunkOverlap, 0)
	if cmd.RAG.ChunkOverlap >= cmd.RAG.ChunkSize {
		cmd.RAG.ChunkOverlap = cmd.RAG.ChunkSize / 4
	}
	if strings.TrimSpace(cmd.RAG.IndexDir) == "" {
		cmd.RAG.IndexDir = defaultRAGIndexDir()
	}
	cmd.RAG.Rerank = normalizeRAGRerank(cmd.RAG.Rerank)
	cmd.Common.Prompt = joinPrompt(fs.Args())
	if cmd.Common.Prompt == "" {
		return Command{}, usageError{message: "missing required argument: prompt", usage: ragUsage()}
	}
	return cmd, nil
}

func parseToolsArgs(args []string) (Command, error) {
	var cmd Command
	cmd.Name = "tools"
	cmd.LLM = defaultLLMOptions()
	cmd.Tools = tools.Options{}

	fs := flag.NewFlagSet("ragcli tools", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	addSharedFlags(fs, &cmd.Common, &cmd.LLM)

	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return Command{}, usageError{usage: toolsUsage()}
		}
		return Command{}, usageError{message: err.Error(), usage: toolsUsage()}
	}

	cmd.Common.Prompt = joinPrompt(fs.Args())
	if cmd.Common.Prompt == "" {
		return Command{}, usageError{message: "missing required argument: prompt", usage: toolsUsage()}
	}
	return cmd, nil
}

func addSharedFlags(fs *flag.FlagSet, common *CommonOptions, llm *LLMOptions) {
	fs.StringVar(&common.InputPath, "f", getEnv("INPUT_FILE", ""), "input file")
	fs.StringVar(&common.InputPath, "file", getEnv("INPUT_FILE", ""), "input file")
	fs.BoolVar(&common.Verbose, "v", getBoolEnv("VERBOSE", false), "verbose")
	fs.BoolVar(&common.Verbose, "verbose", getBoolEnv("VERBOSE", false), "verbose")
	fs.StringVar(&llm.APIURL, "api-url", getEnv("LLM_API_URL", "http://localhost:1234/v1"), "api url")
	fs.StringVar(&llm.Model, "model", getEnv("LLM_MODEL", "local-model"), "model")
	fs.StringVar(&llm.APIKey, "api-key", getEnv("OPENAI_API_KEY", ""), "api key")
	fs.IntVar(&llm.RetryCount, "r", getIntEnv("RETRY", 3), "retry")
}

func defaultLLMOptions() LLMOptions {
	return LLMOptions{
		APIURL:         getEnv("LLM_API_URL", "http://localhost:1234/v1"),
		Model:          getEnv("LLM_MODEL", "local-model"),
		EmbeddingModel: getEnv("EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5"),
		APIKey:         getEnv("OPENAI_API_KEY", ""),
		RetryCount:     getIntEnv("RETRY", 3),
	}
}

func rootUsage() string {
	return "Usage:\n  ragcli map [flags] <prompt>\n  ragcli rag [flags] <prompt>\n  ragcli tools [flags] <prompt>\n  ragcli --version"
}

func mapUsage() string {
	return "Usage:\n  ragcli map [flags] <prompt>"
}

func ragUsage() string {
	return "Usage:\n  ragcli rag [flags] <prompt>"
}

func toolsUsage() string {
	return "Usage:\n  ragcli tools [flags] <prompt>"
}

func joinPrompt(args []string) string {
	return strings.TrimSpace(strings.Join(args, " "))
}

func getEnv(key string, defaultValue string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return defaultValue
}

func getIntEnv(key string, defaultValue int) int {
	if v := os.Getenv(key); v != "" {
		result, err := strconv.Atoi(strings.TrimSpace(v))
		if err == nil {
			return result
		}
	}
	return defaultValue
}

func getBoolEnv(key string, defaultValue bool) bool {
	if v := os.Getenv(key); v != "" {
		switch strings.ToLower(v) {
		case "true", "1", "yes":
			return true
		case "false", "0", "no":
			return false
		}
	}
	return defaultValue
}

func parsePositiveDurationOrDefault(primary string, fallback string, defaultValue time.Duration) time.Duration {
	for _, candidate := range []string{primary, fallback} {
		if strings.TrimSpace(candidate) == "" {
			continue
		}
		duration, err := time.ParseDuration(candidate)
		if err == nil && duration > 0 {
			return duration
		}
	}
	return defaultValue
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
