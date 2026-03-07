package main

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/joho/godotenv"

	"github.com/borro/ragcli/internal/config"
	"github.com/borro/ragcli/internal/llm"
	"github.com/borro/ragcli/internal/processor"
)

var version = "dev"

func main() {
	os.Exit(run(os.Args[1:], os.Stdout))
}

func run(args []string, stdout io.Writer) int {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Загружаем переменные окружения из .env файла
	if err := godotenv.Load(); err != nil {
		config.Log.Warn("failed to load .env file, using system environment variables", "error", err)
	}

	// Обработка сигналов (Ctrl+C)
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigChan
		cancel()
	}()

	// Загружаем конфигурацию с CLI флагами и позиционными аргументами
	cfg, positionalArgs, err := config.LoadWithFlags(args)
	if err != nil {
		config.Log.Error("failed to load config", "error", err)
		return 1
	}

	if cfg.Version {
		if err := printVersion(stdout); err != nil {
			config.Log.Error("failed to print version", "error", err)
			return 1
		}
		return 0
	}

	// Проверяем наличие промпта (позиционного аргумента)
	if len(positionalArgs) == 0 {
		config.Log.Error("missing required argument: prompt (вопрос пользователя)")
		return 1
	}
	prompt := positionalArgs[0]

	if cfg.Verbose {
		printConfigDebug(cfg, prompt)
	}

	// Читаем input (file или stdin)
	inputReader, tempFile, err := getInputReader(cfg.InputPath)
	if err != nil {
		config.Log.Error("failed to get input reader", "error", err)
		return 1
	}
	defer func() {
		if tempFile != "" {
			if err := os.Remove(tempFile); err != nil {
				config.Log.Warn("failed to remove temp file", "error", err)
			}
		}
	}()

	// Создаем клиент LLM с настройками таймаутов из конфига
	client := llm.NewClient(cfg.APIURL, cfg.Model, cfg.APIKey, cfg.RetryCount)

	var result string

	switch strings.ToLower(cfg.Mode) {
	case "map-reduce":
		result, err = processor.RunSRMR(ctx, client, inputReader, cfg, prompt)
	case "rag":
		// Передаём путь к файлу (или tempFile для stdin режима)
		filePath := cfg.InputPath
		if filePath == "" && tempFile != "" {
			filePath = tempFile
		}
		result, err = processor.RunRAG(ctx, client, filePath, prompt)
	default:
		err = fmt.Errorf("unknown mode: %s", cfg.Mode)
	}

	if err != nil {
		config.Log.Error("processing failed", "error", err)
		return 1
	}

	if _, err := fmt.Fprintln(stdout, result); err != nil {
		config.Log.Error("failed to write result", "error", err)
		return 1
	}

	return 0
}

func printVersion(stdout io.Writer) error {
	_, err := fmt.Fprintln(stdout, version)
	return err
}

// getInputReader открывает файл или читает stdin во временный файл
func getInputReader(inputPath string) (io.Reader, string, error) {
	// Проверяем, существует ли файл
	if inputPath != "" {
		if _, err := os.Stat(inputPath); os.IsNotExist(err) {
			return nil, "", fmt.Errorf("file does not exist: %s", inputPath)
		}

		f, err := os.Open(inputPath)
		if err != nil {
			return nil, "", fmt.Errorf("failed to open file: %w", err)
		}
		return f, "", nil
	}

	// Читаем из stdin во временный файл
	tmpFile, err := os.CreateTemp("", "ragcli-stdin-*.txt")
	if err != nil {
		return nil, "", fmt.Errorf("failed to create temp file: %w", err)
	}
	defer func() {
		if err := tmpFile.Close(); err != nil {
			config.Log.Error("failed to close temp file", "error", err)
		}
	}()

	reader := bufio.NewReader(os.Stdin)
	for {
		line, err := reader.ReadString('\n')
		if _, writeErr := tmpFile.WriteString(line); writeErr != nil {
			return nil, "", fmt.Errorf("failed to write to temp file: %w", writeErr)
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, "", fmt.Errorf("failed to read stdin: %w", err)
		}
	}

	// Сбрасываем буфер и возвращаемся в начало файла
	if err := tmpFile.Sync(); err != nil {
		return nil, "", fmt.Errorf("failed to sync temp file: %w", err)
	}
	f, err := os.Open(tmpFile.Name())
	if err != nil {
		return nil, "", fmt.Errorf("failed to reopen temp file: %w", err)
	}
	return f, tmpFile.Name(), nil
}

// printConfigDebug выводит все параметры конфигурации в debug режиме
func printConfigDebug(cfg *config.Config, prompt string) {
	config.Log.Debug("application configuration",
		"input_path", cfg.InputPath,
		"mode", cfg.Mode,
		"api_url", cfg.APIURL,
		"model", cfg.Model,
		"concurrency", cfg.Concurrency,
		"chunk_length", cfg.ChunkLength,
		"retry_count", cfg.RetryCount,
		"request_timeout", cfg.RequestTimeout,
		"dial_timeout", cfg.DialTimeout,
		"tls_timeout", cfg.TLSTimeout,
	)

	config.Log.Debug("prompt (вопрос пользователя)", "prompt", prompt)
}
