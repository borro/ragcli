package config

import (
	"flag"
	"fmt"
	"log/slog"
	"os"
	"strconv"
	"strings"
	"time"
)

// Config содержит все настройки приложения
type Config struct {
	InputPath       string        // Путь к входному файлу (пусто = stdin)
	Mode            string        // Режим работы: "map-reduce" или "tools"
	APIURL          string        // URL API LLM
	Model           string        // Имя модели
	EmbeddingModel  string        // Имя embedding модели для mode=rag
	APIKey          string        // Ключ API (опционально)
	Concurrency     int           // Кол-во параллельных запросов для map-reduce
	ChunkLength     int           // Максимальный размер чанка в байтах
	RetryCount      int           // Количество повторных попыток
	RAGTopK         int           // Количество кандидатов после retrieval
	RAGFinalK       int           // Количество чанков в финальном контексте
	RAGChunkSize    int           // Размер retrieval-чанка в байтах
	RAGChunkOverlap int           // Перекрытие retrieval-чанков в байтах
	RAGIndexTTL     time.Duration // TTL локального индекса
	RAGIndexDir     string        // Базовая директория временного индекса
	RAGRerank       string        // off|heuristic|model
	Verbose         bool          // Включить debug логирование
	Version         bool          // Показать версию и выйти
	RequestTimeout  time.Duration // Таймаут HTTP запроса по умолчанию
	DialTimeout     time.Duration // Таймаут установления соединения
	TLSTimeout      time.Duration // Таймаут TLS рукопожатия
}

// LoadWithFlags загружает конфигурацию из env и CLI флагов.
// args - это массив командных аргументов (обычно os.Args[1:]).
// Возвращает конфигурацию, список позиционных аргументов (после всех флагов) и ошибку.
func LoadWithFlags(args []string) (*Config, []string, error) {
	// Создаём флаговый набор с кастомным использованием (чтобы не выходить при -h)
	fs := flag.NewFlagSet("ragcli", flag.ContinueOnError)

	// Флаги
	file := fs.String("f", getEnv("INPUT_FILE", ""), "путь к входному файлу")
	fileLong := fs.String("file", *file, "путь к входному файлу (alias for -f)")
	mode := fs.String("mode", getEnv("MODE", "map-reduce"), "режим работы: map-reduce или tools")
	apiURL := fs.String("api-url", getEnv("LLM_API_URL", "http://localhost:1234/v1"), "URL API LLM")
	model := fs.String("model", getEnv("LLM_MODEL", "local-model"), "имя модели LLM")
	embeddingModel := fs.String("embedding-model", getEnv("EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5"), "имя embedding модели для mode=rag")
	apiKey := fs.String("api-key", getEnv("OPENAI_API_KEY", ""), "ключ API (опционально)")
	concurrency := fs.Int("c", getIntEnv("CONCURRENCY", 1), "количество параллельных запросов для map-reduce")
	concurrencyLong := fs.Int("concurrency", *concurrency, "количество параллельных запросов для map-reduce (alias for -c)")
	chunkLength := fs.Int("l", getIntEnv("LENGTH", 10000), "максимальный размер чанка в байтах")
	lengthLong := fs.Int("length", *chunkLength, "максимальный размер чанка в байтах (alias for -l)")
	retryCount := fs.Int("r", getIntEnv("RETRY", 3), "количество повторных попыток при ошибках LLM")
	ragTopK := fs.Int("rag-top-k", getIntEnv("RAG_TOP_K", 8), "количество кандидатов после retrieval в mode=rag")
	ragFinalK := fs.Int("rag-final-k", getIntEnv("RAG_FINAL_K", 4), "количество чанков в финальном контексте mode=rag")
	ragChunkSize := fs.Int("rag-chunk-size", getIntEnv("RAG_CHUNK_SIZE", 1800), "размер retrieval чанка в байтах")
	ragChunkOverlap := fs.Int("rag-chunk-overlap", getIntEnv("RAG_CHUNK_OVERLAP", 200), "перекрытие retrieval чанков в байтах")
	ragIndexTTL := fs.String("rag-index-ttl", getEnv("RAG_INDEX_TTL", "24h"), "ttl локального индекса mode=rag")
	ragIndexDir := fs.String("rag-index-dir", getEnv("RAG_INDEX_DIR", defaultRAGIndexDir()), "директория локального индекса mode=rag")
	ragRerank := fs.String("rag-rerank", getEnv("RAG_RERANK", "heuristic"), "rerank режим для mode=rag: off|heuristic|model")
	verbose := fs.Bool("v", getBoolEnv("VERBOSE", false), "debug логирование")
	verboseLong := fs.Bool("verbose", *verbose, "debug логирование (alias for -v)")
	version := fs.Bool("version", false, "показать версию бинаря и выйти")

	// Parse с оставлением остатка аргументов для positional args
	if err := fs.Parse(args); err != nil {
		return nil, nil, fmt.Errorf("failed to parse flags: %w", err)
	}

	// Debug логирование для диагностики позиционных аргументов
	slog.Debug("после парсинга флагов", "все_аргументы_перед_парсингом", args, "positional_args_после_парсинга", fs.Args(), "count_pos_args", len(fs.Args()))

	// Получаем позиционные аргументы (после всех флагов)
	positionalArgs := fs.Args()

	// Если есть несколько позиционных аргументов, объединяем их в один prompt через пробел
	// Это позволяет использовать вопрос без кавычек: ./ragcli вопрос с пробелами
	if len(positionalArgs) > 1 {
		slog.Debug("объединение нескольких позиционных аргументов в prompt", "original_count", len(positionalArgs), "combined_prompt", strings.Join(positionalArgs, " "))
		positionalArgs = []string{strings.Join(positionalArgs, " ")}
	}

	cfg := &Config{}

	// Входной файл
	cfg.InputPath = chooseString(*file, *fileLong, getEnv("INPUT_FILE", ""))

	// Режим работы
	modeVal := *mode
	if modeVal == "" {
		modeVal = "map-reduce"
	}
	cfg.Mode = normalizeMode(modeVal)

	// API URL
	cfg.APIURL = *apiURL

	// Модель
	cfg.Model = *model
	cfg.EmbeddingModel = strings.TrimSpace(*embeddingModel)
	if cfg.EmbeddingModel == "" {
		cfg.EmbeddingModel = "text-embedding-3-small"
	}

	// API Key
	cfg.APIKey = *apiKey

	// Конкурентность
	cfg.Concurrency = chooseIntAtLeast(*concurrency, *concurrencyLong, getIntEnv("CONCURRENCY", 1), 1)
	if cfg.Concurrency < 1 {
		cfg.Concurrency = 1
	}

	// Длина чанка
	cfg.ChunkLength = chooseIntAtLeast(*chunkLength, *lengthLong, getIntEnv("LENGTH", 10000), 1000)
	if cfg.ChunkLength < 1000 {
		cfg.ChunkLength = 1000 // Минимум
	}

	// Retry count
	if *retryCount >= 1 {
		cfg.RetryCount = *retryCount
	} else {
		cfg.RetryCount = getIntEnv("RETRY", 3)
	}
	if cfg.RetryCount < 1 {
		cfg.RetryCount = 1
	}

	cfg.RAGTopK = chooseIntAtLeast(*ragTopK, getIntEnv("RAG_TOP_K", 8), 8, 1)
	cfg.RAGFinalK = chooseIntAtLeast(*ragFinalK, getIntEnv("RAG_FINAL_K", 4), 4, 1)
	if cfg.RAGFinalK > cfg.RAGTopK {
		cfg.RAGFinalK = cfg.RAGTopK
	}
	cfg.RAGChunkSize = chooseIntAtLeast(*ragChunkSize, getIntEnv("RAG_CHUNK_SIZE", 1800), 1800, 1000)
	cfg.RAGChunkOverlap = chooseIntAtLeast(*ragChunkOverlap, getIntEnv("RAG_CHUNK_OVERLAP", 200), 200, 0)
	if cfg.RAGChunkOverlap >= cfg.RAGChunkSize {
		cfg.RAGChunkOverlap = cfg.RAGChunkSize / 4
	}
	cfg.RAGIndexTTL = parsePositiveDurationOrDefault(*ragIndexTTL, getEnv("RAG_INDEX_TTL", "24h"), 24*time.Hour)
	cfg.RAGIndexDir = strings.TrimSpace(*ragIndexDir)
	if cfg.RAGIndexDir == "" {
		cfg.RAGIndexDir = defaultRAGIndexDir()
	}
	cfg.RAGRerank = normalizeRAGRerank(*ragRerank)

	// Verbose
	if *verbose || *verboseLong {
		cfg.Verbose = true
	} else {
		cfg.Verbose = getBoolEnv("VERBOSE", false)
	}

	cfg.Version = *version

	// Таймауты HTTP клиента
	cfg.RequestTimeout = getTimeDurationEnv("HTTP_REQUEST_TIMEOUT", 10*time.Minute)
	cfg.DialTimeout = getTimeDurationEnv("HTTP_DIAL_TIMEOUT", 30*time.Second)
	cfg.TLSTimeout = getTimeDurationEnv("HTTP_TLS_TIMEOUT", 30*time.Second)

	// Устанавливаем уровень логирования
	if cfg.Verbose {
		ConfigureLogger(true)
	} else {
		ConfigureLogger(false)
	}

	return cfg, positionalArgs, nil
}

// getEnv получает переменную окружения с дефолтным значением
func getEnv(key string, defaultValue string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return defaultValue
}

// getIntEnv получает целочисленную переменную окружения
func getIntEnv(key string, defaultValue int) int {
	if v := os.Getenv(key); v != "" {
		result, err := strconv.Atoi(strings.TrimSpace(v))
		if err == nil {
			return result
		}
	}
	return defaultValue
}

func chooseString(primary string, secondary string, fallback string) string {
	if primary != "" {
		return primary
	}
	if secondary != "" {
		return secondary
	}
	return fallback
}

func chooseIntAtLeast(primary int, secondary int, fallback int, min int) int {
	if primary >= min {
		return primary
	}
	if secondary >= min {
		return secondary
	}
	return fallback
}

func normalizeMode(mode string) string {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "", "map-reduce":
		return "map-reduce"
	case "tools":
		return "tools"
	case "rag":
		return "rag"
	default:
		return strings.ToLower(strings.TrimSpace(mode))
	}
}

// getBoolEnv получает булеву переменную окружения
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

// getTimeDurationEnv получает duration из переменной окружения
func getTimeDurationEnv(key string, defaultValue time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" {
		duration, err := time.ParseDuration(v)
		if err == nil && duration > 0 {
			return duration
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
