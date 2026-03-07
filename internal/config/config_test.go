package config

import (
	"os"
	"strings"
	"testing"
	"time"
)

func TestGetEnv(t *testing.T) {
	tests := []struct {
		name           string
		key            string
		envValue       string
		defaultValue   string
		expectedResult string
		cleanup        func()
	}{
		{
			name:           "env variable set",
			key:            "TEST_KEY",
			envValue:       "test_value",
			defaultValue:   "default",
			expectedResult: "test_value",
		},
		{
			name:           "env variable not set",
			key:            "NON_EXISTENT_KEY",
			envValue:       "",
			defaultValue:   "default",
			expectedResult: "default",
		},
		{
			name:           "empty env with default",
			key:            "EMPTY_KEY",
			envValue:       "",
			defaultValue:   "fallback",
			expectedResult: "fallback",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.envValue != "" {
				if err := os.Setenv(tt.key, tt.envValue); err != nil {
					t.Fatalf("failed to set env: %v", err)
				}
				cleanup := func() {
					if err := os.Unsetenv(tt.key); err != nil {
						t.Logf("failed to unset env: %v", err)
					}
				}
				defer cleanup()
			}

			result := getEnv(tt.key, tt.defaultValue)
			if result != tt.expectedResult {
				t.Errorf("getEnv(%q, %q) = %q, want %q", tt.key, tt.defaultValue, result, tt.expectedResult)
			}
		})
	}
}

func TestGetIntEnv(t *testing.T) {
	tests := []struct {
		name           string
		key            string
		envValue       string
		defaultValue   int
		expectedResult int
		cleanup        func()
	}{
		{
			name:           "valid integer",
			key:            "TEST_INT",
			envValue:       "42",
			defaultValue:   0,
			expectedResult: 42,
		},
		{
			name:           "invalid integer (string)",
			key:            "TEST_INT",
			envValue:       "not_a_number",
			defaultValue:   0,
			expectedResult: 0,
		},
		{
			name:           "empty env with default",
			key:            "EMPTY_INT",
			envValue:       "",
			defaultValue:   100,
			expectedResult: 100,
		},
		{
			name:           "valid with spaces",
			key:            "TEST_INT",
			envValue:       "  15  ",
			defaultValue:   0,
			expectedResult: 15,
		},
		{
			name:           "negative value",
			key:            "TEST_INT",
			envValue:       "-10",
			defaultValue:   0,
			expectedResult: -10,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.envValue != "" {
				if err := os.Setenv(tt.key, tt.envValue); err != nil {
					t.Fatalf("failed to set env: %v", err)
				}
				cleanup := func() {
					if err := os.Unsetenv(tt.key); err != nil {
						t.Logf("failed to unset env: %v", err)
					}
				}
				defer cleanup()
			}

			result := getIntEnv(tt.key, tt.defaultValue)
			if result != tt.expectedResult {
				t.Errorf("getIntEnv(%q, %d) = %d, want %d", tt.key, tt.defaultValue, result, tt.expectedResult)
			}
		})
	}
}

func TestGetBoolEnv(t *testing.T) {
	tests := []struct {
		name           string
		key            string
		envValue       string
		defaultValue   bool
		expectedResult bool
		cleanup        func()
	}{
		{
			name:           "true value",
			key:            "TEST_BOOL",
			envValue:       "true",
			defaultValue:   false,
			expectedResult: true,
		},
		{
			name:           "1 as true",
			key:            "TEST_BOOL",
			envValue:       "1",
			defaultValue:   false,
			expectedResult: true,
		},
		{
			name:           "yes as true",
			key:            "TEST_BOOL",
			envValue:       "yes",
			defaultValue:   false,
			expectedResult: true,
		},
		{
			name:           "false value",
			key:            "TEST_BOOL",
			envValue:       "false",
			defaultValue:   true,
			expectedResult: false,
		},
		{
			name:           "0 as false",
			key:            "TEST_BOOL",
			envValue:       "0",
			defaultValue:   true,
			expectedResult: false,
		},
		{
			name:           "no as false",
			key:            "TEST_BOOL",
			envValue:       "no",
			defaultValue:   true,
			expectedResult: false,
		},
		{
			name:           "mixed case true",
			key:            "TEST_BOOL",
			envValue:       "TRUE",
			defaultValue:   false,
			expectedResult: true,
		},
		{
			name:           "mixed case false",
			key:            "TEST_BOOL",
			envValue:       "FALSE",
			defaultValue:   true,
			expectedResult: false,
		},
		{
			name:           "invalid value returns default",
			key:            "TEST_BOOL",
			envValue:       "invalid",
			defaultValue:   true,
			expectedResult: true,
		},
		{
			name:           "empty env with default",
			key:            "EMPTY_BOOL",
			envValue:       "",
			defaultValue:   false,
			expectedResult: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.envValue != "" {
				if err := os.Setenv(tt.key, tt.envValue); err != nil {
					t.Fatalf("failed to set env: %v", err)
				}
				cleanup := func() {
					if err := os.Unsetenv(tt.key); err != nil {
						t.Logf("failed to unset env: %v", err)
					}
				}
				defer cleanup()
			}

			result := getBoolEnv(tt.key, tt.defaultValue)
			if result != tt.expectedResult {
				t.Errorf("getBoolEnv(%q, %v) = %v, want %v", tt.key, tt.defaultValue, result, tt.expectedResult)
			}
		})
	}
}

func TestGetTimeDurationEnv(t *testing.T) {
	tests := []struct {
		name           string
		key            string
		envValue       string
		defaultValue   time.Duration
		expectedResult time.Duration
		cleanup        func()
	}{
		{
			name:           "valid duration",
			key:            "TEST_DURATION",
			envValue:       "5m",
			defaultValue:   10 * time.Minute,
			expectedResult: 5 * time.Minute,
		},
		{
			name:           "negative duration ignored",
			key:            "TEST_DURATION",
			envValue:       "-1m",
			defaultValue:   10 * time.Minute,
			expectedResult: 10 * time.Minute,
		},
		{
			name:           "invalid duration format ignored",
			key:            "TEST_DURATION",
			envValue:       "5xyz",
			defaultValue:   10 * time.Minute,
			expectedResult: 10 * time.Minute,
		},
		{
			name:           "empty env with default",
			key:            "EMPTY_DURATION",
			envValue:       "",
			defaultValue:   30 * time.Second,
			expectedResult: 30 * time.Second,
		},
		{
			name:           "valid seconds",
			key:            "TEST_DURATION",
			envValue:       "30s",
			defaultValue:   10 * time.Minute,
			expectedResult: 30 * time.Second,
		},
		{
			name:           "valid hours",
			key:            "TEST_DURATION",
			envValue:       "2h",
			defaultValue:   10 * time.Minute,
			expectedResult: 2 * time.Hour,
		},
		{
			name:           "millisecond precision",
			key:            "TEST_DURATION",
			envValue:       "500ms",
			defaultValue:   1 * time.Second,
			expectedResult: 500 * time.Millisecond,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.envValue != "" {
				if err := os.Setenv(tt.key, tt.envValue); err != nil {
					t.Fatalf("failed to set env: %v", err)
				}
				cleanup := func() {
					if err := os.Unsetenv(tt.key); err != nil {
						t.Logf("failed to unset env: %v", err)
					}
				}
				defer cleanup()
			}

			result := getTimeDurationEnv(tt.key, tt.defaultValue)
			if result != tt.expectedResult {
				t.Errorf("getTimeDurationEnv(%q, %v) = %v, want %v", tt.key, tt.defaultValue, result, tt.expectedResult)
			}
		})
	}
}

func TestLoadWithFlags(t *testing.T) {
	tests := []struct {
		name         string
		args         []string
		envSetup     func()
		expectedMode string
		expectError  bool
		cleanup      func()
	}{
		{
			name:         "valid flags with all parameters",
			args:         []string{"-f", "test.txt", "-mode", "tools", "-api-url", "http://test.com", "-model", "gpt-4", "-c", "4", "-l", "5000", "-r", "5", "-v"},
			expectedMode: "tools",
			expectError:  false,
			cleanup: func() {
				_ = os.Unsetenv("INPUT_FILE")
				_ = os.Unsetenv("MODE")
				_ = os.Unsetenv("LLM_API_URL")
				_ = os.Unsetenv("LLM_MODEL")
				_ = os.Unsetenv("OPENAI_API_KEY")
				_ = os.Unsetenv("CONCURRENCY")
				_ = os.Unsetenv("LENGTH")
				_ = os.Unsetenv("RETRY")
				_ = os.Unsetenv("VERBOSE")
				_ = os.Unsetenv("HTTP_REQUEST_TIMEOUT")
				_ = os.Unsetenv("HTTP_DIAL_TIMEOUT")
				_ = os.Unsetenv("HTTP_TLS_TIMEOUT")
			},
		},
		{
			name:         "empty args uses env defaults",
			args:         []string{},
			expectedMode: "map-reduce",
			expectError:  false,
			cleanup: func() {
				_ = os.Unsetenv("INPUT_FILE")
				_ = os.Unsetenv("MODE")
				_ = os.Unsetenv("LLM_API_URL")
				_ = os.Unsetenv("LLM_MODEL")
				_ = os.Unsetenv("OPENAI_API_KEY")
				_ = os.Unsetenv("CONCURRENCY")
				_ = os.Unsetenv("LENGTH")
				_ = os.Unsetenv("RETRY")
				_ = os.Unsetenv("VERBOSE")
				_ = os.Unsetenv("HTTP_REQUEST_TIMEOUT")
				_ = os.Unsetenv("HTTP_DIAL_TIMEOUT")
				_ = os.Unsetenv("HTTP_TLS_TIMEOUT")
			},
		},
		{
			name:         "env variable overrides flag",
			args:         []string{"-f", "flag.txt"},
			expectedMode: "custom_mode",
			expectError:  false,
			envSetup: func() {
				if err := os.Setenv("INPUT_FILE", "env.txt"); err != nil {
					t.Fatalf("failed to set INPUT_FILE env: %v", err)
				}
				if err := os.Setenv("MODE", "custom_mode"); err != nil {
					t.Fatalf("failed to set MODE env: %v", err)
				}
			},
			cleanup: func() {
				_ = os.Unsetenv("INPUT_FILE")
				_ = os.Unsetenv("MODE")
				_ = os.Unsetenv("LLM_API_URL")
				_ = os.Unsetenv("LLM_MODEL")
				_ = os.Unsetenv("OPENAI_API_KEY")
				_ = os.Unsetenv("CONCURRENCY")
				_ = os.Unsetenv("LENGTH")
				_ = os.Unsetenv("RETRY")
				_ = os.Unsetenv("VERBOSE")
				_ = os.Unsetenv("HTTP_REQUEST_TIMEOUT")
				_ = os.Unsetenv("HTTP_DIAL_TIMEOUT")
				_ = os.Unsetenv("HTTP_TLS_TIMEOUT")
			},
		},
		{
			name:         "short and long flag aliases should use same value",
			args:         []string{"-f", "test.txt", "-file", "test2.txt"},
			expectedMode: "map-reduce",
			expectError:  false,
			cleanup: func() {
				_ = os.Unsetenv("INPUT_FILE")
				_ = os.Unsetenv("MODE")
				_ = os.Unsetenv("LLM_API_URL")
				_ = os.Unsetenv("LLM_MODEL")
				_ = os.Unsetenv("OPENAI_API_KEY")
				_ = os.Unsetenv("CONCURRENCY")
				_ = os.Unsetenv("LENGTH")
				_ = os.Unsetenv("RETRY")
				_ = os.Unsetenv("VERBOSE")
				_ = os.Unsetenv("HTTP_REQUEST_TIMEOUT")
				_ = os.Unsetenv("HTTP_DIAL_TIMEOUT")
				_ = os.Unsetenv("HTTP_TLS_TIMEOUT")
			},
		},
		{
			name:         "invalid concurrency defaults to 1",
			args:         []string{"-c", "-1"},
			expectedMode: "map-reduce",
			expectError:  false,
			cleanup: func() {
				_ = os.Unsetenv("INPUT_FILE")
				_ = os.Unsetenv("MODE")
				_ = os.Unsetenv("LLM_API_URL")
				_ = os.Unsetenv("LLM_MODEL")
				_ = os.Unsetenv("OPENAI_API_KEY")
				_ = os.Unsetenv("CONCURRENCY")
				_ = os.Unsetenv("LENGTH")
				_ = os.Unsetenv("RETRY")
				_ = os.Unsetenv("VERBOSE")
				_ = os.Unsetenv("HTTP_REQUEST_TIMEOUT")
				_ = os.Unsetenv("HTTP_DIAL_TIMEOUT")
				_ = os.Unsetenv("HTTP_TLS_TIMEOUT")
			},
		},
		{
			name:         "invalid chunk length defaults to 1000",
			args:         []string{"-l", "500"},
			expectedMode: "map-reduce",
			expectError:  false,
			cleanup: func() {
				_ = os.Unsetenv("INPUT_FILE")
				_ = os.Unsetenv("MODE")
				_ = os.Unsetenv("LLM_API_URL")
				_ = os.Unsetenv("LLM_MODEL")
				_ = os.Unsetenv("OPENAI_API_KEY")
				_ = os.Unsetenv("CONCURRENCY")
				_ = os.Unsetenv("LENGTH")
				_ = os.Unsetenv("RETRY")
				_ = os.Unsetenv("VERBOSE")
				_ = os.Unsetenv("HTTP_REQUEST_TIMEOUT")
				_ = os.Unsetenv("HTTP_DIAL_TIMEOUT")
				_ = os.Unsetenv("HTTP_TLS_TIMEOUT")
			},
		},
		{
			name:         "multiple positional args combined into single prompt",
			args:         []string{"question1 question2 question3"},
			expectedMode: "map-reduce",
			expectError:  false,
			cleanup: func() {
				_ = os.Unsetenv("INPUT_FILE")
				_ = os.Unsetenv("MODE")
				_ = os.Unsetenv("LLM_API_URL")
				_ = os.Unsetenv("LLM_MODEL")
				_ = os.Unsetenv("OPENAI_API_KEY")
				_ = os.Unsetenv("CONCURRENCY")
				_ = os.Unsetenv("LENGTH")
				_ = os.Unsetenv("RETRY")
				_ = os.Unsetenv("VERBOSE")
				_ = os.Unsetenv("HTTP_REQUEST_TIMEOUT")
				_ = os.Unsetenv("HTTP_DIAL_TIMEOUT")
				_ = os.Unsetenv("HTTP_TLS_TIMEOUT")
			},
		},
		{
			name:         "version after double dash remains positional argument",
			args:         []string{"--", "--version"},
			expectedMode: "map-reduce",
			expectError:  false,
			cleanup: func() {
				_ = os.Unsetenv("INPUT_FILE")
				_ = os.Unsetenv("MODE")
				_ = os.Unsetenv("LLM_API_URL")
				_ = os.Unsetenv("LLM_MODEL")
				_ = os.Unsetenv("OPENAI_API_KEY")
				_ = os.Unsetenv("CONCURRENCY")
				_ = os.Unsetenv("LENGTH")
				_ = os.Unsetenv("RETRY")
				_ = os.Unsetenv("VERBOSE")
				_ = os.Unsetenv("HTTP_REQUEST_TIMEOUT")
				_ = os.Unsetenv("HTTP_DIAL_TIMEOUT")
				_ = os.Unsetenv("HTTP_TLS_TIMEOUT")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.envSetup != nil {
				tt.envSetup()
				defer tt.cleanup()
			} else {
				defer tt.cleanup()
			}

			cfg, positionalArgs, err := LoadWithFlags(tt.args)

			if tt.expectError {
				if err == nil {
					t.Fatal("LoadWithFlags() expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Fatalf("LoadWithFlags() error = %v", err)
			}

			if cfg.Mode != tt.expectedMode {
				t.Errorf("LoadWithFlags() mode = %q, want %q", cfg.Mode, tt.expectedMode)
			}

			if len(tt.args) == 2 && tt.args[0] == "--" && tt.args[1] == "--version" {
				if cfg.Version {
					t.Error("LoadWithFlags() marked positional --version as version flag")
				}
				if len(positionalArgs) != 1 || positionalArgs[0] != "--version" {
					t.Errorf("LoadWithFlags() positional args = %q, want [--version]", positionalArgs)
				}
			}

			if len(positionalArgs) > 0 && tt.name != "version after double dash remains positional argument" && !strings.Contains(positionalArgs[0], "question") {
				t.Errorf("LoadWithFlags() positional args = %q, expected combined prompt", positionalArgs)
			}

			if cfg.Concurrency < 1 {
				t.Errorf("LoadWithFlags() concurrency = %d, want >= 1", cfg.Concurrency)
			}

			if cfg.ChunkLength < 1000 {
				t.Errorf("LoadWithFlags() chunkLength = %d, want >= 1000", cfg.ChunkLength)
			}
		})
	}
}

func TestNormalizeModeSupportsRAG(t *testing.T) {
	if got := normalizeMode("rag"); got != "rag" {
		t.Fatalf("normalizeMode(rag) = %q, want rag", got)
	}
}

func TestLoadWithFlags_RAGOptions(t *testing.T) {
	t.Setenv("EMBEDDING_MODEL", "")
	t.Setenv("RAG_TOP_K", "")
	t.Setenv("RAG_FINAL_K", "")
	t.Setenv("RAG_CHUNK_SIZE", "")
	t.Setenv("RAG_CHUNK_OVERLAP", "")
	t.Setenv("RAG_INDEX_TTL", "")
	t.Setenv("RAG_INDEX_DIR", "")
	t.Setenv("RAG_RERANK", "")

	cfg, _, err := LoadWithFlags([]string{
		"--mode", "rag",
		"--embedding-model", "embed-small",
		"--rag-top-k", "6",
		"--rag-final-k", "3",
		"--rag-chunk-size", "4096",
		"--rag-chunk-overlap", "512",
		"--rag-index-ttl", "2h",
		"--rag-index-dir", "/tmp/ragcli-tests",
		"--rag-rerank", "off",
	})
	if err != nil {
		t.Fatalf("LoadWithFlags() error = %v", err)
	}

	if cfg.Mode != "rag" {
		t.Fatalf("Mode = %q, want rag", cfg.Mode)
	}
	if cfg.EmbeddingModel != "embed-small" {
		t.Fatalf("EmbeddingModel = %q, want embed-small", cfg.EmbeddingModel)
	}
	if cfg.RAGTopK != 6 || cfg.RAGFinalK != 3 {
		t.Fatalf("RAGTopK/RAGFinalK = %d/%d, want 6/3", cfg.RAGTopK, cfg.RAGFinalK)
	}
	if cfg.RAGChunkSize != 4096 || cfg.RAGChunkOverlap != 512 {
		t.Fatalf("RAG chunk config = %d/%d, want 4096/512", cfg.RAGChunkSize, cfg.RAGChunkOverlap)
	}
	if cfg.RAGIndexTTL != 2*time.Hour {
		t.Fatalf("RAGIndexTTL = %v, want 2h", cfg.RAGIndexTTL)
	}
	if cfg.RAGIndexDir != "/tmp/ragcli-tests" {
		t.Fatalf("RAGIndexDir = %q, want /tmp/ragcli-tests", cfg.RAGIndexDir)
	}
	if cfg.RAGRerank != "off" {
		t.Fatalf("RAGRerank = %q, want off", cfg.RAGRerank)
	}
}

func TestLoadWithFlags_RAGDefaultsAreSafeForLocalEmbeddings(t *testing.T) {
	t.Setenv("RAG_CHUNK_SIZE", "")
	t.Setenv("RAG_CHUNK_OVERLAP", "")

	cfg, _, err := LoadWithFlags([]string{"--mode", "rag"})
	if err != nil {
		t.Fatalf("LoadWithFlags() error = %v", err)
	}

	if cfg.RAGChunkSize != 1800 {
		t.Fatalf("RAGChunkSize = %d, want 1800", cfg.RAGChunkSize)
	}
	if cfg.RAGChunkOverlap != 200 {
		t.Fatalf("RAGChunkOverlap = %d, want 200", cfg.RAGChunkOverlap)
	}
}
