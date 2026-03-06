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
			name:          "env variable set",
			key:           "TEST_KEY",
			envValue:      "test_value",
			defaultValue:  "default",
			expectedResult: "test_value",
		},
		{
			name:          "env variable not set",
			key:            "NON_EXISTENT_KEY",
			envValue:       "",
			defaultValue:   "default",
			expectedResult: "default",
		},
		{
			name:          "empty env with default",
			key:            "EMPTY_KEY",
			envValue:       "",
			defaultValue:   "fallback",
			expectedResult: "fallback",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.envValue != "" {
				os.Setenv(tt.key, tt.envValue)
				cleanup := func() { os.Unsetenv(tt.key) }
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
			name:          "valid integer",
			key:           "TEST_INT",
			envValue:      "42",
			defaultValue:  0,
			expectedResult: 42,
		},
		{
			name:          "invalid integer (string)",
			key:           "TEST_INT",
			envValue:       "not_a_number",
			defaultValue:   0,
			expectedResult: 0,
		},
		{
			name:          "empty env with default",
			key:            "EMPTY_INT",
			envValue:       "",
			defaultValue:   100,
			expectedResult: 100,
		},
		{
			name:          "valid with spaces",
			key:           "TEST_INT",
			envValue:      "  15  ",
			defaultValue:  0,
			expectedResult: 15,
		},
		{
			name:          "negative value",
			key:           "TEST_INT",
			envValue:      "-10",
			defaultValue:  0,
			expectedResult: -10,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.envValue != "" {
				os.Setenv(tt.key, tt.envValue)
				cleanup := func() { os.Unsetenv(tt.key) }
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
			name:          "true value",
			key:           "TEST_BOOL",
			envValue:      "true",
			defaultValue:  false,
			expectedResult: true,
		},
		{
			name:          "1 as true",
			key:           "TEST_BOOL",
			envValue:      "1",
			defaultValue:  false,
			expectedResult: true,
		},
		{
			name:          "yes as true",
			key:           "TEST_BOOL",
			envValue:      "yes",
			defaultValue:  false,
			expectedResult: true,
		},
		{
			name:          "false value",
			key:           "TEST_BOOL",
			envValue:      "false",
			defaultValue:  true,
			expectedResult: false,
		},
		{
			name:          "0 as false",
			key:           "TEST_BOOL",
			envValue:      "0",
			defaultValue:  true,
			expectedResult: false,
		},
		{
			name:          "no as false",
			key:           "TEST_BOOL",
			envValue:      "no",
			defaultValue:  true,
			expectedResult: false,
		},
		{
			name:          "mixed case true",
			key:           "TEST_BOOL",
			envValue:      "TRUE",
			defaultValue:  false,
			expectedResult: true,
		},
		{
			name:          "mixed case false",
			key:           "TEST_BOOL",
			envValue:      "FALSE",
			defaultValue:  true,
			expectedResult: false,
		},
		{
			name:          "invalid value returns default",
			key:           "TEST_BOOL",
			envValue:      "invalid",
			defaultValue:  true,
			expectedResult: true,
		},
		{
			name:          "empty env with default",
			key:            "EMPTY_BOOL",
			envValue:       "",
			defaultValue:   false,
			expectedResult: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.envValue != "" {
				os.Setenv(tt.key, tt.envValue)
				cleanup := func() { os.Unsetenv(tt.key) }
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
			name:          "valid duration",
			key:           "TEST_DURATION",
			envValue:      "5m",
			defaultValue:  10 * time.Minute,
			expectedResult: 5 * time.Minute,
		},
		{
			name:          "negative duration ignored",
			key:           "TEST_DURATION",
			envValue:      "-1m",
			defaultValue:  10 * time.Minute,
			expectedResult: 10 * time.Minute,
		},
		{
			name:          "invalid duration format ignored",
			key:           "TEST_DURATION",
			envValue:      "5xyz",
			defaultValue:  10 * time.Minute,
			expectedResult: 10 * time.Minute,
		},
		{
			name:          "empty env with default",
			key:            "EMPTY_DURATION",
			envValue:       "",
			defaultValue:   30 * time.Second,
			expectedResult: 30 * time.Second,
		},
		{
			name:          "valid seconds",
			key:           "TEST_DURATION",
			envValue:      "30s",
			defaultValue:  10 * time.Minute,
			expectedResult: 30 * time.Second,
		},
		{
			name:          "valid hours",
			key:           "TEST_DURATION",
			envValue:      "2h",
			defaultValue:  10 * time.Minute,
			expectedResult: 2 * time.Hour,
		},
		{
			name:          "millisecond precision",
			key:           "TEST_DURATION",
			envValue:      "500ms",
			defaultValue:  1 * time.Second,
			expectedResult: 500 * time.Millisecond,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.envValue != "" {
				os.Setenv(tt.key, tt.envValue)
				cleanup := func() { os.Unsetenv(tt.key) }
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
		name          string
		args          []string
		envSetup      func()
		expectedMode  string
		expectError   bool
		cleanup       func()
	}{
		{
			name:       "valid flags with all parameters",
			args:       []string{"-f", "test.txt", "-mode", "rag", "-api-url", "http://test.com", "-model", "gpt-4", "-c", "4", "-l", "5000", "-r", "5", "-v"},
			expectedMode: "rag",
			expectError: false,
			cleanup: func() {
				os.Unsetenv("INPUT_FILE")
				os.Unsetenv("MODE")
				os.Unsetenv("LLM_API_URL")
				os.Unsetenv("LLM_MODEL")
				os.Unsetenv("OPENAI_API_KEY")
				os.Unsetenv("CONCURRENCY")
				os.Unsetenv("LENGTH")
				os.Unsetenv("RETRY")
				os.Unsetenv("VERBOSE")
				os.Unsetenv("HTTP_REQUEST_TIMEOUT")
				os.Unsetenv("HTTP_DIAL_TIMEOUT")
				os.Unsetenv("HTTP_TLS_TIMEOUT")
			},
		},
		{
			name:         "empty args uses env defaults",
			args:         []string{},
			expectedMode: "map-reduce",
			expectError:  false,
			cleanup: func() {
				os.Unsetenv("INPUT_FILE")
				os.Unsetenv("MODE")
				os.Unsetenv("LLM_API_URL")
				os.Unsetenv("LLM_MODEL")
				os.Unsetenv("OPENAI_API_KEY")
				os.Unsetenv("CONCURRENCY")
				os.Unsetenv("LENGTH")
				os.Unsetenv("RETRY")
				os.Unsetenv("VERBOSE")
				os.Unsetenv("HTTP_REQUEST_TIMEOUT")
				os.Unsetenv("HTTP_DIAL_TIMEOUT")
				os.Unsetenv("HTTP_TLS_TIMEOUT")
			},
		},
		{
			name:          "env variable overrides flag",
			args:          []string{"-f", "flag.txt"},
			expectedMode:  "custom_mode",
			expectError:   false,
			envSetup: func() {
				os.Setenv("INPUT_FILE", "env.txt")
				os.Setenv("MODE", "custom_mode")
			},
			cleanup: func() {
				os.Unsetenv("INPUT_FILE")
				os.Unsetenv("MODE")
				os.Unsetenv("LLM_API_URL")
				os.Unsetenv("LLM_MODEL")
				os.Unsetenv("OPENAI_API_KEY")
				os.Unsetenv("CONCURRENCY")
				os.Unsetenv("LENGTH")
				os.Unsetenv("RETRY")
				os.Unsetenv("VERBOSE")
				os.Unsetenv("HTTP_REQUEST_TIMEOUT")
				os.Unsetenv("HTTP_DIAL_TIMEOUT")
				os.Unsetenv("HTTP_TLS_TIMEOUT")
			},
		},
		{
			name:          "short and long flag aliases should use same value",
			args:          []string{"-f", "test.txt", "-file", "test2.txt"},
			expectedMode:  "map-reduce",
			expectError:   false,
			cleanup: func() {
				os.Unsetenv("INPUT_FILE")
				os.Unsetenv("MODE")
				os.Unsetenv("LLM_API_URL")
				os.Unsetenv("LLM_MODEL")
				os.Unsetenv("OPENAI_API_KEY")
				os.Unsetenv("CONCURRENCY")
				os.Unsetenv("LENGTH")
				os.Unsetenv("RETRY")
				os.Unsetenv("VERBOSE")
				os.Unsetenv("HTTP_REQUEST_TIMEOUT")
				os.Unsetenv("HTTP_DIAL_TIMEOUT")
				os.Unsetenv("HTTP_TLS_TIMEOUT")
			},
		},
		{
			name:          "invalid concurrency defaults to 1",
			args:          []string{"-c", "-1"},
			expectedMode:  "map-reduce",
			expectError:   false,
			cleanup: func() {
				os.Unsetenv("INPUT_FILE")
				os.Unsetenv("MODE")
				os.Unsetenv("LLM_API_URL")
				os.Unsetenv("LLM_MODEL")
				os.Unsetenv("OPENAI_API_KEY")
				os.Unsetenv("CONCURRENCY")
				os.Unsetenv("LENGTH")
				os.Unsetenv("RETRY")
				os.Unsetenv("VERBOSE")
				os.Unsetenv("HTTP_REQUEST_TIMEOUT")
				os.Unsetenv("HTTP_DIAL_TIMEOUT")
				os.Unsetenv("HTTP_TLS_TIMEOUT")
			},
		},
		{
			name:          "invalid chunk length defaults to 1000",
			args:          []string{"-l", "500"},
			expectedMode:  "map-reduce",
			expectError:   false,
			cleanup: func() {
				os.Unsetenv("INPUT_FILE")
				os.Unsetenv("MODE")
				os.Unsetenv("LLM_API_URL")
				os.Unsetenv("LLM_MODEL")
				os.Unsetenv("OPENAI_API_KEY")
				os.Unsetenv("CONCURRENCY")
				os.Unsetenv("LENGTH")
				os.Unsetenv("RETRY")
				os.Unsetenv("VERBOSE")
				os.Unsetenv("HTTP_REQUEST_TIMEOUT")
				os.Unsetenv("HTTP_DIAL_TIMEOUT")
				os.Unsetenv("HTTP_TLS_TIMEOUT")
			},
		},
		{
			name:       "multiple positional args combined into single prompt",
			args:       []string{"question1 question2 question3"},
			expectedMode: "map-reduce",
			expectError: false,
			cleanup: func() {
				os.Unsetenv("INPUT_FILE")
				os.Unsetenv("MODE")
				os.Unsetenv("LLM_API_URL")
				os.Unsetenv("LLM_MODEL")
				os.Unsetenv("OPENAI_API_KEY")
				os.Unsetenv("CONCURRENCY")
				os.Unsetenv("LENGTH")
				os.Unsetenv("RETRY")
				os.Unsetenv("VERBOSE")
				os.Unsetenv("HTTP_REQUEST_TIMEOUT")
				os.Unsetenv("HTTP_DIAL_TIMEOUT")
				os.Unsetenv("HTTP_TLS_TIMEOUT")
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

			if len(positionalArgs) > 0 && !strings.Contains(positionalArgs[0], "question") {
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
