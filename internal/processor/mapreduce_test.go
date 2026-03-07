package processor

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/borro/ragcli/internal/config"
	"github.com/borro/ragcli/internal/llm"
)

func TestSplitByLines(t *testing.T) {
	tests := []struct {
		name           string
		input          string
		maxLength      int
		expectedChunks []string
	}{
		{
			name:      "simple split",
			input:     "line1\nline2\nline3",
			maxLength: 10,
			expectedChunks: []string{
				"line1",
				"line2",
				"line3",
			},
		},
		{
			name:      "input within single chunk",
			input:     "hello world",
			maxLength: 100,
			expectedChunks: []string{
				"hello world",
			},
		},
		{
			name:           "empty input",
			input:          "",
			maxLength:      100,
			expectedChunks: []string{},
		},
		{
			name:      "new line doesn't fit",
			input:     "hello\nworld",
			maxLength: 5,
			expectedChunks: []string{
				"hello",
				"world",
			},
		},
		{
			name:      "long line larger than chunk",
			input:     "this is a very long line that exceeds the maxLength limit significantly\nanother line",
			maxLength: 30,
			expectedChunks: []string{
				"this is a very long line that exceeds the maxLength limit significantly",
				"another line",
			},
		},
		{
			name:      "multiple lines fit into one chunk",
			input:     "line1\nline2\nline3",
			maxLength: 100,
			expectedChunks: []string{
				"line1\nline2\nline3",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			chunks, err := SplitByLines(strings.NewReader(tt.input), tt.maxLength)
			if err != nil {
				t.Fatalf("SplitByLines() error = %v", err)
			}

			if len(chunks) != len(tt.expectedChunks) {
				t.Fatalf(
					"SplitByLines() returned %d chunks, want %d\nchunks=%v",
					len(chunks),
					len(tt.expectedChunks),
					chunks,
				)
			}

			for i := range chunks {
				if chunks[i] != tt.expectedChunks[i] {
					t.Errorf(
						"chunk[%d] = %q, want %q",
						i,
						chunks[i],
						tt.expectedChunks[i],
					)
				}
			}
		})
	}
}

func TestSplitByLines_ExactBoundary(t *testing.T) {
	input := "12345\n67890"

	chunks, err := SplitByLines(strings.NewReader(input), 5)
	if err != nil {
		t.Fatal(err)
	}

	expected := []string{
		"12345",
		"67890",
	}

	if len(chunks) != len(expected) {
		t.Fatalf("got %v", chunks)
	}
}

func TestSplitByLines_LargeInput(t *testing.T) {
	var builder strings.Builder

	for i := 0; i < 1000; i++ {
		builder.WriteString("line\n")
	}

	_, err := SplitByLines(strings.NewReader(builder.String()), 50)
	if err != nil {
		t.Fatal(err)
	}
}

func TestRunSRMR_EndToEndWithRefine(t *testing.T) {
	client := newSequencedLLMClient(t, []string{
		"fact from chunk 1",
		"fact from chunk 2",
		"fact from chunk 1\nfact from chunk 2",
		"initial answer",
		"SUPPORTED: no\nISSUES:\n- make it more precise",
		"refined answer",
	}, nil)

	result, err := RunSRMR(context.Background(), client, strings.NewReader("aaa\nbbb\n"), &config.Config{
		InputPath:   "input.txt",
		ChunkLength: 3,
		Concurrency: 1,
		RetryCount:  0,
	}, "Что в файле?")
	if err != nil {
		t.Fatalf("RunSRMR() error = %v", err)
	}
	if result != "refined answer" {
		t.Fatalf("RunSRMR() result = %q, want %q", result, "refined answer")
	}
}

func TestRunSRMR_EmptyInput(t *testing.T) {
	client := newSequencedLLMClient(t, nil, nil)

	result, err := RunSRMR(context.Background(), client, strings.NewReader(""), &config.Config{
		InputPath:   "input.txt",
		ChunkLength: 10,
		Concurrency: 1,
		RetryCount:  0,
	}, "Что в файле?")
	if err != nil {
		t.Fatalf("RunSRMR() error = %v", err)
	}
	if result != "Файл пустой" {
		t.Fatalf("RunSRMR() result = %q, want %q", result, "Файл пустой")
	}
}

func TestRunSRMR_NoInfoWhenMapSkipsOrErrors(t *testing.T) {
	t.Run("all map requests skip", func(t *testing.T) {
		client := newSequencedLLMClient(t, []string{"SKIP", "SKIP"}, nil)

		result, err := RunSRMR(context.Background(), client, strings.NewReader("aaa\nbbb\n"), &config.Config{
			InputPath:   "input.txt",
			ChunkLength: 3,
			Concurrency: 1,
			RetryCount:  0,
		}, "Что в файле?")
		if err != nil {
			t.Fatalf("RunSRMR() error = %v", err)
		}
		if result != "Нет информации для ответа" {
			t.Fatalf("RunSRMR() result = %q, want %q", result, "Нет информации для ответа")
		}
	})

	t.Run("map requests fail", func(t *testing.T) {
		client := newSequencedLLMClient(t, nil, []int{http.StatusInternalServerError, http.StatusInternalServerError})

		result, err := RunSRMR(context.Background(), client, strings.NewReader("aaa\nbbb\n"), &config.Config{
			InputPath:   "input.txt",
			ChunkLength: 3,
			Concurrency: 1,
			RetryCount:  0,
		}, "Что в файле?")
		if err != nil {
			t.Fatalf("RunSRMR() error = %v", err)
		}
		if result != "Нет информации для ответа" {
			t.Fatalf("RunSRMR() result = %q, want %q", result, "Нет информации для ответа")
		}
	})
}

func newSequencedLLMClient(t *testing.T, responses []string, statuses []int) *llm.Client {
	t.Helper()

	var mu sync.Mutex
	requestIndex := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		index := requestIndex
		requestIndex++
		mu.Unlock()

		statusCode := http.StatusOK
		if index < len(statuses) && statuses[index] != 0 {
			statusCode = statuses[index]
		}
		if statusCode != http.StatusOK {
			http.Error(w, "upstream error", statusCode)
			return
		}

		content := ""
		if index < len(responses) {
			content = responses[index]
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]any{
			"id":      "chatcmpl-test",
			"object":  "chat.completion",
			"created": 1,
			"model":   "test-model",
			"choices": []map[string]any{
				{
					"index": 0,
					"message": map[string]any{
						"role":    "assistant",
						"content": content,
					},
					"finish_reason": "stop",
				},
			},
			"usage": map[string]any{
				"prompt_tokens":     10,
				"completion_tokens": 5,
				"total_tokens":      15,
			},
		}); err != nil {
			t.Fatalf("failed to encode fake llm response: %v", err)
		}
	}))
	t.Cleanup(server.Close)

	return llm.NewClient(server.URL, "test-model", "", 0)
}
