package processor

import (
	"strings"
	"testing"
)

func TestSplitByLines(t *testing.T) {
	tests := []struct {
		name           string
		input          string
		maxLength      int
		expectedChunks []string
		expectError    bool
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
			expectError: false,
		},
		{
			name:      "input within single chunk",
			input:     "hello world",
			maxLength: 100,
			expectedChunks: []string{
				"hello world",
			},
			expectError: false,
		},
		{
			name:           "empty input returns empty chunks",
			input:          "",
			maxLength:      100,
			expectedChunks: []string{},
			expectError:    false,
		},
		{
			name:      "new line doesn't fit, new chunk created",
			input:     "hello\nworld",
			maxLength: 5,
			expectedChunks: []string{
				"hello",
				"world",
			},
			expectError: false,
		},
		{
			name: "long lines are chunked",
			// Функция добавляет строку в текущий чанк, когда она не превышает maxLength
			input:     "this is a very long line that exceeds the maxLength limit significantly\nanother line",
			maxLength: 30,
			expectedChunks: []string{
				"this is a very long line that exceeds the maxLength limit significantly",
				"another line",
			},
			expectError: false,
		},
		{
			name:      "lines with newlines within limit",
			input:     "line1\nline2\nline3",
			maxLength: 100,
			expectedChunks: []string{
				"line1\nline2\nline3",
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			chunks, err := SplitByLines(strings.NewReader(tt.input), tt.maxLength)

			if tt.expectError {
				if err == nil {
					t.Fatal("SplitByLines() expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Fatalf("SplitByLines() error = %v", err)
			}

			if len(chunks) != len(tt.expectedChunks) {
				t.Errorf("SplitByLines() returned %d chunks, want %d", len(chunks), len(tt.expectedChunks))
				for i, chunk := range chunks {
					t.Logf("chunk[%d] = %q (len=%d)", i, chunk, len(chunk))
				}
				for i, expected := range tt.expectedChunks {
					t.Logf("expected[%d] = %q (len=%d)", i, expected, len(expected))
				}
				return
			}

			for i, chunk := range chunks {
				if chunk != tt.expectedChunks[i] {
					t.Errorf("chunk[%d] = %q, want %q", i, chunk, tt.expectedChunks[i])
				}
			}
		})
	}
}
