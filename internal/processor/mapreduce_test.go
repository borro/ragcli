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
