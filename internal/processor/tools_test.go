package processor

import (
	"os"
	"testing"
)

func TestStdinSearch(t *testing.T) {
	tests := []struct {
		name           string
		input          string
		query          string
		expectedResult string
		expectError    bool
	}{
		{
			name: "found match (case insensitive)",
			input: `первая строка
вторая строка с текстом
третья строка`,
			query:          "ТЕКСТ",
			expectedResult: "Line 2: вторая строка с текстом",
		},
		{
			name:           "no match",
			input:          "первая строка\nвторая строка\n",
			query:          "нет в файле",
			expectedResult: "Нет совпадений в stdin",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			withStdin(t, tt.input, func() {
				result, err := StdinSearch(tt.query)

				if tt.expectError {
					if err == nil {
						t.Fatal("StdinSearch() expected error, got nil")
					}
					return
				}

				if err != nil {
					t.Fatalf("StdinSearch() error = %v", err)
				}

				if result != tt.expectedResult {
					t.Errorf("StdinSearch() result = %q, want %q", result, tt.expectedResult)
				}
			})
		})
	}
}

func TestStdinReadLines(t *testing.T) {
	tests := []struct {
		name           string
		input          string
		startLine      int
		endLine        int
		expectedResult string
		expectError    bool
	}{
		{
			name: "valid range",
			input: `первая строка
вторая строка
третья строка
четвёртая строка`,
			startLine:      2,
			endLine:        3,
			expectedResult: "2: вторая строка\n3: третья строка",
		},
		{
			name:           "out of bounds end",
			input:          "первая строка\nвторая строка\n",
			startLine:      100,
			endLine:        200,
			expectedResult: "Нет строк в stdin в указанном диапазоне",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			withStdin(t, tt.input, func() {
				result, err := StdinReadLines(tt.startLine, tt.endLine)

				if tt.expectError {
					if err == nil {
						t.Fatal("StdinReadLines() expected error, got nil")
					}
					return
				}

				if err != nil {
					t.Fatalf("StdinReadLines() error = %v", err)
				}

				if result != tt.expectedResult {
					t.Errorf("StdinReadLines() result = %q, want %q", result, tt.expectedResult)
				}
			})
		})
	}
}

func TestSearchFile(t *testing.T) {
	filePath := writeTempFile(t, `первая строка
вторая строка с текстом
третья строка
четвёртая строка с текстом
пятая строка
`)

	tests := []struct {
		name           string
		filePath       string
		query          string
		expectedResult string
		expectError    bool
	}{
		{
			name:           "found match (case insensitive)",
			filePath:       filePath,
			query:          "текст",
			expectedResult: "Line 2: вторая строка с текстом\nLine 4: четвёртая строка с текстом",
		},
		{
			name:           "no match",
			filePath:       filePath,
			query:          "нет в файле",
			expectedResult: "Нет совпадений",
		},
		{
			name:        "non-existent file",
			filePath:    "/non/existent/file.txt",
			query:       "test",
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := SearchFile(tt.filePath, tt.query)

			if tt.expectError {
				if err == nil {
					t.Fatal("SearchFile() expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Fatalf("SearchFile() error = %v", err)
			}

			if result != tt.expectedResult {
				t.Errorf("SearchFile() result = %q, want %q", result, tt.expectedResult)
			}
		})
	}
}

func TestReadLines(t *testing.T) {
	filePath := writeTempFile(t, `первая строка
вторая строка
третья строка
четвёртая строка
пятая строка
шестая строка
седьмая строка
восьмая строка
девятая строка
десятая строка
`)

	tests := []struct {
		name           string
		filePath       string
		startLine      int
		endLine        int
		expectedResult string
		expectError    bool
	}{
		{
			name:           "valid range",
			filePath:       filePath,
			startLine:      3,
			endLine:        5,
			expectedResult: "3: третья строка\n4: четвёртая строка\n5: пятая строка",
		},
		{
			name:           "out of bounds start",
			filePath:       filePath,
			startLine:      100,
			endLine:        200,
			expectedResult: "Нет строк в указанном диапазоне",
		},
		{
			name:        "invalid file path",
			filePath:    "/non/existent/file.txt",
			startLine:   1,
			endLine:     5,
			expectError: true,
		},
		{
			name:           "negative start (should be clamped to 1)",
			filePath:       filePath,
			startLine:      -5,
			endLine:        5,
			expectedResult: "1: первая строка\n2: вторая строка\n3: третья строка\n4: четвёртая строка\n5: пятая строка",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := ReadLines(tt.filePath, tt.startLine, tt.endLine)

			if tt.expectError {
				if err == nil {
					t.Fatal("ReadLines() expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Fatalf("ReadLines() error = %v", err)
			}

			if result != tt.expectedResult {
				t.Errorf("ReadLines() result = %q, want %q", result, tt.expectedResult)
			}
		})
	}
}

func writeTempFile(t *testing.T, content string) string {
	t.Helper()

	tmpFile, err := os.CreateTemp("", "processor-tools-*.txt")
	if err != nil {
		t.Fatalf("failed to create temp file: %v", err)
	}

	if _, err := tmpFile.WriteString(content); err != nil {
		t.Fatalf("failed to write temp file: %v", err)
	}

	if err := tmpFile.Close(); err != nil {
		t.Fatalf("failed to close temp file: %v", err)
	}

	t.Cleanup(func() {
		if err := os.Remove(tmpFile.Name()); err != nil {
			t.Logf("failed to remove temp file: %v", err)
		}
	})

	return tmpFile.Name()
}

func withStdin(t *testing.T, input string, fn func()) {
	t.Helper()

	originalStdin := os.Stdin
	reader, writer, err := os.Pipe()
	if err != nil {
		t.Fatalf("failed to create pipe: %v", err)
	}

	if _, err := writer.WriteString(input); err != nil {
		t.Fatalf("failed to write stdin fixture: %v", err)
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("failed to close stdin writer: %v", err)
	}

	os.Stdin = reader
	t.Cleanup(func() {
		os.Stdin = originalStdin
		if err := reader.Close(); err != nil {
			t.Logf("failed to close stdin reader: %v", err)
		}
	})

	fn()
}
