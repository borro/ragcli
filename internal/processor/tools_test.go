package processor

import (
	"os"
	"testing"
)

func TestStdinSearch(t *testing.T) {
	tests := []struct {
		name           string
		query          string
		expectedResult string
		expectError    bool
	}{
		{
			name:           "found match (case insensitive)",
			query:          "текст",
			expectedResult: "Нет совпадений в stdin",
			expectError:    false,
		},
		{
			name:           "no match",
			query:          "нет в файле",
			expectedResult: "Нет совпадений в stdin",
			expectError:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
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
	}
}

func TestStdinReadLines(t *testing.T) {
	tests := []struct {
		name           string
		startLine      int
		endLine        int
		expectedResult string
		expectError    bool
	}{
		{
			name:           "valid range",
			startLine:      2,
			endLine:        4,
			expectedResult: "Нет строк в stdin в указанном диапазоне",
			expectError:    false,
		},
		{
			name:           "out of bounds end",
			startLine:      100,
			endLine:        200,
			expectedResult: "Нет строк в stdin в указанном диапазоне",
			expectError:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
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
	}
}

func TestSearchFile(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "test-search-*.txt")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer os.Remove(tmpFile.Name())

	testContent := `первая строка
вторая строка с текстом
третья строка
четвёртая строка с текстом
пятая строка
`
	if _, err := tmpFile.WriteString(testContent); err != nil {
		t.Fatalf("Failed to write to temp file: %v", err)
	}
	tmpFile.Close()

	tests := []struct {
		name           string
		filePath       string
		query          string
		expectedResult string
		expectError    bool
	}{
		{
			name:           "found match (case insensitive)",
			filePath:       tmpFile.Name(),
			query:          "текст",
			expectedResult: "Line 2: вторая строка с текстом\nLine 4: четвёртая строка с текстом",
			expectError:    false,
		},
		{
			name:           "no match",
			filePath:       tmpFile.Name(),
			query:          "нет в файле",
			expectedResult: "Нет совпадений",
			expectError:    false,
		},
		{
			name:           "non-existent file",
			filePath:       "/non/existent/file.txt",
			query:          "test",
			expectedResult: "",
			expectError:    true,
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
	tmpFile, err := os.CreateTemp("", "test-read-*.txt")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer os.Remove(tmpFile.Name())

	testContent := `первая строка
вторая строка
третья строка
четвёртая строка
пятая строка
шестая строка
седьмая строка
восьмая строка
девятая строка
десятая строка
`
	if _, err := tmpFile.WriteString(testContent); err != nil {
		t.Fatalf("Failed to write to temp file: %v", err)
	}
	tmpFile.Close()

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
			filePath:       tmpFile.Name(),
			startLine:      3,
			endLine:        5,
			expectedResult: "3: третья строка\n4: четвёртая строка\n5: пятая строка",
			expectError:    false,
		},
		{
			name:           "out of bounds start",
			filePath:       tmpFile.Name(),
			startLine:      100,
			endLine:        200,
			expectedResult: "Нет строк в указанном диапазоне",
			expectError:    false,
		},
		{
			name:           "invalid file path",
			filePath:       "/non/existent/file.txt",
			startLine:      1,
			endLine:        5,
			expectedResult: "",
			expectError:    true,
		},
		{
			name:           "negative start (should be clamped to 1)",
			filePath:       tmpFile.Name(),
			startLine:      -5,
			endLine:        5,
			expectedResult: "1: первая строка\n2: вторая строка\n3: третья строка\n4: четвёртая строка\n5: пятая строка",
			expectError:    false,
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
