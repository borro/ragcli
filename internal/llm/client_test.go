package llm

import (
	"testing"

	"github.com/borro/ragcli/internal/config"
)

func TestNewClient(t *testing.T) {
	tests := []struct {
		name           string
		baseURL        string
		model          string
		apiKey         string
		retryCount     int
		expectedModel  string
		expectedRetry  int
	}{
		{
			name:         "valid client creation",
			baseURL:      "http://localhost:1234/v1",
			model:        "gpt-4",
			apiKey:       "test-api-key",
			retryCount:   3,
			expectedModel: "gpt-4",
			expectedRetry: 3,
		},
		{
			name:         "empty baseURL uses default",
			baseURL:      "",
			model:        "gpt-3.5-turbo",
			apiKey:       "another-key",
			retryCount:   5,
			expectedModel: "gpt-3.5-turbo",
			expectedRetry: 5,
		},
		{
			name:         "zero retry count",
			baseURL:      "http://test.com",
			model:        "test-model",
			apiKey:       "",
			retryCount:   0,
			expectedModel: "test-model",
			expectedRetry: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &config.Config{}
			client := NewClient(tt.baseURL, tt.model, tt.apiKey, tt.retryCount, cfg)

			if client.model != tt.expectedModel {
				t.Errorf("NewClient() model = %q, want %q", client.model, tt.expectedModel)
			}
			if client.retryCount != tt.expectedRetry {
				t.Errorf("NewClient() retryCount = %d, want %d", client.retryCount, tt.expectedRetry)
			}
		})
	}
}

func TestClient_Model(t *testing.T) {
	cfg := &config.Config{}
	client := NewClient("http://test.com", "test-model", "api-key", 3, cfg)

	model := client.Model()
	if model != "test-model" {
		t.Errorf("Model() = %q, want %q", model, "test-model")
	}
}
