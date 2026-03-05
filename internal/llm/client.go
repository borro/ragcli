package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"time"

	"github.com/borro/ragcli/internal/config"
)

// Client представляет HTTP клиент для взаимодействия с LLM API
type Client struct {
	baseURL    string
	model      string
	apiKey     string
	httpClient *http.Client
	retryCount int
}

// Message представляет сообщение в чате
type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	Name       string     `json:"name,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// Choice представляет выбор из ответа модели
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

// ChatCompletionRequest представляет запрос к chat completion API
type ChatCompletionRequest struct {
	Model      string      `json:"model"`
	Messages   []Message   `json:"messages"`
	Tools      []Tool      `json:"tools,omitempty"`
	ToolChoice interface{} `json:"tool_choice,omitempty"`
}

// ToolFunction defines the function schema for tool calling
type ToolFunction struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

// Tool представляет собой определение инструмента
type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ChatCompletionResponse представляет ответ от API
type ChatCompletionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
}

// ToolCall represents a tool call from the model
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function ToolCallFunc `json:"function"`
}

// ToolCallFunc represents the function part of a tool call
type ToolCallFunc struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// NewClient создаёт новый LLM клиент с настройками из конфига.
func NewClient(baseURL, model, apiKey string, retryCount int, cfg *config.Config) *Client {
	// Настраиваем HTTP transport с отдельными таймаутами для dial и TLS handshake
	transport := &http.Transport{
		DialContext: (&net.Dialer{
			Timeout: cfg.DialTimeout,
		}).DialContext,
		TLSHandshakeTimeout: cfg.TLSTimeout,
	}

	httpClient := &http.Client{
		Transport: transport,
		Timeout:   cfg.RequestTimeout,
	}

	return &Client{
		baseURL:    baseURL,
		model:      model,
		apiKey:     apiKey,
		httpClient: httpClient,
		retryCount: retryCount,
	}
}

// SendRequest отправляет запрос к LLM API с механизмом retry и exponential backoff
func (c *Client) SendRequest(ctx context.Context, req ChatCompletionRequest) (*ChatCompletionResponse, error) {
	var resp *ChatCompletionResponse
	var err error

	for attempt := 1; attempt <= c.retryCount+1; attempt++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		resp, err = c.doRequest(ctx, req)
		if err == nil {
			return resp, nil
		}

		config.Log.Debug("request failed", slog.Int("attempt", attempt), slog.Any("error", err))

		// Проверяем, стоит ли retry (5xx ошибки или network errors)
		var statusErr *StatusError
		if errors.As(err, &statusErr) {
			if statusErr.statusCode >= 200 && statusErr.statusCode < 500 && statusErr.statusCode != 429 {
				return nil, err // Не retry client errors (кроме 429)
			}
		}

		if attempt <= c.retryCount {
			delay := time.Duration(attempt*attempt) * time.Second
			config.Log.Debug("retrying request", slog.Int("attempt", attempt+1), slog.Duration("delay", delay))
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(delay):
			}
		}
	}

	return nil, fmt.Errorf("request failed after %d attempts: %w", c.retryCount+1, err)
}

// doRequest выполняет один HTTP запрос к API
func (c *Client) doRequest(ctx context.Context, req ChatCompletionRequest) (*ChatCompletionResponse, error) {
	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/chat/completions", c.baseURL)
	config.Log.Debug("sending HTTP request to LLM API", slog.String("url", url), slog.Int("body_bytes", len(reqBody)))

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		httpReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiKey))
	}

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		config.Log.Error("failed to send HTTP request to LLM API", "error", err)
		return nil, &StatusError{err: err, statusCode: 0}
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		config.Log.Debug("received error status from LLM API", slog.Int("status_code", resp.StatusCode), slog.String("response_body", string(body)))
		return nil, &StatusError{
			err:        fmt.Errorf("API returned status %d: %s", resp.StatusCode, string(body)),
			statusCode: resp.StatusCode,
		}
	}

	config.Log.Debug("received successful response from LLM API", slog.Int("status_code", resp.StatusCode), slog.Int("body_bytes", len(body)))

	var apiResp ChatCompletionResponse
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &apiResp, nil
}

// StatusError представляет ошибку HTTP статуса
type StatusError struct {
	err        error
	statusCode int
}

func (s *StatusError) Error() string {
	if s.err != nil {
		return s.err.Error()
	}
	return fmt.Sprintf("HTTP status %d", s.statusCode)
}

func (s *StatusError) Unwrap() error {
	return s.err
}

// GetStatusCode возвращает HTTP статус код
func (s *StatusError) GetStatusCode() int {
	return s.statusCode
}

// Model возвращает имя модели
func (c *Client) Model() string {
	return c.model
}
