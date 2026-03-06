package llm

import (
	"context"
	"fmt"
	"time"

	"github.com/borro/ragcli/internal/config"
	openai "github.com/sashabaranov/go-openai"
)

// Client представляет клиент для взаимодействия с OpenAI API через go-openai
type Client struct {
	client     *openai.Client
	model      string
	retryCount int
}

// Message представляет сообщение в чате
type Message = openai.ChatCompletionMessage

// Choice представляет выбор из ответа модели
type Choice = openai.ChatCompletionChoice

// ChatCompletionRequest представляет запрос к chat completion API
type ChatCompletionRequest struct {
	Model      string                         `json:"model"`
	Messages   []openai.ChatCompletionMessage `json:"messages"`
	Tools      []Tool                         `json:"tools,omitempty"`
	ToolChoice interface{}                    `json:"tool_choice,omitempty"`
}

// FunctionDefinition represents function definition for tool calling
type FunctionDefinition = openai.FunctionDefinition

// Tool представляет собой определение инструмента
type Tool = openai.Tool

// ChatCompletionResponse представляет ответ от API
type ChatCompletionResponse = openai.ChatCompletionResponse

// ToolCall represents a tool call from the model
type ToolCall = openai.ToolCall

// ToolCallFunc represents the function part of a tool call
type ToolCallFunc = openai.FunctionCall

// NewClient создаёт новый LLM клиент, используя go-openai библиотеку
func NewClient(baseURL, model, apiKey string, retryCount int, cfg *config.Config) *Client {
	config := openai.DefaultConfig(apiKey)
	if baseURL != "" {
		config.BaseURL = baseURL
	}

	client := &Client{
		client:     openai.NewClientWithConfig(config),
		model:      model,
		retryCount: retryCount,
	}
	return client
}

// SendRequest отправляет запрос к LLM API с механизмом retry и exponential backoff
func (c *Client) SendRequest(ctx context.Context, req ChatCompletionRequest) (*ChatCompletionResponse, error) {
	for attempt := 1; attempt <= c.retryCount+1; attempt++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		resp, err := c.doRequest(ctx, req)
		if err == nil {
			return &resp, nil
		}

		config.Log.Debug("request failed", "attempt", attempt, "error", err)

		// Проверяем тип ошибки и решаем стоит ли retry
		if attempt <= c.retryCount {
			delay := 1 << (attempt - 1) // exponential backoff: 1, 2, 4, 8 seconds
			config.Log.Debug("retrying request", "attempt", attempt+1, "delay_seconds", delay)
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(time.Duration(delay) * time.Second):
			}
		}
	}

	return nil, fmt.Errorf("request failed after %d attempts", c.retryCount+1)
}

// doRequest выполняет один HTTP запрос к API через go-openai
func (c *Client) doRequest(ctx context.Context, req ChatCompletionRequest) (openai.ChatCompletionResponse, error) {
	openaiReq := openai.ChatCompletionRequest{
		Model:    c.model,
		Messages: req.Messages,
		Tools:    req.Tools,
	}

	if req.ToolChoice != nil {
		openaiReq.ToolChoice = req.ToolChoice
	}

	return c.client.CreateChatCompletion(ctx, openaiReq)
}

// Model возвращает имя модели
func (c *Client) Model() string {
	return c.model
}
