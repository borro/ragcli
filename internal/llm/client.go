package llm

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

// Client представляет клиент для взаимодействия с OpenAI API через go-openai.
type Client struct {
	client     *openai.Client
	httpClient *http.Client
	model      string
	retryCount int
	baseURL    string
	apiKey     string
	sleep      sleepFunc
}

// ChatRequester describes chat completion clients used by application packages.
type ChatRequester interface {
	SendRequestWithMetrics(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, RequestMetrics, error)
}

type AutoContextLengthResolver interface {
	ResolveAutoContextLength(ctx context.Context) (int, error)
}

type ChatAutoContextRequester interface {
	ChatRequester
	AutoContextLengthResolver
}

// Embedder представляет клиент для embeddings API.
type Embedder struct {
	client     *openai.Client
	model      string
	retryCount int
	sleep      sleepFunc
}

// EmbeddingRequester describes embeddings clients used by application packages.
type EmbeddingRequester interface {
	CreateEmbeddingsWithMetrics(ctx context.Context, inputs []string) ([][]float32, EmbeddingMetrics, error)
}

// RequestMetrics содержит метаданные и наблюдаемость по одному запросу к LLM.
type RequestMetrics struct {
	Model            string
	Attempt          int
	MessageCount     int
	ToolCount        int
	ToolChoice       string
	Duration         time.Duration
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
	TokensPerSecond  float64
	ChoiceCount      int
	HasToolCalls     bool
}

// EmbeddingMetrics содержит метрики одного embeddings запроса.
type EmbeddingMetrics struct {
	Model        string
	Attempt      int
	InputCount   int
	VectorCount  int
	Duration     time.Duration
	PromptTokens int
	TotalTokens  int
}

type Config struct {
	BaseURL    string
	Model      string
	APIKey     string
	RetryCount int
	ProxyURL   string
	NoProxy    bool
}

// NewClient создаёт новый LLM клиент, используя go-openai библиотеку.
func NewClient(cfg Config) (*Client, error) {
	config, httpClient, err := newOpenAIConfig(cfg)
	if err != nil {
		return nil, err
	}

	client := &Client{
		client:     openai.NewClientWithConfig(config),
		httpClient: httpClient,
		model:      cfg.Model,
		retryCount: cfg.RetryCount,
		baseURL:    config.BaseURL,
		apiKey:     cfg.APIKey,
		sleep:      sleepContext,
	}
	return client, nil
}

// NewEmbedder создаёт отдельный embeddings клиент.
func NewEmbedder(cfg Config) (*Embedder, error) {
	config, _, err := newOpenAIConfig(cfg)
	if err != nil {
		return nil, err
	}

	embedder := &Embedder{
		client:     openai.NewClientWithConfig(config),
		model:      cfg.Model,
		retryCount: cfg.RetryCount,
		sleep:      sleepContext,
	}
	return embedder, nil
}

func newOpenAIConfig(cfg Config) (openai.ClientConfig, *http.Client, error) {
	config := openai.DefaultConfig(cfg.APIKey)
	if cfg.BaseURL != "" {
		config.BaseURL = cfg.BaseURL
	}

	httpClient, err := newHTTPClient(cfg.ProxyURL, cfg.NoProxy)
	if err != nil {
		return openai.ClientConfig{}, nil, err
	}
	config.HTTPClient = httpClient

	return config, httpClient, nil
}

func newHTTPClient(proxyValue string, noProxy bool) (*http.Client, error) {
	switch {
	case noProxy:
		return &http.Client{Transport: defaultTransport(nil)}, nil
	case strings.TrimSpace(proxyValue) != "":
		parsed, err := parseProxyURL(proxyValue)
		if err != nil {
			return nil, err
		}
		return &http.Client{Transport: defaultTransport(http.ProxyURL(parsed))}, nil
	default:
		return &http.Client{Transport: defaultTransport(http.ProxyFromEnvironment)}, nil
	}
}

func defaultTransport(proxy func(*http.Request) (*url.URL, error)) *http.Transport {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.Proxy = proxy
	return transport
}

func parseProxyURL(raw string) (*url.URL, error) {
	parsed, err := url.Parse(strings.TrimSpace(raw))
	if err != nil {
		return nil, fmt.Errorf("invalid proxy URL: %w", err)
	}
	if parsed.Scheme == "" || parsed.Host == "" {
		return nil, fmt.Errorf("invalid proxy URL: expected scheme://host[:port]")
	}
	return parsed, nil
}
