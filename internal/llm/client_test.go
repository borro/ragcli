package llm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

const testLMStudioModel = "test-org/test-model"
const testBaseURL = "http://example.invalid/v1"

func instantSleep(context.Context, time.Duration) error {
	return nil
}

type transportResponder func(req *http.Request) (*http.Response, error)

type transportStub struct {
	responder transportResponder
}

func newTransportStub(responder transportResponder) *transportStub {
	return &transportStub{responder: responder}
}

func (s *transportStub) Do(req *http.Request) (*http.Response, error) {
	return s.responder(req)
}

func (s *transportStub) RoundTrip(req *http.Request) (*http.Response, error) {
	return s.responder(req)
}

func (s *transportStub) HTTPClient() *http.Client {
	return &http.Client{Transport: s}
}

func mustNewClient(t *testing.T, baseURL, model, apiKey string, retryCount int) *Client {
	t.Helper()
	client, err := NewClient(Config{
		BaseURL:    baseURL,
		Model:      model,
		APIKey:     apiKey,
		RetryCount: retryCount,
	})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	client.sleep = instantSleep
	return client
}

func mustNewEmbedder(t *testing.T, baseURL, model, apiKey string, retryCount int) *Embedder {
	t.Helper()
	embedder, err := NewEmbedder(Config{
		BaseURL:    baseURL,
		Model:      model,
		APIKey:     apiKey,
		RetryCount: retryCount,
	})
	if err != nil {
		t.Fatalf("NewEmbedder() error = %v", err)
	}
	embedder.sleep = instantSleep
	return embedder
}

func mustStubbedClient(t *testing.T, baseURL, model, apiKey string, retryCount int, responder transportResponder) *Client {
	t.Helper()
	client := mustNewClient(t, baseURL, model, apiKey, retryCount)
	setClientTransportStub(t, client, responder)
	return client
}

func mustStubbedEmbedder(t *testing.T, baseURL, model, apiKey string, retryCount int, responder transportResponder) *Embedder {
	t.Helper()
	embedder := mustNewEmbedder(t, baseURL, model, apiKey, retryCount)
	setEmbedderTransportStub(t, embedder, baseURL, apiKey, responder)
	return embedder
}

func setClientTransportStub(t *testing.T, client *Client, responder transportResponder) {
	t.Helper()
	stub := newTransportStub(responder)
	config := openai.DefaultConfig(client.apiKey)
	if client.baseURL != "" {
		config.BaseURL = client.baseURL
	}
	config.HTTPClient = stub
	client.client = openai.NewClientWithConfig(config)
	client.httpClient = stub.HTTPClient()
}

func setEmbedderTransportStub(t *testing.T, embedder *Embedder, baseURL string, apiKey string, responder transportResponder) {
	t.Helper()
	stub := newTransportStub(responder)
	config := openai.DefaultConfig(apiKey)
	if baseURL != "" {
		config.BaseURL = baseURL
	}
	config.HTTPClient = stub
	embedder.client = openai.NewClientWithConfig(config)
}

func decodeJSONBody[T any](t *testing.T, req *http.Request) T {
	t.Helper()
	var payload T
	if err := json.NewDecoder(req.Body).Decode(&payload); err != nil {
		t.Fatalf("Decode request error = %v", err)
	}
	return payload
}

func jsonHTTPResponse(statusCode int, body string) *http.Response {
	return &http.Response{
		StatusCode: statusCode,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}

func textHTTPResponse(statusCode int, body string) *http.Response {
	return &http.Response{
		StatusCode: statusCode,
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}

func notFoundResponse() *http.Response {
	return textHTTPResponse(http.StatusNotFound, "not found")
}

func jsonResponder(body string) transportResponder {
	return func(_ *http.Request) (*http.Response, error) {
		return jsonHTTPResponse(http.StatusOK, body), nil
	}
}

func errorResponder(err error) transportResponder {
	return func(_ *http.Request) (*http.Response, error) {
		return nil, err
	}
}

func routeResponder(routes map[string]transportResponder) transportResponder {
	return func(req *http.Request) (*http.Response, error) {
		if route, ok := routes[req.URL.Path]; ok {
			return route(req)
		}
		return notFoundResponse(), nil
	}
}

// TestNewClient проверяет корректное создание клиента LLM
func TestNewClient(t *testing.T) {
	tests := []struct {
		name          string
		baseURL       string
		model         string
		apiKey        string
		retryCount    int
		expectedModel string
		expectedRetry int
	}{
		{
			name:          "valid client creation",
			baseURL:       testBaseURL,
			model:         "gpt-4",
			apiKey:        "test-api-key",
			retryCount:    3,
			expectedModel: "gpt-4",
			expectedRetry: 3,
		},
		{
			name:          "empty baseURL uses default",
			baseURL:       "",
			model:         "gpt-3.5-turbo",
			apiKey:        "another-key",
			retryCount:    5,
			expectedModel: "gpt-3.5-turbo",
			expectedRetry: 5,
		},
		{
			name:          "zero retry count",
			baseURL:       "http://test.com",
			model:         "test-model",
			apiKey:        "",
			retryCount:    0,
			expectedModel: "test-model",
			expectedRetry: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := mustNewClient(t, tt.baseURL, tt.model, tt.apiKey, tt.retryCount)

			if client.model != tt.expectedModel {
				t.Errorf("NewClient() model = %q, want %q", client.model, tt.expectedModel)
			}
			if client.retryCount != tt.expectedRetry {
				t.Errorf("NewClient() retryCount = %d, want %d", client.retryCount, tt.expectedRetry)
			}
		})
	}
}

func TestSendRequest_RequestScenarios(t *testing.T) {
	wantErr := errors.New("mock transport failure")
	tests := []struct {
		name      string
		retry     int
		req       openai.ChatCompletionRequest
		wantCalls int
		responder func(t *testing.T, calls *int) transportResponder
	}{
		{
			name:  "injects default model and forwards tools",
			retry: 0,
			req: openai.ChatCompletionRequest{
				Messages: []openai.ChatCompletionMessage{
					{Role: openai.ChatMessageRoleUser, Content: "Hello"},
				},
				Tools: []openai.Tool{
					{
						Type: "function",
						Function: &openai.FunctionDefinition{
							Name: "get_weather",
						},
					},
				},
			},
			wantCalls: 1,
			responder: func(t *testing.T, calls *int) transportResponder {
				t.Helper()
				return func(req *http.Request) (*http.Response, error) {
					(*calls)++
					if req.URL.Path != "/v1/chat/completions" {
						t.Fatalf("URL.Path = %q, want /v1/chat/completions", req.URL.Path)
					}
					payload := decodeJSONBody[openai.ChatCompletionRequest](t, req)
					if payload.Model != "test-model" {
						t.Fatalf("Model = %q, want test-model", payload.Model)
					}
					if len(payload.Messages) != 1 {
						t.Fatalf("len(Messages) = %d, want 1", len(payload.Messages))
					}
					if len(payload.Tools) != 1 {
						t.Fatalf("len(Tools) = %d, want 1", len(payload.Tools))
					}
					return nil, wantErr
				}
			},
		},
		{
			name: "retries until exhausted",
			req: openai.ChatCompletionRequest{
				Messages: []openai.ChatCompletionMessage{
					{Role: openai.ChatMessageRoleUser, Content: "Hello"},
				},
			},
			retry:     2,
			wantCalls: 3,
			responder: func(_ *testing.T, calls *int) transportResponder {
				return func(_ *http.Request) (*http.Response, error) {
					(*calls)++
					return nil, wantErr
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calls := 0
			client := mustStubbedClient(t, testBaseURL, "test-model", "test-key", tt.retry, tt.responder(t, &calls))

			_, err := client.SendRequest(context.Background(), tt.req)
			if !errors.Is(err, wantErr) {
				t.Fatalf("SendRequest() err = %v, want wrapped %v", err, wantErr)
			}
			if calls != tt.wantCalls {
				t.Fatalf("calls = %d, want %d", calls, tt.wantCalls)
			}
		})
	}
}

func TestSendRequest_ContextCancelled(t *testing.T) {
	calls := 0
	client := mustStubbedClient(t, testBaseURL, "test-model", "", 3, func(_ *http.Request) (*http.Response, error) {
		calls++
		return nil, errors.New("transport should not be called")
	})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := client.SendRequest(ctx, openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{Role: "user", Content: "Hello"}},
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("SendRequest() err = %v, want context.Canceled", err)
	}
	if calls != 0 {
		t.Fatalf("transport calls = %d, want 0", calls)
	}
}

func TestSendRequestWithMetrics_UsageAndTPS(t *testing.T) {
	client := mustNewClient(t, "", "test-model", "", 0)
	setClientTransportStub(t, client, func(req *http.Request) (*http.Response, error) {
		payload := decodeJSONBody[openai.ChatCompletionRequest](t, req)
		if payload.Model != "test-model" {
			t.Fatalf("Model = %q, want %q", payload.Model, "test-model")
		}
		return jsonHTTPResponse(http.StatusOK, `{"choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}`), nil
	})

	_, metrics, err := client.SendRequestWithMetrics(context.Background(), openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("SendRequestWithMetrics() error = %v", err)
	}

	if metrics.PromptTokens != 10 || metrics.CompletionTokens != 20 || metrics.TotalTokens != 30 {
		t.Fatalf("metrics token usage = %+v, want 10/20/30", metrics)
	}
	if metrics.TokensPerSecond <= 0 {
		t.Fatalf("TokensPerSecond = %v, want > 0", metrics.TokensPerSecond)
	}
	if metrics.Attempt != 1 {
		t.Fatalf("Attempt = %d, want 1", metrics.Attempt)
	}
}

func TestSendRequestWithMetrics_ZeroTotalTokensNoTPS(t *testing.T) {
	client := mustNewClient(t, "", "test-model", "", 0)
	setClientTransportStub(t, client, func(_ *http.Request) (*http.Response, error) {
		return jsonHTTPResponse(http.StatusOK, `{"choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}`), nil
	})

	_, metrics, err := client.SendRequestWithMetrics(context.Background(), openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("SendRequestWithMetrics() error = %v", err)
	}
	if metrics.TokensPerSecond != 0 {
		t.Fatalf("TokensPerSecond = %v, want 0", metrics.TokensPerSecond)
	}
}

func TestSendRequestWithMetrics_RetryUsesSuccessfulAttemptMetrics(t *testing.T) {
	client := mustNewClient(t, "", "test-model", "", 1)
	attempts := 0
	setClientTransportStub(t, client, func(_ *http.Request) (*http.Response, error) {
		attempts++
		if attempts == 1 {
			return nil, errors.New("temporary failure")
		}
		return jsonHTTPResponse(http.StatusOK, `{"choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{"prompt_tokens":7,"completion_tokens":3,"total_tokens":10}}`), nil
	})

	_, metrics, err := client.SendRequestWithMetrics(context.Background(), openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("SendRequestWithMetrics() error = %v", err)
	}

	if attempts != 2 {
		t.Fatalf("attempts = %d, want 2", attempts)
	}
	if metrics.Attempt != 2 {
		t.Fatalf("Attempt = %d, want 2", metrics.Attempt)
	}
	if metrics.TotalTokens != 10 {
		t.Fatalf("TotalTokens = %d, want 10", metrics.TotalTokens)
	}
}

func TestSendRequestWithMetrics_IncludesLastError(t *testing.T) {
	client := mustNewClient(t, "", "test-model", "", 2)
	wantErr := errors.New("remote 503")
	calls := 0
	setClientTransportStub(t, client, func(_ *http.Request) (*http.Response, error) {
		calls++
		return nil, wantErr
	})

	_, _, err := client.SendRequestWithMetrics(context.Background(), openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Fatal("SendRequestWithMetrics() error = nil, want wrapped error")
	}
	if calls != 3 {
		t.Fatalf("SendRequestWithMetrics() calls = %d, want 3", calls)
	}
	if !errors.Is(err, wantErr) {
		t.Fatalf("SendRequestWithMetrics() err = %v, want wrapped %v", err, wantErr)
	}
	if got := err.Error(); got != "request failed after 3 attempts: remote 503" {
		t.Fatalf("SendRequestWithMetrics() error string = %q, want wrapped message", got)
	}
}

func TestSendRequestWithMetrics_ExplicitModelOverridesDefault(t *testing.T) {
	client := mustNewClient(t, "", "default-model", "", 0)
	setClientTransportStub(t, client, func(req *http.Request) (*http.Response, error) {
		payload := decodeJSONBody[openai.ChatCompletionRequest](t, req)
		if payload.Model != "override-model" {
			t.Fatalf("Model = %q, want override-model", payload.Model)
		}
		return jsonHTTPResponse(http.StatusOK, `{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`), nil
	})

	_, metrics, err := client.SendRequestWithMetrics(context.Background(), openai.ChatCompletionRequest{
		Model:    "override-model",
		Messages: []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("SendRequestWithMetrics() error = %v", err)
	}
	if metrics.Model != "override-model" {
		t.Fatalf("metrics.Model = %q, want override-model", metrics.Model)
	}
}

func TestNewEmbedder(t *testing.T) {
	embedder := mustNewEmbedder(t, testBaseURL, "embed-model", "key", 2)
	if embedder.model != "embed-model" {
		t.Fatalf("model = %q, want embed-model", embedder.model)
	}
	if embedder.retryCount != 2 {
		t.Fatalf("retryCount = %d, want 2", embedder.retryCount)
	}
}

func TestNewClientInvalidProxyURL(t *testing.T) {
	_, err := NewClient(Config{
		BaseURL:  testBaseURL,
		Model:    "test-model",
		ProxyURL: "://bad",
	})
	if err == nil {
		t.Fatal("NewClient() error = nil, want invalid proxy URL error")
	}
	if !strings.Contains(err.Error(), "invalid proxy URL") {
		t.Fatalf("error = %v, want invalid proxy URL", err)
	}
}

func TestNewClientNoProxyDisablesTransportProxy(t *testing.T) {
	client := mustNewClient(t, testBaseURL, "test-model", "", 0)
	withNoProxy, err := NewClient(Config{
		BaseURL: testBaseURL,
		Model:   "test-model",
		NoProxy: true,
		APIKey:  "",
	})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}

	defaultTransport, ok := client.httpClient.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("default transport type = %T", client.httpClient.Transport)
	}
	noProxyTransport, ok := withNoProxy.httpClient.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("no-proxy transport type = %T", withNoProxy.httpClient.Transport)
	}
	if defaultTransport.Proxy == nil {
		t.Fatal("default transport Proxy = nil, want environment-based proxy function")
	}
	if noProxyTransport.Proxy != nil {
		t.Fatal("no-proxy transport Proxy != nil, want direct connections")
	}
}

func TestNewClientProxyURLOverridesEnvironment(t *testing.T) {
	client, err := NewClient(Config{
		BaseURL:  testBaseURL,
		Model:    "test-model",
		ProxyURL: "http://127.0.0.1:8080",
	})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}

	transport, ok := client.httpClient.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("transport type = %T, want *http.Transport", client.httpClient.Transport)
	}
	req := httptest.NewRequestWithContext(context.Background(), http.MethodGet, "https://api.example.com/v1/chat", nil)
	proxyURL, err := transport.Proxy(req)
	if err != nil {
		t.Fatalf("transport.Proxy() error = %v", err)
	}
	if proxyURL == nil || proxyURL.String() != "http://127.0.0.1:8080" {
		t.Fatalf("proxyURL = %v, want explicit proxy", proxyURL)
	}
}

func TestCreateEmbeddingsWithMetrics(t *testing.T) {
	embedder := mustStubbedEmbedder(t, "", "embed-model", "", 0, func(req *http.Request) (*http.Response, error) {
		payload := decodeJSONBody[struct {
			Input []string `json:"input"`
		}](t, req)
		inputs := payload.Input
		if len(inputs) != 2 {
			t.Fatalf("len(inputs) = %d, want 2", len(inputs))
		}
		return jsonHTTPResponse(http.StatusOK, `{"data":[{"embedding":[1,0]},{"embedding":[0,1]}],"usage":{"prompt_tokens":12,"total_tokens":12}}`), nil
	})

	vectors, metrics, err := embedder.CreateEmbeddingsWithMetrics(context.Background(), []string{"alpha", "beta"})
	if err != nil {
		t.Fatalf("CreateEmbeddingsWithMetrics() error = %v", err)
	}
	if len(vectors) != 2 {
		t.Fatalf("len(Vectors) = %d, want 2", len(vectors))
	}
	if metrics.InputCount != 2 || metrics.VectorCount != 2 {
		t.Fatalf("metrics = %+v, want input/vector count 2", metrics)
	}
	if metrics.TotalTokens != 12 {
		t.Fatalf("TotalTokens = %d, want 12", metrics.TotalTokens)
	}
}

func TestResponseHasToolCallsAndToolChoiceLabel(t *testing.T) {
	resp := openai.ChatCompletionResponse{
		Choices: []openai.ChatCompletionChoice{
			{Message: openai.ChatCompletionMessage{Role: "assistant"}},
			{Message: openai.ChatCompletionMessage{Role: "assistant", ToolCalls: []openai.ToolCall{{ID: "1"}}}},
		},
	}
	if !responseHasToolCalls(resp) {
		t.Fatal("responseHasToolCalls() = false, want true")
	}
	if responseHasToolCalls(openai.ChatCompletionResponse{}) {
		t.Fatal("responseHasToolCalls(empty) = true, want false")
	}

	tests := []struct {
		name  string
		input interface{}
		want  string
	}{
		{name: "nil", input: nil, want: "auto"},
		{name: "empty string", input: "", want: "auto"},
		{name: "string", input: "required", want: "required"},
		{name: "non string", input: map[string]any{"type": "function"}, want: "map[string]interface {}"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := toolChoiceLabel(tt.input); got != tt.want {
				t.Fatalf("toolChoiceLabel() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestCreateEmbeddingsWithMetrics_ContextCancelled(t *testing.T) {
	embedder := mustStubbedEmbedder(t, "", "embed-model", "", 2, func(req *http.Request) (*http.Response, error) {
		<-req.Context().Done()
		return nil, req.Context().Err()
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, _, err := embedder.CreateEmbeddingsWithMetrics(ctx, []string{"alpha"})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("CreateEmbeddingsWithMetrics() err = %v, want context.Canceled", err)
	}
}

func TestCreateEmbeddingsWithMetrics_IncludesLastError(t *testing.T) {
	wantErr := errors.New("remote 503")
	calls := 0
	embedder := mustStubbedEmbedder(t, "", "embed-model", "", 2, func(_ *http.Request) (*http.Response, error) {
		calls++
		return nil, wantErr
	})

	_, _, err := embedder.CreateEmbeddingsWithMetrics(context.Background(), []string{"alpha"})
	if err == nil {
		t.Fatal("CreateEmbeddingsWithMetrics() error = nil, want wrapped error")
	}
	if calls != 3 {
		t.Fatalf("CreateEmbeddingsWithMetrics() calls = %d, want 3", calls)
	}
	if !errors.Is(err, wantErr) {
		t.Fatalf("CreateEmbeddingsWithMetrics() err = %v, want wrapped %v", err, wantErr)
	}
	if got := err.Error(); got != "request failed after 3 attempts: remote 503" {
		t.Fatalf("CreateEmbeddingsWithMetrics() error string = %q, want wrapped message", got)
	}
}

func TestRunWithRetry_CancelsDuringBackoff(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	attempts := 0
	sleepCalls := 0
	_, err := runWithRetry(
		ctx,
		2,
		func(ctx context.Context, _ time.Duration) error {
			sleepCalls++
			cancel()
			<-ctx.Done()
			return ctx.Err()
		},
		func(context.Context) (string, error) {
			attempts++
			return "", errors.New("temporary failure")
		},
		retryHooks[string]{},
	)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("runWithRetry() err = %v, want context.Canceled", err)
	}
	if attempts != 1 {
		t.Fatalf("runWithRetry() attempts = %d, want 1", attempts)
	}
	if sleepCalls != 1 {
		t.Fatalf("runWithRetry() sleepCalls = %d, want 1", sleepCalls)
	}
}

func TestRunWithRetry_NoSleepAfterLastAttempt(t *testing.T) {
	attempts := 0
	sleepCalls := 0
	wantErr := errors.New("boom")

	_, err := runWithRetry(
		context.Background(),
		3,
		func(context.Context, time.Duration) error {
			sleepCalls++
			return nil
		},
		func(context.Context) (string, error) {
			attempts++
			return "", wantErr
		},
		retryHooks[string]{},
	)
	if !errors.Is(err, wantErr) {
		t.Fatalf("runWithRetry() err = %v, want wrapped %v", err, wantErr)
	}
	if attempts != 3 {
		t.Fatalf("runWithRetry() attempts = %d, want 3", attempts)
	}
	if sleepCalls != 2 {
		t.Fatalf("runWithRetry() sleepCalls = %d, want 2", sleepCalls)
	}
}

func TestSleepContext(t *testing.T) {
	tests := []struct {
		name    string
		ctx     func() context.Context
		delay   time.Duration
		wantErr error
	}{
		{
			name:  "zero delay with active context",
			ctx:   context.Background,
			delay: 0,
		},
		{
			name: "zero delay with canceled context",
			ctx: func() context.Context {
				ctx, cancel := context.WithCancel(context.Background())
				cancel()
				return ctx
			},
			delay:   0,
			wantErr: context.Canceled,
		},
		{
			name: "positive delay with canceled context",
			ctx: func() context.Context {
				ctx, cancel := context.WithCancel(context.Background())
				cancel()
				return ctx
			},
			delay:   time.Millisecond,
			wantErr: context.Canceled,
		},
		{
			name:  "positive delay with active context",
			ctx:   context.Background,
			delay: time.Millisecond,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := sleepContext(tt.ctx(), tt.delay)
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("sleepContext() err = %v, want %v", err, tt.wantErr)
			}
		})
	}
}

func TestRetryHelpers(t *testing.T) {
	t.Run("retryDelay", func(t *testing.T) {
		tests := []struct {
			attempt int
			want    time.Duration
		}{
			{attempt: 0, want: 0},
			{attempt: 1, want: time.Second},
			{attempt: 3, want: 4 * time.Second},
		}

		for _, tt := range tests {
			if got := retryDelay(tt.attempt); got != tt.want {
				t.Fatalf("retryDelay(%d) = %v, want %v", tt.attempt, got, tt.want)
			}
		}
	})

	t.Run("exhaustedAttemptsError", func(t *testing.T) {
		err := exhaustedAttemptsError(2, nil)
		if got := err.Error(); got != "request failed after 2 attempts" {
			t.Fatalf("exhaustedAttemptsError(nil) = %q, want plain attempts message", got)
		}

		lastErr := errors.New("boom")
		err = exhaustedAttemptsError(3, lastErr)
		if !errors.Is(err, lastErr) {
			t.Fatalf("exhaustedAttemptsError() err = %v, want wrapped %v", err, lastErr)
		}
		if got := err.Error(); got != "request failed after 3 attempts: boom" {
			t.Fatalf("exhaustedAttemptsError() = %q, want wrapped attempts message", got)
		}
	})
}

func TestResolveAutoContextLength_Scenarios(t *testing.T) {
	upstreamErr := errors.New("upstream unavailable")
	tests := []struct {
		name            string
		baseURL         string
		model           string
		buildResponder  func(t *testing.T) (transportResponder, func(*testing.T))
		wantValue       int
		wantErrIs       error
		wantErrContains string
	}{
		{
			name:    "uses path api models endpoint first",
			baseURL: "http://example.invalid/lm/v1",
			model:   testLMStudioModel,
			buildResponder: func(_ *testing.T) (transportResponder, func(*testing.T)) {
				hitsPath := 0
				hitsRoot := 0
				return routeResponder(map[string]transportResponder{
						"/lm/api/v1/models": func(_ *http.Request) (*http.Response, error) {
							hitsPath++
							return jsonHTTPResponse(http.StatusOK, `{"models":[{"key":"`+testLMStudioModel+`","loaded_instances":[{"config":{"context_length":32768}}]}]}`), nil
						},
						"/api/v1/models": func(_ *http.Request) (*http.Response, error) {
							hitsRoot++
							return notFoundResponse(), nil
						},
					}), func(t *testing.T) {
						t.Helper()
						if hitsPath != 1 {
							t.Fatalf("path models hits = %d, want 1", hitsPath)
						}
						if hitsRoot != 0 {
							t.Fatalf("root models hits = %d, want 0", hitsRoot)
						}
					}
			},
			wantValue: 32768,
		},
		{
			name:    "falls back to root api models endpoint",
			baseURL: "http://example.invalid/nested/v1",
			model:   testLMStudioModel,
			buildResponder: func(_ *testing.T) (transportResponder, func(*testing.T)) {
				hitsPath := 0
				hitsRoot := 0
				return routeResponder(map[string]transportResponder{
						"/nested/api/v1/models": func(_ *http.Request) (*http.Response, error) {
							hitsPath++
							return notFoundResponse(), nil
						},
						"/api/v1/models": func(_ *http.Request) (*http.Response, error) {
							hitsRoot++
							return jsonHTTPResponse(http.StatusOK, `{"models":[{"key":"`+testLMStudioModel+`","loaded_instances":[{"config":{"context_length":65536}}]}]}`), nil
						},
					}), func(t *testing.T) {
						t.Helper()
						if hitsPath != 1 {
							t.Fatalf("path models hits = %d, want 1", hitsPath)
						}
						if hitsRoot != 1 {
							t.Fatalf("root models hits = %d, want 1", hitsRoot)
						}
					}
			},
			wantValue: 65536,
		},
		{
			name:    "warms up unloaded model and retries models endpoint",
			baseURL: testBaseURL,
			model:   testLMStudioModel,
			buildResponder: func(t *testing.T) (transportResponder, func(*testing.T)) {
				t.Helper()
				modelsCalls := 0
				warmupCalls := 0
				return routeResponder(map[string]transportResponder{
						"/api/v1/models": func(_ *http.Request) (*http.Response, error) {
							modelsCalls++
							if modelsCalls == 1 {
								return jsonHTTPResponse(http.StatusOK, `{"models":[{"key":"`+testLMStudioModel+`","loaded_instances":[]}]}`), nil
							}
							return jsonHTTPResponse(http.StatusOK, `{"models":[{"key":"`+testLMStudioModel+`","loaded_instances":[{"config":{"context_length":150000}}]}]}`), nil
						},
						"/v1/chat/completions": func(req *http.Request) (*http.Response, error) {
							payload := decodeJSONBody[openai.ChatCompletionRequest](t, req)
							warmupCalls++
							if payload.Model != testLMStudioModel {
								t.Fatalf("warmup req.Model = %q, want %s", payload.Model, testLMStudioModel)
							}
							return jsonHTTPResponse(http.StatusOK, `{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`), nil
						},
					}), func(t *testing.T) {
						t.Helper()
						if warmupCalls != 1 {
							t.Fatalf("warmupCalls = %d, want 1", warmupCalls)
						}
						if modelsCalls != 2 {
							t.Fatalf("modelsCalls = %d, want 2", modelsCalls)
						}
					}
			},
			wantValue: 150000,
		},
		{
			name:    "falls back to probe error parsing",
			baseURL: testBaseURL,
			model:   "test-model",
			buildResponder: func(_ *testing.T) (transportResponder, func(*testing.T)) {
				return routeResponder(map[string]transportResponder{
					"/api/v1/models":       func(_ *http.Request) (*http.Response, error) { return notFoundResponse(), nil },
					"/v1/chat/completions": errorResponder(errors.New(`Cannot truncate prompt with n_keep (1368) >= n_ctx (256)`)),
				}), nil
			},
			wantValue: 256,
		},
		{
			name:    "returns wrapped error when detection fails",
			baseURL: testBaseURL,
			model:   "test-model",
			buildResponder: func(_ *testing.T) (transportResponder, func(*testing.T)) {
				return routeResponder(map[string]transportResponder{
					"/api/v1/models":       jsonResponder(`{"models":[{"key":"test-model","loaded_instances":[]}]}`),
					"/v1/chat/completions": errorResponder(upstreamErr),
				}), nil
			},
			wantErrIs:       upstreamErr,
			wantErrContains: "failed to resolve model context length",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			responder, verify := tt.buildResponder(t)
			client := mustStubbedClient(t, tt.baseURL, tt.model, "", 0, responder)

			value, err := client.ResolveAutoContextLength(context.Background())
			if tt.wantErrContains != "" {
				if err == nil {
					t.Fatal("ResolveAutoContextLength() error = nil, want non-nil")
				}
				if !errors.Is(err, tt.wantErrIs) {
					t.Fatalf("ResolveAutoContextLength() err = %v, want wrapped %v", err, tt.wantErrIs)
				}
				if !strings.Contains(err.Error(), tt.wantErrContains) {
					t.Fatalf("ResolveAutoContextLength() error = %q, want substring %q", err.Error(), tt.wantErrContains)
				}
			} else {
				if err != nil {
					t.Fatalf("ResolveAutoContextLength() error = %v", err)
				}
				if value != tt.wantValue {
					t.Fatalf("ResolveAutoContextLength() value = %d, want %d", value, tt.wantValue)
				}
			}
			if verify != nil {
				verify(t)
			}
		})
	}
}

func TestExtractLMStudioContextLength(t *testing.T) {
	tests := []struct {
		name  string
		model lmStudioModel
		want  int
		ok    bool
	}{
		{
			name: "loaded instance config context length",
			model: lmStudioModel{
				LoadedInstances: []lmStudioLoadedInstance{
					{
						Config: lmStudioLoadedInstanceConfig{
							ContextLength: 32768,
						},
					},
				},
			},
			want: 32768,
			ok:   true,
		},
		{
			name: "skips zero context length and uses next loaded instance",
			model: lmStudioModel{
				LoadedInstances: []lmStudioLoadedInstance{
					{
						Config: lmStudioLoadedInstanceConfig{},
					},
					{
						Config: lmStudioLoadedInstanceConfig{
							ContextLength: 65536,
						},
					},
				},
			},
			want: 65536,
			ok:   true,
		},
		{
			name:  "missing context length",
			model: lmStudioModel{},
			want:  0,
			ok:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := extractLMStudioContextLength(tt.model)
			if ok != tt.ok || got != tt.want {
				t.Fatalf("extractLMStudioContextLength() = (%d, %v), want (%d, %v)", got, ok, tt.want, tt.ok)
			}
		})
	}
}
