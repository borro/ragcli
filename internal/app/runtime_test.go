package app

import (
	"bytes"
	"context"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/borro/ragcli/internal/llm"
	mapmode "github.com/borro/ragcli/internal/map"
	"github.com/borro/ragcli/internal/rag"
	openai "github.com/sashabaranov/go-openai"
)

type runtimeChatClient struct {
	responses []string
	requests  []openai.ChatCompletionRequest
}

func (c *runtimeChatClient) SendRequestWithMetrics(_ context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, llm.RequestMetrics, error) {
	c.requests = append(c.requests, req)
	index := len(c.requests) - 1
	if index >= len(c.responses) {
		return nil, llm.RequestMetrics{}, errors.New("unexpected chat request")
	}
	return &openai.ChatCompletionResponse{
		Choices: []openai.ChatCompletionChoice{
			{
				Message: openai.ChatCompletionMessage{
					Role:    openai.ChatMessageRoleAssistant,
					Content: c.responses[index],
				},
			},
		},
	}, llm.RequestMetrics{}, nil
}

func (c *runtimeChatClient) ResolveAutoContextLength(context.Context) (int, error) {
	return 0, nil
}

func newTestRuntime(stdin string) (*appRuntime, *bytes.Buffer, *bytes.Buffer) {
	stdout := &bytes.Buffer{}
	stderr := &bytes.Buffer{}
	runtime := newRuntime(stdout, stderr, strings.NewReader(stdin), "test-version", 1, nil, time.Unix(0, 0))
	return runtime, stdout, stderr
}

func testInvocation(name string, common CommonOptions, llmOpts LLMOptions, payload any) commandInvocation {
	spec, ok := commandSpecByName(name)
	if !ok {
		spec = &commandSpec{name: name}
	}

	return commandInvocation{
		spec:    spec,
		Common:  common,
		LLM:     llmOpts,
		payload: payload,
	}
}

func TestRuntimeExecuteMapDoesNotCreateEmbedder(t *testing.T) {
	runtime, stdout, _ := newTestRuntime("alpha\n")
	chat := &runtimeChatClient{responses: []string{"fact", "answer", "NONE"}}
	embedderCalls := 0
	runtime.newChatClient = func(llm.Config) (llm.ChatAutoContextRequester, error) { return chat, nil }
	runtime.newEmbedder = func(llm.Config) (llm.EmbeddingRequester, error) {
		embedderCalls++
		return nil, errors.New("embedder must not be created")
	}

	err := runtime.execute(context.Background(), testInvocation("map", CommonOptions{
		Prompt: "Что в файле?",
	}, defaultLLMOptions(), mapmode.Options{
		ChunkLength:    1000,
		LengthExplicit: true,
		Concurrency:    1,
	}))
	if err != nil {
		t.Fatalf("runtime.execute(map) error = %v", err)
	}
	if embedderCalls != 0 {
		t.Fatalf("embedderCalls = %d, want 0", embedderCalls)
	}
	if got := stdout.String(); got != "answer\n" {
		t.Fatalf("stdout = %q, want map answer", got)
	}
}

func TestRuntimeExecuteMapWithDirectoryInput(t *testing.T) {
	runtime, stdout, _ := newTestRuntime("")
	chat := &runtimeChatClient{responses: []string{"fact", "answer", "NONE"}}
	runtime.newChatClient = func(llm.Config) (llm.ChatAutoContextRequester, error) { return chat, nil }
	runtime.newEmbedder = func(llm.Config) (llm.EmbeddingRequester, error) {
		return nil, errors.New("embedder must not be created")
	}

	dir := t.TempDir()
	if err := os.WriteFile(dir+"/a.txt", []byte("alpha\n"), 0o600); err != nil {
		t.Fatalf("WriteFile(a.txt) error = %v", err)
	}
	if err := os.Mkdir(dir+"/nested", 0o700); err != nil {
		t.Fatalf("Mkdir(nested) error = %v", err)
	}
	if err := os.WriteFile(dir+"/nested/b.txt", []byte("beta\n"), 0o600); err != nil {
		t.Fatalf("WriteFile(b.txt) error = %v", err)
	}

	err := runtime.execute(context.Background(), testInvocation("map", CommonOptions{
		InputPath: dir,
		Prompt:    "Что в директории?",
	}, defaultLLMOptions(), mapmode.Options{
		ChunkLength:    1000,
		LengthExplicit: true,
		Concurrency:    1,
	}))
	if err != nil {
		t.Fatalf("runtime.execute(map dir) error = %v", err)
	}
	if got := stdout.String(); got != "answer\n" {
		t.Fatalf("stdout = %q, want map answer", got)
	}
}

func TestRuntimeExecuteToolsDoesNotCreateEmbedder(t *testing.T) {
	runtime, stdout, _ := newTestRuntime("alpha\n")
	chat := &runtimeChatClient{responses: []string{"Готовый ответ"}}
	embedderCalls := 0
	runtime.newChatClient = func(llm.Config) (llm.ChatAutoContextRequester, error) { return chat, nil }
	runtime.newEmbedder = func(llm.Config) (llm.EmbeddingRequester, error) {
		embedderCalls++
		return nil, errors.New("embedder must not be created")
	}

	err := runtime.execute(context.Background(), testInvocation("tools", CommonOptions{
		Prompt: "Что в файле?",
	}, defaultLLMOptions(), nil))
	if err != nil {
		t.Fatalf("runtime.execute(tools) error = %v", err)
	}
	if embedderCalls != 0 {
		t.Fatalf("embedderCalls = %d, want 0", embedderCalls)
	}
	if got := stdout.String(); got != "Готовый ответ\n" {
		t.Fatalf("stdout = %q, want tools answer", got)
	}
}

func TestRuntimeExecuteRAGCreatesEmbedder(t *testing.T) {
	runtime, _, _ := newTestRuntime("alpha\n")
	chat := &runtimeChatClient{}
	wantErr := errors.New("embedder sentinel")
	embedderCalls := 0
	runtime.newChatClient = func(llm.Config) (llm.ChatAutoContextRequester, error) { return chat, nil }
	runtime.newEmbedder = func(llm.Config) (llm.EmbeddingRequester, error) {
		embedderCalls++
		return nil, wantErr
	}

	err := runtime.execute(context.Background(), testInvocation("rag", CommonOptions{
		Prompt: "Что в файле?",
	}, func() LLMOptions {
		opts := defaultLLMOptions()
		opts.EmbeddingModel = "embed"
		return opts
	}(), rag.Options{}))
	if !errors.Is(err, wantErr) {
		t.Fatalf("runtime.execute(rag) error = %v, want embedder error", err)
	}
	if embedderCalls != 1 {
		t.Fatalf("embedderCalls = %d, want 1", embedderCalls)
	}
}

func TestRuntimeExecuteUnknownCommandReturnsLocalizedError(t *testing.T) {
	runtime, _, _ := newTestRuntime("alpha\n")
	err := runtime.execute(context.Background(), testInvocation("unknown", CommonOptions{
		Prompt: "Что в файле?",
	}, defaultLLMOptions(), nil))
	if err == nil || err.Error() != unknownSubcommandError("unknown").Error() {
		t.Fatalf("runtime.execute(unknown) error = %v, want localized unknown-subcommand error", err)
	}
}
