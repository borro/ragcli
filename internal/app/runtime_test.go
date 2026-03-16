package app

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/borro/ragcli/internal/hybrid"
	"github.com/borro/ragcli/internal/input"
	"github.com/borro/ragcli/internal/llm"
	"github.com/borro/ragcli/internal/localize"
	mapmode "github.com/borro/ragcli/internal/map"
	"github.com/borro/ragcli/internal/rag"
	"github.com/borro/ragcli/internal/ragcore"
	toolsmode "github.com/borro/ragcli/internal/tools"
	"github.com/borro/ragcli/internal/verbose"
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

type scriptedInteractiveSession struct {
	answers []string
	errs    []error
	asks    []string
	resets  int
}

func (s *scriptedInteractiveSession) Ask(_ context.Context, prompt string, _ *verbose.Plan) (string, error) {
	s.asks = append(s.asks, prompt)
	index := len(s.asks) - 1
	if index < len(s.errs) && s.errs[index] != nil {
		return "", s.errs[index]
	}
	if index >= len(s.answers) {
		return "", errors.New("unexpected interactive ask")
	}
	return s.answers[index], nil
}

func (s *scriptedInteractiveSession) Reset() {
	s.resets++
}

func (s *scriptedInteractiveSession) FollowUpPlan(verbose.Reporter) *verbose.Plan {
	return nil
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
	}, defaultLLMOptions(), toolsmode.Options{}))
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

func TestRuntimeExecuteToolsWithRAGCreatesEmbedder(t *testing.T) {
	runtime, _, _ := newTestRuntime("retry policy enabled\n")
	chat := &runtimeChatClient{}
	wantErr := errors.New("embedder sentinel")
	embedderCalls := 0
	runtime.newChatClient = func(llm.Config) (llm.ChatAutoContextRequester, error) { return chat, nil }
	runtime.newEmbedder = func(llm.Config) (llm.EmbeddingRequester, error) {
		embedderCalls++
		return nil, wantErr
	}

	err := runtime.execute(context.Background(), testInvocation("tools", CommonOptions{
		Prompt: "Что в файле?",
	}, func() LLMOptions {
		opts := defaultLLMOptions()
		opts.EmbeddingModel = "embed"
		return opts
	}(), toolsmode.Options{
		EnableRAG: true,
		RAG: ragcore.SearchOptions{
			TopK:           8,
			ChunkSize:      1000,
			ChunkOverlap:   200,
			IndexTTL:       time.Hour,
			IndexDir:       t.TempDir(),
			Rerank:         "heuristic",
			EmbeddingModel: "embed",
		},
	}))
	if !errors.Is(err, wantErr) {
		t.Fatalf("runtime.execute(tools --rag) error = %v, want embedder error", err)
	}
	if embedderCalls != 1 {
		t.Fatalf("embedderCalls = %d, want 1", embedderCalls)
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

func TestRuntimeExecuteHybridCreatesEmbedder(t *testing.T) {
	runtime, _, _ := newTestRuntime("alpha\n")
	chat := &runtimeChatClient{}
	wantErr := errors.New("embedder sentinel")
	embedderCalls := 0
	runtime.newChatClient = func(llm.Config) (llm.ChatAutoContextRequester, error) { return chat, nil }
	runtime.newEmbedder = func(llm.Config) (llm.EmbeddingRequester, error) {
		embedderCalls++
		return nil, wantErr
	}

	err := runtime.execute(context.Background(), testInvocation("hybrid", CommonOptions{
		Prompt: "Что в файле?",
	}, func() LLMOptions {
		opts := defaultLLMOptions()
		opts.EmbeddingModel = "embed"
		return opts
	}(), hybrid.Options{
		Search: ragcore.SearchOptions{},
	}))
	if !errors.Is(err, wantErr) {
		t.Fatalf("runtime.execute(hybrid) error = %v, want embedder error", err)
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

func TestRunInteractionSupportsAskResetAndExit(t *testing.T) {
	runtime, stdout, _ := newTestRuntime("")
	var interaction bytes.Buffer
	runtime.openREPL = func(_ input.Source, _ io.Reader, _ io.Writer) (*interactionIO, error) {
		return &interactionIO{
			reader: strings.NewReader("follow up\n/reset\nafter reset\n/exit\n"),
			writer: &interaction,
		}, nil
	}

	handle, err := input.Open(writeRuntimeTempFile(t, "alpha\n"), nil)
	if err != nil {
		t.Fatalf("input.Open() error = %v", err)
	}
	defer func() { _ = handle.Close() }()

	repl := &scriptedInteractiveSession{
		answers: []string{"answer 1", "answer 2"},
	}
	session := &commandSession{
		runtime:  runtime,
		ctx:      context.Background(),
		inv:      testInvocation("map", CommonOptions{}, defaultLLMOptions(), nil),
		reporter: runtime.newReporter(io.Discard, false, false),
		source:   handle.Source(),
	}

	if err := session.runInteraction(repl); err != nil {
		t.Fatalf("runInteraction() error = %v", err)
	}
	if got := stdout.String(); got != "answer 1\nanswer 2\n" {
		t.Fatalf("stdout = %q, want follow-up answers", got)
	}
	if got := repl.asks; len(got) != 2 || got[0] != "follow up" || got[1] != "after reset" {
		t.Fatalf("asks = %#v, want both follow-up prompts", got)
	}
	if repl.resets != 1 {
		t.Fatalf("resets = %d, want 1", repl.resets)
	}
	if !strings.Contains(interaction.String(), localize.T("interaction.banner")) {
		t.Fatalf("interaction output = %q, want banner", interaction.String())
	}
	if !strings.Contains(interaction.String(), localize.T("interaction.reset")) {
		t.Fatalf("interaction output = %q, want reset ack", interaction.String())
	}
}

func TestRuntimeExecuteInteractiveKeepsStdinSnapshotUntilLoopEnds(t *testing.T) {
	runtime, stdout, _ := newTestRuntime("from stdin\n")
	var duringInteractionSnapshot string
	runtime.openREPL = func(source input.Source, _ io.Reader, _ io.Writer) (*interactionIO, error) {
		return &interactionIO{
			reader: strings.NewReader("check\n/exit\n"),
			writer: io.Discard,
		}, nil
	}

	customInteractive := &scriptedInteractiveSession{
		answers: []string{"checked"},
	}

	spec := &commandSpec{
		name: "custom",
		execute: func(session *commandSession) (commandResult, error) {
			source, err := session.ensureInputOpened(session.plan.Stage("prepare"))
			if err != nil {
				return commandResult{}, err
			}
			duringInteractionSnapshot = source.SnapshotPath()
			customInteractive.errs = []error{func() error {
				if _, statErr := os.Stat(duringInteractionSnapshot); statErr != nil {
					return statErr
				}
				return nil
			}()}
			customInteractive.answers = []string{"checked"}
			return commandResult{
				Output:      "initial",
				Interactive: customInteractive,
			}, nil
		},
	}

	err := runtime.execute(context.Background(), commandInvocation{
		spec: spec,
		Common: CommonOptions{
			Prompt:      "question",
			Interaction: true,
		},
		LLM: defaultLLMOptions(),
	})
	if err != nil {
		t.Fatalf("runtime.execute() error = %v", err)
	}
	if got := stdout.String(); got != "initial\nchecked\n" {
		t.Fatalf("stdout = %q, want initial and follow-up answers", got)
	}
	if duringInteractionSnapshot == "" {
		t.Fatal("snapshot path is empty")
	}
	if _, statErr := os.Stat(duringInteractionSnapshot); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatalf("snapshot still exists after interaction: %v", statErr)
	}
}

func TestRuntimeExecuteInteractiveStdinRequiresTTY(t *testing.T) {
	runtime, stdout, _ := newTestRuntime("from stdin\n")
	runtime.openREPL = func(source input.Source, _ io.Reader, _ io.Writer) (*interactionIO, error) {
		if source.Kind() != input.KindStdin {
			t.Fatalf("source.Kind() = %q, want stdin", source.Kind())
		}
		return nil, fmt.Errorf("%s", localize.T("error.interaction.tty_required"))
	}

	spec := &commandSpec{
		name: "custom",
		execute: func(session *commandSession) (commandResult, error) {
			if _, err := session.ensureInputOpened(session.plan.Stage("prepare")); err != nil {
				return commandResult{}, err
			}
			return commandResult{
				Output:      "initial",
				Interactive: &scriptedInteractiveSession{},
			}, nil
		},
	}

	err := runtime.execute(context.Background(), commandInvocation{
		spec: spec,
		Common: CommonOptions{
			Prompt:      "question",
			Interaction: true,
		},
		LLM: defaultLLMOptions(),
	})
	if err == nil || err.Error() != localize.T("error.interaction.tty_required") {
		t.Fatalf("runtime.execute() error = %v, want tty-required error", err)
	}
	if got := stdout.String(); got != "initial\n" {
		t.Fatalf("stdout = %q, want initial answer before tty error", got)
	}
}

func writeRuntimeTempFile(t *testing.T, content string) string {
	t.Helper()
	path := t.TempDir() + "/input.txt"
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}
	return path
}
