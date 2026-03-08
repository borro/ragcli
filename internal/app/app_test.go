package app

import (
	"bytes"
	"context"
	"strings"
	"testing"
)

func TestRunVersionCommand(t *testing.T) {
	var stdout bytes.Buffer

	exitCode := Run([]string{"version"}, &stdout, bytes.NewBuffer(nil), "v1.2.3")
	if exitCode != 0 {
		t.Fatalf("Run(version) exit code = %d, want 0", exitCode)
	}
	if stdout.String() != "v1.2.3\n" {
		t.Fatalf("stdout = %q, want version output", stdout.String())
	}
}

func TestRunVersionFlag(t *testing.T) {
	var stdout bytes.Buffer

	exitCode := Run([]string{"--version"}, &stdout, bytes.NewBuffer(nil), "v1.2.3")
	if exitCode != 0 {
		t.Fatalf("Run(--version) exit code = %d, want 0", exitCode)
	}
	if stdout.String() != "v1.2.3\n" {
		t.Fatalf("stdout = %q, want version output", stdout.String())
	}
}

func TestRunShortVerboseFlagDoesNotPrintVersion(t *testing.T) {
	var stdout bytes.Buffer

	exitCode := Run([]string{"-v"}, &stdout, bytes.NewBuffer(nil), "v1.2.3")
	if exitCode != 0 {
		t.Fatalf("Run(-v) exit code = %d, want 0", exitCode)
	}
	output := stdout.String()
	if !strings.Contains(output, "COMMANDS:") {
		t.Fatalf("stdout = %q, want root help output", output)
	}
	if !strings.Contains(output, "--verbose, -v") {
		t.Fatalf("stdout = %q, want verbose flag in root help", output)
	}
	if strings.Contains(strings.TrimSpace(output), "v1.2.3") && !strings.Contains(output, "VERSION:") {
		t.Fatalf("stdout = %q, -v must not behave like version command", output)
	}
}

func TestRunRootHelp(t *testing.T) {
	var stdout bytes.Buffer

	exitCode := Run([]string{"--help"}, &stdout, bytes.NewBuffer(nil), "v1.2.3")
	if exitCode != 0 {
		t.Fatalf("Run(--help) exit code = %d, want 0", exitCode)
	}
	output := stdout.String()
	for _, needle := range []string{"map", "rag", "tools", "version", "help, h", "--version", "--file", "--model", "ragcli [global options] [command [command options]]", "v1.2.3"} {
		if !strings.Contains(output, needle) {
			t.Fatalf("stdout missing %q in help output:\n%s", needle, output)
		}
	}
	if strings.Contains(output, "--version, -v") {
		t.Fatalf("stdout = %q, version flag must not have -v alias", output)
	}
}

func TestRunMapHelp(t *testing.T) {
	var stdout bytes.Buffer

	exitCode := Run([]string{"map", "--help"}, &stdout, bytes.NewBuffer(nil), "v1.2.3")
	if exitCode != 0 {
		t.Fatalf("Run(map --help) exit code = %d, want 0", exitCode)
	}
	output := stdout.String()
	for _, needle := range []string{"--concurrency", "--length", "cat document.txt | ragcli map", "--file document.txt"} {
		if !strings.Contains(output, needle) {
			t.Fatalf("stdout missing %q in map help:\n%s", needle, output)
		}
	}
}

func TestRunHelpMapShowsGlobalFlags(t *testing.T) {
	var stdout bytes.Buffer

	exitCode := Run([]string{"help", "map"}, &stdout, bytes.NewBuffer(nil), "v1.2.3")
	if exitCode != 0 {
		t.Fatalf("Run(help map) exit code = %d, want 0", exitCode)
	}
	output := stdout.String()
	for _, needle := range []string{"GLOBAL OPTIONS:", "--file", "--api-url", "--model", "--retry", "--verbose"} {
		if !strings.Contains(output, needle) {
			t.Fatalf("stdout missing %q in help map output:\n%s", needle, output)
		}
	}
}

func TestRunRAGHelp(t *testing.T) {
	var stdout bytes.Buffer

	exitCode := Run([]string{"rag", "--help"}, &stdout, bytes.NewBuffer(nil), "v1.2.3")
	if exitCode != 0 {
		t.Fatalf("Run(rag --help) exit code = %d, want 0", exitCode)
	}
	output := stdout.String()
	for _, needle := range []string{"--embedding-model", "--rag-top-k", "--rag-index-ttl"} {
		if !strings.Contains(output, needle) {
			t.Fatalf("stdout missing %q in rag help:\n%s", needle, output)
		}
	}
}

func TestRunVersionHelp(t *testing.T) {
	var stdout bytes.Buffer

	exitCode := Run([]string{"version", "--help"}, &stdout, bytes.NewBuffer(nil), "v1.2.3")
	if exitCode != 0 {
		t.Fatalf("Run(version --help) exit code = %d, want 0", exitCode)
	}
	output := stdout.String()
	if !strings.Contains(output, "ragcli version") {
		t.Fatalf("stdout = %q, want version usage", output)
	}
	for _, needle := range []string{"GLOBAL OPTIONS:", "--file", "--api-url", "--model", "--retry", "--verbose"} {
		if !strings.Contains(output, needle) {
			t.Fatalf("stdout missing %q in version help:\n%s", needle, output)
		}
	}
}

func TestRunWithoutSubcommandShowsHelp(t *testing.T) {
	var stdout bytes.Buffer

	exitCode := Run(nil, &stdout, bytes.NewBuffer(nil), "v1.2.3")
	if exitCode != 0 {
		t.Fatalf("Run() exit code = %d, want 0", exitCode)
	}
	if !strings.Contains(stdout.String(), "COMMANDS:") {
		t.Fatalf("stdout = %q, want root help output", stdout.String())
	}
}

func TestCLIMapCommandBinding(t *testing.T) {
	captured, _, err := runCLIForTest([]string{"map", "--file", "doc.txt", "--concurrency", "4", "what", "changed?"})
	if err != nil {
		t.Fatalf("runCLIForTest() error = %v", err)
	}
	if captured.Name != "map" {
		t.Fatalf("Name = %q, want map", captured.Name)
	}
	if captured.Common.InputPath != "doc.txt" {
		t.Fatalf("InputPath = %q, want doc.txt", captured.Common.InputPath)
	}
	if captured.Map.Concurrency != 4 {
		t.Fatalf("Concurrency = %d, want 4", captured.Map.Concurrency)
	}
	if captured.Common.Prompt != "what changed?" {
		t.Fatalf("Prompt = %q, want combined prompt", captured.Common.Prompt)
	}
}

func TestCLIGlobalFlagsApplyBeforeSubcommand(t *testing.T) {
	captured, _, err := runCLIForTest([]string{"--file", "doc.txt", "--model", "qwen2.5", "map", "question"})
	if err != nil {
		t.Fatalf("runCLIForTest() error = %v", err)
	}
	if captured.Name != "map" {
		t.Fatalf("Name = %q, want map", captured.Name)
	}
	if captured.Common.InputPath != "doc.txt" {
		t.Fatalf("InputPath = %q, want doc.txt", captured.Common.InputPath)
	}
	if captured.LLM.Model != "qwen2.5" {
		t.Fatalf("Model = %q, want qwen2.5", captured.LLM.Model)
	}
}

func TestCLIRAGCommandBindingFromEnv(t *testing.T) {
	t.Setenv("EMBEDDING_MODEL", "embed-v2")
	t.Setenv("RAG_TOP_K", "9")

	captured, _, err := runCLIForTest([]string{"rag", "question"})
	if err != nil {
		t.Fatalf("runCLIForTest() error = %v", err)
	}
	if captured.RAG.TopK != 9 {
		t.Fatalf("TopK = %d, want 9", captured.RAG.TopK)
	}
	if captured.LLM.EmbeddingModel != "embed-v2" {
		t.Fatalf("EmbeddingModel = %q, want embed-v2", captured.LLM.EmbeddingModel)
	}
}

func TestCLIStopsFlagParsingAfterPromptStarts(t *testing.T) {
	captured, _, err := runCLIForTest([]string{"map", "what", "--not-a-flag", "means", "this?"})
	if err != nil {
		t.Fatalf("runCLIForTest() error = %v", err)
	}
	if captured.Common.Prompt != "what --not-a-flag means this?" {
		t.Fatalf("Prompt = %q, want prompt to preserve flag-like tokens", captured.Common.Prompt)
	}
}

func TestApplyPromptArgCompatibility(t *testing.T) {
	got := applyPromptArgCompatibility([]string{"map", "--file", "doc.txt", "what", "--looks-like-flag"})
	want := []string{"map", "--file", "doc.txt", "--", "what", "--looks-like-flag"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("applyPromptArgCompatibility() = %q, want %q", got, want)
	}
}

func TestApplyPromptArgCompatibilitySkipsUnsupportedCommands(t *testing.T) {
	got := applyPromptArgCompatibility([]string{"version", "--file", "doc.txt"})
	want := []string{"version", "--file", "doc.txt"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("applyPromptArgCompatibility() = %q, want %q", got, want)
	}
}

func TestRunMapMissingPromptShowsHelp(t *testing.T) {
	var stdout bytes.Buffer

	exitCode := Run([]string{"map"}, &stdout, bytes.NewBuffer(nil), "v1.2.3")
	if exitCode != 1 {
		t.Fatalf("Run(map) exit code = %d, want 1", exitCode)
	}
	output := stdout.String()
	for _, needle := range []string{"ragcli map [arguments...]", "--concurrency", "--length"} {
		if !strings.Contains(output, needle) {
			t.Fatalf("stdout missing %q in map usage output:\n%s", needle, output)
		}
	}
}

func TestRunMapMissingInputFile(t *testing.T) {
	var stdout bytes.Buffer

	exitCode := Run([]string{"map", "--file", "missing.txt", "question"}, &stdout, bytes.NewBuffer(nil), "v1.2.3")
	if exitCode != 1 {
		t.Fatalf("Run(map missing file) exit code = %d, want 1", exitCode)
	}
	if stdout.Len() != 0 {
		t.Fatalf("stdout = %q, want empty output", stdout.String())
	}
}

func TestRunMapWithEmptyStdin(t *testing.T) {
	var stdout bytes.Buffer

	exitCode := Run([]string{"map", "question"}, &stdout, bytes.NewBuffer(nil), "v1.2.3")
	if exitCode != 0 {
		t.Fatalf("Run(map empty stdin) exit code = %d, want 0", exitCode)
	}
	if stdout.String() != "Файл пустой\n" {
		t.Fatalf("stdout = %q, want empty-input result", stdout.String())
	}
}

func TestBindRAGCommandNormalizesValues(t *testing.T) {
	captured, _, err := runCLIForTest([]string{
		"rag",
		"--embedding-model", "",
		"--rag-top-k", "2",
		"--rag-final-k", "9",
		"--rag-chunk-size", "10",
		"--rag-chunk-overlap", "2000",
		"--rag-rerank", "unexpected",
		"question",
	})
	if err != nil {
		t.Fatalf("runCLIForTest() error = %v", err)
	}
	if captured.LLM.EmbeddingModel != "text-embedding-3-small" {
		t.Fatalf("EmbeddingModel = %q, want normalized default", captured.LLM.EmbeddingModel)
	}
	if captured.RAG.TopK != 2 {
		t.Fatalf("TopK = %d, want 2", captured.RAG.TopK)
	}
	if captured.RAG.FinalK != 2 {
		t.Fatalf("FinalK = %d, want 2", captured.RAG.FinalK)
	}
	if captured.RAG.ChunkSize != 1000 {
		t.Fatalf("ChunkSize = %d, want 1000", captured.RAG.ChunkSize)
	}
	if captured.RAG.ChunkOverlap != 250 {
		t.Fatalf("ChunkOverlap = %d, want 250", captured.RAG.ChunkOverlap)
	}
	if captured.RAG.Rerank != "heuristic" {
		t.Fatalf("Rerank = %q, want heuristic", captured.RAG.Rerank)
	}
}

func runCLIForTest(args []string) (Command, string, error) {
	var captured Command
	var stdout bytes.Buffer

	root := newCLI(cliConfig{
		stdout:  &stdout,
		version: "test-version",
		execute: func(_ context.Context, cmd Command) error {
			captured = cmd
			return nil
		},
	})
	err := root.Run(context.Background(), append([]string{root.Name}, applyPromptArgCompatibility(args)...))
	return captured, stdout.String(), err
}
