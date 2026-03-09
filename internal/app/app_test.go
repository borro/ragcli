package app

import (
	"bytes"
	"context"
	"strings"
	"testing"
)

func TestRunVersionCommand(t *testing.T) {
	var stdout bytes.Buffer

	exitCode := Run([]string{"version"}, &stdout, &bytes.Buffer{}, bytes.NewBuffer(nil), "v1.2.3")
	if exitCode != 0 {
		t.Fatalf("Run(version) exit code = %d, want 0", exitCode)
	}
	if stdout.String() != "v1.2.3\n" {
		t.Fatalf("stdout = %q, want version output", stdout.String())
	}
}

func TestRunVersionFlag(t *testing.T) {
	var stdout bytes.Buffer

	exitCode := Run([]string{"--version"}, &stdout, &bytes.Buffer{}, bytes.NewBuffer(nil), "v1.2.3")
	if exitCode != 0 {
		t.Fatalf("Run(--version) exit code = %d, want 0", exitCode)
	}
	if stdout.String() != "v1.2.3\n" {
		t.Fatalf("stdout = %q, want version output", stdout.String())
	}
}

func TestRunShortDebugFlagDoesNotPrintVersion(t *testing.T) {
	var stdout bytes.Buffer

	exitCode := Run([]string{"-d"}, &stdout, &bytes.Buffer{}, bytes.NewBuffer(nil), "v1.2.3")
	if exitCode != 0 {
		t.Fatalf("Run(-d) exit code = %d, want 0", exitCode)
	}
	output := stdout.String()
	if !strings.Contains(output, "COMMANDS:") {
		t.Fatalf("stdout = %q, want root help output", output)
	}
	if !strings.Contains(output, "--debug, -d") {
		t.Fatalf("stdout = %q, want debug flag in root help", output)
	}
	if strings.Contains(strings.TrimSpace(output), "v1.2.3") && !strings.Contains(output, "VERSION:") {
		t.Fatalf("stdout = %q, -d must not behave like version command", output)
	}
}

func TestRunRootHelp(t *testing.T) {
	var stdout bytes.Buffer

	exitCode := Run([]string{"--help"}, &stdout, &bytes.Buffer{}, bytes.NewBuffer(nil), "v1.2.3")
	if exitCode != 0 {
		t.Fatalf("Run(--help) exit code = %d, want 0", exitCode)
	}
	output := stdout.String()
	for _, needle := range []string{"map", "rag", "tools", "version", "help, h", "--version", "--file", "--model", "--raw", "ragcli [global options] [command [command options]]", "v1.2.3"} {
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

	exitCode := Run([]string{"map", "--help"}, &stdout, &bytes.Buffer{}, bytes.NewBuffer(nil), "v1.2.3")
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

	exitCode := Run([]string{"help", "map"}, &stdout, &bytes.Buffer{}, bytes.NewBuffer(nil), "v1.2.3")
	if exitCode != 0 {
		t.Fatalf("Run(help map) exit code = %d, want 0", exitCode)
	}
	output := stdout.String()
	for _, needle := range []string{"GLOBAL OPTIONS:", "--file", "--api-url", "--model", "--retry", "--raw", "--debug"} {
		if !strings.Contains(output, needle) {
			t.Fatalf("stdout missing %q in help map output:\n%s", needle, output)
		}
	}
}

func TestRunRAGHelp(t *testing.T) {
	var stdout bytes.Buffer

	exitCode := Run([]string{"rag", "--help"}, &stdout, &bytes.Buffer{}, bytes.NewBuffer(nil), "v1.2.3")
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

func TestRunHybridHelp(t *testing.T) {
	var stdout bytes.Buffer

	exitCode := Run([]string{"hybrid", "--help"}, &stdout, &bytes.Buffer{}, bytes.NewBuffer(nil), "v1.2.3")
	if exitCode != 0 {
		t.Fatalf("Run(hybrid --help) exit code = %d, want 0", exitCode)
	}
	output := stdout.String()
	for _, needle := range []string{"--hybrid-top-k", "--hybrid-map-k", "--hybrid-fallback", "--embedding-model", "--rag-index-ttl"} {
		if !strings.Contains(output, needle) {
			t.Fatalf("stdout missing %q in hybrid help:\n%s", needle, output)
		}
	}
}

func TestRunVersionHelp(t *testing.T) {
	var stdout bytes.Buffer

	exitCode := Run([]string{"version", "--help"}, &stdout, &bytes.Buffer{}, bytes.NewBuffer(nil), "v1.2.3")
	if exitCode != 0 {
		t.Fatalf("Run(version --help) exit code = %d, want 0", exitCode)
	}
	output := stdout.String()
	if !strings.Contains(output, "ragcli version") {
		t.Fatalf("stdout = %q, want version usage", output)
	}
	for _, needle := range []string{"GLOBAL OPTIONS:", "--file", "--api-url", "--model", "--retry", "--raw", "--debug"} {
		if !strings.Contains(output, needle) {
			t.Fatalf("stdout missing %q in version help:\n%s", needle, output)
		}
	}
}

func TestRunWithoutSubcommandShowsHelp(t *testing.T) {
	var stdout bytes.Buffer

	exitCode := Run(nil, &stdout, &bytes.Buffer{}, bytes.NewBuffer(nil), "v1.2.3")
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
	captured, _, err := runCLIForTest([]string{"--file", "doc.txt", "--model", "qwen2.5", "--raw", "map", "question"})
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
	if !captured.Common.Raw {
		t.Fatalf("Raw = %v, want true", captured.Common.Raw)
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

func TestCLIHybridCommandBinding(t *testing.T) {
	captured, _, err := runCLIForTest([]string{
		"hybrid",
		"--embedding-model", "embed-v2",
		"--hybrid-top-k", "6",
		"--hybrid-final-k", "9",
		"--hybrid-map-k", "7",
		"--hybrid-read-window", "5",
		"--hybrid-fallback", "rag-only",
		"--rag-chunk-size", "1200",
		"question",
	})
	if err != nil {
		t.Fatalf("runCLIForTest() error = %v", err)
	}
	if captured.Name != "hybrid" {
		t.Fatalf("Name = %q, want hybrid", captured.Name)
	}
	if captured.LLM.EmbeddingModel != "embed-v2" {
		t.Fatalf("EmbeddingModel = %q, want embed-v2", captured.LLM.EmbeddingModel)
	}
	if captured.Hybrid.TopK != 6 {
		t.Fatalf("TopK = %d, want 6", captured.Hybrid.TopK)
	}
	if captured.Hybrid.FinalK != 6 {
		t.Fatalf("FinalK = %d, want normalized 6", captured.Hybrid.FinalK)
	}
	if captured.Hybrid.MapK != 6 {
		t.Fatalf("MapK = %d, want normalized 6", captured.Hybrid.MapK)
	}
	if captured.Hybrid.ReadWindow != 5 {
		t.Fatalf("ReadWindow = %d, want 5", captured.Hybrid.ReadWindow)
	}
	if captured.Hybrid.Fallback != "rag-only" {
		t.Fatalf("Fallback = %q, want rag-only", captured.Hybrid.Fallback)
	}
	if captured.Hybrid.ChunkSize != 1200 {
		t.Fatalf("ChunkSize = %d, want 1200", captured.Hybrid.ChunkSize)
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
	var stderr bytes.Buffer

	exitCode := Run([]string{"map"}, &stdout, &stderr, bytes.NewBuffer(nil), "v1.2.3")
	if exitCode != 1 {
		t.Fatalf("Run(map) exit code = %d, want 1", exitCode)
	}
	output := stdout.String()
	for _, needle := range []string{"ragcli map [arguments...]", "--concurrency", "--length"} {
		if !strings.Contains(output, needle) {
			t.Fatalf("stdout missing %q in map usage output:\n%s", needle, output)
		}
	}
	if stderr.String() != "missing required argument: prompt\n" {
		t.Fatalf("stderr = %q, want prompt error", stderr.String())
	}
}

func TestRunMapMissingInputFile(t *testing.T) {
	var stdout bytes.Buffer
	var stderr bytes.Buffer

	exitCode := Run([]string{"map", "--file", "missing.txt", "question"}, &stdout, &stderr, bytes.NewBuffer(nil), "v1.2.3")
	if exitCode != 1 {
		t.Fatalf("Run(map missing file) exit code = %d, want 1", exitCode)
	}
	if stdout.Len() != 0 {
		t.Fatalf("stdout = %q, want empty output", stdout.String())
	}
	if stderr.String() != "failed to open file: open missing.txt: no such file or directory\n" {
		t.Fatalf("stderr = %q, want plain file error", stderr.String())
	}
}

func TestRunMapMissingInputFileDebugLogsToStderr(t *testing.T) {
	var stdout bytes.Buffer
	var stderr bytes.Buffer

	exitCode := Run([]string{"--debug", "map", "--file", "missing.txt", "question"}, &stdout, &stderr, bytes.NewBuffer(nil), "v1.2.3")
	if exitCode != 1 {
		t.Fatalf("Run(debug map missing file) exit code = %d, want 1", exitCode)
	}
	if stdout.Len() != 0 {
		t.Fatalf("stdout = %q, want empty output", stdout.String())
	}
	output := stderr.String()
	if !strings.Contains(output, "level=ERROR") {
		t.Fatalf("stderr = %q, want structured debug log", output)
	}
	if strings.Contains(output, "\nfailed to open file: open missing.txt: no such file or directory\n") {
		t.Fatalf("stderr = %q, want no plain-text duplicate", output)
	}
	if strings.Count(output, "failed to open file: open missing.txt: no such file or directory") != 1 {
		t.Fatalf("stderr = %q, want single fatal error occurrence", output)
	}
}

func TestRunMapInvalidFlagShowsHelpOnStdoutAndErrorOnStderr(t *testing.T) {
	var stdout bytes.Buffer
	var stderr bytes.Buffer

	exitCode := Run([]string{"map", "--unknown", "question"}, &stdout, &stderr, bytes.NewBuffer(nil), "v1.2.3")
	if exitCode != 1 {
		t.Fatalf("Run(map --unknown) exit code = %d, want 1", exitCode)
	}
	if !strings.Contains(stdout.String(), "ragcli map [arguments...]") {
		t.Fatalf("stdout = %q, want map help output", stdout.String())
	}
	if stderr.String() != "flag provided but not defined: -unknown\n" {
		t.Fatalf("stderr = %q, want invalid-flag error", stderr.String())
	}
	if strings.Contains(stdout.String(), "Incorrect Usage:") {
		t.Fatalf("stdout = %q, want no built-in incorrect usage prefix", stdout.String())
	}
}

func TestRunMapWithEmptyStdin(t *testing.T) {
	var stdout bytes.Buffer
	var stderr bytes.Buffer

	exitCode := Run([]string{"map", "question"}, &stdout, &stderr, bytes.NewBuffer(nil), "v1.2.3")
	if exitCode != 0 {
		t.Fatalf("Run(map empty stdin) exit code = %d, want 0", exitCode)
	}
	if stdout.String() != "Файл пустой\n" {
		t.Fatalf("stdout = %q, want empty-input result", stdout.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("stderr = %q, want no stderr output on success", stderr.String())
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
	var stderr bytes.Buffer

	root := newCLI(cliConfig{
		stdout:  &stdout,
		stderr:  &stderr,
		version: "test-version",
		execute: func(_ context.Context, cmd Command) error {
			captured = cmd
			return nil
		},
	})
	err := root.Run(context.Background(), append([]string{root.Name}, applyPromptArgCompatibility(args)...))
	return captured, stdout.String(), err
}
