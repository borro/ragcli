package app

import (
	"bytes"
	"testing"
)

func TestParseArgsVersion(t *testing.T) {
	cmd, err := parseArgs([]string{"--version"})
	if err != nil {
		t.Fatalf("parseArgs() error = %v", err)
	}
	if !cmd.Common.Version {
		t.Fatal("Version = false, want true")
	}
}

func TestParseArgsMapSubcommand(t *testing.T) {
	cmd, err := parseArgs([]string{"map", "--file", "doc.txt", "--concurrency", "4", "what", "changed?"})
	if err != nil {
		t.Fatalf("parseArgs() error = %v", err)
	}
	if cmd.Name != "map" {
		t.Fatalf("Name = %q, want map", cmd.Name)
	}
	if cmd.Common.InputPath != "doc.txt" {
		t.Fatalf("InputPath = %q, want doc.txt", cmd.Common.InputPath)
	}
	if cmd.Map.Concurrency != 4 {
		t.Fatalf("Concurrency = %d, want 4", cmd.Map.Concurrency)
	}
	if cmd.Common.Prompt != "what changed?" {
		t.Fatalf("Prompt = %q, want combined prompt", cmd.Common.Prompt)
	}
}

func TestParseArgsRAGModeSpecificFlags(t *testing.T) {
	cmd, err := parseArgs([]string{"rag", "--rag-top-k", "9", "--embedding-model", "embed-v2", "question"})
	if err != nil {
		t.Fatalf("parseArgs() error = %v", err)
	}
	if cmd.RAG.TopK != 9 {
		t.Fatalf("TopK = %d, want 9", cmd.RAG.TopK)
	}
	if cmd.LLM.EmbeddingModel != "embed-v2" {
		t.Fatalf("EmbeddingModel = %q, want embed-v2", cmd.LLM.EmbeddingModel)
	}
}

func TestParseArgsRejectsUnknownSubcommand(t *testing.T) {
	_, err := parseArgs([]string{"unknown"})
	if err == nil {
		t.Fatal("parseArgs() error = nil, want error")
	}
}

func TestRunVersion(t *testing.T) {
	var stdout bytes.Buffer
	exitCode := Run([]string{"--version"}, &stdout, bytes.NewBuffer(nil), "v1.2.3")
	if exitCode != 0 {
		t.Fatalf("Run(--version) exit code = %d, want 0", exitCode)
	}
	if stdout.String() != "v1.2.3\n" {
		t.Fatalf("stdout = %q, want version output", stdout.String())
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
