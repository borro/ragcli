package app

import (
	mapmode "github.com/borro/ragcli/internal/app/map"
	"github.com/borro/ragcli/internal/app/rag"
	"github.com/borro/ragcli/internal/app/tools"
)

type CommonOptions struct {
	InputPath string
	Prompt    string
	Raw       bool
	Verbose   bool
}

type LLMOptions struct {
	APIURL         string
	Model          string
	EmbeddingModel string
	APIKey         string
	RetryCount     int
}

type Command struct {
	Name   string
	Common CommonOptions
	LLM    LLMOptions
	Map    mapmode.Options
	RAG    rag.Options
	Tools  tools.Options
}
