package app

import (
	"github.com/borro/ragcli/internal/app/hybrid"
	mapmode "github.com/borro/ragcli/internal/app/map"
	"github.com/borro/ragcli/internal/app/rag"
	"github.com/borro/ragcli/internal/app/tools"
)

type CommonOptions struct {
	InputPath string
	Prompt    string
	Raw       bool
	Debug     bool
	Verbose   bool
}

type LLMOptions struct {
	APIURL         string
	Model          string
	EmbeddingModel string
	APIKey         string
	RetryCount     int
	ProxyURL       string
	NoProxy        bool
}

type Command struct {
	Name   string
	Common CommonOptions
	LLM    LLMOptions
	Map    mapmode.Options
	RAG    rag.Options
	Hybrid hybrid.Options
	Tools  tools.Options
}
