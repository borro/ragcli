package tools

import "github.com/borro/ragcli/internal/ragcore"

type Options struct {
	EnableRAG bool
	RAG       ragcore.SearchOptions
}
