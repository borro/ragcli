package tools

import ragruntime "github.com/borro/ragcli/internal/rag"

type Options struct {
	EnableRAG bool
	RAG       ragruntime.SearchOptions
}
