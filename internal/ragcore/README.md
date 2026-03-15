# `internal/ragcore`

## TL;DR

Пакет содержит канонический shared RAG runtime для `rag`, `tools --rag` и `hybrid`: build/load локального индекса, semantic search, fused seed retrieval и concrete tool `search_rag`.

См. также [`doc/architecture.md`](../../doc/architecture.md).

## Зона ответственности

- spool input или hash по корпусу файлов для directory input;
- build/load index с TTL;
- chunking документа или нескольких файлов для retrieval;
- query embedding, retrieval и rerank;
- fused multi-query seed retrieval;
- concrete tool `search_rag` и его JSON-контракт;
- helper-ы для synthetic seed/verification history в `hybrid`;
- финальная генерация ответа и дописывание citation block для standalone `rag`.

## Ключевые entrypoints/types

- `RunOptions`
- `SearchOptions`
- `PreparedSearch`
- `PrepareSearch(ctx, embedder, source, opts, meter)`
- `(*PreparedSearch).Search(...)`
- `(*PreparedSearch).FusedSearch(...)`
- `(*PreparedSearch).FusedSearchWithHook(...)`
- `BuildSeedBundle(ctx, executor, prompt, opts)`
- `NewSearchTool(searcher, embedder)`
- `Chunk`
- `Index`
- `SearchResult`
- `Manifest` alias на `retrieval.Manifest`

## Входящие/исходящие зависимости

Входящие:

- `internal/rag`
- `internal/tools`
- `internal/hybrid`
- `internal/app`

Исходящие:

- `internal/input`
- `internal/llm`
- `internal/retrieval`
- `internal/verbose`
- `internal/localize`
- `internal/aitools`
- `internal/aitools/files`

## Инварианты

- shared helper-ы должны делить те же chunking, hash, TTL и scoring semantics между `rag`, `tools --rag` и `hybrid`;
- standalone `rag` использует этот core для fused seed retrieval, но сам отвечает без tool loop и без synthetic exact reads;
- `hybrid` использует те же seed/fusion helper-ы, но остаётся владельцем orchestration;
- `search_rag` сохраняет стабильный JSON wire contract и pagination semantics;
- индексные файлы должны быть приватными и атомарно опубликованными.
