# `internal/ragcore`

## TL;DR

Пакет содержит канонический shared retrieval runtime для `rag`, `tools --rag` и `hybrid`: build/load локального индекса, semantic search, query variant generation и fused ranking. Tool-контракты и seeded tool history вынесены в `internal/aitools/*`.

См. также [`doc/architecture.md`](../../doc/architecture.md).

## Зона ответственности

- spool input или hash по корпусу файлов для directory input;
- build/load index с TTL;
- chunking документа или нескольких файлов для retrieval;
- query embedding, retrieval и rerank;
- deterministic query variants для fused retrieval;
- pure fused ranking helpers, которые делят standalone `rag` и seeded tool preprocessing;
- retrieval-facing hit/index types для mode- и tool-layer.

## Ключевые entrypoints/types

- `RunOptions`
- `SearchOptions`
- `PreparedSearch`
- `PrepareSearch(ctx, embedder, source, opts, meter)`
- `(*PreparedSearch).Search(...)`
- `(*PreparedSearch).FusedSearch(...)`
- `(*PreparedSearch).FusedSearchWithHook(...)`
- `SeedQueries(prompt)`
- `FuseSeedMatches(matches, topK)`
- `Chunk`
- `Index`
- `FusedSeedHit`
- `FusedSeedMatch`
- `Manifest` alias на `retrieval.Manifest`

## Входящие/исходящие зависимости

Входящие:

- `internal/rag`
- `internal/tools`
- `internal/hybrid`
- `internal/aitools/rag`
- `internal/aitools/seed`

Исходящие:

- `internal/input`
- `internal/llm`
- `internal/retrieval`
- `internal/verbose`

## Инварианты

- shared helper-ы должны делить те же chunking, hash, TTL и scoring semantics между `rag`, `tools --rag` и `hybrid`;
- `ragcore` не знает о `openai.ToolCall`, tool-message JSON envelope и synthetic history;
- standalone `rag` использует этот core для fused retrieval, но отвечает без tool loop и без synthetic exact reads;
- tool-facing retrieval API живёт выше, в `internal/aitools/rag` и `internal/aitools/seed`;
- индексные файлы должны быть приватными и атомарно опубликованными.
