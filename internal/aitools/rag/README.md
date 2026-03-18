# `internal/aitools/rag`

## TL;DR

Пакет содержит retrieval-domain AI tool `search_rag`: tool definition, args normalization, typed payload `SearchResult`, progress/evidence metadata и local tool cache поверх `ragcore.PreparedSearch`.

См. также [`../README.md`](../README.md) и [`doc/architecture.md`](../../../doc/architecture.md).

## Зона ответственности

- concrete tool `search_rag`;
- JSON schema tool definition и localized descriptions;
- args parsing, canonical signature и compact verbose labels;
- typed result payload для semantic search pages;
- retrieval-specific progress keys, pagination hints и evidence ranges.

## Ключевые entrypoints/types

- `NewSearchTool(searcher, embedder)`
- `SearchParams`
- `SearchResult`
- `SearchMatch`
- `NormalizeSearchParams(...)`

## Инварианты

- пакет не строит индекс сам и использует только готовый `ragcore.PreparedSearch`;
- tool возвращает typed `aitools.Execution`, а model-facing JSON envelope кодируется выше, в `internal/aitools`;
- pagination и path filter принадлежат tool-domain слою, а scoring/chunking остаются в `ragcore`.
