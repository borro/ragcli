# `internal/aitools/seed`

## TL;DR

Пакет собирает seeded tool history для `hybrid`: deterministic `search_rag` calls, synthetic verification reads и preloaded assistant/tool batches поверх shared `aitools` contracts.

См. также [`../README.md`](../README.md) и [`doc/architecture.md`](../../../doc/architecture.md).

## Зона ответственности

- построение стартового seed bundle для `hybrid`;
- orchestration synthetic `search_rag`, `read_lines`, `read_around`;
- выбор strongest non-overlapping verification hits;
- сбор preloaded assistant/tool history из общего tool-message envelope.

## Ключевые entrypoints/types

- `BuildSeedBundle(ctx, executor, prompt, opts)`
- `SeedBundle`
- `SyntheticToolExecutor`

## Инварианты

- пакет не знает о CLI и не владеет основным tool loop;
- seed logic использует те же tool contracts, что и обычный `tools` runtime;
- fusion/query variant semantics должны оставаться согласованы с `ragcore`.
