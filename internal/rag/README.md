# `internal/rag`

## TL;DR

Пакет реализует standalone-режим `rag`: он поднимает shared search core из [`internal/ragcore`](../ragcore/README.md), выполняет fused semantic seed без tool loop и синтезирует финальный ответ с `Sources:`.

См. также [`doc/architecture.md`](../../doc/architecture.md).

## Зона ответственности

- standalone orchestration режима `rag`;
- запуск fused seed retrieval через `internal/ragcore`;
- выбор evidence chunks для прямого ответа без synthetic verification reads;
- синтез финального ответа и дописывание `Sources:`.

## Ключевые entrypoints/types

- `Run(ctx, chat, embedder, source, opts, question, plan)`

## Входящие/исходящие зависимости

Входящие:

- `internal/app`

Исходящие:

- `internal/ragcore`
- `internal/retrieval`
- `internal/localize`
- `internal/llm`
- `internal/verbose`

## Основной поток

1. `internal/app` создаёт chat + embeddings клиентов и вызывает `rag.Run`.
2. `rag` через `ragcore.PrepareSearch` строит или грузит индекс.
3. `rag` запускает fused semantic seed по нескольким deterministic query variants исходного вопроса.
4. `rag` отбирает evidence chunks и синтезирует прямой ответ без tool loop.
5. В ответ добавляется секция `Sources:`.

## Инварианты и ошибки

- Режим не должен запускать synthetic `read_lines`/`read_around` и tool orchestration.
- Shared index/search/seed/tool logic меняется в `internal/ragcore`, а mode-specific answer generation и finalization остаются здесь.

## Куда вносить изменения

- Standalone pipeline режима: `rag.go`.
- Shared retrieval/runtime logic: `internal/ragcore`.
