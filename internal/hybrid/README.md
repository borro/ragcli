# `internal/hybrid`

## TL;DR

Пакет реализует режим `hybrid`: сначала он строит fused semantic seed по исходному вопросу, затем предзагружает exact reads по strongest hit'ам и только после этого передаёт управление общему tool-calling loop, чтобы модель проверила релевантность, дочитала локальный контекст и при необходимости повторила retrieval.

См. также [`doc/architecture.md`](../../doc/architecture.md).

## Зона ответственности

- orchestration нового CLI-режима `hybrid`;
- подготовка fused `search_rag` seed и synthetic verification reads до первого LLM turn;
- добавление hybrid-specific prompt instructions поверх общего tool session;
- финализация ответа с `Sources:` по provenance-aware evidence results сессии, разделённым на `Verified` и `Retrieved`.

## Ключевые entrypoints/types

- `Options`
- `Run(ctx, chat, embedder, source, opts, prompt, plan)`

## Входящие/исходящие зависимости

Входящие:

- `internal/app`

Исходящие:

- `internal/tools`
- `internal/rag`
- `internal/aitools/rag`
- `internal/retrieval`
- `internal/localize`

## Основной поток

1. Пакет подготавливает tools session с включённым `search_rag`.
2. До первого model turn выполняются несколько synthetic `search_rag` вызовов по deterministic query variants.
3. Лучшие fused hit'ы дочитываются через synthetic `read_lines` / `read_around`, и обе preloaded phases добавляются в историю как assistant/tool batches.
4. Общий orchestration loop из `internal/tools` продолжает исследование через file tools и повторные `search_rag`.
5. Финальный текст дополняется секцией `Sources:` с `Verified` и `Retrieved` evidence, показанным модели.

## Инварианты

- режим не останавливается только из-за слабого стартового retrieval; после seed всегда может идти tool loop;
- `tools --rag` остаётся отдельным режимом и не меняет своё поведение;
- fused seed и synthetic exact reads выполняются тем же session state, что потом используется в общем loop, но synthetic path не должен загрязнять coverage/cache baseline model-driven tool loop.
