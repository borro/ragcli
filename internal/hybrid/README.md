# `internal/hybrid`

## TL;DR

Пакет реализует режим `hybrid`: сначала он предзагружает результат `search_rag` по исходному вопросу, затем передаёт управление общему tool-calling loop, чтобы модель проверила релевантность, дочитала локальный контекст и при необходимости повторила retrieval.

См. также [`doc/architecture.md`](../../doc/architecture.md).

## Зона ответственности

- orchestration нового CLI-режима `hybrid`;
- подготовка seeded `search_rag` до первого LLM turn;
- добавление hybrid-specific prompt instructions поверх общего tool session;
- финализация ответа с `Sources:` по всем подтверждённым evidence results сессии, сгруппированным по файлу и line ranges.

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
2. До первого model turn выполняется synthetic `search_rag` по исходному вопросу.
3. Seeded tool call и его JSON-результат добавляются в историю как assistant/tool pair.
4. Общий orchestration loop из `internal/tools` продолжает исследование через file tools и повторные `search_rag`.
5. Финальный текст дополняется секцией `Sources:` по подтверждённым semantic/search/read evidence results, показанным модели.

## Инварианты

- режим не останавливается только из-за слабого стартового retrieval; после seed всегда может идти tool loop;
- `tools --rag` остаётся отдельным режимом и не меняет своё поведение;
- seed `search_rag` выполняется тем же session state, что потом используется в общем loop, чтобы сохранить cache/progress semantics.
