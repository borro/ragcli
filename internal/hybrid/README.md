# `internal/hybrid`

## TL;DR

Пакет реализует режим `hybrid`: для небольшого single-file/`stdin` input он может сразу положить весь текст в prompt, а для длинных файлов и директорий сначала запрашивает fused semantic seed и synthetic verification history из `internal/ragcore`, после чего передаёт управление общему tool-calling loop.

См. также [`doc/architecture.md`](../../doc/architecture.md).

## Зона ответственности

- orchestration нового CLI-режима `hybrid`;
- direct-context fast path для single-file/`stdin`, когда весь текст укладывается в safe budget окна модели;
- orchestration fused `search_rag` seed и synthetic verification reads до первого LLM turn через `internal/ragcore`;
- добавление hybrid-specific prompt instructions поверх общего tool session;
- финализация ответа с `Sources:` по provenance-aware evidence results сессии, разделённым на `Verified` и `Retrieved`.
- interactive conversation wrapper для direct-context и seeded tools path.

## Ключевые entrypoints/types

- `Options`
- `Run(ctx, chat, embedder, source, opts, prompt, plan)`
- `StartConversation(...)`

## Входящие/исходящие зависимости

Входящие:

- `internal/app`

Исходящие:

- `internal/tools`
- `internal/ragcore`
- `internal/retrieval`
- `internal/localize`

## Основной поток

1. Для single-file/`stdin` пакет пытается оценить, помещается ли весь текст в safe budget окна модели.
2. Если помещается, весь текст передаётся в один direct-context prompt и режим возвращает прямой ответ без retrieval/tools.
3. Если не помещается, пакет подготавливает tools session с включённым `search_rag`.
4. До первого model turn выполняются несколько synthetic `search_rag` вызовов по deterministic query variants.
5. Лучшие fused hit'ы дочитываются через synthetic `read_lines` / `read_around`, и обе preloaded phases добавляются в историю как assistant/tool batches.
6. Общий orchestration loop из `internal/tools` продолжает исследование через file tools и повторные `search_rag`.
7. На retrieval/tool path финальный текст дополняется секцией `Sources:` с `Verified` и `Retrieved` evidence, показанным модели.
8. На direct-context fast path `Sources:` тоже сохраняется для совместимости CLI-контракта, но с одной grouped citation на весь single-file/`stdin` диапазон строк.
9. При `--interaction` direct-context fast path остаётся history-based chat path, а seeded path переиспользует shared `tools.Conversation` и baseline seed history.

## Инварианты

- режим не останавливается только из-за слабого стартового retrieval; после seed всегда может идти tool loop;
- directory input всегда остаётся на retrieval/tool path, даже если synthetic corpus сам по себе небольшой;
- `tools --rag` остаётся отдельным режимом и не меняет своё поведение;
- fused seed и synthetic exact reads выполняются тем же session state, что потом используется в общем loop, но synthetic path не должен загрязнять coverage/cache baseline model-driven tool loop.
