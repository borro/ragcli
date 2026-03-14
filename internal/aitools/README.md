# `internal/aitools`

## TL;DR

Пакет задаёт общий OpenAI-bound каркас для AI tools: общий `Tool` contract, generic `ExecuteResult`, registry для композиции наборов инструментов, shared `CallDescription` и нейтральные tool-error/logging helpers.

См. также [`doc/architecture.md`](../../doc/architecture.md).

## Зона ответственности

- общий `Tool` interface для AI tools;
- `Registry`, который собирает definitions и dispatch-ит `openai.ToolCall`;
- shared `CallDescription` для structured debug-логов и compact verbose labels;
- generic `ExecuteResult` без domain-specific payload semantics;
- нейтральные tool errors и logging helpers для tool execution.

## Ключевые entrypoints/types

- `Tool`
- `ExecuteResult`
- `CallDescription`
- `Registry`
- `NewRegistry(tools...)`
- `ToolError`
- `AsToolError(err)`
- `NewToolError(code, message, retryable, details)`

## Входящие/исходящие зависимости

Входящие:

- `internal/tools`
- `internal/aitools/files`
- `internal/aitools/rag`
- будущие доменные пакеты `internal/aitools/<domain>`

Исходящие:

- стандартная библиотека
- `go-openai`

## Основной поток

1. Доменные пакеты создают concrete tools, реализующие `Tool`.
2. Вызывающий код собирает нужное сочетание tools через `NewRegistry(tools...)`.
3. Registry отдаёт `ToolDefinition()` для chat request.
4. Для каждого `openai.ToolCall` registry может отдать `CallDescription`: нормализованные аргументы для debug и human-readable label для verbose.
5. Входящий `openai.ToolCall` dispatch-ится по `Name()`.
6. Concrete tool возвращает generic `ExecuteResult`, а orchestration сам решает, как трактовать `ProgressKeys` и `Hints`.

## Инварианты и ошибки

- `aitools` не знает о файловых, retrieval или других domain-specific типах.
- registry не создаёт concrete tools сам, а только композирует переданные экземпляры.
- `CallDescription.Arguments` должен оставаться best-effort summary аргументов, а не полным raw payload по умолчанию.
- `ExecuteResult` должен оставаться generic и пригодным для разных наборов tools.
- unknown tool кодируется через generic `ToolError`.
