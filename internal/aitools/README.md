# `internal/aitools`

## TL;DR

Пакет задаёт общий OpenAI-bound runtime для AI tools: общий `Tool` contract, typed `ToolResult`/`Execution`, registry для композиции доменных инструментов, shared `CallDescription`, единый JSON envelope для `role=tool` и нейтральные tool-error/logging helpers.

См. также [`doc/architecture.md`](../../doc/architecture.md).

## Зона ответственности

- общий `Tool` interface для AI tools;
- `Registry`, который собирает definitions и dispatch-ит `openai.ToolCall`;
- shared `CallDescription` для structured debug-логов и compact verbose labels;
- typed `ToolResult`/`Execution` с payload, progress keys, hints и evidence;
- единый JSON envelope `{"result": ..., "meta": ...}` для tool messages;
- нейтральные tool errors и logging helpers для tool execution.

## Ключевые entrypoints/types

- `Tool`
- `ToolResult`
- `Execution`
- `ToolResponseMeta`
- `CallDescription`
- `Registry`
- `NewRegistry(tools...)`
- `EncodeToolResult(execution, meta)`
- `DecodeToolResult(raw, out)`
- `ToolError`
- `AsToolError(err)`
- `NewToolError(code, message, retryable, details)`

## Входящие/исходящие зависимости

Входящие:

- `internal/tools`
- `internal/aitools/files`
- `internal/aitools/rag`
- `internal/aitools/seed`
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
6. Concrete tool возвращает typed `Execution`, где payload уже отделён от progress/evidence/hints.
7. Orchestration добавляет run-specific meta (`already_seen`, `novel_lines`, `progress_hint`) и сериализует общий envelope через `EncodeToolResult`.

## Инварианты и ошибки

- `aitools` не знает о concrete file/retrieval payload types и не декодирует их сам.
- registry не создаёт concrete tools сам, а только композирует переданные экземпляры.
- `CallDescription.Arguments` должен оставаться best-effort summary аргументов, а не полным raw payload по умолчанию.
- envelope для `role=tool` должен оставаться единым для всех доменов: `result` + `meta`, а error-ответы кодируются отдельно.
- unknown tool кодируется через generic `ToolError`.
