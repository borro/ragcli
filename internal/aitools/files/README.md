# `internal/aitools/files`

## TL;DR

Пакет содержит файловый домен для `aitools`: line-based readers, JSON-контракты file tools и concrete tools `list_files`, `search_file`, `read_lines`, `read_around`.

См. также [`../README.md`](../README.md) и [`doc/architecture.md`](../../../doc/architecture.md).

## Зона ответственности

- ленивое чтение single-file и multi-file источников через `LineReader`;
- `list_files`, `search_file`, `read_lines`, `read_around`;
- concrete tools, реализующие `aitools.Tool`;
- file-specific JSON result/error contracts;
- file-specific progress keys и pagination hints для orchestration;
- standalone file helpers для прямого вызова без registry.

## Ключевые entrypoints/types

- `LineReader`
- `NewFileReader(path)`
- `NewCorpusReader(files)`
- `NewStdinReader()`
- `ToolOptions`
- `NewTools(reader, opts)`
- `SearchParams`
- `ListFilesParams`
- `SearchResult`
- `ReadLinesResult`
- `ReadAroundResult`
- `ListFilesResult`
- `ToolError`
- `MarshalJSON(v)`
- `ExecuteTool(call, reader)`
- `ExecuteToolCalls(ctx, toolCalls, path)`

## Входящие/исходящие зависимости

Входящие:

- `internal/tools`
- `internal/aitools`

Исходящие:

- стандартная библиотека
- `go-openai`
- `internal/aitools`
- `internal/localize`

## Основной поток

1. `cachedLineReader` или `multiFileReader` загружает строки при первом обращении к нужному файлу.
2. Поисковые и read-операции возвращают стабильный JSON wire-format.
3. `NewTools` создаёт concrete file tools для single-file или multi-file режима.
4. Каждый tool валидирует `openai.ToolCall`, выполняет локальную операцию и возвращает generic `aitools.ExecuteResult`.
5. `ProgressKeys` кодируют file/line evidence для orchestration, а pagination hints пишутся в `Hints`.

## Инварианты и ошибки

- `auto` search сначала делает literal search, затем token-overlap fallback.
- В multi-file режиме `path` относится к relative display path внутри attached corpus.
- duplicate-call cache хранится на уровне конкретного tool instance.
- File-specific runtime hints и progress semantics не поднимаются в базовый `aitools`.
