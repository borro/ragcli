# `internal/tools/filetools`

## TL;DR

Пакет даёт локальные инструменты для исследования файла или директории: список файлов, поиск, чтение диапазона строк и чтение окна вокруг строки. Он используется в `tools` и частично в `hybrid`.

См. также [`doc/architecture.md`](../../../doc/architecture.md).

## Зона ответственности

- ленивое чтение single-file и multi-file источников через кэшированный line reader;
- `list_files` с pagination по доступным relative paths;
- `search_file` с literal/regex/auto режимами;
- `read_lines` и `read_around`;
- JSON-результаты и JSON-ошибки для tool calling;
- нормализация параметров поиска и pagination.

## Ключевые entrypoints/types

- `LineReader`
- `NewFileReader(path)`
- `NewCorpusReader(files)`
- `NewStdinReader()`
- `SearchParams`
- `SearchResult`
- `ReadLinesResult`
- `ReadAroundResult`
- `ToolError`

## Входящие/исходящие зависимости

Входящие:

- `internal/tools`
- `internal/hybrid`

Исходящие:

- стандартная библиотека
- `go-openai` только для типов tool calls в helper-части

## Основной поток

1. `cachedLineReader` или `multiFileReader` загружает строки при первом обращении к нужному файлу.
2. `Search` выбирает стратегию поиска и возвращает paginated matches.
3. `ListFiles` возвращает доступные relative paths без чтения содержимого файлов.
4. Для directory corpora search может идти по всему corpus или по конкретному `path`.
5. `ReadLines` и `ReadAround` возвращают file-local диапазоны и требуют `path` для multi-file reader.
6. Результаты сериализуются в JSON и передаются вызывающему коду.

## Инварианты и ошибки

- Источник читается в память только один раз на reader instance.
- `auto` search сначала делает literal search, затем token-overlap fallback.
- В multi-file режиме `path` в JSON и line-tracking относится к relative display path внутри root директории.
- Ошибки инструментов кодируются через `ToolError`, чтобы orchestration loop мог принять решение о retry/fail.
- Максимальный лимит поиска ограничен сверху и нормализуется.

## Что подтверждают тесты

- корректность поиска, pagination и контекстных строк;
- pagination списка файлов, corpus-wide search и path-aware `read_lines`/`read_around`;
- корректность `read_lines` и `read_around` на границах диапазонов;
- кэширование загрузки строк между несколькими вызовами;
- стабильный JSON wire-format и error paths.

## Куда вносить изменения

- Добавление нового инструмента или нового JSON-поля делайте здесь до изменения orchestration-логики в `internal/tools`.
- Если меняется semantics line-numbering, обновляйте тесты и package README.
