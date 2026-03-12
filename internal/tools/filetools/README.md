# `internal/tools/filetools`

## TL;DR

Пакет даёт line-based инструменты для локального исследования файла: поиск, чтение диапазона строк и чтение окна вокруг строки. Он используется в `tools` и частично в `hybrid`.

См. также [`doc/architecture.md`](../../../doc/architecture.md).

## Зона ответственности

- ленивое чтение файла в память через кэшированный line reader;
- `search_file` с literal/regex/auto режимами;
- `read_lines` и `read_around`;
- JSON-результаты и JSON-ошибки для tool calling;
- нормализация параметров поиска и pagination.

## Ключевые entrypoints/types

- `LineReader`
- `NewFileReader(path)`
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

1. `cachedLineReader` загружает строки при первом обращении.
2. `Search` выбирает стратегию поиска и возвращает paginated matches.
3. `ReadLines` возвращает точный диапазон строк.
4. `ReadAround` возвращает окно вокруг целевой строки.
5. Результаты сериализуются в JSON и передаются вызывающему коду.

## Инварианты и ошибки

- Источник читается в память только один раз на reader instance.
- `auto` search сначала делает literal search, затем token-overlap fallback.
- Ошибки инструментов кодируются через `ToolError`, чтобы orchestration loop мог принять решение о retry/fail.
- Максимальный лимит поиска ограничен сверху и нормализуется.

## Что подтверждают тесты

- корректность поиска, pagination и контекстных строк;
- корректность `read_lines` и `read_around` на границах диапазонов;
- кэширование загрузки строк между несколькими вызовами;
- стабильный JSON wire-format и error paths.

## Куда вносить изменения

- Добавление нового инструмента или нового JSON-поля делайте здесь до изменения orchestration-логики в `internal/tools`.
- Если меняется semantics line-numbering, обновляйте тесты и package README.
