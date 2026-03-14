# `internal/rag`

## TL;DR

Пакет реализует retrieval-режим по локальному индексу: строит или переиспользует embeddings index, ищет релевантные чанки и синтезирует ответ с `Sources:`. Тот же пакет отдаёт переиспользуемые helper-ы для `hybrid` и `tools --rag`.

См. также [`doc/architecture.md`](../../doc/architecture.md).

## Зона ответственности

- spool input или hash по корпусу файлов для directory input;
- build/load index с TTL;
- chunking документа или нескольких файлов для retrieval;
- query embedding, retrieval и rerank;
- финальная генерация ответа и дописывание citation block.

## Ключевые entrypoints/types

- `Options`
- `SearchOptions`
- `PreparedSearch`
- `Run(ctx, chat, embedder, source, opts, question, plan)`
- `PrepareSearch(ctx, embedder, source, opts, meter)`
- `Chunk`
- `Index`
- `Manifest` alias на `retrieval.Manifest`

## Входящие/исходящие зависимости

Входящие:

- `internal/app`
- `internal/aitools/rag`
- `internal/hybrid`

Исходящие:

- `internal/input`
- `internal/llm`
- `internal/retrieval`
- `internal/verbose`
- `internal/localize`

## Основной поток

1. Single-file input спулится во временный файл; directory input получает hash по `(relative path, content)`.
2. Пакет пытается загрузить индекс из кэша.
3. При cache miss документ или файлы чанкуются и embed-ятся.
4. Индекс публикуется в файловой системе.
5. Query embed-ится отдельно и сравнивается с embeddings чанков.
6. Лучшие чанки попадают в финальный prompt.
7. Ответ возвращается вместе с секцией `Sources:`.

Shared helper path для `hybrid` и `tools --rag`:

1. `PrepareSearch` заранее строит или загружает индекс.
2. `PreparedSearch.Search(...)` выполняет semantic retrieval по тому же индексу.
3. Concrete tool сам решает, как сериализовать hits в JSON и как ими пользоваться в orchestration loop.

## Инварианты и ошибки

- `rag` отвечает только по найденным evidence chunks.
- Shared helper-ы должны делить с `Run` те же chunking, hash, TTL и scoring semantics.
- Для directory input line numbers и `Sources:` остаются file-local и используют relative paths.
- При слабом retrieval возвращается honest insufficient answer, а не произвольная галлюцинация.
- Индексные файлы должны быть приватными и атомарно опубликованными.
- Источники в `Sources:` дедуплицируются.

## Что подтверждают тесты

- line range сохранение при chunking;
- cache reuse, invalidation и concurrent writer behavior;
- честный insufficient-data ответ;
- наличие citation block;
- приватные права на индексные файлы и schema/TTL validation.

## Куда вносить изменения

- Retrieval pipeline: `rag.go`.
- Общие файловые операции индекса: `internal/retrieval`.
- Если меняется формат `Sources:` или cache semantics, обновляйте общую документацию.
- Локализованные prompts, progress и insufficient-data ответы режима: `i18n/{en,ru}.toml`.
