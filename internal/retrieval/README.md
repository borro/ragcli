# `internal/retrieval`

## TL;DR

Это общее retrieval-ядро для `rag`: spool source, line walking, vector helpers, файловый индекс и общий формат citations.

См. также [`doc/architecture.md`](../../doc/architecture.md).

## Зона ответственности

- `WalkLines` и `SpoolSource` для построчной обработки и hash-based snapshots;
- файловый persistence layer для индексов;
- vector math и token-overlap helpers;
- общий тип `Manifest` и формат citation append.

## Ключевые entrypoints/types

- `Manifest`
- `Filenames`
- `SpoolSource(...)`
- `WalkLines(...)`
- `PersistIndex(...)`
- `LoadIndex(...)`
- `CleanupExpiredIndexes(...)`
- `EmbedQuery(...)`
- `EmbedTexts(...)`
- `AppendSources(...)`

## Входящие/исходящие зависимости

Входящие:

- `internal/rag`

Исходящие:

- `internal/llm`
- стандартная библиотека

## Основной поток

1. Источник при необходимости спулится во временный файл и одновременно хэшируется.
2. Manifest + items + embeddings пишутся в temp dir.
3. Готовый индекс публикуется атомарным rename.
4. Load path проверяет schema version, TTL и согласованность counts.
5. Vector helpers используются режимами для query/chunk embeddings и scoring.

## Инварианты и ошибки

- Schema mismatch, TTL expiry и corruption имеют отдельные sentinel errors.
- Публикация индекса должна быть атомарной и приватной по правам доступа.
- `AppendSources` дедуплицирует ссылки на источники.
- Vector helpers не знают о конкретном режиме и не содержат prompt logic.

## Что подтверждают тесты

- byte offsets при line walking;
- стабильный hash и spool semantics;
- round-trip persist/load индекса;
- schema/TTL/corruption error handling;
- tokenization, overlap и citation deduplication.

## Куда вносить изменения

- Общая файловая логика retrieval идёт сюда, если она нужна больше чем одному режиму.
- Если меняется manifest schema, обновляйте schema version в потребителях и документацию.
