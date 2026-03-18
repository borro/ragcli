# `internal/input/testsource`

## TL;DR

Test-only helper-пакет для создания `input.FileBackedSource` doubles и временных snapshot-файлов в unit-тестах.

## Зона ответственности

- общие helper-ы для test doubles поверх [`internal/input`](../README.md);
- единая логика `DisplayName()` для тестовых источников;
- materialize reader content во временный snapshot для тестов.

## Ключевые entrypoints/types

- `DisplayName(kind, inputPath, snapshotPath, files)`
- `WriteTempFile(reader, name)`

## Входящие/исходящие зависимости

Входящие:

- тесты `internal/ragcore`, `internal/rag`, `internal/tools`

Исходящие:

- стандартная библиотека
- `internal/input`

## Инварианты и ошибки

- Пакет предназначен только для тестов и не должен использоваться production-кодом.
- Helpers не должны добавлять свою бизнес-логику поверх контрактов `internal/input`; они только переиспользуют её в тестах.

## Куда вносить изменения

- Общие helper-ы для `input.FileBackedSource` test doubles держите здесь, если они переиспользуются несколькими пакетами.
