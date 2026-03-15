# `internal/verbose/testutil`

## TL;DR

Test-only helper-пакет для проверки stage-based progress из [`internal/verbose`](../README.md). В production runtime он не участвует.

## Зона ответственности

- лёгкие вспомогательные объекты для unit-тестов progress reporting;
- отсутствие зависимости production-кода от test helpers.

## Ключевые entrypoints/types

- `ProgressRecorder`
- `NewProgressRecorder(total)`

## Входящие/исходящие зависимости

Входящие:

- тесты `internal/tools`, `internal/ragcore`, `internal/map`, `internal/verbose`

Исходящие:

- стандартная библиотека

## Основной поток

1. Тест создаёт recorder с ожидаемым total.
2. Recorder реализует интерфейс progress reporter.
3. Тест читает snapshot и сравнивает последовательность progress updates.

## Инварианты и ошибки

- Пакет не должен импортироваться production-кодом.
- Helpers должны оставаться маленькими и без скрытой бизнес-логики.

## Что подтверждают тесты

- Пакет сам по себе не тестируется отдельно; он используется как инфраструктура для тестов других пакетов.

## Куда вносить изменения

- Helpers, привязанные к контракту `internal/verbose`, держите здесь, а не в общем testutil-каталоге.
