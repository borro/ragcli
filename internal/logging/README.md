# `internal/logging`

## TL;DR

Пакет настраивает `slog` для runtime debug-логов проекта и укорачивает source paths до относительных путей от корня репозитория.

См. также [`doc/requirements.md`](../../doc/requirements.md).

## Зона ответственности

- `Configure(writer, debug)` для глобального logger setup;
- выбор log level;
- отключение debug-логов при обычном пользовательском запуске;
- rewrite source attribute в относительный путь.

## Ключевые entrypoints/types

- `Configure(writer, debug)`
- `newLogger(...)`
- `replaceSourceAttr(...)`

## Входящие/исходящие зависимости

Входящие:

- `internal/app`

Исходящие:

- `log/slog`
- стандартная библиотека

## Основной поток

1. `Configure` создаёт новый logger.
2. При `debug=false` writer подменяется на `io.Discard`.
3. Handler добавляет source info.
4. Source file path сокращается до пути относительно корня проекта.

## Инварианты и ошибки

- Пакет не печатает пользовательские ошибки: это задача `internal/app`.
- При обычном запуске structured logs полностью скрыты.
- Source path должен быть стабильным и компактным в тестах и логах.

## Что подтверждают тесты

- включение/выключение debug level;
- discard поведения по умолчанию;
- корректное сокращение source path.

## Куда вносить изменения

- Любое изменение формата debug-логов начинайте здесь.
- Если появится новый логгер, он должен сохранить инвариант `stderr only`.
