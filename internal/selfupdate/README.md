# `internal/selfupdate`

## TL;DR

Пакет реализует `ragcli self-update` поверх GitHub Releases `borro/ragcli`: выбирает последний стабильный release, проверяет наличие platform-specific asset и `checksums.txt`, а затем атомарно заменяет текущий бинарь.

См. также [`doc/requirements.md`](../../doc/requirements.md) и [`doc/architecture.md`](../../doc/architecture.md).

## Зона ответственности

- policy для current version, stable release selection и asset matching;
- wiring `github.com/creativeprojects/go-selfupdate` с обязательной checksum validation;
- user-facing сообщения и error mapping для CLI команды `self-update`;
- безопасное применение обновления к текущему исполняемому файлу.

## Ключевые entrypoints/types

- `New(Config) (*Manager, error)`
- `Manager.Run(ctx, currentVersion, opts, meter)`
- `Options`
- `Result`

## Входящие/исходящие зависимости

Входящие:

- `internal/app`

Исходящие:

- `internal/localize`
- `internal/verbose`
- `github.com/creativeprojects/go-selfupdate`

## Инварианты и ошибки

- Обновление поддерживается только для released semantic version, а не для `dev`-сборок.
- Источник release-ов жёстко привязан к `borro/ragcli`.
- Пререлизы и draft-релизы игнорируются.
- `checksums.txt` обязателен; без него update завершается ошибкой.
- `self-update --check` не пишет на диск и не применяет обновление.

## Что подтверждают тесты

- отбрасывание `dev`/empty/non-semver текущих версий;
- выбор последнего stable release и platform-specific asset;
- ошибки при отсутствии asset-а или `checksums.txt`;
- checksum mismatch без модификации текущего бинаря;
- успешную замену файла и отсутствие записи в `--check`.
