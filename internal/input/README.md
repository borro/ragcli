# `internal/input`

## TL;DR

Пакет унифицирует вход: открыть файл, собрать директорию в corpus или материализовать `stdin` во временный snapshot и вернуть единый `Handle`/`Source`.

См. также [`doc/architecture.md`](../../doc/architecture.md).

## Зона ответственности

- открыть входной path из `--path`/`--file`;
- рекурсивно собрать директорию в список текстовых файлов и synthetic corpus;
- при отсутствии пути сохранить `stdin` во временный файл;
- вернуть immutable `Source` с metadata, `SnapshotPath()` и `Open()`;
- удалить временный файл при cleanup.

## Ключевые entrypoints/types

- `Open(path, stdin)`
- `Handle`
- `Source`
- `SourceMeta` / `SnapshotSource` / `FileBackedSource`

## Входящие/исходящие зависимости

Входящие:

- `internal/app`

Исходящие:

- стандартная библиотека
- `internal/localize`

## Основной поток

1. Если path указывает на файл, пакет валидирует его и возвращает single-file `Source`.
2. Если path указывает на директорию, пакет рекурсивно собирает пригодные text files.
3. Для директории строится synthetic corpus с file-boundary headers и отдельным snapshot path.
4. Если path не задан, читается `stdin` и материализуется во временный файл.
5. `Handle.Close()` удаляет временный файл, если он был создан пакетом.

## Инварианты и ошибки

- Все режимы получают path-aware источник даже при работе через `stdin`.
- Методы `InputPath()`, `SnapshotPath()`, `DisplayName()`, `BackingFiles()` и `Open()` отделяют user-facing metadata от технического snapshot path.
- Для `stdin` `DisplayName()` равен `stdin`, `InputPath()` пустой, а `SnapshotPath()` указывает на temp file.
- Для директорий `InputPath()` остаётся root path, `SnapshotPath()` указывает на synthetic corpus, а `BackingFiles()` хранит actual path + relative display path.
- Из директории включаются только regular text files; dot-directories, symlink-объекты и файлы с non-text MIME пропускаются.
- Ошибки открытия, записи, закрытия и проверки повторного открытия локализуются.

## Что подтверждают тесты

- открытие файлов и обработка отсутствующего файла;
- рекурсивный обход директорий и relative display paths;
- пропуск hidden/binary/symlink entries;
- materialized snapshot для file/stdin/directory и cleanup temp file;
- error paths при create/read/close/reopen temp file;
- безопасный `Close()` на nil/empty handle.

## Куда вносить изменения

- Любая новая политика работы с директориями, `stdin` или temp files должна жить здесь, а не в режимах.
- Если меняется форма `Source`, проверьте все mode-пакеты и `internal/app`.
