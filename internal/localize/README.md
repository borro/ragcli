# `internal/localize`

## TL;DR

Пакет отвечает за локализацию CLI: собирает зарегистрированные пакетами встроенные каталоги, определяет язык интерфейса и предоставляет `T()` для получения переведённых строк.

См. также [`doc/requirements.md`](../../doc/requirements.md).

## Зона ответственности

- registry встроенных catalog files, зарегистрированных owning-пакетами;
- определение языка по флагу, env и системной locale;
- нормализация locale-кодов;
- выдача локализованных строк через `T()`.

## Ключевые entrypoints/types

- `MustRegister(owner, fsys, paths...)`
- `SetCurrent(code)`
- `Detect(args)`
- `Normalize(raw)`
- `T(id, data...)`
- `Data`

## Входящие/исходящие зависимости

Входящие:

- `internal/app`
- все пакеты с пользовательскими строками

Исходящие:

- `go-i18n`
- `go-locale`
- package-local `embed` catalogs

## Основной поток

1. Пакеты регистрируют свои `embed.FS` каталоги через `MustRegister` в `init`.
2. `Detect` смотрит `--lang`, затем `RAGCLI_LANG`, затем системную locale.
3. `Normalize` сводит ввод к поддерживаемым `ru`/`en`.
4. `SetCurrent` создаёт и кеширует `Localizer`.
5. `T()` локализует строку по message id и template data.

## Инварианты и ошибки

- `Detect` возвращает `(en, false)` для неподдерживаемых override-значений.
- `internal/app` отклоняет неподдерживаемый явный `--lang`, а для env/system locale использует fallback `en`.
- Каталоги вшиты в бинарь через `embed`, но живут рядом с owning-пакетами.
- Регистрация новых каталогов после первой инициализации bundle запрещена.
- Дубликаты `(locale tag, message id)` между каталогами считаются ошибкой и останавливают сборку bundle.
- При отсутствии текущего localizer `T()` возвращает message id как fallback.

## Что подтверждают тесты

- нормализация locale-кодов;
- precedence между флагом, env и системной locale;
- обработка невалидного override.
- сборка bundle из нескольких package registrations;
- защита от duplicate owners и duplicate message ids.

## Куда вносить изменения

- Новые пользовательские строки добавляйте в каталог owning-пакета и регистрируйте его через `MustRegister`.
- Новую локаль добавляйте только вместе с каталогом, нормализацией и тестами.
