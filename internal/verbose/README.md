# `internal/verbose`

## TL;DR

Пакет даёт stage-based progress reporting для пользовательского `--verbose` режима. Он не занимается debug logging и не знает деталей доменной логики режимов.

См. также [`doc/architecture.md`](../../doc/architecture.md).

## Зона ответственности

- контракт `Reporter`;
- безопасный `nop` reporter;
- `Plan` и `Meter` для разметки стадий и прогресса;
- текстовый state reporter в `stderr`.

## Ключевые entrypoints/types

- `Reporter`
- `New(writer, enabled, debug)`
- `Safe(reporter)`
- `StageDef`
- `Plan`
- `Meter`

## Входящие/исходящие зависимости

Входящие:

- `internal/app`
- все mode-пакеты
- пакетные тесты и `internal/verbose/testutil`

Исходящие:

- `internal/localize`
- стандартная библиотека

## Основной поток

1. `commandSpec` определяет stages и slots.
2. `NewPlan` превращает их в именованные `Meter`.
3. Режим вызывает `Start`, `Note`, `Report`, `Done` у соответствующих meter.
4. `stateReporter` печатает одну человекочитаемую строку на каждое новое состояние.

## Инварианты и ошибки

- При выключенном verbose или включённом debug возвращается `nopReporter`.
- Прогресс считается в слотах, а не в процентах режима напрямую.
- `Meter.Slice` позволяет вложенным циклам делить stage на поддиапазоны без знания total plan layout.

## Что подтверждают тесты

- масштабирование progress;
- корректная монотонность и format percent field;
- изоляция progress между отдельными запусками режимов.

## Куда вносить изменения

- Новый формат пользовательского progress — здесь.
- Если меняется шаблон стадий команды, синхронизируйте `internal/app/commandspec.go` и mode logic.
- Локализованные названия stages и fallback-тексты лежат в `i18n/{en,ru}.toml`.
