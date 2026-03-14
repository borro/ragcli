# `internal/tools`

## TL;DR

Пакет реализует режим `tools`: tool-calling orchestration loop, в котором модель исследует файл или директорию через локальные инструменты и только потом формирует ответ.

См. также [`doc/architecture.md`](../../doc/architecture.md).

## Зона ответственности

- построение system/user prompts режима;
- orchestration loop по turns;
- обработка tool calls и cross-turn guards против зацикливания;
- финализация режима после содержательного text answer.

## Ключевые entrypoints/types

- `Run(ctx, client, source, prompt, plan)`
- `NewToolsConfig(prompt, source, definitions)`
- `toolLoopState`
- `orchestrationError`

## Входящие/исходящие зависимости

Входящие:

- `internal/app`

Исходящие:

- `internal/input`
- `internal/llm`
- `internal/aitools`
- `internal/aitools/files`
- `internal/verbose`
- `internal/localize`

## Основной поток

1. Создаются file-domain tools через `aitools/files`, затем из них собирается registry в `aitools`.
2. В каждом turn модель получает chat request с доступными инструментами.
3. Запрошенные tool calls выполняются локально через registry из `aitools`.
4. Результаты сериализуются в JSON и возвращаются модели как `role=tool`.
5. Loop отслеживает cache hits, duplicate calls и отсутствие прогресса.
6. При зацикливании применяются stop/retry/forced-finalization guards.
7. Возвращается финальный текстовый ответ.

## Инварианты и ошибки

- Режим работает только когда `source.SnapshotPath()` непустой; для `stdin` и директорий это обеспечивается `internal/input`.
- JSON — единственный wire-format результатов инструментов.
- Для directory input `list_files` перечисляет доступные relative paths.
- Для directory input `search_file` может искать по всему corpus, а `read_lines`/`read_around` требуют `path`.
- Повторные вызовы тех же инструментов не должны бесконечно расширять context без новых строк.
- Tracking прогресса должен различать одинаковые номера строк в разных файлах.
- Пустой финальный ответ после лимитов превращается в orchestration error.

## Что подтверждают тесты

- single-shot и multi-step tool loop;
- retry при пустом финальном ответе;
- guards на duplicate search/read;
- предупреждение, если backend сразу отвечает без инструментов;
- progress semantics режима.

## Куда вносить изменения

- Orchestration loop и лимиты: `run.go`.
- Добавление нового file tool почти всегда требует и изменения constructors в `internal/aitools/files`.
- При изменении JSON-контрактов обновляйте документацию пакета и общую архитектуру.
- Локализованные prompts и progress режима: `i18n/{en,ru}.toml`.
