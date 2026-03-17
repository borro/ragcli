# `internal/app`

## TL;DR

Это orchestration-слой CLI: он собирает command surface, нормализует опции, открывает input, создаёт клиентов, запускает режимы и печатает результат.

См. также [`doc/requirements.md`](../../doc/requirements.md) и [`doc/architecture.md`](../../doc/architecture.md).

## Зона ответственности

- корневой `Run()` и lifecycle одного запуска CLI;
- описание команд, флагов, defaults и progress templates;
- нормализация `commandInvocation`;
- wiring `input`, `llm`, `verbose`, `logging`, `localize`;
- форматирование финального вывода.
- общий interactive post-run loop по `--interaction`.

## Ключевые entrypoints/types

- `Run(args, stdout, stderr, stdin, version) int`
- `commandSpec`, `commandInvocation`
- `CommonOptions`, `LLMOptions`
- `appRuntime`, `commandSession`
- `TemplateFor()` / `NewPlan()`

## Входящие/исходящие зависимости

Входящие:

- `cmd/ragcli`

Исходящие:

- `internal/input`
- `internal/llm`
- `internal/hybrid`
- `internal/tools`
- `internal/rag`
- `internal/ragcore`
- `internal/map`
- `internal/selfupdate`
- `internal/verbose`
- `internal/localize`
- `internal/logging`

## Основной поток

1. `Run` загружает `.env`, определяет locale и настраивает `slog`.
2. `newCLI` строит дерево команд из `commandSpec`.
3. `bind*Invocation` собирают и нормализуют options.
4. `appRuntime.execute` создаёт signal-aware context и progress plan.
5. `commandSession.ensureInputOpened` открывает файл или материализует `stdin` один раз на весь lifecycle запуска.
6. `withChatInput` / `withChatAndEmbeddingInput` создают клиентов и вызывают режим; `hybrid`, `tools --rag` и `rag` идут через ветку с embedder.
7. `self-update` обходит input/LLM lifecycle и делегирует проверку release-ов и замену бинаря в `internal/selfupdate`.
8. `writeResult` рендерит markdown при необходимости и печатает ответ в `stdout`.
9. При `--interaction` runtime после первого ответа запускает общий REPL, читает follow-up вопросы, умеет `/reset` и `/exit`, а при `stdin` переключает follow-up input на controlling TTY.

## Инварианты и ошибки

- Финальный ответ печатается только в `stdout`.
- Пользовательская ошибка без `--debug` печатается строкой в `stderr`.
- Structured logs пишутся в `stderr` только при `--debug`.
- `--verbose` не активируется одновременно с `--debug`.
- При `--interaction` prompt/banner идут в interaction stream, а follow-up ответы остаются в `stdout`.
- Новая команда должна быть добавлена в `commandSpecs()` и получить progress template.

## Что подтверждают тесты

- help/version/self-update и binding флагов;
- precedence глобальных флагов и prompt compatibility;
- выбор chat/embedder/self-update lifecycle по команде, включая `hybrid` и `tools --rag`;
- markdown rendering и разделение stdout/stderr;
- completeness реестра команд.

## Куда вносить изменения

- Новая команда, новый флаг, новый default: `commandspec.go`.
- Изменение runtime lifecycle: `runtime.go`.
- Изменение output/rendering: `output.go`.
- Изменение CLI UX или help: `cli.go`.
- CLI-строки, help и progress-mode labels: `i18n/{en,ru}.toml`.
