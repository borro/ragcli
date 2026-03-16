# `internal/tools`

## TL;DR

Пакет реализует режим `tools` и общий tool-calling orchestration loop, который также переиспользует `internal/hybrid`. В режиме `tools` модель исследует файл или директорию через локальные инструменты и только потом формирует ответ. При `--rag` режим заранее строит retrieval index и добавляет semantic retrieval tool `search_rag`.

См. также [`doc/architecture.md`](../../doc/architecture.md).

## Зона ответственности

- построение system/user prompts режима;
- orchestration loop по turns;
- обработка tool calls и cross-turn guards против зацикливания;
- финализация режима после содержательного text answer.
- stateful conversation wrapper для `--interaction`, включая `/reset`.

## Ключевые entrypoints/types

- `Options`
- `SessionOptions`
- `ToolExecutionOptions`
- `FirstTurnAnswerPolicy`
- `Session`
- `SessionResult`
- `Conversation`
- `Run(ctx, client, embedder, source, opts, prompt, plan)`
- `StartConversation(...)`
- `PrepareSession(ctx, source, embedder, opts, sessionOpts, indexMeter)`
- `NewToolsConfig(prompt, source, opts, definitions, additionalSystemPrompt, additionalUserPrompt)`
- `toolLoopState`
- `orchestrationError`

## Входящие/исходящие зависимости

Входящие:

- `internal/app`
- `internal/hybrid`

Исходящие:

- `internal/input`
- `internal/llm`
- `internal/aitools`
- `internal/aitools/files`
- `internal/ragcore`
- `internal/verbose`
- `internal/localize`

## Основной поток

1. `PrepareSession` создаёт file-domain tools через `aitools/files`.
2. При `opts.EnableRAG` заранее поднимается `internal/ragcore.PreparedSearch` и добавляется tool `search_rag` из того же shared пакета.
3. `Session.Run` в каждом turn отправляет chat request с доступными инструментами.
4. Запрошенные tool calls выполняются локально через registry из `aitools`.
5. Результаты сериализуются в JSON и возвращаются модели как `role=tool`.
6. Verbose progress использует compact label вызова (`search_file(query="...")` и т.п.), а debug-логи получают нормализованный summary аргументов из того же shared описания вызова.
7. Loop отслеживает cache hits, duplicate calls, отсутствие прогресса и накапливает provenance-aware evidence по успешным tool results.
8. При зацикливании применяются stop/retry/forced-finalization guards.
9. Если backend прислал `<tool_call>...</tool_call>` текстом вместо `ToolCalls`, loop пытается распарсить и исполнить такой fallback; битый markup не печатается пользователю как финальный ответ и приводит к retry.
10. `tools.Run` возвращает финальный текстовый ответ; `internal/hybrid` использует тот же evidence tracking для `Sources:`.
11. При `--interaction` `Conversation` хранит transcript snapshot, клонирует `Session`/`toolLoopState` на follow-up turn и откатывает state к baseline по `/reset`.

## Инварианты и ошибки

- Режим работает только когда `source.SnapshotPath()` непустой; для `stdin` и директорий это обеспечивается `internal/input`.
- JSON — единственный wire-format результатов инструментов.
- Для directory input `list_files` перечисляет доступные relative paths.
- Для directory input `search_file` может искать по всему corpus, а `read_lines`/`read_around` требуют `path`.
- При `--rag` semantic retrieval идёт через `search_rag`, но финальный ответ всё равно собирает orchestration loop, а не сам tool.
- Повторные вызовы тех же инструментов не должны бесконечно расширять context без новых строк.
- Verbose status для tool calls должен показывать ключевые параметры вызова, но скрывать пустые и дефолтные значения.
- Tracking прогресса должен различать одинаковые номера строк в разных файлах.
- Evidence tracking должен собирать evidence из `search_rag`, `search_file`, `read_lines` и `read_around`; `list_files` остаётся discovery-инструментом и не попадает в `Sources:`. `search_rag` маркируется как `retrieved`, а `search_file`/`read_lines`/`read_around` как `verified`. Для `search_file` evidence строится по `match.line_number`.
- Пустой финальный ответ после лимитов превращается в orchestration error.

## Что подтверждают тесты

- single-shot и multi-step tool loop;
- optional `search_rag` и pre-index для `--rag`;
- shared session path для preloaded fused seed history из `internal/hybrid`;
- retry при пустом финальном ответе;
- guards на duplicate search/read;
- предупреждение, если backend сразу отвечает без инструментов;
- progress semantics режима.

## Куда вносить изменения

- Orchestration loop и лимиты: `run.go`.
- Добавление нового tool почти всегда требует и изменения constructors в соответствующем доменном пакете или shared runtime.
- При изменении JSON-контрактов обновляйте документацию пакета и общую архитектуру.
- Локализованные prompts и progress режима: `i18n/{en,ru}.toml`.
