# `internal/tools`

## TL;DR

Пакет реализует режим `tools`: tool-calling orchestration loop, в котором модель исследует файл или директорию через локальные инструменты и только потом формирует ответ. При `--rag` режим заранее строит retrieval index и добавляет semantic retrieval tool `search_rag`.

См. также [`doc/architecture.md`](../../doc/architecture.md).

## Зона ответственности

- построение system/user prompts режима;
- orchestration loop по turns;
- обработка tool calls и cross-turn guards против зацикливания;
- финализация режима после содержательного text answer.

## Ключевые entrypoints/types

- `Options`
- `Run(ctx, client, embedder, source, opts, prompt, plan)`
- `NewToolsConfig(prompt, source, opts, definitions)`
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
- `internal/aitools/rag`
- `internal/rag`
- `internal/verbose`
- `internal/localize`

## Основной поток

1. Создаются file-domain tools через `aitools/files`.
2. При `opts.EnableRAG` заранее поднимается `internal/rag.PreparedSearch` и добавляется tool `search_rag` из `internal/aitools/rag`.
3. В каждом turn модель получает chat request с доступными инструментами.
4. Запрошенные tool calls выполняются локально через registry из `aitools`.
5. Результаты сериализуются в JSON и возвращаются модели как `role=tool`.
6. Loop отслеживает cache hits, duplicate calls и отсутствие прогресса.
7. При зацикливании применяются stop/retry/forced-finalization guards.
8. Возвращается финальный текстовый ответ.

## Инварианты и ошибки

- Режим работает только когда `source.SnapshotPath()` непустой; для `stdin` и директорий это обеспечивается `internal/input`.
- JSON — единственный wire-format результатов инструментов.
- Для directory input `list_files` перечисляет доступные relative paths.
- Для directory input `search_file` может искать по всему corpus, а `read_lines`/`read_around` требуют `path`.
- При `--rag` semantic retrieval идёт через `search_rag`, но финальный ответ всё равно собирает orchestration loop, а не сам tool.
- Повторные вызовы тех же инструментов не должны бесконечно расширять context без новых строк.
- Tracking прогресса должен различать одинаковые номера строк в разных файлах.
- Пустой финальный ответ после лимитов превращается в orchestration error.

## Что подтверждают тесты

- single-shot и multi-step tool loop;
- optional `search_rag` и pre-index для `--rag`;
- retry при пустом финальном ответе;
- guards на duplicate search/read;
- предупреждение, если backend сразу отвечает без инструментов;
- progress semantics режима.

## Куда вносить изменения

- Orchestration loop и лимиты: `run.go`.
- Добавление нового tool почти всегда требует и изменения constructors в соответствующем доменном пакете `internal/aitools/*`.
- При изменении JSON-контрактов обновляйте документацию пакета и общую архитектуру.
- Локализованные prompts и progress режима: `i18n/{en,ru}.toml`.
