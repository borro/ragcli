# Архитектура `ragcli`

## 1. Сводка

`ragcli` построен как CLI-оболочка над набором режимов обработки текста. Архитектурно проект разделён на:

- тонкую точку входа `cmd/ragcli`;
- orchestration-слой `internal/app`;
- mode-пакеты `map`, `rag`, `tools`;
- shared infrastructure: `llm`, `input`, `retrieval`, `aitools`, `verbose`, `localize`, `logging`.

Главный принцип: `internal/app` связывает CLI, input lifecycle, создание клиентов и вывод результата, а domain-specific пайплайны живут в отдельных пакетах режимов.

## 2. Слои и зависимости

```mermaid
flowchart LR
    main["cmd/ragcli"] --> app["internal/app"]
    app --> input["internal/input"]
    app --> llm["internal/llm"]
    app --> verbose["internal/verbose"]
    app --> localize["internal/localize"]
    app --> logging["internal/logging"]
    app --> map["internal/map"]
    app --> rag["internal/rag"]
    app --> tools["internal/tools"]
    rag --> retrieval["internal/retrieval"]
    tools --> aitools["internal/aitools"]
    tools --> aitoolsfiles["internal/aitools/files"]
    llm --> goopenai["go-openai"]
```

### Правила зависимостей

- `cmd/ragcli` не содержит логики кроме вызова `app.Run`.
- `internal/app` знает про все режимы и shared packages; режимы не знают про CLI.
- `internal/map` не зависит от retrieval и tool calling.
- `internal/rag` использует `internal/retrieval` как общее ядро файлового retrieval.
- `internal/tools` оркестрирует tool calling, а общий registry/API делегирует `internal/aitools`, файловый домен — `internal/aitools/files`.

## 3. Жизненный цикл запуска

```mermaid
sequenceDiagram
    participant User
    participant Main as cmd/ragcli
    participant App as internal/app
    participant CLI as urfave/cli
    participant Runtime as commandSession
    participant Mode as mode package

    User->>Main: запуск команды
    Main->>App: Run(args, stdout, stderr, stdin, version)
    App->>App: load .env, detect locale, configure logging
    App->>CLI: build root command + subcommands
    CLI->>Runtime: bind commandInvocation
    Runtime->>Runtime: create context, reporter, plan
    Runtime->>Runtime: open input, create LLM clients
    Runtime->>Mode: execute selected pipeline
    Mode-->>Runtime: result string / error
    Runtime->>App: format output
    App-->>User: stdout result, stderr error/debug/verbose
```

### Ключевые runtime-этапы

1. `app.Run` подгружает `.env`, определяет locale и настраивает `slog`.
2. `newCLI` строит `urfave/cli` дерево из `commandSpec`.
3. `bind*Invocation` собирает `commandInvocation` с нормализованными options.
4. `appRuntime.execute` создаёт `commandSession`, progress plan и сигнал-совместимый context.
5. `withInput` открывает файл или материализует `stdin`.
6. `withChatInput` / `withChatAndEmbeddingInput` создают клиентов и вызывают mode package.
7. `writeResult` форматирует markdown-ответ и печатает его в `stdout`.

## 4. Общие абстракции

### 4.1 `input.Source`

`input.Source` — общий wire-object между orchestration-слоем и режимами:

- `Descriptor` — metadata входа: kind (`stdin` / `file` / `directory`), исходный path и список файлов корпуса.
- `SnapshotPath()` — фактический path до materialized snapshot, который можно переоткрывать в downstream-пайплайнах.
- `Open()` — повторно открывает snapshot как `io.ReadCloser` без хранения живого reader внутри `Source`.
- Методы `InputPath()`, `DisplayName()`, `BackingFiles()` и `IsMultiFile()` скрывают различия между file, directory и `stdin`.
- Поверх concrete type пакет даёт capability-интерфейсы `SourceMeta`, `SnapshotSource` и `FileBackedSource` для mode-пакетов.

Эта абстракция позволяет всем режимам одинаково работать с `--path`, compatibility alias `--file` и `stdin`.

### 4.2 LLM adapters

`internal/llm` даёт два интерфейса:

- `ChatAutoContextRequester` — chat completion + auto context length resolve;
- `EmbeddingRequester` — embeddings API.

Поверх `go-openai` пакет добавляет:

- retry с exponential backoff;
- request/embedding metrics;
- proxy configuration;
- auto-detect контекста модели через LM Studio models endpoint или probe по ошибке.

### 4.3 Progress model

`internal/verbose` задаёт stage-based progress contract:

- `commandSpec.template` описывает stages и slots;
- `verbose.Plan` превращает их в `Meter`;
- конкретные режимы only report progress по ключам стадий, не зная о рендеринге.

Это позволяет тестировать progress отдельно от business logic.

### 4.4 Локализация и логирование

- `internal/localize` собирает зарегистрированные пакетами встроенные TOML-каталоги в общий bundle и даёт `T()` для текстов CLI.
- `internal/logging` конфигурирует `slog` и укорачивает source paths относительно корня проекта.

## 5. Пайплайны режимов

### 5.1 `map`

```mermaid
flowchart TD
    A["input.Source"] --> B["SplitByApproxTokens"]
    B --> C["parallel map calls"]
    C --> D["prepare / filter facts"]
    D --> E["fan-in reduce iterations"]
    E --> F["final answer generation"]
    F --> G["self critique"]
    G --> H["self refine or keep draft"]
    H --> I["stdout result"]
```

Реальный pipeline:

1. `resolveChunkLength` берёт явный `--length` или пытается автоопределить контекст модели.
2. `SplitByApproxTokens` режет текст по строкам с approximate token budget.
3. `runMapParallel` запускает map-запросы к LLM параллельно.
4. Пустые/`SKIP` ответы отбрасываются, оставшиеся факты нормализуются.
5. `fanInReduce` схлопывает результаты батчами до одного блока фактов.
6. Генерируется финальный draft-answer.
7. `selfRefine` делает critique и, если нужно, refine проход.

Сильные стороны:

- независимость от embeddings;
- хороший fit для summary и broad question по очень длинному тексту.

Компромиссы:

- между чанками теряется часть глобального контекста;
- на слабых моделях качество critique/refine может быть неровным.

### 5.2 `rag`

```mermaid
flowchart TD
    A["input.Source"] --> B["SpoolSource + hash"]
    B --> C{"load cached index?"}
    C -- yes --> D["manifest/chunks/embeddings"]
    C -- no --> E["chunk source"]
    E --> F["embed chunks"]
    F --> G["persist index"]
    D --> H["embed query"]
    G --> H
    H --> I["retrieve + rerank"]
    I --> J["select evidence"]
    J --> K["synthesize answer"]
    K --> L["append Sources"]
```

Реальный pipeline:

1. Вход спулится во временный файл и получает hash с учётом параметров индекса.
2. Индекс грузится из кэша или строится заново.
3. Query embed-ится отдельно.
4. Retrieval ранжирует кандидаты по сходству и lexical overlap.
5. Выбираются `final-k` evidence chunks.
6. LLM отвечает только по evidence.
7. В ответ добавляется секция `Sources:`.

Файловые артефакты индекса:

- `manifest.json`
- `chunks.jsonl`
- `embeddings.jsonl`

### 5.3 `tools`

По умолчанию `tools` не строит retrieval index и не загружает файл в prompt целиком. Вместо этого режим оркестрирует диалог модели с локальными file-tools: `list_files`, `search_file`, `read_lines`, `read_around`.

Если передан `--rag`, режим до первого LLM turn строит или загружает тот же локальный embeddings index, что использует `rag`, и добавляет semantic retrieval tool `search_rag`.

Реальный pipeline:

1. Создаётся tools session с `system` и `user` сообщениями.
2. При `--rag` заранее выполняется build/load retrieval index через `internal/rag`.
3. В каждом turn отправляется chat request с tool definitions.
4. Если модель запросила tool calls, `toolLoopState` выполняет их локально.
5. Результаты возвращаются как `role=tool` сообщения в JSON.
6. Loop отслеживает duplicate calls, already-seen строки и уже перечисленные пути, а также no-progress runs.
7. При зацикливании включаются защитные ветки: retry without tools, stop-calls prompt, forced finalization.
8. Режим завершает работу, когда получает содержательный финальный text answer.

Почему `aitools`/`aitools/files`/`aitools/rag` вынесены отдельно:

- общий AI-tool registry и контракты должны переиспользоваться между режимами;
- файловый домен удобно развивать отдельно от orchestration loop и отдельно от будущих не-файловых tools;
- retrieval-domain tools удобно собирать отдельно от file-domain tools, но на том же базовом `aitools.Registry`;
- каждый concrete tool можно подключать в разных сочетаниях через общий `aitools.Registry`.

## 6. Файловые и временные артефакты

| Артефакт | Кто создаёт | Зачем | Lifecycle |
| --- | --- | --- | --- |
| temp file from `stdin` | `internal/input` | дать режимам path + random access | удаляется в `Handle.Close()` |
| spooled source for `rag` | `internal/retrieval` | стабильный hash и повторное чтение | удаляется после завершения run |
| index dir | `internal/rag` | кэш embeddings и metadata | живёт до TTL cleanup |

## 7. Что считать публичным поведением

Публичным поведением проекта считаются:

- CLI-команды, флаги, env-переменные и defaults;
- разделение `stdout`/`stderr`;
- наличие локализации `ru`/`en`;
- semantics режимов `map`, `rag`, `tools`;
- формат источников в ответах `rag`.

Не считаются жёстким public API:

- точные внутренние структуры `commandInvocation`, `pipelineStats`, `toolLoopState`;
- конкретные progress labels;
- конкретные названия внутренних helper functions.

## 8. Куда расширять систему

- Новый режим добавляется через `internal/<mode>` + регистрацию в `internal/app/commandspec.go`.
- Новая общая AI-tool функциональность сначала оценивается на перенос в `internal/aitools` или соответствующий доменный пакет `internal/aitools/<domain>`, а не в отдельный режим.
- Изменения CLI-описания в `internal/app` должны сопровождаться обновлением `doc/requirements.md`, `README.md` и package-level README соответствующих пакетов.
