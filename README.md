# RAG CLI — Инструмент для работы с LLM через RAG и Map-Reduce

[![CI master](https://github.com/borro/ragcli/actions/workflows/ci.yaml/badge.svg?branch=master)](https://github.com/borro/ragcli/actions/workflows/ci.yaml)
[![Coverage](https://codecov.io/gh/borro/ragcli/graph/badge.svg)](https://codecov.io/gh/borro/ragcli)

Командная строка для обработки больших текстовых файлов с помощью LLM (Large Language Models) тремя режимами: **Map-Reduce**, **RAG retrieval** и **Agentic Tool Calling**.

## Какую проблему решает ragcli

LLM плохо работает с большими файлами “в лоб”:
- файл не помещается в контекст модели целиком;
- ручной поиск по логам/документам занимает много времени;
- при разбиении текста легко потерять важный контекст;
- в CLI-скриптах нужен предсказуемый и автоматизируемый способ получать ответы из текста.

`ragcli` закрывает это тремя режимами:
- `map`: быстро и параллельно обрабатывает большие тексты по чанкам;
- `rag`: строит локальный временный индекс, делает retrieval по embeddings и отвечает только по найденным evidence chunks;
- `tools`: позволяет модели точечно искать нужные места в файле через инструменты (`search_file`, `read_lines`, `read_around`).

## Типовые use cases

1. Анализ больших логов и отчётов  
Вопрос: “Где в логах причины 5xx и какие сервисы затронуты?”  
Режим: `map` для скорости на больших объёмах.

2. Поиск точных фрагментов в документации/спецификации  
Вопрос: “Что сказано про retry policy в разделе N?”  
Режим: `tools`, когда важно найти конкретные строки и ответить по ним.

3. Быстрый Q&A по локальным текстовым данным в терминале  
Вопрос: “Какие ключевые выводы в этом файле?”  
Режим: любой, в зависимости от задачи (скорость vs точечный поиск).

4. Интеграция в CI/скрипты и работу с локальными LLM  
`ragcli` работает с OpenAI-compatible API (Ollama, LM Studio, vLLM), поэтому подходит для автоматизации без UI.

## Что важно понимать

В режиме `rag` используется локальный временный индекс с embeddings и retrieval.
Режим `tools` остаётся agentic file exploration поверх файла (`search_file`/`read_lines`/`read_around`), а `map` использует чанкование и агрегацию ответов.


## Структура проекта

```
ragcli/
├── cmd/
│   └── ragcli/
│       └── main.go           # Точка входа приложения
├── internal/
│   ├── app/
│   │   ├── app.go            # Composition root, запуск CLI и runtime execution path
│   │   ├── cli.go            # Root command, subcommands, binding и CLI validation
│   │   ├── map/              # Map pipeline
│   │   ├── rag/              # RAG pipeline
│   │   └── tools/            # Tools orchestration и file tools
│   ├── input/
│   │   └── input.go          # File/stdin materialization
│   ├── llm/
│   │   └── client.go         # HTTP клиент для LLM API с retry/backoff
│   └── logging/
│       └── logger.go         # Runtime logger setup
├── go.mod
├── go.sum
└── README.md
```

## Зависимости

- `github.com/sashabaranov/go-openai` — Client для OpenAI-compatible API

## Компиляция и запуск

```bash
# Сборка бинарника
go build -o ragcli ./cmd/ragcli

# Проверка локальной версии (без ldflags будет dev)
./ragcli --version
./ragcli version

# Help можно вызывать и через встроенную help-команду
./ragcli help map

# Или запуск напрямую
go run ./cmd/ragcli map --help
```

## CLI

Корневые команды:
- `ragcli map [flags] <prompt>`
- `ragcli rag [flags] <prompt>`
- `ragcli tools [flags] <prompt>`
- `ragcli version`
- `ragcli help [command]`

Поведение root-команды:
- `ragcli` без подкоманды печатает help и завершает процесс успешно
- `ragcli --help` печатает root help
- `ragcli help map` и `ragcli map --help` печатают help одной и той же команды
- `ragcli --version` и `ragcli version` выводят версию
- для `map`, `rag` и `tools` `prompt` можно передавать как обычный позиционный аргумент; если внутри него есть токены вида `--like-this`, `ragcli` сохраняет обратную совместимость и корректно трактует их как часть prompt

Общие флаги:
- `-f, --file` или `INPUT_FILE` — путь к входному файлу, иначе читается `stdin`
- `--api-url` или `LLM_API_URL` — URL LLM API
- `--model` или `LLM_MODEL` — chat-модель
- `--api-key` или `OPENAI_API_KEY` — ключ API
- `-r` или `RETRY` — количество повторных попыток
- `-v, --verbose` или `VERBOSE` — включает debug логирование в `stderr`
- `--version` — печатает версию root-команды; короткого алиаса у version-флага нет

Флаги `map`:
- `-c, --concurrency` или `CONCURRENCY`
- `-l, --length` или `LENGTH`

Флаги `rag`:
- `--embedding-model` или `EMBEDDING_MODEL`
- `--rag-top-k`, `--rag-final-k`
- `--rag-chunk-size`, `--rag-chunk-overlap`
- `--rag-index-ttl`, `--rag-index-dir`
- `--rag-rerank`

## Примеры использования

### `map`

```bash
# Обработка файла с использованием 4 параллельных запросов:
./ragcli map --file document.txt -c 4 "Какие основные выводы?"

# С кастомным размером чанка и API:
./ragcli map --file document.txt \
  --api-url http://localhost:1234/v1 \
  --model my-model \
  --length 8000 \
  --verbose \
  "Какие основные выводы?"

# Через stdin:
cat document.txt | ./ragcli map "Какие основные выводы из этого документа?"

# Prompt с flag-like токенами не требует обязательного `--`
./ragcli map --file document.txt "Что означает --retry в этом документе?"
```

### TOOLS режим (Agentic Tool Calling)

В этом режиме модель сама исследует файл через tools и не получает весь файл целиком в контекст.
Цикл многошаговый: модель может несколько раз вызвать `search_file`, `read_lines` и `read_around`, пока не соберёт достаточно контекста для финального ответа.
`ragcli` дополнительно подсказывает модели, что файл уже подключён через tools и просить пользователя повторно прислать файл нельзя, но это остаётся prompt-only guardrail, а не принудительный вызов инструмента.

```bash
./ragcli tools \
  --file file.txt \
  --api-url http://localhost:1234/v1 \
  "Какие ключевые моменты обсуждаются в разделе про архитектуру?"
```

Модель будет вызывать инструменты `search_file`, `read_lines` и `read_around` для исследования файла.
Все инструменты возвращают JSON, чтобы модель могла опираться на структурированные поля, а не на свободный текст.
Если backend или модель не поддерживают OpenAI-compatible tool calling корректно, `tools` может вернуть обычный текстовый ответ без вызова инструментов. Усиленный prompt снижает число ложных ответов вида "пришлите файл", но не гарантирует использование tools на несовместимых серверах.

### RAG режим (retrieval с локальным индексом)

`rag` строит локальный временный индекс из `--file` или `stdin`, сохраняет chunk metadata и embeddings в tempdir, затем:
- встраивает вопрос embedding моделью;
- находит top-k релевантных чанков по cosine similarity;
- делает lightweight rerank;
- формирует ответ только по evidence chunks и добавляет блок `Sources`.

```bash
./ragcli rag \
  --file file.txt \
  --model qwen2.5 \
  --embedding-model text-embedding-3-small \
  "Что сказано про retry policy?"
```

## Режимы работы

### Map

Файл разбивается на чанки, каждый обрабатывается параллельно через LLM (фаза **Map**), затем результаты объединяются для формирования финального ответа (фаза **Reduce**).

**Плюсы:**
- Быстрее при больших файлах благодаря параллелизму
- Предсказуемое поведение

**Минусы:**
- Может теряться контекст между чанками

### Agentic Tool Calling

Используется механизм Function/Tool Calling. Модель получает доступ к инструментам:
- `search_file(query, mode?, limit?, offset?, context_lines?)` — поиск по файлу. `mode=auto` сначала делает case-insensitive literal поиск, затем fallback по token overlap; `mode=regex` включает regex-поиск. Возвращает JSON с `query`, `requested_mode`, `mode`, `match_count`, `total_matches`, `has_more`, `next_offset`, `matches`
- `read_lines(start_line, end_line)` — чтение диапазона строк. Возвращает JSON с `start_line`, `end_line`, `line_count`, `lines`
- `read_around(line, before?, after?)` — чтение окна строк вокруг найденной строки. Возвращает JSON с `line`, `before`, `after`, `start_line`, `end_line`, `line_count`, `lines`

Модель сама решает, какие инструменты и когда вызывать для формирования ответа, и может делать несколько последовательных tool-call шагов.

**Плюсы:**
- Сохраняется полный контекст файла
- Модель может целенаправленно искать информацию

**Минусы:**
- Медленнее (множественные циклы общения с моделью)
- Требует поддержки tool calling от LLM API
- Prompt-only guardrails не могут гарантировать вызов инструментов, если backend их игнорирует

### RAG Retrieval

Используется локальный временный индекс с chunk metadata и embeddings. Модель не получает весь файл и не исследует его агентно; в контекст попадают только retrieval evidence chunks.

**Плюсы:**
- Честный retrieval pipeline с citations
- Повторные запуски могут переиспользовать индекс до истечения TTL
- Финальный ответ ограничен найденными evidence chunks

**Минусы:**
- Требуется embeddings endpoint
- Retrieval quality зависит от chunking и embedding модели

## Логирование

Весь вывод логов идёт в `stderr`, финальный ответ — в `stdout`.

Уровни:
- `ERROR` — по умолчанию
- `DEBUG` — при `-v/--verbose`

Примеры логов:
```bash
# Успешный запуск с --verbose
INFO  command parsed command=map input_path_set=true verbose=true retry_count=3
INFO  input source ready input_source=document.txt has_path=true
DEBUG command options prepared command=map input_path=document.txt has_prompt=true api_url_set=true model=local-model embedding_model=text-embedding-nomic-embed-text-v1.5
INFO  starting map processing input_source=document.txt
DEBUG map-reduce pipeline started chunk_length=10000 concurrency=4 input_path=document.txt
INFO  processing finished command=map input_source=document.txt result_empty=false
DEBUG writing result to stdout command=map

# Ошибка без --verbose
ERROR failed to prepare input stage=open_input command=map input_path=missing.txt error="open missing.txt: no such file or directory"
```

## Обработка ошибок и retry

При ошибках запрос к LLM или embeddings API автоматически повторяется с экспоненциальной задержкой:
- Попроба 1 → запрос без задержки
- Попроба 2 → задержка 1с
- Попроба 3 → задержка 2с
- Попроба 4 → задержка 4с
- Попроба 5 → задержка 8с

Количество попыток задается флагом `-r`.

## Graceful Shutdown

При нажатии `Ctrl+C` (сигнал SIGINT/SIGTERM):
- Все HTTP запросы отменяются через контекст
- Worker pool останавливается
- Временные файлы от stdin удаляются

## Совместимость API

Утилита работает с любым API, совместимым с OpenAI Chat Completions форматом. Протестировано с:
- Ollama (ollama serve)
- LM Studio
- vLLM
- Другие OpenAI-compatible серверы

## CI/CD

Для разработки используется GitHub Actions с автоматической проверкой:
- Линтинг через `golangci-lint`
- Запуск тестов с покрытием кода
- Сборка бинарников для Linux, macOS и Windows

Детали workflow см. в `.github/workflows/ci.yaml`

### Pre-commit хуки

Локальные проверки перед коммитом запускаются через `lefthook`.

Установка:
```bash
go install github.com/evilmartians/lefthook@latest
curl -sSfL https://golangci-lint.run/install.sh | sh -s -- -b $(go env GOPATH)/bin $(cat .golangci-lint-version)
lefthook install
```

Для репозитория зафиксирована версия `golangci-lint` из `.golangci-lint-version`.

Хук `pre-commit` выполняет проверки в таком порядке:
- `gofmt` для staged `.go` файлов с автоматическим добавлением исправлений в индекс
- `golangci-lint run --timeout=5m ./cmd/... ./internal/...`
- `go test ./...`
- `go test ./... -coverprofile=coverage.out`
- `go tool cover -func=coverage.out`

Эти хуки дублируют базовые проверки CI, но не заменяют GitHub Actions.

### Автоматические релизы

При создании тега вида `vX.Y.Z` автоматически создаётся релиз с бинарниками для всех платформ через [goreleaser](https://goreleaser.com/). Конфигурация релиза зафиксирована в `.goreleaser.yml`.

GoReleaser вшивает тег релиза в бинарь, поэтому `./ragcli --version` и `./ragcli version` для релизных артефактов выводят соответствующую версию, например `v1.0.0`. Для локальной сборки без `ldflags` команда выводит `dev`.

Для создания нового релиза:
```bash
git tag v1.0.0 && git push origin v1.0.0
```

### Локальная сборка

Для локальной сборки бинарника для вашей платформы:
```bash
go build -o ragcli ./cmd/ragcli
./ragcli --version   # dev
./ragcli version     # dev
```

Для проверки release-конфига без публикации можно использовать dry-run:
```bash
goreleaser release --clean --snapshot --skip=publish --config .goreleaser.yml
```

Для кроссплатформенной сборки используйте GoReleaser или соберите отдельно для каждой платформы через переменные `GOOS` и `GOARCH`.
