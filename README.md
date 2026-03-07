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
- `map-reduce`: быстро и параллельно обрабатывает большие тексты по чанкам;
- `rag`: строит локальный временный индекс, делает retrieval по embeddings и отвечает только по найденным evidence chunks;
- `tools`: позволяет модели точечно искать нужные места в файле через инструменты (`search_file`, `read_lines`, `read_around`).

## Типовые use cases

1. Анализ больших логов и отчётов  
Вопрос: “Где в логах причины 5xx и какие сервисы затронуты?”  
Режим: `map-reduce` для скорости на больших объёмах.

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
Режим `tools` остаётся agentic file exploration поверх файла (`search_file`/`read_lines`/`read_around`), а `map-reduce` использует чанкование и агрегацию ответов.


## Структура проекта

```
ragcli/
├── cmd/
│   └── ragcli/
│       └── main.go           # Точка входа приложения
├── internal/
│   ├── config/
│   │   └── config.go         # Конфигурация и логирование
│   ├── llm/
│   │   └── client.go         # HTTP клиент для LLM API с retry/backoff
│   ├── processor/
│   │   ├── mapreduce.go      # Map-Reduce режим обработки
│   │   ├── tools_mode.go     # tools (Agentic Tool Calling) режим
│   │   └── tools.go          # Инструменты для tools режима
│   └── rag/
│       └── rag.go            # Локальный RAG retrieval pipeline
├── go.mod
├── go.sum
└── README.md
```

## Зависимости

- `github.com/sashabaranov/go-openai` — Client для OpenAI-compatible API
- `golang.org/x/sync/errgroup` — Worker pool и оркестрация горутин

## Компиляция и запуск

```bash
# Сборка бинарника
go build -o ragcli ./cmd/ragcli

# Проверка локальной версии (без ldflags будет dev)
./ragcli --version

# Или запуск напрямую
go run ./cmd/ragcli --help
```

## CLI Флаги

| Флаг | Альтернатива (env) | По умолчанию | Описание |
|------|-------------------|--------------|----------|
| `--file` или позиционный аргумент | `INPUT_FILE` | stdin | Путь к входному файлу |
| `--mode` | `MODE` | `map-reduce` | Режим: `map-reduce`, `rag` или `tools` |
| `--api-url` | `LLM_API_URL` | `http://localhost:1234/v1` | URL LLM API |
| `--model` | `LLM_MODEL` | `local-model` | Имя модели |
| `--embedding-model` | `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding модель для `rag` |
| `--api-key` | `OPENAI_API_KEY` | (пусто) | Ключ API (опционально) |
| `-c, --concurrency` | `CONCURRENCY` | `1` | Кол-во параллельных запросов для map-reduce |
| `-l, --length` | `LENGTH` | `10000` | Макс. размер чанка в байтах (map-reduce) |
| `--rag-top-k` | `RAG_TOP_K` | `8` | Сколько чанков брать после retrieval |
| `--rag-final-k` | `RAG_FINAL_K` | `4` | Сколько чанков передавать в финальный answer synthesis |
| `--rag-chunk-size` | `RAG_CHUNK_SIZE` | `1800` | Размер retrieval чанка в байтах |
| `--rag-chunk-overlap` | `RAG_CHUNK_OVERLAP` | `200` | Перекрытие retrieval чанков в байтах |
| `--rag-index-ttl` | `RAG_INDEX_TTL` | `24h` | TTL временного локального индекса |
| `--rag-index-dir` | `RAG_INDEX_DIR` | `${TMPDIR}/ragcli-index` | Базовая директория локального индекса |
| `--rag-rerank` | `RAG_RERANK` | `heuristic` | Режим rerank: `off`, `heuristic`, `model` |
| `-r, --retry` | `RETRY` | `3` | Кол-во повторных попыток при ошибках |
| `-v, --verbose` | `VERBOSE` | `false` | Включить debug логирование |
| `--version` | — | `dev` для локальной сборки | Показать версию бинаря и выйти |
| — | `HTTP_REQUEST_TIMEOUT` | `10m` | Таймаут HTTP запроса (например, 5m, 30s) |
| — | `HTTP_DIAL_TIMEOUT` | `30s` | Таймаут установления соединения |
| — | `HTTP_TLS_TIMEOUT` | `30s` | Таймаут TLS рукопожатия |

## Примеры использования

### Map-Reduce режим (по умолчанию)

```bash
# Обработка файла с использованием 4 параллельных запросов:
./ragcli --file document.txt -c 4

# С кастомным размером чанка и API:
./ragcli document.txt \
  --api-url http://localhost:1234/v1 \
  --model my-model \
  --length 8000 \
  --verbose

# Через stdin:
cat document.txt | ./ragcli "Какие основные выводы из этого документа?"
```

### TOOLS режим (Agentic Tool Calling)

В этом режиме модель сама исследует файл через tools и не получает весь файл целиком в контекст.
Цикл многошаговый: модель может несколько раз вызвать `search_file`, `read_lines` и `read_around`, пока не соберёт достаточно контекста для финального ответа.

```bash
./ragcli file.txt \
  --mode tools \
  --api-url http://localhost:1234/v1 \
  "Какие ключевые моменты обсуждаются в разделе про архитектуру?"
```

Модель будет вызывать инструменты `search_file`, `read_lines` и `read_around` для исследования файла.
Все инструменты возвращают JSON, чтобы модель могла опираться на структурированные поля, а не на свободный текст.

### RAG режим (retrieval с локальным индексом)

`rag` строит локальный временный индекс из `--file` или `stdin`, сохраняет chunk metadata и embeddings в tempdir, затем:
- встраивает вопрос embedding моделью;
- находит top-k релевантных чанков по cosine similarity;
- делает lightweight rerank;
- формирует ответ только по evidence chunks и добавляет блок `Sources`.

```bash
./ragcli file.txt \
  --mode rag \
  --model qwen2.5 \
  --embedding-model text-embedding-3-small \
  "Что сказано про retry policy?"
```

## Режимы работы

### Map-Reduce

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
- `INFO` — по умолчанию
- `DEBUG` — при `-v/--verbose`

Примеры логов:
```bash
# Map-reduce режим
INFO starting map-reduce processing  chunk_length=10000 concurrency=4
INFO split into chunks count=5
INFO map phase completed results_count=3
INFO received final answer from model

# TOOLS режим  
INFO starting tools processing file_path=document.txt
INFO received tool calls from model count=2
INFO received final answer from model

# RAG режим
INFO rag processing finished chunk_count=12 embedding_calls=2 cache_hit=false retrieved_k=8 answer_context_chunks=4
```

## Обработка ошибок и retry

При ошибках сети, таймаутах или ответах 5xx от LLM автоматически выполняются повторные попытки с экспоненциальной задержкой:
- Попроба 1 → задержка 0с (сразу)
- Попроба 2 → задержка 1с = 1²
- Попроба 3 → задержка 4с = 2²
- Попроба 4 → задержка 9с = 3²

Количество попыток задается флагом `-r`.

## Настройка HTTP таймаутов

Для работы с медленными LLM серверами или при обработке больших чанков можно настроить HTTP таймауты через переменные окружения:

| Переменная | По умолчанию | Описание | Пример значения |
|------------|--------------|----------|-----------------|
| `HTTP_REQUEST_TIMEOUT` | `10m` | Общий таймаут на весь HTTP запрос (чтение тела ответа) | `5m`, `30s`, `2h` |
| `HTTP_DIAL_TIMEOUT` | `30s` | Таймаут установления TCP соединения | `10s`, `1m` |
| `HTTP_TLS_TIMEOUT` | `30s` | Таймаут TLS рукопожатия (для HTTPS) | `15s`, `1m` |

**Пример использования:**

```bash
# Увеличиваем таймаут до 20 минут для долгой обработки больших чанков
export HTTP_REQUEST_TIMEOUT=20m
./ragcli --file large_document.txt

# Для очень медленных серверов можно настроить все таймауты
export HTTP_DIAL_TIMEOUT=60s
export HTTP_TLS_TIMEOUT=60s
export HTTP_REQUEST_TIMEOUT=15m
./ragcli "Ваш вопрос" --mode tools
```

**Примечание:** Увеличение таймаутов рекомендуется при работе с:
- Локальными LLM серверами, которые медленно загружают модели
- Обработкой очень больших чанков текста
- Медленными сетевыми условиями

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

Эти хуки дублируют базовые проверки CI, но не заменяют GitHub Actions.

### Автоматические релизы

При создании тега вида `vX.Y.Z` автоматически создаётся релиз с бинарниками для всех платформ через [goreleaser](https://goreleaser.com/). Конфигурация релиза зафиксирована в `.goreleaser.yml`.

GoReleaser вшивает тег релиза в бинарь, поэтому `./ragcli --version` для релизных артефактов выводит соответствующую версию, например `v1.0.0`. Для локальной сборки без `ldflags` команда выводит `dev`.

Для создания нового релиза:
```bash
git tag v1.0.0 && git push origin v1.0.0
```

### Локальная сборка

Для локальной сборки бинарника для вашей платформы:
```bash
go build -o ragcli ./cmd/ragcli
./ragcli --version   # dev
```

Для проверки release-конфига без публикации можно использовать dry-run:
```bash
goreleaser release --clean --snapshot --skip=publish --config .goreleaser.yml
```

Для кроссплатформенной сборки используйте GoReleaser или соберите отдельно для каждой платформы через переменные `GOOS` и `GOARCH`.
