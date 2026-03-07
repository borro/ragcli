# RAG CLI — Инструмент для работы с LLM через RAG и Map-Reduce

[![CI main](https://github.com/borro/ragcli/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/borro/ragcli/actions/workflows/ci.yaml)
[![Coverage](https://codecov.io/gh/borro/ragcli/graph/badge.svg)](https://codecov.io/gh/borro/ragcli)

Командная строка для обработки больших текстовых файлов с помощью LLM (Large Language Models) двумя режимами: **Map-Reduce** и **Agentic Tool Calling (RAG)**.

## Какую проблему решает ragcli

LLM плохо работает с большими файлами “в лоб”:
- файл не помещается в контекст модели целиком;
- ручной поиск по логам/документам занимает много времени;
- при разбиении текста легко потерять важный контекст;
- в CLI-скриптах нужен предсказуемый и автоматизируемый способ получать ответы из текста.

`ragcli` закрывает это двумя режимами:
- `map-reduce`: быстро и параллельно обрабатывает большие тексты по чанкам;
- `rag`: позволяет модели точечно искать нужные места в файле через инструменты (`search_file`, `read_lines`).

## Типовые use cases

1. Анализ больших логов и отчётов  
Вопрос: “Где в логах причины 5xx и какие сервисы затронуты?”  
Режим: `map-reduce` для скорости на больших объёмах.

2. Поиск точных фрагментов в документации/спецификации  
Вопрос: “Что сказано про retry policy в разделе N?”  
Режим: `rag`, когда важно найти конкретные строки и ответить по ним.

3. Быстрый Q&A по локальным текстовым данным в терминале  
Вопрос: “Какие ключевые выводы в этом файле?”  
Режим: любой, в зависимости от задачи (скорость vs точечный поиск).

4. Интеграция в CI/скрипты и работу с локальными LLM  
`ragcli` работает с OpenAI-compatible API (Ollama, LM Studio, vLLM), поэтому подходит для автоматизации без UI.

## Что важно понимать

Это не “векторная RAG-система” с индексом и embedding-базой.  
В `rag` режиме используется tool calling поверх файла (`search_file`/`read_lines`), а в `map-reduce` — чанкование и агрегация ответов.


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
│   │   ├── rag.go            # RAG (Agentic Tool Calling) режим
│   │   └── tools.go          # Инструменты для RAG режима
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

# Или запуск напрямую
go run ./cmd/ragcli --help
```

## CLI Флаги

| Флаг | Альтернатива (env) | По умолчанию | Описание |
|------|-------------------|--------------|----------|
| `--file` или позиционный аргумент | `INPUT_FILE` | stdin | Путь к входному файлу |
| `--mode` | `MODE` | `map-reduce` | Режим: `map-reduce` или `rag` |
| `--api-url` | `LLM_API_URL` | `http://localhost:1234/v1` | URL LLM API |
| `--model` | `LLM_MODEL` | `local-model` | Имя модели |
| `--api-key` | `OPENAI_API_KEY` | (пусто) | Ключ API (опционально) |
| `-c, --concurrency` | `CONCURRENCY` | `1` | Кол-во параллельных запросов для map-reduce |
| `-l, --length` | `LENGTH` | `10000` | Макс. размер чанка в байтах (map-reduce) |
| `-r, --retry` | `RETRY` | `3` | Кол-во повторных попыток при ошибках |
| `-v, --verbose` | `VERBOSE` | `false` | Включить debug логирование |
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

### RAG режим (Agentic Tool Calling)

В этом режиме модель сама решает, какие инструменты использовать для поиска информации в файле.

```bash
./ragcli file.txt \
  --mode rag \
  --api-url http://localhost:1234/v1 \
  "Какие ключевые моменты обсуждаются в разделе про архитектуру?"
```

Модель будет вызывать инструменты `search_file` и `read_lines` для исследования файла.

## Режимы работы

### Map-Reduce

Файл разбивается на чанки, каждый обрабатывается параллельно через LLM (фаза **Map**), затем результаты объединяются для формирования финального ответа (фаза **Reduce**).

**Плюсы:**
- Быстрее при больших файлах благодаря параллелизму
- Предсказуемое поведение

**Минусы:**
- Может теряться контекст между чанками

### RAG (Agentic Tool Calling)

Используется механизм Function/Tool Calling. Модель получает доступ к инструментам:
- `search_file(query)` — поиск подстроки в файле с номерами строк
- `read_lines(start_line, end_line)` — чтение диапазона строк

Модель сама решает, какие инструменты и когда вызывать для формирования ответа.

**Плюсы:**
- Сохраняется полный контекст файла
- Модель может целенаправленно искать информацию

**Минусы:**
- Медленнее (множественные циклы общения с моделью)
- Требует поддержки tool calling от LLM API

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

# RAG режим  
INFO starting RAG processing file_path=document.txt
INFO received tool calls from model count=2
INFO received final answer from model
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
./ragcli "Ваш вопрос" --mode rag
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

Для создания нового релиза:
```bash
git tag v1.0.0 && git push origin v1.0.0
```

### Локальная сборка

Для локальной сборки бинарника для вашей платформы:
```bash
go build -o ragcli ./cmd/ragcli
```

Для проверки release-конфига без публикации можно использовать dry-run:
```bash
goreleaser release --clean --snapshot --skip=publish --config .goreleaser.yml
```

Для кроссплатформенной сборки используйте GoReleaser или соберите отдельно для каждой платформы через переменные `GOOS` и `GOARCH`.
