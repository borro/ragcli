# ragcli

[![CI master](https://github.com/borro/ragcli/actions/workflows/ci.yaml/badge.svg?branch=master)](https://github.com/borro/ragcli/actions/workflows/ci.yaml)
[![Codecov](https://codecov.io/gh/borro/ragcli/graph/badge.svg?token=NXBH710VR0)](https://codecov.io/gh/borro/ragcli)

`ragcli` помогает задавать вопросы к большим локальным текстам через LLM из терминала. Он работает с OpenAI-compatible API и поддерживает четыре режима: `map`, retrieval-ориентированный `rag`, гибридный `hybrid` и agentic `tools`.

Подходит для логов, документации, отчётов и других файлов, которые неудобно скармливать модели целиком. Если входной файл не указан, команда читает `stdin`, поэтому `ragcli` удобно встраивать в shell-скрипты и пайплайны.

## Почему `ragcli`

- Не нужно вручную копировать большие файлы в чат.
- Можно выбрать стратегию под задачу: chunked-обработка, retrieval или точечное исследование файла.
- Один и тот же CLI работает с локальными и удалёнными OpenAI-compatible backend'ами.
- В терминале финальный ответ рендерится как Markdown; при `--raw` или выводе в pipe/redirect печатается исходный текст в `stdout`. Ошибки без `--debug` печатаются одной строкой в `stderr`, а runtime-логи при `--debug` идут туда же в structured-виде.

## Когда использовать какой режим

| Режим | Когда выбирать | Сильная сторона | Ограничения |
| --- | --- | --- | --- |
| `map` | Нужно разбить большой файл на чанки и агрегировать ответ по ним | Работает с объёмом, который неудобно отправлять одним куском | На больших файлах может быть медленным, а между чанками теряется часть контекста |
| `rag` | Нужен retrieval по релевантным фрагментам, а не чтение файла целиком | Локальный индекс и ответ по evidence chunks | Нужен совместимый embedding endpoint (`/embeddings`) |
| `hybrid` | Документ длинный, структура смешанная, и нужен баланс между retrieval, дочитыванием соседнего контекста и извлечением фактов | Комбинирует lexical/semantic retrieval, локальное дочитывание и map-style сбор фактов | Нужен совместимый embedding endpoint (`/embeddings`); режим обычно сложнее в тюнинге, чем чистый `rag` |
| `tools` | Нужно найти конкретные строки, секции или причины в файле | Модель умеет вызывать `search_file`, `read_lines`, `read_around` | Требуется корректная поддержка tool calling на backend'е |

Короткая эвристика:
- Для больших логов и summary-задач начинайте с `map`, если файл не помещается в один запрос и вас устраивает более долгий прогон.
- Для Q&A по документу с опорой на релевантные куски используйте `rag`.
- Для длинных документов со смешанной структурой, где одного retrieval мало, попробуйте `hybrid`.
- Для точечного расследования по строкам и секциям используйте `tools`.

## Требования

- Go `1.25.6+` для сборки из исходников.
- OpenAI-compatible API для chat completion.
- Для `rag` и `hybrid` дополнительно нужен совместимый embedding endpoint (`/embeddings`).
- Поддерживаемые сценарии включают локальные backend'ы вроде Ollama, LM Studio и vLLM, если они совместимы по API.

## Quickstart

```bash
go build -o ragcli ./cmd/ragcli
./ragcli --version
```

Сразу после сборки рекомендуется выставить базовые переменные окружения, чтобы дальше не повторять одни и те же флаги:

```bash
export LLM_API_URL=http://localhost:1234/v1
export OPENAI_API_KEY=your-api-key
export LLM_MODEL=your-chat-model
```

Если `LLM_MODEL` не задана, `ragcli` использует значение по умолчанию `local-model`. Во многих локальных OpenAI-compatible backend'ах это означает "возьми последнюю загруженную модель", но это поведение определяется самим backend'ом, а не CLI.

Для режимов `rag` и `hybrid`, если `--embedding-model` и `EMBEDDING_MODEL` не заданы, по умолчанию используется embedding-модель `text-embedding-3-small`. При необходимости её можно переопределить флагом или переменной окружения.

После этого можно запускать команды без постоянной передачи `--api-url`, `--api-key` и `--model`.

### Первый запуск

```bash
./ragcli map --file document.txt "Какие основные выводы?"
./ragcli rag --file spec.txt "Что сказано про retry policy?"
./ragcli tools --file app.log "Где в логах причины 5xx?"
```

## Практические заметки

- `map` полезен, когда файл приходится обрабатывать чанками, но на больших файлах такой прогон может занимать заметное время.
- На локальных моделях и при фактически последовательной обработке `map` обычно не даёт выигрыша по скорости.
- `rag` отвечает только по найденным evidence chunks, поэтому качество зависит от embedding-модели и параметров chunking.
- `hybrid` сочетает lexical/semantic retrieval, точечное дочитывание и map-style извлечение фактов, поэтому обычно полезен для длинных документов со смешанной структурой.
- `tools` не читает весь файл в контекст модели сразу; вместо этого модель вызывает локальные инструменты `search_file`, `read_lines` и `read_around`.
- Если backend плохо поддерживает tool calling, режим `tools` может работать менее надёжно.
- В интерактивном терминале `ragcli` рендерит финальный Markdown-ответ через ANSI-стили; если вывод перенаправлен в файл или pipe, остаётся сырой Markdown без ANSI.
- `--raw` принудительно отключает markdown-рендер и печатает исходный ответ как есть.
- `--debug` включает подробные runtime-логи в `stderr`, не смешивая их с финальным ответом в `stdout`; без него `stderr` остаётся каналом для краткой пользовательской ошибки.
- `--verbose` показывает пользовательский статус выполнения в `stderr` отдельными строками; ранняя подготовка и открытие input отображаются как текстовые этапы, а процентный прогресс начинается внутри выбранного режима.
- Язык интерфейса CLI можно задать через `--lang` или `RAGCLI_LANG`; если override не задан, `ragcli` пытается определить locale системы и в случае неуспеха падает в `en`.
- Язык ответа модели по умолчанию следует языку вопроса пользователя, а не locale интерфейса CLI.

## Примеры

### `map`

```bash
./ragcli map --file report.txt -c 4 "Собери ключевые выводы"
cat report.txt | ./ragcli map "Какие риски и ограничения описаны в документе?"
./ragcli map --file report.txt --length 8000 --debug "Что означает --retry в этом документе?"
./ragcli map --file report.txt --raw "Верни ответ без терминального markdown-рендера"
```

### `rag`

```bash
./ragcli rag --file architecture.md --embedding-model text-embedding-3-small \
  "Какие ограничения описаны в разделе про масштабирование?"

cat spec.txt | ./ragcli rag --rag-top-k 10 --rag-final-k 5 \
  "Какие требования относятся к отказоустойчивости?"
```

### `hybrid`

```bash
./ragcli hybrid --file design.md --verbose \
  "Какие ограничения и компромиссы описаны в документе?"
```

### `tools`

```bash
./ragcli tools --file app.log "Где начинаются ошибки авторизации?"
./ragcli tools --file handbook.md --debug \
  "Что сказано про release process и rollback?"
```

## Команды

Основные команды:

- `ragcli map [options] <prompt>`
- `ragcli rag [options] <prompt>`
- `ragcli hybrid [options] <prompt>`
- `ragcli tools [options] <prompt>`
- `ragcli version`
- `ragcli help [command]`

Поведение CLI:

- `ragcli` без подкоманды печатает help.
- `ragcli --version` и `ragcli version` выводят версию.
- `ragcli help map` и `ragcli map --help` показывают help одной и той же команды.
- `prompt` передаётся как позиционный аргумент.
- `--file` опционален: если его нет, команда читает входные данные из `stdin`.

## Важные флаги и переменные окружения

Глобальные:

| Флаг | Переменная | Значение |
| --- | --- | --- |
| `--file`, `-f` | `INPUT_FILE` | Путь к входному файлу; если не задан, читается `stdin` |
| `--api-url` | `LLM_API_URL` | URL OpenAI-compatible API |
| `--api-key` | `OPENAI_API_KEY` | API key |
| `--model` | `LLM_MODEL` | Chat-модель для `map`, `rag`, `tools` |
| `--retry`, `-r` | `RETRY` | Количество retry для LLM HTTP-клиента |
| `--raw` | `RAW` | Отключить markdown-рендер финального ответа и печатать сырой текст |
| `--debug`, `-d` | `DEBUG` | Подробные runtime-логи в `stderr`; без флага ошибки печатаются одной строкой |
| `--verbose`, `-v` | `VERBOSE` | Печатать в `stderr` пользовательские статусы выполнения по этапам |
| `--lang` | `RAGCLI_LANG` | Язык интерфейса CLI: `en` или `ru`; без override идёт автоопределение по системной locale |

`map`:

- `--concurrency`, `-c` или `CONCURRENCY`
- `--length`, `-l` или `LENGTH` — строгий override лимита токенов на chunk; если не задан, `map` пытается автоопределить context length модели и при неуспехе использует `10000`

`rag`:

- `--embedding-model` или `EMBEDDING_MODEL`
- `--rag-top-k` или `RAG_TOP_K`
- `--rag-final-k` или `RAG_FINAL_K`
- `--rag-chunk-size` или `RAG_CHUNK_SIZE`
- `--rag-chunk-overlap` или `RAG_CHUNK_OVERLAP`
- `--rag-index-ttl` или `RAG_INDEX_TTL`
- `--rag-index-dir` или `RAG_INDEX_DIR`
- `--rag-rerank` или `RAG_RERANK`

`hybrid`:

- `--embedding-model` или `EMBEDDING_MODEL`
- `--rag-index-ttl` или `RAG_INDEX_TTL`
- `--rag-index-dir` или `RAG_INDEX_DIR`
- `--hybrid-top-k` или `HYBRID_TOP_K`
- `--hybrid-final-k` или `HYBRID_FINAL_K`
- `--hybrid-map-k` или `HYBRID_MAP_K`
- `--hybrid-read-window` или `HYBRID_READ_WINDOW`
- `--hybrid-fallback` или `HYBRID_FALLBACK`

Полный справочник по текущим флагам смотрите через встроенный help:

```bash
./ragcli --help
./ragcli map --help
./ragcli rag --help
./ragcli hybrid --help
./ragcli tools --help
```

## Для разработки

Точка входа находится в [`cmd/ragcli/main.go`](/home/borro/projects/llm-code/ragcli/cmd/ragcli/main.go), а CLI-описание и help-тексты определены в [`internal/app/cli.go`](/home/borro/projects/llm-code/ragcli/internal/app/cli.go). Если нужен полный обзор доступных опций и их значений по умолчанию, удобнее начинать именно с help или этих файлов.
