# ragcli

[![CI master](https://github.com/borro/ragcli/actions/workflows/ci.yaml/badge.svg?branch=master)](https://github.com/borro/ragcli/actions/workflows/ci.yaml)
[![Codecov](https://codecov.io/gh/borro/ragcli/graph/badge.svg?token=NXBH710VR0)](https://codecov.io/gh/borro/ragcli)

`ragcli` помогает задавать вопросы к большим локальным текстам через LLM из терминала. Он работает с OpenAI-compatible API и поддерживает четыре режима: проверяющий `hybrid`, agentic `tools`, retrieval-ориентированный `rag` и `map` для fallback chunked-обработки.

Подходит для логов, документации, отчётов и других файлов, которые неудобно скармливать модели целиком. Если входной файл не указан, команда читает `stdin`, поэтому `ragcli` удобно встраивать в shell-скрипты и пайплайны. По флагу `--interaction` (`-i`) после первого ответа CLI остаётся в post-run сессии и принимает follow-up вопросы с командами `/reset` и `/exit`.

## Документация

- Канонический комплект требований и архитектуры лежит в [`doc/README.md`](doc/README.md).
- Общие документы находятся в папке [`doc`](doc/), а package-level обзоры лежат рядом с кодом в `README.md` внутри `cmd/` и `internal/`.
- Если README и код расходятся, источником истины считаются текущий код и проходящие тесты.

## Почему `ragcli`

- Не нужно вручную копировать большие файлы в чат.
- Можно выбрать стратегию под задачу: seeded hybrid-проверку, точечное исследование файла, retrieval по локальному индексу или, если остальные варианты не подходят, chunked-обработку.
- Один и тот же CLI работает с локальными и удалёнными OpenAI-compatible backend'ами.
- В терминале финальный ответ рендерится как Markdown; при `--raw` или выводе в pipe/redirect печатается исходный текст в `stdout`. Ошибки без `--debug` печатаются одной строкой в `stderr`, а runtime-логи при `--debug` идут туда же в structured-виде.

## Когда использовать какой режим

| Режим | Когда выбирать | Сильная сторона | Ограничения |
| --- | --- | --- | --- |
| `hybrid` | Нужен стартовый semantic retrieval, но ответ надо проверить и доуточнить через tools | Fused semantic seed, synthetic exact reads и итоговый `Sources:` с `Verified`/`Retrieved` | Нужны embeddings и рабочий tool calling |
| `tools` | Нужно найти конкретные строки, секции или причины в файле или директории | Модель умеет вызывать `list_files`, `search_file`, `read_lines`, `read_around`, а с `--rag` ещё и `search_rag` | Требуется корректная поддержка tool calling на backend'е |
| `rag` | Нужен retrieval по релевантным фрагментам, а не чтение файла целиком | Fused semantic seed по локальному индексу и ответ по evidence chunks | Нужен совместимый embedding endpoint (`/embeddings`) |
| `map` | Остальные режимы неприменимы, но длинный текст всё равно нужно прогнать чанками | Не требует embeddings и tool calling | Медленнее, хуже держит глобальный контекст и агрегирует ответ по частям |

Короткая эвристика:
- Для вопросов, где нужен быстрый semantic старт и последующая проверка формулировок, используйте `hybrid`.
- Для точечного расследования по строкам и секциям используйте `tools`.
- Для Q&A по документу с опорой на релевантные куски используйте `rag`.
- К `map` переходите как к запасному варианту, если embeddings или tool calling недоступны либо нужен прямой chunked-прогон длинного текста.

## Требования

- Go `1.25.6+` для сборки из исходников.
- OpenAI-compatible API для chat completion.
- Для `hybrid`, `tools --rag` и `rag` дополнительно нужен совместимый embedding endpoint (`/embeddings`).
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
export LLM_PROXY_URL=http://127.0.0.1:1080   # optional explicit proxy
```

Если `LLM_MODEL` не задана, `ragcli` использует значение по умолчанию `local-model`. Во многих локальных OpenAI-compatible backend'ах это означает "возьми последнюю загруженную модель", но это поведение определяется самим backend'ом, а не CLI.

Для режимов `hybrid`, `tools --rag` и `rag`, если `--embedding-model` и `EMBEDDING_MODEL` не заданы, CLI по умолчанию использует embedding-модель `text-embedding-nomic-embed-text-v1.5`. Если backend ожидает другую модель, переопределите её флагом или переменной окружения.

После этого можно запускать команды без постоянной передачи `--api-url`, `--api-key` и `--model`.

Если нужно явно управлять прокси независимо от платформы, используйте `--proxy-url` или `LLM_PROXY_URL`. Чтобы полностью отключить любой прокси для запросов `ragcli`, используйте `--no-proxy` или `LLM_NO_PROXY=true`.

На Windows удобно задавать override явно, не полагаясь на системное поведение `HTTP_PROXY`:

```powershell
$env:LLM_PROXY_URL = "http://127.0.0.1:1080"
./ragcli rag --path spec.txt "Какие ограничения описаны?"
```

### Первый запуск

```bash
./ragcli hybrid --path spec.txt --embedding-model text-embedding-3-small "Что сказано про retry policy?"
./ragcli tools --path app.log "Где в логах причины 5xx?"
./ragcli tools --rag --path spec.txt --embedding-model text-embedding-3-small "Что сказано про retry policy?"
./ragcli rag --path spec.txt "Что сказано про retry policy?"
./ragcli rag --proxy-url http://127.0.0.1:1080 --path spec.txt "Какие ограничения описаны?"
./ragcli map --path document.txt "Какие основные выводы?"
./ragcli tools --path app.log --interaction "Где начинаются ошибки авторизации?"
```

## Практические заметки

- `hybrid` сначала делает fused semantic seed из нескольких deterministic `search_rag` запросов, затем предзагружает synthetic exact reads по лучшим hit'ам и только после этого передаёт управление общему tool loop; итог дополняется `Sources:` с отдельными секциями `Verified` и `Retrieved`. `list_files` остаётся навигационным инструментом и не попадает в источники.
- `tools` не читает весь файл в контекст модели сразу; вместо этого модель вызывает локальные инструменты `list_files`, `search_file`, `read_lines` и `read_around`. С `--rag` режим заранее строит локальный индекс и добавляет semantic retrieval tool `search_rag`.
- `tools --rag` полезен, когда нужен semantic retrieval без обязательного стартового seed: модель сама решает, когда вызывать `search_rag`, а когда достаточно обычных file-tools.
- `rag` сначала делает fused semantic seed по нескольким variants исходного вопроса, а затем отвечает только по выбранным evidence chunks.
- Если backend плохо поддерживает tool calling, режим `tools` может работать менее надёжно.
- `--path` может указывать и на директорию: `hybrid`/`tools`/`rag` сохраняют file-aware пути и line numbers, а `map` в этом случае работает по synthetic corpus.
- `map` стоит держать как fallback-режим: он режет вход на чанки и агрегирует ответ по частям, поэтому на больших файлах может быть заметно медленнее и хуже держать общий контекст.
- На локальных моделях и при фактически последовательной обработке `map` обычно не даёт выигрыша по скорости.
- В интерактивном терминале `ragcli` рендерит финальный Markdown-ответ через ANSI-стили; если вывод перенаправлен в файл или pipe, остаётся сырой Markdown без ANSI.
- `--raw` принудительно отключает markdown-рендер и печатает исходный ответ как есть.
- `--debug` включает подробные runtime-логи в `stderr`, не смешивая их с финальным ответом в `stdout`; без него `stderr` остаётся каналом для краткой пользовательской ошибки.
- `--verbose` показывает пользовательский статус выполнения в `stderr` отдельными строками; ранняя подготовка и открытие input отображаются как текстовые этапы, а процентный прогресс начинается внутри выбранного режима.
- `--interaction` оставляет режим открытым после первого ответа: follow-up ответы продолжают идти в `stdout`, а prompt/banner читаются и печатаются через отдельный interaction stream. Если основной документ пришёл из `stdin`, follow-up вопросы читаются из controlling TTY.
- `--proxy-url` и `LLM_PROXY_URL` принудительно направляют весь LLM-трафик через указанный proxy URL и перекрывают `HTTP_PROXY`/`HTTPS_PROXY`.
- `--no-proxy` и `LLM_NO_PROXY=true` отключают любой proxy для запросов `ragcli`, включая proxy из окружения.
- Язык интерфейса CLI можно задать через `--lang` или `RAGCLI_LANG`; если override не задан, `ragcli` пытается определить locale системы и в случае неуспеха падает в `en`.
- Язык ответа модели по умолчанию следует языку вопроса пользователя, а не locale интерфейса CLI.

## Примеры

### `hybrid`

```bash
./ragcli hybrid --path architecture.md --embedding-model text-embedding-3-small \
  "Что сказано про retry policy?"
./ragcli hybrid --path app.log --rag-top-k 10 \
  "Какие строки объясняют всплески 5xx?"
```

### `tools`

```bash
./ragcli tools --path app.log "Где начинаются ошибки авторизации?"
./ragcli tools --rag --path architecture.md --embedding-model text-embedding-3-small \
  "Что сказано про retry policy?"
./ragcli tools --path handbook.md --debug \
  "Что сказано про release process и rollback?"
```

### `rag`

```bash
./ragcli rag --path architecture.md --embedding-model text-embedding-3-small \
  "Какие ограничения описаны в разделе про масштабирование?"

cat spec.txt | ./ragcli rag --rag-top-k 10 --rag-final-k 5 \
  "Какие требования относятся к отказоустойчивости?"

./ragcli rag --proxy-url http://127.0.0.1:1080 --path architecture.md \
  "Какие ограничения описаны в документе?"

./ragcli rag --no-proxy --path architecture.md \
  "Собери требования без использования системного прокси"
```

### `map`

```bash
./ragcli map --path report.txt -c 4 "Собери ключевые выводы"
cat report.txt | ./ragcli map "Какие риски и ограничения описаны в документе?"
./ragcli map --path report.txt --length 8000 --debug "Что означает --retry в этом документе?"
./ragcli map --path report.txt --raw "Верни ответ без терминального markdown-рендера"
```

## Команды

Основные команды:

- `ragcli hybrid [options] <prompt>`
- `ragcli tools [options] <prompt>`
- `ragcli rag [options] <prompt>`
- `ragcli map [options] <prompt>`
- `ragcli version`
- `ragcli help [command]`

Поведение CLI:

- `ragcli` без подкоманды печатает help.
- `ragcli --version` и `ragcli version` выводят версию.
- `ragcli help map` и `ragcli map --help` показывают help одной и той же команды.
- `prompt` передаётся как позиционный аргумент.
- `--path` опционален: если его нет, команда читает входные данные из `stdin`.
- `--file` и `-f` остаются alias-ами для совместимости.
- `--interaction`, `-i` оставляет REPL открытым после первого успешного ответа; `/reset` возвращает состояние к моменту после первого ответа, `/exit` завершает сессию.

## Важные флаги и переменные окружения

Глобальные:

| Флаг | Переменная | Значение |
| --- | --- | --- |
| `--path` | `INPUT_PATH` | Канонический путь к входному файлу или директории; если не задан, читается `stdin` |
| `--file`, `-f` | `INPUT_FILE` | Alias для совместимости с тем же значением |
| `--api-url` | `LLM_API_URL` | URL OpenAI-compatible API |
| `--api-key` | `OPENAI_API_KEY` | API key |
| `--proxy-url` | `LLM_PROXY_URL` | Явный proxy URL для всех LLM HTTP-запросов |
| `--no-proxy` | `LLM_NO_PROXY` | Отключить любой proxy для запросов `ragcli` |
| `--model` | `LLM_MODEL` | Chat-модель для `hybrid`, `tools`, `rag`, `map` |
| `--retry`, `-r` | `RETRY` | Количество retry для LLM HTTP-клиента |
| `--raw` | `RAW` | Отключить markdown-рендер финального ответа и печатать сырой текст |
| `--debug`, `-d` | `DEBUG` | Подробные runtime-логи в `stderr`; без флага ошибки печатаются одной строкой |
| `--verbose`, `-v` | `VERBOSE` | Печатать в `stderr` пользовательские статусы выполнения по этапам |
| `--interaction`, `-i` | - | После первого ответа оставить post-run REPL для follow-up вопросов |
| `--lang` | `RAGCLI_LANG` | Язык интерфейса CLI: `en` или `ru`; без override идёт автоопределение по системной locale |

`hybrid`:

- `--embedding-model` или `EMBEDDING_MODEL`
- `--rag-top-k` или `RAG_TOP_K`
- `--rag-chunk-size` или `RAG_CHUNK_SIZE`
- `--rag-chunk-overlap` или `RAG_CHUNK_OVERLAP`
- `--rag-index-ttl` или `RAG_INDEX_TTL`
- `--rag-index-dir` или `RAG_INDEX_DIR`
- `--rag-rerank` или `RAG_RERANK`

`tools`:

- `--rag` или `TOOLS_RAG`
- `--embedding-model` или `EMBEDDING_MODEL`
- `--rag-top-k` или `RAG_TOP_K`
- `--rag-chunk-size` или `RAG_CHUNK_SIZE`
- `--rag-chunk-overlap` или `RAG_CHUNK_OVERLAP`
- `--rag-index-ttl` или `RAG_INDEX_TTL`
- `--rag-index-dir` или `RAG_INDEX_DIR`
- `--rag-rerank` или `RAG_RERANK`

`rag`:

- `--embedding-model` или `EMBEDDING_MODEL`
- `--rag-top-k` или `RAG_TOP_K`
- `--rag-final-k` или `RAG_FINAL_K`
- `--rag-chunk-size` или `RAG_CHUNK_SIZE`
- `--rag-chunk-overlap` или `RAG_CHUNK_OVERLAP`
- `--rag-index-ttl` или `RAG_INDEX_TTL`
- `--rag-index-dir` или `RAG_INDEX_DIR`
- `--rag-rerank` или `RAG_RERANK`

`map`:

- `--concurrency`, `-c` или `CONCURRENCY`
- `--length`, `-l` или `LENGTH` — строгий override лимита токенов на chunk; если не задан, `map` пытается автоопределить context length модели и при неуспехе использует `10000`

Полный справочник по текущим флагам смотрите через встроенный help:

```bash
./ragcli --help
./ragcli hybrid --help
./ragcli tools --help
./ragcli rag --help
./ragcli map --help
```

## Для разработки

Точка входа находится в [`cmd/ragcli/main.go`](/home/borro/projects/llm-code/ragcli/cmd/ragcli/main.go), а CLI-описание и help-тексты определены в [`internal/app/cli.go`](/home/borro/projects/llm-code/ragcli/internal/app/cli.go). Если нужен полный обзор доступных опций и их значений по умолчанию, удобнее начинать именно с help или этих файлов.

Архитектурный обзор и пакетная навигация описаны в [`doc/architecture.md`](doc/architecture.md) и package-level `README.md` рядом с соответствующими пакетами.

Live e2e smoke для реальной интеграции вынесен в [`scripts/e2e_live_smoke.sh`](/home/borro/projects/llm-code/ragcli/scripts/e2e_live_smoke.sh). Скрипт один раз собирает бинарь, прогоняет `map`, `tools`, `tools --rag`, `rag` и `hybrid` в режимах `--path` и `stdin`, запускает CLI с `--debug`, проверяет устойчивый маркер в `stdout`, валит сценарий при `level=ERROR` в `stderr` и складывает логи в `${RAGCLI_E2E_ARTIFACT_DIR:-${TMPDIR:-/tmp}/ragcli-e2e-live}/logs`.

Для запуска вручную нужны:

- `LLM_API_URL`
- `OPENAI_API_KEY`
- `LLM_MODEL` или `RAGCLI_E2E_CHAT_MODEL`
- `EMBEDDING_MODEL` или `RAGCLI_E2E_EMBEDDING_MODEL`

Опционально можно задать mode-specific overrides: `RAGCLI_E2E_MAP_MODEL`, `RAGCLI_E2E_TOOLS_MODEL`, `RAGCLI_E2E_TOOLS_RAG_MODEL`, `RAGCLI_E2E_RAG_MODEL`, `RAGCLI_E2E_HYBRID_MODEL`.

```bash
./scripts/e2e_live_smoke.sh
```

`lefthook` запускает этот smoke на `pre-push`, а не на `pre-commit`. Если обязательные env не заданы, hook печатает `skip` и пропускает push; для жёсткого режима выставьте `RAGCLI_E2E_REQUIRED=1`.
