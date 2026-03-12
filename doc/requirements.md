# Требования к `ragcli`

## 1. Назначение

`ragcli` — CLI-инструмент для вопросов к большим локальным текстам через OpenAI-compatible API. Он нужен в случаях, когда документ, лог или отчёт неудобно отправлять в модель одним куском и требуется один из четырёх режимов:

- `map` — chunked map-reduce обработка.
- `rag` — retrieval по локальному индексу.
- `hybrid` — комбинированный retrieval + дочитывание + извлечение фактов.
- `tools` — agentic-исследование файла через tool calling.

## 2. Поддерживаемые сценарии

- Q&A по большим markdown- и plain-text документам.
- Summary и извлечение выводов из длинных отчётов.
- Исследование логов по строкам, диапазонам и локальному контексту.
- Использование как standalone CLI и как элемента shell pipeline через `stdin`.

Не входит в текущий scope:

- Веб-интерфейс.
- Многодокументный корпус с общей коллекцией индексов.
- Внешний persistent metadata store поверх файлового индекса.

## 3. CLI-контракт

### 3.1 Команды

Поддерживаются команды:

- `ragcli map [options] <prompt>`
- `ragcli rag [options] <prompt>`
- `ragcli hybrid [options] <prompt>`
- `ragcli tools [options] <prompt>`
- `ragcli version`
- `ragcli help [command]`

Инварианты CLI:

- `prompt` обязателен для всех режимов обработки.
- `ragcli` без подкоманды показывает help корня.
- `ragcli --version` и `ragcli version` выводят версию.
- `ragcli help <command>` и `<command> --help` должны описывать один и тот же command surface.
- После начала positional prompt парсинг флагов должен останавливаться через внутреннюю compatibility-обвязку `applyPromptArgCompatibility`.

### 3.2 Входные данные

- `--file`/`-f` задаёт путь к входному файлу.
- Если `--file` не задан, вход читается из `stdin`.
- Для унификации downstream-пайплайнов `stdin` материализуется во временный файл и переоткрывается как `input.Source`.
- Временный файл от `stdin` должен удаляться при cleanup.

### 3.3 Конфигурация LLM

Глобальные опции:

| Флаг | Env | Значение | Фактический default |
| --- | --- | --- | --- |
| `--api-url` | `LLM_API_URL` | Базовый URL OpenAI-compatible API | `http://localhost:1234/v1` |
| `--api-key` | `OPENAI_API_KEY` | API key | пусто |
| `--model` | `LLM_MODEL` | Chat-модель для `map`, `rag`, `hybrid`, `tools` | `local-model` |
| `--retry`, `-r` | `RETRY` | Число повторов поверх первой попытки | `3` |
| `--proxy-url` | `LLM_PROXY_URL` | Явный proxy URL для всех LLM-запросов | пусто |
| `--no-proxy` | `LLM_NO_PROXY` | Полностью отключить proxy для LLM-запросов | `false` |

Поведение конфигурации:

- `.env` подгружается автоматически через `godotenv.Load()`.
- Proxy precedence: `--no-proxy` сильнее `--proxy-url`; `--proxy-url` сильнее `HTTP_PROXY`/`HTTPS_PROXY`; иначе используется proxy окружения.
- Для `rag` и `hybrid` дополнительно нужен embeddings endpoint.
- CLI default для `--embedding-model` — `text-embedding-nomic-embed-text-v1.5`.

### 3.4 Вывод и UX

Глобальные UX-опции:

| Флаг | Env | Назначение |
| --- | --- | --- |
| `--raw` | `RAW` | Печатать финальный ответ как сырой текст без markdown renderer |
| `--debug`, `-d` | `DEBUG` | Включить structured runtime-логи в `stderr` |
| `--verbose`, `-v` | `VERBOSE` | Включить пользовательский stage/progress вывод в `stderr` |
| `--lang` | `RAGCLI_LANG` | Язык интерфейса CLI (`ru`/`en`) |

Инварианты вывода:

- Финальный результат печатается только в `stdout`.
- Краткая пользовательская ошибка печатается в `stderr`, если `--debug` выключен.
- Structured runtime-логи через `slog` пишутся в `stderr` только при `--debug`.
- `--verbose` не должен смешиваться с `--debug`: при `--debug` verbose reporter отключён.
- Если `stdout` — TTY и `--raw` не задан, ответ рендерится через `glamour`; иначе печатается raw Markdown.

## 4. Режимы обработки

### 4.1 `map`

Назначение: обработка длинного текста чанками с последующей агрегацией.

Режим должен:

- разбивать вход на чанки по approximate token budget, а не по фиксированным байтам;
- сохранять разбиение по строкам, а oversized single line дорезать по rune offsets;
- уметь параллелить map-фазу через worker pool размера `--concurrency`;
- выполнять reduce fan-in итерациями, пока не останется один блок фактов;
- строить финальный ответ по reduced facts;
- запускать self-critique и self-refine до финального возврата.

Опции `map`:

| Флаг | Env | Значение | Фактический default |
| --- | --- | --- | --- |
| `--concurrency`, `-c` | `CONCURRENCY` | Число параллельных map-вызовов | `1` |
| `--length`, `-l` | `LENGTH` | Явный лимит контекста в approximate tokens | `10000` в help; runtime auto-resolve при отсутствии explicit override |

Дополнительные инварианты:

- если `--length` не задан явно, режим пытается определить размер контекста через `ResolveAutoContextLength`;
- при неуспехе auto-detect используется fallback `10000`;
- пустой файл возвращает ответ об empty input;
- отсутствие полезных map-results возвращает ответ об insufficient information без падения.

### 4.2 `rag`

Назначение: retrieval по локальному файловому индексу.

Режим должен:

- материализовать вход в spool-файл с hash-based identity;
- строить или переиспользовать индекс по hash входа, schema version, chunking params и embedding model;
- хранить manifest, chunks и embeddings в отдельной директории индекса;
- embed-ить query, ранжировать чанки, опционально rerank-ить и выбирать evidence;
- синтезировать ответ только по evidence chunks;
- дописывать в финал секцию `Sources:`.

Опции `rag`:

| Флаг | Env | Значение | Фактический default |
| --- | --- | --- | --- |
| `--embedding-model` | `EMBEDDING_MODEL` | Embedding-модель | `text-embedding-nomic-embed-text-v1.5` |
| `--rag-top-k` | `RAG_TOP_K` | Число кандидатов после retrieval | `8` |
| `--rag-final-k` | `RAG_FINAL_K` | Число evidence chunks для финального ответа | `4` |
| `--rag-chunk-size` | `RAG_CHUNK_SIZE` | Target chunk size | `1800` |
| `--rag-chunk-overlap` | `RAG_CHUNK_OVERLAP` | Overlap между чанками | `200` |
| `--rag-index-ttl` | `RAG_INDEX_TTL` | TTL индекса | `24h` |
| `--rag-index-dir` | `RAG_INDEX_DIR` | Базовая директория индексов | `os.TempDir()/ragcli-index` |
| `--rag-rerank` | `RAG_RERANK` | Стратегия rerank | `heuristic` |

Нормализация и ограничения:

- `final-k` не может быть больше `top-k`;
- `chunk-overlap` не может быть больше или равен `chunk-size`, иначе приводится к `chunk-size / 4`;
- `rag-rerank` принимает `heuristic`, `off`, `model`, неизвестные значения нормализуются в `heuristic`.

### 4.3 `hybrid`

Назначение: retrieval-пайплайн для длинных документов со смешанной структурой, где одного `rag` недостаточно.

Режим должен:

- определять профиль документа: `markdown`, `logs`, `plain`, `unknown`;
- сегментировать документ в meso-level регионы;
- выполнять lexical и semantic retrieval;
- сливать хиты в регионы, расширять их и дочитывать соседний контекст;
- извлекать факты map-style по top regions;
- выполнять coverage check и при необходимости targeted reread;
- синтезировать финальный ответ по фактам и evidence;
- поддерживать fallback, если semantic retrieval или финальная генерация оказались недоступны.

Опции `hybrid`:

| Флаг | Env | Значение | Фактический default |
| --- | --- | --- | --- |
| `--embedding-model` | `EMBEDDING_MODEL` | Embedding-модель | `text-embedding-nomic-embed-text-v1.5` |
| `--rag-chunk-size` | `RAG_CHUNK_SIZE` | Размер сегмента для индексирования | `1800` |
| `--rag-chunk-overlap` | `RAG_CHUNK_OVERLAP` | Overlap | `200` |
| `--rag-index-ttl` | `RAG_INDEX_TTL` | TTL индекса | `24h` |
| `--rag-index-dir` | `RAG_INDEX_DIR` | Базовая директория индексов | `os.TempDir()/ragcli-index` |
| `--hybrid-top-k` | `HYBRID_TOP_K` | Число retrieval кандидатов | `8` |
| `--hybrid-final-k` | `HYBRID_FINAL_K` | Число финальных evidence regions | `4` |
| `--hybrid-map-k` | `HYBRID_MAP_K` | Число regions для map-style fact extraction | `4` |
| `--hybrid-read-window` | `HYBRID_READ_WINDOW` | Ширина дочитывания вокруг лучшей строки | `3` |
| `--hybrid-fallback` | `HYBRID_FALLBACK` | Стратегия fallback | `map` |

Нормализация и ограничения:

- `final-k` не может быть больше `top-k`;
- `map-k` не может быть больше `top-k`;
- `read-window` минимум `1`;
- `hybrid-fallback` принимает `map`, `rag-only`, `fail`, неизвестные значения нормализуются в `map`.

### 4.4 `tools`

Назначение: agentic анализ файла через tool calling вместо загрузки всего текста в prompt.

Режим должен:

- отдавать модели набор локальных инструментов `search_file`, `read_lines`, `read_around`;
- крутить tool loop до финального ответа, лимита ходов или forced finalization;
- кэшировать уже выполненные tool calls;
- отличать отсутствие прогресса от новых строк/нового контекста;
- ограничивать повторы, пустые финальные ответы и дублирующиеся вызовы;
- возвращать orchestration error, если модель зациклилась и не даёт содержательного финального ответа.

Инструменты должны поддерживать:

- `search_file(query, mode, limit, offset, context_lines)`
- `read_lines(start_line, end_line)`
- `read_around(line, before, after)`

Публичные лимиты текущей реализации:

- максимум `20` turns в одном orchestration loop;
- guard на повторные no-progress/duplicate tool runs;
- JSON-результаты инструментов как единственный wire-format между локальным tool runner и моделью.

## 5. Нефункциональные требования

### 5.1 Надёжность

- Все LLM и embeddings-запросы идут через retry wrapper с exponential backoff `1s`, `2s`, `4s`, ... .
- Отмена через `context.Context` должна прерывать запросы, backoff sleep и длинные пайплайны.
- `SIGINT` и `SIGTERM` должны переводиться в command context через `signal.NotifyContext`.

### 5.2 Локальные артефакты и приватность

- `stdin` сохраняется во временный файл только на время обработки и затем удаляется.
- Индексы `rag`/`hybrid` лежат на локальной файловой системе и публикуются атомарно через temp dir + rename.
- Права на индексные файлы и временные директории должны быть приватными (`0700` для директорий, `0600` для файлов).
- Просроченные индексы должны очищаться по TTL.

### 5.3 Наблюдаемость

- Debug-логи должны содержать информацию о стадиях пайплайна, попытках retry, токенах, выбранных режимах proxy и важных runtime-счётчиках.
- Verbose-режим должен показывать человекочитаемый stage/progress, но не заменять debug-логи.
- Source path в debug-логах должен быть укорочен до относительного пути от корня проекта.

### 5.4 Локализация

- Интерфейс CLI локализуется через встроенные каталоги `en` и `ru`.
- Override приоритетов: `--lang`, затем `RAGCLI_LANG`, затем системная locale, затем fallback `en`.
- Неподдерживаемый явный `--lang` должен завершать CLI с пользовательской ошибкой; невалидные env/system locale должны тихо приводиться к fallback `en`.
- Язык интерфейса CLI и язык ответа модели не обязаны совпадать: ответ модели следует вопросу пользователя.

## 6. Формат результата

- `map` возвращает чистый ответ модели без секции `Sources`.
- `rag` и `hybrid` всегда дописывают секцию `Sources:` с deduplicated источниками.
- Если evidence не найдено, `rag` должен возвращать честный insufficient-data ответ и `Sources:\n- none`.
- Raw output перед печатью обрезается по краям `strings.TrimSpace`.

## 7. Критичные инварианты поддержки

- При добавлении новой команды её нужно зарегистрировать в `internal/app/commandspec.go`, включить в help и в progress template.
- При изменении defaults нужно синхронизировать код, тесты, `doc/requirements.md` и корневой `README.md`.
- При изменении поведения режима нужно обновлять также package-level `README.md` соответствующего пакета.
