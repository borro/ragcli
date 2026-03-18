# Требования к `ragcli`

## 1. Назначение

`ragcli` — CLI-инструмент для вопросов к большим локальным текстам через OpenAI-compatible API. Он нужен в случаях, когда документ, лог или отчёт неудобно отправлять в модель одним куском и требуется один из четырёх режимов:

- `hybrid` — seeded retrieval с последующей проверкой через tools.
- `tools` — agentic-исследование файла через tool calling.
- `rag` — retrieval по локальному индексу.
- `map` — запасная chunked map-reduce обработка, когда retrieval или tool calling не подходят.

## 2. Поддерживаемые сценарии

- Q&A по большим markdown- и plain-text документам.
- Q&A по директориям с локальными текстовыми файлами как по одному corpus.
- Summary и извлечение выводов из длинных отчётов.
- Исследование логов по строкам, диапазонам и локальному контексту.
- Использование как standalone CLI и как элемента shell pipeline через `stdin`.

Не входит в текущий scope:

- Веб-интерфейс.
- Глобальная shared-коллекция индексов поверх нескольких независимых директорий.
- Внешний persistent metadata store поверх файлового индекса.

## 3. CLI-контракт

### 3.1 Команды

Поддерживаются команды:

- `ragcli hybrid [options] <prompt>`
- `ragcli tools [options] <prompt>`
- `ragcli rag [options] <prompt>`
- `ragcli map [options] <prompt>`
- `ragcli self-update [--check]`
- `ragcli version`
- `ragcli help [command]`

Инварианты CLI:

- `prompt` обязателен для всех режимов обработки.
- `self-update` не принимает positional prompt и работает только по build-time version текущего бинаря.
- `ragcli` без подкоманды показывает help корня.
- `ragcli --version` и `ragcli version` выводят версию.
- `ragcli self-update` проверяет последний stable GitHub Release `borro/ragcli`, а `ragcli self-update --check` только сообщает о доступности обновления.
- `ragcli help <command>` и `<command> --help` должны описывать один и тот же command surface.
- После начала positional prompt парсинг флагов должен останавливаться через внутреннюю compatibility-обвязку `applyPromptArgCompatibility`.

### 3.2 Входные данные

- `--path` задаёт путь к входному файлу или директории.
- `--file` и `-f` остаются alias-ами для совместимости.
- Если `--path` не задан, вход читается из `stdin`.
- Канонический env var для входа — `INPUT_PATH`, `INPUT_FILE` остаётся compatibility alias.
- Для унификации downstream-пайплайнов `stdin` материализуется во временный snapshot; mode-пакеты дочитывают его через `input.Source.Open()`.
- Если вход — директория, она рекурсивно обходится в лексикографическом порядке по относительным путям.
- В corpus включаются только regular text files; dot-directories, symlink-объекты и бинарные файлы пропускаются.
- Для `map` и map-fallback директорий строится synthetic corpus с file-boundary headers.
- Временный файл от `stdin` должен удаляться при cleanup.
- Если включён `--interaction` и основной input пришёл из `stdin`, follow-up вопросы читаются из controlling TTY; если TTY недоступен, интерактивный loop не стартует и команда возвращает локализованную ошибку после первого ответа.

### 3.3 Конфигурация LLM

Глобальные опции:

| Флаг | Env | Значение | Фактический default |
| --- | --- | --- | --- |
| `--api-url` | `LLM_API_URL` | Базовый URL OpenAI-compatible API | `http://localhost:1234/v1` |
| `--api-key` | `OPENAI_API_KEY` | API key | пусто |
| `--model` | `LLM_MODEL` | Chat-модель для `hybrid`, `tools`, `rag`, `map` | `local-model` |
| `--retry`, `-r` | `RETRY` | Число повторов поверх первой попытки | `3` |
| `--proxy-url` | `LLM_PROXY_URL` | Явный proxy URL для всех LLM-запросов | пусто |
| `--no-proxy` | `LLM_NO_PROXY` | Полностью отключить proxy для LLM-запросов | `false` |

Поведение конфигурации:

- `.env` подгружается автоматически через `godotenv.Load()`.
- Proxy precedence: `--no-proxy` сильнее `--proxy-url`; `--proxy-url` сильнее `HTTP_PROXY`/`HTTPS_PROXY`; иначе используется proxy окружения.
- Для `hybrid`, `tools --rag` и `rag` дополнительно нужен embeddings endpoint.
- CLI default для `--embedding-model` — `text-embedding-nomic-embed-text-v1.5`.
- `self-update` не использует `LLM_PROXY_URL`/`LLM_NO_PROXY` и полагается только на стандартный HTTPS transport Go и proxy env системы.

### 3.4 Вывод и UX

Глобальные UX-опции:

| Флаг | Env | Назначение |
| --- | --- | --- |
| `--raw` | `RAW` | Печатать финальный ответ как сырой текст без markdown renderer |
| `--debug`, `-d` | `DEBUG` | Включить structured runtime-логи в `stderr` |
| `--verbose`, `-v` | `VERBOSE` | Включить пользовательский stage/progress вывод в `stderr` |
| `--interaction`, `-i` | - | После первого успешного ответа оставить сессию открытой для follow-up вопросов |
| `--lang` | `RAGCLI_LANG` | Язык интерфейса CLI (`ru`/`en`) |

Инварианты вывода:

- Финальный результат печатается только в `stdout`.
- В interactive loop follow-up ответы тоже печатаются только в `stdout`.
- Краткая пользовательская ошибка печатается в `stderr`, если `--debug` выключен.
- Structured runtime-логи через `slog` пишутся в `stderr` только при `--debug`.
- `--verbose` не должен смешиваться с `--debug`: при `--debug` verbose reporter отключён.
- Если `stdout` — TTY и `--raw` не задан, ответ рендерится через `glamour`; иначе печатается raw Markdown.
- При `--interaction` CLI принимает follow-up строки до `/exit` или EOF; `/reset` возвращает состояние к моменту сразу после первого ответа, пустые строки игнорируются.

## 4. Режимы обработки

### 4.1 `hybrid`

Назначение: начать с semantic retrieval, а затем дать модели проверить релевантность, дочитать локальный контекст и при необходимости перезапустить retrieval через tools.

Режим должен:

- для `single file` и `stdin` сначала проверять, помещается ли весь текст в safe budget окна chat-модели;
- если помещается, сразу передавать весь текст в prompt как untrusted data и отвечать без индекса, seeded retrieval и tool loop;
- заранее строить или переиспользовать тот же локальный индекс, что используют `tools --rag` и `rag`;
- до первого model turn выполнять fused semantic seed: до 5 deterministic `search_rag` запросов по variants исходного вопроса;
- класть preloaded search phase в историю как assistant/tool batch с несколькими `search_rag` calls и их JSON tool-envelopes;
- после semantic seed выполнять synthetic `read_lines`/`read_around` по strongest non-overlapping hit'ам и добавлять их второй preloaded phase в историю;
- после seeded retrieval запускать тот же orchestration loop и тот же набор инструментов, что и `tools --rag`;
- позволять модели оценить preloaded seed bundle, исследовать файлы через file-tools и при необходимости повторно вызывать `search_rag` с уточнённым запросом;
- при слабом fused seed не принимать первый direct text answer без model-driven tool use и один раз требовать exact verification или refined retrieval;
- всегда дописывать в финал секцию `Sources:` по evidence results текущей tool session, где `Verified` строится из `search_file`/`read_lines`/`read_around`, а `Retrieved` — из `search_rag` после вычитания verified overlaps;
- считать `list_files` discovery-навигацией, а `search_file` учитывать как line-based evidence по найденным match lines;
- не останавливаться только из-за слабого seeded retrieval, если индекс успешно подготовлен.
- для directory input никогда не использовать direct-context fast path, даже если synthetic corpus короткий.
- при `--interaction` direct-context fast path должен оставаться direct-context chat-history path без автоматического переключения в retrieval/tools;
- при `--interaction` retrieval/tool path должен переиспользовать preloaded seed history и тот же tools session state; `/reset` возвращает baseline после первого ответа.

Инструменты `hybrid`:

- `list_files(limit, offset)` при directory input
- `search_file(path?, query, mode, limit, offset, context_lines)`
- `search_rag(path?, query, limit, offset)`
- `read_lines(path?, start_line, end_line)`
- `read_around(path?, line, before, after)`

Примечание по `Sources:`:

- в direct-context fast path секция `Sources:` тоже обязательна; она содержит одну citation на весь диапазон строк single-file/`stdin` input;
- в retrieval/tool path секция `Sources:` обязательна и строится как описано выше.

Опции `hybrid`:

| Flag | Env | Значение | Default |
| --- | --- | --- | --- |
| `--embedding-model` | `EMBEDDING_MODEL` | Embedding-модель для seeded retrieval и повторных `search_rag` | `text-embedding-nomic-embed-text-v1.5` |
| `--rag-top-k` | `RAG_TOP_K` | Максимум semantic candidates для `search_rag` | `8` |
| `--rag-chunk-size` | `RAG_CHUNK_SIZE` | Размер чанка индекса | `1800` |
| `--rag-chunk-overlap` | `RAG_CHUNK_OVERLAP` | Overlap соседних чанков | `200` |
| `--rag-index-ttl` | `RAG_INDEX_TTL` | TTL локального индекса | `24h` |
| `--rag-index-dir` | `RAG_INDEX_DIR` | Базовая директория индексов | `os.TempDir()/ragcli-index` |
| `--rag-rerank` | `RAG_RERANK` | Стратегия rerank для `search_rag` | `heuristic` |

### 4.2 `tools`

Назначение: agentic анализ файла через tool calling вместо загрузки всего текста в prompt.

Режим должен:

- отдавать модели набор локальных инструментов `list_files`, `search_file`, `read_lines`, `read_around`;
- по флагу `--rag` дополнительно прединдексировать вход и отдавать инструмент `search_rag`;
- для directory input давать модели способ перечислить доступные relative paths перед file-local reads;
- для directory input поддерживать corpus-wide search и file-local reads через обязательный `path` у `read_lines`/`read_around`;
- крутить tool loop до финального ответа, лимита ходов или forced finalization;
- кэшировать уже выполненные tool calls;
- отличать отсутствие прогресса от новых строк/нового контекста;
- ограничивать повторы, пустые финальные ответы и дублирующиеся вызовы;
- возвращать orchestration error, если модель зациклилась и не даёт содержательного финального ответа.

Инструменты должны поддерживать:

- `list_files(limit, offset)`
- `search_file(query, mode, limit, offset, context_lines)`
- `search_file(path?, query, mode, limit, offset, context_lines)`
- `search_rag(query, limit, offset)`
- `search_rag(path?, query, limit, offset)`
- `read_lines(path?, start_line, end_line)`
- `read_around(path?, line, before, after)`

Опции `tools`:

| Flag | Env | Значение | Default |
| --- | --- | --- | --- |
| `--rag` | `TOOLS_RAG` | Включить pre-index + `search_rag` | `false` |
| `--embedding-model` | `EMBEDDING_MODEL` | Embedding-модель для `search_rag` | `text-embedding-nomic-embed-text-v1.5` |
| `--rag-top-k` | `RAG_TOP_K` | Максимум semantic candidates для `search_rag` | `8` |
| `--rag-chunk-size` | `RAG_CHUNK_SIZE` | Размер чанка индекса для `tools --rag` | `1800` |
| `--rag-chunk-overlap` | `RAG_CHUNK_OVERLAP` | Overlap соседних чанков | `200` |
| `--rag-index-ttl` | `RAG_INDEX_TTL` | TTL локального индекса | `24h` |
| `--rag-index-dir` | `RAG_INDEX_DIR` | Базовая директория индексов | `os.TempDir()/ragcli-index` |
| `--rag-rerank` | `RAG_RERANK` | Стратегия rerank для `search_rag` | `heuristic` |

Публичные лимиты текущей реализации:

- максимум `20` turns в одном orchestration loop;
- guard на повторные no-progress/duplicate tool runs;
- JSON tool-envelope (`result` + `meta`, либо `error`) как единственный wire-format между локальным tool runner и моделью.
- при `--interaction` follow-up turns должны переиспользовать ту же history, evidence tracking, duplicate cache и progress guards; `/reset` возвращает их к baseline после первого ответа.

### 4.3 `rag`

Назначение: retrieval по локальному файловому индексу.

Режим должен:

- строить или переиспользовать индекс по hash входа, schema version, chunking params и embedding model;
- для директорий считать hash по упорядоченной последовательности `(relative path, raw content)`;
- хранить manifest, chunks и embeddings в отдельной директории индекса;
- при directory input chunk-ать каждый файл отдельно с file-local line numbers;
- выполнять fused semantic seed по нескольким deterministic query variants исходного вопроса;
- ранжировать fused hits, опционально rerank-ить и выбирать evidence;
- синтезировать ответ только по evidence chunks;
- не запускать tool orchestration и не делать synthetic exact verification reads;
- дописывать в финал секцию `Sources:` с relative paths внутри директории.
- при `--interaction` режим должен готовить index один раз и переиспользовать тот же `PreparedSearch` для follow-up вопросов;
- retrieval query follow-up turn должен включать предыдущие user questions как query-context, а финальный synthesis prompt — предыдущий Q/A transcript как fallible context;
- `/reset` должен возвращать transcript к состоянию сразу после первого ответа.

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

### 4.4 `map`

Назначение: запасная обработка длинного текста чанками с последующей агрегацией, если retrieval или tool calling использовать нельзя.

Режим должен:

- разбивать вход на чанки по approximate token budget, а не по фиксированным байтам;
- сохранять разбиение по строкам, а oversized single line дорезать по rune offsets;
- уметь параллелить map-фазу через worker pool размера `--concurrency`;
- выполнять reduce fan-in итерациями, пока не останется один блок фактов;
- строить финальный ответ по reduced facts;
- запускать self-critique и self-refine до финального возврата;
- при directory input работать по детерминированному synthetic corpus, где сохранены границы файлов.

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
- при `--interaction` follow-up turn заново выполняет обычный map pipeline по тому же source/options, но получает contextualized вопрос с предыдущими Q/A;
- `/reset` должен возвращать transcript к состоянию сразу после первого ответа.

### 4.5 `self-update`

Назначение: безопасно обновить `ragcli`, установленный как standalone binary из GitHub Releases.

Команда должна:

- принимать build-time version текущего бинаря и отклонять `dev`, empty и non-semver значения;
- смотреть только public GitHub Releases репозитория `borro/ragcli`;
- игнорировать draft и prerelease релизы;
- подбирать asset строго под текущие `GOOS/GOARCH`;
- требовать наличия `checksums.txt` в release assets;
- при `--check` не скачивать asset и не писать на диск;
- при обычном запуске скачивать asset и `checksums.txt`, валидировать checksum до любой замены бинаря и затем применять atomic replace;
- не использовать package-manager detection, target version selection, background checks или auto-update.

Опции `self-update`:

| Flag | Env | Значение | Default |
| --- | --- | --- | --- |
| `--check` | - | Только проверить доступность нового stable release без записи на диск | `false` |

## 5. Нефункциональные требования

### 5.1 Надёжность

- Все LLM и embeddings-запросы идут через retry wrapper с exponential backoff `1s`, `2s`, `4s`, ... .
- Отмена через `context.Context` должна прерывать запросы, backoff sleep и длинные пайплайны.
- `SIGINT` и `SIGTERM` должны переводиться в command context через `signal.NotifyContext`.
- `self-update` должен валидировать checksum до filesystem commit и по возможности оставлять старый бинарь нетронутым при любой ошибке download/validation.

### 5.2 Локальные артефакты и приватность

- `stdin` сохраняется во временный файл только на время обработки и затем удаляется.
- Индексы `hybrid`, `tools --rag` и `rag` лежат на локальной файловой системе и публикуются атомарно через temp dir + rename.
- Права на индексные файлы и временные директории должны быть приватными (`0700` для директорий, `0600` для файлов).
- Просроченные индексы должны очищаться по TTL.

### 5.3 Наблюдаемость

- Debug-логи должны содержать информацию о стадиях пайплайна, попытках retry, токенах, выбранных режимах proxy и важных runtime-счётчиках.
- Для AI tool calls debug-логи должны показывать ключевые нормализованные аргументы хотя бы в событиях старта и ошибки.
- Verbose-режим должен показывать человекочитаемый stage/progress и compact параметры tool-вызова, но не заменять debug-логи.
- Source path в debug-логах должен быть укорочен до относительного пути от корня проекта.

### 5.4 Локализация

- Интерфейс CLI локализуется через встроенные каталоги `en` и `ru`, распределённые по owning-пакетам и зарегистрированные в `internal/localize`.
- Override приоритетов: `--lang`, затем `RAGCLI_LANG`, затем системная locale, затем fallback `en`.
- Неподдерживаемый явный `--lang` должен завершать CLI с пользовательской ошибкой; невалидные env/system locale должны тихо приводиться к fallback `en`.
- Язык интерфейса CLI и язык ответа модели не обязаны совпадать: ответ модели следует вопросу пользователя.

## 6. Формат результата

- `hybrid` всегда дописывает секцию `Sources:`; на retrieval/tool path внутри неё секции `Verified:` и `Retrieved:` строятся по evidence, а на direct-context fast path выводится одна citation на весь приложенный single-file/`stdin` диапазон строк. Пути группируются по файлу, строки сортируются, пересекающиеся диапазоны схлопываются, одиночная строка выводится как одно число.
- `rag` всегда дописывает секцию `Sources:` с тем же форматом grouped-by-file line ranges.
- `map` возвращает чистый ответ модели без секции `Sources`.
- Если evidence не найдено, `rag` должен возвращать честный insufficient-data ответ и `Sources:\n- none`.
- Raw output перед печатью обрезается по краям `strings.TrimSpace`.

## 7. Критичные инварианты поддержки

- При добавлении новой команды её нужно зарегистрировать в `internal/app/commandspec.go`, включить в help и в progress template.
- При изменении defaults нужно синхронизировать код, тесты, `doc/requirements.md` и корневой `README.md`.
- При изменении поведения режима нужно обновлять также package-level `README.md` соответствующего пакета.
- При изменении release asset naming или validation policy нужно синхронизировать `.goreleaser.yml`, CI и `internal/selfupdate`.
