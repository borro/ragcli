# AGENTS

## Mission

- Работай как кодинг-агент по репозиторию `ragcli`: сначала пойми текущий код, потом меняй минимально нужные места.
- Считай задачу завершённой только после проверки релевантных тестов, ссылок и связанных docs.
- Не дублируй архитектуру в этом файле: используй его как оперативный handbook и навигатор.

## Source Of Truth

- Источник истины: текущий код и проходящие тесты.
- Общие требования и архитектура лежат в [`doc/README.md`](doc/README.md), [`doc/requirements.md`](doc/requirements.md), [`doc/architecture.md`](doc/architecture.md).
- Контекст по пакетам лежит рядом с кодом в `README.md` внутри `cmd/` и `internal/`.
- Если docs и код расходятся, сначала исправляй код или docs так, чтобы они снова совпали; не закрепляй drift.

## Repo Map

- [`cmd/ragcli`](cmd/ragcli) — тонкая точка входа, только передаёт управление в `internal/app`.
- [`internal/app`](internal/app) — CLI surface, binding флагов, lifecycle команды, wiring input/llm/output/progress.
- [`internal/map`](internal/map), [`internal/rag`](internal/rag), [`internal/tools`](internal/tools) — три режима обработки.
- [`internal/ragcore`](internal/ragcore) — единый shared RAG runtime для `rag`, `tools --rag` и `hybrid`.
- [`internal/aitools`](internal/aitools) — общий OpenAI-bound каркас для AI tools и registry-композиции.
- [`internal/aitools/files`](internal/aitools/files) — файловый домен AI tools: line-based readers, JSON-контракты и concrete file tools.
- [`internal/testutil`](internal/testutil) — shared test-only helpers; здесь живёт обёртка для `rapid` и mutation-mode стабилизации property-based тестов.
- [`internal/llm`](internal/llm), [`internal/input`](internal/input), [`internal/retrieval`](internal/retrieval), [`internal/verbose`](internal/verbose), [`internal/localize`](internal/localize), [`internal/logging`](internal/logging) — shared infrastructure.

## Standard Commands

- Форматирование Go-кода: `gofmt -w <files>`
- Линт как в CI: `golangci-lint run --timeout=5m`
- Базовая проверка: `go test ./...`
- Полная CI-like проверка локально: `go test -v -race -coverprofile=coverage.out ./...`
- Углублённый локальный property-based прогон: `go test ./... -args -rapid.checks=500`
- Локальный mutation dry/full run: `./scripts/mutation_test.sh [origin/master]`
- Live smoke script с реальными CLI-сценариями: `./scripts/e2e_live_smoke.sh`

## Non-Negotiable Invariants

- Не раздувай [`cmd/ragcli`](cmd/ragcli): бизнес-логика должна жить в `internal/app` и пакетах режимов.
- При изменении CLI-флагов, defaults или mode-пайплайнов обновляй соответствующие docs в `doc/` и package-level `README.md`.
- При изменении строк или ключей локализации вноси изменения во все файлы локализации, чтобы набор переводов оставался синхронным.
- При изменении локальных quality checks или CI-пайплайна держи `.github/workflows/ci.yaml` и `lefthook.yml` синхронными по набору обязательных проверок, если нет явно задокументированного исключения.
- Mutation testing — как раз такое задокументированное исключение: оно живёт отдельно в `.github/workflows/mutation.yaml`, `.gremlins.yaml` и `scripts/mutation_test.sh`, а не в `lefthook` и не в основном `ci.yaml`.
- Если трогаешь property-based тесты на `rapid` или добавляешь новые, используй shared helper из `internal/testutil`, чтобы mutation-mode (`RAGCLI_MUTATION=1`) оставался детерминированным и не расходился по пакетам.
- Для нового пакета с самостоятельной ролью добавляй рядом `README.md`.
- Сохраняй контракт вывода: финальный ответ в `stdout`, пользовательские ошибки и debug/verbose — в `stderr`.
- По возможности меняй один subsystem локально, а не размазывай правки по всему репозиторию без необходимости.
