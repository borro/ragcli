# Документация `ragcli`

Этот каталог содержит канонический набор документов по требованиям и архитектуре проекта.

Repo-level инструкции для кодинг-агентов лежат в [`../AGENTS.md`](../AGENTS.md).

## Правило источника истины

- Источником истины считается текущий код в репозитории и проходящие тесты.
- Если `README`, старые заметки и код расходятся, приоритет у кода и тестов.
- Общие документы живут в `doc/`, а package-level контекст лежит рядом с пакетами в `README.md`.

## Общие документы

- [`requirements.md`](requirements.md) — продуктовые и технические требования, CLI-контракт, режимы, конфигурация и нефункциональные инварианты.
- [`architecture.md`](architecture.md) — слои системы, зависимости, пайплайны режимов, файловые артефакты и диаграммы.

## Package-Level Документы

### Точка входа и orchestration

- [`../cmd/ragcli/README.md`](../cmd/ragcli/README.md)
- [`../internal/app/README.md`](../internal/app/README.md)

### Режимы обработки

- [`../internal/hybrid/README.md`](../internal/hybrid/README.md)
- [`../internal/tools/README.md`](../internal/tools/README.md)
- [`../internal/rag/README.md`](../internal/rag/README.md)
- [`../internal/map/README.md`](../internal/map/README.md)

### Общая инфраструктура

- [`../internal/aitools/README.md`](../internal/aitools/README.md)
- [`../internal/aitools/files/README.md`](../internal/aitools/files/README.md)
- [`../internal/ragcore/README.md`](../internal/ragcore/README.md)
- [`../internal/llm/README.md`](../internal/llm/README.md)
- [`../internal/input/README.md`](../internal/input/README.md)
- [`../internal/retrieval/README.md`](../internal/retrieval/README.md)
- [`../internal/verbose/README.md`](../internal/verbose/README.md)
- [`../internal/verbose/testutil/README.md`](../internal/verbose/testutil/README.md)
- [`../internal/localize/README.md`](../internal/localize/README.md)
- [`../internal/logging/README.md`](../internal/logging/README.md)

## Как поддерживать документацию

- При изменении CLI-флагов или defaults сначала обновляйте код и тесты, затем `doc/requirements.md` и корневой `README.md`.
- При изменении пайплайна одного режима обновляйте `doc/architecture.md` и package-level `README.md` соответствующего пакета.
- При добавлении нового пакета с самостоятельной ролью добавляйте рядом с ним `README.md` и ссылку на него из этого индекса.
- Live smoke с реальным backend'ом запускается через `scripts/e2e_live_smoke.sh`; `lefthook` вешает его на `pre-push`, а не на `pre-commit`.
