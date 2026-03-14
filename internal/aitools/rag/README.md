# `internal/aitools/rag`

## TL;DR

Пакет добавляет retrieval-domain AI tool `search_rag` для режима `tools`: он использует заранее подготовленный индекс из `internal/rag` и возвращает релевантные чанки в JSON.

## Зона ответственности

- concrete tool `search_rag`;
- JSON-контракт semantic retrieval результата, включая `chunk_id`, raw `similarity` и `overlap` для fused hybrid seed;
- локальный tool-level cache и pagination hints;
- преобразование retrieval hits в progress keys для orchestration loop.

## Входящие/исходящие зависимости

Входящие:

- `internal/tools`
- `internal/rag`

Исходящие:

- `internal/aitools`
- `internal/llm`
- `internal/localize`

## Инварианты

- индекс должен быть подготовлен заранее до первого LLM turn;
- tool не строит финальный ответ и не добавляет `Sources:`;
- duplicate cache и pagination ведут себя так же, как у file-tools.
