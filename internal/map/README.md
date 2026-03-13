# `internal/map`

## TL;DR

Пакет реализует режим `map`: approximate-token chunking, parallel map-фазу, iterative reduce и self-refine финального ответа.

См. также [`doc/architecture.md`](../../doc/architecture.md).

## Зона ответственности

- разбиение входного текста на чанки;
- построение map/reduce/final/critique/refine prompts;
- параллельный вызов LLM на map-фазе;
- схлопывание промежуточных фактов;
- self-critique и refine финального ответа.

## Ключевые entrypoints/types

- `Options`
- `Run(ctx, client, source, opts, question, plan)`
- `SplitByApproxTokens(...)`

## Входящие/исходящие зависимости

Входящие:

- `internal/app`
- `internal/hybrid` как fallback target

Исходящие:

- `internal/input`
- `internal/llm`
- `internal/verbose`
- `internal/localize`

## Основной поток

1. Определяется effective chunk length: явный `--length` или auto-detect.
2. Текст режется на чанки по approximate token budget.
3. По чанкам выполняются map-вызовы к LLM.
4. `SKIP` и пустые результаты отбрасываются.
5. Reduce-фаза схлопывает факты до одного блока.
6. Генерируется draft answer.
7. Draft проходит critique/refine и возвращается как результат режима.

## Инварианты и ошибки

- Пакет не использует embeddings.
- Chunking старается сохранять границы строк; oversized line режется отдельно.
- Пустой файл и отсутствие полезных map-results не считаются ошибкой исполнения.
- Approximate token budgeting завязан на эвристики, а не на точный tokenizer.

## Что подтверждают тесты

- корректность approximate token counting и chunk splitting;
- устойчивость к большим входам и отмене context;
- runtime поведение map-пайплайна и progress semantics.

## Куда вносить изменения

- Изменение chunking-эвристики: `chunking.go`.
- Изменение prompts или фаз пайплайна: `map.go`.
- Изменение пользовательских инвариантов режима: синхронизировать с `doc/requirements.md`.
- Локализованные prompts, progress и user-facing ответы режима: `i18n/{en,ru}.toml`.
