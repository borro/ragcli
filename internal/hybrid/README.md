# `internal/hybrid`

## TL;DR

Пакет реализует режим `hybrid`: profile document, segment, выполнить lexical+semantic retrieval, дочитать контекст, извлечь факты, проверить coverage и при необходимости fallback-нуться в `map`.

См. также [`doc/architecture.md`](../../doc/architecture.md).

## Зона ответственности

- profiling документа или каждого файла корпуса как `markdown` / `logs` / `plain` / `unknown`;
- сегментация документа или нескольких файлов в регионы;
- lexical и semantic retrieval;
- fusion hits в regions;
- line-grounding и дочитывание вокруг лучших мест;
- map-style extraction фактов;
- coverage check и targeted reread;
- fallback semantics.

## Ключевые entrypoints/types

- `Options`
- `Run(ctx, chat, embedder, source, opts, question, plan)`
- `DocumentProfile`
- `Segment`
- `Region`
- `EvidenceBlock`
- `CoverageReport`

## Входящие/исходящие зависимости

Входящие:

- `internal/app`

Исходящие:

- `internal/input`
- `internal/llm`
- `internal/retrieval`
- `internal/tools/filetools`
- `internal/map`
- `internal/verbose`
- `internal/localize`

## Основной поток

1. Single-file input открывается через `input.SnapshotSource.Open()` и спулится во временный snapshot; directory input читается по списку файлов из `input.FileBackedSource.BackingFiles()`.
2. Документ или corpus сегментируется в meso-segments с file-aware metadata.
3. Выполняются lexical и semantic retrieval.
4. Хиты сливаются в regions и расширяются.
5. По лучшим regions дочитывается локальный контекст.
6. LLM извлекает факты по top regions.
7. Coverage check определяет, хватает ли evidence и нет ли конфликтов.
8. При необходимости строится follow-up query и запускается targeted reread.
9. Финальный ответ синтезируется по фактам и evidence либо выполняется fallback.

## Инварианты и ошибки

- Это единственный режим, который одновременно использует retrieval, line-based reread и map-style extraction.
- Для directory input регионы, evidence и citations не переходят через границы файлов.
- Fallback управляется только нормализованным `Options.Fallback`.
- Пустой input и отсутствие сегментов обрабатываются как корректные user-facing ответы.
- `hybrid` зависит от `map` только как от fallback pipeline, а не как от обязательной фазы.

## Что подтверждают тесты

- document profiling и segmentation для markdown/plain/logs;
- fallback на `map` при ошибках embeddings;
- schema/TTL validation индексного кэша;
- targeted reread и coverage-driven follow-up;
- наличие секции `Sources:` в финальном ответе.

## Куда вносить изменения

- Profiling/segmentation/fusion: `hybrid.go`.
- Общие retrieval инварианты: `internal/retrieval`.
- Если меняется fallback semantics, обязательно обновляйте `doc/requirements.md`.
- Локализованные prompts, progress и user-facing ответы режима: `i18n/{en,ru}.toml`.
