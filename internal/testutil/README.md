# `internal/testutil`

## TL;DR

Пакет хранит shared test-only helpers, которые можно переиспользовать между пакетами без протекания тестовой инфраструктуры в production code.

## Текущая роль

- `RapidCheck(...)` оборачивает `pgregory.net/rapid.Check`;
- при `RAGCLI_MUTATION=1` один раз стабилизирует `rapid`-флаги для mutation testing;
- вне mutation-режима не меняет обычное поведение property-based тестов.

## Инварианты

- пакет предназначен только для использования из тестов;
- mutation defaults не должны перекрывать явно переданные пользователем `rapid.*` flags;
- env overrides вида `RAGCLI_MUTATION_RAPID_*` используются только как tuning knobs для mutation runs.
