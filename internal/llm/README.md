# `internal/llm`

## TL;DR

Пакет адаптирует `go-openai` под нужды проекта: chat/embeddings клиенты, retries, proxy configuration, metrics и auto-detect model context length.

См. также [`doc/architecture.md`](../../doc/architecture.md).

## Зона ответственности

- создание chat и embeddings клиентов;
- единый `Config` для LLM runtime;
- retry wrapper с exponential backoff;
- request/embedding metrics;
- proxy/no-proxy transport behavior;
- auto context length resolve для режима `map`;
- sanitization helpers для untrusted retrieved/generated text blocks.

## Ключевые entrypoints/types

- `Config`
- `NewClient(cfg)`
- `NewEmbedder(cfg)`
- `ChatRequester`
- `ChatAutoContextRequester`
- `EmbeddingRequester`
- `RequestMetrics`
- `EmbeddingMetrics`
- `SanitizeUntrustedBlock(raw)`
- `IsInstructionLikeLine(line)`

## Входящие/исходящие зависимости

Входящие:

- `internal/app`
- `internal/tools`
- `internal/ragcore`
- `internal/retrieval`
- `internal/map`

Исходящие:

- `github.com/sashabaranov/go-openai`
- `net/http`

## Основной поток

1. `Config` превращается в `go-openai` client config и HTTP transport.
2. Chat и embeddings запросы выполняются через `runWithRetry`.
3. После успешного запроса пакет возвращает ответ вместе с метриками.
4. Для `map` пакет умеет автоопределять контекст модели через LM Studio endpoint или probe по ошибке.

## Инварианты и ошибки

- Retry delay растёт экспоненциально: 1s, 2s, 4s, ...
- `NoProxy` сильнее `ProxyURL`.
- Метрики всегда соответствуют успешной попытке, а не последней неуспешной.
- Auto context resolve может завершиться fallback-ошибкой; вызывающий код обязан иметь дефолтное поведение.

## Что подтверждают тесты

- retry behavior и cancellation during backoff;
- proxy precedence и invalid proxy URL;
- request/embedding metrics и explicit model override;
- detection model context length по разным сценариям.

## Куда вносить изменения

- Изменение retry policy: `retry.go`.
- Изменение chat/embeddings transport или metrics: `chat.go`, `embedder.go`, `client.go`.
- Изменение auto context resolve: `auto_context.go`.
