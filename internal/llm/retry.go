package llm

import (
	"context"
	"fmt"
	"time"
)

type sleepFunc func(ctx context.Context, delay time.Duration) error

type retryHooks[T any] struct {
	OnStart   func(attempt int)
	OnSuccess func(attempt int, value T, duration time.Duration)
	OnFailure func(attempt int, err error, duration time.Duration)
	OnRetry   func(attempt int, err error, delay time.Duration)
}

func runWithRetry[T any](
	ctx context.Context,
	attempts int,
	sleep sleepFunc,
	call func(context.Context) (T, error),
	hooks retryHooks[T],
) (T, error) {
	var zero T
	if sleep == nil {
		sleep = sleepContext
	}

	var lastErr error
	for attempt := 1; attempt <= attempts; attempt++ {
		if err := ctx.Err(); err != nil {
			return zero, err
		}
		if hooks.OnStart != nil {
			hooks.OnStart(attempt)
		}

		startedAt := time.Now()
		value, err := call(ctx)
		elapsed := time.Since(startedAt)
		if err == nil {
			if hooks.OnSuccess != nil {
				hooks.OnSuccess(attempt, value, elapsed)
			}
			return value, nil
		}

		lastErr = err
		if hooks.OnFailure != nil {
			hooks.OnFailure(attempt, err, elapsed)
		}

		if attempt < attempts {
			delay := retryDelay(attempt)
			if hooks.OnRetry != nil {
				hooks.OnRetry(attempt, err, delay)
			}
			if err := sleep(ctx, delay); err != nil {
				return zero, err
			}
		}
	}

	return zero, exhaustedAttemptsError(attempts, lastErr)
}

func retryDelay(attempt int) time.Duration {
	if attempt < 1 {
		return 0
	}
	return time.Second * time.Duration(1<<(attempt-1))
}

func exhaustedAttemptsError(attempts int, lastErr error) error {
	if lastErr == nil {
		return fmt.Errorf("request failed after %d attempts", attempts)
	}
	return fmt.Errorf("request failed after %d attempts: %w", attempts, lastErr)
}

func sleepContext(ctx context.Context, delay time.Duration) error {
	if delay <= 0 {
		if err := ctx.Err(); err != nil {
			return err
		}
		return nil
	}

	timer := time.NewTimer(delay)
	defer timer.Stop()

	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}
