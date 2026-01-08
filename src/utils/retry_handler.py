"""
Retry Handler with Exponential Backoff.

Handles rate limits and transient failures for LLM calls.
"""
import asyncio
import random
import logging
from dataclasses import dataclass
from typing import Callable, Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 5
    min_wait: float = 1.0
    max_wait: float = 60.0
    multiplier: float = 2.0
    jitter: bool = True

    # Exceptions to retry on
    retryable_exceptions: tuple = (
        Exception,  # Will be refined based on actual exceptions
    )


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted."""
    pass


async def retry_with_backoff(
    func: Callable[..., T],
    *args,
    config: RetryConfig = None,
    on_retry: Callable[[int, Exception, float], None] = None,
    **kwargs
) -> T:
    """
    Execute async function with exponential backoff retry.

    Args:
        func: Async function to execute
        *args: Positional arguments for func
        config: Retry configuration
        on_retry: Callback called on each retry (attempt, exception, wait_time)
        **kwargs: Keyword arguments for func

    Returns:
        Result from successful function call

    Raises:
        RetryExhaustedError: If all attempts fail
    """
    if config is None:
        config = RetryConfig()

    last_exception = None
    wait_time = config.min_wait

    for attempt in range(1, config.max_attempts + 1):
        try:
            return await func(*args, **kwargs)

        except config.retryable_exceptions as e:
            last_exception = e

            # Check if this is the last attempt
            if attempt >= config.max_attempts:
                logger.error(f"All {config.max_attempts} retry attempts exhausted")
                raise RetryExhaustedError(
                    f"Failed after {config.max_attempts} attempts: {e}"
                ) from e

            # Calculate wait time with optional jitter
            if config.jitter:
                wait_time = wait_time * (1 + random.random() * 0.5)
            wait_time = min(wait_time, config.max_wait)

            # Check if rate limit error
            is_rate_limit = _is_rate_limit_error(e)

            logger.warning(
                f"Attempt {attempt}/{config.max_attempts} failed: {e}. "
                f"Retrying in {wait_time:.1f}s..."
            )

            # Call retry callback if provided
            if on_retry:
                on_retry(attempt, e, wait_time)

            # Wait before retry
            await asyncio.sleep(wait_time)

            # Increase wait time for next attempt
            wait_time *= config.multiplier

    # Should not reach here, but just in case
    raise RetryExhaustedError(f"Unexpected retry exhaustion: {last_exception}")


def _is_rate_limit_error(exception: Exception) -> bool:
    """Check if exception is a rate limit error."""
    error_str = str(exception).lower()
    rate_limit_indicators = [
        "rate limit",
        "rate_limit",
        "ratelimit",
        "429",
        "too many requests",
        "quota exceeded",
        "resource exhausted",
    ]
    return any(indicator in error_str for indicator in rate_limit_indicators)


def sync_retry_with_backoff(
    func: Callable[..., T],
    *args,
    config: RetryConfig = None,
    **kwargs
) -> T:
    """
    Synchronous version of retry_with_backoff.
    """
    import time

    if config is None:
        config = RetryConfig()

    last_exception = None
    wait_time = config.min_wait

    for attempt in range(1, config.max_attempts + 1):
        try:
            return func(*args, **kwargs)

        except config.retryable_exceptions as e:
            last_exception = e

            if attempt >= config.max_attempts:
                raise RetryExhaustedError(
                    f"Failed after {config.max_attempts} attempts: {e}"
                ) from e

            if config.jitter:
                wait_time = wait_time * (1 + random.random() * 0.5)
            wait_time = min(wait_time, config.max_wait)

            logger.warning(
                f"Attempt {attempt}/{config.max_attempts} failed: {e}. "
                f"Retrying in {wait_time:.1f}s..."
            )

            time.sleep(wait_time)
            wait_time *= config.multiplier

    raise RetryExhaustedError(f"Unexpected retry exhaustion: {last_exception}")
