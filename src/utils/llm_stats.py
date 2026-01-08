"""
LLM Statistics Tracking.

Tracks token usage, costs, timing, and retry stats across all LLM calls.
"""
from dataclasses import dataclass, field
from typing import Any
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class LLMStatistics:
    """Statistics for LLM usage."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0

    # Token tracking
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    # Timing
    total_time: float = 0.0

    # Retry stats
    rate_limit_errors: int = 0
    retry_attempts: int = 0
    retry_successes: int = 0
    total_retry_wait_time: float = 0.0

    # Per-shard tracking
    shard_stats: dict = field(default_factory=dict)

    def add_call(
        self,
        success: bool,
        shard_id: str = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        elapsed_time: float = 0.0
    ):
        """Record an LLM call."""
        self.total_calls += 1
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_time += elapsed_time

        # Track per-shard
        if shard_id:
            if shard_id not in self.shard_stats:
                self.shard_stats[shard_id] = {"calls": 0, "time": 0.0, "success": True}
            self.shard_stats[shard_id]["calls"] += 1
            self.shard_stats[shard_id]["time"] += elapsed_time
            if not success:
                self.shard_stats[shard_id]["success"] = False

    def add_retry(self, wait_time: float = 0.0, is_rate_limit: bool = False):
        """Record a retry attempt."""
        self.retry_attempts += 1
        self.total_retry_wait_time += wait_time
        if is_rate_limit:
            self.rate_limit_errors += 1

    def add_retry_success(self):
        """Record a successful retry."""
        self.retry_successes += 1

    def get_summary(self) -> dict[str, Any]:
        """Get summary of statistics."""
        total_tokens = self.total_prompt_tokens + self.total_completion_tokens

        # Cost estimation (Gemini 2.5 Flash pricing - adjust as needed)
        # Input: $0.075/1M tokens, Output: $0.30/1M tokens
        estimated_cost = (
            (self.total_prompt_tokens * 0.000000075) +
            (self.total_completion_tokens * 0.0000003)
        )

        return {
            "calls": {
                "total": self.total_calls,
                "successful": self.successful_calls,
                "failed": self.failed_calls,
                "success_rate": f"{(self.successful_calls / max(self.total_calls, 1)) * 100:.1f}%"
            },
            "tokens": {
                "total": total_tokens,
                "prompt": self.total_prompt_tokens,
                "completion": self.total_completion_tokens,
            },
            "timing": {
                "total_seconds": round(self.total_time, 2),
                "avg_per_call": round(self.total_time / max(self.total_calls, 1), 2),
            },
            "retries": {
                "attempts": self.retry_attempts,
                "successes": self.retry_successes,
                "rate_limit_errors": self.rate_limit_errors,
                "total_wait_time": round(self.total_retry_wait_time, 2),
            },
            "cost": {
                "estimated_usd": round(estimated_cost, 6),
            },
            "per_shard": self.shard_stats,
        }


# Global statistics instance
_stats = LLMStatistics()


def get_stats() -> LLMStatistics:
    """Get global statistics instance."""
    return _stats


def get_stats_summary() -> dict[str, Any]:
    """Get statistics summary."""
    return _stats.get_summary()


def reset_stats():
    """Reset statistics."""
    global _stats
    _stats = LLMStatistics()


class StatsTimer:
    """Context manager for timing LLM calls."""

    def __init__(self, shard_id: str = None):
        self.shard_id = shard_id
        self.start_time = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start_time
        return False  # Don't suppress exceptions
