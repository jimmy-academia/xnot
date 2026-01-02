"""
LLM token usage tracking and cost calculation.

Provides thread-safe tracking of all LLM API calls with:
- Token counts (prompt, completion)
- Cost calculation based on model pricing
- Latency measurement
- Aggregation and summary
"""

import json
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any


# Model pricing (per 1M tokens)
MODEL_PRICING = {
    # OpenAI models
    "gpt-5-nano": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "o1": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 3.00, "output": 12.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    # Anthropic models
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-opus-4-5-20251101": {"input": 15.00, "output": 75.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    # Default for unknown models
    "default": {"input": 1.00, "output": 3.00},
}


@dataclass
class LLMUsageRecord:
    """Record of a single LLM API call."""
    timestamp: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float


class LLMUsageTracker:
    """Thread-safe global tracker for LLM usage."""

    _instance: Optional['LLMUsageTracker'] = None
    _init_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._records: List[LLMUsageRecord] = []
                    instance._records_lock = threading.Lock()
                    cls._instance = instance
        return cls._instance

    def record(self, model: str, provider: str,
               prompt_tokens: int, completion_tokens: int,
               latency_ms: float) -> None:
        """Record a single LLM call.

        Args:
            model: Model name (e.g., "gpt-5-nano")
            provider: Provider name ("openai", "anthropic", "local")
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            latency_ms: API call latency in milliseconds
        """
        total_tokens = prompt_tokens + completion_tokens
        cost = self._calculate_cost(model, prompt_tokens, completion_tokens)

        record = LLMUsageRecord(
            timestamp=datetime.now().isoformat(),
            model=model,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
        )

        with self._records_lock:
            self._records.append(record)

    def _calculate_cost(self, model: str, prompt_tokens: int,
                        completion_tokens: int) -> float:
        """Calculate cost in USD based on model pricing."""
        pricing = MODEL_PRICING.get(model, MODEL_PRICING["default"])
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def get_summary(self) -> Dict[str, Any]:
        """Get aggregate summary of all recorded calls.

        Returns:
            Dict with total_calls, token counts, costs, and by_model breakdown
        """
        with self._records_lock:
            if not self._records:
                return {
                    "total_calls": 0,
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "total_tokens": 0,
                    "total_cost_usd": 0.0,
                    "total_latency_ms": 0.0,
                    "avg_latency_ms": 0.0,
                    "by_model": {},
                }

            total_prompt = sum(r.prompt_tokens for r in self._records)
            total_completion = sum(r.completion_tokens for r in self._records)
            total_cost = sum(r.cost_usd for r in self._records)
            total_latency = sum(r.latency_ms for r in self._records)

            # Group by model
            by_model: Dict[str, Dict[str, Any]] = {}
            for r in self._records:
                if r.model not in by_model:
                    by_model[r.model] = {
                        "calls": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "cost_usd": 0.0,
                    }
                by_model[r.model]["calls"] += 1
                by_model[r.model]["prompt_tokens"] += r.prompt_tokens
                by_model[r.model]["completion_tokens"] += r.completion_tokens
                by_model[r.model]["cost_usd"] += r.cost_usd

            # Round cost values for cleaner output
            for stats in by_model.values():
                stats["cost_usd"] = round(stats["cost_usd"], 6)

            return {
                "total_calls": len(self._records),
                "total_prompt_tokens": total_prompt,
                "total_completion_tokens": total_completion,
                "total_tokens": total_prompt + total_completion,
                "total_cost_usd": round(total_cost, 6),
                "total_latency_ms": round(total_latency, 2),
                "avg_latency_ms": round(total_latency / len(self._records), 2),
                "by_model": by_model,
            }

    def get_records(self) -> List[Dict[str, Any]]:
        """Get all records as list of dicts."""
        with self._records_lock:
            return [asdict(r) for r in self._records]

    def reset(self) -> None:
        """Clear all records. Call at start of new evaluation run."""
        with self._records_lock:
            self._records = []

    def save_to_file(self, path: Path) -> None:
        """Save all records to a JSONL file.

        Args:
            path: Output file path (typically usage.jsonl)
        """
        with self._records_lock:
            with open(path, "w") as f:
                for r in self._records:
                    f.write(json.dumps(asdict(r)) + "\n")


def get_usage_tracker() -> LLMUsageTracker:
    """Get the global usage tracker singleton."""
    return LLMUsageTracker()
