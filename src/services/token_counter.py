"""Rough token count + cost estimate for LLM calls."""
import os
from typing import Optional


def estimate_tokens(text: str) -> int:
    """Rough heuristic: ~4 chars per token."""
    return max(1, len(text) // 4)


def estimate_cost(prompt_tokens: int, completion_tokens: int = 0, model: Optional[str] = None) -> float:
    """Cost in USD. Defaults to a rough estimate for mercury-2."""
    model = model or os.getenv("MERCURY_MODEL", "mercury-2")
    pricing = {
        "mercury-2": {"in": 0.25 / 1_000_000, "out": 0.75 / 1_000_000},
        "mercury-2-mini": {"in": 0.10 / 1_000_000, "out": 0.30 / 1_000_000},
    }
    p = pricing.get(model, pricing["mercury-2"])
    return prompt_tokens * p["in"] + completion_tokens * p["out"]
