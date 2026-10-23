"""Provider interface. Every model backend turns a neutral conversation into one model turn."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

# Neutral message format used by the runner:
#   {"role": "user", "content": str}
#   {"role": "assistant", "text": str, "tool_calls": [ToolCall-as-dict, ...]}
#   {"role": "tool", "results": [{"id", "name", "content": str, "is_error": bool}, ...]}


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class ModelTurn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    # Set by simulated providers; real providers leave None and the runner measures wall time.
    simulated_latency_s: float | None = None


class Provider(Protocol):
    name: str

    def chat(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelTurn: ...


class ProviderError(Exception):
    """The model call itself failed (auth, rate limit, network). Counted separately from agent mistakes."""


# USD per 1M tokens (input, output). Override per run with --price model=in,out.
# Real-provider prices change; check the vendor's pricing page before trusting cost numbers.
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "sim:frontier": (3.00, 15.00),
    "sim:fast": (0.80, 4.00),
    "sim:small-oss": (0.10, 0.40),
}


def price_for(model: str, overrides: dict[str, tuple[float, float]]) -> tuple[float, float]:
    if model in overrides:
        return overrides[model]
    if model in DEFAULT_PRICES:
        return DEFAULT_PRICES[model]
    if model.startswith(("ollama:", "vllm:")):
        return (0.0, 0.0)  # self-hosted: no per-token bill
    return (0.0, 0.0)
