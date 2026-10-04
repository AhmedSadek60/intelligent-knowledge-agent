"""LLM settings read from the server-side environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping

from .base import GenerationConfig, LLMConfigurationError

DEFAULT_MODEL = "gemma-4-26b-a4b-it"


@dataclass(frozen=True)
class LLMSettings:
    # repr=False keeps the key out of logs and tracebacks.
    api_key: str | None = field(default=None, repr=False)
    model: str = DEFAULT_MODEL
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    history_limit: int = 20

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "LLMSettings":
        env = os.environ if env is None else env
        defaults = GenerationConfig()
        temperature = _number(env, "GEMMA_TEMPERATURE", defaults.temperature, float)
        if not 0.0 <= temperature <= 2.0:
            raise LLMConfigurationError("GEMMA_TEMPERATURE must be between 0 and 2.")
        return cls(
            api_key=(env.get("GOOGLE_API_KEY") or "").strip() or None,
            model=(env.get("GEMMA_MODEL") or "").strip() or DEFAULT_MODEL,
            generation=GenerationConfig(
                temperature=temperature,
                max_output_tokens=_positive(env, "GEMMA_MAX_OUTPUT_TOKENS", defaults.max_output_tokens, int),
                timeout_seconds=_positive(env, "GEMMA_TIMEOUT_SECONDS", defaults.timeout_seconds, float),
            ),
            history_limit=_positive(env, "GEMMA_HISTORY_LIMIT", 20, int),
        )


def _number(env: Mapping[str, str], name: str, default, cast):
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError:
        raise LLMConfigurationError(f"{name} must be a number, got {raw!r}.") from None


def _positive(env: Mapping[str, str], name: str, default, cast):
    value = _number(env, name, default, cast)
    if value <= 0:
        raise LLMConfigurationError(f"{name} must be greater than 0.")
    return value
