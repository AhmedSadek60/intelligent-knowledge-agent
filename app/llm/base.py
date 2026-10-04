"""Provider-agnostic LLM interface.

Nothing in this module knows about Google, retrieval, vector search, memory,
Tavily or citation validation. Adapters implement ``LLMProvider``; the rest of
the application depends only on the types defined here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.2
    max_output_tokens: int = 2048
    timeout_seconds: float = 60.0


@dataclass(frozen=True)
class ToolDeclaration:
    """A tool the model may ask for. ``parameters`` is a JSON Schema object."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation requested by the model. The application executes it."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    finish_reason: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    usage: dict[str, int] = field(default_factory=dict)


class LLMError(Exception):
    """Base class for every error raised by an LLM provider."""


class LLMConfigurationError(LLMError):
    """Missing or invalid configuration (API key, model name, numeric settings)."""


class LLMAuthError(LLMError):
    """The provider rejected the credentials."""


class LLMRateLimitError(LLMError):
    """Quota or rate limit exceeded."""


class LLMTimeoutError(LLMError):
    """The request did not complete within the configured timeout."""


class LLMProviderError(LLMError):
    """Any other failure reported by the provider or the network."""


class LLMResponseError(LLMError):
    """The provider answered, but with no usable content (blocked, empty, truncated)."""


class LLMProvider(ABC):
    """Text generation over a list of conversation messages."""

    model: str

    @abstractmethod
    def generate(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        config: GenerationConfig | None = None,
        tools: Sequence[ToolDeclaration] | None = None,
    ) -> LLMResponse:
        """Generate the next assistant turn.

        If ``tools`` are given the response may contain ``tool_calls`` instead
        of (or alongside) text. Providers never execute tools themselves.
        """

    @abstractmethod
    def check(self) -> None:
        """Verify credentials and model availability. Raises ``LLMError`` on failure."""
