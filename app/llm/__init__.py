"""LLM integration for the Intelligent Knowledge Assistant."""

from __future__ import annotations

from .base import (
    GenerationConfig,
    LLMAuthError,
    LLMConfigurationError,
    LLMError,
    LLMProvider,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponse,
    LLMResponseError,
    LLMTimeoutError,
    Message,
    ToolCall,
    ToolDeclaration,
)
from .config import DEFAULT_MODEL, LLMSettings
from .conversation import MessageStore, SQLiteMessageStore
from .prompts import INSUFFICIENT_EVIDENCE, RAG_SYSTEM_PROMPT, Passage, build_rag_user_message
from .service import KnowledgeAssistant
from .tools import ToolRegistry, ToolValidationError, run_with_tools


def create_provider(settings: LLMSettings | None = None) -> LLMProvider:
    """Build the configured provider. The Google SDK is imported only here."""
    from .gemma import GoogleGemmaProvider

    settings = settings or LLMSettings.from_env()
    return GoogleGemmaProvider(
        api_key=settings.api_key, model=settings.model, default_config=settings.generation
    )


__all__ = [
    "DEFAULT_MODEL",
    "INSUFFICIENT_EVIDENCE",
    "RAG_SYSTEM_PROMPT",
    "GenerationConfig",
    "KnowledgeAssistant",
    "LLMAuthError",
    "LLMConfigurationError",
    "LLMError",
    "LLMProvider",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMResponse",
    "LLMResponseError",
    "LLMSettings",
    "LLMTimeoutError",
    "Message",
    "MessageStore",
    "Passage",
    "SQLiteMessageStore",
    "ToolCall",
    "ToolDeclaration",
    "ToolRegistry",
    "ToolValidationError",
    "build_rag_user_message",
    "create_provider",
    "run_with_tools",
]
