"""Glue between conversation history, retrieved passages and the LLM provider.

Retrieval, vector search, memory, Tavily and citation validation live elsewhere:
callers pass in already-retrieved passages and validate the returned citations
themselves. This module does not write to the database either; persisting the
question and the answer stays with the application.
"""

from __future__ import annotations

from typing import Sequence

from .base import GenerationConfig, LLMProvider, LLMResponse, Message
from .conversation import MessageStore
from .prompts import RAG_SYSTEM_PROMPT, Passage, build_rag_user_message


class KnowledgeAssistant:
    def __init__(self, provider: LLMProvider, store: MessageStore, history_limit: int = 20) -> None:
        self._provider = provider
        self._store = store
        self._history_limit = history_limit

    def build_messages(
        self, conversation_id: str, question: str, passages: Sequence[Passage]
    ) -> list[Message]:
        history = self._store.load_messages(conversation_id, self._history_limit)
        # A truncated window can start mid-exchange; the API expects the dialogue to open with a user turn.
        while history and history[0].role != "user":
            history = history[1:]
        return [*history, Message("user", build_rag_user_message(question, passages))]

    def answer(
        self,
        conversation_id: str,
        question: str,
        passages: Sequence[Passage],
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        messages = self.build_messages(conversation_id, question, passages)
        return self._provider.generate(messages, system=RAG_SYSTEM_PROMPT, config=config)
