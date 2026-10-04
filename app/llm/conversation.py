"""Loading persisted conversation history from the application's own database.

The assistant depends only on the ``MessageStore`` protocol. ``SQLiteMessageStore``
is a read-only loader for a conventional ``messages`` table; it never creates or
alters schema. If the existing database differs, implement ``MessageStore``
against it instead.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Protocol

from .base import Message


class MessageStore(Protocol):
    def load_messages(self, conversation_id: str, limit: int) -> list[Message]:
        """Return up to ``limit`` most recent messages, oldest first."""
        ...


class SQLiteMessageStore:
    """Reads ``(conversation_id, role, content, created_at, id)`` rows from an existing table."""

    def __init__(self, db_path: str, table: str = "messages") -> None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
            raise ValueError(f"Invalid table name: {table!r}")
        self._db_path = db_path
        self._table = table

    def load_messages(self, conversation_id: str, limit: int) -> list[Message]:
        query = (
            f"SELECT role, content FROM {self._table} "
            "WHERE conversation_id = ? ORDER BY created_at DESC, id DESC LIMIT ?"
        )
        connection = sqlite3.connect(self._db_path)
        try:
            rows = connection.execute(query, (conversation_id, limit)).fetchall()
        finally:
            connection.close()
        # Rows with other roles (system, tool) are not part of the model-visible dialogue.
        return [Message(role, content) for role, content in reversed(rows) if role in ("user", "assistant")]
