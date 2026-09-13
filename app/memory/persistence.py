"""Conversation History Persistence Subsystem for J.A.R.V.I.S. Phase 24.

Provides thread-safe, atomic, and bounded persistence for ConversationMemory records
to/from JSON files (defaulting to data/conversation_history.json). Handles missing
files, directory creation, corrupted JSON gracefully, redacts sensitive keys/tokens,
and guarantees zero crashes on startup or during disk I/O.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Any, Final, Optional, Sequence, Union

from app.core.logger import get_logger
from app.memory.models import (
    DEFAULT_CONVERSATION_ID,
    ConversationMemory,
    InvalidMemoryError,
)

DEFAULT_HISTORY_FILE: Final[str] = os.path.join("data", "conversation_history.json")
DEFAULT_MAX_PERSISTED_ENTRIES: Final[int] = 100

# Patterns for redacting sensitive secrets before writing to disk
REDACT_PATTERNS: Final[list[re.Pattern]] = [
    re.compile(r"(?i)((?:api[_-]?)?key|secret[_-]?key|access[_-]?token|password|auth[_-]?token|token)\s*[:=]\s*['\"]?([A-Za-z0-9_\-\.]{8,})['\"]?"),
    re.compile(r"(?i)bearer\s+([A-Za-z0-9_\-\.]{16,})"),
    re.compile(r"(AIzaSy[A-Za-z0-9_\-]{20,45})"),  # Google API key pattern
    re.compile(r"(sk-[A-Za-z0-9_\-]{20,60})"),     # OpenAI API key pattern
]


def redact_secrets(text: str) -> str:
    """Sanitize text to redact API keys, tokens, and credentials before persistence.

    Args:
        text: Raw text string.

    Returns:
        Sanitized string with sensitive tokens masked.
    """
    if not isinstance(text, str) or not text:
        return ""

    sanitized = text
    for pattern in REDACT_PATTERNS:
        sanitized = pattern.sub(r"[REDACTED_SECRET]", sanitized)
    return sanitized


class ConversationHistoryPersistence:
    """Thread-safe, atomic, bounded file persistence adapter for ConversationMemory.

    Features:
    - Atomic writes via NamedTemporaryFile + fsync + os.replace.
    - Graceful recovery: corrupt or missing JSON files return empty history without crashing.
    - Enforced maximum history bounds (default 100 entries).
    - Secret redaction protection.
    - Synchronous and asynchronous lifecycle binding to MemoryManager.
    """

    def __init__(
        self,
        file_path: Optional[Union[str, Path]] = None,
        *,
        max_entries: int = DEFAULT_MAX_PERSISTED_ENTRIES,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize ConversationHistoryPersistence.

        Args:
            file_path: Optional path to JSON persistence file. Defaults to 'data/conversation_history.json'.
            max_entries: Maximum number of conversation records to retain on disk.
            logger: Optional Logger instance.
        """
        self._lock = threading.RLock()
        self._max_entries = max(1, int(max_entries))
        self._logger = logger if logger is not None else get_logger("CONVERSATION_PERSISTENCE")

        resolved_path = Path(file_path) if file_path is not None else Path(DEFAULT_HISTORY_FILE)
        # Ensure absolute path resolution
        self._file_path = resolved_path.resolve()

        self._ensure_directory()

    @property
    def file_path(self) -> Path:
        """Return the target history JSON file path."""
        return self._file_path

    @property
    def max_entries(self) -> int:
        """Return maximum entries limit."""
        return self._max_entries

    def _ensure_directory(self) -> None:
        """Ensure parent directory exists safely."""
        try:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self._logger.warning("Could not create history directory '%s': %s", self._file_path.parent, exc)

    def load_history(self) -> list[ConversationMemory]:
        """Load conversation history from the persistence file safely.

        Returns:
            List of chronologically ordered ConversationMemory instances.
            Returns empty list if file is missing, empty, or corrupt.
        """
        with self._lock:
            if not self._file_path.exists():
                self._logger.debug("History file '%s' does not exist yet; returning empty list.", self._file_path)
                return []

            try:
                with open(self._file_path, mode="r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if not content:
                        return []
                    data = json.loads(content)

                if not isinstance(data, list):
                    self._logger.warning("Corrupt history format in '%s' (not a JSON list); returning empty list.", self._file_path)
                    return []

                memories: list[ConversationMemory] = []
                for item in data:
                    if isinstance(item, dict):
                        try:
                            mem = ConversationMemory.from_dict(item)
                            memories.append(mem)
                        except Exception as exc:
                            self._logger.debug("Skipping invalid record in history: %s", exc)

                # Clamp to max entries
                if len(memories) > self._max_entries:
                    memories = memories[-self._max_entries:]

                return memories

            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                self._logger.warning("Malformed JSON in conversation history '%s': %s. Recovering gracefully.", self._file_path, exc)
                return []
            except Exception as exc:
                self._logger.error("Unexpected error reading history file '%s': %s", self._file_path, exc)
                return []

    def save_history(self, memories: Sequence[ConversationMemory]) -> bool:
        """Persist conversation memories atomically to the target file.

        Args:
            memories: Sequence of ConversationMemory instances.

        Returns:
            True if write succeeded, False otherwise.
        """
        with self._lock:
            try:
                self._ensure_directory()

                # Bound entries
                bounded = list(memories)[-self._max_entries:] if len(memories) > self._max_entries else list(memories)

                # Serialize records and sanitize secrets
                records: list[dict[str, Any]] = []
                for mem in bounded:
                    if isinstance(mem, ConversationMemory):
                        rec = mem.to_dict()
                        rec["content"] = redact_secrets(str(rec.get("content", "")))
                        records.append(rec)
                    elif isinstance(mem, dict):
                        rec = dict(mem)
                        rec["content"] = redact_secrets(str(rec.get("content", "")))
                        records.append(rec)

                json_str = json.dumps(records, ensure_ascii=False, indent=2)

                # Atomic write using NamedTemporaryFile in the target directory
                temp_fd, temp_path = tempfile.mkstemp(
                    dir=str(self._file_path.parent),
                    prefix="jarvis_conv_",
                    suffix=".tmp",
                )
                try:
                    with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                        f.write(json_str)
                        f.flush()
                        os.fsync(f.fileno())

                    # Atomic replacement
                    os.replace(temp_path, str(self._file_path))
                    return True
                except Exception as exc:
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except Exception:
                            pass
                    raise exc

            except Exception as exc:
                self._logger.error("Failed to persist conversation history to '%s': %s", self._file_path, exc)
                return False

    def append_turn(
        self,
        user_content: str,
        assistant_content: Optional[str] = None,
        *,
        conversation_id: str = DEFAULT_CONVERSATION_ID,
        source: str = "text",
        user_metadata: Optional[dict[str, Any]] = None,
        assistant_metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Convenience method to append a complete user/assistant dialogue turn safely.

        Args:
            user_content: Text of user command or query.
            assistant_content: Optional text of assistant response.
            conversation_id: Session identifier.
            source: Source channel ('text', 'voice', etc.).
            user_metadata: Metadata for user memory.
            assistant_metadata: Metadata for assistant memory.

        Returns:
            True if persistence succeeded, False otherwise.
        """
        with self._lock:
            history = self.load_history()

            if user_content and user_content.strip():
                history.append(
                    ConversationMemory(
                        content=user_content.strip(),
                        role="user",
                        conversation_id=conversation_id,
                        source=source,
                        metadata=user_metadata or {},
                    )
                )

            if assistant_content and assistant_content.strip():
                history.append(
                    ConversationMemory(
                        content=assistant_content.strip(),
                        role="assistant",
                        conversation_id=conversation_id,
                        source=source,
                        metadata=assistant_metadata or {},
                    )
                )

            return self.save_history(history)

    def clear_history(self) -> bool:
        """Clear all conversation history from the persistence file.

        Returns:
            True if cleared successfully.
        """
        with self._lock:
            return self.save_history([])

    def bind_to_manager(self, manager: Any) -> int:
        """Bind this persistence adapter to an existing MemoryManager instance.

        Loads historical records into the manager's store and hooks listener
        callbacks for future additions.

        Args:
            manager: MemoryManager instance.

        Returns:
            Count of historical memories populated.
        """
        with self._lock:
            loaded = self.load_history()
            count = 0
            if hasattr(manager, "_store"):
                for mem in loaded:
                    try:
                        manager._store.add(mem)
                        count += 1
                    except Exception as exc:
                        self._logger.debug("Failed adding loaded memory to store: %s", exc)

            self._logger.info("Bound persistence to MemoryManager, restored %d records.", count)

            # Subscribe to memory manager event bus if present
            bus = getattr(manager, "_event_bus", None)
            if bus is not None and hasattr(bus, "subscribe"):
                def _on_memory_added(event: Any) -> None:
                    try:
                        if hasattr(manager, "get_recent"):
                            recent = manager.get_recent(limit=self._max_entries)
                            self.save_history(recent)
                    except Exception as exc:
                        self._logger.debug("Failed syncing memory to disk on memory.added: %s", exc)

                def _on_memory_cleared(event: Any) -> None:
                    try:
                        self.clear_history()
                    except Exception as exc:
                        self._logger.debug("Failed clearing disk memory on memory.cleared: %s", exc)

                try:
                    bus.subscribe("memory.added", _on_memory_added)
                    bus.subscribe("memory.cleared", _on_memory_cleared)
                except Exception as exc:
                    self._logger.warning("Could not subscribe persistence to event bus: %s", exc)

            return count
