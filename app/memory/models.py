"""Memory models and data structures for J.A.R.V.I.S.

Defines the ConversationMemory dataclass representing a discrete dialogue turn,
role taxonomy, importance scoring, multi-conversation tracking, contextual
metadata, and disk-persistence-ready serialization.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Final, Optional, Union

from app.core.container import JarvisException

DEFAULT_MEMORY_CAPACITY: Final[int] = 100
DEFAULT_CONVERSATION_ID: Final[str] = "default"
VALID_ROLES: Final[frozenset[str]] = frozenset(
    {"user", "assistant", "system", "tool", "function"}
)
VALID_SOURCES: Final[frozenset[str]] = frozenset(
    {"voice", "text", "system", "automation", "tool", "cli", "api"}
)


class MemoryError(JarvisException):
    """Base exception for all memory subsystem errors."""


class InvalidMemoryError(MemoryError):
    """Raised when memory content or parameters are invalid."""


class MemoryCapacityError(MemoryError):
    """Raised when an invalid memory capacity is configured."""


@dataclass
class ConversationMemory:
    """Represents a discrete conversational dialogue turn with rich metadata.

    Attributes:
        content: Textual content or message payload.
        role: Entity generating the message ('user', 'assistant', 'system', 'tool').
        id: Unique UUID tracking identifier across logs, events, AI, and memory.
        conversation_id: Multi-conversation tracking session ID (default: 'default').
        source: Input channel source ('voice', 'text', 'system', 'automation', 'tool').
        importance: Numerical score from 1 to 10 for future pruning/promotion (default: 1).
        timestamp: Unix epoch timestamp marking creation.
        metadata: Arbitrary contextual metadata (intent_id, confidence, window, etc.).
        tags: Categorical tags for keyword indexing and discovery.
    """

    content: str
    role: str = "user"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = DEFAULT_CONVERSATION_ID
    source: str = "text"
    importance: int = 1
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate and sanitize memory fields upon initialization."""
        # 1. Content validation
        if not isinstance(self.content, str) or not self.content.strip():
            raise InvalidMemoryError("Memory content must be a non-empty string.")
        self.content = self.content.strip()

        # 2. Role normalization & validation
        if not isinstance(self.role, str) or not self.role.strip():
            self.role = "user"
        else:
            clean_role = self.role.strip().lower()
            self.role = clean_role if clean_role in VALID_ROLES else "user"

        # 3. UUID validation
        if not self.id:
            self.id = str(uuid.uuid4())
        elif isinstance(self.id, uuid.UUID):
            self.id = str(self.id)
        else:
            self.id = str(self.id).strip()

        # 4. Conversation ID
        if not isinstance(self.conversation_id, str) or not self.conversation_id.strip():
            self.conversation_id = DEFAULT_CONVERSATION_ID
        else:
            self.conversation_id = self.conversation_id.strip()

        # 5. Source channel validation
        if not isinstance(self.source, str) or not self.source.strip():
            self.source = "text"
        else:
            clean_source = self.source.strip().lower()
            self.source = clean_source if clean_source in VALID_SOURCES else "text"

        # 6. Importance validation (1 to 10)
        try:
            imp = int(self.importance)
            self.importance = max(1, min(10, imp))
        except (ValueError, TypeError):
            self.importance = 1

        # 7. Timestamp validation
        if not isinstance(self.timestamp, (int, float)):
            self.timestamp = time.time()

        # 8. Metadata validation
        if self.metadata is None or not isinstance(self.metadata, dict):
            self.metadata = {}
        else:
            self.metadata = dict(self.metadata)

        # 9. Tags validation
        if self.tags is None or not isinstance(self.tags, (list, tuple, set)):
            self.tags = []
        else:
            self.tags = [str(t).strip().lower() for t in self.tags if str(t).strip()]

    def to_dict(self) -> dict[str, Any]:
        """Convert the ConversationMemory instance to a dictionary for serialization.

        Returns:
            JSON-serializable dictionary representation of the memory.
        """
        return {
            "id": self.id,
            "content": self.content,
            "role": self.role,
            "conversation_id": self.conversation_id,
            "source": self.source,
            "importance": self.importance,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
            "tags": list(self.tags),
        }

    def to_json(self) -> str:
        """Serialize memory record to JSON string format for persistence.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationMemory:
        """Create a ConversationMemory instance from a dictionary.

        Args:
            data: Dictionary containing memory attributes.

        Returns:
            Instantiated and validated ConversationMemory.

        Raises:
            TypeError: If data is not a dictionary.
            InvalidMemoryError: If required fields are missing or invalid.
        """
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict for from_dict, got {type(data).__name__}")

        if "content" not in data:
            raise InvalidMemoryError("Missing required 'content' field in memory dictionary.")

        return cls(
            content=str(data.get("content", "")),
            role=str(data.get("role", "user")),
            id=str(data.get("id", uuid.uuid4())),
            conversation_id=str(data.get("conversation_id", DEFAULT_CONVERSATION_ID)),
            source=str(data.get("source", "text")),
            importance=int(data.get("importance", 1)),
            timestamp=float(data.get("timestamp", time.time())),
            metadata=dict(data.get("metadata", {})),
            tags=list(data.get("tags", [])),
        )

    @classmethod
    def from_json(cls, json_str: str) -> ConversationMemory:
        """Create a ConversationMemory instance from a JSON string.

        Args:
            json_str: JSON formatted string.

        Returns:
            Instantiated ConversationMemory.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def matches(
        self,
        query: Optional[str] = None,
        role: Optional[str] = None,
        conversation_id: Optional[str] = None,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Evaluate if this memory record matches search criteria.

        Args:
            query: Optional substring query against content or tags.
            role: Optional role filter.
            conversation_id: Optional conversation_id filter.
            metadata_filter: Optional key-value dictionary matching required metadata.

        Returns:
            True if all active filter conditions match, False otherwise.
        """
        # Role filter
        if role is not None and isinstance(role, str) and role.strip():
            if self.role != role.strip().lower():
                return False

        # Conversation ID filter
        if conversation_id is not None and isinstance(conversation_id, str) and conversation_id.strip():
            if self.conversation_id != conversation_id.strip():
                return False

        # Metadata filter
        if metadata_filter and isinstance(metadata_filter, dict):
            for k, v in metadata_filter.items():
                if self.metadata.get(k) != v:
                    return False

        # Query filter
        if query is not None and isinstance(query, str) and query.strip():
            q = query.strip().lower()
            in_content = q in self.content.lower()
            in_tags = any(q in tag for tag in self.tags)
            in_role = q == self.role
            if not (in_content or in_tags or in_role):
                return False

        return True

    def __repr__(self) -> str:
        """Developer-friendly string representation."""
        preview = self.content[:25] + "..." if len(self.content) > 25 else self.content
        return (
            f"<ConversationMemory(id={self.id[:8]}..., conv={self.conversation_id!r}, "
            f"role={self.role!r}, importance={self.importance}, content={preview!r})>"
        )
