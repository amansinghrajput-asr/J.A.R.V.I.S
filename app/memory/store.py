"""Short-term In-Memory Store for J.A.R.V.I.S.

Provides an O(1) indexed, thread-safe bounded FIFO memory buffer with
O(1) existence checks, O(1) removal, O(1) latest record access, and
comprehensive multi-dimensional search over content, tags, role, conversation ID,
and metadata.
"""

from __future__ import annotations

import collections
import threading
from typing import Any, Optional

from app.memory.models import (
    DEFAULT_MEMORY_CAPACITY,
    ConversationMemory,
    InvalidMemoryError,
    MemoryCapacityError,
)


class MemoryStore:
    """Thread-safe bounded in-memory store for short-term conversation context.

    Uses an OrderedDict to provide strict O(1) additions, O(1) removals,
    O(1) lookups, O(1) existence checks, and O(1) retrieval of the newest record.
    """

    def __init__(self, capacity: int = DEFAULT_MEMORY_CAPACITY) -> None:
        """Initialize the MemoryStore.

        Args:
            capacity: Maximum number of memories to retain. Defaults to 100.

        Raises:
            MemoryCapacityError: If capacity is less than 1.
        """
        if not isinstance(capacity, int) or capacity < 1:
            raise MemoryCapacityError(f"Memory capacity must be a positive integer, got: {capacity!r}")

        self._lock = threading.RLock()
        self._capacity = capacity
        self._entries: collections.OrderedDict[str, ConversationMemory] = collections.OrderedDict()

    @property
    def capacity(self) -> int:
        """Retrieve the maximum memory capacity."""
        with self._lock:
            return self._capacity

    @capacity.setter
    def capacity(self, new_capacity: int) -> None:
        """Dynamically update maximum memory capacity and trim if necessary.

        Args:
            new_capacity: New positive integer capacity.

        Raises:
            MemoryCapacityError: If new_capacity is less than 1.
        """
        if not isinstance(new_capacity, int) or new_capacity < 1:
            raise MemoryCapacityError(
                f"Memory capacity must be a positive integer, got: {new_capacity!r}"
            )

        with self._lock:
            self._capacity = new_capacity
            # Trim oldest items if current size exceeds new capacity
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)

    def add(self, memory: ConversationMemory) -> ConversationMemory:
        """Add a memory to the store in O(1) time.

        If the store is at maximum capacity, the oldest record is evicted in O(1).

        Args:
            memory: ConversationMemory instance to store.

        Returns:
            The stored ConversationMemory instance.

        Raises:
            InvalidMemoryError: If memory is not an instance of ConversationMemory.
        """
        if not isinstance(memory, ConversationMemory):
            raise InvalidMemoryError(
                f"Expected ConversationMemory instance, got {type(memory).__name__}"
            )

        with self._lock:
            # If item already exists, pop first so insertion moves it to the newest end
            if memory.id in self._entries:
                del self._entries[memory.id]
            elif len(self._entries) >= self._capacity:
                # Evict oldest entry in O(1)
                self._entries.popitem(last=False)

            self._entries[memory.id] = memory
            return memory

    def get(self, memory_id: str) -> Optional[ConversationMemory]:
        """Retrieve a specific memory by its unique ID in O(1) time.

        Args:
            memory_id: Unique UUID string identifier.

        Returns:
            The matching ConversationMemory instance if found, None otherwise.
        """
        if not isinstance(memory_id, str) or not memory_id.strip():
            return None

        clean_id = memory_id.strip()
        with self._lock:
            return self._entries.get(clean_id)

    def get_last(self) -> Optional[ConversationMemory]:
        """Retrieve the newest memory record in O(1) time.

        Returns:
            The newest ConversationMemory instance, or None if store is empty.
        """
        with self._lock:
            if not self._entries:
                return None
            return next(reversed(self._entries.values()))

    def exists(self, memory_id: str) -> bool:
        """Check if a memory record exists in O(1) time.

        Args:
            memory_id: Unique UUID string identifier.

        Returns:
            True if present, False otherwise.
        """
        if not isinstance(memory_id, str) or not memory_id.strip():
            return False

        clean_id = memory_id.strip()
        with self._lock:
            return clean_id in self._entries

    def remove(self, memory_id: str) -> bool:
        """Remove a memory record by ID in O(1) time.

        Args:
            memory_id: Unique UUID string identifier.

        Returns:
            True if the record was found and removed, False otherwise.
        """
        if not isinstance(memory_id, str) or not memory_id.strip():
            return False

        clean_id = memory_id.strip()
        with self._lock:
            return self._entries.pop(clean_id, None) is not None

    def get_recent(
        self,
        limit: Optional[int] = None,
        role: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> list[ConversationMemory]:
        """Retrieve recent conversation memories in chronological order.

        Args:
            limit: Maximum number of recent memories to return. If None, returns all.
            role: Optional role filter ('user', 'assistant', 'system', 'tool').
            conversation_id: Optional conversation ID filter.

        Returns:
            Chronologically ordered list of ConversationMemory instances.
        """
        with self._lock:
            memories = list(self._entries.values())

        # Filter by conversation_id
        if conversation_id is not None and isinstance(conversation_id, str) and conversation_id.strip():
            clean_conv = conversation_id.strip()
            memories = [m for m in memories if m.conversation_id == clean_conv]

        # Filter by role
        if role is not None and isinstance(role, str) and role.strip():
            clean_role = role.strip().lower()
            memories = [m for m in memories if m.role == clean_role]

        if limit is not None:
            if not isinstance(limit, int) or limit < 0:
                return []
            if limit > 0:
                memories = memories[-limit:]
            else:
                return []

        return memories

    def search(
        self,
        query: Optional[str] = None,
        limit: Optional[int] = None,
        role: Optional[str] = None,
        conversation_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[ConversationMemory]:
        """Search memories across content, tags, role, conversation_id, and metadata.

        Args:
            query: Keyword or substring query to search across content and tags.
            limit: Maximum number of search results to return.
            role: Optional role filter.
            conversation_id: Optional conversation ID filter.
            metadata: Optional key-value dictionary filter for metadata matching.

        Returns:
            Chronologically ordered list of matching memories.
        """
        with self._lock:
            memories = list(self._entries.values())

        matches = [
            m
            for m in memories
            if m.matches(
                query=query,
                role=role,
                conversation_id=conversation_id,
                metadata_filter=metadata,
            )
        ]

        if limit is not None:
            if not isinstance(limit, int) or limit <= 0:
                return []
            matches = matches[-limit:]

        return matches

    def clear(self) -> int:
        """Remove all memories from the store.

        Returns:
            The number of memories that were cleared.
        """
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            return count

    def count(self) -> int:
        """Return the current number of memories in the store."""
        with self._lock:
            return len(self._entries)

    def all(self) -> list[ConversationMemory]:
        """Return a snapshot of all stored memories in chronological order."""
        with self._lock:
            return list(self._entries.values())

    def __len__(self) -> int:
        """Support len(store) syntax."""
        return self.count()

    def __contains__(self, memory_id: str) -> bool:
        """Support `memory_id in store` syntax."""
        return self.exists(memory_id)

    def __repr__(self) -> str:
        """Developer-friendly string representation."""
        with self._lock:
            return f"<MemoryStore(count={len(self._entries)}, capacity={self._capacity})>"
