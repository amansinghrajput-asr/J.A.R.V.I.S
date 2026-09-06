"""Memory Manager Subsystem for J.A.R.V.I.S.

Coordinates short-term conversational context storage, retrieval, keyword and
metadata search, capacity enforcement, item removal, and event notifications
across the system Event Bus.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Final, Optional, Union

from app.core.config import Settings, settings
from app.core.container import ServiceContainer, container
from app.core.event_bus import EventBus, event_bus
from app.core.logger import get_logger
from app.memory.models import (
    DEFAULT_CONVERSATION_ID,
    DEFAULT_MEMORY_CAPACITY,
    ConversationMemory,
    InvalidMemoryError,
    MemoryCapacityError,
    MemoryError,
)
from app.memory.store import MemoryStore


class MemoryManager:
    """Thread-safe, async-ready coordinator for short-term conversational memory.

    Manages an O(1) indexed bounded memory buffer, supports multi-conversation
    session tracking, handles message addition, removal, newest memory lookups,
    context retrieval, multi-attribute search, buffer clearing, and emits
    complete lifecycle events (added, retrieved, removed, cleared).

    Integrates with Config, Logger, Event Bus, and Service Container.
    """

    def __init__(
        self,
        capacity: Optional[int] = None,
        config: Optional[Settings] = None,
        logger: Optional[logging.Logger] = None,
        container_instance: Optional[ServiceContainer] = None,
        event_bus_instance: Optional[EventBus] = None,
        *,
        auto_register_in_container: bool = True,
    ) -> None:
        """Initialize the MemoryManager.

        Args:
            capacity: Optional maximum short-term memory capacity. Defaults to 100.
            config: Optional Settings instance. If None, resolves from container or global settings.
            logger: Optional Logger instance. If None, creates 'MEMORY' logger.
            container_instance: Optional ServiceContainer. If None, uses global container.
            event_bus_instance: Optional EventBus. If None, resolves from container or global event bus.
            auto_register_in_container: If True, registers this MemoryManager into container.
        """
        self._lock = threading.RLock()

        # 1. Dependency resolution: Service Container
        self._container = container_instance if container_instance is not None else container

        # 2. Dependency resolution: Logger
        self._logger = logger if logger is not None else get_logger("MEMORY")

        # 3. Dependency resolution: Configuration
        if config is not None:
            self._config = config
        elif self._container.exists("settings"):
            self._config = self._container.resolve("settings")
        elif self._container.exists("config"):
            self._config = self._container.resolve("config")
        else:
            self._config = settings

        # 4. Dependency resolution: Event Bus
        if event_bus_instance is not None:
            self._event_bus = event_bus_instance
        elif self._container.exists("event_bus"):
            self._event_bus = self._container.resolve("event_bus")
        else:
            self._event_bus = event_bus

        # 5. Determine capacity
        resolved_capacity = DEFAULT_MEMORY_CAPACITY
        if capacity is not None:
            resolved_capacity = capacity
        elif hasattr(self._config, "memory_capacity"):
            resolved_capacity = int(getattr(self._config, "memory_capacity"))

        self._store = MemoryStore(capacity=resolved_capacity)
        self._logger.debug(f"Initialized MemoryStore with capacity={resolved_capacity}")

        # 6. Self-registration in Service Container
        if auto_register_in_container:
            try:
                self._container.register_singleton("memory_manager", self, allow_override=True)
                self._container.register_singleton("memory", self, allow_override=True)
                self._logger.debug("Registered 'memory_manager' singleton into Service Container.")
            except Exception as exc:
                self._logger.warning(f"Could not register MemoryManager into container: {exc}")

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------

    @property
    def config(self) -> Settings:
        """Retrieve the active configuration instance."""
        return self._config

    @property
    def logger(self) -> logging.Logger:
        """Retrieve the active logger instance."""
        return self._logger

    @property
    def container(self) -> ServiceContainer:
        """Retrieve the active service container instance."""
        return self._container

    @property
    def event_bus(self) -> EventBus:
        """Retrieve the active event bus instance."""
        return self._event_bus

    @property
    def store(self) -> MemoryStore:
        """Retrieve the underlying MemoryStore instance."""
        return self._store

    @property
    def capacity(self) -> int:
        """Retrieve the current maximum capacity."""
        return self._store.capacity

    @capacity.setter
    def capacity(self, new_capacity: int) -> None:
        """Update maximum memory capacity dynamically."""
        with self._lock:
            self._store.capacity = new_capacity
            self._logger.info(f"Updated memory capacity to {new_capacity}")

    def set_capacity(self, new_capacity: int) -> None:
        """Convenience method to set memory capacity."""
        self.capacity = new_capacity

    # --------------------------------------------------------------------------
    # Synchronous Memory Operations
    # --------------------------------------------------------------------------

    def add(
        self,
        content: Union[str, ConversationMemory],
        role: str = "user",
        conversation_id: str = DEFAULT_CONVERSATION_ID,
        source: str = "text",
        importance: int = 1,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
    ) -> ConversationMemory:
        """Add a memory entry and publish 'memory.added' event.

        Args:
            content: Text string or existing ConversationMemory object.
            role: Entity role ('user', 'assistant', 'system', 'tool'). Defaults to 'user'.
            conversation_id: Multi-conversation tracking session ID (default: 'default').
            source: Input channel source ('voice', 'text', 'system', etc.).
            importance: Numerical score from 1 to 10.
            metadata: Optional dictionary of contextual metadata.
            tags: Optional list of keyword tags.

        Returns:
            The stored ConversationMemory instance.

        Raises:
            InvalidMemoryError: If content is empty or invalid.
        """
        if isinstance(content, ConversationMemory):
            mem = content
        else:
            mem = ConversationMemory(
                content=content,
                role=role,
                conversation_id=conversation_id,
                source=source,
                importance=importance,
                metadata=metadata or {},
                tags=tags or [],
            )

        with self._lock:
            stored = self._store.add(mem)
            current_count = self._store.count()

        # Publish memory.added event
        self._event_bus.publish(
            "memory.added",
            payload={
                "memory": stored.to_dict(),
                "total_count": current_count,
                "timestamp": time.time(),
            },
            source="memory_manager",
        )
        self._logger.debug(
            f"Added memory '{stored.id[:8]}...' (role={stored.role!r}, conv={stored.conversation_id!r})"
        )
        return stored

    def get_last(self) -> Optional[ConversationMemory]:
        """Retrieve the newest memory record in O(1) time.

        Returns:
            The latest ConversationMemory or None if empty.
        """
        return self._store.get_last()

    def get(self, memory_id: str) -> Optional[ConversationMemory]:
        """Retrieve a specific memory by ID in O(1) time."""
        return self._store.get(memory_id)

    def exists(self, memory_id: str) -> bool:
        """Check if a memory record exists in O(1) time."""
        return self._store.exists(memory_id)

    def remove(self, memory_id: str) -> bool:
        """Remove a memory record by ID and publish 'memory.removed' event.

        Args:
            memory_id: Unique UUID string identifier.

        Returns:
            True if removed, False otherwise.
        """
        with self._lock:
            removed = self._store.remove(memory_id)

        if removed:
            self._event_bus.publish(
                "memory.removed",
                payload={
                    "memory_id": memory_id,
                    "timestamp": time.time(),
                },
                source="memory_manager",
            )
            self._logger.debug(f"Removed memory '{memory_id}'")

        return removed

    def get_recent(
        self,
        limit: Optional[int] = None,
        role: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> list[ConversationMemory]:
        """Retrieve recent memories and publish 'memory.retrieved' event.

        Args:
            limit: Maximum number of recent memories to return. If None, returns all.
            role: Optional role filter ('user', 'assistant', 'system', 'tool').
            conversation_id: Optional conversation ID filter.

        Returns:
            Chronologically ordered list of recent ConversationMemory objects.
        """
        with self._lock:
            memories = self._store.get_recent(
                limit=limit,
                role=role,
                conversation_id=conversation_id,
            )

        # Publish memory.retrieved event
        self._event_bus.publish(
            "memory.retrieved",
            payload={
                "count": len(memories),
                "filter": {"limit": limit, "role": role, "conversation_id": conversation_id},
                "memories": [m.to_dict() for m in memories],
                "timestamp": time.time(),
            },
            source="memory_manager",
        )
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

        Publishes 'memory.retrieved' event.

        Args:
            query: Keyword or phrase to search across content and tags.
            limit: Maximum number of results to return.
            role: Optional role filter.
            conversation_id: Optional conversation ID filter.
            metadata: Optional dictionary filter matching metadata keys and values.

        Returns:
            Chronologically ordered list of matching memories.
        """
        with self._lock:
            matches = self._store.search(
                query=query,
                limit=limit,
                role=role,
                conversation_id=conversation_id,
                metadata=metadata,
            )

        # Publish memory.retrieved event
        self._event_bus.publish(
            "memory.retrieved",
            payload={
                "count": len(matches),
                "query": query,
                "filter": {
                    "limit": limit,
                    "role": role,
                    "conversation_id": conversation_id,
                    "metadata": metadata,
                },
                "memories": [m.to_dict() for m in matches],
                "timestamp": time.time(),
            },
            source="memory_manager",
        )
        return matches

    def clear(self) -> int:
        """Clear all memories from the store and publish 'memory.cleared' event.

        Returns:
            The number of memories cleared.
        """
        with self._lock:
            cleared_count = self._store.clear()

        # Publish memory.cleared event
        self._event_bus.publish(
            "memory.cleared",
            payload={
                "cleared_count": cleared_count,
                "timestamp": time.time(),
            },
            source="memory_manager",
        )
        self._logger.info(f"Cleared {cleared_count} memory records.")
        return cleared_count

    # --------------------------------------------------------------------------
    # Asynchronous Memory Operations
    # --------------------------------------------------------------------------

    async def add_async(
        self,
        content: Union[str, ConversationMemory],
        role: str = "user",
        conversation_id: str = DEFAULT_CONVERSATION_ID,
        source: str = "text",
        importance: int = 1,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
    ) -> ConversationMemory:
        """Asynchronously add a memory entry and publish 'memory.added' event.

        Args:
            content: Text string or existing ConversationMemory object.
            role: Entity role ('user', 'assistant', 'system', 'tool').
            conversation_id: Multi-conversation session ID.
            source: Input channel source.
            importance: Numerical score (1 to 10).
            metadata: Optional dictionary of contextual metadata.
            tags: Optional list of keyword tags.

        Returns:
            The stored ConversationMemory instance.
        """
        if isinstance(content, ConversationMemory):
            mem = content
        else:
            mem = ConversationMemory(
                content=content,
                role=role,
                conversation_id=conversation_id,
                source=source,
                importance=importance,
                metadata=metadata or {},
                tags=tags or [],
            )

        with self._lock:
            stored = self._store.add(mem)
            current_count = self._store.count()

        await self._event_bus.publish_async(
            "memory.added",
            payload={
                "memory": stored.to_dict(),
                "total_count": current_count,
                "timestamp": time.time(),
            },
            source="memory_manager",
        )
        self._logger.debug(f"Async added memory '{stored.id[:8]}...' (role={stored.role!r})")
        return stored

    async def remove_async(self, memory_id: str) -> bool:
        """Asynchronously remove a memory record by ID and publish 'memory.removed' event.

        Args:
            memory_id: Unique UUID string identifier.

        Returns:
            True if removed, False otherwise.
        """
        with self._lock:
            removed = self._store.remove(memory_id)

        if removed:
            await self._event_bus.publish_async(
                "memory.removed",
                payload={
                    "memory_id": memory_id,
                    "timestamp": time.time(),
                },
                source="memory_manager",
            )
            self._logger.debug(f"Async removed memory '{memory_id}'")

        return removed

    async def get_recent_async(
        self,
        limit: Optional[int] = None,
        role: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> list[ConversationMemory]:
        """Asynchronously retrieve recent memories and publish 'memory.retrieved' event.

        Args:
            limit: Maximum number of recent memories to return.
            role: Optional role filter.
            conversation_id: Optional conversation ID filter.

        Returns:
            Chronologically ordered list of recent ConversationMemory objects.
        """
        with self._lock:
            memories = self._store.get_recent(
                limit=limit,
                role=role,
                conversation_id=conversation_id,
            )

        await self._event_bus.publish_async(
            "memory.retrieved",
            payload={
                "count": len(memories),
                "filter": {"limit": limit, "role": role, "conversation_id": conversation_id},
                "memories": [m.to_dict() for m in memories],
                "timestamp": time.time(),
            },
            source="memory_manager",
        )
        return memories

    async def search_async(
        self,
        query: Optional[str] = None,
        limit: Optional[int] = None,
        role: Optional[str] = None,
        conversation_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[ConversationMemory]:
        """Asynchronously search memories and publish 'memory.retrieved' event.

        Args:
            query: Keyword or phrase to search across content and tags.
            limit: Maximum number of results to return.
            role: Optional role filter.
            conversation_id: Optional conversation ID filter.
            metadata: Optional dictionary filter for metadata matching.

        Returns:
            Chronologically ordered list of matching memories.
        """
        with self._lock:
            matches = self._store.search(
                query=query,
                limit=limit,
                role=role,
                conversation_id=conversation_id,
                metadata=metadata,
            )

        await self._event_bus.publish_async(
            "memory.retrieved",
            payload={
                "count": len(matches),
                "query": query,
                "filter": {
                    "limit": limit,
                    "role": role,
                    "conversation_id": conversation_id,
                    "metadata": metadata,
                },
                "memories": [m.to_dict() for m in matches],
                "timestamp": time.time(),
            },
            source="memory_manager",
        )
        return matches

    async def clear_async(self) -> int:
        """Asynchronously clear all memories and publish 'memory.cleared' event.

        Returns:
            The number of memories cleared.
        """
        with self._lock:
            cleared_count = self._store.clear()

        await self._event_bus.publish_async(
            "memory.cleared",
            payload={
                "cleared_count": cleared_count,
                "timestamp": time.time(),
            },
            source="memory_manager",
        )
        self._logger.info(f"Async cleared {cleared_count} memory records.")
        return cleared_count

    # --------------------------------------------------------------------------
    # Lookups & Helpers
    # --------------------------------------------------------------------------

    def count(self) -> int:
        """Return the total number of memories currently stored."""
        return self._store.count()

    def all(self) -> list[ConversationMemory]:
        """Return all memories in chronological order."""
        return self._store.all()

    def __len__(self) -> int:
        """Return total memory count."""
        return self.count()

    def __contains__(self, memory_id: str) -> bool:
        """Check if memory ID exists in store."""
        return self._store.exists(memory_id)

    def __repr__(self) -> str:
        """Developer-friendly string representation."""
        return f"<MemoryManager(count={self.count()}, capacity={self.capacity})>"


# Global default MemoryManager singleton
memory_manager: Final[MemoryManager] = MemoryManager()
