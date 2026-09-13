"""Memory Subsystem for J.A.R.V.I.S.

Provides short-term conversational context storage, search, dynamic capacity
enforcement, and lifecycle event streaming across the J.A.R.V.I.S architecture.
"""

from __future__ import annotations

from app.memory.models import (
    DEFAULT_MEMORY_CAPACITY,
    VALID_ROLES,
    ConversationMemory,
    InvalidMemoryError,
    MemoryCapacityError,
    MemoryError,
)
from app.memory.store import MemoryStore
from app.memory.manager import MemoryManager, memory_manager
from app.memory.persistence import ConversationHistoryPersistence

__all__ = [
    "ConversationMemory",
    "ConversationHistoryPersistence",
    "MemoryStore",
    "MemoryManager",
    "memory_manager",
    "DEFAULT_MEMORY_CAPACITY",
    "VALID_ROLES",
    "MemoryError",
    "InvalidMemoryError",
    "MemoryCapacityError",
]
