"""Shared Execution Memory & Blackboard Workspace for Phase 18.0.

Provides thread-safe multi-agent blackboard storage, scoped key-value namespaces,
an artifact store for passing large data by reference, and versioned concurrency locks.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("app.ai.planner.multi_agent.memory")


class ArtifactStore:
    """Thread-safe store for intermediate execution artifacts (reports, code, tables)."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._artifacts: Dict[str, Dict[str, Any]] = {}

    def put_artifact(
        self,
        artifact_id: str,
        content: Any,
        created_by: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store an artifact and return its identifier."""
        with self._lock:
            self._artifacts[artifact_id] = {
                "artifact_id": artifact_id,
                "content": content,
                "created_by": created_by,
                "created_at": time.time(),
                "metadata": dict(metadata or {}),
            }
            return artifact_id

    def get_artifact(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an artifact dictionary by ID."""
        with self._lock:
            item = self._artifacts.get(artifact_id)
            return dict(item) if item is not None else None

    def list_artifacts(self) -> List[Dict[str, Any]]:
        """List summary of all stored artifacts."""
        with self._lock:
            return [
                {
                    "artifact_id": a["artifact_id"],
                    "created_by": a["created_by"],
                    "created_at": a["created_at"],
                    "metadata": a["metadata"],
                }
                for a in self._artifacts.values()
            ]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize artifacts."""
        with self._lock:
            return {k: dict(v) for k, v in self._artifacts.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ArtifactStore:
        """Restore artifacts from dictionary."""
        store = cls()
        store._artifacts = {k: dict(v) for k, v in data.items()}
        return store


class SharedAgentMemory:
    """Thread-safe blackboard workspace for multi-agent coordination.

    Features:
    - Global shared workspace for cross-agent discovery and synthesis
    - Namespaced agent workspaces for private notes and intermediate computation
    - Task-scoped workspaces for upstream output sharing
    - Versioned keys with conflict detection
    - High-volume artifact storage
    """

    def __init__(self, artifact_store: Optional[ArtifactStore] = None) -> None:
        self._lock = threading.RLock()
        self._global_store: Dict[str, Any] = {}
        self._key_versions: Dict[str, int] = {}
        self._agent_stores: Dict[str, Dict[str, Any]] = {}
        self._task_stores: Dict[str, Dict[str, Any]] = {}
        self.artifacts = artifact_store or ArtifactStore()

    # --------------------------------------------------------------------------
    # Global Workspace
    # --------------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from the global shared blackboard."""
        with self._lock:
            return self._global_store.get(key, default)

    def set(self, key: str, value: Any) -> int:
        """Set a value in the global blackboard and return the new version number."""
        with self._lock:
            self._global_store[key] = value
            v = self._key_versions.get(key, 0) + 1
            self._key_versions[key] = v
            return v

    def set_with_version(self, key: str, value: Any, expected_version: int) -> Tuple[bool, int]:
        """Optimistically set a key only if the current version matches expected.

        Returns:
            Tuple of (success, current_or_new_version).
        """
        with self._lock:
            curr = self._key_versions.get(key, 0)
            if curr != expected_version:
                logger.warning("Version mismatch for key '%s': expected %d, got %d", key, expected_version, curr)
                return False, curr

            self._global_store[key] = value
            new_v = curr + 1
            self._key_versions[key] = new_v
            return True, new_v

    def get_version(self, key: str) -> int:
        """Get current version of a global key (0 if not set)."""
        with self._lock:
            return self._key_versions.get(key, 0)

    def has(self, key: str) -> bool:
        """Check if global key exists."""
        with self._lock:
            return key in self._global_store

    def delete(self, key: str) -> bool:
        """Delete key from global blackboard."""
        with self._lock:
            if key in self._global_store:
                del self._global_store[key]
                self._key_versions[key] = self._key_versions.get(key, 0) + 1
                return True
            return False

    def list_keys(self) -> List[str]:
        """List all keys in global blackboard."""
        with self._lock:
            return sorted(self._global_store.keys())

    # --------------------------------------------------------------------------
    # Agent Namespaced Workspace
    # --------------------------------------------------------------------------

    def get_agent_data(self, agent_id: str, key: str, default: Any = None) -> Any:
        """Get private data belonging to a specific agent."""
        with self._lock:
            store = self._agent_stores.get(agent_id, {})
            return store.get(key, default)

    def set_agent_data(self, agent_id: str, key: str, value: Any) -> None:
        """Set private data belonging to a specific agent."""
        with self._lock:
            self._agent_stores.setdefault(agent_id, {})[key] = value

    def list_agent_keys(self, agent_id: str) -> List[str]:
        """List all keys owned by an agent."""
        with self._lock:
            return sorted(self._agent_stores.get(agent_id, {}).keys())

    # --------------------------------------------------------------------------
    # Task Namespaced Workspace
    # --------------------------------------------------------------------------

    def get_task_data(self, task_id: str, key: str, default: Any = None) -> Any:
        """Get data associated with a specific task."""
        with self._lock:
            store = self._task_stores.get(task_id, {})
            return store.get(key, default)

    def set_task_data(self, task_id: str, key: str, value: Any) -> None:
        """Set data associated with a specific task."""
        with self._lock:
            self._task_stores.setdefault(task_id, {})[key] = value

    # --------------------------------------------------------------------------
    # Serialization
    # --------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize memory and blackboard contents."""
        with self._lock:
            return {
                "global_store": dict(self._global_store),
                "key_versions": dict(self._key_versions),
                "agent_stores": {aid: dict(data) for aid, data in self._agent_stores.items()},
                "task_stores": {tid: dict(data) for tid, data in self._task_stores.items()},
                "artifacts": self.artifacts.to_dict(),
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SharedAgentMemory:
        """Restore shared memory from dictionary."""
        art_store = ArtifactStore.from_dict(data.get("artifacts", {}))
        mem = cls(artifact_store=art_store)
        mem._global_store = dict(data.get("global_store", {}))
        mem._key_versions = dict(data.get("key_versions", {}))
        mem._agent_stores = {aid: dict(sub) for aid, sub in data.get("agent_stores", {}).items()}
        mem._task_stores = {tid: dict(sub) for tid, sub in data.get("task_stores", {}).items()}
        return mem
