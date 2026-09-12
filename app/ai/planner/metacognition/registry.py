"""Dynamic Tool Registry for Phase 20 Metacognition.

Provides thread-safe runtime storage, resolution, duplicate detection,
and hot-binding of synthesized tools without global state.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("app.ai.planner.metacognition.registry")


class DynamicToolRegistry:
    """Thread-safe runtime registry for dynamically synthesized action callables."""

    def __init__(self) -> None:
        """Initialize an isolated dynamic tool registry."""
        self._lock = threading.RLock()
        self._tools: Dict[str, Callable[..., Any]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        action_name: str,
        callable_obj: Callable[..., Any],
        overwrite: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Register an action callable into the runtime registry.

        Args:
            action_name: Action identifier string.
            callable_obj: Python callable executing the action.
            overwrite: Whether to replace an existing tool with the same action name.
            metadata: Optional dictionary of tool attributes.

        Returns:
            True if registration succeeded.

        Raises:
            ValueError: If action_name already exists and overwrite is False, or inputs are invalid.
        """
        if not action_name or not action_name.strip():
            raise ValueError("Action name cannot be empty.")
        if not callable(callable_obj):
            raise ValueError(f"Provided object for '{action_name}' is not callable.")

        clean_name = action_name.strip().lower()

        with self._lock:
            if clean_name in self._tools and not overwrite:
                raise ValueError(f"Tool '{clean_name}' is already registered in DynamicToolRegistry.")

            self._tools[clean_name] = callable_obj
            self._metadata[clean_name] = dict(metadata or {})
            logger.debug("Dynamic tool '%s' registered (overwrite=%s).", clean_name, overwrite)
            return True

    def resolve(self, action_name: str) -> Optional[Callable[..., Any]]:
        """Resolve a registered callable by action name.

        Args:
            action_name: Action name to resolve.

        Returns:
            The callable object, or None if not found.
        """
        if not action_name:
            return None
        clean_name = action_name.strip().lower()
        with self._lock:
            return self._tools.get(clean_name)

    def contains(self, action_name: str) -> bool:
        """Check whether an action name is currently registered.

        Args:
            action_name: Action name to check.

        Returns:
            True if registered, False otherwise.
        """
        if not action_name:
            return False
        clean_name = action_name.strip().lower()
        with self._lock:
            return clean_name in self._tools

    def unregister(self, action_name: str) -> bool:
        """Remove an action from the registry.

        Args:
            action_name: Action name to unregister.

        Returns:
            True if tool was found and removed, False otherwise.
        """
        if not action_name:
            return False
        clean_name = action_name.strip().lower()
        with self._lock:
            if clean_name in self._tools:
                del self._tools[clean_name]
                self._metadata.pop(clean_name, None)
                logger.debug("Dynamic tool '%s' unregistered.", clean_name)
                return True
            return False

    def list_tools(self) -> List[str]:
        """Return a deterministically sorted list of all registered tool names.

        Returns:
            Alphabetically sorted list of action names.
        """
        with self._lock:
            return sorted(list(self._tools.keys()))

    def get_metadata(self, action_name: str) -> Optional[Dict[str, Any]]:
        """Return metadata for a registered tool.

        Args:
            action_name: Action name to query.

        Returns:
            Metadata dictionary copy, or None if not found.
        """
        if not action_name:
            return None
        clean_name = action_name.strip().lower()
        with self._lock:
            meta = self._metadata.get(clean_name)
            return dict(meta) if meta is not None else None

    def clear(self) -> None:
        """Remove all registered tools from this registry instance."""
        with self._lock:
            self._tools.clear()
            self._metadata.clear()
