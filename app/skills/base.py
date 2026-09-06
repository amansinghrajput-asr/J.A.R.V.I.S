"""Base Skill definition and exception hierarchy for J.A.R.V.I.S.

Provides the foundational contract, lifecycle hooks, execution interfaces,
metadata introspection, and dependency injection integration for all J.A.R.V.I.S skills.
"""

from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from typing import Any, Final, Optional, Union

from app.core.config import Settings, settings
from app.core.container import JarvisException, ServiceContainer, container
from app.core.event_bus import EventBus, event_bus
from app.core.logger import get_logger

DEFAULT_SKILL_PRIORITY: Final[int] = 100
DEFAULT_SKILL_VERSION: Final[str] = "0.1.0"


class SkillError(JarvisException):
    """Base exception for all skill framework errors."""


class SkillNotFoundError(SkillError):
    """Raised when a requested skill or matching skill cannot be found."""


class SkillAlreadyRegisteredError(SkillError):
    """Raised when attempting to register a skill with an existing name."""


class SkillExecutionError(SkillError):
    """Raised when an error occurs during skill execution."""


class SkillInitializationError(SkillError):
    """Raised when an error occurs during skill initialization."""


class InvalidSkillError(SkillError):
    """Raised when a skill configuration or instance is invalid."""


class SkillDiscoveryError(SkillError):
    """Raised when an error occurs during dynamic skill discovery."""


class BaseSkill(ABC):
    """Abstract base class for all J.A.R.V.I.S skills.

    Defines the contract for skill identity, execution lifecycle hooks,
    command handling, permission architecture, priority ordering, and
    dependency injection access.

    Subclasses must define `name` and implement `can_handle` and `execute`.
    They may optionally implement `initialize` and `shutdown` for lifecycle hooks.

    Attributes:
        name: Unique identifier for the skill (e.g., 'system_control', 'web_search').
        description: Human-readable explanation of what the skill does.
        version: Semantic version string of the skill.
        enabled: Boolean flag indicating if the skill is currently active.
        priority: Numeric execution priority (higher value runs first).
        tags: List of descriptive keyword tags for skill discovery.
        permissions: Set of system permission identifiers required by the skill.
    """

    name: str = ""
    description: str = ""
    version: str = DEFAULT_SKILL_VERSION
    enabled: bool = True
    priority: int = DEFAULT_SKILL_PRIORITY
    tags: list[str] = []
    permissions: set[str] = set()

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        version: Optional[str] = None,
        enabled: Optional[bool] = None,
        priority: Optional[int] = None,
        tags: Optional[list[str]] = None,
        permissions: Optional[Union[set[str], list[str]]] = None,
        *,
        config: Optional[Settings] = None,
        logger: Optional[logging.Logger] = None,
        container: Optional[ServiceContainer] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        """Initialize the BaseSkill instance.

        Args:
            name: Optional override for skill name.
            description: Optional override for skill description.
            version: Optional override for semantic version.
            enabled: Optional override for skill enabled state.
            priority: Optional execution priority (default: 100).
            tags: Optional list of discovery tags.
            permissions: Optional set or list of required permissions.
            config: Injected system configuration settings.
            logger: Injected logger instance.
            container: Injected service container.
            event_bus: Injected event bus instance.

        Raises:
            InvalidSkillError: If the skill name is not defined or invalid.
        """
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description
        if version is not None:
            self.version = version
        if enabled is not None:
            self.enabled = bool(enabled)
        if priority is not None:
            self.priority = int(priority)
        if tags is not None:
            self.tags = list(tags)
        else:
            self.tags = list(getattr(self.__class__, "tags", []))
        if permissions is not None:
            self.permissions = set(permissions)
        else:
            self.permissions = set(getattr(self.__class__, "permissions", set()))

        # Validate skill name
        if not isinstance(self.name, str) or not self.name.strip():
            raise InvalidSkillError(
                f"Skill '{self.__class__.__name__}' must specify a non-empty string name."
            )
        self.name = self.name.strip()

        # Dependency injection holders
        self._config: Optional[Settings] = config
        self._logger: Optional[logging.Logger] = logger
        self._container: Optional[ServiceContainer] = container
        self._event_bus: Optional[EventBus] = event_bus

    @property
    def config(self) -> Settings:
        """Retrieve the active configuration instance.

        Returns:
            The injected Settings or resolved system settings.
        """
        if self._config is not None:
            return self._config
        if self._container is not None and self._container.exists("settings"):
            return self._container.resolve("settings")
        if self._container is not None and self._container.exists("config"):
            return self._container.resolve("config")
        return settings

    @property
    def logger(self) -> logging.Logger:
        """Retrieve the logger configured for this skill.

        Returns:
            The injected logger or a module-specific logger.
        """
        if self._logger is not None:
            return self._logger
        return get_logger(f"SKILL.{self.name.upper()}")

    @property
    def container(self) -> ServiceContainer:
        """Retrieve the service container instance.

        Returns:
            The injected ServiceContainer or global container singleton.
        """
        if self._container is not None:
            return self._container
        return container

    @property
    def event_bus(self) -> EventBus:
        """Retrieve the event bus instance.

        Returns:
            The injected EventBus or resolved/global event bus singleton.
        """
        if self._event_bus is not None:
            return self._event_bus
        if self._container is not None and self._container.exists("event_bus"):
            return self._container.resolve("event_bus")
        return event_bus

    def bind(
        self,
        config: Optional[Settings] = None,
        logger: Optional[logging.Logger] = None,
        container: Optional[ServiceContainer] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        """Bind system dependencies into the skill.

        Called automatically during registration by SkillManager, or manually
        for testing and standalone invocation.

        Args:
            config: Optional Settings instance to bind.
            logger: Optional Logger instance to bind.
            container: Optional ServiceContainer to bind.
            event_bus: Optional EventBus to bind.
        """
        if config is not None:
            self._config = config
        if logger is not None:
            self._logger = logger
        if container is not None:
            self._container = container
        if event_bus is not None:
            self._event_bus = event_bus

    def initialize(self) -> None:
        """Hook called when the skill is registered with the SkillManager.

        Override this method to perform one-time setup, load models, allocate
        resources, or initialize external client sessions.
        """

    def shutdown(self) -> None:
        """Hook called when the skill is unregistered or the system shuts down.

        Override this method to perform graceful cleanup, close file handles,
        terminate background workers, or disconnect network sessions.
        """

    @abstractmethod
    def can_handle(self, command: Any) -> bool:
        """Evaluate whether this skill can process the given command.

        Args:
            command: The command payload, text string, or intent object to evaluate.

        Returns:
            True if this skill is capable of handling the command, False otherwise.
        """

    @abstractmethod
    def execute(self, command: Any) -> Any:
        """Execute the skill logic against the specified command.

        Args:
            command: The command payload, text string, or intent object to process.

        Returns:
            The result of the skill execution (arbitrary output, payload, or status).

        Raises:
            Exception: Any domain exception raised during skill execution.
        """

    async def execute_async(self, command: Any) -> Any:
        """Asynchronously execute the skill logic against the specified command.

        Default implementation delegates to execute() and awaits if the result is a coroutine.
        Subclasses may override this method directly with native async logic.

        Args:
            command: The command payload, text string, or intent object to process.

        Returns:
            The result of the skill execution.

        Raises:
            Exception: Any domain exception raised during skill execution.
        """
        result = self.execute(command)
        if inspect.iscoroutine(result):
            return await result
        return result

    def enable(self) -> None:
        """Enable this skill."""
        self.enabled = True

    def disable(self) -> None:
        """Disable this skill."""
        self.enabled = False

    def metadata(self) -> dict[str, Any]:
        """Export standardized metadata for skill introspection and discovery.

        Returns:
            Dictionary containing the skill's name, description, version,
            enabled status, priority, tags, and permissions.
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "enabled": self.enabled,
            "priority": self.priority,
            "tags": list(self.tags),
            "permissions": sorted(self.permissions),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convenience alias for `metadata()`."""
        return self.metadata()

    def __repr__(self) -> str:
        """Developer-friendly string representation."""
        return (
            f"<{self.__class__.__name__}(name={self.name!r}, "
            f"version={self.version!r}, priority={self.priority}, "
            f"enabled={self.enabled})>"
        )
