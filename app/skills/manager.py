"""Skill Manager Subsystem for J.A.R.V.I.S.

Coordinates the registration, lifecycle management, dependency injection,
command routing, prioritization, and event notification for all system skills.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import inspect
import logging
import os
from pathlib import Path
import pkgutil
import sys
import threading
import time
from typing import Any, Final, Optional, Union

from app.core.config import Settings, settings
from app.core.container import ServiceContainer, container
from app.core.event_bus import EventBus, event_bus
from app.core.logger import get_logger
from app.skills.base import (
    BaseSkill,
    InvalidSkillError,
    SkillAlreadyRegisteredError,
    SkillDiscoveryError,
    SkillExecutionError,
    SkillInitializationError,
    SkillNotFoundError,
)


class SkillManager:
    """Thread-safe registry and execution coordinator for J.A.R.V.I.S skills.

    Manages skill lifecycles, orchestrates dependency injection across configuration,
    logging, container, and event bus subsystems, routes commands based on skill
    capabilities and priority, and emits lifecycle events.

    Attributes:
        config: System configuration instance.
        logger: Centralized logger instance.
        container: Service container for dependency resolution.
        event_bus: Publish-subscribe event bus instance.
    """

    def __init__(
        self,
        config: Optional[Settings] = None,
        logger: Optional[logging.Logger] = None,
        container_instance: Optional[ServiceContainer] = None,
        event_bus_instance: Optional[EventBus] = None,
        *,
        auto_register_in_container: bool = True,
    ) -> None:
        """Initialize the SkillManager.

        Args:
            config: Optional Settings instance. If None, resolves from container or global settings.
            logger: Optional Logger instance. If None, creates 'SKILL_MANAGER' logger.
            container_instance: Optional ServiceContainer. If None, uses global container.
            event_bus_instance: Optional EventBus. If None, resolves from container or global event bus.
            auto_register_in_container: If True, registers this SkillManager singleton into container.
        """
        self._lock = threading.RLock()
        self._skills: dict[str, BaseSkill] = {}

        # 1. Dependency resolution: Container
        self._container = container_instance if container_instance is not None else container

        # 2. Dependency resolution: Logger
        self._logger = logger if logger is not None else get_logger("SKILL_MANAGER")

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

        # 5. Optional self-registration in Service Container
        if auto_register_in_container:
            try:
                self._container.register_singleton(
                    "skill_manager", self, allow_override=True
                )
                self._logger.debug("Registered 'skill_manager' singleton into Service Container.")
            except Exception as exc:
                self._logger.warning(
                    f"Could not register SkillManager into container: {exc}"
                )

    @property
    def config(self) -> Settings:
        """Return the active configuration settings."""
        return self._config

    @property
    def logger(self) -> logging.Logger:
        """Return the active logger instance."""
        return self._logger

    @property
    def container(self) -> ServiceContainer:
        """Return the active service container."""
        return self._container

    @property
    def event_bus(self) -> EventBus:
        """Return the active event bus."""
        return self._event_bus

    def register(self, skill: BaseSkill, *, allow_override: bool = False) -> None:
        """Register a new skill into the manager.

        Performs validation, injects subsystem dependencies, executes skill initialization,
        records the skill in the internal registry, and publishes a 'skill.registered' event.

        Args:
            skill: An instance of BaseSkill to register.
            allow_override: If True, permits replacing an already registered skill with the same name.
                Defaults to False.

        Raises:
            InvalidSkillError: If skill is not an instance of BaseSkill or has an invalid name.
            SkillAlreadyRegisteredError: If a skill with the same name is registered and
                allow_override is False.
            SkillInitializationError: If the skill's `initialize()` hook raises an exception.
        """
        if not isinstance(skill, BaseSkill):
            raise InvalidSkillError(
                f"Expected instance of BaseSkill, got {type(skill).__name__}."
            )

        skill_name = skill.name
        if not isinstance(skill_name, str) or not skill_name.strip():
            raise InvalidSkillError("Skill name must be a non-empty string.")
        clean_name = skill_name.strip()

        with self._lock:
            if clean_name in self._skills and not allow_override:
                raise SkillAlreadyRegisteredError(
                    f"Skill '{clean_name}' is already registered. Set allow_override=True to replace."
                )

            # Dependency Injection into Skill
            skill.bind(
                config=self._config,
                logger=self._logger,
                container=self._container,
                event_bus=self._event_bus,
            )

            # Initialize skill lifecycle hook
            try:
                skill.initialize()
            except Exception as exc:
                self._logger.exception(
                    f"Failed to initialize skill '{clean_name}': {exc}"
                )
                self._event_bus.publish(
                    "skill.failed",
                    payload={
                        "skill_name": clean_name,
                        "stage": "initialize",
                        "error": str(exc),
                        "exception_type": type(exc).__name__,
                    },
                    source="skill_manager",
                )
                raise SkillInitializationError(
                    f"Initialization of skill '{clean_name}' failed: {exc}"
                ) from exc

            self._skills[clean_name] = skill

            self._logger.info(
                f"Registered skill: '{clean_name}' (v{skill.version}, "
                f"priority={skill.priority}, enabled={skill.enabled})"
            )

        # Publish skill.registered event
        self._event_bus.publish(
            "skill.registered",
            payload=skill.metadata(),
            source="skill_manager",
        )

    def unregister(self, skill_name: str) -> bool:
        """Unregister a skill by name and invoke its shutdown lifecycle hook.

        Args:
            skill_name: The unique identifier of the skill to unregister.

        Returns:
            True if the skill was found and removed, False otherwise.
        """
        if not isinstance(skill_name, str) or not skill_name.strip():
            return False
        clean_name = skill_name.strip()

        with self._lock:
            skill = self._skills.pop(clean_name, None)
            if skill is None:
                return False

        # Execute shutdown hook outside the lock to prevent deadlocks
        try:
            skill.shutdown()
        except Exception as exc:
            self._logger.exception(
                f"Error during shutdown of skill '{clean_name}': {exc}"
            )

        self._logger.info(f"Unregistered skill: '{clean_name}'")
        self._event_bus.publish(
            "skill.unregistered",
            payload={"skill_name": clean_name},
            source="skill_manager",
        )
        return True

    def get(self, skill_name: str) -> Optional[BaseSkill]:
        """Retrieve a registered skill by name.

        Args:
            skill_name: Unique identifier of the desired skill.

        Returns:
            The BaseSkill instance if found, None otherwise.
        """
        if not isinstance(skill_name, str) or not skill_name.strip():
            return None
        clean_name = skill_name.strip()

        with self._lock:
            return self._skills.get(clean_name)

    def list_skills(
        self,
        *,
        enabled_only: bool = False,
        tag: Optional[str] = None,
    ) -> list[BaseSkill]:
        """Return a sorted list of registered skills.

        Skills are ordered by priority (highest first), then alphabetically by name.

        Args:
            enabled_only: If True, filters out disabled skills. Defaults to False.
            tag: Optional tag filter; only skills containing this tag are returned.

        Returns:
            List of matching BaseSkill instances.
        """
        with self._lock:
            skills = list(self._skills.values())

        if enabled_only:
            skills = [s for s in skills if s.enabled]

        if tag is not None and isinstance(tag, str) and tag.strip():
            clean_tag = tag.strip().lower()
            skills = [s for s in skills if any(t.lower() == clean_tag for t in s.tags)]

        # Sort: highest priority first (-priority), then alphabetical by name
        skills.sort(key=lambda s: (-s.priority, s.name))
        return skills

    def execute(self, command: Any, *, skill_name: Optional[str] = None) -> Any:
        """Execute a command synchronously by routing to an appropriate skill.

        If `skill_name` is provided, routes directly to that skill. Otherwise,
        finds the highest-priority enabled skill whose `can_handle(command)` returns True.

        Publishes 'skill.executed' on success or 'skill.failed' on error.

        Args:
            command: The command payload, query string, or intent object to process.
            skill_name: Optional specific skill name to target directly.

        Returns:
            The result of skill execution.

        Raises:
            SkillNotFoundError: If no registered/enabled skill can handle the command,
                or if targeted skill_name is not registered.
            SkillExecutionError: If the skill is disabled or execution raises an exception.
        """
        target_skill = self._resolve_target_skill(command, skill_name=skill_name)
        start_time = time.perf_counter()

        try:
            result = target_skill.execute(command)

            # Handle returned coroutine if a coroutine skill was invoked in sync context
            if inspect.iscoroutine(result):
                try:
                    loop = asyncio.get_running_loop()
                    result = loop.create_task(result)
                except RuntimeError:
                    result = asyncio.run(result)

            duration = time.perf_counter() - start_time

            # Publish skill.executed event
            self._event_bus.publish(
                "skill.executed",
                payload={
                    "skill_name": target_skill.name,
                    "command": command,
                    "result": result,
                    "duration": duration,
                },
                source="skill_manager",
            )
            return result

        except Exception as exc:
            duration = time.perf_counter() - start_time

            # Publish skill.failed event
            self._event_bus.publish(
                "skill.failed",
                payload={
                    "skill_name": target_skill.name,
                    "command": command,
                    "error": str(exc),
                    "exception_type": type(exc).__name__,
                    "duration": duration,
                },
                source="skill_manager",
            )
            self._logger.exception(
                f"Execution failed for skill '{target_skill.name}': {exc}"
            )
            raise SkillExecutionError(
                f"Skill '{target_skill.name}' execution failed: {exc}"
            ) from exc

    async def execute_async(self, command: Any, *, skill_name: Optional[str] = None) -> Any:
        """Execute a command asynchronously by routing to an appropriate skill.

        Supports both synchronous skills and coroutines (`async def execute(...)`).

        Publishes 'skill.executed' on success or 'skill.failed' on error.

        Args:
            command: The command payload, query string, or intent object to process.
            skill_name: Optional specific skill name to target directly.

        Returns:
            The result of skill execution.

        Raises:
            SkillNotFoundError: If no registered/enabled skill can handle the command.
            SkillExecutionError: If skill execution raises an exception.
        """
        target_skill = self._resolve_target_skill(command, skill_name=skill_name)
        start_time = time.perf_counter()

        try:
            result = await target_skill.execute_async(command)

            duration = time.perf_counter() - start_time

            await self._event_bus.publish_async(
                "skill.executed",
                payload={
                    "skill_name": target_skill.name,
                    "command": command,
                    "result": result,
                    "duration": duration,
                },
                source="skill_manager",
            )
            return result

        except Exception as exc:
            duration = time.perf_counter() - start_time

            await self._event_bus.publish_async(
                "skill.failed",
                payload={
                    "skill_name": target_skill.name,
                    "command": command,
                    "error": str(exc),
                    "exception_type": type(exc).__name__,
                    "duration": duration,
                },
                source="skill_manager",
            )
            self._logger.exception(
                f"Async execution failed for skill '{target_skill.name}': {exc}"
            )
            raise SkillExecutionError(
                f"Skill '{target_skill.name}' async execution failed: {exc}"
            ) from exc

    def _resolve_target_skill(
        self,
        command: Any,
        skill_name: Optional[str] = None,
    ) -> BaseSkill:
        """Resolve the target skill for a command according to target name or priority.

        Args:
            command: Command being routed.
            skill_name: Optional specific skill name.

        Returns:
            The resolved BaseSkill.

        Raises:
            SkillNotFoundError: If no skill matches or target skill does not exist.
            SkillExecutionError: If target skill exists but is disabled.
        """
        if skill_name is not None:
            clean_name = skill_name.strip() if isinstance(skill_name, str) else ""
            with self._lock:
                skill = self._skills.get(clean_name)

            if skill is None:
                raise SkillNotFoundError(f"Skill '{skill_name}' is not registered.")
            if not skill.enabled:
                raise SkillExecutionError(
                    f"Cannot execute disabled skill '{skill.name}'."
                )
            return skill

        # Route dynamically via can_handle across candidates ordered by priority
        candidates = self.list_skills(enabled_only=True)
        for cand in candidates:
            try:
                if cand.can_handle(command):
                    return cand
            except Exception as exc:
                self._logger.warning(
                    f"Error in 'can_handle' for skill '{cand.name}': {exc}"
                )

        raise SkillNotFoundError(
            f"No enabled skill found capable of handling command: {command!r}"
        )

    def enable_skill(self, skill_name: str) -> bool:
        """Enable a registered skill.

        Args:
            skill_name: Unique identifier of the skill to enable.

        Returns:
            True if the skill was found and enabled, False otherwise.
        """
        if not isinstance(skill_name, str) or not skill_name.strip():
            return False
        clean_name = skill_name.strip()

        with self._lock:
            skill = self._skills.get(clean_name)
            if skill is None:
                return False
            skill.enabled = True

        self._logger.info(f"Enabled skill: '{skill.name}'")
        self._event_bus.publish(
            "skill.enabled",
            payload={"skill_name": skill.name},
            source="skill_manager",
        )
        return True

    def disable_skill(self, skill_name: str) -> bool:
        """Disable a registered skill.

        Args:
            skill_name: Unique identifier of the skill to disable.

        Returns:
            True if the skill was found and disabled, False otherwise.
        """
        if not isinstance(skill_name, str) or not skill_name.strip():
            return False
        clean_name = skill_name.strip()

        with self._lock:
            skill = self._skills.get(clean_name)
            if skill is None:
                return False
            skill.enabled = False

        self._logger.info(f"Disabled skill: '{skill.name}'")
        self._event_bus.publish(
            "skill.disabled",
            payload={"skill_name": skill.name},
            source="skill_manager",
        )
        return True

    def get_skills_by_tag(
        self,
        tag: str,
        *,
        enabled_only: bool = False,
    ) -> list[BaseSkill]:
        """Return registered skills tagged with the specified tag keyword.

        Args:
            tag: The tag string to match against skill.tags (case-insensitive).
            enabled_only: If True, only returns active/enabled skills.

        Returns:
            List of matching BaseSkill instances sorted by priority (highest first).
        """
        return self.list_skills(enabled_only=enabled_only, tag=tag)

    def get_skills_by_permission(
        self,
        permission: str,
        *,
        enabled_only: bool = False,
    ) -> list[BaseSkill]:
        """Return registered skills requesting the specified permission.

        Args:
            permission: The permission string identifier.
            enabled_only: If True, only returns active/enabled skills.

        Returns:
            List of matching BaseSkill instances sorted by priority (highest first).
        """
        if not isinstance(permission, str) or not permission.strip():
            return []
        clean_perm = permission.strip().lower()

        with self._lock:
            skills = list(self._skills.values())

        if enabled_only:
            skills = [s for s in skills if s.enabled]

        matches = [
            s for s in skills
            if any(p.lower() == clean_perm for p in s.permissions)
        ]
        matches.sort(key=lambda s: (-s.priority, s.name))
        return matches

    def find_skills(
        self,
        query: Optional[str] = None,
        *,
        tag: Optional[str] = None,
        permission: Optional[str] = None,
        enabled_only: bool = False,
    ) -> list[BaseSkill]:
        """Discover and filter registered skills matching composite search criteria.

        Args:
            query: Substring to match in skill name or description (case-insensitive).
            tag: Tag to filter by (case-insensitive).
            permission: Required permission identifier.
            enabled_only: If True, only returns active/enabled skills.

        Returns:
            List of matching BaseSkill instances sorted by priority (highest first).
        """
        skills = self.list_skills(enabled_only=enabled_only, tag=tag)

        if permission is not None and isinstance(permission, str) and permission.strip():
            clean_perm = permission.strip().lower()
            skills = [
                s for s in skills
                if any(p.lower() == clean_perm for p in s.permissions)
            ]

        if query is not None and isinstance(query, str) and query.strip():
            clean_q = query.strip().lower()
            skills = [
                s for s in skills
                if clean_q in s.name.lower() or clean_q in s.description.lower()
            ]

        return skills

    def discover(
        self,
        package_or_path: Union[str, Path, None] = None,
        *,
        recursive: bool = True,
        allow_override: bool = False,
    ) -> list[str]:
        """Dynamically discover and register skills from a package, module, or filesystem directory.

        Inspects modules for concrete subclasses of `BaseSkill`, instantiates them,
        and registers them into this SkillManager.

        Args:
            package_or_path: A dotted Python package name (e.g. 'app.skills'),
                a filesystem directory path (e.g. 'plugins/' or Path object),
                or None to scan default locations.
            recursive: If True, recursively scans sub-packages or subdirectories.
            allow_override: If True, allows discovered skills to overwrite already registered ones.

        Returns:
            List of registered skill names discovered in this run.

        Raises:
            SkillDiscoveryError: If package or directory cannot be found or imported.
        """
        if package_or_path is None:
            return []

        path_obj = Path(package_or_path) if not isinstance(package_or_path, Path) else package_or_path
        discovered_names: list[str] = []
        modules: list[Any] = []
        seen_classes: set[type] = set()

        if path_obj.exists() and path_obj.is_dir():
            pattern = "**/*.py" if recursive else "*.py"
            py_files = sorted(path_obj.glob(pattern))

            for py_file in py_files:
                if py_file.name.startswith((".", "_")) or py_file.name in ("__init__.py", "base.py", "manager.py"):
                    continue
                module_name = f"_jarvis_discovered_{py_file.stem}_{abs(hash(str(py_file.resolve())))}"
                try:
                    spec = importlib.util.spec_from_file_location(module_name, str(py_file.resolve()))
                    if spec is not None and spec.loader is not None:
                        mod = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = mod
                        spec.loader.exec_module(mod)
                        modules.append(mod)
                except Exception as exc:
                    self._logger.warning(
                        f"Failed to load module from file '{py_file}': {exc}"
                    )
        elif path_obj.exists() and path_obj.is_file() and path_obj.suffix == ".py":
            module_name = f"_jarvis_discovered_{path_obj.stem}_{abs(hash(str(path_obj.resolve())))}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, str(path_obj.resolve()))
                if spec is not None and spec.loader is not None:
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = mod
                    spec.loader.exec_module(mod)
                    modules.append(mod)
            except Exception as exc:
                raise SkillDiscoveryError(
                    f"Failed to load skill file '{package_or_path}': {exc}"
                ) from exc
        else:
            pkg_str = str(package_or_path)
            try:
                pkg = importlib.import_module(pkg_str)
                modules.append(pkg)
                if hasattr(pkg, "__path__"):
                    prefix = pkg.__name__ + "."
                    walk_iter = (
                        pkgutil.walk_packages(pkg.__path__, prefix)
                        if recursive
                        else pkgutil.iter_modules(pkg.__path__, prefix)
                    )
                    for info in walk_iter:
                        try:
                            submod = importlib.import_module(info.name)
                            modules.append(submod)
                        except Exception as sub_exc:
                            self._logger.warning(
                                f"Failed to import discovered sub-module '{info.name}': {sub_exc}"
                            )
            except ImportError as exc:
                raise SkillDiscoveryError(
                    f"Failed to discover skills from package or path '{package_or_path}': {exc}"
                ) from exc

        for mod in modules:
            for attr_name, obj in inspect.getmembers(mod, inspect.isclass):
                if (
                    issubclass(obj, BaseSkill)
                    and obj is not BaseSkill
                    and not inspect.isabstract(obj)
                    and obj not in seen_classes
                ):
                    seen_classes.add(obj)
                    try:
                        skill_inst = obj()
                        self.register(skill_inst, allow_override=allow_override)
                        discovered_names.append(skill_inst.name)
                    except Exception as exc:
                        self._logger.warning(
                            f"Failed to instantiate or register discovered skill class '{attr_name}' from '{mod.__name__}': {exc}"
                        )

        if discovered_names:
            self._event_bus.publish(
                "skills.discovered",
                payload={"skills": list(discovered_names), "count": len(discovered_names)},
                source="skill_manager",
            )
            self._logger.info(
                f"Discovered and registered {len(discovered_names)} skill(s): {discovered_names}"
            )

        return discovered_names

    discover_skills = discover

    def has_skill(self, skill_name: str) -> bool:
        """Check if a skill is registered.

        Args:
            skill_name: Unique identifier of the skill.

        Returns:
            True if registered, False otherwise.
        """
        if not isinstance(skill_name, str) or not skill_name.strip():
            return False
        clean_name = skill_name.strip()
        with self._lock:
            return clean_name in self._skills

    def clear(self) -> None:
        """Shut down and unregister all skills.

        Useful for test teardown and system shutdown.
        """
        with self._lock:
            skills = list(self._skills.values())
            self._skills.clear()

        for skill in skills:
            try:
                skill.shutdown()
            except Exception as exc:
                self._logger.exception(
                    f"Error during shutdown of skill '{skill.name}': {exc}"
                )

        self._logger.debug("Cleared all registered skills.")

    def __contains__(self, skill_name: str) -> bool:
        """Support 'in' operator syntax: `'weather' in skill_manager`."""
        return self.has_skill(skill_name)

    def __getitem__(self, skill_name: str) -> BaseSkill:
        """Support dict-like indexing syntax: `skill_manager['weather']`."""
        skill = self.get(skill_name)
        if skill is None:
            raise KeyError(f"Skill '{skill_name}' is not registered.")
        return skill

    def __len__(self) -> int:
        """Return total number of registered skills."""
        with self._lock:
            return len(self._skills)

    def __iter__(self):
        """Iterate over registered skills in priority order."""
        return iter(self.list_skills())

    def __repr__(self) -> str:
        """Developer-friendly string representation."""
        with self._lock:
            count = len(self._skills)
            enabled_count = sum(1 for s in self._skills.values() if s.enabled)
        return (
            f"<SkillManager(total_skills={count}, enabled_skills={enabled_count})>"
        )


# Global default SkillManager singleton
skill_manager: Final[SkillManager] = SkillManager()
