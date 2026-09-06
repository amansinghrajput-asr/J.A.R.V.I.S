"""Command Router Subsystem for J.A.R.V.I.S.

Coordinates the intake, normalization, alias expansion, preprocessing, middleware
chains, intent classification, and dispatch of user commands to the SkillManager,
while publishing comprehensive lifecycle events across the system Event Bus.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
import threading
import time
import uuid
from typing import Any, Callable, Final, Optional, Union

from app.core.config import Settings, settings
from app.core.container import JarvisException, ServiceContainer, container
from app.core.event_bus import EventBus, event_bus
from app.core.logger import get_logger
from app.router.intent import Intent
from app.skills.base import BaseSkill, SkillError, SkillNotFoundError
from app.skills.manager import SkillManager, skill_manager


class RouterError(JarvisException):
    """Base exception for all command router errors."""


class RoutingError(RouterError):
    """Raised when routing a command fails or no capable skill is found."""


class CommandPreprocessError(RouterError):
    """Raised when an error occurs during command preprocessing."""


class MiddlewareError(RouterError):
    """Raised when an error occurs within a router middleware."""


class InvalidCommandError(RouterError):
    """Raised when an empty or malformed command input is provided."""


class _PreprocessorEntry:
    """Internal container storing a registered preprocessor callback."""

    __slots__ = ("priority", "order", "callback")

    def __init__(self, priority: int, order: int, callback: Callable[..., Any]) -> None:
        self.priority = priority
        self.order = order
        self.callback = callback


class _MiddlewareEntry:
    """Internal container storing a registered middleware callback."""

    __slots__ = ("priority", "order", "callback")

    def __init__(self, priority: int, order: int, callback: Callable[..., Any]) -> None:
        self.priority = priority
        self.order = order
        self.callback = callback


class _IntentRuleEntry:
    """Internal container storing a pattern-based intent classification rule."""

    __slots__ = ("intent_name", "pattern", "parameters_extractor", "priority", "order")

    def __init__(
        self,
        intent_name: str,
        pattern: re.Pattern[str],
        parameters_extractor: Optional[Callable[[re.Match[str]], dict[str, Any]]],
        priority: int,
        order: int,
    ) -> None:
        self.intent_name = intent_name
        self.pattern = pattern
        self.parameters_extractor = parameters_extractor
        self.priority = priority
        self.order = order


class CommandRouter:
    """Thread-safe, async-ready Command Router for J.A.R.V.I.S.

    Normalizes user text, resolves aliases, applies preprocessors, executes
    middleware chains, parses intents, routes commands to the SkillManager,
    and publishes full lifecycle events (received, routed, completed, failed).

    Integrates with Config, Logger, Event Bus, Service Container, and SkillManager.
    """

    def __init__(
        self,
        config: Optional[Settings] = None,
        logger: Optional[logging.Logger] = None,
        container_instance: Optional[ServiceContainer] = None,
        event_bus_instance: Optional[EventBus] = None,
        skill_manager_instance: Optional[SkillManager] = None,
        *,
        auto_register_in_container: bool = True,
    ) -> None:
        """Initialize the CommandRouter.

        Args:
            config: Optional Settings instance. If None, resolves from container or global settings.
            logger: Optional Logger instance. If None, creates 'COMMAND_ROUTER' logger.
            container_instance: Optional ServiceContainer. If None, uses global container.
            event_bus_instance: Optional EventBus. If None, resolves from container or global event bus.
            skill_manager_instance: Optional SkillManager. If None, resolves from container or global skill manager.
            auto_register_in_container: If True, registers this CommandRouter into container.
        """
        self._lock = threading.RLock()
        self._aliases: dict[str, str] = {}
        self._preprocessors: list[_PreprocessorEntry] = []
        self._middleware: list[_MiddlewareEntry] = []
        self._intent_rules: list[_IntentRuleEntry] = []
        self._order_counter: int = 0

        # 1. Dependency resolution: Service Container
        self._container = container_instance if container_instance is not None else container

        # 2. Dependency resolution: Logger
        self._logger = logger if logger is not None else get_logger("COMMAND_ROUTER")

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

        # 5. Dependency resolution: SkillManager
        self._skill_manager_explicit: Optional[SkillManager] = skill_manager_instance
        if skill_manager_instance is not None:
            self._skill_manager = skill_manager_instance
        elif self._container.exists("skill_manager"):
            self._skill_manager = self._container.resolve("skill_manager")
        else:
            self._skill_manager = skill_manager

        # 6. Self-registration in Service Container
        if auto_register_in_container:
            try:
                self._container.register_singleton("command_router", self, allow_override=True)
                self._container.register_singleton("router", self, allow_override=True)
                self._logger.debug("Registered 'command_router' singleton into Service Container.")
            except Exception as exc:
                self._logger.warning(f"Could not register CommandRouter into container: {exc}")

    # --------------------------------------------------------------------------
    # Properties for System Dependencies
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
    def skill_manager(self) -> SkillManager:
        """Retrieve the active skill manager instance."""
        if self._skill_manager_explicit is not None:
            return self._skill_manager_explicit
        if self._container is not None and self._container.exists("skill_manager"):
            return self._container.resolve("skill_manager")
        if self._skill_manager is not None:
            return self._skill_manager
        from app.skills.manager import skill_manager as global_sm
        return global_sm

    # --------------------------------------------------------------------------
    # Command Normalization
    # --------------------------------------------------------------------------

    def normalize(self, command: str) -> str:
        """Normalize a raw command string.

        Performs:
        - Leading and trailing whitespace stripping.
        - Whitespace collapsing (converting tabs, newlines, multi-spaces to single space).
        - Wake-word prefix stripping (e.g., 'hey jarvis', 'jarvis').
        - Trailing punctuation stripping (e.g., '.', '!', '?').
        - Case folding (lowercasing).

        Args:
            command: The command text to normalize.

        Returns:
            The normalized canonical command string.
        """
        if not isinstance(command, str):
            command = str(command) if command is not None else ""

        text = command.strip()
        if not text:
            return ""

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)

        # Lowercase for canonical processing
        lowered = text.lower()

        # Wake-word removal
        wake_words = ["hey jarvis", "jarvis"]
        configured_wake = getattr(self._config, "wake_word", None)
        if configured_wake and isinstance(configured_wake, str) and configured_wake.strip():
            clean_wake = configured_wake.strip().lower()
            if clean_wake not in wake_words:
                wake_words.append(clean_wake)

        # Sort wake words longest first so "hey jarvis" matches before "jarvis"
        wake_words.sort(key=len, reverse=True)

        for ww in wake_words:
            if lowered == ww:
                lowered = ""
                break
            if lowered.startswith(ww + ","):
                lowered = lowered[len(ww) + 1 :].lstrip()
                break
            if lowered.startswith(ww + " "):
                lowered = lowered[len(ww) + 1 :].lstrip()
                break
            if lowered.startswith(ww + ":"):
                lowered = lowered[len(ww) + 1 :].lstrip()
                break

        # Strip trailing punctuation if present (. ! ?)
        lowered = re.sub(r"[.!?]+$", "", lowered).strip()

        # Collapse internal whitespace again after stripping
        return re.sub(r"\s+", " ", lowered).strip()

    # --------------------------------------------------------------------------
    # Alias Management
    # --------------------------------------------------------------------------

    def register_alias(self, alias: str, target: str) -> None:
        """Register a command alias.

        Supports exact command alias substitution and prefix alias substitution
        with parameter preservation.

        Args:
            alias: The alias trigger keyword or phrase.
            target: The target command or prefix to substitute.

        Raises:
            ValueError: If alias or target is empty or invalid.
        """
        if not isinstance(alias, str) or not alias.strip():
            raise ValueError("Alias trigger must be a non-empty string.")
        if not isinstance(target, str) or not target.strip():
            raise ValueError("Alias target must be a non-empty string.")

        clean_alias = alias.strip().lower()
        clean_target = target.strip()

        with self._lock:
            self._aliases[clean_alias] = clean_target
            self._logger.debug(f"Registered alias: '{clean_alias}' -> '{clean_target}'")

    def unregister_alias(self, alias: str) -> bool:
        """Unregister a previously registered command alias.

        Args:
            alias: The alias trigger to remove.

        Returns:
            True if the alias was found and removed, False otherwise.
        """
        if not isinstance(alias, str) or not alias.strip():
            return False

        clean_alias = alias.strip().lower()
        with self._lock:
            removed = self._aliases.pop(clean_alias, None) is not None

        if removed:
            self._logger.debug(f"Unregistered alias: '{clean_alias}'")
        return removed

    def has_alias(self, alias: str) -> bool:
        """Check if an alias is registered.

        Args:
            alias: The alias trigger to check.

        Returns:
            True if registered, False otherwise.
        """
        if not isinstance(alias, str) or not alias.strip():
            return False
        with self._lock:
            return alias.strip().lower() in self._aliases

    def get_alias(self, alias: str) -> Optional[str]:
        """Retrieve the target for a given alias.

        Args:
            alias: The alias trigger.

        Returns:
            The target command string if found, None otherwise.
        """
        if not isinstance(alias, str) or not alias.strip():
            return None
        with self._lock:
            return self._aliases.get(alias.strip().lower())

    def list_aliases(self) -> dict[str, str]:
        """Retrieve a copy of all registered aliases.

        Returns:
            Dictionary mapping alias triggers to targets.
        """
        with self._lock:
            return dict(self._aliases)

    def clear_aliases(self) -> None:
        """Remove all registered aliases."""
        with self._lock:
            self._aliases.clear()
            self._logger.debug("Cleared all command aliases.")

    def resolve_alias(self, command: str, max_depth: int = 5) -> str:
        """Resolve a command against registered aliases, preventing recursion cycles.

        Args:
            command: The command text to expand.
            max_depth: Maximum recursion depth to prevent infinite loops.

        Returns:
            The expanded command text.
        """
        current = command.strip()
        visited: set[str] = set()
        depth = 0

        with self._lock:
            aliases_snapshot = dict(self._aliases)

        while depth < max_depth:
            depth += 1
            lowered = current.lower()

            if lowered in visited:
                break
            visited.add(lowered)

            # 1. Exact match
            if lowered in aliases_snapshot:
                current = aliases_snapshot[lowered]
                continue

            # 2. Prefix match (alias + space + arguments)
            matched = False
            for alias_key, alias_val in aliases_snapshot.items():
                prefix = alias_key + " "
                if lowered.startswith(prefix):
                    args = current[len(prefix) :].strip()
                    current = f"{alias_val} {args}".strip()
                    matched = True
                    break

            if not matched:
                break

        return current

    # --------------------------------------------------------------------------
    # Preprocessor Management
    # --------------------------------------------------------------------------

    def register_preprocessor(
        self,
        callback: Optional[Callable[..., Any]] = None,
        *,
        priority: int = 100,
    ) -> Any:
        """Register a command preprocessor callback or use as a decorator.

        Preprocessors receive a command string or Intent and return a transformed
        command string or Intent. Higher priority preprocessors run first.

        Args:
            callback: Callable preprocessor.
            priority: Execution priority (default: 100).

        Returns:
            Unsubscribe callable `() -> bool` if callback provided, else decorator.
        """

        def _register(fn: Callable[..., Any]) -> Callable[[], bool]:
            if not callable(fn):
                raise TypeError(f"Preprocessor must be callable, got {type(fn).__name__}")

            with self._lock:
                self._order_counter += 1
                entry = _PreprocessorEntry(
                    priority=int(priority),
                    order=self._order_counter,
                    callback=fn,
                )
                self._preprocessors.append(entry)
                self._preprocessors.sort(key=lambda p: (-p.priority, p.order))
                fn_name = getattr(fn, "__name__", repr(fn))
                self._logger.debug(f"Registered preprocessor '{fn_name}' (priority={priority})")

            def _unregister() -> bool:
                return self.unregister_preprocessor(fn)

            return _unregister

        if callback is None:
            def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
                _register(fn)
                return fn

            return decorator

        return _register(callback)

    def unregister_preprocessor(self, callback: Callable[..., Any]) -> bool:
        """Unregister a previously registered preprocessor.

        Args:
            callback: The preprocessor callable to remove.

        Returns:
            True if found and removed, False otherwise.
        """
        with self._lock:
            original_len = len(self._preprocessors)
            self._preprocessors = [p for p in self._preprocessors if p.callback != callback]
            removed = len(self._preprocessors) < original_len

        if removed:
            fn_name = getattr(callback, "__name__", repr(callback))
            self._logger.debug(f"Unregistered preprocessor '{fn_name}'")
        return removed

    def clear_preprocessors(self) -> None:
        """Remove all registered preprocessors."""
        with self._lock:
            self._preprocessors.clear()
            self._logger.debug("Cleared all command preprocessors.")

    def _apply_preprocessors(self, command: str) -> str:
        """Synchronously execute registered preprocessors in priority order."""
        with self._lock:
            preprocessors = list(self._preprocessors)

        current = command
        for entry in preprocessors:
            try:
                res = entry.callback(current)
                if inspect.iscoroutine(res):
                    try:
                        loop = asyncio.get_running_loop()
                        res = loop.create_task(res)
                    except RuntimeError:
                        res = asyncio.run(res)

                if isinstance(res, str):
                    current = res
                elif isinstance(res, Intent):
                    current = res.normalized_command
            except Exception as exc:
                fn_name = getattr(entry.callback, "__name__", repr(entry.callback))
                self._logger.exception(f"Error in preprocessor '{fn_name}': {exc}")
                raise CommandPreprocessError(f"Preprocessor '{fn_name}' failed: {exc}") from exc

        return current

    async def _apply_preprocessors_async(self, command: str) -> str:
        """Asynchronously execute registered preprocessors in priority order."""
        with self._lock:
            preprocessors = list(self._preprocessors)

        current = command
        for entry in preprocessors:
            try:
                res = entry.callback(current)
                if inspect.iscoroutine(res):
                    res = await res

                if isinstance(res, str):
                    current = res
                elif isinstance(res, Intent):
                    current = res.normalized_command
            except Exception as exc:
                fn_name = getattr(entry.callback, "__name__", repr(entry.callback))
                self._logger.exception(f"Error in async preprocessor '{fn_name}': {exc}")
                raise CommandPreprocessError(
                    f"Async preprocessor '{fn_name}' failed: {exc}"
                ) from exc

        return current

    # --------------------------------------------------------------------------
    # Middleware Architecture
    # --------------------------------------------------------------------------

    def register_middleware(
        self,
        middleware: Optional[Callable[..., Any]] = None,
        *,
        priority: int = 100,
    ) -> Any:
        """Register a middleware callback in the execution pipeline or use as decorator.

        Middleware intercept the Intent before skill execution and can execute logic
        both before and after the downstream handler:
        `middleware(intent: Intent, next_handler: Callable[[Intent], Any]) -> Any`

        Examples for future expansion: Authentication, Logging, Rate limiting,
        Telemetry, Memory injection, and AI reasoning.

        Args:
            middleware: Callable middleware function.
            priority: Execution priority (default: 100). Higher runs first.

        Returns:
            Unsubscribe callable `() -> bool` if callback provided, else decorator.
        """

        def _register(fn: Callable[..., Any]) -> Callable[[], bool]:
            if not callable(fn):
                raise TypeError(f"Middleware must be callable, got {type(fn).__name__}")

            with self._lock:
                self._order_counter += 1
                entry = _MiddlewareEntry(
                    priority=int(priority),
                    order=self._order_counter,
                    callback=fn,
                )
                self._middleware.append(entry)
                self._middleware.sort(key=lambda m: (-m.priority, m.order))
                fn_name = getattr(fn, "__name__", repr(fn))
                self._logger.debug(f"Registered middleware '{fn_name}' (priority={priority})")

            def _unregister() -> bool:
                return self.unregister_middleware(fn)

            return _unregister

        if middleware is None:
            def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
                _register(fn)
                return fn

            return decorator

        return _register(middleware)

    def unregister_middleware(self, middleware: Callable[..., Any]) -> bool:
        """Unregister a previously registered middleware callback.

        Args:
            middleware: The middleware callable to remove.

        Returns:
            True if found and removed, False otherwise.
        """
        with self._lock:
            original_len = len(self._middleware)
            self._middleware = [m for m in self._middleware if m.callback != middleware]
            removed = len(self._middleware) < original_len

        if removed:
            fn_name = getattr(middleware, "__name__", repr(middleware))
            self._logger.debug(f"Unregistered middleware '{fn_name}'")
        return removed

    def list_middleware(self) -> list[Callable[..., Any]]:
        """Retrieve a list of registered middleware callables in execution order."""
        with self._lock:
            return [m.callback for m in self._middleware]

    def clear_middleware(self) -> None:
        """Remove all registered middleware callbacks."""
        with self._lock:
            self._middleware.clear()
            self._logger.debug("Cleared all router middleware.")

    def _execute_middleware_chain(
        self,
        intent: Intent,
        terminal_handler: Callable[[Intent], Any],
    ) -> Any:
        """Construct and execute synchronous middleware chain around terminal handler."""
        with self._lock:
            middleware_list = [m.callback for m in self._middleware]

        if not middleware_list:
            return terminal_handler(intent)

        def build_chain(index: int) -> Callable[[Intent], Any]:
            if index >= len(middleware_list):
                return terminal_handler

            current_middleware = middleware_list[index]
            next_handler = build_chain(index + 1)

            def step(target_intent: Intent) -> Any:
                res = current_middleware(target_intent, next_handler)
                if inspect.iscoroutine(res):
                    try:
                        loop = asyncio.get_running_loop()
                        res = loop.create_task(res)
                    except RuntimeError:
                        res = asyncio.run(res)
                return res

            return step

        pipeline = build_chain(0)
        return pipeline(intent)

    async def _execute_middleware_chain_async(
        self,
        intent: Intent,
        terminal_handler: Callable[[Intent], Any],
    ) -> Any:
        """Construct and execute asynchronous middleware chain around terminal handler."""
        with self._lock:
            middleware_list = [m.callback for m in self._middleware]

        if not middleware_list:
            res = terminal_handler(intent)
            if inspect.iscoroutine(res):
                res = await res
            return res

        def build_chain(index: int) -> Callable[[Intent], Any]:
            if index >= len(middleware_list):
                return terminal_handler

            current_middleware = middleware_list[index]
            next_handler = build_chain(index + 1)

            async def step(target_intent: Intent) -> Any:
                res = current_middleware(target_intent, next_handler)
                if inspect.iscoroutine(res):
                    res = await res
                return res

            return step

        pipeline = build_chain(0)
        res = pipeline(intent)
        if inspect.iscoroutine(res):
            res = await res
        return res

    # --------------------------------------------------------------------------
    # Intent Classification Rules (Deterministic)
    # --------------------------------------------------------------------------

    def register_intent_rule(
        self,
        intent_name: str,
        pattern: Union[str, re.Pattern[str]],
        parameters_extractor: Optional[Callable[[re.Match[str]], dict[str, Any]]] = None,
        *,
        priority: int = 100,
    ) -> Callable[[], bool]:
        """Register a pattern-based intent classification rule.

        Enables fast, zero-cost deterministic intent resolution without external LLM calls.

        Args:
            intent_name: Target intent identifier (e.g., 'open_app', 'weather_query').
            pattern: Regex pattern string or compiled regex pattern to match.
            parameters_extractor: Optional callable taking the regex Match and returning parameters dict.
            priority: Priority for evaluation order (higher evaluated first).

        Returns:
            Unsubscribe callable `() -> bool`.
        """
        if not isinstance(intent_name, str) or not intent_name.strip():
            raise ValueError("Intent name must be a non-empty string.")

        compiled_pattern = (
            pattern if isinstance(pattern, re.Pattern) else re.compile(pattern, re.IGNORECASE)
        )

        with self._lock:
            self._order_counter += 1
            rule = _IntentRuleEntry(
                intent_name=intent_name.strip(),
                pattern=compiled_pattern,
                parameters_extractor=parameters_extractor,
                priority=int(priority),
                order=self._order_counter,
            )
            self._intent_rules.append(rule)
            self._intent_rules.sort(key=lambda r: (-r.priority, r.order))
            self._logger.debug(
                f"Registered intent rule '{rule.intent_name}' for pattern '{compiled_pattern.pattern}'"
            )

        def _unregister() -> bool:
            with self._lock:
                if rule in self._intent_rules:
                    self._intent_rules.remove(rule)
                    return True
                return False

        return _unregister

    def clear_intent_rules(self) -> None:
        """Clear all registered intent classification rules."""
        with self._lock:
            self._intent_rules.clear()
            self._logger.debug("Cleared all intent rules.")

    def parse_intent(
        self,
        command: Union[str, Intent, dict[str, Any]],
        *,
        normalized_override: Optional[str] = None,
        source: str = "text",
        context: Optional[dict[str, Any]] = None,
    ) -> Intent:
        """Parse or normalize a command into a standardized Intent object.

        Args:
            command: Input raw command string, dictionary, or existing Intent.
            normalized_override: Optional pre-normalized text to use.
            source: Source channel ('voice', 'text', 'cli', 'api', 'automation').
            context: Optional initial contextual dictionary.

        Returns:
            Constructed and validated Intent instance.
        """
        initial_ctx = dict(context) if context else {}

        if isinstance(command, Intent):
            if source != "text" and command.source == "text":
                command.source = source
            if initial_ctx:
                command.context.update(initial_ctx)
            return command

        if isinstance(command, dict):
            parsed = Intent.from_dict(command)
            if source != "text" and parsed.source == "text":
                parsed.source = source
            if initial_ctx:
                parsed.context.update(initial_ctx)
            return parsed

        raw_str = str(command) if command is not None else ""
        norm_str = normalized_override if normalized_override is not None else self.normalize(raw_str)

        # Evaluate registered intent rules
        with self._lock:
            rules_snapshot = list(self._intent_rules)

        for rule in rules_snapshot:
            match = rule.pattern.search(norm_str)
            if match:
                params: dict[str, Any] = {}
                if rule.parameters_extractor is not None:
                    try:
                        extracted = rule.parameters_extractor(match)
                        if isinstance(extracted, dict):
                            params = extracted
                    except Exception as exc:
                        self._logger.warning(
                            f"Error extracting parameters for intent '{rule.intent_name}': {exc}"
                        )
                elif match.groupdict():
                    params = match.groupdict()

                return Intent(
                    raw_command=raw_str,
                    normalized_command=norm_str,
                    intent_name=rule.intent_name,
                    confidence=1.0,
                    parameters=params,
                    timestamp=time.time(),
                    id=str(uuid.uuid4()),
                    source=source,
                    context=initial_ctx,
                )

        # Default fallback intent parsing: first token as verb or 'unknown'
        tokens = norm_str.split()
        intent_name = tokens[0] if tokens else "unknown"
        parameters = {"args": tokens[1:]} if len(tokens) > 1 else {}

        return Intent(
            raw_command=raw_str,
            normalized_command=norm_str,
            intent_name=intent_name,
            confidence=0.8 if tokens else 0.0,
            parameters=parameters,
            timestamp=time.time(),
            id=str(uuid.uuid4()),
            source=source,
            context=initial_ctx,
        )

    # --------------------------------------------------------------------------
    # Helper: Find Matched Skill and Command Payload
    # --------------------------------------------------------------------------

    def _find_matching_skill(
        self,
        intent: Intent,
        targeted_skill_name: Optional[str] = None,
    ) -> tuple[Optional[BaseSkill], Any]:
        """Find the matching skill and the appropriate command payload.

        Evaluates registered skills in priority order:
        1. Targeted skill explicitly requested by name.
        2. Specialized skills (evaluated first in priority order, excluding AISkill).
        3. AISkill fallback (from SkillManager or ServiceContainer).

        Args:
            intent: Standardized user Intent object.
            targeted_skill_name: Optional explicit skill name override.

        Returns:
            Tuple of (matched_skill, command_payload) or (None, intent).
        """
        sm = self.skill_manager

        # 1. Targeted skill requested directly
        if targeted_skill_name:
            clean_target = targeted_skill_name.strip()
            skill = sm.get(clean_target)
            if skill is not None:
                if skill.can_handle(intent):
                    return skill, intent
                if skill.can_handle(intent.normalized_command):
                    return skill, intent.normalized_command
                if skill.can_handle(intent.raw_command):
                    return skill, intent.raw_command
                return skill, intent
            if self._container is not None and self._container.exists(clean_target):
                c_skill = self._container.resolve(clean_target)
                if isinstance(c_skill, BaseSkill):
                    return c_skill, intent
            return None, intent

        # 2. Retrieve enabled skills from SkillManager (ordered by -priority, name)
        skills = sm.list_skills(enabled_only=True)

        specialized_skills: list[BaseSkill] = []
        ai_skills: list[BaseSkill] = []

        for s in skills:
            if s.name == "ai" or getattr(s, "name", "") == "ai":
                ai_skills.append(s)
            else:
                specialized_skills.append(s)

        # 3. Evaluate specialized skills first in priority order
        for s in specialized_skills:
            try:
                if s.can_handle(intent):
                    return s, intent
                if s.can_handle(intent.normalized_command):
                    return s, intent.normalized_command
                if s.can_handle(intent.raw_command):
                    return s, intent.raw_command
            except Exception:
                continue

        # 4. If no specialized skill matched, evaluate registered AI fallback skill
        for s in ai_skills:
            try:
                if s.can_handle(intent):
                    return s, intent
                if s.can_handle(intent.normalized_command):
                    return s, intent.normalized_command
                if s.can_handle(intent.raw_command):
                    return s, intent.raw_command
            except Exception:
                continue

        # 5. Check ServiceContainer for 'ai_skill' or 'ai' if not registered in SkillManager
        if self._container is not None:
            for key in ("ai_skill", "ai"):
                if self._container.exists(key):
                    try:
                        c_ai = self._container.resolve(key)
                        if isinstance(c_ai, BaseSkill) and getattr(c_ai, "enabled", True):
                            if c_ai.can_handle(intent):
                                return c_ai, intent
                            if c_ai.can_handle(intent.normalized_command):
                                return c_ai, intent.normalized_command
                            if c_ai.can_handle(intent.raw_command):
                                return c_ai, intent.raw_command
                    except Exception:
                        pass

        return None, intent

    def _find_matching_skill_name(
        self,
        intent: Intent,
        targeted_skill_name: Optional[str] = None,
    ) -> str:
        """Find the name of the skill that can handle or was targeted for this intent."""
        skill, _ = self._find_matching_skill(intent, targeted_skill_name)
        return skill.name if skill is not None else "unknown"

    # --------------------------------------------------------------------------
    # Synchronous Command Routing
    # --------------------------------------------------------------------------

    def route(
        self,
        command: Union[str, Intent, dict[str, Any]],
        *,
        skill_name: Optional[str] = None,
        source: str = "text",
        context: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Route a user command synchronously to an appropriate Skill.

        Lifecycle:
        1. Validate input and publish 'command.received' event.
        2. Apply normalization, preprocessors, and alias resolution.
        3. Parse command into standardized Intent dataclass (with UUID, source, context).
        4. Publish 'command.routed' when target skill is identified.
        5. Pass Intent through Middleware Chain.
        6. Publish 'router.skill.started' before skill execution.
        7. Forward command to SkillManager / matched skill.
        8. Publish 'router.skill.completed' on success or 'router.skill.failed' on error.
        9. Publish 'command.completed' upon execution completion.
        10. Publish 'command.failed' on error.

        Args:
            command: Raw command text, dictionary, or Intent instance.
            skill_name: Optional explicit skill name to target directly.
            source: Input source channel ('voice', 'text', 'cli', 'api', 'automation').
            context: Optional initial contextual dictionary.

        Returns:
            The execution result from the handling Skill.

        Raises:
            InvalidCommandError: If command is empty or invalid.
            RoutingError: If no skill can handle the command or execution fails.
        """
        start_time = time.perf_counter()

        # Extract raw representation
        if isinstance(command, Intent):
            raw_command = command.raw_command
        elif isinstance(command, dict):
            raw_command = str(command.get("raw_command", command.get("command", "")))
        else:
            raw_command = str(command) if command is not None else ""

        if not raw_command.strip():
            self._event_bus.publish(
                "command.failed",
                payload={
                    "command": raw_command,
                    "intent": None,
                    "error": "Command cannot be empty.",
                    "exception_type": "InvalidCommandError",
                    "duration": 0.0,
                    "timestamp": time.time(),
                },
                source="command_router",
            )
            raise InvalidCommandError("Command cannot be empty or whitespace only.")

        # 1. Publish command.received event
        self._event_bus.publish(
            "command.received",
            payload={
                "command": raw_command,
                "raw_command": raw_command,
                "source": source,
                "timestamp": time.time(),
            },
            source="command_router",
        )

        intent: Optional[Intent] = None
        matched_skill_name = "unknown"
        try:
            # 2. Preprocess & Normalize
            if isinstance(command, Intent):
                intent = command
                if source != "text" and intent.source == "text":
                    intent.source = source
                if context:
                    intent.context.update(context)
            else:
                normalized = self.normalize(raw_command)
                preprocessed = self._apply_preprocessors(normalized)
                expanded = self.resolve_alias(preprocessed)
                final_normalized = self.normalize(expanded)
                intent = self.parse_intent(
                    raw_command,
                    normalized_override=final_normalized,
                    source=source,
                    context=context,
                )

            matched_skill, _ = self._find_matching_skill(intent, skill_name)
            matched_skill_name = matched_skill.name if matched_skill is not None else "unknown"

            # 3. Publish command.routed event (routing established)
            self._event_bus.publish(
                "command.routed",
                payload={
                    "command": intent.normalized_command,
                    "raw_command": intent.raw_command,
                    "intent": intent.to_dict(),
                    "skill_name": matched_skill_name,
                    "timestamp": time.time(),
                },
                source="command_router",
            )

            # 4. Terminal execution handler invoking matched skill
            def terminal_dispatch(target_intent: Intent) -> Any:
                target_skill, payload_to_send = self._find_matching_skill(target_intent, skill_name)

                if target_skill is None:
                    # Fallback to standard SkillManager routing to raise SkillNotFoundError
                    try:
                        return self.skill_manager.execute(target_intent, skill_name=skill_name)
                    except SkillNotFoundError:
                        try:
                            return self.skill_manager.execute(
                                target_intent.normalized_command, skill_name=skill_name
                            )
                        except SkillNotFoundError:
                            return self.skill_manager.execute(
                                target_intent.raw_command, skill_name=skill_name
                            )

                skill_start = time.perf_counter()
                self._event_bus.publish(
                    "router.skill.started",
                    payload={
                        "command": target_intent.raw_command,
                        "raw_command": target_intent.raw_command,
                        "normalized_command": target_intent.normalized_command,
                        "intent": target_intent.to_dict(),
                        "skill_name": target_skill.name,
                        "timestamp": time.time(),
                    },
                    source="command_router",
                )

                try:
                    if self.skill_manager.has_skill(target_skill.name):
                        res = self.skill_manager.execute(payload_to_send, skill_name=target_skill.name)
                    else:
                        res = target_skill.execute(payload_to_send)

                    if inspect.iscoroutine(res):
                        try:
                            loop = asyncio.get_running_loop()
                            res = loop.create_task(res)
                        except RuntimeError:
                            res = asyncio.run(res)

                    skill_duration = time.perf_counter() - skill_start
                    self._event_bus.publish(
                        "router.skill.completed",
                        payload={
                            "command": target_intent.raw_command,
                            "intent": target_intent.to_dict(),
                            "skill_name": target_skill.name,
                            "result": res,
                            "duration": skill_duration,
                            "success": True,
                            "timestamp": time.time(),
                        },
                        source="command_router",
                    )
                    return res

                except Exception as exc:
                    skill_duration = time.perf_counter() - skill_start
                    self._event_bus.publish(
                        "router.skill.failed",
                        payload={
                            "command": target_intent.raw_command,
                            "intent": target_intent.to_dict(),
                            "skill_name": target_skill.name,
                            "error": str(exc),
                            "exception_type": type(exc).__name__,
                            "duration": skill_duration,
                            "timestamp": time.time(),
                        },
                        source="command_router",
                    )
                    raise

            # 5. Execute through Middleware Chain
            result = self._execute_middleware_chain(intent, terminal_dispatch)

            # Handle returned coroutine if invoked in sync context
            if inspect.iscoroutine(result):
                try:
                    loop = asyncio.get_running_loop()
                    result = loop.create_task(result)
                except RuntimeError:
                    result = asyncio.run(result)

            duration = time.perf_counter() - start_time

            # 6. Publish command.completed event (execution complete)
            self._event_bus.publish(
                "command.completed",
                payload={
                    "intent": intent.to_dict(),
                    "skill_name": matched_skill_name,
                    "duration": duration,
                    "success": True,
                    "result": result,
                    "timestamp": time.time(),
                },
                source="command_router",
            )
            return result

        except Exception as exc:
            duration = time.perf_counter() - start_time

            # 7. Publish command.completed (failed state) and command.failed
            self._event_bus.publish(
                "command.completed",
                payload={
                    "intent": intent.to_dict() if intent is not None else None,
                    "skill_name": matched_skill_name,
                    "duration": duration,
                    "success": False,
                    "result": None,
                    "timestamp": time.time(),
                },
                source="command_router",
            )

            self._event_bus.publish(
                "command.failed",
                payload={
                    "command": raw_command,
                    "intent": intent.to_dict() if intent is not None else None,
                    "error": str(exc),
                    "exception_type": type(exc).__name__,
                    "duration": duration,
                    "timestamp": time.time(),
                },
                source="command_router",
            )
            self._logger.exception(f"Routing failed for command '{raw_command}': {exc}")

            if isinstance(exc, RouterError):
                raise
            raise RoutingError(f"Failed to route command '{raw_command}': {exc}") from exc

    # --------------------------------------------------------------------------
    # Asynchronous Command Routing
    # --------------------------------------------------------------------------

    async def route_async(
        self,
        command: Union[str, Intent, dict[str, Any]],
        *,
        skill_name: Optional[str] = None,
        source: str = "text",
        context: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Route a user command asynchronously to an appropriate Skill.

        Lifecycle:
        1. Validate input and publish 'command.received' event asynchronously.
        2. Apply normalization, async preprocessors, and alias resolution.
        3. Parse command into standardized Intent dataclass (with UUID, source, context).
        4. Publish 'command.routed' when target skill is identified.
        5. Pass Intent through Async Middleware Chain.
        6. Forward command to SkillManager (awaiting coroutine skills).
        7. Publish 'command.completed' upon execution completion.
        8. Publish 'command.failed' on error.

        Args:
            command: Raw command text, dictionary, or Intent instance.
            skill_name: Optional explicit skill name to target directly.
            source: Input source channel ('voice', 'text', 'cli', 'api', 'automation').
            context: Optional initial contextual dictionary.

        Returns:
            The execution result from the handling Skill.

        Raises:
            InvalidCommandError: If command is empty or invalid.
            RoutingError: If no skill can handle the command or execution fails.
        """
        start_time = time.perf_counter()

        # Extract raw representation
        if isinstance(command, Intent):
            raw_command = command.raw_command
        elif isinstance(command, dict):
            raw_command = str(command.get("raw_command", command.get("command", "")))
        else:
            raw_command = str(command) if command is not None else ""

        if not raw_command.strip():
            await self._event_bus.publish_async(
                "command.failed",
                payload={
                    "command": raw_command,
                    "intent": None,
                    "error": "Command cannot be empty.",
                    "exception_type": "InvalidCommandError",
                    "duration": 0.0,
                    "timestamp": time.time(),
                },
                source="command_router",
            )
            raise InvalidCommandError("Command cannot be empty or whitespace only.")

        # 1. Publish command.received event
        await self._event_bus.publish_async(
            "command.received",
            payload={
                "command": raw_command,
                "raw_command": raw_command,
                "source": source,
                "timestamp": time.time(),
            },
            source="command_router",
        )

        intent: Optional[Intent] = None
        matched_skill_name = "unknown"
        try:
            # 2. Preprocess & Normalize
            if isinstance(command, Intent):
                intent = command
                if source != "text" and intent.source == "text":
                    intent.source = source
                if context:
                    intent.context.update(context)
            else:
                normalized = self.normalize(raw_command)
                preprocessed = await self._apply_preprocessors_async(normalized)
                expanded = self.resolve_alias(preprocessed)
                final_normalized = self.normalize(expanded)
                intent = self.parse_intent(
                    raw_command,
                    normalized_override=final_normalized,
                    source=source,
                    context=context,
                )

            matched_skill, _ = self._find_matching_skill(intent, skill_name)
            matched_skill_name = matched_skill.name if matched_skill is not None else "unknown"

            # 3. Publish command.routed event
            await self._event_bus.publish_async(
                "command.routed",
                payload={
                    "command": intent.normalized_command,
                    "raw_command": intent.raw_command,
                    "intent": intent.to_dict(),
                    "skill_name": matched_skill_name,
                    "timestamp": time.time(),
                },
                source="command_router",
            )

            # 4. Terminal async dispatch handler
            async def terminal_dispatch_async(target_intent: Intent) -> Any:
                target_skill, payload_to_send = self._find_matching_skill(target_intent, skill_name)

                if target_skill is None:
                    # Fallback to standard SkillManager routing to raise SkillNotFoundError
                    try:
                        res = await self.skill_manager.execute_async(
                            target_intent, skill_name=skill_name
                        )
                    except SkillNotFoundError:
                        try:
                            res = await self.skill_manager.execute_async(
                                target_intent.normalized_command, skill_name=skill_name
                            )
                        except SkillNotFoundError:
                            res = await self.skill_manager.execute_async(
                                target_intent.raw_command, skill_name=skill_name
                            )
                    if inspect.iscoroutine(res):
                        res = await res
                    return res

                skill_start = time.perf_counter()
                await self._event_bus.publish_async(
                    "router.skill.started",
                    payload={
                        "command": target_intent.raw_command,
                        "raw_command": target_intent.raw_command,
                        "normalized_command": target_intent.normalized_command,
                        "intent": target_intent.to_dict(),
                        "skill_name": target_skill.name,
                        "timestamp": time.time(),
                    },
                    source="command_router",
                )

                try:
                    if self.skill_manager.has_skill(target_skill.name):
                        res = await self.skill_manager.execute_async(
                            payload_to_send, skill_name=target_skill.name
                        )
                    else:
                        res = await target_skill.execute_async(payload_to_send)

                    if inspect.iscoroutine(res):
                        res = await res

                    skill_duration = time.perf_counter() - skill_start
                    await self._event_bus.publish_async(
                        "router.skill.completed",
                        payload={
                            "command": target_intent.raw_command,
                            "intent": target_intent.to_dict(),
                            "skill_name": target_skill.name,
                            "result": res,
                            "duration": skill_duration,
                            "success": True,
                            "timestamp": time.time(),
                        },
                        source="command_router",
                    )
                    return res

                except Exception as exc:
                    skill_duration = time.perf_counter() - skill_start
                    await self._event_bus.publish_async(
                        "router.skill.failed",
                        payload={
                            "command": target_intent.raw_command,
                            "intent": target_intent.to_dict(),
                            "skill_name": target_skill.name,
                            "error": str(exc),
                            "exception_type": type(exc).__name__,
                            "duration": skill_duration,
                            "timestamp": time.time(),
                        },
                        source="command_router",
                    )
                    raise

            # 5. Execute through Async Middleware Chain
            result = await self._execute_middleware_chain_async(
                intent, terminal_dispatch_async
            )

            duration = time.perf_counter() - start_time

            # 6. Publish command.completed event
            await self._event_bus.publish_async(
                "command.completed",
                payload={
                    "intent": intent.to_dict(),
                    "skill_name": matched_skill_name,
                    "duration": duration,
                    "success": True,
                    "result": result,
                    "timestamp": time.time(),
                },
                source="command_router",
            )
            return result

        except Exception as exc:
            duration = time.perf_counter() - start_time

            # 7. Publish command.completed (failed state) and command.failed
            await self._event_bus.publish_async(
                "command.completed",
                payload={
                    "intent": intent.to_dict() if intent is not None else None,
                    "skill_name": matched_skill_name,
                    "duration": duration,
                    "success": False,
                    "result": None,
                    "timestamp": time.time(),
                },
                source="command_router",
            )

            await self._event_bus.publish_async(
                "command.failed",
                payload={
                    "command": raw_command,
                    "intent": intent.to_dict() if intent is not None else None,
                    "error": str(exc),
                    "exception_type": type(exc).__name__,
                    "duration": duration,
                    "timestamp": time.time(),
                },
                source="command_router",
            )
            self._logger.exception(f"Async routing failed for command '{raw_command}': {exc}")

            if isinstance(exc, RouterError):
                raise
            raise RoutingError(f"Failed to route command '{raw_command}': {exc}") from exc

    def __repr__(self) -> str:
        """Developer-friendly string representation."""
        with self._lock:
            alias_count = len(self._aliases)
            prep_count = len(self._preprocessors)
            mid_count = len(self._middleware)
            rule_count = len(self._intent_rules)
        return (
            f"<CommandRouter(aliases={alias_count}, preprocessors={prep_count}, "
            f"middleware={mid_count}, rules={rule_count})>"
        )


# Global default CommandRouter singleton
command_router: Final[CommandRouter] = CommandRouter()
