"""Production Conversational AI Skill for J.A.R.V.I.S.

Connects the CommandRouter and SkillManager with the AI subsystem (AIManager),
serving as the default natural language conversational fallback skill.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Final, Optional, Union

from app.core.config import Settings
from app.core.container import ServiceContainer
from app.core.event_bus import EventBus
from app.core.logger import get_logger
from app.skills.base import BaseSkill, SkillExecutionError

if TYPE_CHECKING:
    from app.ai.manager import AIManager
    from app.router.intent import Intent

DEFAULT_AI_SKILL_NAME: Final[str] = "ai"
DEFAULT_AI_SKILL_DESCRIPTION: Final[str] = (
    "General conversational AI skill powered by Gemini."
)
DEFAULT_AI_SKILL_PRIORITY: Final[int] = -100


class AISkill(BaseSkill):
    """Default conversational AI skill powered by Google Gemini.

    Acts as the fallback conversational handler for J.A.R.V.I.S. Given a lower
    priority (-100) than deterministic or tool-based skills (default 100), it
    captures queries that do not match specialized commands and routes them to
    the AIManager for generative reasoning.

    Attributes:
        name: Unique identifier ('ai').
        description: Informative summary of the skill's capabilities.
        version: Semantic version string.
        enabled: Boolean flag indicating if the skill is currently active.
        priority: Low execution priority (-100) to serve as conversational fallback.
        tags: Discovery tags for metadata filtering.
    """

    name: str = DEFAULT_AI_SKILL_NAME
    description: str = DEFAULT_AI_SKILL_DESCRIPTION
    version: str = "1.0.0"
    enabled: bool = True
    priority: int = DEFAULT_AI_SKILL_PRIORITY
    tags: list[str] = ["ai", "conversation", "gemini", "llm", "fallback"]
    permissions: set[str] = set()

    def __init__(
        self,
        ai_manager_instance: Optional[AIManager] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        version: Optional[str] = None,
        enabled: Optional[bool] = None,
        priority: Optional[int] = None,
        tags: Optional[list[str]] = None,
        permissions: Optional[Union[set[str], list[str]]] = None,
        config: Optional[Settings] = None,
        logger: Optional[logging.Logger] = None,
        container: Optional[ServiceContainer] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        """Initialize the AISkill instance.

        All parameters have default values to allow automatic discovery and
        instantiation by SkillManager.

        Args:
            ai_manager_instance: Optional explicit AIManager dependency.
            name: Optional override for skill name (default: 'ai').
            description: Optional override for description.
            version: Optional override for semantic version.
            enabled: Optional override for enabled status.
            priority: Optional execution priority (default: -100).
            tags: Optional discovery tags.
            permissions: Optional required permissions.
            config: Injected Settings instance.
            logger: Injected Logger instance.
            container: Injected ServiceContainer instance.
            event_bus: Injected EventBus instance.
        """
        super().__init__(
            name=name if name is not None else self.name,
            description=description if description is not None else self.description,
            version=version if version is not None else self.version,
            enabled=enabled if enabled is not None else self.enabled,
            priority=priority if priority is not None else self.priority,
            tags=tags if tags is not None else list(self.tags),
            permissions=permissions if permissions is not None else set(self.permissions),
            config=config,
            logger=logger,
            container=container,
            event_bus=event_bus,
        )
        self._ai_manager: Optional[AIManager] = ai_manager_instance
        self._lock = threading.RLock()
        self._execution_count: int = 0
        self._error_count: int = 0

    @property
    def ai_manager(self) -> AIManager:
        """Resolve the AIManager instance from injection, container, or global scope.

        Returns:
            The active AIManager instance.

        Raises:
            SkillExecutionError: If AIManager cannot be resolved.
        """
        if self._ai_manager is not None:
            return self._ai_manager

        # Resolve from ServiceContainer
        container_inst = self.container
        if container_inst is not None:
            if container_inst.exists("ai_manager"):
                return container_inst.resolve("ai_manager")
            if container_inst.exists("ai"):
                return container_inst.resolve("ai")

        # Fallback to global singleton
        try:
            from app.ai.manager import ai_manager as global_ai_mgr

            return global_ai_mgr
        except Exception as exc:
            raise SkillExecutionError(
                f"AIManager could not be resolved from container or global scope: {exc}"
            ) from exc

    @ai_manager.setter
    def ai_manager(self, manager: AIManager) -> None:
        """Explicitly set or override the AIManager instance."""
        with self._lock:
            self._ai_manager = manager

    def bind(
        self,
        config: Optional[Settings] = None,
        logger: Optional[logging.Logger] = None,
        container: Optional[ServiceContainer] = None,
        event_bus: Optional[EventBus] = None,
        ai_manager: Optional[AIManager] = None,
    ) -> None:
        """Bind subsystem dependencies into the skill.

        Args:
            config: Optional Settings instance.
            logger: Optional Logger instance.
            container: Optional ServiceContainer instance.
            event_bus: Optional EventBus instance.
            ai_manager: Optional AIManager instance.
        """
        super().bind(
            config=config,
            logger=logger,
            container=container,
            event_bus=event_bus,
        )
        if ai_manager is not None:
            with self._lock:
                self._ai_manager = ai_manager

    @property
    def execution_count(self) -> int:
        """Return the total number of successful skill executions."""
        with self._lock:
            return self._execution_count

    @property
    def error_count(self) -> int:
        """Return the total number of failed skill executions."""
        with self._lock:
            return self._error_count

    def _extract_query_and_context(self, command: Any) -> tuple[str, str]:
        """Extract user query text and conversation identifier from diverse command types.

        Supports strings, dictionaries, Intent dataclass objects, and generic objects.

        Args:
            command: The command payload to parse.

        Returns:
            A tuple of (query_text, conversation_id).
        """
        if command is None:
            return "", "default"

        # 1. String command
        if isinstance(command, str):
            return command.strip(), "default"

        # 2. Intent object (app.router.intent.Intent)
        if hasattr(command, "raw_command") or hasattr(command, "normalized_command"):
            raw_cmd = getattr(command, "raw_command", "") or ""
            norm_cmd = getattr(command, "normalized_command", "") or ""
            text = str(raw_cmd if raw_cmd.strip() else norm_cmd).strip()
            ctx = getattr(command, "context", {}) or {}
            conv_id = "default"
            if isinstance(ctx, dict):
                conv_id = str(ctx.get("conversation_id", ctx.get("session_id", "default")))
            return text, conv_id

        # 3. Dictionary payload
        if isinstance(command, dict):
            text = str(
                command.get("query")
                or command.get("command")
                or command.get("text")
                or command.get("raw_command", "")
            ).strip()
            conv_id = str(
                command.get("conversation_id")
                or command.get("session_id", "default")
            )
            return text, conv_id

        # 4. Fallback: string conversion
        return str(command).strip(), "default"

    def can_handle(self, command: Any) -> bool:
        """Evaluate whether this skill can handle the given command.

        As the universal conversational AI fallback, returns True for any non-empty
        command or query text.

        Args:
            command: The command text, intent, dictionary, or payload to inspect.

        Returns:
            True if non-empty text was provided, False otherwise.
        """
        if command is None:
            return False
        text, _ = self._extract_query_and_context(command)
        return bool(text and text.strip())

    def execute(self, command: Any) -> Any:
        """Execute the conversational skill synchronously using AIManager.

        Lifecycle:
        1. Parse query and conversation context.
        2. Publish 'ai.skill.started' event on EventBus.
        3. Invoke `ai_manager.generate(query=text)`.
        4. Publish 'ai.skill.completed' event on EventBus.
        5. Return response (or publish 'ai.skill.failed' and raise SkillExecutionError on error).

        Args:
            command: The command string, intent, or dictionary to process.

        Returns:
            The AI generation response (AIResponse or string).

        Raises:
            SkillExecutionError: If command is empty or AI generation encounters an error.
        """
        text, conversation_id = self._extract_query_and_context(command)
        if not text:
            raise SkillExecutionError("Command query cannot be empty for AI skill.")

        self.logger.info(f"Executing AI skill for query: '{text[:60]}...'")

        # 1. Publish ai.skill.started
        self.event_bus.publish(
            "ai.skill.started",
            payload={
                "query": text,
                "skill_name": self.name,
                "conversation_id": conversation_id,
                "timestamp": time.time(),
            },
            source=f"skill.{self.name}",
        )

        start_time = time.perf_counter()
        try:
            mgr = self.ai_manager
            if conversation_id and conversation_id != "default":
                response = mgr.generate(query=text, conversation_id=conversation_id)
            else:
                response = mgr.generate(query=text)

            duration = time.perf_counter() - start_time
            with self._lock:
                self._execution_count += 1

            resp_text = getattr(response, "content", str(response))

            # 2. Publish ai.skill.completed
            self.event_bus.publish(
                "ai.skill.completed",
                payload={
                    "query": text,
                    "response": resp_text,
                    "result": response,
                    "duration": duration,
                    "skill_name": self.name,
                    "conversation_id": conversation_id,
                    "timestamp": time.time(),
                },
                source=f"skill.{self.name}",
            )
            return response

        except Exception as exc:
            duration = time.perf_counter() - start_time
            with self._lock:
                self._error_count += 1

            self.logger.error(
                f"AI skill execution failed for query '{text[:60]}...': {exc}",
                exc_info=True,
            )

            # 3. Publish ai.skill.failed
            self.event_bus.publish(
                "ai.skill.failed",
                payload={
                    "query": text,
                    "error": str(exc),
                    "exception_type": type(exc).__name__,
                    "duration": duration,
                    "skill_name": self.name,
                    "conversation_id": conversation_id,
                    "timestamp": time.time(),
                },
                source=f"skill.{self.name}",
            )
            raise SkillExecutionError(f"AI skill execution failed: {exc}") from exc

    async def execute_async(self, command: Any) -> Any:
        """Execute the conversational skill asynchronously using AIManager.

        Lifecycle:
        1. Parse query and conversation context.
        2. Asynchronously publish 'ai.skill.started' event on EventBus.
        3. Invoke `await ai_manager.generate_async(query=text)`.
        4. Asynchronously publish 'ai.skill.completed' event on EventBus.
        5. Return response (or asynchronously publish 'ai.skill.failed' and raise SkillExecutionError on error).

        Args:
            command: The command string, intent, or dictionary to process.

        Returns:
            The AI generation response (AIResponse or string).

        Raises:
            SkillExecutionError: If command is empty or AI generation encounters an error.
        """
        text, conversation_id = self._extract_query_and_context(command)
        if not text:
            raise SkillExecutionError("Command query cannot be empty for AI skill.")

        self.logger.info(f"Executing async AI skill for query: '{text[:60]}...'")

        # 1. Publish ai.skill.started
        await self.event_bus.publish_async(
            "ai.skill.started",
            payload={
                "query": text,
                "skill_name": self.name,
                "conversation_id": conversation_id,
                "timestamp": time.time(),
            },
            source=f"skill.{self.name}",
        )

        start_time = time.perf_counter()
        try:
            mgr = self.ai_manager
            if conversation_id and conversation_id != "default":
                response = await mgr.generate_async(query=text, conversation_id=conversation_id)
            else:
                response = await mgr.generate_async(query=text)

            duration = time.perf_counter() - start_time
            with self._lock:
                self._execution_count += 1

            resp_text = getattr(response, "content", str(response))

            # 2. Publish ai.skill.completed
            await self.event_bus.publish_async(
                "ai.skill.completed",
                payload={
                    "query": text,
                    "response": resp_text,
                    "result": response,
                    "duration": duration,
                    "skill_name": self.name,
                    "conversation_id": conversation_id,
                    "timestamp": time.time(),
                },
                source=f"skill.{self.name}",
            )
            return response

        except Exception as exc:
            duration = time.perf_counter() - start_time
            with self._lock:
                self._error_count += 1

            self.logger.error(
                f"Async AI skill execution failed for query '{text[:60]}...': {exc}",
                exc_info=True,
            )

            # 3. Publish ai.skill.failed
            await self.event_bus.publish_async(
                "ai.skill.failed",
                payload={
                    "query": text,
                    "error": str(exc),
                    "exception_type": type(exc).__name__,
                    "duration": duration,
                    "skill_name": self.name,
                    "conversation_id": conversation_id,
                    "timestamp": time.time(),
                },
                source=f"skill.{self.name}",
            )
            raise SkillExecutionError(f"Async AI skill execution failed: {exc}") from exc


# ------------------------------------------------------------------------------
# Module-level default singleton instance
# ------------------------------------------------------------------------------
ai_skill: Final[AISkill] = AISkill()

__all__ = [
    "AISkill",
    "DEFAULT_AI_SKILL_DESCRIPTION",
    "DEFAULT_AI_SKILL_NAME",
    "DEFAULT_AI_SKILL_PRIORITY",
    "ai_skill",
]
