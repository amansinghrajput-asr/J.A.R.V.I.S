"""AI Manager Subsystem for J.A.R.V.I.S.

Coordinates LLM generation, prompt construction, memory context retrieval,
service container dependency injection, and event lifecycle dispatching.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Final, Optional

from app.ai.models import (
    AIError,
    AIResponse,
    GenerationConfig,
)
from app.ai.prompt import PromptBuilder
from app.ai.provider import GeminiProvider
from app.core.config import Settings, settings
from app.core.container import ServiceContainer, container
from app.core.event_bus import EventBus, event_bus
from app.core.logger import get_logger
from app.memory.manager import MemoryManager, memory_manager


class AIManager:
    """Central coordinator for cognition, reasoning, and conversational intelligence.

    Integrates Google Gemini model providers with the short-term MemoryManager,
    system persona PromptBuilder, central EventBus, and ServiceContainer.
    """

    def __init__(
        self,
        config: Optional[Settings] = None,
        logger: Optional[logging.Logger] = None,
        container_instance: Optional[ServiceContainer] = None,
        event_bus_instance: Optional[EventBus] = None,
        memory_manager_instance: Optional[MemoryManager] = None,
        provider_instance: Optional[GeminiProvider] = None,
        prompt_builder_instance: Optional[PromptBuilder] = None,
        *,
        auto_register_in_container: bool = True,
    ) -> None:
        """Initialize the AIManager.

        Args:
            config: Optional Settings instance.
            logger: Optional Logger instance.
            container_instance: Optional ServiceContainer instance.
            event_bus_instance: Optional EventBus instance.
            memory_manager_instance: Optional MemoryManager instance.
            provider_instance: Optional GeminiProvider instance.
            prompt_builder_instance: Optional PromptBuilder instance.
            auto_register_in_container: If True, self-registers into ServiceContainer.
        """
        self._lock = threading.RLock()

        # 1. Dependency resolution: Service Container
        self._container = container_instance if container_instance is not None else container

        # 2. Dependency resolution: Logger
        self._logger = logger if logger is not None else get_logger("AI_MANAGER")

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

        # 5. Dependency resolution: Memory Manager
        if memory_manager_instance is not None:
            self._memory_manager: Optional[MemoryManager] = memory_manager_instance
        elif self._container.exists("memory_manager"):
            self._memory_manager = self._container.resolve("memory_manager")
        elif self._container.exists("memory"):
            self._memory_manager = self._container.resolve("memory")
        else:
            self._memory_manager = memory_manager

        # 6. Dependency resolution: Prompt Builder
        self._prompt_builder = (
            prompt_builder_instance if prompt_builder_instance is not None else PromptBuilder()
        )

        # 7. Dependency resolution: Gemini Provider
        if provider_instance is not None:
            self._provider = provider_instance
        else:
            self._provider = GeminiProvider(logger=self._logger)

        # 8. Self-registration in Service Container
        if auto_register_in_container:
            try:
                self._container.register_singleton("ai_manager", self, allow_override=True)
                self._container.register_singleton("ai", self, allow_override=True)
                self._logger.debug("Registered 'ai_manager' and 'ai' into Service Container.")
            except Exception as exc:
                self._logger.warning(f"Could not register AIManager into container: {exc}")

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------

    @property
    def config(self) -> Settings:
        """Return the active settings configuration."""
        return self._config

    @property
    def logger(self) -> logging.Logger:
        """Return the active logger."""
        return self._logger

    @property
    def container(self) -> ServiceContainer:
        """Return the service container."""
        return self._container

    @property
    def event_bus(self) -> EventBus:
        """Return the event bus."""
        return self._event_bus

    @property
    def memory_manager(self) -> Optional[MemoryManager]:
        """Return the active memory manager."""
        return self._memory_manager

    @property
    def provider(self) -> GeminiProvider:
        """Return the underlying Gemini provider."""
        return self._provider

    @property
    def prompt_builder(self) -> PromptBuilder:
        """Return the active prompt builder."""
        return self._prompt_builder

    # --------------------------------------------------------------------------
    # Core Operations
    # --------------------------------------------------------------------------

    def generate(
        self,
        query: str,
        *,
        conversation_id: str = "default",
        use_history: bool = True,
        system_prompt: Optional[str] = None,
        extra_context: Optional[dict[str, Any] | str] = None,
        config: Optional[GenerationConfig] = None,
        record_memory: bool = True,
        history_limit: Optional[int] = None,
    ) -> AIResponse:
        """Execute synchronous conversational generation with full system integration.

        Args:
            query: User's command or question.
            conversation_id: Conversation tracking ID.
            use_history: If True, retrieves recent conversation history from MemoryManager.
            system_prompt: Optional custom system prompt override.
            extra_context: Optional contextual dictionary or string.
            config: Optional sampling parameters (temperature, tokens, etc.).
            record_memory: If True, records user query and assistant response in MemoryManager.
            history_limit: Maximum historical turns to provide to model. If None, uses settings.ai.history_limit.

        Returns:
            The structured AIResponse.

        Raises:
            AIError: If generation fails after all provider retries.
        """
        resolved_limit = (
            history_limit
            if history_limit is not None
            else getattr(self._config.ai, "history_limit", 20)
        )

        with self._lock:
            history = None
            if use_history and self._memory_manager is not None:
                history = self._memory_manager.get_recent(
                    limit=resolved_limit,
                    conversation_id=conversation_id,
                )

            # Record user query in memory
            if record_memory and self._memory_manager is not None:
                self._memory_manager.add(
                    content=query,
                    role="user",
                    conversation_id=conversation_id,
                    source="user",
                )

            # Build provider payload
            payload = self._prompt_builder.build_payload(
                query=query,
                history=history,
                system_prompt=system_prompt,
                extra_context=extra_context,
            )

        # Emit ai.started event
        self._event_bus.publish(
            "ai.started",
            payload={
                "query": query,
                "model": self._provider.model,
                "conversation_id": conversation_id,
            },
            source="ai_manager",
        )

        try:
            response = self._provider.generate(payload, config=config)

            # Record assistant response in memory
            if record_memory and self._memory_manager is not None:
                self._memory_manager.add(
                    content=response.content,
                    role="assistant",
                    conversation_id=conversation_id,
                    source="ai",
                    metadata={
                        "model": response.model,
                        "tokens": response.total_tokens,
                        "duration": response.duration,
                    },
                )

            # Emit ai.completed event
            self._event_bus.publish(
                "ai.completed",
                payload={
                    "query": query,
                    "response": response.content,
                    "model": response.model,
                    "duration": response.duration,
                    "tokens": response.total_tokens,
                    "conversation_id": conversation_id,
                },
                source="ai_manager",
            )

            # Emit ai.generated event
            self._event_bus.publish(
                "ai.generated",
                payload={
                    "query": query,
                    "response": response.content,
                    "model": response.model,
                    "duration": response.duration,
                    "tokens": response.total_tokens,
                    "conversation_id": conversation_id,
                },
                source="ai_manager",
            )
            return response

        except Exception as exc:
            self._logger.exception(f"AIManager generation failed for query '{query[:50]}...': {exc}")
            self._event_bus.publish(
                "ai.failed",
                payload={
                    "query": query,
                    "error": str(exc),
                    "exception_type": type(exc).__name__,
                    "conversation_id": conversation_id,
                },
                source="ai_manager",
            )
            if isinstance(exc, AIError):
                raise
            raise AIError(f"Generation failed: {exc}") from exc

    async def generate_async(
        self,
        query: str,
        *,
        conversation_id: str = "default",
        use_history: bool = True,
        system_prompt: Optional[str] = None,
        extra_context: Optional[dict[str, Any] | str] = None,
        config: Optional[GenerationConfig] = None,
        record_memory: bool = True,
        history_limit: Optional[int] = None,
    ) -> AIResponse:
        """Execute asynchronous conversational generation with full system integration.

        Args:
            query: User's command or question.
            conversation_id: Conversation tracking ID.
            use_history: If True, retrieves recent conversation history from MemoryManager.
            system_prompt: Optional custom system prompt override.
            extra_context: Optional contextual dictionary or string.
            config: Optional sampling parameters.
            record_memory: If True, records user query and assistant response in MemoryManager.
            history_limit: Maximum historical turns to provide to model. If None, uses settings.ai.history_limit.

        Returns:
            The structured AIResponse.

        Raises:
            AIError: If generation fails after all provider retries.
        """
        resolved_limit = (
            history_limit
            if history_limit is not None
            else getattr(self._config.ai, "history_limit", 20)
        )

        with self._lock:
            history = None
            if use_history and self._memory_manager is not None:
                history = self._memory_manager.get_recent(
                    limit=resolved_limit,
                    conversation_id=conversation_id,
                )

            # Record user query in memory
            if record_memory and self._memory_manager is not None:
                self._memory_manager.add(
                    content=query,
                    role="user",
                    conversation_id=conversation_id,
                    source="user",
                )

            payload = self._prompt_builder.build_payload(
                query=query,
                history=history,
                system_prompt=system_prompt,
                extra_context=extra_context,
            )

        # Emit ai.started event asynchronously
        await self._event_bus.publish_async(
            "ai.started",
            payload={
                "query": query,
                "model": self._provider.model,
                "conversation_id": conversation_id,
            },
            source="ai_manager",
        )

        try:
            response = await self._provider.generate_async(payload, config=config)

            # Record assistant response in memory
            if record_memory and self._memory_manager is not None:
                self._memory_manager.add(
                    content=response.content,
                    role="assistant",
                    conversation_id=conversation_id,
                    source="ai",
                    metadata={
                        "model": response.model,
                        "tokens": response.total_tokens,
                        "duration": response.duration,
                    },
                )

            # Emit ai.completed event
            await self._event_bus.publish_async(
                "ai.completed",
                payload={
                    "query": query,
                    "response": response.content,
                    "model": response.model,
                    "duration": response.duration,
                    "tokens": response.total_tokens,
                    "conversation_id": conversation_id,
                },
                source="ai_manager",
            )

            # Emit ai.generated event
            await self._event_bus.publish_async(
                "ai.generated",
                payload={
                    "query": query,
                    "response": response.content,
                    "model": response.model,
                    "duration": response.duration,
                    "tokens": response.total_tokens,
                    "conversation_id": conversation_id,
                },
                source="ai_manager",
            )
            return response

        except Exception as exc:
            self._logger.exception(f"AIManager async generation failed: {exc}")
            await self._event_bus.publish_async(
                "ai.failed",
                payload={
                    "query": query,
                    "error": str(exc),
                    "exception_type": type(exc).__name__,
                    "conversation_id": conversation_id,
                },
                source="ai_manager",
            )
            if isinstance(exc, AIError):
                raise
            raise AIError(f"Async generation failed: {exc}") from exc

    def stream_generate(
        self,
        query: str,
        *,
        conversation_id: str = "default",
        **kwargs: Any,
    ) -> Any:
        """Stream conversational generation synchronously (future capability).

        Raises:
            NotImplementedError: Streaming generation is not yet implemented.
        """
        raise NotImplementedError("Streaming generation is not yet implemented.")

    async def stream_generate_async(
        self,
        query: str,
        *,
        conversation_id: str = "default",
        **kwargs: Any,
    ) -> Any:
        """Stream conversational generation asynchronously (future capability).

        Raises:
            NotImplementedError: Asynchronous streaming generation is not yet implemented.
        """
        raise NotImplementedError("Asynchronous streaming generation is not yet implemented.")

    def chat(self, query: str, **kwargs: Any) -> str:
        """Convenience method returning the textual response directly.

        Args:
            query: User's command or question.
            **kwargs: Extra parameters passed to generate().

        Returns:
            Generated response content string.
        """
        response = self.generate(query, **kwargs)
        return response.content

    async def chat_async(self, query: str, **kwargs: Any) -> str:
        """Convenience async method returning the textual response directly.

        Args:
            query: User's command or question.
            **kwargs: Extra parameters passed to generate_async().

        Returns:
            Generated response content string.
        """
        response = await self.generate_async(query, **kwargs)
        return response.content


# Global default AIManager singleton
ai_manager: Final[AIManager] = AIManager()
