"""ProviderRouter for J.A.R.V.I.S AI Subsystem.

Responsible for maintaining the provider registry, registering concrete
AI providers (Gemini, Ollama, and future extensions), and resolving the
active provider instance based on configuration or explicit overrides.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Callable, Final, Optional

from app.ai.models import AIConfigError, AIError
from app.ai.ollama_provider import OllamaProvider
from app.ai.provider import GeminiProvider
from app.core.config import Settings, settings
from app.core.container import ServiceContainer, container
from app.core.logger import get_logger

# Type alias for a provider factory callable
ProviderFactory = Callable[[Settings, logging.Logger], Any]


def _gemini_factory(cfg: Settings, logger: logging.Logger) -> GeminiProvider:
    """Factory to instantiate a configured GeminiProvider."""
    ai_cfg = getattr(cfg, "ai", None)
    return GeminiProvider(
        api_key=getattr(ai_cfg, "gemini_api_key", None),
        model=getattr(ai_cfg, "gemini_model", None) or getattr(ai_cfg, "model", None),
        timeout=getattr(ai_cfg, "timeout", None),
        max_retries=getattr(ai_cfg, "max_retries", None),
        logger=logger,
    )


def _ollama_factory(cfg: Settings, logger: logging.Logger) -> OllamaProvider:
    """Factory to instantiate a configured OllamaProvider."""
    ai_cfg = getattr(cfg, "ai", None)
    model = getattr(ai_cfg, "ollama_model", None) or os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    host = getattr(ai_cfg, "ollama_host", None) or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    return OllamaProvider(model=model, host=host)


class ProviderRouter:
    """Central registry and routing engine for AI LLM providers.

    Decouples provider selection and instantiation from cognitive orchestration,
    enabling declarative provider selection via configuration and dynamic runtime
    registration of new providers.
    """

    def __init__(
        self,
        config: Optional[Settings] = None,
        logger: Optional[logging.Logger] = None,
        container_instance: Optional[ServiceContainer] = None,
        *,
        auto_register_in_container: bool = True,
    ) -> None:
        """Initialize the ProviderRouter.

        Args:
            config: Optional Settings instance.
            logger: Optional Logger instance.
            container_instance: Optional ServiceContainer instance.
            auto_register_in_container: If True, self-registers into ServiceContainer.
        """
        self._lock = threading.RLock()

        # 1. Dependency resolution: Service Container
        self._container = container_instance if container_instance is not None else container

        # 2. Dependency resolution: Logger
        self._logger = logger if logger is not None else get_logger("AI.PROVIDER_ROUTER")

        # 3. Dependency resolution: Configuration
        if config is not None:
            self._config = config
        elif self._container.exists("settings"):
            self._config = self._container.resolve("settings")
        elif self._container.exists("config"):
            self._config = self._container.resolve("config")
        else:
            self._config = settings

        # 4. Registry and active override state
        self._registry: dict[str, ProviderFactory] = {}
        self._active_override: Optional[str] = None

        # 5. Register built-in providers
        self.register_provider("gemini", _gemini_factory)
        self.register_provider("ollama", _ollama_factory)

        # 6. Self-registration in Service Container
        if auto_register_in_container:
            try:
                self._container.register_singleton("provider_router", self, allow_override=True)
                self._container.register_singleton("ai_provider_router", self, allow_override=True)
                self._logger.debug("Registered 'provider_router' into Service Container.")
            except Exception as exc:
                self._logger.warning(f"Could not register ProviderRouter into container: {exc}")

    def register_provider(self, name: str, factory: ProviderFactory) -> None:
        """Register a provider factory for dynamic provider selection.

        Args:
            name: Case-insensitive provider identifier (e.g., 'gemini', 'ollama', 'openai').
            factory: Callable accepting (Settings, Logger) and returning a provider instance.

        Raises:
            ValueError: If name is empty or factory is not callable.
        """
        clean_name = (name or "").strip().lower()
        if not clean_name:
            raise ValueError("Provider name must not be empty.")
        if not callable(factory):
            raise ValueError(f"Provider factory for '{clean_name}' must be callable.")

        with self._lock:
            self._registry[clean_name] = factory
            self._logger.debug(f"Registered AI provider factory for '{clean_name}'.")

    def get_registered_providers(self) -> list[str]:
        """Return a sorted list of all registered provider names."""
        with self._lock:
            return sorted(list(self._registry.keys()))

    def set_active_provider(self, name: str) -> None:
        """Set an active provider override at runtime.

        Args:
            name: Case-insensitive provider identifier.

        Raises:
            AIConfigError: If the provider is not registered.
        """
        clean_name = (name or "").strip().lower()
        with self._lock:
            if clean_name not in self._registry:
                raise AIConfigError(
                    f"Cannot set active provider to '{clean_name}': provider is not registered. "
                    f"Available providers: {self.get_registered_providers()}"
                )
            self._active_override = clean_name
            self._logger.info(f"Active AI provider override set to '{clean_name}'.")

    @property
    def active_provider_name(self) -> str:
        """Return the identifier of the active provider."""
        with self._lock:
            if self._active_override is not None:
                return self._active_override

            ai_config = getattr(self._config, "ai", None)
            configured = (
                getattr(ai_config, "provider", None)
                or getattr(self._config, "ai_provider", None)
                or os.getenv("AI_PROVIDER", "gemini")
            )
            return str(configured).strip().lower()

    def resolve(
        self,
        provider_name: Optional[str] = None,
        *,
        instance_override: Optional[Any] = None,
    ) -> Any:
        """Resolve and return an initialized provider instance.

        Resolution precedence:
        1. instance_override (if provided)
        2. provider_name parameter (if provided)
        3. active_provider_name (from active override or Settings.ai.provider)
        4. Fallback ("gemini")

        Args:
            provider_name: Optional explicit provider identifier to resolve.
            instance_override: Optional pre-constructed provider instance.

        Returns:
            An instantiated and configured provider instance.

        Raises:
            AIConfigError: If the target provider is not registered in the router.
        """
        if instance_override is not None:
            return instance_override

        target_name = (provider_name or self.active_provider_name).strip().lower()

        with self._lock:
            factory = self._registry.get(target_name)
            if factory is None:
                raise AIConfigError(
                    f"Unsupported AI provider: '{target_name}'. "
                    f"Registered providers: {self.get_registered_providers()}"
                )

        try:
            return factory(self._config, self._logger)
        except Exception as exc:
            self._logger.exception(f"Failed to instantiate AI provider '{target_name}': {exc}")
            if isinstance(exc, AIError):
                raise
            raise AIConfigError(f"Failed to instantiate provider '{target_name}': {exc}") from exc


# ------------------------------------------------------------------------------
# Default Singleton Instance Export
# ------------------------------------------------------------------------------
provider_router: Final[ProviderRouter] = ProviderRouter()
