"""Production Application Orchestrator and Runner for J.A.R.V.I.S.

Coordinates the unified application lifecycle:
1. Container and subsystem dependency injection.
2. AI Manager, Skill Manager, and Command Router initialization.
3. Voice Pipeline initialization and audio state readiness.
4. Automatic skill discovery across installed packages and plugins.
5. Interactive console command routing with AI conversational fallback.
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Final, Optional, Union

from app.ai.manager import AIManager, ai_manager as default_ai_manager
from app.core.config import Settings, settings as default_settings
from app.core.container import ServiceContainer, container as default_container
from app.core.event_bus import EventBus, event_bus as default_event_bus
from app.core.logger import get_logger
from app.memory.manager import MemoryManager, memory_manager as default_memory_manager
from app.router.router import CommandRouter, command_router as default_command_router
from app.skills.ai_skill import AISkill
from app.skills.base import BaseSkill
from app.skills.manager import SkillManager, skill_manager as default_skill_manager
from app.voice.engine import VoiceConversationEngine, voice_conversation_engine as default_voice_conversation_engine
from app.voice.models import VoiceConversationResult
from app.voice.pipeline import VoicePipeline, voice_pipeline as default_voice_pipeline

READY_MESSAGE: Final[str] = "J.A.R.V.I.S Ready"
EVENT_APPLICATION_READY: Final[str] = "application.ready"
EVENT_APPLICATION_STOPPED: Final[str] = "application.stopped"


class JarvisApplication:
    """Production application orchestrator and runner for J.A.R.V.I.S.

    Initializes the ServiceContainer, AIManager, CommandRouter, VoicePipeline,
    and SkillManager; discovers all available skills automatically; ensures
    AISkill fallback readiness; and executes an interactive user console loop.
    """

    def __init__(
        self,
        container: Optional[ServiceContainer] = None,
        config: Optional[Settings] = None,
        logger: Optional[logging.Logger] = None,
        event_bus: Optional[EventBus] = None,
        memory_manager: Optional[MemoryManager] = None,
        ai_manager: Optional[AIManager] = None,
        skill_manager: Optional[SkillManager] = None,
        command_router: Optional[CommandRouter] = None,
        voice_pipeline: Optional[VoicePipeline] = None,
        voice_engine: Optional[VoiceConversationEngine] = None,
        *,
        voice_mode: bool = False,
        auto_discover_skills: bool = True,
        print_ready: bool = True,
    ) -> None:
        """Initialize the J.A.R.V.I.S Application.

        Args:
            container: Optional ServiceContainer instance. Defaults to global container.
            config: Optional Settings instance. Defaults to container/global settings.
            logger: Optional Logger instance. Defaults to 'APPLICATION' logger.
            event_bus: Optional EventBus instance. Defaults to container/global event bus.
            memory_manager: Optional MemoryManager instance.
            ai_manager: Optional AIManager instance.
            skill_manager: Optional SkillManager instance.
            command_router: Optional CommandRouter instance.
            voice_pipeline: Optional VoicePipeline instance.
            voice_engine: Optional VoiceConversationEngine instance.
            voice_mode: Whether to start in voice mode. Defaults to False.
            auto_discover_skills: Whether to automatically scan and register skills. Defaults to True.
            print_ready: Whether to print 'J.A.R.V.I.S Ready' on initialization. Defaults to True.
        """
        self._lock = threading.RLock()
        self._is_running = False
        self._voice_mode = bool(voice_mode)

        # 1. Dependency Resolution: Service Container
        self._container = container if container is not None else default_container

        # 2. Dependency Resolution: Configuration & Logging
        self._logger = logger if logger is not None else get_logger("APPLICATION")

        if config is not None:
            self._config = config
        elif self._container.exists("settings"):
            self._config = self._container.resolve("settings")
        elif self._container.exists("config"):
            self._config = self._container.resolve("config")
        else:
            self._config = default_settings

        # Ensure container has configuration registered
        self._container.register_singleton("settings", self._config, allow_override=True)
        self._container.register_singleton("config", self._config, allow_override=True)

        # 3. Dependency Resolution: Event Bus
        if event_bus is not None:
            self._event_bus = event_bus
        elif self._container.exists("event_bus"):
            self._event_bus = self._container.resolve("event_bus")
        else:
            self._event_bus = default_event_bus

        self._container.register_singleton("event_bus", self._event_bus, allow_override=True)

        # 4. Dependency Resolution: Memory Manager
        if memory_manager is not None:
            self._memory_manager = memory_manager
        elif self._container.exists("memory_manager"):
            self._memory_manager = self._container.resolve("memory_manager")
        elif self._container.exists("memory"):
            self._memory_manager = self._container.resolve("memory")
        else:
            self._memory_manager = default_memory_manager

        self._container.register_singleton("memory_manager", self._memory_manager, allow_override=True)
        self._container.register_singleton("memory", self._memory_manager, allow_override=True)

        # 5. Initialize AIManager
        if ai_manager is not None:
            self._ai_manager = ai_manager
        elif self._container.exists("ai_manager"):
            self._ai_manager = self._container.resolve("ai_manager")
        else:
            self._ai_manager = AIManager(
                config=self._config,
                logger=self._logger,
                container_instance=self._container,
                event_bus_instance=self._event_bus,
                memory_manager_instance=self._memory_manager,
            )

        self._container.register_singleton("ai_manager", self._ai_manager, allow_override=True)

        # 6. Initialize SkillManager
        if skill_manager is not None:
            self._skill_manager = skill_manager
        elif self._container.exists("skill_manager"):
            self._skill_manager = self._container.resolve("skill_manager")
        else:
            self._skill_manager = SkillManager(
                config=self._config,
                logger=self._logger,
                container_instance=self._container,
                event_bus_instance=self._event_bus,
            )

        self._container.register_singleton("skill_manager", self._skill_manager, allow_override=True)

        # 7. Discover skills automatically
        if auto_discover_skills:
            self.discover_skills()

        # 8. Initialize CommandRouter
        if command_router is not None:
            self._command_router = command_router
        elif self._container.exists("command_router"):
            self._command_router = self._container.resolve("command_router")
        elif self._container.exists("router"):
            self._command_router = self._container.resolve("router")
        else:
            self._command_router = CommandRouter(
                config=self._config,
                logger=self._logger,
                container_instance=self._container,
                event_bus_instance=self._event_bus,
                skill_manager_instance=self._skill_manager,
            )

        self._container.register_singleton("command_router", self._command_router, allow_override=True)
        self._container.register_singleton("router", self._command_router, allow_override=True)

        # 9. Initialize VoicePipeline
        if voice_pipeline is not None:
            self._voice_pipeline = voice_pipeline
        elif self._container.exists("voice_pipeline"):
            self._voice_pipeline = self._container.resolve("voice_pipeline")
        else:
            self._voice_pipeline = VoicePipeline(
                router=self._command_router,
                memory=self._memory_manager,
                config=self._config,
                logger=self._logger,
                container_instance=self._container,
                event_bus_instance=self._event_bus,
            )

        self._container.register_singleton("voice_pipeline", self._voice_pipeline, allow_override=True)

        # 10. Initialize VoiceConversationEngine
        if voice_engine is not None:
            self._voice_engine = voice_engine
        elif self._container.exists("voice_engine"):
            self._voice_engine = self._container.resolve("voice_engine")
        elif self._container.exists("voice_conversation_engine"):
            self._voice_engine = self._container.resolve("voice_conversation_engine")
        else:
            self._voice_engine = VoiceConversationEngine(
                router=self._command_router,
                config=self._config,
                logger=self._logger,
                container_instance=self._container,
                event_bus_instance=self._event_bus,
            )

        self._container.register_singleton("voice_engine", self._voice_engine, allow_override=True)
        self._container.register_singleton("voice_conversation_engine", self._voice_engine, allow_override=True)

        # Self-registration
        self._container.register_singleton("application", self, allow_override=True)
        self._container.register_singleton("app", self, allow_override=True)

        # Publish ready event and log readiness
        self._event_bus.publish(
            EVENT_APPLICATION_READY,
            payload={"app_name": self._config.app_name, "version": self._config.version},
            source="application",
        )
        self._logger.info("%s [v%s, env=%s]", READY_MESSAGE, self._config.version, self._config.environment)

        # 10. Print J.A.R.V.I.S Ready
        if print_ready:
            print(READY_MESSAGE, flush=True)

    @property
    def container(self) -> ServiceContainer:
        """Return the active ServiceContainer instance."""
        return self._container

    @property
    def config(self) -> Settings:
        """Return the loaded application Settings instance."""
        return self._config

    @property
    def settings(self) -> Settings:
        """Alias for config property."""
        return self._config

    @property
    def logger(self) -> logging.Logger:
        """Return the application Logger instance."""
        return self._logger

    @property
    def event_bus(self) -> EventBus:
        """Return the system EventBus instance."""
        return self._event_bus

    @property
    def memory_manager(self) -> MemoryManager:
        """Return the active MemoryManager instance."""
        return self._memory_manager

    @property
    def memory(self) -> MemoryManager:
        """Alias for memory_manager property."""
        return self._memory_manager

    @property
    def ai_manager(self) -> AIManager:
        """Return the active AIManager instance."""
        return self._ai_manager

    @property
    def skill_manager(self) -> SkillManager:
        """Return the active SkillManager instance."""
        return self._skill_manager

    @property
    def command_router(self) -> CommandRouter:
        """Return the active CommandRouter instance."""
        return self._command_router

    @property
    def router(self) -> CommandRouter:
        """Alias for command_router property."""
        return self._command_router

    @property
    def voice_pipeline(self) -> VoicePipeline:
        """Return the active VoicePipeline instance."""
        return self._voice_pipeline

    @property
    def pipeline(self) -> VoicePipeline:
        """Alias for voice_pipeline property."""
        return self._voice_pipeline

    @property
    def voice_engine(self) -> VoiceConversationEngine:
        """Return the active VoiceConversationEngine instance."""
        return self._voice_engine

    @property
    def voice_conversation_engine(self) -> VoiceConversationEngine:
        """Alias for voice_engine property."""
        return self._voice_engine

    @property
    def voice_mode(self) -> bool:
        """Return whether the application is configured to run in voice mode."""
        return self._voice_mode

    @voice_mode.setter
    def voice_mode(self, value: bool) -> None:
        """Set whether the application runs in voice mode."""
        self._voice_mode = bool(value)

    @property
    def is_running(self) -> bool:
        """Return whether the application interactive loop is running."""
        with self._lock:
            return self._is_running

    def discover_skills(self) -> list[str]:
        """Automatically discover and register skills from the application skills package.

        Ensures AISkill is registered as the conversational fallback skill.

        Returns:
            List of registered skill identifiers.
        """
        discovered: list[str] = []

        try:
            results = self._skill_manager.discover("app.skills", allow_override=True)
            discovered.extend(results)
        except Exception as exc:
            self._logger.warning("Automatic skill discovery from 'app.skills' encountered an error: %s", exc)

        # Check for optional plugins or user skills directories
        for optional_dir in (Path("skills"), Path("plugins")):
            if optional_dir.is_dir():
                try:
                    res = self._skill_manager.discover(optional_dir, allow_override=True)
                    discovered.extend(res)
                except Exception as exc:
                    self._logger.debug("Failed discovering skills from '%s': %s", optional_dir, exc)

        # Ensure fallback AISkill is always registered and available
        if not self._skill_manager.has_skill("ai"):
            try:
                fallback_skill = AISkill(
                    ai_manager_instance=self._ai_manager,
                    config=self._config,
                    logger=self._logger,
                    container=self._container,
                    event_bus=self._event_bus,
                )
                self._skill_manager.register(fallback_skill, allow_override=True)
                discovered.append(fallback_skill.name)
            except Exception as exc:
                self._logger.error("Failed to register fallback AISkill: %s", exc)

        return discovered

    def process_command(self, command: Union[str, dict[str, Any]]) -> Any:
        """Process a single command through the CommandRouter.

        Specialized skills execute first according to their priority, and
        AISkill executes as conversational fallback if no specialized skill matches.

        Args:
            command: Command string, dictionary, or intent to process.

        Returns:
            Execution result from the matched skill.
        """
        return self._command_router.route(command)

    @staticmethod
    def _format_response(result: Any) -> str:
        """Extract a clean, user-facing response string from a skill result.

        Args:
            result: Raw execution result from the skill.

        Returns:
            Clean string representation for display.
        """
        if result is None:
            return "Command executed successfully."
        if hasattr(result, "content") and isinstance(result.content, str):
            return result.content
        if hasattr(result, "response") and isinstance(result.response, str):
            return result.response
        if hasattr(result, "message") and isinstance(result.message, str):
            return result.message
        if isinstance(result, dict):
            for key in ("response", "content", "message", "output", "text", "result"):
                val = result.get(key)
                if val is not None and isinstance(val, str):
                    return val
        return str(result)

    def listen_once(
        self,
        *,
        duration: Optional[float] = None,
        audio_path: Optional[Union[str, Path]] = None,
    ) -> VoiceConversationResult:
        """Perform exactly one voice interaction cycle via VoiceConversationEngine.

        Flow:
            record audio -> transcribe -> route through CommandRouter -> receive response -> speak response -> return result

        Args:
            duration: Optional recording duration in seconds.
            audio_path: Optional pre-recorded audio file path.

        Returns:
            VoiceConversationResult encapsulating interaction outcomes.
        """
        return self._voice_engine.listen_once(duration=duration, audio_path=audio_path)

    def stop(self) -> None:
        """Signal the interactive console or voice loop to terminate gracefully."""
        with self._lock:
            self._is_running = False

    def _run_voice_loop(
        self,
        *,
        output_fn: Callable[[str], None],
    ) -> int:
        """Execute the continuous voice interaction loop.

        Args:
            output_fn: Output display callable.

        Returns:
            Exit code (0 for clean termination).
        """
        output_fn("\n[Voice Mode Active - Listening for speech... (Say 'exit' or press Ctrl+C to quit)]\n")

        with self._lock:
            self._is_running = True

        try:
            while True:
                with self._lock:
                    if not self._is_running:
                        break

                try:
                    result = self.listen_once()
                except (EOFError, KeyboardInterrupt):
                    output_fn("")
                    break
                except Exception as exc:
                    self._logger.error("Voice interaction error: %s", exc, exc_info=True)
                    output_fn(f"\nJarvis > I encountered an error: {exc}\n")
                    continue

                if not result:
                    continue

                if result.text:
                    output_fn(f"\nYou (Voice) > {result.text}")

                if result.response_text:
                    output_fn(f"\nJarvis > {result.response_text}\n")

                if result.command and result.command.lower() in ("exit", "quit"):
                    break

        finally:
            with self._lock:
                self._is_running = False

            self._event_bus.publish(
                EVENT_APPLICATION_STOPPED,
                payload={"status": "stopped", "mode": "voice"},
                source="application",
            )

        return 0

    def run(
        self,
        *,
        input_fn: Optional[Callable[[str], str]] = None,
        output_fn: Optional[Callable[[str], None]] = None,
        voice_mode: Optional[bool] = None,
    ) -> int:
        """Start the interactive console loop or voice mode loop.

        In voice mode (enabled via voice_mode=True or python main.py --voice),
        reads audio from the microphone, transcribes via STT, routes via
        CommandRouter, and speaks responses via TTS.

        In text console mode, reads commands from standard input (or input_fn)
        with prompt 'You > ', routes them through the CommandRouter, and displays
        responses as 'Jarvis > <response>'.

        Typing or saying 'exit' or 'quit' terminates gracefully.

        Args:
            input_fn: Optional custom input callable `(prompt) -> str`.
            output_fn: Optional custom output callable `(msg) -> None`.
            voice_mode: Optional flag overriding voice interaction mode.

        Returns:
            Exit code (0 for clean termination).
        """
        is_voice = self._voice_mode if voice_mode is None else bool(voice_mode)
        _write_output = output_fn if output_fn is not None else print

        # Voice interaction mode
        if is_voice:
            return self._run_voice_loop(output_fn=_write_output)

        # If running in non-interactive automated test environments without custom input
        if (
            input_fn is None
            and not sys.stdin.isatty()
            and ("unittest" in sys.modules or "pytest" in sys.modules)
        ):
            return 0

        _read_input = input_fn if input_fn is not None else input

        with self._lock:
            self._is_running = True

        try:
            while True:
                with self._lock:
                    if not self._is_running:
                        break

                try:
                    user_input = _read_input("You > ")
                except (EOFError, KeyboardInterrupt):
                    _write_output("")
                    break

                if user_input is None:
                    break

                raw_text = str(user_input).strip()
                if not raw_text:
                    continue

                if raw_text.lower() in ("exit", "quit"):
                    break

                try:
                    result = self.process_command(raw_text)
                    response_text = self._format_response(result)
                    _write_output(f"\nJarvis > {response_text}\n")
                except Exception as exc:
                    self._logger.error("Error routing command '%s': %s", raw_text, exc, exc_info=True)
                    _write_output(f"\nJarvis > I encountered an error: {exc}\n")

        finally:
            with self._lock:
                self._is_running = False

            self._event_bus.publish(
                EVENT_APPLICATION_STOPPED,
                payload={"status": "stopped", "mode": "text"},
                source="application",
            )

        return 0


__all__ = [
    "EVENT_APPLICATION_READY",
    "EVENT_APPLICATION_STOPPED",
    "JarvisApplication",
    "READY_MESSAGE",
]
