"""Text-based Wake Word Detector Subsystem for J.A.R.V.I.S.

Provides fast, case-insensitive, punctuation-stripped, whitespace-normalized
wake word detection integrated with Config, Logger, Event Bus, and Service Container.
"""

from __future__ import annotations

import logging
import re
import string
import threading
import time
from typing import Any, Final, Optional, Sequence

from app.core.config import Settings, settings
from app.core.constants import DEFAULT_WAKE_PHRASES
from app.core.container import ServiceContainer, container
from app.core.event_bus import EventBus, event_bus
from app.core.logger import get_logger
from app.wakeword.models import InvalidInputError, WakeWordResult

# Event Topic Constants
EVENT_WAKEWORD_DETECTED: Final[str] = "wakeword.detected"
EVENT_WAKEWORD_IGNORED: Final[str] = "wakeword.ignored"
EVENT_SOURCE_WAKEWORD: Final[str] = "wakeword"

# Pattern to collapse punctuation and whitespace clusters into single spaces
_PUNCTUATION_AND_SPACE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[\s" + re.escape(string.punctuation) + r"“”‘’—–…«»]+",
    re.UNICODE,
)


def normalize_text(text: Optional[str]) -> str:
    """Normalize input text for wake word evaluation.

    Performs case-folding (lowercase), strips all punctuation characters
    (commas, periods, exclamation marks, question marks, quotes, hyphens, etc.),
    collapses multiple consecutive whitespace characters into a single space,
    and trims leading and trailing whitespace.

    Args:
        text: Raw text string to normalize. If None, treated as empty string.

    Returns:
        Cleaned, lowercased, punctuation-free string.

    Examples:
        >>> normalize_text("  Hey,   Jarvis...  ")
        'hey jarvis'
        >>> normalize_text("Activate Jarvis!")
        'activate jarvis'
        >>> normalize_text("Jarvish, Activate???")
        'jarvish activate'
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # 1. Lowercase
    cleaned = text.lower()
    # 2. Replace punctuation characters and whitespace clusters with a single space
    cleaned = _PUNCTUATION_AND_SPACE_PATTERN.sub(" ", cleaned)
    # 3. Strip leading and trailing whitespace
    return cleaned.strip()


class WakeWordDetector:
    """Thread-safe, async-ready text-based Wake Word Detector for J.A.R.V.I.S.

    Evaluates text input for configured wake phrases, normalizing casing,
    punctuation, and whitespace before applying word-boundary matching.
    Emits lifecycle events on the Event Bus upon detection or rejection.
    """

    def __init__(
        self,
        wake_phrases: Optional[Sequence[str]] = None,
        config: Optional[Settings] = None,
        logger: Optional[logging.Logger] = None,
        container_instance: Optional[ServiceContainer] = None,
        event_bus_instance: Optional[EventBus] = None,
        *,
        auto_register_in_container: bool = True,
    ) -> None:
        """Initialize the WakeWordDetector.

        Args:
            wake_phrases: Optional custom sequence of wake phrases. If None,
                resolves from config or uses system default phrases.
            config: Optional Settings instance. If None, resolves from container
                or global settings.
            logger: Optional Logger instance. If None, creates 'WAKEWORD' logger.
            container_instance: Optional ServiceContainer. If None, uses global container.
            event_bus_instance: Optional EventBus. If None, resolves from container
                or global event bus.
            auto_register_in_container: If True, registers this detector into container.
        """
        self._lock = threading.RLock()

        # 1. Dependency resolution: Service Container
        self._container = (
            container_instance if container_instance is not None else container
        )

        # 2. Dependency resolution: Logger
        self._logger = logger if logger is not None else get_logger("WAKEWORD")

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

        # 5. Initialize wake phrases
        self._wake_phrases: list[str] = []
        self._compiled_patterns: list[tuple[str, str, re.Pattern[str]]] = []

        if wake_phrases is not None:
            self.set_wake_phrases(wake_phrases)
        elif hasattr(self._config, "wake_phrases") and self._config.wake_phrases:
            self.set_wake_phrases(self._config.wake_phrases)
        else:
            self.set_wake_phrases(DEFAULT_WAKE_PHRASES)

        # 6. Auto-registration in Service Container
        if auto_register_in_container:
            try:
                self._container.register_singleton(
                    "wakeword_detector", self, allow_override=True
                )
                self._container.register_singleton(
                    "wake_word_detector", self, allow_override=True
                )
                self._container.register_singleton(
                    "wakeword", self, allow_override=True
                )
                self._logger.debug(
                    "Registered 'wakeword_detector' singleton into Service Container."
                )
            except Exception as exc:
                self._logger.warning(
                    f"Could not register WakeWordDetector into container: {exc}"
                )

    # --------------------------------------------------------------------------
    # Dependency Access Properties
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
    def event_bus(self) -> EventBus:
        """Retrieve the active event bus instance."""
        return self._event_bus

    @property
    def container(self) -> ServiceContainer:
        """Retrieve the active service container instance."""
        return self._container

    @property
    def wake_phrases(self) -> tuple[str, ...]:
        """Return a copy of the currently configured wake phrases."""
        with self._lock:
            return tuple(self._wake_phrases)

    @property
    def normalized_wake_phrases(self) -> tuple[str, ...]:
        """Return a copy of the normalized wake phrases used for matching."""
        with self._lock:
            return tuple(norm for _, norm, _ in self._compiled_patterns)

    # --------------------------------------------------------------------------
    # Wake Phrase Configuration Management
    # --------------------------------------------------------------------------

    def set_wake_phrases(self, phrases: Sequence[str]) -> None:
        """Replace active wake phrases with a new sequence of phrases.

        Args:
            phrases: Sequence of wake phrase strings.

        Raises:
            InvalidInputError: If phrases is not an iterable sequence or contains
                no valid non-empty phrases.
        """
        if phrases is None or not hasattr(phrases, "__iter__"):
            raise InvalidInputError("Wake phrases must be an iterable sequence of strings.")

        with self._lock:
            new_phrases: list[str] = []
            compiled: list[tuple[str, str, re.Pattern[str]]] = []

            for p in phrases:
                if not isinstance(p, str):
                    continue
                p_clean = p.strip()
                if not p_clean:
                    continue
                norm = normalize_text(p_clean)
                if not norm:
                    continue

                if p_clean not in new_phrases:
                    new_phrases.append(p_clean)
                    # Compile regex pattern using word boundaries for precise matching
                    pattern = re.compile(rf"\b{re.escape(norm)}\b")
                    compiled.append((p_clean, norm, pattern))

            # Sort compiled patterns by normalized length descending (most specific first)
            compiled.sort(key=lambda item: (-len(item[1].split()), -len(item[1])))

            self._wake_phrases = new_phrases
            self._compiled_patterns = compiled
            self._logger.debug(
                f"Configured {len(self._wake_phrases)} wake phrases: {self._wake_phrases}"
            )

    def add_wake_phrase(self, phrase: str) -> None:
        """Add a single wake phrase to the detector.

        Args:
            phrase: Wake phrase to add.

        Raises:
            InvalidInputError: If phrase is empty or not a string.
        """
        if not isinstance(phrase, str) or not phrase.strip():
            raise InvalidInputError("Wake phrase must be a non-empty string.")

        with self._lock:
            if phrase.strip() not in self._wake_phrases:
                current = list(self._wake_phrases)
                current.append(phrase.strip())
                self.set_wake_phrases(current)

    def remove_wake_phrase(self, phrase: str) -> bool:
        """Remove a wake phrase from the detector.

        Args:
            phrase: Wake phrase to remove.

        Returns:
            True if phrase was found and removed, False otherwise.
        """
        if not isinstance(phrase, str):
            return False

        with self._lock:
            clean = phrase.strip()
            norm_target = normalize_text(clean)
            found = False
            remaining: list[str] = []

            for p in self._wake_phrases:
                if p == clean or normalize_text(p) == norm_target:
                    found = True
                else:
                    remaining.append(p)

            if found:
                self.set_wake_phrases(remaining)
            return found

    def reset_wake_phrases(self) -> None:
        """Reset wake phrases to configuration defaults or DEFAULT_WAKE_PHRASES."""
        with self._lock:
            if hasattr(self._config, "wake_phrases") and self._config.wake_phrases:
                self.set_wake_phrases(self._config.wake_phrases)
            else:
                self.set_wake_phrases(DEFAULT_WAKE_PHRASES)

    # --------------------------------------------------------------------------
    # Detection Core Logic
    # --------------------------------------------------------------------------

    def _evaluate(self, text: Optional[str]) -> WakeWordResult:
        """Internal synchronous evaluation of input text against wake phrases."""
        timestamp = time.time()
        normalized = normalize_text(text)

        if not normalized:
            return WakeWordResult(
                detected=False,
                matched_phrase=None,
                normalized_text="",
                timestamp=timestamp,
            )

        with self._lock:
            for original_phrase, norm_phrase, pattern in self._compiled_patterns:
                if pattern.search(normalized):
                    return WakeWordResult(
                        detected=True,
                        matched_phrase=original_phrase,
                        normalized_text=normalized,
                        timestamp=timestamp,
                    )

        return WakeWordResult(
            detected=False,
            matched_phrase=None,
            normalized_text=normalized,
            timestamp=timestamp,
        )

    def _create_event_payload(self, result: WakeWordResult) -> dict[str, Any]:
        """Format the event payload for publication."""
        return {
            "detected": result.detected,
            "matched_phrase": result.matched_phrase,
            "normalized_text": result.normalized_text,
            "timestamp": result.timestamp,
            "result": result,
        }

    def detect(self, text: Optional[str], *, publish_event: bool = True) -> WakeWordResult:
        """Synchronously detect if any configured wake phrase is in the text.

        Args:
            text: Input text string to evaluate.
            publish_event: If True, publishes lifecycle events on the Event Bus.
                Defaults to True.

        Returns:
            WakeWordResult indicating detection status and details.
        """
        result = self._evaluate(text)

        if result.detected:
            self._logger.info(
                f"Wake word detected: '{result.matched_phrase}' in '{result.normalized_text}'"
            )
            if publish_event and self._event_bus is not None:
                self._event_bus.publish(
                    event=EVENT_WAKEWORD_DETECTED,
                    payload=self._create_event_payload(result),
                    source=EVENT_SOURCE_WAKEWORD,
                )
        else:
            self._logger.debug(
                f"Wake word ignored for input: '{result.normalized_text}'"
            )
            if publish_event and self._event_bus is not None:
                self._event_bus.publish(
                    event=EVENT_WAKEWORD_IGNORED,
                    payload=self._create_event_payload(result),
                    source=EVENT_SOURCE_WAKEWORD,
                )

        return result

    async def detect_async(
        self, text: Optional[str], *, publish_event: bool = True
    ) -> WakeWordResult:
        """Asynchronously detect if any configured wake phrase is in the text.

        Args:
            text: Input text string to evaluate.
            publish_event: If True, publishes lifecycle events asynchronously
                on the Event Bus. Defaults to True.

        Returns:
            WakeWordResult indicating detection status and details.
        """
        result = self._evaluate(text)

        if result.detected:
            self._logger.info(
                f"Wake word detected (async): '{result.matched_phrase}' in '{result.normalized_text}'"
            )
            if publish_event and self._event_bus is not None:
                await self._event_bus.publish_async(
                    event=EVENT_WAKEWORD_DETECTED,
                    payload=self._create_event_payload(result),
                    source=EVENT_SOURCE_WAKEWORD,
                )
        else:
            self._logger.debug(
                f"Wake word ignored (async) for input: '{result.normalized_text}'"
            )
            if publish_event and self._event_bus is not None:
                await self._event_bus.publish_async(
                    event=EVENT_WAKEWORD_IGNORED,
                    payload=self._create_event_payload(result),
                    source=EVENT_SOURCE_WAKEWORD,
                )

        return result

    def is_wake_phrase(self, text: Optional[str]) -> bool:
        """Convenience helper returning True if a wake word was detected."""
        return self.detect(text, publish_event=False).detected


# ------------------------------------------------------------------------------
# Default Singleton Instance Export
# ------------------------------------------------------------------------------
wake_word_detector: Final[WakeWordDetector] = WakeWordDetector()

__all__ = [
    "EVENT_SOURCE_WAKEWORD",
    "EVENT_WAKEWORD_DETECTED",
    "EVENT_WAKEWORD_IGNORED",
    "WakeWordDetector",
    "normalize_text",
    "wake_word_detector",
]
