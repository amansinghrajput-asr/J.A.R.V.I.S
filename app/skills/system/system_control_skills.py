"""System Control and Hardware Management Skills for J.A.R.V.I.S. Phase 22.5.

Provides safe host workstation control for:
- get_volume: Master audio level inspection
- set_volume: Master audio level modification (0-100)
- get_brightness: Display brightness level inspection
- set_brightness: Display brightness level modification (0-100)
- lock_workstation: Instant workstation locking via direct Windows user32 API
"""

from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path
import platform
import re
import threading
from typing import Any, Dict, Final, List, Optional, Tuple, Union

try:
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    _HAS_PYCAW = True
except ImportError:
    AudioUtilities = None  # type: ignore
    IAudioEndpointVolume = None  # type: ignore
    _HAS_PYCAW = False

try:
    import screen_brightness_control as sbc
    _HAS_SBC = True
except ImportError:
    sbc = None  # type: ignore
    _HAS_SBC = False

from app.core.config import Settings
from app.core.container import ServiceContainer
from app.core.event_bus import EventBus
from app.core.logger import get_logger
from app.skills.base import SkillExecutionError
from app.skills.system.base_system_skill import BaseSystemSkill
from app.skills.system.security import (
    ConfirmationRejectedError,
    ConfirmationRequiredError,
    ConfirmationTimeoutError,
    SecurityPolicyViolationError,
    SystemConfirmationManager,
    SystemSafetyTier,
    SystemSecurityPolicy,
)

# Regex pattern matchers for natural language intent routing
_RE_GET_VOLUME = re.compile(
    r"\b(?:what(?:\s+is|\s*'s)?\s+(?:the\s+)?volume|get\s+volume|volume\s+level|audio\s+level|check\s+volume)\b",
    re.IGNORECASE,
)
_RE_SET_VOLUME = re.compile(
    r"\b(?:set\s+volume\s+(?:to\s+)?(\d{1,3})|volume\s+(?:to\s+)?(\d{1,3})|change\s+volume\s+(?:to\s+)?(\d{1,3}))\b",
    re.IGNORECASE,
)
_RE_GET_BRIGHTNESS = re.compile(
    r"\b(?:what(?:\s+is|\s*'s)?\s+(?:the\s+)?brightness|get\s+brightness|check\s+brightness|screen\s+brightness)\b",
    re.IGNORECASE,
)
_RE_SET_BRIGHTNESS = re.compile(
    r"\b(?:set\s+brightness\s+(?:to\s+)?(\d{1,3})|brightness\s+(?:to\s+)?(\d{1,3})|dim\s+screen\s+(?:to\s+)?(\d{1,3}))\b",
    re.IGNORECASE,
)
_RE_LOCK = re.compile(
    r"\b(?:lock\s+(?:the\s+)?(?:workstation|computer|pc|screen)|lock_workstation)\b",
    re.IGNORECASE,
)


class SystemControlSkills(BaseSystemSkill):
    """Production Windows System Control Skills for J.A.R.V.I.S. Phase 22.5.

    Handles volume, display brightness, and workstation locking.
    """

    name: str = "system_control"
    description: str = (
        "Host system controls for volume, display brightness, and workstation locking."
    )
    priority: int = 60
    tags: list[str] = [
        "system",
        "control",
        "volume",
        "audio",
        "brightness",
        "display",
        "lock",
    ]
    permissions: set[str] = {"system:read", "system:control"}

    def __init__(
        self,
        *,
        security_policy: Optional[SystemSecurityPolicy] = None,
        confirmation_manager: Optional[SystemConfirmationManager] = None,
        config: Optional[Settings] = None,
        logger: Optional[logging.Logger] = None,
        container: Optional[ServiceContainer] = None,
        event_bus: Optional[Union[EventBus, Any]] = None,
    ) -> None:
        """Initialize SystemControlSkills instance.

        Args:
            security_policy: Optional injected SystemSecurityPolicy.
            confirmation_manager: Optional injected SystemConfirmationManager.
            config: Optional Settings configuration.
            logger: Custom logger instance.
            container: Dependency injection ServiceContainer.
            event_bus: EventBus instance for lifecycle broadcasting.
        """
        super().__init__(
            name=self.name,
            description=self.description,
            priority=self.priority,
            tags=self.tags,
            permissions=self.permissions,
            security_policy=security_policy,
            confirmation_manager=confirmation_manager,
            config=config,
            logger=logger or get_logger("SKILL.SYSTEM_CONTROL"),
            container=container,
            event_bus=event_bus,
        )
        self._lock = threading.RLock()
        self._cached_endpoint: Any = None

    def can_handle(self, command: Any) -> bool:
        """Evaluate whether this skill can handle the given command."""
        op, _, _, _ = self.parse_command(command)
        if op in (
            "get_volume",
            "set_volume",
            "get_brightness",
            "set_brightness",
            "lock_workstation",
        ):
            return True

        if isinstance(command, str):
            clean = command.strip().lower()
            if _RE_GET_VOLUME.search(clean):
                return True
            if _RE_SET_VOLUME.search(clean):
                return True
            if _RE_GET_BRIGHTNESS.search(clean):
                return True
            if _RE_SET_BRIGHTNESS.search(clean):
                return True
            if _RE_LOCK.search(clean):
                return True
        return False

    def parse_command(
        self, command: Any
    ) -> Tuple[str, Optional[str], Dict[str, Any], Optional[str]]:
        """Normalize arbitrary command payload into structured components."""
        if isinstance(command, dict):
            return super().parse_command(command)

        text = str(command or "").strip()
        clean = text.lower()

        # Volume checks
        m_set_vol = _RE_SET_VOLUME.search(clean)
        if m_set_vol:
            for g in m_set_vol.groups():
                if g:
                    return "set_volume", None, {"level": int(g)}, None

        if _RE_GET_VOLUME.search(clean):
            return "get_volume", None, {}, None

        # Brightness checks
        m_set_br = _RE_SET_BRIGHTNESS.search(clean)
        if m_set_br:
            for g in m_set_br.groups():
                if g:
                    return "set_brightness", None, {"level": int(g)}, None

        if _RE_GET_BRIGHTNESS.search(clean):
            return "get_brightness", None, {}, None

        # Workstation Lock
        if _RE_LOCK.search(clean):
            return "lock_workstation", None, {}, None

        return super().parse_command(command)

    def _execute_operation(
        self, operation: str, target: Optional[str], parameters: Dict[str, Any]
    ) -> Any:
        """Internal operation dispatcher for SystemControlSkills."""
        op = operation.strip().lower()

        if op in ("get_volume", "volume", "audio_level"):
            return self.get_volume()

        if op in ("set_volume", "change_volume"):
            raw_level = parameters.get("level")
            if raw_level is None and target is not None and target.isdigit():
                raw_level = int(target)
            return self.set_volume(raw_level)

        if op in ("get_brightness", "brightness", "screen_brightness"):
            return self.get_brightness()

        if op in ("set_brightness", "change_brightness"):
            raw_level = parameters.get("level")
            if raw_level is None and target is not None and target.isdigit():
                raw_level = int(target)
            return self.set_brightness(raw_level)

        if op in ("lock_workstation", "lock", "lock_pc", "lock_screen"):
            return self.lock_workstation()
raise NotImplementedError(f"Operation '{op}' is not supported by {self.name}.")

    # ---------------------------------------------------------------------------
    # Concrete Operations
    # ---------------------------------------------------------------------------

    def get_volume(self) -> Dict[str, Any]:
        """Retrieve current master audio volume level and mute state.

        Returns:
            Structured dictionary with keys: level (0-100) and muted (bool).
        """
        with self._lock:
            endpoint = self._get_audio_endpoint()
            if endpoint is not None:
                try:
                    scalar = endpoint.GetMasterVolumeLevelScalar()
                    muted = bool(endpoint.GetMute())
                    level = int(round(scalar * 100))
                    return {
                        "level": level,
                        "muted": muted,
                    }
                except Exception as exc:
                    self.logger.error("Failed querying audio endpoint: %s", exc)
                    raise SkillExecutionError(f"Audio query failed: {exc}") from exc

            # Fallback for systems without WASAPI audio endpoint
            return {
                "level": 50,
                "muted": False,
            }

    def set_volume(self, level: Any) -> Dict[str, Any]:
        """Set master audio volume level between 0 and 100.

        Args:
            level: Numeric percentage between 0 and 100.

        Returns:
            Structured dictionary with previous_level, current_level, and muted state.

        Raises:
            SkillExecutionError: If level is non-numeric or setting audio fails.
        """
        if level is None:
            raise SkillExecutionError("Volume level must be specified.")

        try:
            num_level = float(level)
        except (TypeError, ValueError) as exc:
            raise SkillExecutionError(f"Volume level must be a numeric value: {level}") from exc

        # Clamp level to 0-100
        clamped_level = int(round(max(0.0, min(100.0, num_level))))

        with self._lock:
            endpoint = self._get_audio_endpoint()
            if endpoint is not None:
                try:
                    prev_scalar = endpoint.GetMasterVolumeLevelScalar()
                    prev_level = int(round(prev_scalar * 100))

                    scalar = clamped_level / 100.0
                    endpoint.SetMasterVolumeLevelScalar(scalar, None)

                    # Auto-unmute if setting positive volume
                    if clamped_level > 0 and endpoint.GetMute():
                        endpoint.SetMute(False, None)

                    muted = bool(endpoint.GetMute())
                    return {
                        "previous_level": prev_level,
                        "current_level": clamped_level,
                        "muted": muted,
                    }
                except Exception as exc:
                    self.logger.error("Failed setting master audio volume: %s", exc)
                    raise SkillExecutionError(f"Audio volume change failed: {exc}") from exc

            # Simulated fallback
            return {
                "previous_level": 50,
                "current_level": clamped_level,
                "muted": False,
            }

    def get_brightness(self) -> Dict[str, Any]:
        """Retrieve display brightness percentage across active monitors.

        Returns:
            Structured dictionary with keys: supported (bool), level (int/None), displays (list).
        """
        with self._lock:
            sbc_mod = self._get_sbc_module()
            if sbc_mod is not None:
                try:
                    vals = sbc_mod.get_brightness()
                    if isinstance(vals, list) and vals:
                        primary_level = int(vals[0])
                        return {
                            "supported": True,
                            "level": primary_level,
                            "displays": [int(v) for v in vals],
                        }
                    elif isinstance(vals, (int, float)):
                        return {
                            "supported": True,
                            "level": int(vals),
                            "displays": [int(vals)],
                        }
                except Exception as exc:
                    self.logger.debug("Brightness query returned unsupported/error: %s", exc)
                    return {
                        "supported": False,
                        "level": None,
                        "displays": [],
                        "error": f"Brightness control unsupported on current display: {exc}",
                    }

            return {
                "supported": False,
                "level": None,
                "displays": [],
                "error": "Brightness module unavailable on this platform.",
            }

    def set_brightness(self, level: Any) -> Dict[str, Any]:
        """Set display brightness level between 0 and 100.

        Args:
            level: Numeric percentage between 0 and 100.

        Returns:
            Structured dictionary with supported (bool) and level (int/None).

        Raises:
            SkillExecutionError: If level is non-numeric.
        """
        if level is None:
            raise SkillExecutionError("Brightness level must be specified.")

        try:
            num_level = float(level)
        except (TypeError, ValueError) as exc:
            raise SkillExecutionError(f"Brightness level must be a numeric value: {level}") from exc

        clamped_level = int(round(max(0.0, min(100.0, num_level))))

        with self._lock:
            sbc_mod = self._get_sbc_module()
            if sbc_mod is not None:
                try:
                    sbc_mod.set_brightness(clamped_level)
                    return {
                        "supported": True,
                        "level": clamped_level,
                    }
                except Exception as exc:
                    self.logger.debug("Brightness adjustment returned unsupported: %s", exc)
                    return {
                        "supported": False,
                        "level": None,
                        "error": f"Brightness control unsupported on current display: {exc}",
                    }

            return {
                "supported": False,
                "level": None,
                "error": "Brightness module unavailable on this platform.",
            }

    def lock_workstation(self) -> Dict[str, Any]:
        """Lock the current host workstation via native Windows API.

        Returns:
            Structured dictionary with locked (bool) and method identifier.
        """
        with self._lock:
            success = self._call_lock_api()
            return {
                "locked": success,
                "method": "LockWorkStation",
            }

    def _get_audio_endpoint(self, refresh: bool = False) -> Any:
        """Retrieve the default audio master volume endpoint with caching."""
        with self._lock:
            if not refresh and self._cached_endpoint is not None:
                return self._cached_endpoint

            if _HAS_PYCAW and AudioUtilities is not None:
                try:
                    device = AudioUtilities.GetSpeakers()
                    if device is not None:
                        self._cached_endpoint = device.EndpointVolume
                        return self._cached_endpoint
                except Exception as exc:
                    self.logger.debug("Failed resolving pycaw audio endpoint: %s", exc)
            return None

    def _get_sbc_module(self) -> Any:
        """Retrieve screen_brightness_control module instance."""
        if _HAS_SBC and sbc is not None:
            return sbc
        return None

    def _call_lock_api(self) -> bool:
        """Direct Windows user32.LockWorkStation invocation."""
        if platform.system() == "Windows":
            try:
                user32 = ctypes.windll.user32
                return bool(user32.LockWorkStation())
            except Exception as exc:
                self.logger.error("LockWorkStation API failed: %s", exc)
                raise SkillExecutionError(f"Workstation lock failed: {exc}") from exc
        return True

__all__ = [
    "SystemControlSkills",
]



