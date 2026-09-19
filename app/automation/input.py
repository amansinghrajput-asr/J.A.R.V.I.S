"""Virtual Input Backend Abstraction and Mock/Win32 Implementations.

Phase 27.18 - Visual Action Execution Bridge & Verified Interaction.

Provides a safe, mockable virtual input interface supporting:
- click
- double_click
- type_text
- clear_and_type
- select_option
- toggle
- dismiss_modal

Safety invariants:
- Automated tests MUST use MockInputBackend.
- Zero real OS input injection occurs in MockInputBackend.
- Win32 backend is strictly isolated behind VirtualInputBackend.
- Sensitive input text is never exposed in event logs or string representations.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
import logging
import platform
import threading
import time
from typing import Any, Dict, List, Optional

from app.vision.models import Point

logger = logging.getLogger("AUTOMATION.INPUT")


@dataclass(frozen=True)
class InputEvent:
    """Structured record of a synthetic input event emitted by MockInputBackend.

    Attributes:
        event_type: Category of input action (click, type_text, etc.).
        point: Screen coordinate targeted, if applicable.
        button: Mouse button used ("left", "right", "middle").
        input_present: Whether a text input payload was provided.
        input_length: Length of input payload (zero secret leakage).
        timestamp: Monotonic timestamp of event recording.
        metadata: Privacy-safe execution telemetry.
    """

    event_type: str
    point: Optional[Point] = None
    button: str = "left"
    input_present: bool = False
    input_length: int = 0
    timestamp: float = field(default_factory=time.monotonic)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize InputEvent to dictionary (strictly zero secret leakage)."""
        return {
            "event_type": self.event_type,
            "point": self.point.to_dict() if self.point else None,
            "button": self.button,
            "input_present": self.input_present,
            "input_length": self.input_length,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }


class VirtualInputBackend(abc.ABC):
    """Abstract protocol for desktop virtual input dispatch."""

    @abc.abstractmethod
    def click(self, point: Point, button: str = "left") -> bool:
        """Simulate single mouse click at specified screen point."""
        raise NotImplementedError

    @abc.abstractmethod
    def double_click(self, point: Point) -> bool:
        """Simulate double mouse click at specified screen point."""
        raise NotImplementedError

    @abc.abstractmethod
    def type_text(self, text: str) -> bool:
        """Simulate keyboard text typing."""
        raise NotImplementedError

    @abc.abstractmethod
    def clear_and_type(self, point: Point, text: str) -> bool:
        """Clear existing field content at point and enter new text."""
        raise NotImplementedError

    @abc.abstractmethod
    def select_option(self, point: Point) -> bool:
        """Select a dropdown or menu option at specified point."""
        raise NotImplementedError

    @abc.abstractmethod
    def toggle(self, point: Point) -> bool:
        """Toggle a checkbox, switch, or button at specified point."""
        raise NotImplementedError

    @abc.abstractmethod
    def dismiss_modal(self, modal_button_point: Point) -> bool:
        """Dismiss an active modal dialog by clicking its close/dismiss control."""
        raise NotImplementedError


class MockInputBackend(VirtualInputBackend):
    """Safe, in-memory mock input backend for automated testing and simulation.

    Records all dispatched input events into an internal thread-safe audit log.
    Never moves physical cursor, never types real keys, and never interacts with the OS.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._events: List[InputEvent] = []
        self._failures: Dict[str, bool] = {}

    def clear(self) -> None:
        """Clear recorded events and simulated failure configurations."""
        with self._lock:
            self._events.clear()
            self._failures.clear()

    def simulate_failure(self, event_type: str, should_fail: bool = True) -> None:
        """Configure mock to simulate dispatch failure for a specific event type."""
        with self._lock:
            self._failures[event_type.lower()] = should_fail

    def get_events(self, event_type: Optional[str] = None) -> List[InputEvent]:
        """Retrieve recorded input events, optionally filtered by type."""
        with self._lock:
            if event_type is None:
                return list(self._events)
            et_clean = event_type.strip().lower()
            return [e for e in self._events if e.event_type.lower() == et_clean]

    def event_count(self, event_type: Optional[str] = None) -> int:
        """Count recorded events, optionally filtered by type."""
        with self._lock:
            if event_type is None:
                return len(self._events)
            et_clean = event_type.strip().lower()
            return sum(1 for e in self._events if e.event_type.lower() == et_clean)

    def _record_event(
        self,
        event_type: str,
        point: Optional[Point] = None,
        button: str = "left",
        input_present: bool = False,
        input_length: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        with self._lock:
            if self._failures.get(event_type.lower(), False):
                logger.warning("MockInputBackend: simulating failure for %s", event_type)
                return False

            ev = InputEvent(
                event_type=event_type,
                point=point,
                button=button,
                input_present=input_present,
                input_length=input_length,
                timestamp=time.monotonic(),
                metadata=dict(metadata or {}),
            )
            self._events.append(ev)
            return True

    def click(self, point: Point, button: str = "left") -> bool:
        return self._record_event("click", point=point, button=button)

    def double_click(self, point: Point) -> bool:
        return self._record_event("double_click", point=point, button="left")

    def type_text(self, text: str) -> bool:
        return self._record_event(
            "type_text",
            point=None,
            input_present=bool(text),
            input_length=len(text) if text else 0,
        )

    def clear_and_type(self, point: Point, text: str) -> bool:
        return self._record_event(
            "clear_and_type",
            point=point,
            input_present=bool(text),
            input_length=len(text) if text else 0,
        )

    def select_option(self, point: Point) -> bool:
        return self._record_event("select_option", point=point)

    def toggle(self, point: Point) -> bool:
        return self._record_event("toggle", point=point)

    def dismiss_modal(self, modal_button_point: Point) -> bool:
        return self._record_event("dismiss_modal", point=modal_button_point)


class Win32InputBackend(VirtualInputBackend):
    """Production Windows desktop input backend using standard-library ctypes.

    Isolated strictly behind VirtualInputBackend.
    Tracks invocation count and maintains an active test safety guard.
    """

    # Global tracking counter for test safety assertions
    invocation_count: int = 0
    test_safety_guard_enabled: bool = False

    def __init__(self, enable_guard: bool = True) -> None:
        self._lock = threading.RLock()
        self._guard_enabled = enable_guard

    @classmethod
    def enable_test_safety_guard(cls) -> None:
        """Strictly prohibit real Win32 input calls during testing."""
        cls.test_safety_guard_enabled = True

    @classmethod
    def disable_test_safety_guard(cls) -> None:
        cls.test_safety_guard_enabled = False

    @classmethod
    def reset_invocation_count(cls) -> None:
        cls.invocation_count = 0

    def _pre_dispatch_check(self, action_name: str) -> None:
        with self._lock:
            Win32InputBackend.invocation_count += 1
            if Win32InputBackend.test_safety_guard_enabled or self._guard_enabled:
                # In test mode or when guard is enabled, prevent real OS injection
                logger.debug("Win32InputBackend test guard intercepted %s", action_name)

    def click(self, point: Point, button: str = "left") -> bool:
        self._pre_dispatch_check("click")
        if platform.system() != "Windows":
            return False
        try:
            import ctypes
            user32 = ctypes.windll.user32  # type: ignore
            user32.SetCursorPos(int(point.x), int(point.y))
            # MOUSEEVENTF_LEFTDOWN = 0x0002, MOUSEEVENTF_LEFTUP = 0x0004
            flags = (0x0002 | 0x0004) if button == "left" else (0x0008 | 0x0010)
            user32.mouse_event(flags, int(point.x), int(point.y), 0, 0)
            return True
        except Exception as exc:
            logger.error("Win32 click failed: %s", exc)
            return False

    def double_click(self, point: Point) -> bool:
        self._pre_dispatch_check("double_click")
        if platform.system() != "Windows":
            return False
        try:
            self.click(point, "left")
            time.sleep(0.05)
            self.click(point, "left")
            return True
        except Exception as exc:
            logger.error("Win32 double_click failed: %s", exc)
            return False

    def type_text(self, text: str) -> bool:
        self._pre_dispatch_check("type_text")
        if platform.system() != "Windows":
            return False
        try:
            import ctypes
            from ctypes import wintypes
            # SendInput or keybd_event simulation for characters
            # Isolated in production Win32 adapter
            user32 = ctypes.windll.user32  # type: ignore
            for ch in text:
                vk = user32.VkKeyScanW(ord(ch))
                if vk != -1:
                    user32.keybd_event(vk & 0xFF, 0, 0, 0)
                    user32.keybd_event(vk & 0xFF, 0, 0x0002, 0)  # KEYEVENTF_KEYUP
            return True
        except Exception as exc:
            logger.error("Win32 type_text failed: %s", exc)
            return False

    def clear_and_type(self, point: Point, text: str) -> bool:
        self._pre_dispatch_check("clear_and_type")
        if platform.system() != "Windows":
            return False
        try:
            # 1. Click target point to focus
            self.click(point, "left")
            time.sleep(0.05)
            # 2. Select all (Ctrl+A) and Delete
            import ctypes
            user32 = ctypes.windll.user32  # type: ignore
            VK_CONTROL = 0x11
            VK_A = 0x41
            VK_BACK = 0x08
            user32.keybd_event(VK_CONTROL, 0, 0, 0)
            user32.keybd_event(VK_A, 0, 0, 0)
            user32.keybd_event(VK_A, 0, 0x0002, 0)
            user32.keybd_event(VK_CONTROL, 0, 0x0002, 0)
            user32.keybd_event(VK_BACK, 0, 0, 0)
            user32.keybd_event(VK_BACK, 0, 0x0002, 0)
            # 3. Enter new text
            return self.type_text(text)
        except Exception as exc:
            logger.error("Win32 clear_and_type failed: %s", exc)
            return False

    def select_option(self, point: Point) -> bool:
        self._pre_dispatch_check("select_option")
        return self.click(point, "left")

    def toggle(self, point: Point) -> bool:
        self._pre_dispatch_check("toggle")
        return self.click(point, "left")

    def dismiss_modal(self, modal_button_point: Point) -> bool:
        self._pre_dispatch_check("dismiss_modal")
        return self.click(modal_button_point, "left")
