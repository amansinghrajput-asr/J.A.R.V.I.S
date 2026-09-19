"""Unit Tests for VirtualInputBackend, MockInputBackend, and Win32InputBackend.

Phase 27.18 - Visual Action Execution Bridge & Verified Interaction.

Ensures:
- MockInputBackend records structured synthetic events correctly.
- All 7 supported visual action primitives are supported:
  click, double_click, type_text, clear_and_type, select_option, toggle, dismiss_modal.
- Zero physical cursor movement or keyboard typing during tests.
- Win32InputBackend invocation count remains strictly 0 throughout all tests.
"""

from __future__ import annotations

import threading
import time
import pytest

from app.automation.input import (
    InputEvent,
    MockInputBackend,
    VirtualInputBackend,
    Win32InputBackend,
)
from app.vision.models import Point


@pytest.fixture(autouse=True)
def ensure_zero_real_win32_calls() -> None:
    """Safeguard: verify Win32 real input is never invoked in tests."""
    Win32InputBackend.reset_invocation_count()
    Win32InputBackend.enable_test_safety_guard()
    yield
    # Explicit Safety Assertion (Section 18)
    assert Win32InputBackend.invocation_count == 0, (
        f"CRITICAL SAFETY VIOLATION: Win32InputBackend was invoked {Win32InputBackend.invocation_count} times during tests!"
    )


class TestInputAdapter:
    """Test suite for MockInputBackend operations and safety guarantees."""

    def test_01_mock_click_event_generation(self) -> None:
        """Verify click records structured synthetic event with correct point and button."""
        mock = MockInputBackend()
        pt = Point(150, 250)

        ok = mock.click(pt, button="left")
        assert ok is True
        assert mock.event_count() == 1
        assert mock.event_count("click") == 1

        events = mock.get_events()
        assert len(events) == 1
        ev = events[0]
        assert ev.event_type == "click"
        assert ev.point == pt
        assert ev.button == "left"
        assert ev.input_present is False
        assert ev.input_length == 0
        assert isinstance(ev.timestamp, float)

    def test_02_mock_double_click_event_generation(self) -> None:
        """Verify double click records double_click event."""
        mock = MockInputBackend()
        pt = Point(300, 400)

        ok = mock.double_click(pt)
        assert ok is True
        assert mock.event_count("double_click") == 1
        ev = mock.get_events("double_click")[0]
        assert ev.point == pt

    def test_03_mock_type_text_privacy(self) -> None:
        """Verify type_text records length and presence without storing raw secret text."""
        mock = MockInputBackend()
        secret = "SuperSecretPassword123!"

        ok = mock.type_text(secret)
        assert ok is True
        assert mock.event_count("type_text") == 1

        ev = mock.get_events("type_text")[0]
        assert ev.event_type == "type_text"
        assert ev.input_present is True
        assert ev.input_length == len(secret)
        # Verify secret text is NOT stored anywhere in ev or ev.to_dict()
        assert secret not in str(ev)
        assert secret not in str(ev.to_dict())

    def test_04_mock_clear_and_type_event_generation(self) -> None:
        """Verify clear_and_type records point and length."""
        mock = MockInputBackend()
        pt = Point(200, 100)
        text = "Hello World"

        ok = mock.clear_and_type(pt, text)
        assert ok is True
        assert mock.event_count("clear_and_type") == 1

        ev = mock.get_events("clear_and_type")[0]
        assert ev.point == pt
        assert ev.input_present is True
        assert ev.input_length == len(text)
        assert text not in str(ev)

    def test_05_mock_select_option_event_generation(self) -> None:
        """Verify select_option records event at target point."""
        mock = MockInputBackend()
        pt = Point(500, 300)

        ok = mock.select_option(pt)
        assert ok is True
        assert mock.event_count("select_option") == 1
        assert mock.get_events("select_option")[0].point == pt

    def test_06_mock_toggle_event_generation(self) -> None:
        """Verify toggle records event at target point."""
        mock = MockInputBackend()
        pt = Point(120, 140)

        ok = mock.toggle(pt)
        assert ok is True
        assert mock.event_count("toggle") == 1
        assert mock.get_events("toggle")[0].point == pt

    def test_07_mock_dismiss_modal_event_generation(self) -> None:
        """Verify dismiss_modal records event at target button point."""
        mock = MockInputBackend()
        pt = Point(800, 200)

        ok = mock.dismiss_modal(pt)
        assert ok is True
        assert mock.event_count("dismiss_modal") == 1
        assert mock.get_events("dismiss_modal")[0].point == pt

    def test_08_mock_simulate_failure(self) -> None:
        """Verify simulated failure returns False and does not record successful dispatch."""
        mock = MockInputBackend()
        mock.simulate_failure("click", should_fail=True)

        ok = mock.click(Point(100, 100))
        assert ok is False
        assert mock.event_count("click") == 0

        # Unaffected event types still succeed
        ok_toggle = mock.toggle(Point(200, 200))
        assert ok_toggle is True
        assert mock.event_count("toggle") == 1

    def test_09_mock_clear(self) -> None:
        """Verify clear resets all recorded events and failure simulations."""
        mock = MockInputBackend()
        mock.click(Point(10, 20))
        mock.simulate_failure("toggle", should_fail=True)
        assert mock.event_count() == 1

        mock.clear()
        assert mock.event_count() == 0
        assert mock.toggle(Point(10, 20)) is True

    def test_10_mock_thread_safety(self) -> None:
        """Verify concurrent dispatches to MockInputBackend are thread-safe."""
        mock = MockInputBackend()
        threads = []

        def worker(idx: int) -> None:
            for i in range(25):
                mock.click(Point(idx, i))

        for t_idx in range(4):
            t = threading.Thread(target=worker, args=(t_idx,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert mock.event_count("click") == 100
