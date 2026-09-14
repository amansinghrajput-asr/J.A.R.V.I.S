"""Hardware-independent deterministic unit tests for Phase 27.1 Desktop Capture Engine.

SAFETY GUARANTEES:
- All Win32 GDI and User32 calls are 100% mocked.
- ZERO interaction with real desktop hardware or display drivers.
- ZERO screenshots are persisted to disk.
- ZERO OS/window states are modified.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import unittest
from unittest.mock import MagicMock, call, patch

from app.core.container import ServiceContainer
from app.core.event_bus import EventBus
from app.vision.capture import (
    DesktopCaptureEngine,
    MockCaptureBackend,
    Win32GdiCaptureBackend,
    init_dpi_awareness,
)
from app.vision.models import (
    CaptureError,
    GdiResourceError,
    InvalidBoundsError,
    MonitorInfo,
    Point,
    ScreenCapture,
    UnsupportedPlatformError,
    WindowBounds,
)


class TestDesktopCaptureEngineWithMockBackend(unittest.TestCase):
    """Test DesktopCaptureEngine behaviors using deterministic MockCaptureBackend."""

    def setUp(self) -> None:
        self.container = ServiceContainer()
        self.event_bus = EventBus()
        self.primary_bounds = WindowBounds(left=0, top=0, right=1920, bottom=1080)
        self.secondary_bounds = WindowBounds(left=-1920, top=0, right=0, bottom=1080)
        self.virtual_bounds = WindowBounds(left=-1920, top=0, right=1920, bottom=1080)

        self.monitors = [
            MonitorInfo(
                handle=1,
                name="PRIMARY_DISPLAY",
                bounds=self.primary_bounds,
                is_primary=True,
            ),
            MonitorInfo(
                handle=2,
                name="LEFT_DISPLAY",
                bounds=self.secondary_bounds,
                is_primary=False,
            ),
        ]
        self.mock_backend = MockCaptureBackend(
            monitors=self.monitors,
            virtual_bounds=self.virtual_bounds,
        )
        self.engine = DesktopCaptureEngine(
            backend=self.mock_backend,
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=True,
        )

    def test_container_registration(self) -> None:
        """Verify capture engine registers cleanly into ServiceContainer."""
        self.assertTrue(self.container.exists("desktop_capture_engine"))
        resolved = self.container.resolve("desktop_capture_engine")
        self.assertIs(resolved, self.engine)

    def test_default_screen_capture_uses_primary_monitor(self) -> None:
        """Verify capturing screen without explicit bounds targets the primary display."""
        cap = self.engine.capture_screen()
        self.assertEqual(cap.width, 1920)
        self.assertEqual(cap.height, 1080)
        self.assertEqual(cap.bounds, self.primary_bounds)
        self.assertEqual(cap.source, "screen")
        self.assertEqual(self.mock_backend.capture_count, 1)

    def test_capture_arbitrary_rectangle(self) -> None:
        """Verify arbitrary rectangle capture."""
        target_bounds = WindowBounds(left=200, top=150, right=600, bottom=450)
        cap = self.engine.capture_screen(target_bounds, source="custom_region")
        self.assertEqual(cap.width, 400)
        self.assertEqual(cap.height, 300)
        self.assertEqual(cap.bounds, target_bounds)
        self.assertEqual(cap.source, "custom_region")

    def test_capture_negative_coordinates(self) -> None:
        """Verify capturing region on secondary monitor with negative coordinates."""
        cap = self.engine.capture_screen(self.secondary_bounds, source="secondary")
        self.assertEqual(cap.width, 1920)
        self.assertEqual(cap.height, 1080)
        self.assertEqual(cap.bounds, self.secondary_bounds)
        self.assertEqual(cap.bounds.left, -1920)

    def test_capture_monitor_by_index(self) -> None:
        """Verify capturing specific monitor indices."""
        cap0 = self.engine.capture_monitor(0)
        self.assertEqual(cap0.bounds, self.primary_bounds)
        self.assertEqual(cap0.source, "monitor_0")

        cap1 = self.engine.capture_monitor(1)
        self.assertEqual(cap1.bounds, self.secondary_bounds)
        self.assertEqual(cap1.source, "monitor_1")

        # Invalid monitor index
        with self.assertRaises(CaptureError):
            self.engine.capture_monitor(99)

    def test_capture_virtual_desktop(self) -> None:
        """Verify virtual desktop capture spanning all displays."""
        cap = self.engine.capture_virtual_desktop()
        self.assertEqual(cap.bounds, self.virtual_bounds)
        self.assertEqual(cap.width, 3840)
        self.assertEqual(cap.height, 1080)
        self.assertEqual(cap.source, "virtual_desktop")

    def test_capture_active_window(self) -> None:
        """Verify active foreground window capture."""
        cap = self.engine.capture_active_window()
        self.assertEqual(cap.source, "window")
        self.assertEqual(cap.metadata.get("hwnd"), 12345)
        self.assertEqual(cap.width, 800)
        self.assertEqual(cap.height, 600)

    def test_capture_window_by_handle(self) -> None:
        """Verify capture of known window handle."""
        cap = self.engine.capture_window(12345)
        self.assertEqual(cap.width, 800)
        self.assertEqual(cap.height, 600)

        # Non-existent window handle
        with self.assertRaises(CaptureError):
            self.engine.capture_window(999999)

    def test_invalid_bounds_rejected(self) -> None:
        """Verify zero or inverted bounds raise InvalidBoundsError."""
        zero_bounds = WindowBounds(left=100, top=100, right=100, bottom=100)
        with self.assertRaises(InvalidBoundsError):
            self.engine.capture_screen(zero_bounds)

        neg_bounds = WindowBounds(left=500, top=500, right=200, bottom=200)
        with self.assertRaises(InvalidBoundsError):
            self.engine.capture_screen(neg_bounds)

    def test_event_bus_emission_on_capture(self) -> None:
        """Verify EVENT_SCREEN_CAPTURED is published upon capture."""
        received_events = []
        self.event_bus.subscribe("vision.screen_captured", lambda e: received_events.append(e))

        self.engine.capture_screen()
        self.assertEqual(len(received_events), 1)
        ev = received_events[0]
        payload = ev.payload if hasattr(ev, "payload") else ev
        self.assertEqual(payload["width"], 1920)
        self.assertEqual(payload["height"], 1080)

    def test_async_capture_methods(self) -> None:
        """Verify non-blocking async wrapper execution."""
        async def _test():
            cap = await self.engine.capture_screen_async()
            self.assertEqual(cap.width, 1920)

            act_cap = await self.engine.capture_active_window_async()
            self.assertEqual(act_cap.width, 800)

            v_cap = await self.engine.capture_virtual_desktop_async()
            self.assertEqual(v_cap.width, 3840)

        asyncio.run(_test())

    def test_concurrent_captures_are_thread_safe(self) -> None:
        """Verify thread safety under concurrent requests."""
        results = []
        def _worker():
            c = self.engine.capture_screen()
            results.append(c.width)

        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(_worker) for _ in range(12)]
            for f in futs:
                f.result()

        self.assertEqual(len(results), 12)
        self.assertTrue(all(w == 1920 for w in results))
        self.assertEqual(self.mock_backend.capture_count, 12)

    def test_window_skills_integration(self) -> None:
        """Verify engine delegates to WindowSkills if available in container."""
        mock_ws = MagicMock()
        mock_ws._api_get_foreground_window.return_value = 54321
        mock_ws.get_window_bounds.return_value = WindowBounds(left=50, top=50, right=450, bottom=350)
        self.container.register_singleton("window_skills", mock_ws, allow_override=True)

        hwnd = self.engine.get_active_window_handle()
        self.assertEqual(hwnd, 54321)

        bounds = self.engine.get_active_window_bounds()
        self.assertEqual(bounds, WindowBounds(left=50, top=50, right=450, bottom=350))


class TestWin32GdiCaptureBackendMocked(unittest.TestCase):
    """Surgical GDI lifecycle and resource cleanup tests with mocked Win32 APIs."""

    def setUp(self) -> None:
        self.backend = Win32GdiCaptureBackend()

    def test_gdi_cleanup_on_success(self) -> None:
        """Verify all GDI handles are released cleanly on successful capture."""
        bounds = WindowBounds(left=0, top=0, right=100, bottom=100)

        # Mock all low-level API calls
        self.backend._is_win = True
        self.backend._api_get_desktop_dc = MagicMock(return_value=1001)
        self.backend._api_create_compatible_dc = MagicMock(return_value=1002)
        self.backend._api_create_compatible_bitmap = MagicMock(return_value=1003)
        self.backend._api_select_object = MagicMock(return_value=1004)
        self.backend._api_bit_blt = MagicMock(return_value=True)
        self.backend._api_get_di_bits = MagicMock(return_value=100)
        self.backend._api_delete_object = MagicMock(return_value=True)
        self.backend._api_delete_dc = MagicMock(return_value=True)
        self.backend._api_release_dc = MagicMock(return_value=1)

        cap = self.backend.capture_rect(bounds)

        self.assertEqual(cap.width, 100)
        self.assertEqual(cap.height, 100)
        self.assertEqual(cap.size_bytes, 100 * 100 * 4)

        # Verify GDI release sequence
        self.backend._api_select_object.assert_called_with(1002, 1004)
        self.backend._api_delete_object.assert_called_with(1003)
        self.backend._api_delete_dc.assert_called_with(1002)
        self.backend._api_release_dc.assert_called_with(0, 1001)

    def test_gdi_cleanup_when_desktop_dc_fails(self) -> None:
        """Verify error handling when GetDC fails."""
        bounds = WindowBounds(left=0, top=0, right=100, bottom=100)
        self.backend._is_win = True
        self.backend._api_get_desktop_dc = MagicMock(return_value=0)

        with self.assertRaises(GdiResourceError):
            self.backend.capture_rect(bounds)

    def test_gdi_cleanup_when_bitmap_creation_fails(self) -> None:
        """Verify cleanup of DC when CreateCompatibleBitmap fails."""
        bounds = WindowBounds(left=0, top=0, right=100, bottom=100)
        self.backend._is_win = True
        self.backend._api_get_desktop_dc = MagicMock(return_value=2001)
        self.backend._api_create_compatible_dc = MagicMock(return_value=2002)
        self.backend._api_create_compatible_bitmap = MagicMock(return_value=0)  # Failure
        self.backend._api_delete_dc = MagicMock(return_value=True)
        self.backend._api_release_dc = MagicMock(return_value=1)

        with self.assertRaises(GdiResourceError):
            self.backend.capture_rect(bounds)

        # Ensure acquired DCs are freed
        self.backend._api_delete_dc.assert_called_with(2002)
        self.backend._api_release_dc.assert_called_with(0, 2001)

    def test_gdi_cleanup_when_bitblt_fails(self) -> None:
        """Verify full handle cleanup when BitBlt returns False."""
        bounds = WindowBounds(left=0, top=0, right=100, bottom=100)
        self.backend._is_win = True
        self.backend._api_get_desktop_dc = MagicMock(return_value=3001)
        self.backend._api_create_compatible_dc = MagicMock(return_value=3002)
        self.backend._api_create_compatible_bitmap = MagicMock(return_value=3003)
        self.backend._api_select_object = MagicMock(return_value=3004)
        self.backend._api_bit_blt = MagicMock(return_value=False)  # BitBlt fails
        self.backend._api_delete_object = MagicMock(return_value=True)
        self.backend._api_delete_dc = MagicMock(return_value=True)
        self.backend._api_release_dc = MagicMock(return_value=1)

        with self.assertRaises(GdiResourceError):
            self.backend.capture_rect(bounds)

        self.backend._api_delete_object.assert_called_with(3003)
        self.backend._api_delete_dc.assert_called_with(3002)
        self.backend._api_release_dc.assert_called_with(0, 3001)

    def test_gdi_cleanup_when_getdibits_fails(self) -> None:
        """Verify full handle cleanup when GetDIBits returns incomplete scanlines."""
        bounds = WindowBounds(left=0, top=0, right=50, bottom=50)
        self.backend._is_win = True
        self.backend._api_get_desktop_dc = MagicMock(return_value=4001)
        self.backend._api_create_compatible_dc = MagicMock(return_value=4002)
        self.backend._api_create_compatible_bitmap = MagicMock(return_value=4003)
        self.backend._api_select_object = MagicMock(return_value=4004)
        self.backend._api_bit_blt = MagicMock(return_value=True)
        self.backend._api_get_di_bits = MagicMock(return_value=25)  # Expected 50
        self.backend._api_delete_object = MagicMock(return_value=True)
        self.backend._api_delete_dc = MagicMock(return_value=True)
        self.backend._api_release_dc = MagicMock(return_value=1)

        with self.assertRaises(GdiResourceError):
            self.backend.capture_rect(bounds)

        self.backend._api_delete_object.assert_called_with(4003)
        self.backend._api_delete_dc.assert_called_with(4002)
        self.backend._api_release_dc.assert_called_with(0, 4001)

    def test_unsupported_platform_handling(self) -> None:
        """Verify capture on unsupported platforms raises clean UnsupportedPlatformError."""
        self.backend._is_win = False
        with self.assertRaises(UnsupportedPlatformError):
            self.backend.capture_rect(WindowBounds(left=0, top=0, right=100, bottom=100))


class TestCaptureSafetyInvariants(unittest.TestCase):
    """Verify Phase 27.1 Safety Invariants: No disk writes, read-only desktop."""

    def test_zero_filesystem_artifacts_created(self) -> None:
        """Verify capturing does NOT write any image files to disk."""
        initial_cwd_files = set(os.listdir("."))
        mock_backend = MockCaptureBackend()
        engine = DesktopCaptureEngine(backend=mock_backend)

        cap = engine.capture_screen()
        # Verify in memory
        self.assertIsInstance(cap.raw_data, bytes)
        png = cap.to_png_bytes()
        self.assertIsInstance(png, bytes)

        after_cwd_files = set(os.listdir("."))
        # CWD must not contain new screenshots
        new_files = after_cwd_files - initial_cwd_files
        self.assertEqual(len(new_files), 0, f"Detected unauthorized disk writes: {new_files}")


if __name__ == "__main__":
    unittest.main()
