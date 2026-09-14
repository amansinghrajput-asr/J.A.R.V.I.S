"""Desktop and Window Screen Capture Subsystem for J.A.R.V.I.S. Phase 27.1.

Implements native Windows screen capture via Win32 GDI (BitBlt / GetDIBits)
with strict resource management, multi-monitor enumeration, negative coordinate
handling, active-window bounds tracking, and mockable testing backends.

Safety Invariants:
1. READ-ONLY: Never moves, resizes, focuses, minimizes, or closes user windows.
2. ZERO DISK PERSISTENCE: Captures are kept strictly in-memory as ScreenCapture buffers.
3. GDI LEAK IMMUNITY: Every DC, bitmap handle, and memory object is released in finally blocks.
4. THREAD SAFETY: Synchronous operations are thread-safe and non-blocking to the GUI loop.
5. NO CONTINUOUS CAPTURE: On-demand only; zero background screen recording loops.
"""

from __future__ import annotations

import abc
import asyncio
import ctypes
from ctypes import wintypes
import logging
import platform
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from app.core.container import ServiceContainer, container as default_container
from app.core.event_bus import EventBus, event_bus as default_event_bus
from app.core.logger import get_logger
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

logger = get_logger("VISION.CAPTURE")

# --------------------------------------------------------------------------
# Win32 GDI & User32 Constants
# --------------------------------------------------------------------------

SRCCOPY: int = 0x00CC0020
CAPTUREBLT: int = 0x40000000
DIB_RGB_COLORS: int = 0
BI_RGB: int = 0

# System Metric Indices
SM_CXSCREEN: int = 0
SM_CYSCREEN: int = 1
SM_XVIRTUALSCREEN: int = 76
SM_YVIRTUALSCREEN: int = 77
SM_CXVIRTUALSCREEN: int = 78
SM_CYVIRTUALSCREEN: int = 79

# DWM Constants
DWMWA_EXTENDED_FRAME_BOUNDS: int = 9

# Monitor Flags
MONITORINFOF_PRIMARY: int = 0x00000001


# --------------------------------------------------------------------------
# Win32 Native Structures (ctypes)
# --------------------------------------------------------------------------


class _RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class _MONITORINFOEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", _RECT),
        ("rcWork", _RECT),
        ("dwFlags", wintypes.DWORD),
        ("szDevice", wintypes.WCHAR * 32),
    ]


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", ctypes.c_long),
        ("biHeight", ctypes.c_long),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", ctypes.c_long),
        ("biYPelsPerMeter", ctypes.c_long),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


# --------------------------------------------------------------------------
# DPI Awareness Helper
# --------------------------------------------------------------------------


def init_dpi_awareness() -> bool:
    """Safely configure process DPI awareness for accurate pixel coordinates.

    Tries Per-Monitor V2 without interfering with existing GUI contexts.
    Returns True if successfully set or already configured.
    """
    if platform.system() != "Windows":
        return False
    try:
        user32 = ctypes.windll.user32
        if hasattr(user32, "SetProcessDpiAwarenessContext"):
            # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
            context = ctypes.c_void_p(-4)
            res = user32.SetProcessDpiAwarenessContext(context)
            return bool(res)
    except Exception as exc:
        logger.debug("SetProcessDpiAwarenessContext ignored: %s", exc)
    return False


# --------------------------------------------------------------------------
# Capture Backend Interface
# --------------------------------------------------------------------------


class CaptureBackend(abc.ABC):
    """Abstract interface defining low-level desktop capture and monitor inspection."""

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True if backend is operational on the current platform."""

    @abc.abstractmethod
    def capture_rect(self, bounds: WindowBounds, source: str = "screen") -> ScreenCapture:
        """Capture an arbitrary desktop rectangular coordinate region."""

    @abc.abstractmethod
    def get_monitors(self) -> List[MonitorInfo]:
        """Enumerate connected physical/virtual display monitors."""

    @abc.abstractmethod
    def get_virtual_desktop_bounds(self) -> WindowBounds:
        """Retrieve total bounding box across all active displays."""

    @abc.abstractmethod
    def get_foreground_window(self) -> int:
        """Retrieve HWND of the current active foreground window."""

    @abc.abstractmethod
    def get_window_bounds(self, hwnd: int) -> Optional[WindowBounds]:
        """Retrieve bounding box of a specific window handle."""


# --------------------------------------------------------------------------
# Native Win32 GDI Capture Backend
# --------------------------------------------------------------------------


class Win32GdiCaptureBackend(CaptureBackend):
    """Native Windows desktop screen capture backend using Win32 GDI BitBlt and GetDIBits.

    All low-level Win32 calls are structured as private instance methods to enable
    surgical mocking during tests without touching real display hardware.
    """

    def __init__(self, logger_instance: Optional[logging.Logger] = None) -> None:
        self.logger = logger_instance or logger
        self._is_win = platform.system() == "Windows"
        if self._is_win:
            init_dpi_awareness()

    def is_available(self) -> bool:
        return self._is_win

    # ----------------------------------------------------------------------
    # Low-level Win32 API Wrappers (Mockable)
    # ----------------------------------------------------------------------

    def _api_get_desktop_dc(self) -> int:
        return int(ctypes.windll.user32.GetDC(0))

    def _api_release_dc(self, hwnd: int, hdc: int) -> int:
        return int(ctypes.windll.user32.ReleaseDC(wintypes.HWND(hwnd), wintypes.HDC(hdc)))

    def _api_create_compatible_dc(self, hdc: int) -> int:
        return int(ctypes.windll.gdi32.CreateCompatibleDC(wintypes.HDC(hdc)))

    def _api_delete_dc(self, hdc: int) -> bool:
        return bool(ctypes.windll.gdi32.DeleteDC(wintypes.HDC(hdc)))

    def _api_create_compatible_bitmap(self, hdc: int, width: int, height: int) -> int:
        return int(ctypes.windll.gdi32.CreateCompatibleBitmap(wintypes.HDC(hdc), width, height))

    def _api_select_object(self, hdc: int, hgdiobj: int) -> int:
        return int(ctypes.windll.gdi32.SelectObject(wintypes.HDC(hdc), wintypes.HGDIOBJ(hgdiobj)))

    def _api_delete_object(self, hgdiobj: int) -> bool:
        return bool(ctypes.windll.gdi32.DeleteObject(wintypes.HGDIOBJ(hgdiobj)))

    def _api_bit_blt(
        self,
        hdc_dest: int,
        x_dest: int,
        y_dest: int,
        width: int,
        height: int,
        hdc_src: int,
        x_src: int,
        y_src: int,
        rop: int,
    ) -> bool:
        return bool(
            ctypes.windll.gdi32.BitBlt(
                wintypes.HDC(hdc_dest),
                x_dest,
                y_dest,
                width,
                height,
                wintypes.HDC(hdc_src),
                x_src,
                y_src,
                rop,
            )
        )

    def _api_get_di_bits(
        self,
        hdc: int,
        hbmp: int,
        start_scan: int,
        scan_lines: int,
        buf: Any,
        bmi_ptr: Any,
        usage: int,
    ) -> int:
        return int(
            ctypes.windll.gdi32.GetDIBits(
                wintypes.HDC(hdc),
                wintypes.HBITMAP(hbmp),
                start_scan,
                scan_lines,
                buf,
                bmi_ptr,
                usage,
            )
        )

    def _api_get_system_metrics(self, n_index: int) -> int:
        return int(ctypes.windll.user32.GetSystemMetrics(n_index))

    def _api_get_foreground_window(self) -> int:
        return int(ctypes.windll.user32.GetForegroundWindow() or 0)

    def _api_is_window(self, hwnd: int) -> bool:
        return bool(ctypes.windll.user32.IsWindow(wintypes.HWND(hwnd)))

    def _api_get_window_rect(self, hwnd: int) -> Optional[Tuple[int, int, int, int]]:
        """Retrieve bounding rectangle for a window handle.

        Prefers DwmGetWindowAttribute with DWMWA_EXTENDED_FRAME_BOUNDS to retrieve
        the true visible window bounds without invisible Aero shadow padding.
        Falls back to GetWindowRect.
        """
        rect = _RECT()
        try:
            dwmapi = ctypes.windll.dwmapi
            res = dwmapi.DwmGetWindowAttribute(
                wintypes.HWND(hwnd),
                wintypes.DWORD(DWMWA_EXTENDED_FRAME_BOUNDS),
                ctypes.byref(rect),
                ctypes.sizeof(rect),
            )
            if res == 0:
                return (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))
        except Exception:
            pass

        try:
            if ctypes.windll.user32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(rect)):
                return (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))
        except Exception:
            pass

        return None

    # ----------------------------------------------------------------------
    # Core Implementations
    # ----------------------------------------------------------------------

    def capture_rect(self, bounds: WindowBounds, source: str = "screen") -> ScreenCapture:
        """Capture specified desktop rectangular coordinates via GDI BitBlt.

        Supports negative monitor coordinates (e.g. secondary monitor to the left
        or above the primary display).
        Guarantees zero GDI handle leaks via robust finally-block release.
        """
        if not self.is_available():
            raise UnsupportedPlatformError("Win32 GDI capture is only supported on Windows.")

        if bounds.width <= 0 or bounds.height <= 0:
            raise InvalidBoundsError(
                f"Cannot capture invalid bounds: width={bounds.width}, height={bounds.height}."
            )

        width = bounds.width
        height = bounds.height
        left = bounds.left
        top = bounds.top

        hdc_screen = 0
        hdc_mem = 0
        hbitmap = 0
        old_bmp = 0

        try:
            # 1. Acquire screen device context (DC) for whole virtual desktop
            hdc_screen = self._api_get_desktop_dc()
            if not hdc_screen:
                raise GdiResourceError("Failed to acquire Desktop DC (GetDC returned 0).")

            # 2. Create memory DC compatible with screen
            hdc_mem = self._api_create_compatible_dc(hdc_screen)
            if not hdc_mem:
                raise GdiResourceError("Failed to create compatible memory DC.")

            # 3. Create compatible bitmap of target dimensions
            hbitmap = self._api_create_compatible_bitmap(hdc_screen, width, height)
            if not hbitmap:
                raise GdiResourceError(f"Failed to create compatible bitmap ({width}x{height}).")

            # 4. Select bitmap into memory DC
            old_bmp = self._api_select_object(hdc_mem, hbitmap)
            if not old_bmp:
                raise GdiResourceError("Failed to select bitmap into memory DC.")

            # 5. BitBlt from screen DC (at virtual left, top) to memory DC (at 0, 0)
            rop = SRCCOPY | CAPTUREBLT
            success = self._api_bit_blt(
                hdc_mem, 0, 0, width, height,
                hdc_screen, left, top,
                rop,
            )
            if not success:
                raise GdiResourceError(
                    f"BitBlt failed to transfer screen region ({left}, {top}, {width}x{height})."
                )

            # 6. Prepare BITMAPINFO for top-down 32-bit BGRA extraction
            bmi = _BITMAPINFOHEADER()
            bmi.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
            bmi.biWidth = width
            bmi.biHeight = -height  # Negative indicates top-down scanlines
            bmi.biPlanes = 1
            bmi.biBitCount = 32
            bmi.biCompression = BI_RGB
            bmi.biSizeImage = width * height * 4

            buf_size = width * height * 4
            buffer = ctypes.create_string_buffer(buf_size)

            lines = self._api_get_di_bits(
                hdc_mem,
                hbitmap,
                0,
                height,
                buffer,
                ctypes.byref(bmi),
                DIB_RGB_COLORS,
            )
            if lines != height:
                raise GdiResourceError(
                    f"GetDIBits failed to copy all scanlines: read {lines}/{height} lines."
                )

            raw_bytes = bytes(buffer.raw)

            return ScreenCapture(
                raw_data=raw_bytes,
                width=width,
                height=height,
                channels=4,
                pixel_format="BGRA",
                stride=width * 4,
                timestamp=time.time(),
                source=source,
                bounds=bounds,
                metadata={"captured_at": time.time(), "rop": rop},
            )

        finally:
            # Strict GDI Resource Cleanup
            if hdc_mem and old_bmp:
                try:
                    self._api_select_object(hdc_mem, old_bmp)
                except Exception:
                    pass
            if hbitmap:
                try:
                    self._api_delete_object(hbitmap)
                except Exception:
                    pass
            if hdc_mem:
                try:
                    self._api_delete_dc(hdc_mem)
                except Exception:
                    pass
            if hdc_screen:
                try:
                    self._api_release_dc(0, hdc_screen)
                except Exception:
                    pass

    def get_monitors(self) -> List[MonitorInfo]:
        """Enumerate display monitors using Win32 EnumDisplayMonitors."""
        if not self.is_available():
            return []

        user32 = ctypes.windll.user32
        monitors: List[MonitorInfo] = []

        def _enum_proc(hmon: Any, _hdc: Any, _lprc: Any, _lparam: Any) -> bool:
            try:
                mi = _MONITORINFOEXW()
                mi.cbSize = ctypes.sizeof(_MONITORINFOEXW)
                if user32.GetMonitorInfoW(hmon, ctypes.byref(mi)):
                    h_val = int(hmon)
                    name = str(mi.szDevice).strip() or f"DISPLAY_{len(monitors) + 1}"
                    bounds = WindowBounds(
                        left=int(mi.rcMonitor.left),
                        top=int(mi.rcMonitor.top),
                        right=int(mi.rcMonitor.right),
                        bottom=int(mi.rcMonitor.bottom),
                    )
                    work = WindowBounds(
                        left=int(mi.rcWork.left),
                        top=int(mi.rcWork.top),
                        right=int(mi.rcWork.right),
                        bottom=int(mi.rcWork.bottom),
                    )
                    is_primary = bool(mi.dwFlags & MONITORINFOF_PRIMARY)
                    monitors.append(
                        MonitorInfo(
                            handle=h_val,
                            name=name,
                            bounds=bounds,
                            work_area=work,
                            is_primary=is_primary,
                        )
                    )
            except Exception as exc:
                self.logger.debug("Monitor enumeration error on handle %s: %s", hmon, exc)
            return True

        monitor_proc_type = ctypes.WINFUNCTYPE(
            ctypes.c_bool, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(_RECT), wintypes.LPARAM
        )
        proc_instance = monitor_proc_type(_enum_proc)

        try:
            user32.EnumDisplayMonitors(None, None, proc_instance, 0)
        except Exception as exc:
            self.logger.warning("EnumDisplayMonitors failed: %s", exc)

        # Fallback if no monitors returned: create single monitor from primary metrics
        if not monitors:
            vw = self._api_get_system_metrics(SM_CXSCREEN) or 1920
            vh = self._api_get_system_metrics(SM_CYSCREEN) or 1080
            monitors.append(
                MonitorInfo(
                    handle=1,
                    name="DISPLAY1",
                    bounds=WindowBounds(left=0, top=0, right=vw, bottom=vh),
                    is_primary=True,
                )
            )

        return monitors

    def get_virtual_desktop_bounds(self) -> WindowBounds:
        """Return the bounding rectangle of the total virtual desktop."""
        if not self.is_available():
            return WindowBounds(left=0, top=0, right=1920, bottom=1080)

        vx = self._api_get_system_metrics(SM_XVIRTUALSCREEN)
        vy = self._api_get_system_metrics(SM_YVIRTUALSCREEN)
        vw = self._api_get_system_metrics(SM_CXVIRTUALSCREEN)
        vh = self._api_get_system_metrics(SM_CYVIRTUALSCREEN)

        if vw <= 0 or vh <= 0:
            vw = self._api_get_system_metrics(SM_CXSCREEN) or 1920
            vh = self._api_get_system_metrics(SM_CYSCREEN) or 1080
            vx = 0
            vy = 0

        return WindowBounds(left=vx, top=vy, right=vx + vw, bottom=vy + vh)

    def get_foreground_window(self) -> int:
        if not self.is_available():
            return 0
        return self._api_get_foreground_window()

    def get_window_bounds(self, hwnd: int) -> Optional[WindowBounds]:
        if not self.is_available() or not self._api_is_window(hwnd):
            return None
        rect = self._api_get_window_rect(hwnd)
        if rect is None:
            return None
        return WindowBounds(left=rect[0], top=rect[1], right=rect[2], bottom=rect[3])


# --------------------------------------------------------------------------
# Mock Capture Backend (Hardware-Independent Testing)
# --------------------------------------------------------------------------


class MockCaptureBackend(CaptureBackend):
    """Deterministic, hardware-independent mock capture backend.

    Generates synthetic frames without interacting with real desktop hardware or Windows APIs.
    """

    def __init__(
        self,
        monitors: Optional[List[MonitorInfo]] = None,
        virtual_bounds: Optional[WindowBounds] = None,
        default_color: Tuple[int, int, int, int] = (128, 200, 255, 255),
    ) -> None:
        self.default_color = default_color
        self.monitors = monitors if monitors is not None else [
            MonitorInfo(
                handle=1,
                name="MOCK_DISPLAY_1",
                bounds=WindowBounds(left=0, top=0, right=1920, bottom=1080),
                work_area=WindowBounds(left=0, top=0, right=1920, bottom=1040),
                is_primary=True,
            )
        ]
        self.virtual_bounds = virtual_bounds or WindowBounds(left=0, top=0, right=1920, bottom=1080)
        self.active_hwnd: int = 12345
        self.window_bounds_map: Dict[int, WindowBounds] = {
            12345: WindowBounds(left=100, top=100, right=900, bottom=700),
        }
        self.capture_count: int = 0
        self.fail_on_capture: bool = False

    def is_available(self) -> bool:
        return True

    def capture_rect(self, bounds: WindowBounds, source: str = "screen") -> ScreenCapture:
        if self.fail_on_capture:
            raise GdiResourceError("Simulated GDI capture failure.")

        if bounds.width <= 0 or bounds.height <= 0:
            raise InvalidBoundsError(f"Invalid bounds: width={bounds.width}, height={bounds.height}")

        self.capture_count += 1
        width = bounds.width
        height = bounds.height

        # Generate synthetic BGRA pixel pattern
        b, g, r, a = self.default_color
        row = bytes([b, g, r, a]) * width
        raw_data = row * height

        return ScreenCapture(
            raw_data=raw_data,
            width=width,
            height=height,
            channels=4,
            pixel_format="BGRA",
            stride=width * 4,
            timestamp=time.time(),
            source=source,
            bounds=bounds,
            metadata={"mock": True, "count": self.capture_count},
        )

    def get_monitors(self) -> List[MonitorInfo]:
        return list(self.monitors)

    def get_virtual_desktop_bounds(self) -> WindowBounds:
        return self.virtual_bounds

    def get_foreground_window(self) -> int:
        return self.active_hwnd

    def get_window_bounds(self, hwnd: int) -> Optional[WindowBounds]:
        return self.window_bounds_map.get(hwnd)


# --------------------------------------------------------------------------
# High-Level Desktop Capture Engine
# --------------------------------------------------------------------------


class DesktopCaptureEngine:
    """Thread-safe, on-demand desktop and window capture facade.

    Features:
    - Primary display, arbitrary rectangle, multi-monitor, and active-window capture.
    - Automatic fallback to mock backend in non-Windows or test environments.
    - DI integration with ServiceContainer and WindowSkills.
    - Asynchronous non-blocking wrappers.
    - Clean resource management with zero disk persistence.
    """

    def __init__(
        self,
        backend: Optional[CaptureBackend] = None,
        container_instance: Optional[ServiceContainer] = None,
        event_bus_instance: Optional[EventBus] = None,
        logger_instance: Optional[logging.Logger] = None,
        *,
        auto_register_in_container: bool = True,
    ) -> None:
        self._lock = threading.RLock()
        self._logger = logger_instance or logger
        self._container = container_instance if container_instance is not None else default_container
        self._event_bus = event_bus_instance if event_bus_instance is not None else default_event_bus

        # Initialize Backend
        if backend is not None:
            self._backend = backend
        elif platform.system() == "Windows":
            self._backend = Win32GdiCaptureBackend(logger_instance=self._logger)
        else:
            self._backend = MockCaptureBackend()

        if auto_register_in_container and self._container is not None:
            try:
                self._container.register_singleton(
                    "desktop_capture_engine", self, allow_override=True
                )
                self._logger.debug("Registered 'desktop_capture_engine' singleton in ServiceContainer.")
            except Exception as exc:
                self._logger.warning("Could not register capture engine in container: %s", exc)

    @property
    def backend(self) -> CaptureBackend:
        """Return active capture backend."""
        return self._backend

    @property
    def is_available(self) -> bool:
        """Check if capture backend is operational."""
        return self._backend.is_available()

    # ----------------------------------------------------------------------
    # WindowSkills Integration Helper
    # ----------------------------------------------------------------------

    def _resolve_window_skills(self) -> Optional[Any]:
        """Resolve WindowSkills from container if available."""
        if self._container is not None and self._container.exists("window_skills"):
            return self._container.resolve("window_skills")
        if self._container is not None and self._container.exists("skills"):
            try:
                sm = self._container.resolve("skills")
                if hasattr(sm, "get_skill"):
                    return sm.get_skill("window")
            except Exception:
                pass
        return None

    # ----------------------------------------------------------------------
    # Display & Monitor Telemetry
    # ----------------------------------------------------------------------

    def get_monitors(self) -> List[MonitorInfo]:
        """Enumerate all connected display monitors."""
        with self._lock:
            return self._backend.get_monitors()

    def get_primary_monitor(self) -> Optional[MonitorInfo]:
        """Return primary monitor metadata, or first monitor if primary flag unset."""
        monitors = self.get_monitors()
        for m in monitors:
            if m.is_primary:
                return m
        return monitors[0] if monitors else None

    def get_virtual_desktop_bounds(self) -> WindowBounds:
        """Retrieve total bounding box spanning all active displays."""
        with self._lock:
            return self._backend.get_virtual_desktop_bounds()

    # ----------------------------------------------------------------------
    # Window Bounds Inspection
    # ----------------------------------------------------------------------

    def get_active_window_handle(self) -> int:
        """Return HWND of the current active foreground window."""
        with self._lock:
            # Check WindowSkills first if registered
            ws = self._resolve_window_skills()
            if ws and hasattr(ws, "_api_get_foreground_window"):
                try:
                    return int(ws._api_get_foreground_window())
                except Exception:
                    pass
            return self._backend.get_foreground_window()

    def get_window_bounds(self, hwnd: int) -> Optional[WindowBounds]:
        """Retrieve screen coordinates for specified window handle."""
        with self._lock:
            ws = self._resolve_window_skills()
            if ws and hasattr(ws, "get_window_bounds"):
                try:
                    bounds = ws.get_window_bounds(hwnd)
                    if isinstance(bounds, WindowBounds):
                        return bounds
                except Exception:
                    pass
            return self._backend.get_window_bounds(hwnd)

    def get_active_window_bounds(self) -> Optional[WindowBounds]:
        """Retrieve screen coordinates of currently active foreground window."""
        hwnd = self.get_active_window_handle()
        if not hwnd:
            return None
        return self.get_window_bounds(hwnd)

    # ----------------------------------------------------------------------
    # Synchronous Capture Methods
    # ----------------------------------------------------------------------

    def capture_screen(
        self,
        bounds: Optional[WindowBounds] = None,
        source: str = "screen",
    ) -> ScreenCapture:
        """Capture an arbitrary rectangular region of the desktop.

        If bounds is None, defaults to the primary monitor.
        """
        with self._lock:
            target_bounds = bounds
            if target_bounds is None:
                primary = self.get_primary_monitor()
                if primary:
                    target_bounds = primary.bounds
                else:
                    target_bounds = self.get_virtual_desktop_bounds()

            if target_bounds.is_empty:
                raise InvalidBoundsError(f"Target bounds {target_bounds} are empty.")

            capture = self._backend.capture_rect(target_bounds, source=source)
            self._notify_captured(capture)
            return capture

    def capture_monitor(self, monitor_index: int = 0) -> ScreenCapture:
        """Capture the screen of a specific display monitor by index."""
        with self._lock:
            monitors = self.get_monitors()
            if not monitors:
                raise CaptureError("No display monitors detected.")
            if monitor_index < 0 or monitor_index >= len(monitors):
                raise CaptureError(
                    f"Invalid monitor index {monitor_index}. Available count: {len(monitors)}."
                )
            target = monitors[monitor_index]
            return self.capture_screen(target.bounds, source=f"monitor_{monitor_index}")

    def capture_virtual_desktop(self) -> ScreenCapture:
        """Capture the entire multi-monitor virtual desktop."""
        with self._lock:
            v_bounds = self.get_virtual_desktop_bounds()
            return self.capture_screen(v_bounds, source="virtual_desktop")

    def capture_window(self, hwnd: int) -> ScreenCapture:
        """Capture the visible region of a specific window handle."""
        with self._lock:
            bounds = self.get_window_bounds(hwnd)
            if bounds is None or bounds.is_empty:
                raise CaptureError(
                    f"Cannot capture window {hwnd}: window not found or is minimized/hidden."
                )

            capture = self.capture_screen(bounds, source="window")
            capture.metadata["hwnd"] = hwnd
            return capture

    def capture_active_window(self) -> ScreenCapture:
        """Capture the current active foreground window."""
        with self._lock:
            hwnd = self.get_active_window_handle()
            if not hwnd:
                raise CaptureError("No active foreground window found.")
            return self.capture_window(hwnd)

    # ----------------------------------------------------------------------
    # Asynchronous Wrappers (Non-blocking)
    # ----------------------------------------------------------------------

    async def capture_screen_async(
        self,
        bounds: Optional[WindowBounds] = None,
        source: str = "screen",
    ) -> ScreenCapture:
        """Asynchronously capture screen region on a worker thread."""
        return await asyncio.to_thread(self.capture_screen, bounds, source)

    async def capture_active_window_async(self) -> ScreenCapture:
        """Asynchronously capture active foreground window on a worker thread."""
        return await asyncio.to_thread(self.capture_active_window)

    async def capture_virtual_desktop_async(self) -> ScreenCapture:
        """Asynchronously capture entire virtual desktop on a worker thread."""
        return await asyncio.to_thread(self.capture_virtual_desktop)

    # ----------------------------------------------------------------------
    # Event Notification
    # ----------------------------------------------------------------------

    def _notify_captured(self, capture: ScreenCapture) -> None:
        """Emit telemetry event on event bus if configured."""
        if self._event_bus is not None and hasattr(self._event_bus, "publish"):
            try:
                self._event_bus.publish(
                    "vision.screen_captured",
                    {
                        "source": capture.source,
                        "width": capture.width,
                        "height": capture.height,
                        "size_bytes": capture.size_bytes,
                        "timestamp": capture.timestamp,
                    },
                )
            except Exception:
                pass


__all__ = [
    "CaptureBackend",
    "DesktopCaptureEngine",
    "MockCaptureBackend",
    "Win32GdiCaptureBackend",
    "init_dpi_awareness",
]
