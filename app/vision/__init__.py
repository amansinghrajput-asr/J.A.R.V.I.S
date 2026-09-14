"""Vision and Screen Intelligence Package for J.A.R.V.I.S. Phase 27.

Provides native Windows desktop screen capture, multi-monitor topology inspection,
active-window tracking, and typed in-memory capture models.
"""

from __future__ import annotations

from app.vision.capture import (
    CaptureBackend,
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
    VisionError,
    WindowBounds,
)

__all__ = [
    "CaptureBackend",
    "CaptureError",
    "DesktopCaptureEngine",
    "GdiResourceError",
    "InvalidBoundsError",
    "MockCaptureBackend",
    "MonitorInfo",
    "Point",
    "ScreenCapture",
    "UnsupportedPlatformError",
    "VisionError",
    "Win32GdiCaptureBackend",
    "WindowBounds",
    "init_dpi_awareness",
]
