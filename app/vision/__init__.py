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
    BufferExpiredError,
    CaptureAuthorization,
    CaptureBlockedError,
    CaptureCategory,
    CaptureDecision,
    CaptureError,
    GdiResourceError,
    InvalidBoundsError,
    MonitorInfo,
    Point,
    ScreenCapture,
    ScreenObservation,
    UnsupportedPlatformError,
    VisionError,
    VisionSecurityError,
    WindowBounds,
)
from app.vision.security import (
    EphemeralBufferManager,
    SecureVisionManager,
    SensitiveWindowRule,
    VisionSecurityPolicy,
)

__all__ = [
    "BufferExpiredError",
    "CaptureAuthorization",
    "CaptureBackend",
    "CaptureBlockedError",
    "CaptureCategory",
    "CaptureDecision",
    "CaptureError",
    "DesktopCaptureEngine",
    "EphemeralBufferManager",
    "GdiResourceError",
    "InvalidBoundsError",
    "MockCaptureBackend",
    "MonitorInfo",
    "Point",
    "ScreenCapture",
    "ScreenObservation",
    "SecureVisionManager",
    "SensitiveWindowRule",
    "UnsupportedPlatformError",
    "VisionError",
    "VisionSecurityError",
    "VisionSecurityPolicy",
    "Win32GdiCaptureBackend",
    "WindowBounds",
    "init_dpi_awareness",
]
