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
    OCRResult,
    OCRTextBlock,
    Point,
    ScreenCapture,
    ScreenObservation,
    UnsupportedPlatformError,
    VisionError,
    VisionSecurityError,
    VisualAnalysisResult,
    WindowBounds,
)
from app.vision.ocr import (
    MockOCRProvider,
    MultimodalVisionOCRAdapter,
    OCRProvider,
    WindowsMediaOCRProvider,
)
from app.vision.preprocessing import (
    ImagePreprocessor,
    capture_to_image_part,
    default_preprocessor,
    preprocess_capture,
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
    "ImagePreprocessor",
    "InvalidBoundsError",
    "MockCaptureBackend",
    "MockOCRProvider",
    "MonitorInfo",
    "MultimodalVisionOCRAdapter",
    "OCRProvider",
    "OCRResult",
    "OCRTextBlock",
    "Point",
    "ScreenCapture",
    "ScreenObservation",
    "SecureVisionManager",
    "SensitiveWindowRule",
    "UnsupportedPlatformError",
    "VisionError",
    "VisionSecurityError",
    "VisionSecurityPolicy",
    "VisualAnalysisResult",
    "Win32GdiCaptureBackend",
    "WindowBounds",
    "WindowsMediaOCRProvider",
    "capture_to_image_part",
    "default_preprocessor",
    "init_dpi_awareness",
    "preprocess_capture",
]
