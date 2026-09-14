"""Data models and structures for the J.A.R.V.I.S Vision subsystem.

Provides typed, immutable, and serializable representations for:
- 2D Screen coordinates (Point)
- Rectangular window and screen regions (WindowBounds)
- Physical and virtual display metadata (MonitorInfo)
- In-memory screen capture buffers and metadata (ScreenCapture)
- Subsystem exception hierarchy (VisionError, CaptureError, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import struct
import time
from typing import Any, Dict, Optional, Tuple, Union
import uuid
import zlib

from app.core.container import JarvisException


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class VisionError(JarvisException):
    """Base exception for all vision subsystem errors."""


class CaptureError(VisionError):
    """Raised when screen or window capture fails."""


class InvalidBoundsError(CaptureError):
    """Raised when capture dimensions or bounds are zero, negative, or invalid."""


class GdiResourceError(CaptureError):
    """Raised when Win32 GDI resource allocation (DC, bitmap, memory) fails."""


class UnsupportedPlatformError(VisionError):
    """Raised when desktop vision capabilities are invoked on an unsupported OS."""


class VisionSecurityError(VisionError):
    """Base exception for vision privacy and security policy violations."""


class CaptureBlockedError(VisionSecurityError):
    """Raised when a screen capture request is blocked by vision security guardrails."""

    def __init__(self, message: str, authorization: Optional[Any] = None) -> None:
        super().__init__(message)
        self.authorization = authorization


class BufferExpiredError(VisionError):
    """Raised when attempting to access an expired ephemeral screen observation."""


# --------------------------------------------------------------------------
# Coordinate and Geometric Models
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Point:
    """Represents an integer 2D coordinate on the desktop or virtual screen."""

    x: int
    y: int

    def to_tuple(self) -> Tuple[int, int]:
        """Return coordinate as (x, y) tuple."""
        return (self.x, self.y)

    def to_dict(self) -> Dict[str, int]:
        """Serialize Point to dictionary."""
        return {"x": self.x, "y": self.y}


@dataclass(frozen=True)
class WindowBounds:
    """Represents a rectangular region on a display, window, or virtual desktop.

    Coordinates follow standard screen convention:
    - left: X-coordinate of left edge (can be negative in multi-monitor layouts)
    - top: Y-coordinate of top edge (can be negative in multi-monitor layouts)
    - right: X-coordinate of right edge
    - bottom: Y-coordinate of bottom edge
    """

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        """Horizontal dimension in pixels."""
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        """Vertical dimension in pixels."""
        return max(0, self.bottom - self.top)

    @property
    def is_empty(self) -> bool:
        """Return True if area is zero or negative."""
        return self.width <= 0 or self.height <= 0

    @property
    def area(self) -> int:
        """Area in square pixels."""
        return self.width * self.height

    @property
    def center(self) -> Point:
        """Center coordinate of the rectangle."""
        return Point(
            x=self.left + self.width // 2,
            y=self.top + self.height // 2,
        )

    def contains_point(self, point: Union[Point, Tuple[int, int]]) -> bool:
        """Return True if the point is strictly inside the bounds."""
        px = point.x if isinstance(point, Point) else point[0]
        py = point.y if isinstance(point, Point) else point[1]
        return (self.left <= px < self.right) and (self.top <= py < self.bottom)

    def intersects(self, other: WindowBounds) -> bool:
        """Return True if this bounding box intersects with another."""
        return not (
            self.right <= other.left
            or self.left >= other.right
            or self.bottom <= other.top
            or self.top >= other.bottom
        )

    def intersection(self, other: WindowBounds) -> Optional[WindowBounds]:
        """Return overlapping rectangle if intersecting, or None."""
        if not self.intersects(other):
            return None
        return WindowBounds(
            left=max(self.left, other.left),
            top=max(self.top, other.top),
            right=min(self.right, other.right),
            bottom=min(self.bottom, other.bottom),
        )

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Return bounds as (left, top, right, bottom) tuple."""
        return (self.left, self.top, self.right, self.bottom)

    def to_dict(self) -> Dict[str, int]:
        """Serialize WindowBounds to dictionary."""
        return {
            "left": self.left,
            "top": self.top,
            "right": self.right,
            "bottom": self.bottom,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_xywh(cls, x: int, y: int, width: int, height: int) -> WindowBounds:
        """Construct WindowBounds from top-left (x, y) and dimensions (width, height)."""
        return cls(left=x, top=y, right=x + width, bottom=y + height)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WindowBounds:
        """Construct WindowBounds from dictionary."""
        if "width" in data and "height" in data and "right" not in data:
            return cls.from_xywh(
                x=int(data.get("left", data.get("x", 0))),
                y=int(data.get("top", data.get("y", 0))),
                width=int(data["width"]),
                height=int(data["height"]),
            )
        return cls(
            left=int(data.get("left", 0)),
            top=int(data.get("top", 0)),
            right=int(data.get("right", 0)),
            bottom=int(data.get("bottom", 0)),
        )


# --------------------------------------------------------------------------
# Display / Monitor Telemetry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MonitorInfo:
    """Metadata describing a physical or virtual display monitor."""

    handle: int
    name: str
    bounds: WindowBounds
    work_area: Optional[WindowBounds] = None
    is_primary: bool = False
    device_pixel_ratio: float = 1.0

    @property
    def width(self) -> int:
        """Monitor horizontal resolution."""
        return self.bounds.width

    @property
    def height(self) -> int:
        """Monitor vertical resolution."""
        return self.bounds.height

    def to_dict(self) -> Dict[str, Any]:
        """Serialize MonitorInfo to dictionary."""
        return {
            "handle": self.handle,
            "name": self.name,
            "bounds": self.bounds.to_dict(),
            "work_area": self.work_area.to_dict() if self.work_area else None,
            "is_primary": self.is_primary,
            "device_pixel_ratio": self.device_pixel_ratio,
            "width": self.width,
            "height": self.height,
        }


# --------------------------------------------------------------------------
# In-Memory Screen Capture Buffer
# --------------------------------------------------------------------------


def _encode_raw_rgba_to_png(raw_rgba: bytes, width: int, height: int) -> bytes:
    """Encode raw RGBA bytes into standard PNG format entirely in memory.

    Zero-dependency implementation using standard library struct and zlib.
    Never touches disk.
    """
    if width <= 0 or height <= 0:
        return b""

    # PNG Signature: 89 50 4E 47 0D 0A 1A 0A
    signature = b"\x89PNG\r\n\x1a\n"

    # IHDR chunk (Width, Height, Bit depth=8, ColorType=6 (RGBA), Comp=0, Filter=0, Interlace=0)
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    ihdr_crc = struct.pack(">I", zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF)
    ihdr_chunk = struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data + ihdr_crc

    # IDAT chunk: filter byte 0 (None) prefixed to each scanline
    stride = width * 4
    scanlines = bytearray(height * (stride + 1))
    in_offset = 0
    out_offset = 0
    for _ in range(height):
        scanlines[out_offset] = 0  # Filter byte None
        out_offset += 1
        scanlines[out_offset : out_offset + stride] = raw_rgba[in_offset : in_offset + stride]
        out_offset += stride
        in_offset += stride

    compressed = zlib.compress(bytes(scanlines), level=6)
    idat_crc = struct.pack(">I", zlib.crc32(b"IDAT" + compressed) & 0xFFFFFFFF)
    idat_chunk = struct.pack(">I", len(compressed)) + b"IDAT" + compressed + idat_crc

    # IEND chunk
    iend_crc = struct.pack(">I", zlib.crc32(b"IEND") & 0xFFFFFFFF)
    iend_chunk = struct.pack(">I", 0) + b"IEND" + iend_crc

    return signature + ihdr_chunk + idat_chunk + iend_chunk


@dataclass
class ScreenCapture:
    """Represents an uncompressed, in-memory screen or window capture.

    Attributes:
        raw_data: Uncompressed pixel byte buffer (held strictly in RAM).
        width: Frame width in pixels.
        height: Frame height in pixels.
        channels: Number of color channels (default: 4).
        pixel_format: Pixel order format, e.g. 'BGRA' or 'RGBA' (default: 'BGRA').
        stride: Byte length of each row (default: width * channels).
        timestamp: Unix epoch timestamp marking capture moment.
        source: Provenance label ('screen', 'window', 'region', 'monitor').
        bounds: Screen region coordinates represented by this frame.
        metadata: Arbitrary capture telemetry (HWND, window title, PID, scale, etc.).
    """

    raw_data: bytes
    width: int
    height: int
    channels: int = 4
    pixel_format: str = "BGRA"
    stride: int = 0
    timestamp: float = field(default_factory=time.time)
    source: str = "screen"
    bounds: Optional[WindowBounds] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize stride and validate basic geometry."""
        if self.stride <= 0:
            self.stride = self.width * self.channels

    @property
    def size_bytes(self) -> int:
        """Length of internal raw byte buffer."""
        return len(self.raw_data)

    @property
    def is_empty(self) -> bool:
        """Check if frame contains zero pixels or empty buffer."""
        return self.width <= 0 or self.height <= 0 or len(self.raw_data) == 0

    def to_rgba(self) -> bytes:
        """Return raw pixel bytes converted to RGBA order.

        If already RGBA, returns raw_data directly.
        Performs in-memory channel swap without disk persistence.
        """
        if self.pixel_format.upper() == "RGBA":
            return self.raw_data

        if self.pixel_format.upper() == "BGRA":
            # Fast vectorized conversion if numpy is available
            try:
                import numpy as np

                arr = np.frombuffer(self.raw_data, dtype=np.uint8)
                if arr.size == self.width * self.height * 4:
                    reshaped = arr.reshape((self.height, self.width, 4))
                    return reshaped[:, :, [2, 1, 0, 3]].tobytes()
            except ImportError:
                pass

            # Standard Python fallback
            src = bytearray(self.raw_data)
            for i in range(0, len(src), 4):
                src[i], src[i + 2] = src[i + 2], src[i]
            return bytes(src)

        return self.raw_data

    def to_png_bytes(self) -> bytes:
        """Encode frame to standard PNG format strictly in memory.

        No files are created on disk. The returned bytes can be directly passed
        to OCR engines or multimodal AI payloads.
        """
        if self.is_empty:
            return b""
        rgba_bytes = self.to_rgba()
        return _encode_raw_rgba_to_png(rgba_bytes, self.width, self.height)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize capture metadata to dictionary.

        Omit raw byte array to prevent telemetry/log buffer bloat.
        """
        return {
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "pixel_format": self.pixel_format,
            "size_bytes": self.size_bytes,
            "timestamp": self.timestamp,
            "source": self.source,
            "bounds": self.bounds.to_dict() if self.bounds else None,
            "metadata": dict(self.metadata),
        }


# --------------------------------------------------------------------------
# Security & Observation Models (Phase 27.2)
# --------------------------------------------------------------------------


class CaptureDecision(str, Enum):
    """Authorization verdict for screen/window capture requests."""

    ALLOW = "ALLOW"
    BLOCK = "BLOCK"


class CaptureCategory(str, Enum):
    """Classification category explaining capture authorization outcomes."""

    SAFE = "safe"
    SENSITIVE_APPLICATION = "sensitive_application"
    CREDENTIAL_INTERFACE = "credential_interface"
    PRIVATE_BROWSING = "private_browsing"
    INVALID_CONTEXT = "invalid_context"
    POLICY_RESTRICTION = "policy_restriction"


@dataclass(frozen=True)
class CaptureAuthorization:
    """Represents a deterministic authorization verdict for a capture request."""

    decision: CaptureDecision
    category: CaptureCategory
    reason: str
    rule_name: Optional[str] = None
    target: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    @property
    def is_allowed(self) -> bool:
        """Return True if capture is permitted."""
        return self.decision == CaptureDecision.ALLOW

    def to_dict(self) -> Dict[str, Any]:
        """Serialize authorization to dictionary."""
        return {
            "decision": self.decision.value,
            "category": self.category.value,
            "reason": self.reason,
            "rule_name": self.rule_name,
            "target": self.target,
            "timestamp": self.timestamp,
            "is_allowed": self.is_allowed,
        }


@dataclass
class ScreenObservation:
    """Represents a secure, ephemeral in-memory observation of screen content.

    Attributes:
        id: Unique tracking UUID.
        capture: Underlying ScreenCapture frame buffer (in RAM only), or None if blocked.
        authorization: Security authorization verdict governing this observation.
        source: Capture provenance ('screen', 'window', 'region', 'monitor').
        timestamp: Unix epoch timestamp when observation was generated.
        expires_at: Epoch timestamp when this observation expires from memory (TTL).
        metadata: Arbitrary contextual telemetry (window title, process, etc.).
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    capture: Optional[ScreenCapture] = None
    authorization: Optional[CaptureAuthorization] = None
    source: str = "screen"
    timestamp: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Return True if observation lifetime has lapsed."""
        if self.expires_at is None:
            return False
        return time.time() >= self.expires_at

    @property
    def is_valid(self) -> bool:
        """Return True if observation has not expired and holds a valid capture."""
        return not self.is_expired and self.capture is not None and not self.capture.is_empty

    def to_dict(self) -> Dict[str, Any]:
        """Serialize observation metadata (omitting raw image bytes)."""
        return {
            "id": self.id,
            "source": self.source,
            "timestamp": self.timestamp,
            "expires_at": self.expires_at,
            "is_expired": self.is_expired,
            "is_valid": self.is_valid,
            "authorization": self.authorization.to_dict() if self.authorization else None,
            "capture": self.capture.to_dict() if self.capture else None,
            "metadata": dict(self.metadata),
        }


# --------------------------------------------------------------------------
# Structured OCR and Visual Analysis Models (Phase 27.3)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OCRTextBlock:
    """Represents an individual recognized block, line, or span of text.

    Attributes:
        text: Extracted plain text string.
        confidence: Normalized confidence value [0.0, 1.0].
        bounds: Optional desktop bounding rectangle coordinates for the text.
    """

    text: str
    confidence: float = 1.0
    bounds: Optional[WindowBounds] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize text block to dictionary."""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bounds": self.bounds.to_dict() if self.bounds else None,
        }


@dataclass(frozen=True)
class OCRResult:
    """Structured result from an optical character recognition operation.

    Attributes:
        text: Full consolidated text extracted from the frame.
        blocks: Sequence of granular text blocks with bounding coordinates.
        language: BCP-47 language tag (e.g. 'en', 'hi').
        duration: Processing wall-clock duration in seconds.
    """

    text: str = ""
    blocks: Tuple[OCRTextBlock, ...] = field(default_factory=tuple)
    language: str = "en"
    duration: float = 0.0

    @property
    def is_empty(self) -> bool:
        """Return True if no text was recognized."""
        return not self.text.strip() and len(self.blocks) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize OCR result to dictionary."""
        return {
            "text": self.text,
            "blocks": [b.to_dict() for b in self.blocks],
            "language": self.language,
            "duration": self.duration,
            "is_empty": self.is_empty,
        }


@dataclass(frozen=True)
class VisualAnalysisResult:
    """Structured output from multimodal image analysis and visual understanding.

    Attributes:
        summary: Concise natural-language visual description or analysis.
        extracted_text: Optional plain text extracted from the visual target.
        ocr_result: Optional granular OCR result if OCR was performed.
        detected_elements: Tuple of identified visual UI elements, windows, or artifacts.
        observation_id: UUID string of the corresponding ScreenObservation.
        duration: Total execution duration in seconds.
        metadata: Privacy-safe operational metadata (never containing raw image bytes).
    """

    summary: str
    extracted_text: Optional[str] = None
    ocr_result: Optional[OCRResult] = None
    detected_elements: Tuple[str, ...] = field(default_factory=tuple)
    observation_id: str = ""
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize visual analysis result to dictionary."""
        return {
            "summary": self.summary,
            "extracted_text": self.extracted_text,
            "ocr_result": self.ocr_result.to_dict() if self.ocr_result else None,
            "detected_elements": list(self.detected_elements),
            "observation_id": self.observation_id,
            "duration": self.duration,
            "metadata": dict(self.metadata),
        }


__all__ = [
    "BufferExpiredError",
    "CaptureAuthorization",
    "CaptureBlockedError",
    "CaptureCategory",
    "CaptureDecision",
    "CaptureError",
    "GdiResourceError",
    "InvalidBoundsError",
    "MonitorInfo",
    "OCRResult",
    "OCRTextBlock",
    "Point",
    "ScreenCapture",
    "ScreenObservation",
    "UnsupportedPlatformError",
    "VisionError",
    "VisionSecurityError",
    "VisualAnalysisResult",
    "WindowBounds",
]
