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
    def observation_id(self) -> str:
        """Alias for id attribute."""
        return self.id

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

    @property
    def bounds(self) -> Optional[WindowBounds]:
        """Return capture bounding box coordinates if capture exists."""
        return self.capture.bounds if self.capture else None

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



# --------------------------------------------------------------------------
# Visual Grounding & Element Localization Models (Phase 27.8)
# --------------------------------------------------------------------------


class UIElementType(str, Enum):
    """Semantic classification of visual UI elements."""

    BUTTON = "button"
    INPUT = "input"
    TEXT = "text"
    ICON = "icon"
    LINK = "link"
    CHECKBOX = "checkbox"
    DROPDOWN = "dropdown"
    MENU = "menu"
    TAB = "tab"
    DIALOG = "dialog"
    CONTAINER = "container"
    UNKNOWN = "unknown"


class GroundingSource(str, Enum):
    """Identifies the underlying engine that resolved the element coordinates."""

    OCR_EXACT = "ocr_exact"
    MULTIMODAL_SEMANTIC = "multimodal_semantic"
    HYBRID_FUSED = "hybrid_fused"
    UNKNOWN = "unknown"


class SpatialRelation(str, Enum):
    """Spatial relationship between a target element and a reference anchor."""

    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    ABOVE = "above"
    BELOW = "below"
    NEAR = "near"
    INSIDE = "inside"


@dataclass(frozen=True)
class UIElement:
    """Represents a spatially grounded, typed user interface element.

    Attributes:
        name: Extracted or target label describing the element.
        element_type: UIElementType category (button, input, icon, etc.).
        bounds: Window-relative or desktop-relative WindowBounds.
        center: Calibrated center Point(x, y) for precision targeting.
        confidence: Normalized confidence value [0.0, 1.0].
        source: GroundingSource indicating whether OCR, Multimodal, or Fusion resolved it.
        text_content: Optional literal text recognized inside the element bounds.
        metadata: Privacy-safe metadata (never containing raw pixel data).
    """

    name: str
    element_type: UIElementType
    bounds: WindowBounds
    center: Point
    confidence: float = 1.0
    source: GroundingSource = GroundingSource.UNKNOWN
    text_content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate confidence range [0.0, 1.0]."""
        clamped = max(0.0, min(1.0, float(self.confidence)))
        if clamped != self.confidence:
            object.__setattr__(self, "confidence", clamped)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize UIElement to dictionary."""
        return {
            "name": self.name,
            "element_type": self.element_type.value if isinstance(self.element_type, UIElementType) else str(self.element_type),
            "bounds": self.bounds.to_dict() if self.bounds else None,
            "center": self.center.to_dict() if self.center else None,
            "confidence": self.confidence,
            "source": self.source.value if isinstance(self.source, GroundingSource) else str(self.source),
            "text_content": self.text_content,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class VisualGroundingResult:
    """Consolidated outcome of a visual element localization operation.

    Attributes:
        target: The natural-language query or element description requested.
        element: Resolved UIElement, or None if not found or uncertain.
        is_found: True if an element was grounded with acceptable confidence.
        confidence: Overall confidence score [0.0, 1.0].
        observation_id: UUID of the underlying ScreenObservation.
        duration: Processing wall-clock duration in seconds.
        summary: Natural-language explanation of the localization result.
        metadata: Operational telemetry (never containing raw image bytes).
    """

    target: str
    element: Optional[UIElement] = None
    is_found: bool = False
    confidence: float = 0.0
    observation_id: str = ""
    duration: float = 0.0
    summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate confidence range [0.0, 1.0]."""
        clamped = max(0.0, min(1.0, float(self.confidence)))
        if clamped != self.confidence:
            object.__setattr__(self, "confidence", clamped)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize VisualGroundingResult to dictionary."""
        return {
            "target": self.target,
            "element": self.element.to_dict() if self.element else None,
            "is_found": self.is_found,
            "confidence": self.confidence,
            "observation_id": self.observation_id,
            "duration": self.duration,
            "summary": self.summary,
            "metadata": dict(self.metadata),
        }


# --------------------------------------------------------------------------
# Visual Delta and Change Detection Models (Phase 27.10)
# --------------------------------------------------------------------------


class VisualDeltaType(str, Enum):
    """Categorization of visual differences detected between observations."""

    NO_MEANINGFUL_CHANGE = "no_meaningful_change"
    WINDOW_CHANGED = "window_changed"
    ELEMENT_APPEARED = "element_appeared"
    ELEMENT_DISAPPEARED = "element_disappeared"
    ELEMENT_MOVED = "element_moved"
    ELEMENT_RESIZED = "element_resized"
    TEXT_CHANGED = "text_changed"
    REGION_CHANGED = "region_changed"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class UIElementChange:
    """Represents a localized UI element change between two observations.

    Attributes:
        change_type: Categorized change type.
        before_element: Corresponding element in the baseline observation, if any.
        after_element: Corresponding element in the follow-up observation, if any.
        displacement: Relative (dx, dy) centroid translation if element moved.
        text_similarity: Normalized string similarity [0.0, 1.0].
        iou: Spatial Intersection-over-Union [0.0, 1.0].
        confidence: Confidence score of this change classification [0.0, 1.0].
        metadata: Privacy-safe structural metadata (never raw pixel bytes).
    """

    change_type: VisualDeltaType
    before_element: Optional[UIElement] = None
    after_element: Optional[UIElement] = None
    displacement: Optional[Point] = None
    text_similarity: float = 0.0
    iou: float = 0.0
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate confidence, text_similarity, and iou ranges."""
        clamped_conf = max(0.0, min(1.0, float(self.confidence)))
        if clamped_conf != self.confidence:
            object.__setattr__(self, "confidence", clamped_conf)

        clamped_ts = max(0.0, min(1.0, float(self.text_similarity)))
        if clamped_ts != self.text_similarity:
            object.__setattr__(self, "text_similarity", clamped_ts)

        clamped_iou = max(0.0, min(1.0, float(self.iou)))
        if clamped_iou != self.iou:
            object.__setattr__(self, "iou", clamped_iou)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize UIElementChange to dictionary."""
        return {
            "change_type": self.change_type.value if isinstance(self.change_type, VisualDeltaType) else str(self.change_type),
            "before_element": self.before_element.to_dict() if self.before_element else None,
            "after_element": self.after_element.to_dict() if self.after_element else None,
            "displacement": self.displacement.to_dict() if self.displacement else None,
            "text_similarity": self.text_similarity,
            "iou": self.iou,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class VisualDeltaResult:
    """Consolidated outcome of a screen observation pair comparison.

    Attributes:
        delta_id: Unique UUID for this delta calculation.
        before_observation_id: Observation ID of the baseline observation (T0).
        after_observation_id: Observation ID of the follow-up observation (T1).
        time_delta_seconds: Elapsed time in seconds between T0 and T1.
        primary_change_type: High-level classification of the visual change.
        element_changes: Detailed element-level changes detected.
        added_texts: List of textual tokens/lines newly appeared.
        removed_texts: List of textual tokens/lines that vanished.
        modified_texts: List of textual tokens/lines altered.
        window_changed: True if active window identity/process changed.
        meaningful_change_detected: True if a non-trivial visual change occurred.
        confidence: Overall confidence score [0.0, 1.0].
        explanation: Natural-language explanation of what changed.
        metadata: Safe telemetry metadata (never raw pixel bytes).
    """

    delta_id: str
    before_observation_id: str
    after_observation_id: str
    time_delta_seconds: float = 0.0
    primary_change_type: VisualDeltaType = VisualDeltaType.NO_MEANINGFUL_CHANGE
    element_changes: Tuple[UIElementChange, ...] = field(default_factory=tuple)
    added_texts: Tuple[str, ...] = field(default_factory=tuple)
    removed_texts: Tuple[str, ...] = field(default_factory=tuple)
    modified_texts: Tuple[str, ...] = field(default_factory=tuple)
    window_changed: bool = False
    meaningful_change_detected: bool = False
    confidence: float = 1.0
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate confidence range [0.0, 1.0]."""
        clamped = max(0.0, min(1.0, float(self.confidence)))
        if clamped != self.confidence:
            object.__setattr__(self, "confidence", clamped)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize VisualDeltaResult to dictionary."""
        return {
            "delta_id": self.delta_id,
            "before_observation_id": self.before_observation_id,
            "after_observation_id": self.after_observation_id,
            "time_delta_seconds": self.time_delta_seconds,
            "primary_change_type": self.primary_change_type.value if isinstance(self.primary_change_type, VisualDeltaType) else str(self.primary_change_type),
            "element_changes": [c.to_dict() for c in self.element_changes],
            "added_texts": list(self.added_texts),
            "removed_texts": list(self.removed_texts),
            "modified_texts": list(self.modified_texts),
            "window_changed": self.window_changed,
            "meaningful_change_detected": self.meaningful_change_detected,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "metadata": dict(self.metadata),
        }


# --------------------------------------------------------------------------
# Visual UI Scene Parsing & Interactive Element Mapping Models (Phase 27.11)
# --------------------------------------------------------------------------


class UIContainerType(str, Enum):
    """Categorization of structural UI regions in an application window."""

    HEADER = "header"
    SIDEBAR = "sidebar"
    TOOLBAR = "toolbar"
    FORM = "form"
    DIALOG = "dialog"
    TABLE = "table"
    TAB_PANEL = "tab_panel"
    STATUS_BAR = "status_bar"
    CONTENT_AREA = "content_area"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class UIContainer:
    """Represents a bounded structural UI container region.

    Attributes:
        container_id: Unique identifier for this container.
        container_type: Categorized container region type.
        bounds: Rectangular bounding box of the container.
        elements: Nested UI elements residing inside this container.
        label: Optional title or header label describing the container.
        confidence: Confidence score of this container classification [0.0, 1.0].
        metadata: Safe structural metadata (never raw pixels).
    """

    container_id: str
    container_type: UIContainerType
    bounds: WindowBounds
    elements: Tuple[UIElement, ...] = field(default_factory=tuple)
    label: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and clamp confidence to [0.0, 1.0]."""
        clamped = max(0.0, min(1.0, float(self.confidence)))
        if clamped != self.confidence:
            object.__setattr__(self, "confidence", clamped)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize UIContainer to dictionary."""
        return {
            "container_id": self.container_id,
            "container_type": self.container_type.value if isinstance(self.container_type, UIContainerType) else str(self.container_type),
            "bounds": self.bounds.to_dict() if self.bounds else None,
            "elements": [e.to_dict() for e in self.elements],
            "label": self.label,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class FormField:
    """Represents a bound form field pairing a label with an interactive input widget.

    Attributes:
        field_id: Unique identifier for this form field.
        label: Textual descriptor label (e.g. 'Username:', 'Password:').
        label_bounds: Bounding box of the label.
        input_element: Associated interactive input element.
        is_required: True if marked as required (e.g. trailing asterisk).
        confidence: Association confidence score [0.0, 1.0].
        metadata: Safe structural metadata (never raw pixels).
    """

    field_id: str
    label: str
    label_bounds: WindowBounds
    input_element: UIElement
    is_required: bool = False
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and clamp confidence to [0.0, 1.0]."""
        clamped = max(0.0, min(1.0, float(self.confidence)))
        if clamped != self.confidence:
            object.__setattr__(self, "confidence", clamped)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize FormField to dictionary."""
        return {
            "field_id": self.field_id,
            "label": self.label,
            "label_bounds": self.label_bounds.to_dict() if self.label_bounds else None,
            "input_element": self.input_element.to_dict() if self.input_element else None,
            "is_required": self.is_required,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class UIScene:
    """Holistic structured representation of a visual application screen.

    Attributes:
        scene_id: Unique UUID for this parsed scene.
        observation_id: Originating ScreenObservation ID.
        window_title: Title of the active application window.
        process_name: Process name of the active application.
        window_bounds: Bounding box of the application window.
        containers: Structural container regions (Header, Sidebar, Form, Dialog, etc.).
        interactive_elements: Complete inventory of actionable UI elements.
        form_fields: Bound label-to-input field associations.
        summary: Natural-language overview of the UI layout and controls.
        confidence: Overall parsing confidence score [0.0, 1.0].
        metadata: Safe telemetry and performance metadata (never raw pixels).
    """

    scene_id: str
    observation_id: str
    window_title: str
    process_name: Optional[str] = None
    window_bounds: Optional[WindowBounds] = None
    containers: Tuple[UIContainer, ...] = field(default_factory=tuple)
    interactive_elements: Tuple[UIElement, ...] = field(default_factory=tuple)
    form_fields: Tuple[FormField, ...] = field(default_factory=tuple)
    summary: str = ""
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and clamp confidence to [0.0, 1.0]."""
        clamped = max(0.0, min(1.0, float(self.confidence)))
        if clamped != self.confidence:
            object.__setattr__(self, "confidence", clamped)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize UIScene to dictionary."""
        return {
            "scene_id": self.scene_id,
            "observation_id": self.observation_id,
            "window_title": self.window_title,
            "process_name": self.process_name,
            "window_bounds": self.window_bounds.to_dict() if self.window_bounds else None,
            "containers": [c.to_dict() for c in self.containers],
            "interactive_elements": [e.to_dict() for e in self.interactive_elements],
            "form_fields": [f.to_dict() for f in self.form_fields],
            "summary": self.summary,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }



# --------------------------------------------------------------------------
# UI Control State, Interactive Affordance & Semantic Scene Models (Phase 27.12)
# --------------------------------------------------------------------------


class ControlVisualState(str, Enum):
    """Visual or operational state of an interactive UI control."""

    ENABLED = "enabled"
    DISABLED = "disabled"
    FOCUSED = "focused"
    CHECKED = "checked"
    UNCHECKED = "unchecked"
    INDETERMINATE = "indeterminate"
    EMPTY = "empty"
    POPULATED = "populated"
    UNCERTAIN = "uncertain"


class ControlAffordance(str, Enum):
    """Primary interaction affordance supported by a UI control."""

    CLICKABLE = "clickable"
    EDITABLE = "editable"
    TOGGLEABLE = "toggleable"
    SELECTABLE = "selectable"
    SCROLLABLE = "scrollable"
    READ_ONLY = "read_only"


@dataclass(frozen=True)
class ElementAffordance:
    """Represents the operational state and interaction affordance of a UIElement.

    Attributes:
        element: Target UIElement.
        detected_state: Classified visual/operational state.
        primary_affordance: Primary action affordance.
        confidence: Classification confidence score [0.0, 1.0].
        evidence: Natural-language explanation of detected evidence.
        metadata: Safe telemetry metadata (never raw pixels).
    """

    element: UIElement
    detected_state: ControlVisualState
    primary_affordance: ControlAffordance
    confidence: float = 1.0
    evidence: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and clamp confidence to [0.0, 1.0]."""
        clamped = max(0.0, min(1.0, float(self.confidence)))
        if clamped != self.confidence:
            object.__setattr__(self, "confidence", clamped)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ElementAffordance to dictionary."""
        return {
            "element": self.element.to_dict() if self.element else None,
            "detected_state": self.detected_state.value if isinstance(self.detected_state, ControlVisualState) else str(self.detected_state),
            "primary_affordance": self.primary_affordance.value if isinstance(self.primary_affordance, ControlAffordance) else str(self.primary_affordance),
            "confidence": self.confidence,
            "evidence": self.evidence,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SceneQueryAnswer:
    """Structured response to a visual scene state or affordance query.

    Attributes:
        target_element: Name or identifier of target element if applicable.
        detected_state: State of target element if determined.
        verified_condition: Boolean verdict if answering a verification question.
        text_content: Non-sensitive text content associated with target.
        confidence: Answer confidence score [0.0, 1.0].
        summary: Speakable summary suitable for TTS.
        element: Associated UIElement if resolved.
        metadata: Safe operational metadata (never raw pixels).
    """

    target_element: Optional[str] = None
    detected_state: Optional[ControlVisualState] = None
    verified_condition: Optional[bool] = None
    text_content: Optional[str] = None
    confidence: float = 1.0
    summary: str = ""
    element: Optional[UIElement] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and clamp confidence to [0.0, 1.0]."""
        clamped = max(0.0, min(1.0, float(self.confidence)))
        if clamped != self.confidence:
            object.__setattr__(self, "confidence", clamped)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize SceneQueryAnswer to dictionary."""
        return {
            "target_element": self.target_element,
            "detected_state": self.detected_state.value if isinstance(self.detected_state, ControlVisualState) else (str(self.detected_state) if self.detected_state else None),
            "verified_condition": self.verified_condition,
            "text_content": self.text_content,
            "confidence": self.confidence,
            "summary": self.summary,
            "element": self.element.to_dict() if self.element else None,
            "metadata": dict(self.metadata),
        }


__all__ = [
    "BufferExpiredError",
    "CaptureAuthorization",
    "CaptureBlockedError",
    "CaptureCategory",
    "CaptureDecision",
    "CaptureError",
    "ControlAffordance",
    "ControlVisualState",
    "ElementAffordance",
    "FormField",
    "GdiResourceError",
    "GroundingSource",
    "InvalidBoundsError",
    "MonitorInfo",
    "OCRResult",
    "OCRTextBlock",
    "Point",
    "SceneQueryAnswer",
    "ScreenCapture",
    "ScreenObservation",
    "SpatialRelation",
    "UIContainer",
    "UIContainerType",
    "UIElement",
    "UIElementChange",
    "UIElementType",
    "UIScene",
    "UnsupportedPlatformError",
    "VisionError",
    "VisionSecurityError",
    "VisualAnalysisResult",
    "VisualDeltaResult",
    "VisualDeltaType",
    "VisualGroundingResult",
    "WindowBounds",
]
