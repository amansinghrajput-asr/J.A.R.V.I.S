"""In-memory image preprocessing and downsampling for J.A.R.V.I.S Vision (Phase 27.3).

Provides CPU-only, in-memory dimension limiting and format normalization for
multimodal AI payloads and OCR extraction.

Key Invariants:
- Zero disk I/O (all transformations held strictly in RAM).
- Zero GPU/VRAM consumption (avoids contention with Whisper STT on 4GB VRAM).
- Area block-averaging downsampling via NumPy to preserve font legibility.
- Graceful preservation of original resolution when dimensions fall within bounds.
"""

from __future__ import annotations

import io
import math
from typing import Any, Final, Optional

from app.ai.models import ImagePart
from app.core.logger import get_logger
from app.vision.models import (
    InvalidBoundsError,
    ScreenCapture,
    _encode_raw_rgba_to_png,
)

logger = get_logger("VISION.PREPROCESSING")

DEFAULT_MAX_DIMENSION: Final[int] = 1600
MINIMUM_DIMENSION: Final[int] = 1


class ImagePreprocessor:
    """Lightweight, thread-safe in-memory image preprocessor for multimodal payloads."""

    def __init__(
        self,
        default_max_dimension: int = DEFAULT_MAX_DIMENSION,
    ) -> None:
        """Initialize ImagePreprocessor.

        Args:
            default_max_dimension: Maximum horizontal or vertical resolution limit (default 1600px).
        """
        self._max_dimension = max(MINIMUM_DIMENSION, int(default_max_dimension))

    @property
    def max_dimension(self) -> int:
        """Return the active maximum dimension limit."""
        return self._max_dimension

    def process(
        self,
        capture: ScreenCapture,
        max_dimension: Optional[int] = None,
    ) -> bytes:
        """Downsample and encode capture frame to PNG bytes entirely in RAM.

        Args:
            capture: The source ScreenCapture instance.
            max_dimension: Optional dimension limit override.

        Returns:
            Standard PNG encoded bytes in memory.

        Raises:
            InvalidBoundsError: If capture dimensions are zero, negative, or buffer is empty.
        """
        if capture is None or capture.is_empty:
            raise InvalidBoundsError("Cannot preprocess an empty or uninitialized ScreenCapture.")

        limit = max_dimension if max_dimension is not None else self._max_dimension
        limit = max(MINIMUM_DIMENSION, int(limit))

        w, h = capture.width, capture.height
        if w <= 0 or h <= 0:
            raise InvalidBoundsError(f"Invalid capture dimensions: width={w}, height={h}.")

        # 1. Within dimension limit: pass directly through existing zero-loss PNG encoder
        if w <= limit and h <= limit:
            return capture.to_png_bytes()

        # 2. Try Pillow if available in environment (bicubic downsampling)
        try:
            from PIL import Image

            rgba_bytes = capture.to_rgba()
            img = Image.frombytes("RGBA", (w, h), rgba_bytes)
            scale = limit / max(w, h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            resample_filter = getattr(Image, "Resampling", Image).BILINEAR
            resized = img.resize((new_w, new_h), resample_filter)
            buf = io.BytesIO()
            resized.save(buf, format="PNG")
            return buf.getvalue()
        except ImportError:
            pass
        except Exception as exc:
            logger.debug("Pillow downsampling failed, falling back to NumPy: %s", exc)

        # 3. High-quality block-averaging downsampling via NumPy
        try:
            import numpy as np

            factor = math.ceil(max(w / limit, h / limit))
            if factor <= 1:
                return capture.to_png_bytes()

            rgba = capture.to_rgba()
            arr = np.frombuffer(rgba, dtype=np.uint8)
            if arr.size == w * h * 4:
                arr = arr.reshape((h, w, 4))
                trim_h = (h // factor) * factor
                trim_w = (w // factor) * factor
                trimmed = arr[:trim_h, :trim_w, :]

                # Mathematical block average across (factor x factor) cells
                # Preserves text legibility and anti-aliasing far better than point subsampling
                downsampled = (
                    trimmed.reshape(trim_h // factor, factor, trim_w // factor, factor, 4)
                    .mean(axis=(1, 3))
                    .astype(np.uint8)
                )
                new_h, new_w = downsampled.shape[:2]
                return _encode_raw_rgba_to_png(downsampled.tobytes(), new_w, new_h)
        except Exception as exc:
            logger.warning("NumPy downsampling failed, preserving original resolution: %s", exc)

        # 4. Fallback: return full-resolution lossless PNG
        return capture.to_png_bytes()

    def to_image_part(
        self,
        capture: ScreenCapture,
        max_dimension: Optional[int] = None,
    ) -> ImagePart:
        """Preprocess capture and wrap into an immutable ImagePart for Gemini payloads.

        Args:
            capture: The source ScreenCapture instance.
            max_dimension: Optional dimension limit override.

        Returns:
            Immutable ImagePart holding in-memory PNG bytes.
        """
        png_bytes = self.process(capture, max_dimension=max_dimension)
        return ImagePart(data=png_bytes, mime_type="image/png")


# Module-level default instance
default_preprocessor = ImagePreprocessor()


def preprocess_capture(
    capture: ScreenCapture,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
) -> bytes:
    """Convenience function to preprocess a ScreenCapture into PNG bytes."""
    return default_preprocessor.process(capture, max_dimension=max_dimension)


def capture_to_image_part(
    capture: ScreenCapture,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
) -> ImagePart:
    """Convenience function to convert a ScreenCapture into an ImagePart."""
    return default_preprocessor.to_image_part(capture, max_dimension=max_dimension)
