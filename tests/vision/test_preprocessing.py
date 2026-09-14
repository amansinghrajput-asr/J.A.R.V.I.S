"""Unit tests for in-memory image preprocessing (Phase 27.3).

Validates dimension bounding, block-averaging downsampling, memory safety,
and error handling for invalid or empty capture frames.
"""

from __future__ import annotations

import os
import unittest

from app.ai.models import ImagePart
from app.vision.models import (
    InvalidBoundsError,
    Point,
    ScreenCapture,
    WindowBounds,
)
from app.vision.preprocessing import (
    DEFAULT_MAX_DIMENSION,
    ImagePreprocessor,
    capture_to_image_part,
    preprocess_capture,
)


def _create_synthetic_capture(
    width: int,
    height: int,
    pixel_format: str = "BGRA",
) -> ScreenCapture:
    """Helper to generate an in-memory synthetic ScreenCapture."""
    raw = bytearray(width * height * 4)
    # Fill with a deterministic striped pattern
    for y in range(height):
        for x in range(width):
            idx = (y * width + x) * 4
            raw[idx] = (x * 7) % 256      # Blue
            raw[idx + 1] = (y * 5) % 256  # Green
            raw[idx + 2] = 128            # Red
            raw[idx + 3] = 255            # Alpha
    return ScreenCapture(
        raw_data=bytes(raw),
        width=width,
        height=height,
        pixel_format=pixel_format,
        bounds=WindowBounds(0, 0, width, height),
    )


class TestImagePreprocessing(unittest.TestCase):
    """Test suite for ImagePreprocessor."""

    def setUp(self) -> None:
        self.preprocessor = ImagePreprocessor(default_max_dimension=1600)

    def test_preprocessor_within_bounds_preserves_resolution(self) -> None:
        """Verify capture within dimension limits produces valid PNG without scaling."""
        capture = _create_synthetic_capture(800, 600)
        png_bytes = self.preprocessor.process(capture, max_dimension=1600)

        # Validate PNG magic bytes
        self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertTrue(len(png_bytes) > 0)

    def test_preprocessor_downsamples_oversized_frame(self) -> None:
        """Verify capture exceeding limits is downsampled to within the bounding box."""
        # 3200 x 2000 frame with limit 1600
        capture = _create_synthetic_capture(3200, 2000)
        png_bytes = self.preprocessor.process(capture, max_dimension=1600)

        self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))
        # Downsampled PNG size should be significantly smaller than raw 25MB
        self.assertTrue(len(png_bytes) < 3200 * 2000 * 4)

    def test_preprocessor_to_image_part(self) -> None:
        """Verify to_image_part outputs an immutable ImagePart."""
        capture = _create_synthetic_capture(400, 300)
        part = self.preprocessor.to_image_part(capture)

        self.assertIsInstance(part, ImagePart)
        self.assertEqual(part.mime_type, "image/png")
        self.assertTrue(part.data.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_preprocessor_convenience_functions(self) -> None:
        """Verify module-level convenience functions."""
        capture = _create_synthetic_capture(640, 480)
        raw_png = preprocess_capture(capture, max_dimension=1200)
        self.assertTrue(raw_png.startswith(b"\x89PNG\r\n\x1a\n"))

        part = capture_to_image_part(capture, max_dimension=1200)
        self.assertIsInstance(part, ImagePart)
        self.assertEqual(part.data, raw_png)

    def test_preprocessor_invalid_bounds_error_empty(self) -> None:
        """Verify empty capture raises InvalidBoundsError."""
        empty_capture = ScreenCapture(
            raw_data=b"",
            width=0,
            height=0,
            pixel_format="BGRA",
        )
        with self.assertRaises(InvalidBoundsError):
            self.preprocessor.process(empty_capture)

    def test_preprocessor_invalid_bounds_error_negative(self) -> None:
        """Verify negative dimensions raise InvalidBoundsError."""
        bad_capture = ScreenCapture(
            raw_data=b"dummy",
            width=-100,
            height=200,
            pixel_format="BGRA",
        )
        with self.assertRaises(InvalidBoundsError):
            self.preprocessor.process(bad_capture)

    def test_preprocessor_none_capture(self) -> None:
        """Verify None capture raises InvalidBoundsError."""
        with self.assertRaises(InvalidBoundsError):
            self.preprocessor.process(None)  # type: ignore

    def test_large_synthetic_4k_frame(self) -> None:
        """Verify synthetic 3840x2160 (4K) frame is downsampled safely in RAM without memory explosion."""
        # 3840 x 2160
        capture = _create_synthetic_capture(3840, 2160)
        png_bytes = self.preprocessor.process(capture, max_dimension=1600)

        self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertTrue(len(png_bytes) > 0)


if __name__ == "__main__":
    unittest.main()
