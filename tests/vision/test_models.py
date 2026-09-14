"""Hardware-independent deterministic unit tests for Phase 27.1 Vision Models.

Tests:
- Point coordinates and serialization
- WindowBounds geometry, intersections, center, and negative offsets
- MonitorInfo display metadata
- ScreenCapture in-memory buffers, RGBA conversion, and PNG encoding
- Exceptions hierarchy
"""

from __future__ import annotations

import unittest

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


class TestVisionModels(unittest.TestCase):
    """Deterministic unit tests for Point, WindowBounds, MonitorInfo, and ScreenCapture."""

    def test_point_creation_and_properties(self) -> None:
        """Verify Point construction, coordinates, and immutability."""
        p = Point(x=150, y=300)
        self.assertEqual(p.x, 150)
        self.assertEqual(p.y, 300)
        self.assertEqual(p.to_tuple(), (150, 300))
        self.assertEqual(p.to_dict(), {"x": 150, "y": 300})

        # Test negative coordinate support
        neg_p = Point(x=-1920, y=-1080)
        self.assertEqual(neg_p.x, -1920)
        self.assertEqual(neg_p.y, -1080)

    def test_window_bounds_geometry_and_properties(self) -> None:
        """Verify WindowBounds geometry calculations and empty state."""
        b = WindowBounds(left=100, top=200, right=900, bottom=700)
        self.assertEqual(b.width, 800)
        self.assertEqual(b.height, 500)
        self.assertEqual(b.area, 400000)
        self.assertFalse(b.is_empty)
        self.assertEqual(b.center, Point(x=500, y=450))
        self.assertEqual(b.to_tuple(), (100, 200, 900, 700))

        # Test from_xywh factory
        b2 = WindowBounds.from_xywh(100, 200, 800, 500)
        self.assertEqual(b, b2)

        # Test from_dict factory
        b_dict = b.to_dict()
        self.assertEqual(b_dict["width"], 800)
        self.assertEqual(b_dict["height"], 500)
        self.assertEqual(WindowBounds.from_dict(b_dict), b)

    def test_window_bounds_negative_coordinates(self) -> None:
        """Verify WindowBounds with negative coordinates (multi-monitor topology)."""
        # Secondary monitor placed to the left of the primary display
        left_mon = WindowBounds(left=-1920, top=0, right=0, bottom=1080)
        self.assertEqual(left_mon.width, 1920)
        self.assertEqual(left_mon.height, 1080)
        self.assertEqual(left_mon.center, Point(x=-960, y=540))
        self.assertFalse(left_mon.is_empty)

        # Monitor placed above the primary display
        top_mon = WindowBounds(left=0, top=-1080, right=1920, bottom=0)
        self.assertEqual(top_mon.width, 1920)
        self.assertEqual(top_mon.height, 1080)
        self.assertEqual(top_mon.center, Point(x=960, y=-540))

    def test_window_bounds_empty_and_inverted(self) -> None:
        """Verify zero-sized and inverted bounds are treated as empty."""
        zero_b = WindowBounds(left=100, top=100, right=100, bottom=200)
        self.assertEqual(zero_b.width, 0)
        self.assertEqual(zero_b.height, 100)
        self.assertTrue(zero_b.is_empty)

        inv_b = WindowBounds(left=500, top=500, right=200, bottom=100)
        self.assertEqual(inv_b.width, 0)
        self.assertEqual(inv_b.height, 0)
        self.assertTrue(inv_b.is_empty)

    def test_window_bounds_containment_and_intersection(self) -> None:
        """Verify point containment and bounding box intersection logic."""
        b = WindowBounds(left=100, top=100, right=500, bottom=400)

        # Inside
        self.assertTrue(b.contains_point(Point(150, 150)))
        self.assertTrue(b.contains_point((200, 300)))
        self.assertTrue(b.contains_point((100, 100)))  # Top-left corner is inclusive

        # Outside
        self.assertFalse(b.contains_point((500, 400)))  # Bottom-right corner is exclusive
        self.assertFalse(b.contains_point((50, 50)))
        self.assertFalse(b.contains_point((600, 200)))

        # Intersections
        overlapping = WindowBounds(left=300, top=200, right=700, bottom=600)
        self.assertTrue(b.intersects(overlapping))
        self.assertTrue(overlapping.intersects(b))

        sect = b.intersection(overlapping)
        self.assertIsNotNone(sect)
        assert sect is not None
        self.assertEqual(sect, WindowBounds(left=300, top=200, right=500, bottom=400))

        # Disjoint
        disjoint = WindowBounds(left=600, top=600, right=800, bottom=800)
        self.assertFalse(b.intersects(disjoint))
        self.assertIsNone(b.intersection(disjoint))

    def test_monitor_info(self) -> None:
        """Verify MonitorInfo properties and dictionary serialization."""
        bounds = WindowBounds(left=0, top=0, right=1920, bottom=1080)
        work = WindowBounds(left=0, top=0, right=1920, bottom=1040)
        mon = MonitorInfo(
            handle=101,
            name=r"\\.\DISPLAY1",
            bounds=bounds,
            work_area=work,
            is_primary=True,
            device_pixel_ratio=1.25,
        )
        self.assertEqual(mon.width, 1920)
        self.assertEqual(mon.height, 1080)
        self.assertTrue(mon.is_primary)
        d = mon.to_dict()
        self.assertEqual(d["handle"], 101)
        self.assertEqual(d["name"], r"\\.\DISPLAY1")
        self.assertTrue(d["is_primary"])
        self.assertEqual(d["bounds"]["width"], 1920)
        self.assertEqual(d["work_area"]["height"], 1040)

    def test_screen_capture_in_memory_buffer(self) -> None:
        """Verify ScreenCapture construction, properties, and serialization."""
        width = 10
        height = 10
        # 10x10 BGRA pixels = 400 bytes (Blue: 255, Green: 0, Red: 0, Alpha: 255)
        raw_bgra = bytes([255, 0, 0, 255]) * (width * height)

        bounds = WindowBounds(left=0, top=0, right=10, bottom=10)
        cap = ScreenCapture(
            raw_data=raw_bgra,
            width=width,
            height=height,
            channels=4,
            pixel_format="BGRA",
            source="test",
            bounds=bounds,
        )

        self.assertEqual(cap.width, 10)
        self.assertEqual(cap.height, 10)
        self.assertEqual(cap.size_bytes, 400)
        self.assertFalse(cap.is_empty)
        self.assertEqual(cap.stride, 40)

        # Dictionary should NOT dump raw bytes to prevent log bloat
        d = cap.to_dict()
        self.assertNotIn("raw_data", d)
        self.assertEqual(d["width"], 10)
        self.assertEqual(d["size_bytes"], 400)
        self.assertEqual(d["source"], "test")

    def test_screen_capture_rgba_conversion(self) -> None:
        """Verify conversion from BGRA to RGBA in memory."""
        # Single pixel: B=200, G=150, R=100, A=255
        pixel_bgra = bytes([200, 150, 100, 255])
        cap = ScreenCapture(
            raw_data=pixel_bgra,
            width=1,
            height=1,
            pixel_format="BGRA",
        )
        rgba = cap.to_rgba()
        # Expect R=100, G=150, B=200, A=255
        self.assertEqual(rgba, bytes([100, 150, 200, 255]))

        # Idempotent if already RGBA
        cap_rgba = ScreenCapture(
            raw_data=rgba,
            width=1,
            height=1,
            pixel_format="RGBA",
        )
        self.assertEqual(cap_rgba.to_rgba(), rgba)

    def test_screen_capture_png_encoding(self) -> None:
        """Verify deterministic in-memory PNG encoding without disk writes."""
        width = 4
        height = 4
        raw_bgra = bytes([0, 255, 0, 255]) * (width * height)  # Solid green
        cap = ScreenCapture(
            raw_data=raw_bgra,
            width=width,
            height=height,
            pixel_format="BGRA",
        )

        png_bytes = cap.to_png_bytes()
        self.assertIsInstance(png_bytes, bytes)
        self.assertGreater(len(png_bytes), 0)

        # Validate standard PNG magic header: \x89PNG\r\n\x1a\n
        self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))
        # Validate contains IHDR and IEND chunks
        self.assertIn(b"IHDR", png_bytes)
        self.assertIn(b"IDAT", png_bytes)
        self.assertTrue(png_bytes.endswith(b"IEND\xaeB`\x82"))

        # Empty capture produces empty PNG
        empty_cap = ScreenCapture(raw_data=b"", width=0, height=0)
        self.assertEqual(empty_cap.to_png_bytes(), b"")

    def test_exception_hierarchy(self) -> None:
        """Verify vision exception classes inherit correctly from JarvisException."""
        self.assertTrue(issubclass(CaptureError, VisionError))
        self.assertTrue(issubclass(InvalidBoundsError, CaptureError))
        self.assertTrue(issubclass(GdiResourceError, CaptureError))
        self.assertTrue(issubclass(UnsupportedPlatformError, VisionError))


if __name__ == "__main__":
    unittest.main()
