"""Unit tests for OCR abstraction, structured result models, and adapters (Phase 27.3).

Validates:
- OCRTextBlock, OCRResult, VisualAnalysisResult typed data models
- MockOCRProvider deterministic behavior (sync & async)
- WindowsMediaOCRProvider lazy WinRT detection
- MultimodalVisionOCRAdapter integration with AI provider
- Security and ephemeral observation boundary compatibility
"""

from __future__ import annotations

import asyncio
import unittest
from typing import Any, Optional

from app.ai.models import AIResponse, UnsupportedModalityError
from app.vision.models import (
    CaptureAuthorization,
    CaptureCategory,
    CaptureDecision,
    OCRResult,
    OCRTextBlock,
    ScreenCapture,
    ScreenObservation,
    UnsupportedPlatformError,
    VisualAnalysisResult,
    WindowBounds,
)
from app.vision.ocr import (
    MockOCRProvider,
    MultimodalVisionOCRAdapter,
    OCRProvider,
    WindowsMediaOCRProvider,
)


def _create_synthetic_capture(text: str = "Test") -> ScreenCapture:
    """Helper to generate a lightweight valid ScreenCapture."""
    width, height = 300, 100
    raw = bytes([128] * (width * height * 4))
    return ScreenCapture(
        raw_data=raw,
        width=width,
        height=height,
        pixel_format="BGRA",
        bounds=WindowBounds(100, 100, 400, 200),
        metadata={"label": text},
    )


class TestStructuredOCRModels(unittest.TestCase):
    """Test suite for structured visual analysis and OCR data models."""

    def test_ocr_text_block_properties_and_serialization(self) -> None:
        """Verify OCRTextBlock creation, immutability, and to_dict()."""
        bounds = WindowBounds(10, 20, 110, 50)
        block = OCRTextBlock(text="Total: $42.50", confidence=0.98, bounds=bounds)

        self.assertEqual(block.text, "Total: $42.50")
        self.assertEqual(block.confidence, 0.98)
        self.assertEqual(block.bounds, bounds)

        d = block.to_dict()
        self.assertEqual(d["text"], "Total: $42.50")
        self.assertEqual(d["confidence"], 0.98)
        self.assertIsNotNone(d["bounds"])
        self.assertEqual(d["bounds"]["left"], 10)

        # Immutability check
        with self.assertRaises(Exception):
            block.text = "Changed"  # type: ignore

    def test_ocr_result_properties_and_is_empty(self) -> None:
        """Verify OCRResult fields, is_empty helper, and serialization."""
        empty_res = OCRResult(text="", blocks=(), duration=0.01)
        self.assertTrue(empty_res.is_empty)

        b1 = OCRTextBlock(text="Line 1", confidence=0.95)
        b2 = OCRTextBlock(text="Line 2", confidence=0.92)
        res = OCRResult(text="Line 1\nLine 2", blocks=(b1, b2), language="en", duration=0.15)

        self.assertFalse(res.is_empty)
        self.assertEqual(len(res.blocks), 2)
        d = res.to_dict()
        self.assertEqual(d["text"], "Line 1\nLine 2")
        self.assertEqual(len(d["blocks"]), 2)
        self.assertEqual(d["language"], "en")
        self.assertEqual(d["duration"], 0.15)
        self.assertFalse(d["is_empty"])

    def test_visual_analysis_result_model(self) -> None:
        """Verify VisualAnalysisResult serialization without raw bytes."""
        ocr = OCRResult(text="Welcome to J.A.R.V.I.S")
        res = VisualAnalysisResult(
            summary="A desktop window displaying the greeting screen.",
            extracted_text="Welcome to J.A.R.V.I.S",
            ocr_result=ocr,
            detected_elements=("greeting_label", "start_button"),
            observation_id="test-obs-1234",
            duration=0.45,
            metadata={"source": "active_window"},
        )

        d = res.to_dict()
        self.assertEqual(d["summary"], "A desktop window displaying the greeting screen.")
        self.assertEqual(d["extracted_text"], "Welcome to J.A.R.V.I.S")
        self.assertEqual(d["observation_id"], "test-obs-1234")
        self.assertEqual(d["duration"], 0.45)
        self.assertEqual(d["detected_elements"], ["greeting_label", "start_button"])
        self.assertEqual(d["ocr_result"]["text"], "Welcome to J.A.R.V.I.S")

        # Crucial security check: no raw bytes or base64 data in serialization
        dump_str = str(d)
        self.assertNotIn("raw_data", dump_str)
        self.assertNotIn("inlineData", dump_str)


class TestMockOCRProvider(unittest.TestCase):
    """Test suite for MockOCRProvider."""

    def test_mock_ocr_sync_extraction(self) -> None:
        """Verify deterministic text and block extraction synchronously."""
        provider = MockOCRProvider(canned_text="File Edit View Help")
        capture = _create_synthetic_capture()
        result = provider.extract_text(capture)

        self.assertIsInstance(result, OCRResult)
        self.assertEqual(result.text, "File Edit View Help")
        self.assertEqual(len(result.blocks), 1)
        self.assertEqual(result.blocks[0].text, "File Edit View Help")
        self.assertTrue(result.duration >= 0.0)

    def test_mock_ocr_async_extraction(self) -> None:
        """Verify asynchronous execution without blocking."""
        provider = MockOCRProvider(canned_text="Line One\nLine Two")
        capture = _create_synthetic_capture()

        async def _run() -> OCRResult:
            return await provider.extract_text_async(capture)

        result = asyncio.run(_run())
        self.assertEqual(result.text, "Line One\nLine Two")
        self.assertEqual(len(result.blocks), 2)
        self.assertEqual(result.blocks[0].text, "Line One")
        self.assertEqual(result.blocks[1].text, "Line Two")

    def test_mock_ocr_empty_capture(self) -> None:
        """Verify empty capture yields empty OCRResult."""
        provider = MockOCRProvider(canned_text="Should not return")
        empty_capture = ScreenCapture(raw_data=b"", width=0, height=0)
        res = provider.extract_text(empty_capture)
        self.assertTrue(res.is_empty)

    def test_mock_ocr_update_canned_text(self) -> None:
        """Verify dynamic update of mock response."""
        provider = MockOCRProvider(canned_text="Initial")
        provider.set_canned_text("Updated Text")
        res = provider.extract_text(_create_synthetic_capture())
        self.assertEqual(res.text, "Updated Text")


class TestWindowsMediaOCRProvider(unittest.TestCase):
    """Test suite for WindowsMediaOCRProvider lazy detection."""

    def test_lazy_availability_detection(self) -> None:
        """Verify is_available() returns a boolean and does not raise exceptions."""
        provider = WindowsMediaOCRProvider()
        avail = provider.is_available()
        self.assertIsInstance(avail, bool)

    def test_extract_text_graceful_failure_when_unavailable(self) -> None:
        """Verify UnsupportedPlatformError is raised if invoked when unavailable."""
        provider = WindowsMediaOCRProvider()
        if not provider.is_available():
            with self.assertRaises(UnsupportedPlatformError):
                provider.extract_text(_create_synthetic_capture())


class FakeMultimodalProvider:
    """Mock AI provider supporting multimodal generation."""

    def __init__(self, response_text: str = "Extracted Text From Image", supports_multimodal: bool = True):
        self.response_text = response_text
        self.supports_multimodal = supports_multimodal
        self.last_payload: Optional[dict[str, Any]] = None

    def generate(self, payload: dict[str, Any], config: Any = None) -> AIResponse:
        self.last_payload = payload
        return AIResponse(content=self.response_text)

    async def generate_async(self, payload: dict[str, Any], config: Any = None) -> AIResponse:
        self.last_payload = payload
        return AIResponse(content=self.response_text)


class TestMultimodalVisionOCRAdapter(unittest.TestCase):
    """Test suite for MultimodalVisionOCRAdapter."""

    def test_adapter_sync_extraction(self) -> None:
        """Verify synchronous extraction formats OCR prompt and parses response."""
        fake_ai = FakeMultimodalProvider(response_text="Header Title\nSubtitle Text")
        adapter = MultimodalVisionOCRAdapter(provider=fake_ai)

        capture = _create_synthetic_capture()
        result = adapter.extract_text(capture)

        self.assertIsInstance(result, OCRResult)
        self.assertEqual(result.text, "Header Title\nSubtitle Text")
        self.assertEqual(len(result.blocks), 2)
        self.assertEqual(result.blocks[0].text, "Header Title")
        self.assertEqual(result.blocks[1].text, "Subtitle Text")

        # Verify payload sent to provider has inlineData
        self.assertIsNotNone(fake_ai.last_payload)
        parts = fake_ai.last_payload["contents"][0]["parts"]
        self.assertTrue(any("inlineData" in p for p in parts))

    def test_adapter_async_extraction(self) -> None:
        """Verify non-blocking asynchronous extraction."""
        fake_ai = FakeMultimodalProvider(response_text="Async Line 1\nAsync Line 2")
        adapter = MultimodalVisionOCRAdapter(provider=fake_ai)

        capture = _create_synthetic_capture()

        async def _run() -> OCRResult:
            return await adapter.extract_text_async(capture)

        result = asyncio.run(_run())
        self.assertEqual(result.text, "Async Line 1\nAsync Line 2")
        self.assertEqual(len(result.blocks), 2)

    def test_adapter_unsupported_modality_rejection(self) -> None:
        """Verify adapter raises UnsupportedModalityError when provider does not support vision."""
        text_only_ai = FakeMultimodalProvider(supports_multimodal=False)
        adapter = MultimodalVisionOCRAdapter(provider=text_only_ai)

        capture = _create_synthetic_capture()
        with self.assertRaises(UnsupportedModalityError):
            adapter.extract_text(capture)


class TestSecurityAndObservationIntegration(unittest.TestCase):
    """Test suite verifying security boundary and ephemeral buffer interaction."""

    def test_expired_observation_handling(self) -> None:
        """Verify expired ScreenObservation can be detected and handled safely."""
        capture = _create_synthetic_capture()
        obs = ScreenObservation(
            capture=capture,
            expires_at=0.0,  # Expired in 1970
        )
        self.assertTrue(obs.is_expired)
        self.assertFalse(obs.is_valid)

    def test_blocked_capture_authorization(self) -> None:
        """Verify blocked authorization contains no capture frame."""
        auth = CaptureAuthorization(
            decision=CaptureDecision.BLOCK,
            category=CaptureCategory.SENSITIVE_APPLICATION,
            reason="Blocked by Bitwarden policy",
            rule_name="password_managers",
        )
        obs = ScreenObservation(
            capture=None,
            authorization=auth,
        )
        self.assertFalse(obs.is_valid)
        self.assertIsNone(obs.capture)


if __name__ == "__main__":
    unittest.main()
