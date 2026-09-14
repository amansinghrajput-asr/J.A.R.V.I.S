"""Unit tests for Multimodal AI Integration (Phase 27.3).

Validates ImagePart model, PromptBuilder multimodal payload construction,
backward compatibility with text-only prompts, and provider capability detection.
"""

from __future__ import annotations

import base64
import unittest
from typing import Any

import httpx

from app.ai.models import (
    AIResponse,
    ImagePart,
    UnsupportedModalityError,
)
from app.ai.ollama_provider import OllamaProvider
from app.ai.prompt import PromptBuilder
from app.ai.provider import GeminiProvider
from app.memory.models import ConversationMemory


class TestImagePart(unittest.TestCase):
    """Test suite for the in-memory ImagePart data model."""

    def test_image_part_creation_and_properties(self) -> None:
        """Verify initialization with valid bytes and MIME type."""
        raw_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        part = ImagePart(data=raw_bytes, mime_type="image/png")
        self.assertEqual(part.data, raw_bytes)
        self.assertEqual(part.mime_type, "image/png")

    def test_image_part_default_mime(self) -> None:
        """Verify default MIME type is 'image/png'."""
        part = ImagePart(data=b"test_image_bytes")
        self.assertEqual(part.mime_type, "image/png")

    def test_image_part_validation_empty_bytes(self) -> None:
        """Verify empty byte buffer raises ValueError."""
        with self.assertRaises(ValueError):
            ImagePart(data=b"")

    def test_image_part_validation_invalid_type(self) -> None:
        """Verify non-bytes data raises ValueError."""
        with self.assertRaises(ValueError):
            ImagePart(data="not_bytes")  # type: ignore

    def test_image_part_validation_invalid_mime(self) -> None:
        """Verify invalid or non-image MIME type raises ValueError."""
        with self.assertRaises(ValueError):
            ImagePart(data=b"dummy", mime_type="text/plain")

        with self.assertRaises(ValueError):
            ImagePart(data=b"dummy", mime_type="")

    def test_image_part_immutability(self) -> None:
        """Verify ImagePart is a frozen dataclass."""
        part = ImagePart(data=b"sample_data")
        with self.assertRaises(Exception):
            part.data = b"new_data"  # type: ignore

    def test_image_part_repr_safety(self) -> None:
        """Verify string and repr representations never expose raw bytes."""
        secret_bytes = b"SECRET_PIXEL_DATA_NEVER_LOG"
        part = ImagePart(data=secret_bytes, mime_type="image/jpeg")
        repr_str = repr(part)
        self.assertNotIn("SECRET_PIXEL_DATA", repr_str)
        self.assertIn("size_bytes=27", repr_str)
        self.assertIn("image/jpeg", repr_str)
        self.assertEqual(str(part), repr_str)

    def test_image_part_gemini_serialization(self) -> None:
        """Verify serialization adheres to Google Gemini REST inlineData schema."""
        raw = b"\x89PNG\r\n\x1a\n_dummy_content"
        part = ImagePart(data=raw, mime_type="image/png")
        d = part.to_gemini_dict()

        self.assertIn("inlineData", d)
        self.assertEqual(d["inlineData"]["mimeType"], "image/png")
        expected_b64 = base64.b64encode(raw).decode("ascii")
        self.assertEqual(d["inlineData"]["data"], expected_b64)


class TestPromptBuilderMultimodal(unittest.TestCase):
    """Test suite for multimodal prompt payload assembly and backward compatibility."""

    def setUp(self) -> None:
        self.builder = PromptBuilder()

    def test_backward_compatibility_text_only(self) -> None:
        """Verify that omitting images generates byte-for-byte identical text payloads."""
        payload_none = self.builder.build_payload("What is on my screen?", images=None)
        payload_empty = self.builder.build_payload("What is on my screen?", images=[])

        self.assertEqual(payload_none, payload_empty)
        contents = payload_none["contents"]
        self.assertEqual(len(contents), 1)
        self.assertEqual(contents[0]["role"], "user")
        self.assertEqual(contents[0]["parts"], [{"text": "What is on my screen?"}])

    def test_multimodal_payload_with_image_parts(self) -> None:
        """Verify image parts are appended alongside text in user turn."""
        img1 = ImagePart(data=b"img1_bytes", mime_type="image/png")
        img2 = ImagePart(data=b"img2_bytes", mime_type="image/jpeg")

        payload = self.builder.build_payload("Analyze this", images=[img1, img2])
        contents = payload["contents"]
        self.assertEqual(len(contents), 1)
        self.assertEqual(contents[0]["role"], "user")

        parts = contents[0]["parts"]
        self.assertEqual(len(parts), 3)
        self.assertEqual(parts[0], {"text": "Analyze this"})
        self.assertEqual(parts[1], img1.to_gemini_dict())
        self.assertEqual(parts[2], img2.to_gemini_dict())

    def test_multimodal_payload_with_raw_bytes_normalization(self) -> None:
        """Verify raw bytes passed to images are safely normalized to ImagePart."""
        raw_png = b"\x89PNG_raw_content"
        payload = self.builder.build_payload("Read text", images=[raw_png])

        parts = payload["contents"][0]["parts"]
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0], {"text": "Read text"})
        self.assertEqual(parts[1]["inlineData"]["mimeType"], "image/png")
        self.assertEqual(
            parts[1]["inlineData"]["data"], base64.b64encode(raw_png).decode("ascii")
        )

    def test_multimodal_payload_with_history(self) -> None:
        """Verify conversational history is preserved before the multimodal user turn."""
        history = [
            ConversationMemory(role="user", content="Hello Jarvis"),
            ConversationMemory(role="assistant", content="Good day, Sir."),
        ]
        img = ImagePart(data=b"sample_frame")
        payload = self.builder.build_payload("Check this", history=history, images=[img])

        contents = payload["contents"]
        self.assertEqual(len(contents), 3)
        self.assertEqual(contents[0]["role"], "user")
        self.assertEqual(contents[1]["role"], "model")
        self.assertEqual(contents[2]["role"], "user")
        self.assertEqual(len(contents[2]["parts"]), 2)
        self.assertEqual(contents[2]["parts"][0], {"text": "Check this"})
        self.assertIn("inlineData", contents[2]["parts"][1])


class TestProviderMultimodalCapability(unittest.TestCase):
    """Test suite for provider capability flags and error boundaries."""

    def test_gemini_provider_supports_multimodal(self) -> None:
        """Verify GeminiProvider declares supports_multimodal = True."""
        provider = GeminiProvider(api_key="test_key", model="gemini-2.0-flash")
        self.assertTrue(provider.supports_multimodal)

    def test_ollama_provider_supports_multimodal_is_false(self) -> None:
        """Verify OllamaProvider declares supports_multimodal = False."""
        provider = OllamaProvider(model="qwen2.5:3b")
        self.assertFalse(provider.supports_multimodal)

    def test_unsupported_modality_error(self) -> None:
        """Verify UnsupportedModalityError inheritance and messaging."""
        err = UnsupportedModalityError("Ollama does not support vision.")
        self.assertIsInstance(err, Exception)
        self.assertIn("Ollama does not support vision", str(err))

    def test_gemini_mock_multimodal_execution(self) -> None:
        """Verify GeminiProvider successfully parses mock response for multimodal payload."""
        def mock_handler(request: httpx.Request) -> httpx.Response:
            self.assertIn("key=test_api_key", str(request.url))
            mock_json = {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "I see a desktop window with code."}],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 150,
                    "candidatesTokenCount": 10,
                    "totalTokenCount": 160,
                },
            }
            return httpx.Response(200, json=mock_json)

        client = httpx.Client(transport=httpx.MockTransport(mock_handler))
        provider = GeminiProvider(api_key="test_api_key", model="gemini-2.0-flash", client=client)

        builder = PromptBuilder()
        img = ImagePart(data=b"\x89PNG_mock_screen")
        payload = builder.build_payload("Describe this screen", images=[img])

        resp: AIResponse = provider.generate(payload)
        self.assertEqual(resp.content, "I see a desktop window with code.")
        self.assertEqual(resp.finish_reason, "STOP")
        self.assertEqual(resp.prompt_tokens, 150)


if __name__ == "__main__":
    unittest.main()
