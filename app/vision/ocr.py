"""OCR (Optical Character Recognition) Abstraction and Providers for J.A.R.V.I.S (Phase 27.3).

Provides a pluggable, hardware-safe OCR architecture:
1. OCRProvider: Abstract base interface supporting sync and non-blocking async execution.
2. MockOCRProvider: Hardware-free, deterministic provider for testing and headless CI.
3. WindowsMediaOCRProvider: Optional adapter for Windows.Media.Ocr (lazy WinRT detection).
4. MultimodalVisionOCRAdapter: Gemini multimodal vision adapter for text extraction.

Safety Invariants:
- Zero real desktop capture required during testing.
- Zero external binary requirements (no Tesseract.exe).
- Lazy optional WinRT imports (never crashes if winsdk/winrt runtime is absent).
- Raw image bytes never appear in logs or error messages.
"""

from __future__ import annotations

import abc
import asyncio
import re
import time
from typing import Any, Final, List, Optional, Sequence

from app.ai.models import UnsupportedModalityError
from app.ai.prompt import PromptBuilder
from app.core.container import ServiceContainer, container as default_container
from app.core.logger import get_logger
from app.vision.models import (
    CaptureError,
    OCRResult,
    OCRTextBlock,
    ScreenCapture,
    UnsupportedPlatformError,
    WindowBounds,
)
from app.vision.preprocessing import ImagePreprocessor, default_preprocessor

logger = get_logger("VISION.OCR")

DEFAULT_OCR_PROMPT: Final[str] = (
    "Extract all visible text from this image exactly as displayed. "
    "Preserve line breaks where appropriate. "
    "Do not include any commentary, explanations, reasoning, or markdown fences."
)


class OCRProvider(abc.ABC):
    """Abstract base class for all optical character recognition providers."""

    @abc.abstractmethod
    def extract_text(self, capture: ScreenCapture) -> OCRResult:
        """Synchronously extract visible text from an in-memory ScreenCapture.

        Args:
            capture: ScreenCapture holding uncompressed pixel data.

        Returns:
            Structured OCRResult containing extracted text and blocks.
        """

    async def extract_text_async(self, capture: ScreenCapture) -> OCRResult:
        """Asynchronously extract visible text without blocking GUI or calling threads.

        Default implementation offloads synchronous extraction to worker thread.
        """
        return await asyncio.to_thread(self.extract_text, capture)

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True if this OCR provider is supported and available in the current environment."""


# --------------------------------------------------------------------------
# Mock OCR Provider (Deterministic / Testing)
# --------------------------------------------------------------------------


class MockOCRProvider(OCRProvider):
    """Deterministic, hardware-free OCR provider for automated testing and simulation."""

    def __init__(
        self,
        canned_text: str = "J.A.R.V.I.S Vision Intelligence Test",
        canned_blocks: Optional[Sequence[OCRTextBlock]] = None,
        available: bool = True,
        simulate_duration: float = 0.0,
    ) -> None:
        """Initialize MockOCRProvider.

        Args:
            canned_text: Default synthetic text string to return.
            canned_blocks: Optional explicit sequence of OCRTextBlock items.
            available: Whether is_available() should return True.
            simulate_duration: Optional simulated processing delay in seconds.
        """
        self._canned_text = canned_text
        self._canned_blocks = tuple(canned_blocks) if canned_blocks is not None else None
        self._available = available
        self._simulate_duration = max(0.0, float(simulate_duration))

    def is_available(self) -> bool:
        """Return configured availability status."""
        return self._available

    def set_canned_text(self, text: str, blocks: Optional[Sequence[OCRTextBlock]] = None) -> None:
        """Update mock response text for subsequent calls."""
        self._canned_text = text
        self._canned_blocks = tuple(blocks) if blocks is not None else None

    def extract_text(self, capture: ScreenCapture) -> OCRResult:
        """Return canned OCR results deterministically."""
        start_time = time.perf_counter()

        if self._simulate_duration > 0:
            time.sleep(self._simulate_duration)

        if capture is None or capture.is_empty:
            return OCRResult(text="", blocks=(), duration=time.perf_counter() - start_time)

        if self._canned_blocks is not None:
            blocks = self._canned_blocks
            text = self._canned_text
        else:
            lines = [line.strip() for line in self._canned_text.splitlines() if line.strip()]
            generated_blocks: List[OCRTextBlock] = []
            for i, line in enumerate(lines):
                b = None
                if capture.bounds:
                    line_height = max(10, capture.bounds.height // max(1, len(lines)))
                    b = WindowBounds(
                        left=capture.bounds.left,
                        top=capture.bounds.top + i * line_height,
                        right=capture.bounds.right,
                        bottom=capture.bounds.top + (i + 1) * line_height,
                    )
                generated_blocks.append(OCRTextBlock(text=line, confidence=0.99, bounds=b))
            blocks = tuple(generated_blocks)
            text = self._canned_text

        duration = time.perf_counter() - start_time
        return OCRResult(text=text, blocks=blocks, language="en", duration=duration)


# --------------------------------------------------------------------------
# Native Windows Media OCR Provider (Optional / Lazy WinRT)
# --------------------------------------------------------------------------


class WindowsMediaOCRProvider(OCRProvider):
    """Adapter for native Windows 10/11 Windows.Media.Ocr.

    Maintains completely lazy and guarded imports. Never crashes during instantiation
    if WinRT runtime packages (winsdk / winrt) are absent.
    """

    def __init__(self) -> None:
        self._winrt_available: Optional[bool] = None

    def is_available(self) -> bool:
        """Check if Windows native WinRT OCR runtime is importable."""
        if self._winrt_available is not None:
            return self._winrt_available

        try:
            import platform

            if platform.system().lower() != "windows":
                self._winrt_available = False
                return False

            # Guarded attempt to import WinRT OCR projection
            try:
                import winsdk.windows.media.ocr as _  # type: ignore # noqa: F401
                self._winrt_available = True
            except ImportError:
                try:
                    import winrt.windows.media.ocr as _  # type: ignore # noqa: F401
                    self._winrt_available = True
                except ImportError:
                    self._winrt_available = False
        except Exception:
            self._winrt_available = False

        return self._winrt_available

    def extract_text(self, capture: ScreenCapture) -> OCRResult:
        """Extract text using Windows.Media.Ocr if available.

        Raises:
            UnsupportedPlatformError: If WinRT OCR projection is unavailable.
        """
        if not self.is_available():
            raise UnsupportedPlatformError(
                "Windows.Media.Ocr is not available in the current environment. "
                "winsdk or winrt runtime is not installed."
            )

        # If available in future environment, implementation projects to Windows.Media.Ocr.OcrEngine
        raise NotImplementedError("Native WinRT OCR projection execution is reserved for future phases.")


# --------------------------------------------------------------------------
# Multimodal Vision OCR Adapter (Gemini Provider Integration)
# --------------------------------------------------------------------------


class MultimodalVisionOCRAdapter(OCRProvider):
    """Performs optical character recognition via the existing AI multimodal provider."""

    def __init__(
        self,
        provider: Optional[Any] = None,
        preprocessor: Optional[ImagePreprocessor] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        container_instance: Optional[ServiceContainer] = None,
        custom_prompt: Optional[str] = None,
    ) -> None:
        """Initialize MultimodalVisionOCRAdapter.

        Args:
            provider: Multimodal AI provider instance (e.g. GeminiProvider).
            preprocessor: In-memory image downsampling preprocessor.
            prompt_builder: Prompt builder instance for payload serialization.
            container_instance: Optional ServiceContainer instance.
            custom_prompt: Optional override for the concise OCR extraction prompt.
        """
        self._container = container_instance or default_container
        self._preprocessor = preprocessor or default_preprocessor
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._custom_prompt = custom_prompt or DEFAULT_OCR_PROMPT

        # Resolve provider
        if provider is not None:
            self._provider = provider
        elif self._container.exists("ai_provider"):
            self._provider = self._container.resolve("ai_provider")
        elif self._container.exists("gemini_provider"):
            self._provider = self._container.resolve("gemini_provider")
        elif self._container.exists("ai_manager"):
            ai_mgr = self._container.resolve("ai_manager")
            self._provider = getattr(ai_mgr, "provider", None)
        else:
            from app.ai.provider import GeminiProvider
            self._provider = GeminiProvider()

    @property
    def provider(self) -> Any:
        """Return active underlying AI provider."""
        return self._provider

    def is_available(self) -> bool:
        """Check whether underlying provider is multimodal capable."""
        if self._provider is None:
            return False
        return getattr(self._provider, "supports_multimodal", False)

    def _prepare_payload(self, capture: ScreenCapture) -> dict[str, Any]:
        """Convert capture to ImagePart and construct Gemini REST payload."""
        if not self.is_available():
            raise UnsupportedModalityError(
                f"Selected AI provider '{getattr(self._provider, 'model', 'unknown')}' "
                "does not support multimodal vision inputs."
            )

        image_part = self._preprocessor.to_image_part(capture)
        return self._prompt_builder.build_payload(
            query=self._custom_prompt,
            images=[image_part],
        )

    def _parse_ai_ocr_response(self, text: str, duration: float) -> OCRResult:
        """Parse raw LLM response text into structured OCRResult and OCRTextBlock items."""
        clean_text = (text or "").strip()
        if not clean_text:
            return OCRResult(text="", blocks=(), duration=duration)

        lines = [line for line in clean_text.splitlines() if line.strip()]
        blocks = tuple(
            OCRTextBlock(text=line.strip(), confidence=0.95, bounds=None)
            for line in lines
        )
        return OCRResult(text=clean_text, blocks=blocks, language="en", duration=duration)

    def extract_text(self, capture: ScreenCapture) -> OCRResult:
        """Extract text synchronously via multimodal AI."""
        start_time = time.perf_counter()
        if capture is None or capture.is_empty:
            return OCRResult(text="", blocks=(), duration=time.perf_counter() - start_time)

        payload = self._prepare_payload(capture)
        response = self._provider.generate(payload)
        duration = time.perf_counter() - start_time

        return self._parse_ai_ocr_response(response.content, duration)

    async def extract_text_async(self, capture: ScreenCapture) -> OCRResult:
        """Extract text asynchronously using non-blocking HTTP provider client."""
        start_time = time.perf_counter()
        if capture is None or capture.is_empty:
            return OCRResult(text="", blocks=(), duration=time.perf_counter() - start_time)

        payload = self._prepare_payload(capture)

        if hasattr(self._provider, "generate_async"):
            response = await self._provider.generate_async(payload)
        else:
            response = await asyncio.to_thread(self._provider.generate, payload)

        duration = time.perf_counter() - start_time
        return self._parse_ai_ocr_response(response.content, duration)
