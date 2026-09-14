"""Vision Skills for J.A.R.V.I.S. Phase 27.4.

Provides safe, privacy-guarded visual perception, desktop screen capture,
hybrid OCR extraction, active-window explanation, and visual error diagnosis.

Safety and Architectural Invariants Enforced:
1. Every capture operation MUST pass through SecureVisionManager and VisionSecurityPolicy.
   Direct calls to DesktopCaptureEngine or Win32 GDI APIs from this skill are strictly prohibited.
2. Sensitive contexts (password managers, banking credentials, incognito sessions) are blocked
   before raw pixels are captured.
3. Screenshots exist strictly in transient RAM buffers managed by EphemeralBufferManager.
   Screenshots are NEVER written to disk, logged, or placed into conversation memory.
4. No raw image bytes or base64 pixel strings are returned in SystemSkillResult data,
   telemetry payloads, or EventBus events.
5. Multimodal visual queries verify provider.supports_multimodal and fail gracefully
   when connected to text-only providers (e.g. Ollama default) without silent provider switching.
6. Execution is asynchronous and non-blocking to prevent freezing the GUI thread.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from typing import Any, Dict, Final, List, Optional, Tuple, Union

from app.ai.models import ImagePart, UnsupportedModalityError
from app.ai.prompt import PromptBuilder
from app.core.config import Settings
from app.core.container import ServiceContainer, container as default_container
from app.core.event_bus import EventBus
from app.core.logger import get_logger
from app.skills.base import SkillExecutionError
from app.skills.system.base_system_skill import BaseSystemSkill, SystemSkillResult
from app.skills.system.security import (
    ConfirmationRequiredError,
    SecurityPolicyViolationError,
    SystemConfirmationManager,
    SystemSecurityPolicy,
)
from app.vision.models import (
    BufferExpiredError,
    CaptureBlockedError,
    CaptureError,
    OCRResult,
    ScreenCapture,
    ScreenObservation,
    UnsupportedPlatformError,
    VisionError,
    VisionSecurityError,
    WindowBounds,
)
from app.vision.ocr import (
    MockOCRProvider,
    MultimodalVisionOCRAdapter,
    OCRProvider,
    WindowsMediaOCRProvider,
)
from app.vision.preprocessing import ImagePreprocessor, default_preprocessor
from app.vision.security import (
    EphemeralBufferManager,
    SecureVisionManager,
    VisionSecurityPolicy,
)

logger = get_logger("SYSTEM.VISION_SKILLS")

# ---------------------------------------------------------------------------
# Regex Matchers for Natural Language Vision Commands
# ---------------------------------------------------------------------------

_RE_CAPTURE_SCREEN: Final[re.Pattern[str]] = re.compile(
    r"^(?:take\s+(?:a\s+)?screenshot|capture\s+(?:the\s+|my\s+)?screen|screenshot\s+(?:my\s+)?screen)$",
    re.IGNORECASE,
)

_RE_READ_TEXT: Final[re.Pattern[str]] = re.compile(
    r"^(?:read\s+(?:the\s+)?text\s+on\s+(?:my\s+|the\s+)?screen|"
    r"read\s+screen\s+text|"
    r"read\s+what(?:'s|\s+is)\s+on\s+my\s+screen|"
    r"ocr\s+(?:the\s+)?(?:screen|window))$",
    re.IGNORECASE,
)

_RE_EXPLAIN_WINDOW: Final[re.Pattern[str]] = re.compile(
    r"^(?:explain\s+(?:this\s+window|this\s+screen|my\s+screen|what(?:'s|\s+is)\s+on\s+my\s+screen)|"
    r"what\s+is\s+on\s+my\s+screen\??|"
    r"what(?:'s|\s+is)\s+on\s+my\s+screen\??|"
    r"look\s+at\s+my\s+screen|"
    r"describe\s+(?:this\s+screen|my\s+screen|this\s+window))$",
    re.IGNORECASE,
)

_RE_DIAGNOSE_ERROR: Final[re.Pattern[str]] = re.compile(
    r"^(?:what\s+does\s+this\s+error\s+mean\??|"
    r"diagnose\s+this\s+error|"
    r"why\s+is\s+this\s+screen\s+failing\??|"
    r"what\s+is\s+wrong\s+on\s+my\s+screen\??|"
    r"what\s+error\s+is\s+this\??)$",
    re.IGNORECASE,
)


class VisionSkills(BaseSystemSkill):
    """Production desktop vision and screen intelligence skill for J.A.R.V.I.S.

    Inherits from BaseSystemSkill, coordinating secure screen capture, hybrid OCR,
    multimodal window comprehension, and visual error diagnosis through the
    strictly authorized SecureVisionManager boundary.
    """

    name: str = "vision"
    description: str = "Visual perception, desktop screen capture, OCR, and multimodal desktop intelligence."
    priority: int = 52
    tags: list[str] = ["vision", "screen", "ocr", "multimodal", "intelligence"]
    permissions: set[str] = {"system:read", "vision:capture", "vision:analyze"}

    def __init__(
        self,
        *,
        secure_vision_manager: Optional[SecureVisionManager] = None,
        ocr_provider: Optional[OCRProvider] = None,
        ai_provider: Optional[Any] = None,
        preprocessor: Optional[ImagePreprocessor] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        security_policy: Optional[SystemSecurityPolicy] = None,
        confirmation_manager: Optional[SystemConfirmationManager] = None,
        config: Optional[Settings] = None,
        logger: Optional[logging.Logger] = None,
        container: Optional[ServiceContainer] = None,
        event_bus: Optional[Union[EventBus, Any]] = None,
    ) -> None:
        """Initialize VisionSkills instance."""
        super().__init__(
            name=self.name,
            description=self.description,
            priority=self.priority,
            tags=self.tags,
            permissions=self.permissions,
            security_policy=security_policy,
            confirmation_manager=confirmation_manager,
            config=config,
            logger=logger or get_logger("SKILL.VISION"),
            container=container,
            event_bus=event_bus,
        )
        self._lock = threading.RLock()
        self._secure_vision_manager = secure_vision_manager
        self._ocr_provider = ocr_provider
        self._ai_provider = ai_provider
        self._preprocessor = preprocessor or default_preprocessor
        self._prompt_builder = prompt_builder or PromptBuilder()

    # -----------------------------------------------------------------------
    # Dependency Resolution
    # -----------------------------------------------------------------------

    @property
    def secure_vision_manager(self) -> SecureVisionManager:
        """Resolve the active SecureVisionManager, preferring injected or container instance."""
        if self._secure_vision_manager is not None:
            return self._secure_vision_manager
        if self._container is not None and self._container.exists("secure_vision_manager"):
            return self._container.resolve("secure_vision_manager")
        # Default fallback instance
        self._secure_vision_manager = SecureVisionManager(
            container_instance=self._container,
            event_bus_instance=self._event_bus,
            logger_instance=self.logger,
            auto_register_in_container=False,
        )
        return self._secure_vision_manager

    @property
    def ai_provider(self) -> Optional[Any]:
        """Resolve the active multimodal AI provider."""
        if self._ai_provider is not None:
            return self._ai_provider
        if self._container is not None:
            for key in ("ai_provider", "gemini_provider"):
                if self._container.exists(key):
                    return self._container.resolve(key)
            if self._container.exists("ai_manager"):
                mgr = self._container.resolve("ai_manager")
                return getattr(mgr, "provider", None)
        return None

    @property
    def ocr_provider(self) -> Optional[OCRProvider]:
        """Resolve the configured OCR provider."""
        if self._ocr_provider is not None:
            return self._ocr_provider
        if self._container is not None and self._container.exists("ocr_provider"):
            return self._container.resolve("ocr_provider")
        return None

    # -----------------------------------------------------------------------
    # Command Recognition & Parsing
    # -----------------------------------------------------------------------

    def can_handle(self, command: Any) -> bool:
        """Evaluate whether this skill can handle the given command."""
        op, _, _, _ = self.parse_command(command)
        if op in (
            "capture_screen",
            "read_screen_text",
            "explain_active_window",
            "diagnose_screen_error",
        ):
            return True

        if isinstance(command, str):
            clean = " ".join(command.strip().lower().split())
            if _RE_CAPTURE_SCREEN.match(clean):
                return True
            if _RE_READ_TEXT.match(clean):
                return True
            if _RE_EXPLAIN_WINDOW.match(clean):
                return True
            if _RE_DIAGNOSE_ERROR.match(clean):
                return True

        return False

    def parse_command(
        self, command: Any
    ) -> Tuple[str, Optional[str], Dict[str, Any], Optional[str]]:
        """Parse natural language command or dictionary into structured components."""
        if isinstance(command, dict):
            return super().parse_command(command)

        text = str(command or "").strip()
        clean = " ".join(text.lower().split())

        if _RE_CAPTURE_SCREEN.match(clean):
            return "capture_screen", "screen", {}, None

        if _RE_READ_TEXT.match(clean):
            return "read_screen_text", "active_window", {}, None

        if _RE_EXPLAIN_WINDOW.match(clean):
            return "explain_active_window", "active_window", {}, None

        if _RE_DIAGNOSE_ERROR.match(clean):
            return "diagnose_screen_error", "active_window", {}, None

        return super().parse_command(command)

    # -----------------------------------------------------------------------
    # Asynchronous Execution
    # -----------------------------------------------------------------------

    async def execute_async(self, command: Any) -> SystemSkillResult:
        """Execute vision command asynchronously without blocking calling or GUI threads."""
        return await asyncio.to_thread(self.execute, command)

    # -----------------------------------------------------------------------
    # Core Operation Dispatcher
    # -----------------------------------------------------------------------

    def _execute_operation(
        self, operation: str, target: Optional[str], parameters: Dict[str, Any]
    ) -> Any:
        """Dispatch operation to specific internal handlers."""
        op = operation.strip().lower()

        if op == "capture_screen":
            return self._handle_capture_screen(target, parameters)

        if op == "read_screen_text":
            return self._handle_read_screen_text(target, parameters)

        if op == "explain_active_window":
            return self._handle_explain_active_window(target, parameters)

        if op == "diagnose_screen_error":
            return self._handle_diagnose_screen_error(target, parameters)

        raise SkillExecutionError(f"Unsupported vision operation: '{op}'")

    # -----------------------------------------------------------------------
    # Helper: Target Resolution and Capture via SecureVisionManager
    # -----------------------------------------------------------------------

    def _acquire_observation(
        self,
        target: Optional[str],
        default_target: str = "active_window",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> ScreenObservation:
        """Acquire an authorized ScreenObservation via SecureVisionManager.

        Raises:
            CaptureBlockedError: If security policy denies capture.
            CaptureError: If acquisition fails.
        """
        params = parameters or {}
        chosen_target = (target or params.get("target") or default_target).strip().lower()

        mgr = self.secure_vision_manager
        ttl = params.get("ttl_seconds")
        if ttl is not None:
            try:
                ttl = float(ttl)
            except (ValueError, TypeError):
                ttl = None

        if chosen_target in ("active_window", "window", "active"):
            return mgr.capture_active_window(raise_on_blocked=True, ttl_seconds=ttl)
        else:
            bounds = params.get("bounds")
            if bounds is not None and not isinstance(bounds, WindowBounds):
                if isinstance(bounds, dict):
                    bounds = WindowBounds.from_dict(bounds)
                else:
                    bounds = None
            return mgr.capture_screen(bounds=bounds, source="screen", raise_on_blocked=True, ttl_seconds=ttl)

    # -----------------------------------------------------------------------
    # 1. capture_screen
    # -----------------------------------------------------------------------

    def _handle_capture_screen(
        self, target: Optional[str], parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Capture screen or active window securely, returning metadata only."""
        # capture_screen defaults to full screen unless active_window explicitly requested
        observation = self._acquire_observation(
            target=target,
            default_target="screen",
            parameters=parameters,
        )

        cap = observation.capture
        if cap is None:
            raise CaptureError("Capture completed but produced no observation frame.")

        return {
            "observation_id": observation.observation_id,
            "width": cap.width,
            "height": cap.height,
            "timestamp": cap.timestamp,
            "target": target or "screen",
            "is_active_window": cap.source == "window",
            "bounds": cap.bounds.to_dict() if cap.bounds else None,
            "category": observation.authorization.category.value if observation.authorization else "general",
        }

    # -----------------------------------------------------------------------
    # 2. read_screen_text
    # -----------------------------------------------------------------------

    def _resolve_ocr_provider(self) -> OCRProvider:
        """Resolve the most suitable OCR provider according to the hybrid strategy."""
        # 1. Injected or container OCRProvider
        configured = self.ocr_provider
        if configured is not None and configured.is_available():
            return configured

        # 2. Native Windows OCR
        native_provider = WindowsMediaOCRProvider()
        if native_provider.is_available():
            return native_provider

        # 3. Multimodal AI provider fallback
        ai_p = self.ai_provider
        if ai_p is not None and getattr(ai_p, "supports_multimodal", False):
            return MultimodalVisionOCRAdapter(
                provider=ai_p,
                preprocessor=self._preprocessor,
                prompt_builder=self._prompt_builder,
                container_instance=self._container,
            )

        raise SkillExecutionError(
            "No capable OCR provider is available. Windows native OCR runtime is absent "
            "and active AI provider does not support multimodal vision."
        )

    def _handle_read_screen_text(
        self, target: Optional[str], parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Read text from screen or window using the hybrid OCR pipeline."""
        # read_screen_text defaults to active_window
        observation = self._acquire_observation(
            target=target,
            default_target="active_window",
            parameters=parameters,
        )

        cap = observation.capture
        if cap is None or cap.is_empty:
            raise CaptureError("Screen capture produced an empty frame.")

        provider = self._resolve_ocr_provider()
        ocr_result: OCRResult = provider.extract_text(cap)

        return {
            "text": ocr_result.text,
            "line_count": len([line for line in ocr_result.text.splitlines() if line.strip()]),
            "block_count": len(ocr_result.blocks),
            "observation_id": observation.observation_id,
            "target": target or "active_window",
            "duration_ms": round(ocr_result.duration * 1000.0, 2),
        }

    # -----------------------------------------------------------------------
    # 3. explain_active_window
    # -----------------------------------------------------------------------

    def _handle_explain_active_window(
        self, target: Optional[str], parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Capture active window and generate concise visual explanation via multimodal AI."""
        ai_p = self.ai_provider
        if ai_p is None or not getattr(ai_p, "supports_multimodal", False):
            model_name = getattr(ai_p, "model", "unknown") if ai_p else "none"
            raise SkillExecutionError(
                f"Selected AI provider '{model_name}' does not support multimodal vision inputs."
            )

        # explain_active_window is strictly active_window
        observation = self._acquire_observation(
            target="active_window",
            default_target="active_window",
            parameters=parameters,
        )

        cap = observation.capture
        if cap is None or cap.is_empty:
            raise CaptureError("Failed to capture active window content.")

        # Downsample and convert to ImagePart
        image_part = self._preprocessor.to_image_part(cap)

        prompt = (
            "Provide a clear, concise visual description of this active window. "
            "Summarize what application or document is open and highlight key visible "
            "UI elements, status indicators, or contents. Do not speculate or invent information."
        )

        payload = self._prompt_builder.build_payload(
            query=prompt,
            images=[image_part],
        )

        response = ai_p.generate(payload)
        summary_text = getattr(response, "content", str(response)).strip()

        win_title = cap.metadata.get("window_title") or observation.metadata.get("window_title", "")
        proc_name = observation.metadata.get("process_name")

        return {
            "summary": summary_text,
            "window_title": win_title,
            "process_name": proc_name,
            "observation_id": observation.observation_id,
            "target": "active_window",
        }

    # -----------------------------------------------------------------------
    # 4. diagnose_screen_error
    # -----------------------------------------------------------------------

    def _handle_diagnose_screen_error(
        self, target: Optional[str], parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Capture active window and visually diagnose errors without executing fixes."""
        ai_p = self.ai_provider
        if ai_p is None or not getattr(ai_p, "supports_multimodal", False):
            model_name = getattr(ai_p, "model", "unknown") if ai_p else "none"
            raise SkillExecutionError(
                f"Selected AI provider '{model_name}' does not support multimodal vision inputs."
            )

        observation = self._acquire_observation(
            target=target,
            default_target="active_window",
            parameters=parameters,
        )

        cap = observation.capture
        if cap is None or cap.is_empty:
            raise CaptureError("Failed to capture screen content for error diagnosis.")

        image_part = self._preprocessor.to_image_part(cap)

        prompt = (
            "Analyze this window image for any visible error dialogs, crash messages, warning banners, "
            "or stack traces. Respond with clear structured information:\n"
            "Error Found: [True/False]\n"
            "Error Title: [Brief error title or 'None']\n"
            "Root Cause: [Underlying cause or 'N/A']\n"
            "Recommended Fix: [Suggested safe action or 'None']\n"
            "If no error or warning is present, explicitly state that no clear error was detected."
        )

        payload = self._prompt_builder.build_payload(
            query=prompt,
            images=[image_part],
        )

        response = ai_p.generate(payload)
        raw_text = getattr(response, "content", str(response)).strip()

        # Parse structured output safely
        error_found = True
        lower = raw_text.lower()
        if (
            "error found: false" in lower
            or "no error" in lower
            or "no visible error" in lower
            or "no clear error" in lower
            or "no issues found" in lower
        ):
            error_found = False

        # Extract fields via regex if present
        m_title = re.search(r"Error Title:\s*(.+)", raw_text, re.IGNORECASE)
        m_cause = re.search(r"Root Cause:\s*(.+)", raw_text, re.IGNORECASE)
        m_fix = re.search(r"Recommended Fix:\s*(.+)", raw_text, re.IGNORECASE)

        error_title = m_title.group(1).strip() if m_title else ("Visible Screen Error" if error_found else None)
        root_cause = m_cause.group(1).strip() if m_cause else ("See diagnosis details" if error_found else None)
        recommended_fix = m_fix.group(1).strip() if m_fix else None

        return {
            "error_found": error_found,
            "error_title": error_title,
            "root_cause": root_cause,
            "recommended_fix": recommended_fix,
            "raw_diagnosis": raw_text,
            "observation_id": observation.observation_id,
            "target": target or "active_window",
        }


__all__ = ["VisionSkills"]
