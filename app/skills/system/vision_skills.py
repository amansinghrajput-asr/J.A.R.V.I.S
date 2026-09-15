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

_RE_ASK_SCREEN_PREFIX: Final[re.Pattern[str]] = re.compile(
    r"^(?:ask\s+(?:the\s+|my\s+)?screen|screen\s+query|vqa)\s*:?\s*(.+)$",
    re.IGNORECASE,
)

_RE_VERIFY_SCREEN_PREFIX: Final[re.Pattern[str]] = re.compile(
    r"^(?:verify\s+(?:screen\s+state|on\s+screen|screen)|check\s+screen\s+state)\s*:?\s*(.+)$",
    re.IGNORECASE,
)

_RE_VISUAL_QUESTION: Final[re.Pattern[str]] = re.compile(
    r"^(?:what|which|does|is|are|how|where|check|can\s+you\s+see)\b.*"
    r"\b(?:on\s+(?:my\s+|the\s+)?screen|in\s+(?:this|the|my)\s+(?:window|dialog|terminal|screen)|"
    r"shown\s+(?:on|in)\s+(?:the\s+|my\s+)?(?:screen|window|terminal|dialog)|"
    r"visible\s+on\s+(?:my\s+|the\s+)?screen|"
    r"dialog\s+say|terminal\s+show|download\s+(?:finished|complete))\b.*?\??$",
    re.IGNORECASE,
)

_RE_TEMPORAL_QUERY: Final[re.Pattern[str]] = re.compile(
    r"\b(?:"
    r"now|currently|current|latest|right\s+now|at\s+this\s+moment|"
    r"what\s+changed|what\s+just\s+changed|did\s+anything\s+change|has\s+it\s+changed|has\s+anything\s+changed|"
    r"what\s+is\s+happening\s+(?:right\s+)?now|what's\s+happening\s+(?:right\s+)?now|"
    r"is\s+it\s+still\s+there|"
    r"did\s+that\s+work|did\s+it\s+work|did\s+it\s+open|did\s+the\s+page\s+open|did\s+the\s+app\s+open|"
    r"is\s+it\s+happening\s+now"
    r")\b",
    re.IGNORECASE,
)

_RE_VERIFICATION_QUESTION: Final[re.Pattern[str]] = re.compile(
    r"^(?:did\s+(?:it|that|the\s+page|the\s+app|the\s+window)\s+(?:open|work|load|finish)|"
    r"verify\s+screen\s+state)\??$",
    re.IGNORECASE,
)

_RE_VISUAL_FOLLOWUP: Final[re.Pattern[str]] = re.compile(
    r"^(?:"
    r"(?:what|how)\s+about\s+(?:the|that|this)\s+(?:button|icon|text|error|link|dialog|window|box|menu|tab|field|input|label|image|checkbox|dropdown|message|panel|one\s+on\s+the\s+(?:right|left|top|bottom)|right|left|top|bottom)[\w\s]*\??|"
    r"and\s+(?:the|that|this)\s+(?:button|icon|text|error|link|dialog|window|box|menu|tab|field|input|label|image|message|one|part)[\w\s]*\??|"
    r"what\s+does\s+(?:that|this|it)\s+(?:say|read)\??|"
    r"where\s+is\s+(?:that|the|this)\s+(?:button|icon|error|link|dialog|window|box|menu|tab|text)[\w\s]*\??|"
    r"can\s+you\s+see\s+(?:the|that|this)\s+(?:error|button|icon|text|dialog|warning|link|message)[\w\s]*\??|"
    r"is\s+(?:that|this)\s+(?:the|a|an)?\s*[\w\s]*(?:button|link|icon|error|dialog|window|text|box|menu|tab)[\w\s]*\??|"
    r"what\s+is\s+that\s+(?:button|icon|error|link|text|menu|dialog|box|window|below|above|on\s+the\s+(?:right|left|top|bottom))[\w\s]*\??|"
    r"what\s+color\s+is\s+(?:that|this|the)\s+[\w\s]+|"
    r"read\s+(?:that|this)\s*(?:text|part|section|dialog|box)?\??|"
    r"tell\s+me\s+(?:more\s+)?about\s+(?:that|this)\s+(?:button|icon|text|error|link|dialog|window|box|menu|tab)[\w\s]*"
    r")$",
    re.IGNORECASE,
)

_UI_MUTATING_OPERATIONS: Final[frozenset[str]] = frozenset({
    "focus_window",
    "minimize_window",
    "maximize_window",
    "restore_window",
    "close_window",
    "open_app",
    "launch_app",
    "close_app",
    "restart_app",
    "open_url",
    "browse_url",
    "open_link",
    "search_web",
    "open_browser",
    "type_text",
    "send_keys",
    "click",
    "double_click",
    "press_key",
    "hotkey",
    "lock_workstation",
})



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
        self._last_qa: Optional[Dict[str, Any]] = None
        self._event_listeners_setup: bool = False
        self._event_bus_subscribed: bool = False
        self._planner_bus_subscribed: bool = False
        self._setup_event_listeners()

    def bind_system_services(
        self,
        *,
        security_policy: Optional[SystemSecurityPolicy] = None,
        confirmation_manager: Optional[SystemConfirmationManager] = None,
        planner_event_bus: Optional[Any] = None,
    ) -> None:
        """Bind system-specific security and confirmation dependencies."""
        super().bind_system_services(
            security_policy=security_policy,
            confirmation_manager=confirmation_manager,
            planner_event_bus=planner_event_bus,
        )
        self._setup_event_listeners()

    def _setup_event_listeners(self) -> None:
        """Subscribe to system/planner events for automatic cache invalidation on UI mutations."""
        with self._lock:
            bus = self._event_bus
            p_bus = self.planner_event_bus

            # Return immediately if listeners are already successfully registered for all active buses
            if self._event_listeners_setup and (bus is None or self._event_bus_subscribed) and (p_bus is None or self._planner_bus_subscribed):
                return

            # Register EventBus listeners once
            if bus is not None and hasattr(bus, "subscribe") and not self._event_bus_subscribed:
                try:
                    bus.subscribe("system.skill_completed", self._on_event_bus_skill_completed)
                    bus.subscribe("*", self._on_event_bus_mutation)
                    self._event_bus_subscribed = True
                except Exception as exc:
                    self.logger.debug("Could not subscribe to EventBus: %s", exc)

            # Register PlannerEventBus listener once
            if p_bus is not None and hasattr(p_bus, "subscribe") and not self._planner_bus_subscribed:
                try:
                    from app.ai.planner.events import SystemSkillCompleted
                    p_bus.subscribe(SystemSkillCompleted, self._on_planner_skill_completed)
                    self._planner_bus_subscribed = True
                except Exception as exc:
                    self.logger.debug("Could not subscribe to PlannerEventBus: %s", exc)

            # Set the completion guard only after required active subscriptions have actually succeeded
            if (self._event_bus_subscribed and bus is not None) or (self._planner_bus_subscribed and p_bus is not None):
                if (bus is None or self._event_bus_subscribed) and (p_bus is None or self._planner_bus_subscribed):
                    self._event_listeners_setup = True

    def _on_planner_skill_completed(self, event: Any) -> None:
        """Handle SystemSkillCompleted on PlannerEventBus to invalidate cache on UI mutation."""
        op = str(getattr(event, "operation", "") or "").lower()
        skill_name = str(getattr(event, "skill_name", "") or "").lower()
        if skill_name != "vision" and (op in _UI_MUTATING_OPERATIONS or skill_name in ("window", "app", "browser", "system_control")):
            self.invalidate_observation_cache(reason=f"planner_skill_completed:{skill_name}:{op}")

    def _on_event_bus_skill_completed(self, event: Any) -> None:
        """Handle system.skill_completed on EventBus to invalidate cache on UI mutation."""
        payload = getattr(event, "payload", {}) if hasattr(event, "payload") else (event if isinstance(event, dict) else {})
        op = str(payload.get("operation") or "").lower()
        skill = str(payload.get("skill_name") or payload.get("skill") or "").lower()
        if skill != "vision" and (op in _UI_MUTATING_OPERATIONS or skill in ("window", "app", "browser", "system_control")):
            self.invalidate_observation_cache(reason=f"event_bus_completed:{skill}:{op}")

    def _on_event_bus_mutation(self, event: Any) -> None:
        """Handle window.*, app.*, or system.* mutation events on EventBus."""
        name = str(getattr(event, "name", "") if hasattr(event, "name") else "").lower()
        if name.startswith(("window.", "app.", "system.")):
            self.invalidate_observation_cache(reason=f"event_bus_mutation:{name}")

    def invalidate_observation_cache(self, reason: str = "explicit_invalidation") -> None:
        """Explicitly invalidate cached screen observations and conversational visual context."""
        with self._lock:
            self._last_qa = None
            self.secure_vision_manager.invalidate_cache(reason=reason)

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
        if hasattr(command, "raw_command") or hasattr(command, "normalized_command"):
            cmd_text = getattr(command, "normalized_command", "") or getattr(command, "raw_command", "")
            return self.can_handle(str(cmd_text))

        op, _, _, _ = self.parse_command(command)
        if op in (
            "capture_screen",
            "read_screen_text",
            "explain_active_window",
            "diagnose_screen_error",
            "ask_screen",
            "verify_screen_state",
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
            if _RE_ASK_SCREEN_PREFIX.match(clean):
                return True
            if _RE_VERIFY_SCREEN_PREFIX.match(clean):
                return True
            if _RE_VISUAL_QUESTION.match(clean):
                return True
            if _RE_VERIFICATION_QUESTION.match(clean):
                return True
            if _RE_VISUAL_FOLLOWUP.match(clean):
                # Follow-up questions require an active, unexpired, valid observation in ephemeral cache
                try:
                    mgr = self.secure_vision_manager
                    latest = mgr.get_latest_observation()
                    if latest is not None and latest.is_valid and not latest.is_expired:
                        return True
                except Exception:
                    pass
                return False

        return False

    def parse_command(
        self, command: Any
    ) -> Tuple[str, Optional[str], Dict[str, Any], Optional[str]]:
        """Parse natural language command or dictionary into structured components."""
        if hasattr(command, "raw_command") or hasattr(command, "normalized_command"):
            raw_text = getattr(command, "raw_command", "") or ""
            norm_text = getattr(command, "normalized_command", "") or raw_text
            text = str(raw_text or norm_text).strip()
        elif isinstance(command, dict):
            return super().parse_command(command)
        else:
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

        m_ask = _RE_ASK_SCREEN_PREFIX.match(clean)
        if m_ask:
            q = m_ask.group(1).strip()
            return "ask_screen", "active_window", {"question": q}, None

        m_ver = _RE_VERIFY_SCREEN_PREFIX.match(clean)
        if m_ver:
            c = m_ver.group(1).strip()
            return "verify_screen_state", "active_window", {"condition": c}, None

        if _RE_VERIFICATION_QUESTION.match(clean):
            cond = text
            if "page" in clean and "open" in clean:
                cond = "the requested page is open"
            elif "app" in clean or "window" in clean:
                cond = "the requested window or application is open"
            elif "work" in clean:
                cond = "the previous action or operation worked"
            elif "open" in clean:
                cond = "it opened successfully"
            return "verify_screen_state", "active_window", {"condition": cond}, None

        if _RE_VISUAL_QUESTION.match(clean):
            return "ask_screen", "active_window", {"question": text}, None

        if _RE_VISUAL_FOLLOWUP.match(clean):
            try:
                latest = self.secure_vision_manager.get_latest_observation()
                if latest is not None and latest.is_valid and not latest.is_expired:
                    return "ask_screen", "active_window", {"question": text, "reuse_cache": True, "is_followup": True}, None
            except Exception:
                pass

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

        if op == "ask_screen":
            return self._handle_ask_screen(target, parameters)

        if op == "verify_screen_state":
            return self._handle_verify_screen_state(target, parameters)

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

    def _acquire_observation_with_reuse(
        self,
        target: Optional[str] = "active_window",
        default_target: str = "active_window",
        parameters: Optional[Dict[str, Any]] = None,
        allow_reuse: bool = False,
        query_text: Optional[str] = None,
    ) -> Tuple[ScreenObservation, bool]:
        """Acquire a screen observation, safely reusing an existing observation if permitted.

        Reuse conditions:
        1. Explicitly requested via allow_reuse or parameters['reuse_cache'] == True.
        2. Query does not contain temporal keywords (now, currently, latest, etc.).
        3. Current foreground window is verified and authorized by VisionSecurityPolicy.
        4. Observation is <= 5.0 seconds old and not expired.
        5. Active window identity (hwnd, title, process, bounds) matches cached metadata.

        Returns:
            Tuple of (ScreenObservation, was_reused: bool).
        """
        params = parameters or {}
        reuse_requested = bool(allow_reuse or params.get("reuse_cache", False))
        obs_id = params.get("observation_id")
        mgr = self.secure_vision_manager

        # Explicit fresh capture requests or temporal queries MUST force fresh capture
        if bool(params.get("force_fresh") or params.get("fresh")):
            reuse_requested = False
            obs_id = None
        elif query_text and _RE_TEMPORAL_QUERY.search(query_text):
            reuse_requested = False
            obs_id = None

        # 1. If explicit observation_id provided (e.g. planner intra-step)
        if obs_id:
            auth = mgr.check_active_window_authorized(source="window")
            if not auth.is_allowed:
                raise CaptureBlockedError(auth.reason, authorization=auth)

            cached = mgr.get_observation(str(obs_id).strip())
            if cached is not None and cached.is_valid:
                age = time.time() - cached.timestamp
                if age <= 5.0:
                    return (cached, True)

        # 2. If general observation reuse requested
        if reuse_requested:
            # Enforce security check on current foreground window first
            auth = mgr.check_active_window_authorized(source="window")
            if not auth.is_allowed:
                raise CaptureBlockedError(auth.reason, authorization=auth)

            cached = mgr.get_latest_observation()
            if cached is not None and cached.is_valid:
                age = time.time() - cached.timestamp
                if age <= 5.0:
                    # Verify compound window identity
                    curr_hwnd, curr_title, curr_proc, curr_bounds = mgr.get_active_window_identity()
                    cached_hwnd = cached.metadata.get("hwnd")
                    cached_title = cached.metadata.get("window_title")
                    cached_proc = cached.metadata.get("process_name")
                    cached_bounds = cached.metadata.get("bounds")

                    if (
                        curr_hwnd == cached_hwnd
                        and curr_title == cached_title
                        and curr_proc == cached_proc
                        and curr_bounds == cached_bounds
                    ):
                        return (cached, True)

        # 3. Default: Fresh authorized capture
        fresh = self._acquire_observation(
            target=target,
            default_target=default_target,
            parameters=parameters,
        )
        return (fresh, False)

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

    # -----------------------------------------------------------------------
    # 5. ask_screen (Visual Question Answering)
    # -----------------------------------------------------------------------

    def _handle_ask_screen(
        self, target: Optional[str], parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Answer arbitrary natural-language question using authorized visual observation."""
        question = str(
            parameters.get("question")
            or parameters.get("query")
            or parameters.get("prompt")
            or target
            or ""
        ).strip()
        if not question:
            raise SkillExecutionError("No question provided for ask_screen operation.")

        ai_p = self.ai_provider
        if ai_p is None or not getattr(ai_p, "supports_multimodal", False):
            model_name = getattr(ai_p, "model", "unknown") if ai_p else "none"
            raise SkillExecutionError(
                f"Selected AI provider '{model_name}' does not support multimodal vision inputs."
            )

        allow_reuse = bool(parameters.get("reuse_cache", False))
        observation, reused = self._acquire_observation_with_reuse(
            target=target,
            default_target="active_window",
            parameters=parameters,
            allow_reuse=allow_reuse,
            query_text=question,
        )

        cap = observation.capture
        if cap is None or cap.is_empty:
            raise CaptureError("Failed to capture screen content for question answering.")

        image_part = self._preprocessor.to_image_part(cap)

        # Check for contextual previous discussion if reusing observation
        prev_ctx = parameters.get("previous_context") or parameters.get("context")
        prev_q = None
        prev_a = None
        if isinstance(prev_ctx, dict):
            prev_q = prev_ctx.get("question")
            prev_a = prev_ctx.get("answer") or prev_ctx.get("content")
        elif isinstance(prev_ctx, str):
            prev_a = prev_ctx
        elif reused and self._last_qa and self._last_qa.get("observation_id") == observation.observation_id:
            prev_q = self._last_qa.get("question")
            prev_a = self._last_qa.get("answer")

        if prev_q or prev_a:
            context_block = "Previous Visual Discussion:\n"
            if prev_q:
                context_block += f"User: {prev_q}\n"
            if prev_a:
                context_block += f"Assistant: {prev_a}\n"

            prompt = (
                "You are J.A.R.V.I.S Vision Intelligence. The user is asking a follow-up question about the provided screenshot:\n\n"
                f"{context_block}\n"
                f"Current Follow-up Question: {question}\n\n"
                "Instructions:\n"
                "- Answer the follow-up question using visible evidence in the screenshot and the previous context.\n"
                "- Avoid speculating, guessing, or inventing information.\n"
                "- If the requested information is not clearly visible, state: 'I can't determine that from the visible screen.'\n"
                "- Distinguish clearly between direct visual observation and inference.\n"
                "- Keep your response concise, accurate, and speakable.\n"
                "- Directly answer the user's question."
            )
        else:
            prompt = (
                "You are J.A.R.V.I.S Vision Intelligence. Answer the following question about the provided screenshot:\n\n"
                f"Question: {question}\n\n"
                "Instructions:\n"
                "- Answer ONLY based on visible evidence in the image.\n"
                "- Avoid speculating, guessing, or inventing information.\n"
                "- If the requested information is not clearly visible, state: 'I can't determine that from the visible screen.'\n"
                "- Distinguish clearly between direct visual observation and inference.\n"
                "- Keep your response concise, accurate, and speakable.\n"
                "- Directly answer the user's question rather than describing the entire screen unnecessarily."
            )

        payload = self._prompt_builder.build_payload(
            query=prompt,
            images=[image_part],
        )

        response = ai_p.generate(payload)
        answer_text = getattr(response, "content", str(response)).strip()

        # Update ephemeral single-turn visual QA context (safe textual metadata only)
        with self._lock:
            self._last_qa = {
                "observation_id": observation.observation_id,
                "question": question,
                "answer": answer_text,
                "timestamp": time.time(),
            }

        win_title = cap.metadata.get("window_title") or observation.metadata.get("window_title", "")
        proc_name = observation.metadata.get("process_name")

        return {
            "question": question,
            "answer": answer_text,
            "observation_id": observation.observation_id,
            "target": target or "active_window",
            "window_title": win_title,
            "process_name": proc_name,
            "reused_cache": reused,
            "has_context": bool(prev_q or prev_a),
        }

    # -----------------------------------------------------------------------
    # 6. verify_screen_state (Planner Visual State Verification)
    # -----------------------------------------------------------------------

    def _handle_verify_screen_state(
        self, target: Optional[str], parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Verify whether an expected visual state or condition is met using a fresh screen observation."""
        condition = str(
            parameters.get("condition")
            or parameters.get("expected")
            or parameters.get("state")
            or target
            or ""
        ).strip()
        if not condition:
            raise SkillExecutionError("No condition specified for visual state verification.")

        ai_p = self.ai_provider
        if ai_p is None or not getattr(ai_p, "supports_multimodal", False):
            model_name = getattr(ai_p, "model", "unknown") if ai_p else "none"
            raise SkillExecutionError(
                f"Selected AI provider '{model_name}' does not support multimodal vision inputs."
            )

        # MANDATORY FRESH CAPTURE: verify_screen_state never reuses stale cache
        observation = self._acquire_observation(
            target=target,
            default_target="active_window",
            parameters=parameters,
        )

        cap = observation.capture
        if cap is None or cap.is_empty:
            raise CaptureError("Failed to capture screen content for visual verification.")

        image_part = self._preprocessor.to_image_part(cap)

        prompt = (
            "You are J.A.R.V.I.S Visual Verification Engine. Inspect the provided screenshot and determine "
            f"whether the following condition is true:\n\n"
            f"Condition: {condition}\n\n"
            "Instructions:\n"
            "1. Assess whether the visible evidence confirms or refutes this condition.\n"
            "2. Your response MUST begin with exactly one of these three verdicts:\n"
            "   - VERIFIED: <concise explanation of why the condition is met>\n"
            "   - NOT VERIFIED: <concise explanation of why the condition is not met>\n"
            "   - UNCERTAIN: <concise explanation of why visible evidence is insufficient>\n"
            "3. Be concise, direct, and factual. Base your judgment strictly on visible evidence."
        )

        payload = self._prompt_builder.build_payload(
            query=prompt,
            images=[image_part],
        )

        response = ai_p.generate(payload)
        content = getattr(response, "content", str(response)).strip()

        verified: Optional[bool] = None
        status = "uncertain"
        reason = content

        upper = content.upper()
        if upper.startswith("VERIFIED"):
            verified = True
            status = "verified"
            parts = content.split(":", 1)
            reason = parts[1].strip() if len(parts) > 1 else content
        elif upper.startswith("NOT VERIFIED"):
            verified = False
            status = "not_verified"
            parts = content.split(":", 1)
            reason = parts[1].strip() if len(parts) > 1 else content
        elif upper.startswith("UNCERTAIN") or "CANNOT DETERMINE" in upper or "UNABLE TO DETERMINE" in upper:
            verified = None
            status = "uncertain"
            parts = content.split(":", 1)
            reason = parts[1].strip() if len(parts) > 1 else content
        else:
            if "IS VERIFIED" in upper or "CONDITION IS MET" in upper:
                verified = True
                status = "verified"
            elif "NOT VERIFIED" in upper or "CONDITION IS NOT MET" in upper:
                verified = False
                status = "not_verified"
            else:
                verified = None
                status = "uncertain"

        win_title = cap.metadata.get("window_title") or observation.metadata.get("window_title", "")
        proc_name = observation.metadata.get("process_name")

        return {
            "condition": condition,
            "verified": verified,
            "status": status,
            "reason": reason,
            "observation_id": observation.observation_id,
            "target": target or "active_window",
            "window_title": win_title,
            "process_name": proc_name,
            "reused_cache": False,
        }


__all__ = ["VisionSkills"]
