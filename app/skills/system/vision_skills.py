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
from app.vision.delta import VisualDeltaEngine
from app.vision.grounding import VisualGroundingEngine
from app.vision.models import (
    BufferExpiredError,
    CaptureBlockedError,
    CaptureError,
    FormField,
    ControlAffordance,
    ControlVisualState,
    ElementAffordance,
    OCRResult,
    SceneQueryAnswer,
    ScreenCapture,
    ScreenObservation,
    UIContainer,
    UIContainerType,
    UIElement,
    UIElementChange,
    UIElementType,
    UIScene,
    UnsupportedPlatformError,
    VisionError,
    VisionSecurityError,
    VisualActionFeasibilityStatus,
    VisualActionGroundingResult,
    VisualActionSafetyTier,
    VisualActionTarget,
    VisualActionType,
    VisualDeltaResult,
    VisualDeltaType,
    VisualElementTrack,
    VisualEvidenceItem,
    VisualGoalCriterion,
    VisualGoalSpec,
    VisualGroundingResult,
    VisualOutcomeType,
    VisualSituation,
    VisualSituationResult,
    VisualSituationType,
    VisualTemporalEvent,
    VisualTemporalHistoryResult,
    VisualTrackStatus,
    VisualTrackingResult,
    VisualVerificationResult,
    WindowBounds,
)
from app.vision.action_grounding import VisualActionGroundingEngine
from app.vision.affordance import VisualAffordanceEngine
from app.vision.scene import VisualSceneParser
from app.vision.situation import VisualSituationEngine
from app.vision.temporal import VisualTemporalEngine
from app.vision.tracking import VisualTrackingEngine
from app.vision.verification import VisualVerificationEngine, parse_visual_goal
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

_RE_DETECT_CHANGE: Final[re.Pattern[str]] = re.compile(
    r"^(?:"
    r"what(?:\s+has|\s+is|\s+'s)?\s+changed(?:\s+on(?:\s+my|\s+the)?\s+screen|\s+in(?:\s+this|\s+the)?\s+window)?\??|"
    r"what\s+just\s+changed\??|"
    r"what\s+changed(?:\s+after\s+that)?\??|"
    r"did(?:\s+the|\s+my)?\s+screen\s+change\??|"
    r"did\s+anything\s+change\??|"
    r"has\s+anything\s+changed\??|"
    r"what\s+happened\s+after\s+that\??|"
    r"detect\s+(?:screen\s+)?changes?"
    r")$",
    re.IGNORECASE,
)

_RE_VERIFY_CHANGE: Final[re.Pattern[str]] = re.compile(
    r"^(?:"
    r"did\s+(?:the\s+|a\s+|an\s+)?(.+?)\s+(appear|disappear|vanish|move|change)\??|"
    r"is\s+(?:the\s+|a\s+|an\s+)?(.+?)\s+still\s+(?:there|visible|present)\??|"
    r"did\s+(?:the\s+)?UI\s+element\s+move\??|"
    r"did\s+(?:the\s+)?button\s+appear\??|"
    r"did\s+(?:the\s+)?popup\s+disappear\??"
    r")$",
    re.IGNORECASE,
)

_RE_MAP_SCENE: Final[re.Pattern[str]] = re.compile(
    r"^(?:"
    r"map\s+(?:the\s+|my\s+)?(?:ui\s+)?(?:active\s+)?(?:scene|layout|controls?|screen|window|elements?)(?:\s+(?:of|on|in)\s+(?:the\s+|this\s+|my\s+)?(?:screen|window|application|app))?|"
    r"map\s+(?:the\s+|my\s+)?ui(?:\s+(?:of|on|in)\s+(?:the\s+|this\s+|my\s+)?(?:screen|window|application|app))?|"
    r"what\s+(?:interactive\s+)?(?:controls?|elements?|buttons?|inputs?|fields?|buttons?\s+and\s+(?:input\s+)?fields?)\s+are\s+(?:available\s+|present\s+|here\s+)?(?:on\s+(?:this\s+|my\s+|the\s+)?screen|in\s+(?:this\s+|the\s+|my\s+)?(?:window|app|application))?\??|"
    r"list\s+(?:the\s+|all\s+)?(?:interactive\s+)?(?:controls?|elements?|buttons?|inputs?|form\s+fields?|buttons?\s+and\s+inputs?)(?:\s+(?:on|in)\s+(?:the\s+|this\s+|my\s+)?(?:screen|window|application|app))?|"
    r"what\s+forms?(?:\s+or\s+dialogs?)?\s+are\s+(?:open|available|present|visible)\??|"
    r"what\s+buttons?\s+and\s+(?:input\s+)?fields?\s+are\s+(?:here|available|on\s+screen)\??|"
    r"what\s+interactive\s+elements\s+are\s+here\??|"
    r"inspect\s+(?:the\s+)?(?:ui\s+)?(?:layout|scene|controls?)"
    r")$",
    re.IGNORECASE,
)

_RE_INSPECT_CONTROL_STATE: Final[re.Pattern[str]] = re.compile(
    r"^(?:"
    r"is\s+(?:the\s+|a\s+|an\s+)?(.+?)\s+(?:button|control|checkbox|field|input)\s+(enabled|disabled|checked|unchecked|empty|populated|focused|editable)\??|"
    r"is\s+(?:the\s+|a\s+|an\s+)?(submit|save|cancel|ok|apply|next|previous|login|continue|register)\s+button\s+(enabled|disabled|clickable)\??|"
    r"is\s+(?:the\s+|a\s+|an\s+)?(.+?)\s+(checked|unchecked|empty|populated|focused|editable)\??|"
    r"what\s+is\s+the\s+state\s+of\s+(?:the\s+|this\s+)?(.+?)\??|"
    r"inspect\s+(?:the\s+)?(?:control\s+state|state\s+of)\s+(.+)"
    r")$",
    re.IGNORECASE,
)

_RE_QUERY_SCENE_STATE: Final[re.Pattern[str]] = re.compile(
    r"^(?:"
    r"can\s+i\s+submit\s+(?:this\s+|the\s+)?form\??|"
    r"is\s+(?:the\s+|this\s+)?form\s+complete\??|"
    r"which\s+(?:required\s+)?(?:fields?|inputs?)\s+are\s+(?:empty|incomplete|missing)\??|"
    r"query\s+(?:the\s+)?(?:scene\s+state|ui\s+state)\s*:?\s*(.+)"
    r")$",
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

_RE_RELATIONAL_LOCATE: Final[re.Pattern[str]] = re.compile(
    r"^(?:"
    r"(?:where\s+is\s+(?:the\s+|a\s+|an\s+)|find\s+(?:the\s+|a\s+|an\s+)?|locate\s+(?:the\s+|a\s+|an\s+)?|show\s+me\s+where\s+(?:the\s+|a\s+|an\s+)?)(.+?)\s+"
    r"|"
    r"which\s+(.+?)\s+is\s+"
    r")"
    r"(to\s+the\s+left\s+of|left\s+of|to\s+the\s+right\s+of|right\s+of|above|on\s+top\s+of|below|under|underneath|beneath|inside|within|near|close\s+to|next\s+to|beside)\s+"
    r"(.+?)(?:\s+(?:is|located|at))?\??$",
    re.IGNORECASE,
)

_RE_LOCATE_ELEMENT: Final[re.Pattern[str]] = re.compile(
    r"^(?:where\s+is\s+(?:the\s+|a\s+|an\s+)|"
    r"find\s+(?:the\s+|a\s+|an\s+)?|"
    r"locate\s+(?:the\s+|a\s+|an\s+)?|"
    r"show\s+me\s+where\s+(?:the\s+|a\s+|an\s+)?)"
    r"(.+?)(?:\s+(?:is|located|at))?\??$",
    re.IGNORECASE,
)

_RE_TRACK_ELEMENT: Final[re.Pattern[str]] = re.compile(
    r"^(?:"
    r"track\s+(?:the\s+|this\s+|a\s+|an\s+)?(.+?)(?:\s+element|\s+button|\s+control)?|"
    r"where\s+did\s+(?:the\s+|this\s+|a\s+|an\s+)?(.+?)\s+move\??|"
    r"where\s+is\s+the\s+tracked\s+(.+?)\??|"
    r"is\s+it\s+the\s+same\s+(.+?)\??|"
    r"did\s+(?:this\s+|the\s+)?control\s+change\s+position\??"
    r")$",
    re.IGNORECASE,
)

_RE_GET_TRACKS: Final[re.Pattern[str]] = re.compile(
    r"^(?:"
    r"get\s+(?:all\s+)?(?:visual\s+)?tracks?|"
    r"list\s+(?:all\s+)?(?:visual\s+)?tracks?|"
    r"what\s+(?:elements\s+are\s+being\s+|is\s+being\s+)?tracked\??|"
    r"show\s+(?:all\s+)?(?:tracked\s+elements|tracks)"
    r")$",
    re.IGNORECASE,
)

_RE_GET_RECENT_EVENTS: Final[re.Pattern[str]] = re.compile(
    r"^(?:"
    r"what\s+happened\s+recently\??|"
    r"what\s+just\s+changed\s+on\s+(?:my\s+|the\s+)?screen\??|"
    r"show\s+(?:recent\s+)?(?:visual\s+)?events?|"
    r"get\s+recent\s+(?:visual\s+)?events?|"
    r"recent\s+(?:visual\s+)?(?:events?|changes?)"
    r")$",
    re.IGNORECASE,
)

_RE_QUERY_EVENT_HISTORY: Final[re.Pattern[str]] = re.compile(
    r"^(?:"
    r"what\s+happened\s+to\s+(?:the\s+|this\s+|a\s+|an\s+)?(.+?)\??|"
    r"did\s+any\s+(?:buttons?|controls?|elements?)\s+change\s+state\??|"
    r"did\s+(?:the\s+|this\s+)?(.+?)\s+change\s+state\??|"
    r"what\s+elements?\s+disappeared\??|"
    r"show\s+history\s+for\s+(?:the\s+|this\s+)?(.+?)\??"
    r")$",
    re.IGNORECASE,
)

_RE_GET_VISUAL_SITUATION: Final[re.Pattern[str]] = re.compile(
    r"^(?:"
    r"what\s+is\s+happening\s+on\s+(?:my\s+|the\s+)?screen\??|"
    r"what\s+is\s+happening\s+right\s+now\??|"
    r"what\s+is\s+happening\??|"
    r"what(?:'s|\s+is)\s+(?:the\s+)?current\s+situation\??|"
    r"what\s+is\s+the\s+situation(?:\s+on\s+(?:my\s+|the\s+)?screen)?\??|"
    r"is\s+there\s+a\s+popup\??|"
    r"is\s+something\s+blocking\s+(?:the\s+|my\s+)?screen\??|"
    r"(?:get\s+)?visual\s+situation|"
    r"situation\s+summary|"
    r"what\s+is\s+going\s+on\s+on\s+(?:my\s+|the\s+)?screen\??|"
    r"describe\s+(?:the\s+)?current\s+situation"
    r")$",
    re.IGNORECASE,
)

_RE_GROUND_ACTION: Final[re.Pattern[str]] = re.compile(
    r"^(?:"
    r"where\s+should\s+i\s+click(?:\s+to\s+(.+?))?\??|"
    r"which\s+button\s+should\s+i\s+click(?:\s+to\s+(.+?))?\??|"
    r"can\s+i\s+submit(?:\s+this|\s+the)?\s+form\??|"
    r"is\s+this\s+button\s+ready\??|"
    r"how\s+do\s+i\s+dismiss\s+(?:this\s+|the\s+)?(?:popup|modal|dialog)\??|"
    r"locate\s+(?:the\s+)?control\s+for\s+(.+?)\??|"
    r"identify\s+(?:the\s+)?control\s+for\s+(.+?)\??|"
    r"ground\s+(?:visual\s+)?action(?:\s+(.+?))?\??|"
    r"can\s+i\s+click\s+(.+?)\??"
    r")$",
    re.IGNORECASE,
)


def _normalize_relation_phrase(phrase: str) -> str:
    """Map natural language relation phrases to standardized SpatialRelation string value."""
    p = phrase.strip().lower()
    if p in ("to the left of", "left of", "left"):
        return "left_of"
    if p in ("to the right of", "right of", "right"):
        return "right_of"
    if p in ("above", "on top of", "over"):
        return "above"
    if p in ("below", "under", "underneath", "beneath"):
        return "below"
    if p in ("inside", "within", "in"):
        return "inside"
    if p in ("near", "close to", "next to", "beside"):
        return "near"
    return p

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
        if self._security_policy is not None and hasattr(self._security_policy, "_safe_operations"):
            self._security_policy._safe_operations.add("detect_screen_change")
            self._security_policy._safe_operations.add("map_ui_scene")
            self._security_policy._safe_operations.add("inspect_control_state")
            self._security_policy._safe_operations.add("query_scene_state")
            self._security_policy._safe_operations.add("verify_goal")
            self._security_policy._safe_operations.add("verify_screen_state")
            self._security_policy._safe_operations.add("track_elements")
            self._security_policy._safe_operations.add("get_visual_tracks")
            self._security_policy._safe_operations.add("get_recent_events")
            self._security_policy._safe_operations.add("query_event_history")
            self._security_policy._safe_operations.add("get_visual_situation")
            self._security_policy._safe_operations.add("ground_visual_action")
        self._lock = threading.RLock()
        self._secure_vision_manager = secure_vision_manager
        self._ocr_provider = ocr_provider
        self._ai_provider = ai_provider
        self._preprocessor = preprocessor or default_preprocessor
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._verification_engine: Optional[VisualVerificationEngine] = None
        self._tracking_engine: Optional[VisualTrackingEngine] = None
        self._temporal_engine: Optional[VisualTemporalEngine] = None
        self._situation_engine: Optional[VisualSituationEngine] = None
        self._action_grounding_engine: Optional[VisualActionGroundingEngine] = None
        self._last_qa: Optional[Dict[str, Any]] = None
        self._event_listeners_setup: bool = False
        self._event_bus_subscribed: bool = False
        self._planner_bus_subscribed: bool = False
        self._setup_event_listeners()

    @property
    def verification_engine(self) -> VisualVerificationEngine:
        """Lazily initialize and return the VisualVerificationEngine instance."""
        with self._lock:
            if self._verification_engine is None:
                self._verification_engine = VisualVerificationEngine(
                    ocr_provider=self._ocr_provider,
                    ai_provider=self._ai_provider,
                    preprocessor=self._preprocessor,
                    prompt_builder=self._prompt_builder,
                )
            return self._verification_engine

    @property
    def tracking_engine(self) -> VisualTrackingEngine:
        """Lazily initialize and return the VisualTrackingEngine instance."""
        with self._lock:
            if self._tracking_engine is None:
                self._tracking_engine = VisualTrackingEngine(
                    logger_instance=self.logger,
                )
            return self._tracking_engine

    @property
    def temporal_engine(self) -> VisualTemporalEngine:
        """Lazily initialize and return the VisualTemporalEngine instance."""
        with self._lock:
            if self._temporal_engine is None:
                self._temporal_engine = VisualTemporalEngine()
            return self._temporal_engine

    @property
    def situation_engine(self) -> VisualSituationEngine:
        """Lazily initialize and return the VisualSituationEngine instance."""
        with self._lock:
            if self._situation_engine is None:
                self._situation_engine = VisualSituationEngine(
                    logger_instance=self.logger,
                )
            return self._situation_engine

    @property
    def action_grounding_engine(self) -> VisualActionGroundingEngine:
        """Lazily initialize and return the VisualActionGroundingEngine instance."""
        with self._lock:
            if self._action_grounding_engine is None:
                self._action_grounding_engine = VisualActionGroundingEngine(
                    logger_instance=self.logger,
                )
            return self._action_grounding_engine

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
        if self._security_policy is not None and hasattr(self._security_policy, "_safe_operations"):
            self._security_policy._safe_operations.add("detect_screen_change")
            self._security_policy._safe_operations.add("map_ui_scene")
            self._security_policy._safe_operations.add("inspect_control_state")
            self._security_policy._safe_operations.add("query_scene_state")
            self._security_policy._safe_operations.add("verify_goal")
            self._security_policy._safe_operations.add("verify_screen_state")
            self._security_policy._safe_operations.add("track_elements")
            self._security_policy._safe_operations.add("get_visual_tracks")
            self._security_policy._safe_operations.add("get_recent_events")
            self._security_policy._safe_operations.add("query_event_history")
            self._security_policy._safe_operations.add("get_visual_situation")
            self._security_policy._safe_operations.add("ground_visual_action")
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
    def security_policy(self) -> SystemSecurityPolicy:
        """Retrieve active security policy, ensuring vision operations are classified as SAFE."""
        policy = super().security_policy
        if hasattr(policy, "_safe_operations"):
            policy._safe_operations.add("detect_screen_change")
            policy._safe_operations.add("map_ui_scene")
            policy._safe_operations.add("inspect_control_state")
            policy._safe_operations.add("query_scene_state")
            policy._safe_operations.add("verify_goal")
            policy._safe_operations.add("verify_screen_state")
            policy._safe_operations.add("track_elements")
            policy._safe_operations.add("get_visual_tracks")
            policy._safe_operations.add("get_recent_events")
            policy._safe_operations.add("query_event_history")
            policy._safe_operations.add("get_visual_situation")
        return policy

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
            "verify_goal",
            "locate_element",
            "detect_screen_change",
            "map_ui_scene",
            "inspect_control_state",
            "query_scene_state",
            "track_elements",
            "get_visual_tracks",
            "get_recent_events",
            "query_event_history",
            "get_visual_situation",
            "ground_visual_action",
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
            if _RE_DETECT_CHANGE.match(clean):
                return True
            if _RE_VERIFY_CHANGE.match(clean):
                return True
            if _RE_MAP_SCENE.match(clean):
                return True
            if _RE_INSPECT_CONTROL_STATE.match(clean):
                return True
            if _RE_QUERY_SCENE_STATE.match(clean):
                return True
            if _RE_RELATIONAL_LOCATE.match(clean):
                return True
            if _RE_LOCATE_ELEMENT.match(clean):
                return True
            if _RE_TRACK_ELEMENT.match(clean):
                return True
            if _RE_GET_TRACKS.match(clean):
                return True
            if _RE_GET_RECENT_EVENTS.match(clean):
                return True
            if _RE_QUERY_EVENT_HISTORY.match(clean):
                return True
            if _RE_GET_VISUAL_SITUATION.match(clean):
                return True
            if _RE_GROUND_ACTION.match(clean):
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

        if _RE_DETECT_CHANGE.match(clean):
            return "detect_screen_change", "active_window", {}, None

        if _RE_MAP_SCENE.match(clean):
            return "map_ui_scene", "active_window", {}, None

        m_ctrl = _RE_INSPECT_CONTROL_STATE.match(clean)
        if m_ctrl:
            tgt = m_ctrl.group(1) or m_ctrl.group(3) or m_ctrl.group(5) or m_ctrl.group(6) or m_ctrl.group(7) or ""
            prop = m_ctrl.group(2) or m_ctrl.group(4) or ""
            return "inspect_control_state", "active_window", {"target": tgt.strip(), "expected_state": prop.strip()}, None

        m_qscene = _RE_QUERY_SCENE_STATE.match(clean)
        if m_qscene:
            q_txt = m_qscene.group(1) if m_qscene.lastindex else text
            return "query_scene_state", "active_window", {"query": (q_txt or text).strip()}, None

        m_vchg = _RE_VERIFY_CHANGE.match(clean)
        if m_vchg:
            raw_tgt = (m_vchg.group(1) or "").strip()
            raw_act = (m_vchg.group(2) or "").strip()
            params: Dict[str, Any] = {}
            if raw_tgt:
                params["target"] = raw_tgt
            if raw_act:
                params["expected_change"] = raw_act
            return "detect_screen_change", "active_window", params, None

        if _RE_GET_TRACKS.match(clean):
            return "get_visual_tracks", "active_window", {}, None

        m_track = _RE_TRACK_ELEMENT.match(clean)
        if m_track:
            raw_tgt = (m_track.group(1) or m_track.group(2) or m_track.group(3) or m_track.group(4) or m_track.group(5) or "").strip()
            if raw_tgt.endswith("?"):
                raw_tgt = raw_tgt[:-1].strip()
            return "track_elements", "active_window", {"target": raw_tgt} if raw_tgt else {}, None

        if _RE_GET_RECENT_EVENTS.match(clean):
            return "get_recent_events", "active_window", {}, None

        if _RE_GET_VISUAL_SITUATION.match(clean):
            return "get_visual_situation", "active_window", {}, None

        if _RE_GROUND_ACTION.match(clean):
            return "ground_visual_action", "active_window", {"intent": text}, None

        m_qevt = _RE_QUERY_EVENT_HISTORY.match(clean)
        if m_qevt:
            tgt = (m_qevt.group(1) or m_qevt.group(2) or m_qevt.group(3) or "").strip()
            if tgt.endswith("?"):
                tgt = tgt[:-1].strip()
            params: Dict[str, Any] = {}
            if "change state" in clean:
                params["event_type"] = "STATE_CHANGED"
            elif "disappeared" in clean:
                params["event_type"] = "DISAPPEARED"
            if tgt and tgt.lower() not in ("any", "elements", "controls", "buttons"):
                params["target"] = tgt
            return "query_event_history", "active_window", params, None

        m_rel = _RE_RELATIONAL_LOCATE.match(clean)
        if m_rel:
            raw_tgt = (m_rel.group(1) or m_rel.group(2) or "").strip()
            rel_phrase = (m_rel.group(3) or "").strip()
            raw_ref = (m_rel.group(4) or "").strip()
            if raw_ref.lower().startswith("the "):
                raw_ref = raw_ref[4:].strip()
            elif raw_ref.lower().startswith("a "):
                raw_ref = raw_ref[2:].strip()
            elif raw_ref.lower().startswith("an "):
                raw_ref = raw_ref[3:].strip()
            if raw_ref.endswith("?"):
                raw_ref = raw_ref[:-1].strip()

            rel_norm = _normalize_relation_phrase(rel_phrase)
            return "locate_element", "active_window", {
                "target": raw_tgt,
                "relation": rel_norm,
                "reference_target": raw_ref,
            }, None

        m_loc = _RE_LOCATE_ELEMENT.match(clean)
        if m_loc:
            tgt = m_loc.group(1).strip()
            if tgt.endswith("?"):
                tgt = tgt[:-1].strip()
            return "locate_element", "active_window", {"target": tgt}, None

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

        if op in ("verify_screen_state", "verify_goal"):
            return self._handle_verify_screen_state(target, parameters)

        if op == "locate_element":
            return self._handle_locate_element(target, parameters)

        if op == "detect_screen_change":
            return self._handle_detect_screen_change(target, parameters)

        if op == "map_ui_scene":
            return self._handle_map_ui_scene(target, parameters)

        if op == "inspect_control_state":
            return self._handle_inspect_control_state(target, parameters)

        if op == "query_scene_state":
            return self._handle_query_scene_state(target, parameters)

        if op == "track_elements":
            return self._handle_track_elements(target, parameters)

        if op == "get_visual_tracks":
            return self._handle_get_visual_tracks(target, parameters)

        if op == "get_recent_events":
            return self._handle_get_recent_events(target, parameters)

        if op == "query_event_history":
            return self._handle_query_event_history(target, parameters)

        if op == "get_visual_situation":
            return self._handle_get_visual_situation(target, parameters)

        if op == "ground_visual_action":
            return self._handle_ground_visual_action(target, parameters)

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
        chosen_target = (params.get("capture_target") or target or default_target).strip().lower()

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
        goal_spec = parameters.get("goal_spec")
        condition = str(
            parameters.get("condition")
            or parameters.get("expected")
            or parameters.get("state")
            or (parameters.get("target") if not goal_spec else "")
            or target
            or ""
        ).strip()
        if not condition and not goal_spec:
            raise SkillExecutionError("No condition specified for visual state verification.")

        # Determine spec
        if goal_spec is not None:
            spec = goal_spec if isinstance(goal_spec, VisualGoalSpec) else VisualGoalSpec.from_dict(goal_spec)
        else:
            spec = parse_visual_goal(condition)

        ai_p = self.ai_provider
        # If plain string condition was passed without explicit goal_spec, enforce backward compatibility
        if goal_spec is None and ai_p is not None and not getattr(ai_p, "supports_multimodal", False):
            model_name = getattr(ai_p, "model", "unknown")
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

        prior_obs = parameters.get("prior_observation")
        v_res = self.verification_engine.verify_goal(
            spec, observation, prior_observation=prior_obs
        )

        verified: Optional[bool] = None
        if v_res.outcome == VisualOutcomeType.VERIFIED:
            verified = True
            status = "verified"
        elif v_res.outcome == VisualOutcomeType.NOT_VERIFIED:
            verified = False
            status = "not_verified"
        elif v_res.outcome == VisualOutcomeType.BLOCKED:
            verified = False
            status = "blocked"
        else:
            verified = None
            status = "uncertain"

        reason = v_res.explanation
        win_title = cap.metadata.get("window_title") or observation.metadata.get("window_title", "")
        proc_name = observation.metadata.get("process_name")

        return {
            "condition": condition or spec.target,
            "verified": verified,
            "status": status,
            "reason": reason,
            "outcome": v_res.outcome.value,
            "confidence": v_res.confidence,
            "evidence_chain": [e.to_dict() for e in v_res.evidence_chain],
            "evaluation_source": v_res.evaluation_source,
            "verification_result": v_res.to_dict(),
            "observation_id": observation.observation_id,
            "target": target or "active_window",
            "window_title": win_title,
            "process_name": proc_name,
            "reused_cache": False,
        }

    def verify_goal(
        self,
        goal: Union[VisualGoalSpec, Dict[str, Any], str],
        target: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> VisualVerificationResult:
        """Execute post-condition or goal verification through SecureVisionManager."""
        params = dict(parameters or {})
        params["goal_spec"] = goal
        res = self._handle_verify_screen_state(target, params)
        v_dict = res.get("verification_result")
        if v_dict:
            outcome_val = v_dict.get("outcome", VisualOutcomeType.UNCERTAIN.value)
            try:
                outcome = VisualOutcomeType(outcome_val)
            except Exception:
                outcome = VisualOutcomeType.UNCERTAIN
            spec_dict = v_dict.get("goal_spec") or {}
            spec_obj = VisualGoalSpec.from_dict(spec_dict)
            evidence_items = tuple(
                VisualEvidenceItem(
                    source=e.get("source", ""),
                    description=e.get("description", ""),
                    confidence=float(e.get("confidence", 1.0)),
                )
                for e in v_dict.get("evidence_chain", [])
            )
            return VisualVerificationResult(
                verification_id=v_dict.get("verification_id", ""),
                observation_id=v_dict.get("observation_id", ""),
                outcome=outcome,
                goal_spec=spec_obj,
                confidence=float(v_dict.get("confidence", 1.0)),
                evidence_chain=evidence_items,
                explanation=v_dict.get("explanation", ""),
                evaluation_source=v_dict.get("evaluation_source", "deterministic"),
                metadata=v_dict.get("metadata", {}),
            )
        outcome = (
            VisualOutcomeType.VERIFIED
            if res.get("verified")
            else (
                VisualOutcomeType.NOT_VERIFIED
                if res.get("verified") is False
                else VisualOutcomeType.UNCERTAIN
            )
        )
        spec_obj = (
            goal
            if isinstance(goal, VisualGoalSpec)
            else (
                VisualGoalSpec.from_dict(goal)
                if isinstance(goal, dict)
                else parse_visual_goal(str(goal))
            )
        )
        return VisualVerificationResult(
            verification_id=f"verif-{uuid.uuid4().hex[:8]}",
            observation_id=str(res.get("observation_id", "")),
            outcome=outcome,
            goal_spec=spec_obj,
            confidence=float(res.get("confidence", 1.0)),
            explanation=str(res.get("reason", "")),
            evaluation_source=str(res.get("evaluation_source", "deterministic")),
        )

    # -----------------------------------------------------------------------
    # 7. locate_element (Visual Grounding & UI Element Localization)
    # -----------------------------------------------------------------------

    def _handle_locate_element(
        self, target: Optional[str], parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Locate a UI element on the active window using hybrid OCR and multimodal visual grounding."""
        target_name = (
            parameters.get("target")
            or parameters.get("element")
            or parameters.get("query")
            or target
            or ""
        ).strip()
        if not target_name:
            raise SkillExecutionError("locate_element requires a 'target' parameter specifying the element to locate.")

        obs, reused = self._acquire_observation_with_reuse(
            target=target,
            default_target="active_window",
            parameters=parameters,
            allow_reuse=bool(parameters.get("reuse_cache", False)),
            query_text=target_name,
        )
        if obs is None or not obs.is_valid or obs.capture is None:
            raise SkillExecutionError("Could not acquire authorized screen observation for element localization.")

        engine = VisualGroundingEngine(
            ocr_provider=self.ocr_provider,
            ai_provider=self.ai_provider,
            prompt_builder=self._prompt_builder,
            logger_instance=self.logger,
            container_instance=self._container,
        )

        relation = parameters.get("relation")
        reference_target = parameters.get("reference_target") or parameters.get("reference")

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                if relation and reference_target:
                    res = loop.run_until_complete(
                        engine.locate_relative_element_async(
                            target=target_name,
                            relation=relation,
                            reference_target=reference_target,
                            observation=obs,
                            element_type_hint=parameters.get("element_type"),
                            force_multimodal=bool(parameters.get("force_multimodal", False)),
                        )
                    )
                else:
                    res = loop.run_until_complete(
                        engine.locate_element_async(
                            target=target_name,
                            observation=obs,
                            element_type_hint=parameters.get("element_type"),
                            force_multimodal=bool(parameters.get("force_multimodal", False)),
                        )
                    )
            finally:
                loop.close()
        except Exception as exc:
            self.logger.warning("Visual grounding execution failed: %s", exc)
            res = VisualGroundingResult(
                target=target_name,
                element=None,
                is_found=False,
                confidence=0.0,
                observation_id=obs.id if obs else "",
                summary=f"Could not locate '{target_name}': {exc}",
            )

        win_title = obs.capture.metadata.get("window_title") or obs.metadata.get("window_title", "")
        proc_name = obs.metadata.get("process_name")

        return {
            "target": target_name,
            "relation": relation,
            "reference_target": reference_target,
            "is_found": res.is_found,
            "confidence": res.confidence,
            "element": res.element.to_dict() if res.element else None,
            "summary": res.summary,
            "observation_id": obs.observation_id,
            "window_title": win_title,
            "process_name": proc_name,
            "reused_cache": reused,
        }

    # -----------------------------------------------------------------------
    # 8. detect_screen_change (Visual Delta & Change Detection)
    # -----------------------------------------------------------------------

    def _handle_detect_screen_change(
        self, target: Optional[str], parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare screen observations and detect visual changes between T0 (before) and T1 (after)."""
        mgr = self.secure_vision_manager
        target_name = (
            parameters.get("target")
            or parameters.get("element")
            or parameters.get("query")
            or target
            or ""
        ).strip()
        if target_name.lower() in ("active_window", "screen", "null", "none"):
            target_name = ""

        # Baseline Policy:
        # Case A: Check if a valid recent unexpired T0 exists in buffer manager.
        prev_obs = mgr.get_latest_observation()
        if prev_obs is not None and prev_obs.is_valid and not prev_obs.is_expired:
            obs_before = prev_obs
            # Capture fresh T1
            obs_after = self._acquire_observation(
                target=target,
                default_target="active_window",
                parameters=parameters,
            )
        else:
            # Case B: No valid T0 or T0 expired -> capture fresh baseline T0, then fresh T1
            obs_before = self._acquire_observation(
                target=target,
                default_target="active_window",
                parameters=parameters,
            )
            obs_after = self._acquire_observation(
                target=target,
                default_target="active_window",
                parameters=parameters,
            )

        if obs_before is None or obs_after is None or obs_before.capture is None or obs_after.capture is None:
            raise SkillExecutionError("Failed to acquire authorized screen observations for change detection.")

        engine = VisualDeltaEngine(
            ocr_provider=self.ocr_provider,
            ai_provider=self.ai_provider,
            preprocessor=self._preprocessor,
            prompt_builder=self._prompt_builder,
            logger_instance=self.logger,
            container_instance=self._container,
        )

        force_mm = bool(parameters.get("force_multimodal", False))
        res = engine.compare_observations(
            before=obs_before,
            after=obs_after,
            target_filter=target_name if target_name else None,
            force_multimodal=force_mm,
        )

        win_title = obs_after.capture.metadata.get("window_title") or obs_after.metadata.get("window_title", "")
        proc_name = obs_after.metadata.get("process_name")

        return {
            "target": target_name,
            "primary_change_type": res.primary_change_type.value if isinstance(res.primary_change_type, VisualDeltaType) else str(res.primary_change_type),
            "meaningful_change_detected": res.meaningful_change_detected,
            "window_changed": res.window_changed,
            "explanation": res.explanation,
            "confidence": res.confidence,
            "element_changes": [c.to_dict() for c in res.element_changes],
            "added_texts": list(res.added_texts),
            "removed_texts": list(res.removed_texts),
            "modified_texts": list(res.modified_texts),
            "before_observation_id": obs_before.observation_id,
            "after_observation_id": obs_after.observation_id,
            "time_delta_seconds": res.time_delta_seconds,
            "window_title": win_title,
            "process_name": proc_name,
            "metadata": dict(res.metadata),
        }

    # -----------------------------------------------------------------------
    # 9. map_ui_scene (Visual UI Scene Parsing & Interactive Element Mapping)
    # -----------------------------------------------------------------------

    def _handle_map_ui_scene(
        self, target: Optional[str], parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse active window into structured UIScene containing containers, widgets, and form fields."""
        container_filter = (
            parameters.get("container_filter")
            or parameters.get("filter")
            or parameters.get("type")
        )
        if container_filter and str(container_filter).lower() in ("active_window", "screen", "none", "null"):
            container_filter = None

        obs, reused = self._acquire_observation_with_reuse(
            target=target,
            default_target="active_window",
            parameters=parameters,
            allow_reuse=bool(parameters.get("reuse_cache", False)),
            query_text=str(container_filter or ""),
        )

        if obs is None or not obs.is_valid or obs.capture is None:
            raise SkillExecutionError("Could not acquire authorized screen observation for scene parsing.")

        parser = VisualSceneParser(
            ocr_provider=self.ocr_provider,
            ai_provider=self.ai_provider,
            preprocessor=self._preprocessor,
            prompt_builder=self._prompt_builder,
            logger_instance=self.logger,
            container_instance=self._container,
        )

        force_mm = bool(parameters.get("force_multimodal", False))
        scene: UIScene = parser.parse_scene(
            observation=obs,
            container_filter=container_filter,
            force_multimodal=force_mm,
        )

        win_title = obs.capture.metadata.get("window_title") or obs.metadata.get("window_title", "")
        proc_name = obs.metadata.get("process_name")

        return {
            "scene_id": scene.scene_id,
            "observation_id": obs.observation_id,
            "window_title": win_title,
            "process_name": proc_name,
            "window_bounds": scene.window_bounds.to_dict() if scene.window_bounds else None,
            "containers": [c.to_dict() for c in scene.containers],
            "interactive_elements": [e.to_dict() for e in scene.interactive_elements],
            "form_fields": [f.to_dict() for f in scene.form_fields],
            "summary": scene.summary,
            "confidence": scene.confidence,
            "reused_cache": reused,
            "metadata": dict(scene.metadata),
        }

    # -----------------------------------------------------------------------
    # 10. inspect_control_state & 11. query_scene_state (Phase 27.12)
    # -----------------------------------------------------------------------

    def _handle_inspect_control_state(
        self, target: Optional[str], parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Inspect the operational state and affordances of a target UI control."""
        target_name = (
            parameters.get("target")
            or parameters.get("element")
            or parameters.get("name")
            or target
            or ""
        ).strip()
        expected_state = (
            parameters.get("expected_state")
            or parameters.get("state")
            or parameters.get("property")
            or ""
        ).strip()

        obs, reused = self._acquire_observation_with_reuse(
            target=target,
            default_target="active_window",
            parameters=parameters,
            allow_reuse=bool(parameters.get("reuse_cache", False)),
            query_text=target_name,
        )

        if obs is None or not obs.is_valid or obs.capture is None:
            raise SkillExecutionError("Could not acquire authorized screen observation for control state inspection.")

        parser = VisualSceneParser(
            ocr_provider=self.ocr_provider,
            ai_provider=self.ai_provider,
            preprocessor=self._preprocessor,
            prompt_builder=self._prompt_builder,
            logger_instance=self.logger,
            container_instance=self._container,
        )
        scene = parser.parse_scene(observation=obs)

        engine = VisualAffordanceEngine(
            ai_provider=self.ai_provider,
            preprocessor=self._preprocessor,
            prompt_builder=self._prompt_builder,
            logger_instance=self.logger,
            container_instance=self._container,
        )

        # Resolve element
        matched_el: Optional[UIElement] = None
        if target_name:
            t_low = target_name.lower()
            for el in scene.interactive_elements:
                if el.name and (t_low == el.name.lower() or t_low in el.name.lower() or el.name.lower() in t_low):
                    matched_el = el
                    break
            if matched_el is None and scene.form_fields:
                for f in scene.form_fields:
                    if t_low in f.label.lower() or f.label.lower() in t_low:
                        matched_el = f.input_element
                        break

        if matched_el is not None:
            aff = engine.inspect_element_affordance(matched_el, capture=obs.capture)
            verified: Optional[bool] = None
            if expected_state:
                exp_low = expected_state.lower()
                if exp_low == "enabled":
                    verified = (aff.detected_state == ControlVisualState.ENABLED)
                elif exp_low == "disabled":
                    verified = (aff.detected_state == ControlVisualState.DISABLED)
                elif exp_low == "checked":
                    verified = (aff.detected_state == ControlVisualState.CHECKED)
                elif exp_low == "unchecked":
                    verified = (aff.detected_state == ControlVisualState.UNCHECKED)
                elif exp_low == "empty":
                    verified = (aff.detected_state == ControlVisualState.EMPTY)
                elif exp_low == "populated":
                    verified = (aff.detected_state == ControlVisualState.POPULATED)
                elif exp_low == "focused":
                    verified = (aff.detected_state == ControlVisualState.FOCUSED)
                elif exp_low == "editable":
                    verified = (aff.primary_affordance == ControlAffordance.EDITABLE)
                elif exp_low == "clickable":
                    verified = (aff.primary_affordance == ControlAffordance.CLICKABLE)

            summary = f"The {matched_el.name} appears {aff.detected_state.value}."
            if aff.detected_state == ControlVisualState.UNCERTAIN:
                summary = "I can't determine the control state confidently from the current screen."

            return {
                "target": target_name or matched_el.name,
                "detected_state": aff.detected_state.value,
                "primary_affordance": aff.primary_affordance.value,
                "confidence": aff.confidence,
                "evidence": aff.evidence,
                "verified": verified,
                "summary": summary,
                "element": matched_el.to_dict(),
                "observation_id": obs.observation_id,
                "reused_cache": reused,
            }

        # Fallback to query_scene_state
        q_text = f"is {target_name} {expected_state}" if expected_state else f"what is the state of {target_name}"
        ans = engine.query_scene_state(scene=scene, query=q_text, capture=obs.capture)
        return {
            "target": target_name,
            "detected_state": ans.detected_state.value if ans.detected_state else "uncertain",
            "primary_affordance": "read_only",
            "confidence": ans.confidence,
            "verified": ans.verified_condition,
            "summary": ans.summary,
            "element": ans.element.to_dict() if ans.element else None,
            "observation_id": obs.observation_id,
            "reused_cache": reused,
        }

    def _handle_query_scene_state(
        self, target: Optional[str], parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Query state and affordances of controls across the active window UIScene."""
        query_text = (
            parameters.get("query")
            or parameters.get("question")
            or target
            or ""
        ).strip()
        if not query_text:
            raise SkillExecutionError("query_scene_state requires a 'query' parameter.")

        obs, reused = self._acquire_observation_with_reuse(
            target=target,
            default_target="active_window",
            parameters=parameters,
            allow_reuse=bool(parameters.get("reuse_cache", False)),
            query_text=query_text,
        )

        if obs is None or not obs.is_valid or obs.capture is None:
            raise SkillExecutionError("Could not acquire authorized screen observation for scene state querying.")

        parser = VisualSceneParser(
            ocr_provider=self.ocr_provider,
            ai_provider=self.ai_provider,
            preprocessor=self._preprocessor,
            prompt_builder=self._prompt_builder,
            logger_instance=self.logger,
            container_instance=self._container,
        )
        scene = parser.parse_scene(observation=obs)

        engine = VisualAffordanceEngine(
            ai_provider=self.ai_provider,
            preprocessor=self._preprocessor,
            prompt_builder=self._prompt_builder,
            logger_instance=self.logger,
            container_instance=self._container,
        )

        ans = engine.query_scene_state(
            scene=scene,
            query=query_text,
            capture=obs.capture,
            force_multimodal=bool(parameters.get("force_multimodal", False)),
        )

        return {
            "query": query_text,
            "target_element": ans.target_element,
            "detected_state": ans.detected_state.value if ans.detected_state else None,
            "verified": ans.verified_condition,
            "confidence": ans.confidence,
            "summary": ans.summary,
            "element": ans.element.to_dict() if ans.element else None,
            "observation_id": obs.observation_id,
            "reused_cache": reused,
            "metadata": dict(ans.metadata),
        }

    def _handle_track_elements(
        self, target: Optional[str], parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Track UI visual elements across observations using VisualTrackingEngine."""
        target_name = (parameters.get("target") or "").strip()
        if not target_name and target and target.strip().lower() not in ("active_window", "active", "window", "screen"):
            target_name = target.strip()
        allow_reuse = bool(parameters.get("reuse_cache", False))

        obs, reused = self._acquire_observation_with_reuse(
            target=target,
            default_target="active_window",
            parameters=parameters,
            allow_reuse=allow_reuse,
            query_text=target_name,
        )

        if obs is None or not obs.is_valid or obs.capture is None:
            res = self.tracking_engine.track_observation(
                observation=obs,
                target_filter=target_name or None,
            )
            try:
                self.temporal_engine.derive_events_from_tracking(
                    tracking_result=res,
                    observation=obs,
                )
            except Exception as exc:
                self.logger.debug("Temporal event derivation failed: %s", exc)

            return {
                "tracking_id": res.tracking_id,
                "observation_id": obs.observation_id if obs else "",
                "summary": res.summary,
                "active_tracks": [t.to_dict() for t in res.active_tracks],
                "new_tracks": [t.to_dict() for t in res.new_tracks],
                "moved_tracks": [t.to_dict() for t in res.moved_tracks],
                "updated_tracks": [t.to_dict() for t in res.updated_tracks],
                "missing_tracks": [t.to_dict() for t in res.missing_tracks],
                "reappeared_tracks": [t.to_dict() for t in res.reappeared_tracks],
                "uncertain_tracks": [t.to_dict() for t in res.uncertain_tracks],
                "confidence": res.confidence,
                "target": target_name or None,
                "reused_cache": reused,
                "metadata": dict(res.metadata),
            }

        # Parse scene if possible to provide container context
        scene: Optional[UIScene] = None
        try:
            parser = VisualSceneParser(
                ocr_provider=self.ocr_provider,
                ai_provider=self.ai_provider,
                preprocessor=self._preprocessor,
                prompt_builder=self._prompt_builder,
                logger_instance=self.logger,
                container_instance=self._container,
            )
            scene = parser.parse_scene(observation=obs)
        except Exception as exc:
            self.logger.debug("Scene parsing during tracking fallback: %s", exc)

        res = self.tracking_engine.track_observation(
            observation=obs,
            scene=scene,
            target_filter=target_name or None,
        )

        try:
            self.temporal_engine.derive_events_from_tracking(
                tracking_result=res,
                scene=scene,
                observation=obs,
            )
        except Exception as exc:
            self.logger.debug("Temporal event derivation failed: %s", exc)

        return {
            "tracking_id": res.tracking_id,
            "observation_id": obs.observation_id,
            "summary": res.summary,
            "active_tracks": [t.to_dict() for t in res.active_tracks],
            "new_tracks": [t.to_dict() for t in res.new_tracks],
            "moved_tracks": [t.to_dict() for t in res.moved_tracks],
            "updated_tracks": [t.to_dict() for t in res.updated_tracks],
            "missing_tracks": [t.to_dict() for t in res.missing_tracks],
            "reappeared_tracks": [t.to_dict() for t in res.reappeared_tracks],
            "uncertain_tracks": [t.to_dict() for t in res.uncertain_tracks],
            "confidence": res.confidence,
            "target": target_name or None,
            "reused_cache": reused,
            "metadata": dict(res.metadata),
        }

    def _handle_get_visual_tracks(
        self, target: Optional[str], parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Query currently active in-memory visual tracks without capturing screen."""
        target_name = (parameters.get("target") or "").strip()
        if not target_name and target and target.strip().lower() not in ("active_window", "active", "window", "screen"):
            target_name = target.strip()
        active = self.tracking_engine.get_active_tracks(target_filter=target_name or None)
        count = len(active)
        summary = f"Currently tracking {count} visual element{'s' if count != 1 else ''}."
        if target_name:
            summary = f"Currently tracking {count} element{'s' if count != 1 else ''} matching '{target_name}'."
        return {
            "active_tracks": [t.to_dict() for t in active],
            "count": count,
            "target": target_name or None,
            "summary": summary,
        }

    def _handle_get_recent_events(
        self, target: Optional[str], parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Query recent visual temporal events from bounded history without capturing screen."""
        limit = int(parameters.get("limit", 10))
        event_type = parameters.get("event_type")
        window_filter = parameters.get("window_filter") or (
            target if target and target.strip().lower() not in ("active_window", "active", "window", "screen") else None
        )
        res = self.temporal_engine.get_recent_events(
            limit=limit,
            event_type=event_type,
            window_filter=window_filter,
        )
        return res.to_dict()

    def _handle_query_event_history(
        self, target: Optional[str], parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Query visual temporal event history for a specific element, track_id, or event category."""
        target_name = (parameters.get("target") or "").strip()
        if not target_name and target and target.strip().lower() not in ("active_window", "active", "window", "screen"):
            target_name = target.strip()
        track_id = parameters.get("track_id")
        event_type = parameters.get("event_type")
        window_filter = parameters.get("window_filter")

        if event_type and str(event_type).upper() == "STATE_CHANGED":
            res = self.temporal_engine.get_state_changes(target=target_name or None)
        elif event_type and str(event_type).upper() == "DISAPPEARED":
            res = self.temporal_engine.get_disappeared_elements(window_filter=window_filter)
        elif target_name or track_id:
            res = self.temporal_engine.get_element_history(target=target_name, track_id=track_id)
        else:
            res = self.temporal_engine.get_recent_events(
                limit=int(parameters.get("limit", 10)),
                event_type=event_type,
                window_filter=window_filter,
            )
        return res.to_dict()

    # -----------------------------------------------------------------------
    # 15. get_visual_situation (Visual Context Fusion & Situation Understanding)
    # -----------------------------------------------------------------------

    def _handle_get_visual_situation(
        self, target: Optional[str], parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fuse structured visual outputs into a deterministic, privacy-safe VisualSituationResult."""
        obs, reused = self._acquire_observation_with_reuse(
            target=target,
            default_target="active_window",
            parameters=parameters,
            allow_reuse=bool(parameters.get("reuse_cache", False)),
            query_text=parameters.get("query"),
        )

        if obs is None:
            raise SkillExecutionError("Could not acquire authorized screen observation for visual situation understanding.")

        # If observation is sensitive, blocked, or unauthorized, let situation_engine fail closed
        is_blocked = (
            getattr(obs, "is_sensitive", False)
            or bool(obs.metadata.get("is_sensitive", False))
            or bool(obs.metadata.get("blocked", False))
            or (obs.authorization is not None and not obs.authorization.is_allowed)
            or not getattr(obs, "is_valid", True)
        )

        scene: Optional[UIScene] = None
        if not is_blocked and bool(parameters.get("parse_scene", True)) and obs.capture is not None and not obs.capture.is_empty:
            try:
                parser = VisualSceneParser(
                    ocr_provider=self.ocr_provider,
                    ai_provider=self.ai_provider,
                    preprocessor=self._preprocessor,
                    prompt_builder=self._prompt_builder,
                    logger_instance=self.logger,
                    container_instance=self._container,
                )
                scene = parser.parse_scene(observation=obs)
            except Exception as exc:
                self.logger.debug("Scene parsing in get_visual_situation fallback: %s", exc)

        # Retrieve recent temporal events if available and safe
        temporal_history: Optional[VisualTemporalHistoryResult] = None
        if not is_blocked:
            try:
                temporal_history = self.temporal_engine.get_recent_events(limit=5)
            except Exception as exc:
                self.logger.debug("Temporal events in get_visual_situation fallback: %s", exc)

        # Evaluate situation
        sit_res = self.situation_engine.evaluate_situation(
            observation=obs,
            scene=scene,
            affordances=parameters.get("affordances"),
            tracking_result=parameters.get("tracking_result"),
            temporal_history=temporal_history,
            reused_cache=reused,
        )

        data = sit_res.to_dict()
        data["situation_id"] = sit_res.situation.situation_id
        data["situation_type"] = (
            sit_res.situation.situation_type.value
            if isinstance(sit_res.situation.situation_type, VisualSituationType)
            else str(sit_res.situation.situation_type)
        )
        data["observation_id"] = sit_res.situation.observation_id
        data["window_title"] = sit_res.situation.window_title
        data["process_name"] = sit_res.situation.process_name
        data["confidence"] = sit_res.situation.confidence
        return data

    # -----------------------------------------------------------------------
    # 16. ground_visual_action (Visual Action Grounding & Precondition Validation)
    # -----------------------------------------------------------------------

    def _handle_ground_visual_action(
        self, target: Optional[str], parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ground and validate an intended user action against current visual state."""
        intent_text = (
            parameters.get("intent")
            or parameters.get("action")
            or parameters.get("target")
            or target
            or ""
        )
        obs, reused = self._acquire_observation_with_reuse(
            target=target,
            default_target="active_window",
            parameters=parameters,
            allow_reuse=bool(parameters.get("reuse_cache", False)),
            query_text=intent_text,
        )

        if obs is None:
            raise SkillExecutionError("Could not acquire authorized screen observation for visual action grounding.")

        # If observation is sensitive, blocked, or unauthorized, let action_grounding_engine fail closed
        is_blocked = (
            getattr(obs, "is_sensitive", False)
            or bool(obs.metadata.get("is_sensitive", False))
            or bool(obs.metadata.get("blocked", False))
            or (obs.authorization is not None and not obs.authorization.is_allowed)
            or not getattr(obs, "is_valid", True)
        )

        scene: Optional[UIScene] = parameters.get("scene")
        if scene is None and not is_blocked and bool(parameters.get("parse_scene", True)) and obs.capture is not None and not obs.capture.is_empty:
            try:
                parser = VisualSceneParser(
                    ocr_provider=self.ocr_provider,
                    ai_provider=self.ai_provider,
                    preprocessor=self._preprocessor,
                    prompt_builder=self._prompt_builder,
                    logger_instance=self.logger,
                    container_instance=self._container,
                )
                scene = parser.parse_scene(observation=obs)
            except Exception as exc:
                self.logger.debug("Scene parsing in ground_visual_action fallback: %s", exc)

        situation: Optional[VisualSituation] = parameters.get("situation")
        if situation is None and not is_blocked and scene is not None:
            try:
                sit_res = self.situation_engine.evaluate_situation(
                    observation=obs,
                    scene=scene,
                    reused_cache=reused,
                )
                situation = sit_res.situation
            except Exception as exc:
                self.logger.debug("Situation evaluation in ground_visual_action fallback: %s", exc)

        res = self.action_grounding_engine.ground_action(
            intent=str(intent_text),
            observation=obs,
            situation=situation,
            scene=scene,
            affordances=parameters.get("affordances"),
            reused_cache=reused,
        )

        data = res.to_dict()
        data["status"] = (
            res.status.value
            if isinstance(res.status, VisualActionFeasibilityStatus)
            else str(res.status)
        )
        data["summary"] = res.summary
        if res.target:
            data["target_id"] = res.target.target_id
            data["target_element_name"] = res.target.target_element_name
            data["feasibility"] = (
                res.target.feasibility.value
                if isinstance(res.target.feasibility, VisualActionFeasibilityStatus)
                else str(res.target.feasibility)
            )
            data["safety_tier"] = (
                res.target.safety_tier.value
                if isinstance(res.target.safety_tier, VisualActionSafetyTier)
                else str(res.target.safety_tier)
            )
            data["requires_confirmation"] = res.target.requires_confirmation
            data["confidence"] = res.target.confidence
            data["reason"] = res.target.reason
        return data


__all__ = ["VisionSkills"]
