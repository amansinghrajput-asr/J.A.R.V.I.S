"""Visual Context Fusion and Situation Understanding Engine for J.A.R.V.I.S. Phase 27.16.

Fuses already-computed structured visual intelligence:
- ScreenObservation context (active window, process, authorization)
- Hierarchical UIScene (spatial containers and interactive element nodes)
- Control affordances and states (actionability, focus, enablement)
- Cross-observation visual tracks (active element identities)
- Bounded visual temporal history (recent semantic transitions)

Responsibilities:
1. Pure aggregation / interpretation layer - never captures screen, never runs OCR,
   never tracks elements, never clusters scenes, never polls the OS.
2. Evaluates deterministic, evidence-based VisualSituationType classifications.
3. Produces concise, voice-safe situation summaries for conversational TTS and planning.
4. Independently fails closed on sensitive, blocked, unauthorized, or invalid observations.
5. Ensures sensitive window transition isolation so pre-sensitive context is not
   propagated across protected boundaries.
6. Enforces strict privacy boundaries (zero raw pixels, secret redaction).
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import uuid

from app.core.logger import get_logger
from app.vision.models import (
    CaptureAuthorization,
    CaptureDecision,
    ControlAffordance,
    ControlVisualState,
    ElementAffordance,
    GroundingSource,
    Point,
    ScreenObservation,
    UIContainer,
    UIContainerType,
    UIElement,
    UIElementType,
    UIScene,
    VisualElementTrack,
    VisualSituation,
    VisualSituationResult,
    VisualSituationType,
    VisualTemporalEvent,
    VisualTemporalHistoryResult,
    VisualTrackingResult,
)

logger = logging.getLogger("VISUAL_SITUATION")

# Secret redaction pattern
_RE_SECRET_PATTERNS = re.compile(
    r"(?i)\b(password|passwd|pwd|token|api[_-]?key|secret|pin|bearer|authorization|"
    r"auth[_-]?token|credentials?|private[_-]?key|card[_-]?number|cvv|ssn)\b"
)

# Progress indicator patterns
_RE_PROGRESS_KEYWORDS = re.compile(
    r"(?i)\b(progress|loading|please wait|downloading|uploading|installing|updating|"
    r"processing|synchronizing|buffering|fetching|connecting|waiting for)\b"
)

# Error / Alert indicator patterns
_RE_ERROR_KEYWORDS = re.compile(
    r"(?i)\b(error|exception|failed|failure|alert|warning|fatal|crash|invalid|denied|"
    r"aborted|unhandled|critical|access denied|permission denied|could not)\b"
)

# Desktop / Idle indicator patterns
_RE_DESKTOP_PROCESSES = {
    "explorer.exe",
    "dwm.exe",
    "shellexperiencehost.exe",
    "searchapp.exe",
    "startmenuexperiencehost.exe",
}
_RE_DESKTOP_TITLES = re.compile(r"(?i)^(program manager|desktop|taskbar|start)$")


def _sanitize_text(text: Optional[str]) -> str:
    """Strip, normalize whitespace, and redact sensitive secret terms."""
    if not text:
        return ""
    cleaned = " ".join(str(text).split())
    if _RE_SECRET_PATTERNS.search(cleaned):
        return "[REDACTED]"
    return cleaned


def _redact_element(element: UIElement) -> UIElement:
    """Return a privacy-safe sanitized clone of UIElement."""
    raw_name = getattr(element, "name", "") or getattr(element, "canonical_name", "") or ""
    raw_txt = getattr(element, "text_content", "") or getattr(element, "text", "") or ""
    name = _sanitize_text(raw_name)
    txt = _sanitize_text(raw_txt)

    # Redact metadata dict
    clean_meta: Dict[str, Any] = {}
    for k, v in element.metadata.items():
        if isinstance(v, str):
            clean_meta[k] = _sanitize_text(v)
        else:
            clean_meta[k] = v

    center = getattr(element, "center", None)
    if center is None and element.bounds is not None:
        center = element.bounds.center

    return UIElement(
        name=name,
        element_type=element.element_type,
        bounds=element.bounds,
        center=center or Point(0, 0),
        confidence=element.confidence,
        source=getattr(element, "source", GroundingSource.UNKNOWN),
        text_content=txt or None,
        metadata=clean_meta,
    )


def _redact_container(container: UIContainer) -> UIContainer:
    """Return a privacy-safe sanitized clone of UIContainer."""
    raw_label = getattr(container, "label", None) or getattr(container, "title", None) or ""
    label = _sanitize_text(raw_label)
    clean_elements = tuple(_redact_element(e) for e in container.elements)

    clean_meta: Dict[str, Any] = {}
    for k, v in container.metadata.items():
        if isinstance(v, str):
            clean_meta[k] = _sanitize_text(v)
        else:
            clean_meta[k] = v

    return UIContainer(
        container_id=container.container_id,
        container_type=container.container_type,
        bounds=container.bounds,
        elements=clean_elements,
        label=label or None,
        confidence=container.confidence,
        metadata=clean_meta,
    )


class VisualSituationEngine:
    """Deterministic-first Visual Context Fusion and Situation Understanding Engine.

    Aggregates already-computed visual perception outputs (scene, affordances, tracking,
    temporal history) into a unified, privacy-safe, voice-ready visual situation model.
    """

    def __init__(self, logger_instance: Optional[logging.Logger] = None) -> None:
        """Initialize the situation engine."""
        self._logger = logger_instance or logger
        self._lock = threading.RLock()

        # Ephemeral session context for transition tracking
        self._last_window_title: Optional[str] = None
        self._last_process_name: Optional[str] = None
        self._last_situation_type: Optional[VisualSituationType] = None
        self._last_situation_id: Optional[str] = None

    def reset(self) -> None:
        """Reset internal situation state and transition context."""
        with self._lock:
            self._last_window_title = None
            self._last_process_name = None
            self._last_situation_type = None
            self._last_situation_id = None

    def fuse_situation(
        self,
        observation: Optional[ScreenObservation] = None,
        scene: Optional[UIScene] = None,
        affordances: Optional[Sequence[ControlAffordance]] = None,
        tracking_result: Optional[VisualTrackingResult] = None,
        temporal_history: Optional[VisualTemporalHistoryResult] = None,
        current_time: Optional[float] = None,
    ) -> VisualSituation:
        """Fuse structured visual outputs into a consolidated VisualSituation snapshot.

        Args:
            observation: Active ScreenObservation context.
            scene: Optional parsed hierarchical UIScene.
            affordances: Optional sequence of evaluated ControlAffordance items.
            tracking_result: Optional VisualTrackingResult from VisualTrackingEngine.
            temporal_history: Optional VisualTemporalHistoryResult from VisualTemporalEngine.
            current_time: Optional deterministic timestamp injection (defaults to time.time()).

        Returns:
            Consolidated VisualSituation snapshot.
        """
        now = current_time if current_time is not None else time.time()
        sit_id = f"sit_{uuid.uuid4().hex[:8]}"

        # ------------------------------------------------------------------
        # 1. Independent Security Guard: Fail Closed on Protected Observations
        # ------------------------------------------------------------------
        if observation is not None:
            is_blocked = (
                getattr(observation, "is_sensitive", False)
                or bool(observation.metadata.get("is_sensitive", False))
                or bool(observation.metadata.get("blocked", False))
                or (observation.authorization is not None and not observation.authorization.is_allowed)
                or not getattr(observation, "is_valid", True)
            )
            if is_blocked:
                self._logger.warning("VisualSituationEngine: sensitive or blocked observation intercepted. Failing closed.")
                with self._lock:
                    # Protected transition isolation: invalidate prior window and situation context
                    # so safe context cannot infer transitions across this protected boundary
                    self._last_window_title = None
                    self._last_process_name = None
                    self._last_situation_type = None
                    self._last_situation_id = None

                return VisualSituation(
                    situation_id=sit_id,
                    timestamp=now,
                    observation_id=getattr(observation, "observation_id", getattr(observation, "id", "")),
                    situation_type=VisualSituationType.SENSITIVE_PROTECTED,
                    window_title=None,
                    process_name=None,
                    primary_container=None,
                    active_modal=None,
                    focused_element=None,
                    primary_actions=(),
                    recent_events_summary="",
                    summary="Current screen or active window is protected by security policy.",
                    confidence=1.0,
                    metadata={"blocked": True, "policy": "fail_closed"},
                )

        obs_id = ""
        raw_win_title = ""
        raw_proc_name = ""
        if observation is not None:
            obs_id = getattr(observation, "observation_id", getattr(observation, "id", ""))
            raw_win_title = str(observation.metadata.get("window_title") or "")
            raw_proc_name = str(observation.metadata.get("process_name") or "")

        win_title = _sanitize_text(raw_win_title)
        proc_name = _sanitize_text(raw_proc_name)

        # ------------------------------------------------------------------
        # 2. Extract Structural Evidence
        # ------------------------------------------------------------------
        containers: List[UIContainer] = list(scene.containers) if scene and scene.containers else []
        elements: List[UIElement] = []
        if scene:
            if hasattr(scene, "interactive_elements") and scene.interactive_elements:
                elements.extend(scene.interactive_elements)
            elif hasattr(scene, "elements") and getattr(scene, "elements", None):
                elements.extend(getattr(scene, "elements"))
            for c in containers:
                for el in c.elements:
                    if el not in elements:
                        elements.append(el)
        affordance_list: List[ControlAffordance] = list(affordances) if affordances else []

        # Find active modal / dialog
        active_modal: Optional[UIContainer] = None
        for c in containers:
            c_label = getattr(c, "label", None) or getattr(c, "title", None) or str(c.metadata.get("title") or "")
            is_modal = c.container_type == UIContainerType.DIALOG or bool(c.metadata.get("is_modal")) or getattr(c, "is_modal", False)
            if is_modal:
                active_modal = _redact_container(c)
                break
            if "dialog" in c_label.lower() or "modal" in c_label.lower():
                active_modal = _redact_container(c)
                break

        # Find primary container (active modal takes precedence, then form, then main container)
        primary_container: Optional[UIContainer] = None
        if active_modal is not None:
            primary_container = active_modal
        else:
            for c in containers:
                if c.container_type == UIContainerType.FORM:
                    primary_container = _redact_container(c)
                    break
            if primary_container is None and containers:
                # Pick largest or first meaningful container
                primary_container = _redact_container(containers[0])

        # Find focused element
        focused_element: Optional[UIElement] = None
        for aff in affordance_list:
            aff_state = getattr(aff, "detected_state", None) or getattr(aff, "state", None)
            if aff_state == ControlVisualState.FOCUSED and aff.element is not None:
                focused_element = _redact_element(aff.element)
                break
        if focused_element is None:
            for el in elements:
                if el.metadata.get("is_focused") or el.metadata.get("focused"):
                    focused_element = _redact_element(el)
                    break

        # Extract primary actions (actionable affordances, sanitized)
        primary_actions: List[ElementAffordance] = []
        for aff in affordance_list:
            aff_state = getattr(aff, "detected_state", None) or getattr(aff, "state", None) or ControlVisualState.ENABLED
            if aff_state not in (ControlVisualState.DISABLED, ControlVisualState.UNCERTAIN):
                # Redact target element
                clean_el = _redact_element(aff.element) if aff.element else None
                aff_type = getattr(aff, "primary_affordance", None) or getattr(aff, "affordance_type", None) or ControlAffordance.CLICKABLE
                evidence_txt = _sanitize_text(getattr(aff, "evidence", "") or getattr(aff, "action_hint", ""))
                clean_aff = ElementAffordance(
                    element=clean_el,
                    primary_affordance=aff_type,
                    detected_state=aff_state,
                    confidence=aff.confidence,
                    evidence=evidence_txt,
                    metadata=dict(aff.metadata),
                )
                primary_actions.append(clean_aff)
                if len(primary_actions) >= 5:  # Clamp to top 5 primary actions
                    break

        # Extract recent events summary
        recent_events_summary = ""
        if temporal_history is not None:
            recent_events_summary = _sanitize_text(temporal_history.summary)
        elif tracking_result is not None and tracking_result.summary:
            recent_events_summary = _sanitize_text(tracking_result.summary)

        # ------------------------------------------------------------------
        # 3. Deterministic Situation Classification
        # ------------------------------------------------------------------
        situation_type, confidence = self._classify_situation(
            win_title=win_title,
            proc_name=proc_name,
            containers=containers,
            elements=elements,
            active_modal=active_modal,
            affordances=affordance_list,
            observation=observation,
        )

        # ------------------------------------------------------------------
        # 4. Construct Voice-Safe Natural Language Summary
        # ------------------------------------------------------------------
        summary = self._generate_situation_summary(
            situation_type=situation_type,
            win_title=win_title,
            proc_name=proc_name,
            active_modal=active_modal,
            primary_container=primary_container,
            focused_element=focused_element,
            primary_actions=primary_actions,
            recent_events_summary=recent_events_summary,
        )

        with self._lock:
            self._last_window_title = win_title or None
            self._last_process_name = proc_name or None
            self._last_situation_type = situation_type
            self._last_situation_id = sit_id

        return VisualSituation(
            situation_id=sit_id,
            timestamp=now,
            observation_id=obs_id,
            situation_type=situation_type,
            window_title=win_title or None,
            process_name=proc_name or None,
            primary_container=primary_container,
            active_modal=active_modal,
            focused_element=focused_element,
            primary_actions=tuple(primary_actions),
            recent_events_summary=recent_events_summary,
            summary=summary,
            confidence=confidence,
            metadata={
                "container_count": len(containers),
                "element_count": len(elements),
                "action_count": len(primary_actions),
            },
        )

    def evaluate_situation(
        self,
        observation: Optional[ScreenObservation] = None,
        scene: Optional[UIScene] = None,
        affordances: Optional[Sequence[ControlAffordance]] = None,
        tracking_result: Optional[VisualTrackingResult] = None,
        temporal_history: Optional[VisualTemporalHistoryResult] = None,
        reused_cache: bool = False,
        current_time: Optional[float] = None,
    ) -> VisualSituationResult:
        """Evaluate visual situation and return structured VisualSituationResult."""
        situation = self.fuse_situation(
            observation=observation,
            scene=scene,
            affordances=affordances,
            tracking_result=tracking_result,
            temporal_history=temporal_history,
            current_time=current_time,
        )
        return VisualSituationResult(
            situation=situation,
            reused_cache=reused_cache,
            evaluation_source="deterministic_fusion",
            summary=situation.summary,
            metadata=dict(situation.metadata),
        )

    # -----------------------------------------------------------------------
    # Classification Rules Core
    # -----------------------------------------------------------------------

    def _classify_situation(
        self,
        win_title: str,
        proc_name: str,
        containers: List[UIContainer],
        elements: List[UIElement],
        active_modal: Optional[UIContainer],
        affordances: List[ControlAffordance],
        observation: Optional[ScreenObservation],
    ) -> Tuple[VisualSituationType, float]:
        """Classify visual situation using deterministic, conservative evidence ordering."""
        # Check empty / invalid context
        if not win_title and not proc_name and not containers and not elements:
            return VisualSituationType.UNKNOWN, 0.5

        # Check Desktop Idle
        is_desktop_proc = proc_name.lower() in _RE_DESKTOP_PROCESSES
        is_desktop_title = bool(_RE_DESKTOP_TITLES.match(win_title))
        if (is_desktop_proc and is_desktop_title) or (is_desktop_title and len(containers) <= 1 and len(elements) == 0):
            return VisualSituationType.DESKTOP_IDLE, 0.95

        # Check Error Alert (prominent error banner, crash dialog, or error text)
        error_found = False
        for c in containers:
            c_label = getattr(c, "label", None) or getattr(c, "title", None) or str(c.metadata.get("title") or "")
            if _RE_ERROR_KEYWORDS.search(c_label):
                error_found = True
                break
            for el in c.elements:
                el_txt = getattr(el, "text_content", None) or getattr(el, "text", None) or el.name
                if el.metadata.get("is_error") or _RE_ERROR_KEYWORDS.search(el_txt or ""):
                    error_found = True
                    break
            if error_found:
                break
        if not error_found:
            for el in elements:
                el_txt = getattr(el, "text_content", None) or getattr(el, "text", None) or el.name
                if el.metadata.get("is_error") or _RE_ERROR_KEYWORDS.search(el_txt or ""):
                    error_found = True
                    break

        if error_found:
            return VisualSituationType.ERROR_ALERT, 0.95

        # Check Progress / Busy indicator
        progress_found = False
        for el in elements:
            el_txt = getattr(el, "text_content", None) or getattr(el, "text", None) or el.name
            if el.metadata.get("is_progress") or _RE_PROGRESS_KEYWORDS.search(el_txt or ""):
                progress_found = True
                break
        if not progress_found:
            for c in containers:
                c_label = getattr(c, "label", None) or getattr(c, "title", None) or str(c.metadata.get("title") or "")
                if _RE_PROGRESS_KEYWORDS.search(c_label):
                    progress_found = True
                    break

        if progress_found:
            return VisualSituationType.PROGRESS_BUSY, 0.9

        # Check Modal Dialog
        if active_modal is not None:
            return VisualSituationType.MODAL_DIALOG, 0.95

        # Check Form Input
        form_found = False
        input_count = 0
        for c in containers:
            if c.container_type == UIContainerType.FORM:
                form_found = True
                break
            for el in c.elements:
                if el.element_type in (UIElementType.INPUT, UIElementType.CHECKBOX) or el.metadata.get("is_interactive", True):
                    input_count += 1
        if not form_found and input_count >= 2:
            form_found = True

        if form_found:
            return VisualSituationType.FORM_INPUT, 0.9

        # Normal Application View
        if win_title or proc_name:
            return VisualSituationType.APPLICATION_ACTIVE, 0.85

        return VisualSituationType.UNKNOWN, 0.5

    # -----------------------------------------------------------------------
    # Summary Generation
    # -----------------------------------------------------------------------

    def _generate_situation_summary(
        self,
        situation_type: VisualSituationType,
        win_title: str,
        proc_name: str,
        active_modal: Optional[UIContainer],
        primary_container: Optional[UIContainer],
        focused_element: Optional[UIElement],
        primary_actions: List[ControlAffordance],
        recent_events_summary: str,
    ) -> str:
        """Construct concise, voice-safe natural language summary."""
        app_label = win_title or proc_name or "active window"

        if situation_type == VisualSituationType.SENSITIVE_PROTECTED:
            return "Current screen is protected by security policy."

        if situation_type == VisualSituationType.DESKTOP_IDLE:
            return "The desktop is currently idle with no active application in foreground."

        if situation_type == VisualSituationType.ERROR_ALERT:
            base = f"An error or alert is displayed in {app_label}."
            modal_title = (active_modal.label or str(active_modal.metadata.get("title") or "")) if active_modal else ""
            if modal_title:
                base = f"An alert dialog '{modal_title}' is active in {app_label}."
            if primary_actions:
                action_names = [a.element.name for a in primary_actions if a.element and a.element.name]
                if action_names:
                    base += f" Available actions include {', '.join(action_names[:2])}."
            return base

        if situation_type == VisualSituationType.PROGRESS_BUSY:
            return f"{app_label} is currently busy processing or loading."

        if situation_type == VisualSituationType.MODAL_DIALOG:
            dlg_title = (active_modal.label or str(active_modal.metadata.get("title") or "")) if active_modal else "Dialog"
            base = f"A modal dialog '{dlg_title}' is currently open in {app_label}."
            if primary_actions:
                action_names = [a.element.name for a in primary_actions if a.element and a.element.name]
                if action_names:
                    base += f" Actions available: {', '.join(action_names[:3])}."
            return base

        if situation_type == VisualSituationType.FORM_INPUT:
            base = f"You are viewing a form in {app_label}."
            if focused_element and focused_element.name:
                base += f" Focused field: '{focused_element.name}'."
            elif primary_actions:
                first_act = primary_actions[0].element.name if primary_actions[0].element else ""
                if first_act:
                    base += f" Ready to submit or interact with '{first_act}'."
            return base

        if situation_type == VisualSituationType.APPLICATION_ACTIVE:
            base = f"{app_label} is active in foreground."
            if recent_events_summary:
                base += f" {recent_events_summary}"
            return base

        return "Active visual situation is currently unclassified."


__all__ = [
    "VisualSituationEngine",
]
