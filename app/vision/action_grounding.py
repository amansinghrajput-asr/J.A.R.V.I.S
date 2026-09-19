"""Visual Action Grounding and Precondition Validation Engine for J.A.R.V.I.S. Phase 27.17.

Bridges user action intent with the visual intelligence perception stack:
- ScreenObservation context (security boundary, active window)
- Hierarchical UIScene (containers and interactive element nodes)
- VisualSituation (modal detection, primary actions, focused element)
- ElementAffordance (actionability, enablement, control state)
- Post-action verification (VisualGoalSpec pairing)

Responsibilities:
1. Identifies WHAT visual control corresponds to an intended action.
2. Validates WHETHER that action is currently feasible given UI preconditions.
3. Computes WHERE a safe interior target coordinate exists without touching borders.
4. Classifies WHETHER human confirmation is required (conservative safety tiering).
5. Generates expected VisualGoalSpec post-condition for future Phase 27.13 verification.
6. Strictly read-only: NEVER moves mouse, clicks, types, or executes OS actions.
7. Independently fails closed on sensitive, blocked, unauthorized, or invalid observations.
8. Enforces modal dialog focus isolation (background controls blocked by modal).
9. Ensures safe transition isolation so protected observations cannot leak prior context.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import uuid

from app.core.logger import get_logger
from app.vision.models import (
    CaptureAuthorization,
    CaptureDecision,
    ControlAffordance,
    ControlVisualState,
    ElementAffordance,
    Point,
    ScreenObservation,
    UIContainer,
    UIContainerType,
    UIElement,
    UIElementType,
    UIScene,
    VisualActionFeasibilityStatus,
    VisualActionGroundingResult,
    VisualActionSafetyTier,
    VisualActionTarget,
    VisualActionType,
    VisualGoalCriterion,
    VisualGoalSpec,
    VisualSituation,
    VisualSituationType,
    WindowBounds,
)

logger = logging.getLogger("VISUAL_ACTION_GROUNDING")

# Secret redaction pattern
_RE_SECRET_PATTERNS = re.compile(
    r"(?i)\b(password|passwd|pwd|token|api[_-]?key|secret|pin|bearer|authorization|"
    r"auth[_-]?token|credentials?|private[_-]?key|card[_-]?number|cvv|ssn)\b"
)

# Destructive action indicator patterns
_RE_DESTRUCTIVE_KEYWORDS = re.compile(
    r"(?i)\b(delete|discard|remove|destroy|erase|wipe|format|drop|purge|kill|terminate|"
    r"overwrite|cancel\s+unsaved|reset|factory\s+reset|clear\s+all|uninstall|"
    r"pay|payment|transfer|checkout|submit\s+order|buy|purchase)\b"
)

# Safe / Non-mutating action patterns
_RE_SAFE_KEYWORDS = re.compile(
    r"(?i)\b(focus|inspect|view|read|navigate|switch\s+tab|next\s+tab|prev\s+tab|"
    r"scroll|zoom|expand|collapse|look|highlight|hover|examine|check\s+status)\b"
)

# Dismiss modal indicator patterns
_RE_DISMISS_PATTERNS = re.compile(
    r"(?i)\b(dismiss|close|cancel|exit|leave|hide|abort|escape|x)\b"
)

# Toggle indicator patterns
_RE_TOGGLE_PATTERNS = re.compile(
    r"(?i)\b(toggle|check|uncheck|enable|disable|switch|turn\s+(?:on|off))\b"
)

# Type text indicator patterns
_RE_TYPE_PATTERNS = re.compile(
    r"(?i)\b(type|enter|input|write|fill|fill\s+in|insert|set\s+text)\b"
)


def _sanitize_text(text: Optional[str]) -> str:
    """Strip, normalize whitespace, and redact sensitive secret terms."""
    if not text:
        return ""
    cleaned = " ".join(str(text).split())
    if _RE_SECRET_PATTERNS.search(cleaned):
        return "[REDACTED]"
    return cleaned


class VisualActionGroundingEngine:
    """Deterministic, read-only engine for grounding action intents on visual controls.

    Evaluates user intent against structured scene, affordance, and situation data to
    resolve target controls, enforce modal isolation, check preconditions, compute safe
    interior interaction coordinates, and assign conservative safety tiers.
    """

    def __init__(self, logger_instance: Optional[logging.Logger] = None) -> None:
        """Initialize VisualActionGroundingEngine."""
        self._lock = threading.RLock()
        self._logger = logger_instance or logger
        self._last_observation_id: Optional[str] = None
        self._last_was_sensitive: bool = False

    def ground_action(
        self,
        intent: str,
        observation: ScreenObservation,
        *,
        situation: Optional[VisualSituation] = None,
        scene: Optional[UIScene] = None,
        affordances: Optional[Sequence[ElementAffordance]] = None,
        reused_cache: bool = False,
    ) -> VisualActionGroundingResult:
        """Ground and validate an intended user action against current visual state.

        Args:
            intent: Natural language or structured user action intent.
            observation: Authorized ScreenObservation representing current screen state.
            situation: Optional pre-computed VisualSituation context.
            scene: Optional pre-parsed UIScene hierarchy.
            affordances: Optional pre-computed ElementAffordance sequence.
            reused_cache: Whether observation cache was reused.

        Returns:
            VisualActionGroundingResult with resolved VisualActionTarget and feasibility status.
        """
        with self._lock:
            # 1. Independent fail-closed security guard
            is_sensitive_obs = (
                getattr(observation, "is_sensitive", False)
                or bool(observation.metadata.get("is_sensitive", False))
                or bool(observation.metadata.get("blocked", False))
                or (observation.authorization is not None and not observation.authorization.is_allowed)
                or not getattr(observation, "is_valid", True)
                or (observation.capture is None)
                or (getattr(observation.capture, "is_empty", False))
            )

            current_obs_id = getattr(observation, "id", "") or str(uuid.uuid4())

            if is_sensitive_obs:
                # Isolate context: reset any prior tracked state
                self._last_observation_id = current_obs_id
                self._last_was_sensitive = True

                self._logger.warning(
                    "VisualActionGroundingEngine rejected sensitive, blocked, or invalid observation."
                )
                return VisualActionGroundingResult(
                    target=None,
                    status=VisualActionFeasibilityStatus.SENSITIVE_PROTECTED,
                    summary="Action grounding blocked: active screen context contains protected sensitive information.",
                    reused_cache=reused_cache,
                    metadata={"blocked": True, "reason": "sensitive_or_blocked_observation"},
                )

            # Safe A -> Protected -> Safe B isolation: if previous was sensitive, clear any cached continuity
            if self._last_was_sensitive:
                self._last_was_sensitive = False

            self._last_observation_id = current_obs_id

            # 2. Parse action intent deterministically
            sanitized_intent = _sanitize_text(intent)
            if not sanitized_intent or sanitized_intent == "[REDACTED]":
                return VisualActionGroundingResult(
                    target=None,
                    status=VisualActionFeasibilityStatus.UNCERTAIN,
                    summary="Action intent is empty or contains redacted sensitive tokens.",
                    reused_cache=reused_cache,
                    metadata={"reason": "empty_or_redacted_intent"},
                )

            action_type, target_keyword, requested_value = self._parse_intent_components(sanitized_intent)

            # 3. Resolve candidates from UIScene, VisualSituation, and affordances
            candidates = self._gather_candidate_elements(scene=scene, situation=situation, affordances=affordances)

            if not candidates:
                return VisualActionGroundingResult(
                    target=None,
                    status=VisualActionFeasibilityStatus.TARGET_NOT_FOUND,
                    summary=f"Could not locate any interactive controls on the active screen.",
                    reused_cache=reused_cache,
                    metadata={"target_keyword": target_keyword, "candidates_count": 0},
                )

            # 4. Match target control against candidates
            matched_element, match_confidence, ambiguity_detected = self._match_candidate(
                candidates=candidates,
                action_type=action_type,
                target_keyword=target_keyword,
                situation=situation,
            )

            # Ambiguity handling: if multiple elements match equally well, fail closed
            if ambiguity_detected or (matched_element is None and target_keyword):
                if ambiguity_detected:
                    return VisualActionGroundingResult(
                        target=None,
                        status=VisualActionFeasibilityStatus.UNCERTAIN,
                        summary=f"Multiple controls match '{target_keyword}'. Action requires explicit confirmation to disambiguate.",
                        reused_cache=reused_cache,
                        metadata={
                            "reason": "ambiguous_target_candidates",
                            "requires_confirmation": True,
                        },
                    )
                return VisualActionGroundingResult(
                    target=None,
                    status=VisualActionFeasibilityStatus.TARGET_NOT_FOUND,
                    summary=f"Could not locate a control matching '{target_keyword}' on the screen.",
                    reused_cache=reused_cache,
                    metadata={"target_keyword": target_keyword},
                )

            if matched_element is None:
                return VisualActionGroundingResult(
                    target=None,
                    status=VisualActionFeasibilityStatus.TARGET_NOT_FOUND,
                    summary="No matching actionable control could be determined.",
                    reused_cache=reused_cache,
                    metadata={"intent": sanitized_intent},
                )

            # 5. Check Modal Dialog Isolation
            is_blocked_by_modal, modal_reason = self._check_modal_isolation(
                element=matched_element,
                situation=situation,
                scene=scene,
                action_type=action_type,
            )

            if is_blocked_by_modal:
                target_name = getattr(matched_element, "name", "") or "control"
                target_id = f"act_{uuid.uuid4().hex[:8]}"
                modal_hwnd: Optional[int] = None
                if observation is not None and isinstance(observation.metadata, dict):
                    raw_h = observation.metadata.get("hwnd")
                    if raw_h is not None:
                        try:
                            modal_hwnd = int(raw_h)
                        except (ValueError, TypeError):
                            pass
                target = VisualActionTarget(
                    target_id=target_id,
                    action_type=action_type,
                    target_element_name=target_name,
                    target_point=None,
                    bounds=matched_element.bounds,
                    safety_tier=VisualActionSafetyTier.SAFE,
                    feasibility=VisualActionFeasibilityStatus.BLOCKED_BY_MODAL,
                    requires_confirmation=True,
                    confidence=match_confidence,
                    reason=modal_reason,
                    expected_outcome=None,
                    metadata={"element_name": target_name},
                    window_handle=modal_hwnd,
                    grounded_at=time.monotonic(),
                    input_text=None,
                )
                return VisualActionGroundingResult(
                    target=target,
                    status=VisualActionFeasibilityStatus.BLOCKED_BY_MODAL,
                    summary=f"Action on '{target_name}' is blocked by an active modal dialog.",
                    reused_cache=reused_cache,
                    metadata={"blocked_by_modal": True, "reason": modal_reason},
                )

            # 6. Check Control Affordances, Enablement & Preconditions
            affordance = self._find_element_affordance(matched_element, affordances)
            feasibility, precondition_reason = self._check_preconditions(
                element=matched_element,
                affordance=affordance,
                action_type=action_type,
                scene=scene,
            )

            # 7. Classify Action Safety Tier (Conservative)
            safety_tier, requires_conf = self._classify_action_safety(
                action_type=action_type,
                element=matched_element,
                intent_text=sanitized_intent,
                feasibility=feasibility,
            )

            # 8. Compute Safe Interior Target Point
            target_point: Optional[Point] = None
            if feasibility == VisualActionFeasibilityStatus.FEASIBLE:
                target_point, point_valid, point_reason = self._compute_safe_interior_point(matched_element.bounds)
                if not point_valid:
                    feasibility = VisualActionFeasibilityStatus.UNCERTAIN
                    precondition_reason = point_reason
                    requires_conf = True

            # 9. Pair with Post-Condition VisualGoalSpec
            expected_outcome: Optional[VisualGoalSpec] = None
            if feasibility == VisualActionFeasibilityStatus.FEASIBLE or feasibility == VisualActionFeasibilityStatus.BLOCKED_CONTROL_DISABLED:
                expected_outcome = self._create_post_condition_goal(
                    action_type=action_type,
                    element=matched_element,
                    target_keyword=target_keyword,
                    situation=situation,
                )

            # 10. Assemble VisualActionTarget
            target_id = f"act_{uuid.uuid4().hex[:8]}"
            target_name = getattr(matched_element, "name", "") or "control"
            final_reason = precondition_reason or f"Target control '{target_name}' is ready and feasible."

            # Resolve window_handle if safely available
            target_hwnd: Optional[int] = None
            if observation is not None and isinstance(observation.metadata, dict):
                raw_h = observation.metadata.get("hwnd")
                if raw_h is None:
                    raw_h = getattr(observation, "window_handle", None)
                if raw_h is not None:
                    try:
                        target_hwnd = int(raw_h)
                    except (ValueError, TypeError):
                        target_hwnd = None
            if target_hwnd is None and scene is not None and isinstance(scene.metadata, dict):
                raw_h = scene.metadata.get("hwnd")
                if raw_h is not None:
                    try:
                        target_hwnd = int(raw_h)
                    except (ValueError, TypeError):
                        target_hwnd = None

            # Populate input_text only when the action actually requires it
            action_input_text: Optional[str] = None
            if action_type in (VisualActionType.TYPE_TEXT, VisualActionType.CLEAR_AND_TYPE):
                if requested_value and isinstance(requested_value, str) and requested_value.strip():
                    action_input_text = requested_value.strip()

            action_target = VisualActionTarget(
                target_id=target_id,
                action_type=action_type,
                target_element_name=target_name,
                target_point=target_point,
                bounds=matched_element.bounds,
                safety_tier=safety_tier,
                feasibility=feasibility,
                requires_confirmation=requires_conf,
                confidence=match_confidence,
                reason=final_reason,
                expected_outcome=expected_outcome,
                metadata={
                    "element_type": (
                        matched_element.element_type.value
                        if isinstance(matched_element.element_type, UIElementType)
                        else str(matched_element.element_type)
                    ),
                    "element_name": target_name,
                },
                window_handle=target_hwnd,
                grounded_at=time.monotonic(),
                input_text=action_input_text,
            )

            # Summary formulation
            summary = self._generate_voice_summary(action_target)

            return VisualActionGroundingResult(
                target=action_target,
                status=feasibility,
                summary=summary,
                reused_cache=reused_cache,
                metadata={
                    "safety_tier": safety_tier.value,
                    "requires_confirmation": requires_conf,
                    "target_name": target_name,
                },
            )

    # -----------------------------------------------------------------------
    # Internal Intent Parsing
    # -----------------------------------------------------------------------

    def _parse_intent_components(
        self, intent: str
    ) -> Tuple[VisualActionType, str, Optional[str]]:
        """Parse sanitized intent into action type, target keyword, and optional input text."""
        norm = intent.strip()
        lower = norm.lower()

        # Check dismiss modal
        if _RE_DISMISS_PATTERNS.search(lower) and any(
            k in lower for k in ("popup", "modal", "dialog", "alert", "window", "notice")
        ):
            return VisualActionType.DISMISS_MODAL, "close", None

        # Check toggle
        if _RE_TOGGLE_PATTERNS.search(lower):
            m = re.search(
                r"(?i)\b(?:toggle|check|uncheck|enable|disable)\s+(?:the\s+|a\s+|an\s+)?(.+?)(?:\s+checkbox|\s+switch|\s+toggle|\s+button)?$",
                lower,
            )
            tgt = m.group(1).strip() if m else "checkbox"
            return VisualActionType.TOGGLE, tgt, None

        # Check clear and type
        if "clear and type" in lower or "clear and enter" in lower or "clear and input" in lower or "clear and fill" in lower:
            m_clear = re.search(
                r"(?i)\b(?:clear\s+and\s+(?:type|enter|input|fill(?:\s+in)?))\s+['\"]?(.+?)['\"]?\s+(?:in|into|for)\s+(?:the\s+|a\s+|an\s+)?(.+?)$",
                lower,
            )
            if m_clear:
                val = m_clear.group(1).strip()
                field_name = m_clear.group(2).strip()
                return VisualActionType.CLEAR_AND_TYPE, field_name, val
            m_clear_simple = re.search(
                r"(?i)\b(?:clear\s+and\s+(?:type|enter|input|fill))\s+(?:the\s+|a\s+|an\s+)?(.+?)(?:\s+field|\s+input|\s+box)?$",
                lower,
            )
            tgt = m_clear_simple.group(1).strip() if m_clear_simple else "input"
            return VisualActionType.CLEAR_AND_TYPE, tgt, None

        # Check select option / choose option
        if "select option" in lower or "choose option" in lower or "pick option" in lower or ("select " in lower and any(w in lower for w in ("dropdown", "option", "combo", "item"))):
            m_sel = re.search(
                r"(?i)\b(?:select|choose|pick)\s+(?:option\s+)?['\"]?(.+?)['\"]?\s+(?:from|in|for)\s+(?:the\s+|a\s+|an\s+)?(.+?)$",
                lower,
            )
            if m_sel:
                val = m_sel.group(1).strip()
                field_name = m_sel.group(2).strip()
                return VisualActionType.SELECT_OPTION, field_name, val
            m_sel_simple = re.search(
                r"(?i)\b(?:select|choose|pick)\s+(?:the\s+|a\s+|an\s+)?(.+?)(?:\s+option|\s+item|\s+dropdown)?$",
                lower,
            )
            tgt = m_sel_simple.group(1).strip() if m_sel_simple else "option"
            return VisualActionType.SELECT_OPTION, tgt, None

        # Check type text
        if _RE_TYPE_PATTERNS.search(lower):
            m_type = re.search(
                r"(?i)\b(?:type|enter|input|write|fill\s+in)\s+['\"]?(.+?)['\"]?\s+(?:in|into|for)\s+(?:the\s+|a\s+|an\s+)?(.+?)$",
                lower,
            )
            if m_type:
                val = m_type.group(1).strip()
                field_name = m_type.group(2).strip()
                return VisualActionType.TYPE_TEXT, field_name, val
            m_simple = re.search(
                r"(?i)\b(?:type|enter|input|fill)\s+(?:the\s+|a\s+|an\s+)?(.+?)(?:\s+field|\s+input|\s+box)?$",
                lower,
            )
            tgt = m_simple.group(1).strip() if m_simple else "input"
            return VisualActionType.TYPE_TEXT, tgt, None

        # Check double click
        if "double click" in lower or "double-click" in lower:
            m_dbl = re.search(
                r"(?i)\b(?:double\s+click|double-click)\s+(?:on\s+)?(?:the\s+|a\s+|an\s+)?(.+?)(?:\s+button|\s+control|\s+icon)?$",
                lower,
            )
            tgt = m_dbl.group(1).strip() if m_dbl else ""
            return VisualActionType.DOUBLE_CLICK, tgt, None

        # Check click / button queries:
        # e.g., "where should i click to save", "which button should i click to save", "identify the control for saving"
        m_where = re.search(
            r"(?i)\b(?:where\s+should\s+i\s+click\s+to|which\s+button\s+should\s+i\s+click\s+to|"
            r"locate\s+the\s+control\s+for|identify\s+the\s+control\s+for)\s+(.+?)(?:\s+button)?$",
            lower,
        )
        if m_where:
            tgt = m_where.group(1).strip()
            # Normalize gerund: saving -> save, submitting -> submit
            if tgt.endswith("ing"):
                if tgt == "saving":
                    tgt = "save"
                elif tgt == "submitting":
                    tgt = "submit"
                elif tgt == "closing":
                    tgt = "close"
            return VisualActionType.CLICK, tgt, None

        # Check "can i submit this form", "is this button ready"
        if "can i submit" in lower or "submit this form" in lower or "submit the form" in lower:
            return VisualActionType.CLICK, "submit", None

        # Check general click
        m_click = re.search(
            r"(?i)\b(?:click|press|hit|tap)\s+(?:on\s+)?(?:the\s+|a\s+|an\s+)?(.+?)(?:\s+button|\s+control)?$",
            lower,
        )
        if m_click:
            tgt = m_click.group(1).strip()
            return VisualActionType.CLICK, tgt, None

        # Check navigate / switch to / view
        m_nav = re.search(
            r"(?i)\b(?:navigate(?:\s+to)?|switch\s+to|focus(?:\s+on)?|go\s+to|view|inspect)\s+(?:the\s+|a\s+|an\s+)?(.+?)(?:\s+tab|\s+button|\s+page|\s+screen)?$",
            lower,
        )
        if m_nav:
            tgt = m_nav.group(1).strip()
            return VisualActionType.CLICK, tgt, None

        # Fallback keyword extraction: e.g. "save this", "submit", "save document"
        words = [w for w in lower.split() if w not in ("the", "a", "an", "this", "that", "it", "my", "to", "for", "please")]
        tgt = words[0] if words else ""
        return VisualActionType.CLICK, tgt, None

    # -----------------------------------------------------------------------
    # Candidate Gathering & Matching
    # -----------------------------------------------------------------------

    def _gather_candidate_elements(
        self,
        scene: Optional[UIScene],
        situation: Optional[VisualSituation],
        affordances: Optional[Sequence[ElementAffordance]],
    ) -> List[UIElement]:
        """Gather deduplicated interactive UIElement candidates."""
        seen_names: Set[str] = set()
        candidates: List[UIElement] = []

        def add_elem(el: Optional[UIElement]) -> None:
            if el is None:
                return
            name = getattr(el, "name", "") or ""
            key = (name.lower(), el.bounds.to_tuple() if el.bounds else None)
            if key not in seen_names:
                seen_names.add(key)
                candidates.append(el)

        # 1. Gather from scene interactive elements
        if scene is not None:
            for el in scene.interactive_elements:
                add_elem(el)
            for cnt in scene.containers:
                for el in cnt.elements:
                    add_elem(el)

        # 2. Gather from situation focused element & primary actions
        if situation is not None:
            if situation.focused_element is not None:
                add_elem(situation.focused_element)
            for aff in situation.primary_actions:
                add_elem(aff.element)

        # 3. Gather from affordances
        if affordances:
            for aff in affordances:
                add_elem(aff.element)

        return candidates

    def _match_candidate(
        self,
        candidates: Sequence[UIElement],
        action_type: VisualActionType,
        target_keyword: str,
        situation: Optional[VisualSituation],
    ) -> Tuple[Optional[UIElement], float, bool]:
        """Match candidates against target keyword and action type.

        Returns:
            (matched_element, confidence, ambiguity_detected)
        """
        keyword_clean = target_keyword.strip().lower()
        if not keyword_clean:
            # If no keyword specified, check situation focused element or primary action
            if situation and situation.focused_element:
                return situation.focused_element, 0.85, False
            if candidates:
                # Ambiguous if multiple candidates exist without target keyword
                if len(candidates) > 1:
                    return None, 0.5, True
                return candidates[0], 0.8, False
            return None, 0.0, False

        # Match scoring
        scored: List[Tuple[float, UIElement]] = []

        # Canonical aliases
        aliases = {keyword_clean}
        if keyword_clean in ("close", "dismiss", "cancel", "x", "popup"):
            aliases.update(["close", "dismiss", "cancel", "x", "ok", "abort", "exit"])
        elif keyword_clean in ("save", "apply"):
            aliases.update(["save", "apply", "commit", "save changes", "update"])
        elif keyword_clean in ("submit", "confirm", "send", "done", "finish"):
            aliases.update(["submit", "confirm", "send", "done", "finish", "ok", "proceed", "continue"])
        elif keyword_clean in ("remember me", "remember", "agree", "accept"):
            aliases.update(["remember me", "remember", "agree", "accept", "terms"])

        for el in candidates:
            score = 0.0
            el_name = (getattr(el, "name", "") or "").lower()
            el_text = (getattr(el, "text_content", "") or "").lower()

            # Exact match on name or text
            if el_name in aliases or el_text in aliases:
                score = 1.0
            elif any(alias in el_name for alias in aliases) or any(alias in el_text for alias in aliases):
                score = 0.9
            elif any(alias in el_name.split() for alias in aliases) or any(alias in el_text.split() for alias in aliases):
                score = 0.85
            elif keyword_clean in el_name or keyword_clean in el_text:
                score = 0.8

            # Action type compatibility bonus
            if action_type in (VisualActionType.CLICK, VisualActionType.DOUBLE_CLICK, VisualActionType.DISMISS_MODAL) and el.element_type == UIElementType.BUTTON:
                score += 0.05
            elif action_type in (VisualActionType.TYPE_TEXT, VisualActionType.CLEAR_AND_TYPE) and el.element_type in (
                UIElementType.INPUT,
                UIElementType.TEXT,
            ):
                score += 0.05
            elif action_type == VisualActionType.TOGGLE and el.element_type == UIElementType.CHECKBOX:
                score += 0.05
            elif action_type == VisualActionType.SELECT_OPTION and el.element_type in (
                UIElementType.DROPDOWN,
                UIElementType.MENU,
            ):
                score += 0.05

            if score > 0.6:
                scored.append((score, el))

        if not scored:
            return None, 0.0, False

        # Sort descending by score
        scored.sort(key=lambda x: x[0], reverse=True)

        top_score, top_el = scored[0]

        # Check for ambiguity: multiple top candidates with identical score
        if len(scored) > 1:
            second_score, second_el = scored[1]
            if top_score - second_score < 0.05 and (top_el.name != second_el.name or top_el.bounds != second_el.bounds):
                return None, top_score, True

        return top_el, min(1.0, top_score), False

    # -----------------------------------------------------------------------
    # Modal Dialog Isolation
    # -----------------------------------------------------------------------

    def _check_modal_isolation(
        self,
        element: UIElement,
        situation: Optional[VisualSituation],
        scene: Optional[UIScene],
        action_type: VisualActionType,
    ) -> Tuple[bool, str]:
        """Check if target control is blocked by an active modal dialog.

        Returns:
            (is_blocked, reason)
        """
        modal_container: Optional[UIContainer] = None

        # Check situation active modal
        if situation and situation.active_modal:
            modal_container = situation.active_modal
        elif situation and situation.situation_type == VisualSituationType.MODAL_DIALOG:
            modal_container = situation.primary_container

        # Check scene containers for modal
        if modal_container is None and scene:
            for cnt in scene.containers:
                if cnt.container_type == UIContainerType.DIALOG or cnt.metadata.get("is_modal"):
                    modal_container = cnt
                    break

        if modal_container is None:
            return False, ""

        # If a modal exists, check if element is inside the modal
        # Check by element membership
        for el in modal_container.elements:
            if el.name == element.name and el.bounds == element.bounds:
                return False, ""

        # Check by geometric containment
        if element.bounds is not None and modal_container.bounds is not None:
            mb = modal_container.bounds
            eb = element.bounds
            if (
                eb.left >= mb.left
                and eb.right <= mb.right
                and eb.top >= mb.top
                and eb.bottom <= mb.bottom
            ):
                return False, ""

        # Element is outside the active modal
        return True, "Target control is located outside the active modal dialog and is blocked."

    # -----------------------------------------------------------------------
    # Precondition Validation & Control State
    # -----------------------------------------------------------------------

    def _find_element_affordance(
        self, element: UIElement, affordances: Optional[Sequence[ElementAffordance]]
    ) -> Optional[ElementAffordance]:
        """Find matching ElementAffordance for UIElement."""
        if not affordances:
            return None
        for aff in affordances:
            if aff.element.name == element.name and aff.element.bounds == element.bounds:
                return aff
        return None

    def _check_preconditions(
        self,
        element: UIElement,
        affordance: Optional[ElementAffordance],
        action_type: VisualActionType,
        scene: Optional[UIScene],
    ) -> Tuple[VisualActionFeasibilityStatus, str]:
        """Validate UI preconditions, disabled controls, and unfilled prerequisites."""
        # 1. Check disabled control
        is_disabled = False
        if affordance is not None:
            if affordance.detected_state == ControlVisualState.DISABLED:
                is_disabled = True
        elif element.metadata.get("disabled") is True or element.metadata.get("state") == "disabled":
            is_disabled = True

        if is_disabled:
            return (
                VisualActionFeasibilityStatus.BLOCKED_CONTROL_DISABLED,
                f"Target control '{element.name}' is currently disabled.",
            )

        # 2. Check unfilled prerequisites for submission actions
        el_name_lower = (getattr(element, "name", "") or "").lower()
        if action_type == VisualActionType.CLICK and any(
            k in el_name_lower for k in ("submit", "confirm", "save", "apply", "register", "login")
        ):
            # Check scene form fields or inputs for empty required controls
            if scene is not None:
                for cand in scene.interactive_elements:
                    if cand.element_type == UIElementType.INPUT:
                        is_required = cand.metadata.get("is_required", False) or cand.metadata.get("required", False)
                        text_val = (getattr(cand, "text_content", "") or "").strip()
                        state_val = cand.metadata.get("state")
                        if is_required and (not text_val or state_val == "empty" or state_val == ControlVisualState.EMPTY):
                            return (
                                VisualActionFeasibilityStatus.BLOCKED_UNFILLED_PREREQUISITES,
                                f"Required form field '{cand.name}' is empty or incomplete.",
                            )

        # 3. Check non-actionable control
        if affordance is not None and affordance.primary_affordance == ControlAffordance.READ_ONLY:
            return (
                VisualActionFeasibilityStatus.UNCERTAIN,
                f"Target control '{element.name}' does not have an actionable interactive affordance.",
            )

        return VisualActionFeasibilityStatus.FEASIBLE, ""

    # -----------------------------------------------------------------------
    # Action Safety Classification
    # -----------------------------------------------------------------------

    def _classify_action_safety(
        self,
        action_type: VisualActionType,
        element: UIElement,
        intent_text: str,
        feasibility: VisualActionFeasibilityStatus,
    ) -> Tuple[VisualActionSafetyTier, bool]:
        """Classify safety tier conservatively.

        CRITICAL REQUIREMENT:
        Keyword matching alone MUST NOT prove an action is safe.
        If the engine cannot confidently determine safety:
        safety_tier = DESTRUCTIVE, requires_confirmation = True.
        Never silently downgrade uncertain actions to SAFE.
        """
        el_name = (getattr(element, "name", "") or "").lower()
        combined_text = f"{intent_text} {el_name}"

        # 1. Check DESTRUCTIVE indicators
        if _RE_DESTRUCTIVE_KEYWORDS.search(combined_text):
            return VisualActionSafetyTier.DESTRUCTIVE, True

        # 2. Check MUTATING indicators
        if action_type in (
            VisualActionType.TYPE_TEXT,
            VisualActionType.CLEAR_AND_TYPE,
            VisualActionType.TOGGLE,
            VisualActionType.SELECT_OPTION,
        ):
            # Mutating actions are MUTATING; if feasibility is blocked or uncertain, require confirmation
            requires_conf = feasibility != VisualActionFeasibilityStatus.FEASIBLE
            return VisualActionSafetyTier.MUTATING, requires_conf

        # 3. Check SAFE indicators
        if action_type == VisualActionType.DISMISS_MODAL:
            return VisualActionSafetyTier.SAFE, False

        el_words = set(el_name.split())
        if action_type == VisualActionType.CLICK:
            if any(k in el_words for k in ("close", "cancel", "dismiss", "back", "next", "tab")) or any(k in el_name for k in ("close", "cancel", "dismiss")):
                return VisualActionSafetyTier.SAFE, False
            if any(k in el_words for k in ("save", "submit", "apply", "ok", "confirm", "send")) or any(k in el_name for k in ("save", "submit", "apply", "confirm")):
                # Submitting/saving modifies application state -> MUTATING
                return VisualActionSafetyTier.MUTATING, False

        if _RE_SAFE_KEYWORDS.search(combined_text):
            return VisualActionSafetyTier.SAFE, False

        # 4. If safety cannot be confidently determined: fail closed to DESTRUCTIVE + confirmation required
        return VisualActionSafetyTier.DESTRUCTIVE, True

    # -----------------------------------------------------------------------
    # Coordinate Safety
    # -----------------------------------------------------------------------

    def _compute_safe_interior_point(
        self, bounds: Optional[WindowBounds]
    ) -> Tuple[Optional[Point], bool, str]:
        """Compute safe interior point strictly inside control bounds with margin.

        Returns:
            (point, is_valid, reason)
        """
        if bounds is None:
            return None, False, "Control bounds are missing or null."

        width = bounds.right - bounds.left
        height = bounds.bottom - bounds.top

        if width < 2 or height < 2:
            return None, False, f"Control bounds dimensions ({width}x{height}) are invalid or too small."

        # Compute safe interior point (centered with minimum 1px inset)
        cx = bounds.left + (width // 2)
        cy = bounds.top + (height // 2)

        # Strict containment check
        if not (bounds.left < cx < bounds.right and bounds.top < cy < bounds.bottom):
            return None, False, "Computed interior coordinate is not strictly inside bounds."

        return Point(x=cx, y=cy), True, ""

    # -----------------------------------------------------------------------
    # Post-Condition Pairing
    # -----------------------------------------------------------------------

    def _create_post_condition_goal(
        self,
        action_type: VisualActionType,
        element: UIElement,
        target_keyword: str,
        situation: Optional[VisualSituation],
    ) -> VisualGoalSpec:
        """Pair grounded action with a VisualGoalSpec for post-action verification."""
        el_name = getattr(element, "name", "") or target_keyword

        if action_type == VisualActionType.DISMISS_MODAL:
            # Expected post-condition: modal dialog disappears
            modal_title = "dialog"
            if situation and situation.active_modal:
                modal_title = situation.active_modal.label or "dialog"
            return VisualGoalSpec(
                criterion=VisualGoalCriterion.WINDOW_ABSENT,
                target=modal_title,
                metadata={"action": "dismiss_modal", "target_element": el_name},
            )

        if action_type == VisualActionType.TOGGLE:
            # Expected post-condition: element toggles state
            return VisualGoalSpec(
                criterion=VisualGoalCriterion.ELEMENT_STATE,
                target=el_name,
                expected_state=ControlVisualState.CHECKED,
                metadata={"action": "toggle", "target_element": el_name},
            )

        if action_type in (VisualActionType.TYPE_TEXT, VisualActionType.CLEAR_AND_TYPE):
            return VisualGoalSpec(
                criterion=VisualGoalCriterion.ELEMENT_STATE,
                target=el_name,
                expected_state=ControlVisualState.POPULATED,
                metadata={"action": "type_text", "target_element": el_name},
            )

        # Default for click / other: custom semantic verification
        return VisualGoalSpec(
            criterion=VisualGoalCriterion.CUSTOM_SEMANTIC,
            target=el_name,
            metadata={"action": "click", "element_name": el_name},
        )

    # -----------------------------------------------------------------------
    # Voice-Safe Summary Generation
    # -----------------------------------------------------------------------

    def _generate_voice_summary(self, target: VisualActionTarget) -> str:
        """Produce a concise, voice-safe summary without raw coordinates or leaked secrets."""
        name = target.target_element_name or "control"

        if target.feasibility == VisualActionFeasibilityStatus.FEASIBLE:
            if target.requires_confirmation or target.safety_tier == VisualActionSafetyTier.DESTRUCTIVE:
                return f"Grounded action on '{name}'. Confirmation is required before execution because this action is destructive."
            return f"Control '{name}' is ready and feasible for {target.action_type.value.lower().replace('_', ' ')}."

        if target.feasibility == VisualActionFeasibilityStatus.BLOCKED_CONTROL_DISABLED:
            return f"Control '{name}' was identified, but it is currently disabled."

        if target.feasibility == VisualActionFeasibilityStatus.BLOCKED_BY_MODAL:
            return f"Control '{name}' is blocked by an active modal dialog."

        if target.feasibility == VisualActionFeasibilityStatus.BLOCKED_UNFILLED_PREREQUISITES:
            return f"Cannot perform action on '{name}' because required form fields are incomplete."

        if target.feasibility == VisualActionFeasibilityStatus.TARGET_NOT_FOUND:
            return f"Could not find a control matching '{name}' on the screen."

        return f"Action feasibility for '{name}' is uncertain: {target.reason}"


__all__ = ["VisualActionGroundingEngine"]
