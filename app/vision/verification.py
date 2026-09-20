"""Visual Task Outcome & Goal-State Verification Engine for J.A.R.V.I.S. Phase 27.13.

Connects the visual perception stack to structured task outcome verification.
Evaluates whether expected visual post-conditions and goal states have been
achieved following task execution or user query.

Architectural and Security Invariants Enforced:
1. Deterministic-First: Relies on window/process metadata, local OCR, structural
   UIScene containers, and ControlVisualState affordances before any VLM fallback.
2. Optional Multimodal Fallback: Multimodal VLM is invoked ONLY when criterion is
   CUSTOM_SEMANTIC or deterministic evidence is genuinely ambiguous. If VLM is
   unavailable, returns conservative deterministic result or UNCERTAIN.
3. Evidence & Uncertainty: Produces an ordered evidence chain of sanitized
   VisualEvidenceItems. When evidence conflicts or is insufficient, returns
   UNCERTAIN rather than guessing.
4. Read-Only Execution: Performs ZERO mouse, keyboard, window manipulation, or OS actions.
5. Privacy and Leak Prevention: Never logs, serializes, or persists raw pixels,
   base64 strings, passwords, or sensitive input contents.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from app.core.logger import get_logger
from app.vision.affordance import VisualAffordanceEngine
from app.vision.delta import VisualDeltaEngine
from app.vision.models import (
    ControlAffordance,
    ControlVisualState,
    ElementAffordance,
    OCRResult,
    ScreenObservation,
    UIContainer,
    UIContainerType,
    UIElement,
    UIScene,
    VisualDeltaResult,
    VisualEvidenceItem,
    VisualGoalCriterion,
    VisualGoalSpec,
    VisualOutcomeType,
    VisualVerificationResult,
)
from app.vision.ocr import OCRProvider
from app.vision.preprocessing import ImagePreprocessor, default_preprocessor
from app.vision.scene import VisualSceneParser

logger = get_logger("VISION.VERIFICATION")

# ---------------------------------------------------------------------------
# Regex Matchers for Natural Language Goal Parsing
# ---------------------------------------------------------------------------

_RE_WINDOW_OPEN = re.compile(
    r"^(?:verify\s+(?:that\s+)?|check\s+(?:if\s+|whether\s+)?|is\s+|did\s+)?"
    r"(.+?)\s+(?:is\s+)?(?:open|opened|launched|running|visible|active)\??$",
    re.IGNORECASE,
)

_RE_WINDOW_CLOSED = re.compile(
    r"^(?:verify\s+(?:that\s+)?|check\s+(?:if\s+|whether\s+)?|is\s+|did\s+)?"
    r"(.+?)\s+(?:is\s+)?(?:closed|close|exited|terminated|gone|vanished|disappeared)\??$",
    re.IGNORECASE,
)

_RE_ELEMENT_STATE = re.compile(
    r"^(?:verify\s+(?:that\s+)?|check\s+(?:if\s+|whether\s+)?|is\s+)?"
    r"(?:the\s+|a\s+|an\s+)?(.+?)\s+(?:button|control|checkbox|field|input)?\s*"
    r"(enabled|disabled|checked|unchecked|empty|populated|focused|editable|clickable)\??$",
    re.IGNORECASE,
)

_RE_CONTAINER_PRESENT = re.compile(
    r"^(?:verify\s+(?:that\s+)?|check\s+(?:if\s+|whether\s+)?|is\s+)?"
    r"(?:the\s+|a\s+|an\s+)?(.+?)\s*(dialog|form|sidebar|toolbar|header|tab\s+panel|status\s+bar)\s*"
    r"(?:is\s+)?(?:open|visible|present|showing)\??$",
    re.IGNORECASE,
)

_RE_TEXT_ABSENT = re.compile(
    r"^(?:verify\s+(?:that\s+)?|check\s+(?:if\s+|whether\s+)?|did\s+)?"
    r"(?:the\s+)?(.+?)\s*(?:error|alert|warning|message|text)?\s*"
    r"(?:disappear|disappeared|vanish|vanished|gone|closed)\??$",
    re.IGNORECASE,
)

_RE_VISUAL_DELTA = re.compile(
    r"^(?:verify\s+(?:(?:the\s+)?screen\s+)?changes?|did\s+(?:the\s+screen\s+|the\s+|anything\s+)?change|what\s+changed)\??$",
    re.IGNORECASE,
)


def parse_visual_goal(text: str) -> VisualGoalSpec:
    """Parse a natural language condition or verification request into a typed VisualGoalSpec.

    Maps common phrases deterministically to appropriate VisualGoalCriterion.
    If ambiguous or complex, returns a CUSTOM_SEMANTIC criterion.
    """
    clean = " ".join(str(text or "").strip().split())
    if not clean:
        return VisualGoalSpec(
            criterion=VisualGoalCriterion.CUSTOM_SEMANTIC,
            target="",
        )

    # 1. Visual delta change query
    if _RE_VISUAL_DELTA.match(clean):
        return VisualGoalSpec(
            criterion=VisualGoalCriterion.VISUAL_DELTA,
            target="screen_change",
        )

    # 2. Element state query: e.g. "is submit button enabled", "is checkbox checked"
    m_state = _RE_ELEMENT_STATE.match(clean)
    if m_state:
        tgt = m_state.group(1).strip()
        state_str = m_state.group(2).strip().lower()
        mapping = {
            "enabled": ControlVisualState.ENABLED,
            "clickable": ControlVisualState.ENABLED,
            "disabled": ControlVisualState.DISABLED,
            "checked": ControlVisualState.CHECKED,
            "unchecked": ControlVisualState.UNCHECKED,
            "empty": ControlVisualState.EMPTY,
            "populated": ControlVisualState.POPULATED,
            "focused": ControlVisualState.FOCUSED,
            "editable": ControlVisualState.ENABLED,
        }
        if state_str in mapping and tgt:
            return VisualGoalSpec(
                criterion=VisualGoalCriterion.ELEMENT_STATE,
                target=tgt,
                expected_state=mapping[state_str],
            )

    # 3. Container query: e.g. "is login dialog open", "verify form is visible"
    m_cont = _RE_CONTAINER_PRESENT.match(clean)
    if m_cont:
        tgt = m_cont.group(1).strip()
        c_type_str = m_cont.group(2).strip().lower().replace(" ", "_")
        c_mapping = {
            "dialog": UIContainerType.DIALOG,
            "form": UIContainerType.FORM,
            "sidebar": UIContainerType.SIDEBAR,
            "toolbar": UIContainerType.TOOLBAR,
            "header": UIContainerType.HEADER,
            "tab_panel": UIContainerType.TAB_PANEL,
            "status_bar": UIContainerType.STATUS_BAR,
        }
        if c_type_str in c_mapping:
            return VisualGoalSpec(
                criterion=VisualGoalCriterion.CONTAINER_PRESENT,
                target=tgt or c_type_str,
                container_type=c_mapping[c_type_str],
            )

    # 4. Text absent / error vanished: e.g. "did the error disappear", "is the error gone"
    m_txt_abs = _RE_TEXT_ABSENT.match(clean)
    if m_txt_abs and ("error" in clean.lower() or "alert" in clean.lower() or "warning" in clean.lower() or "disappear" in clean.lower()):
        tgt = m_txt_abs.group(1).strip() or "error"
        return VisualGoalSpec(
            criterion=VisualGoalCriterion.TEXT_ABSENT,
            target=tgt,
        )

    # 5. Window closed / absent: e.g. "did notepad close", "verify calculator is closed"
    m_closed = _RE_WINDOW_CLOSED.match(clean)
    if m_closed and ("close" in clean.lower() or "exit" in clean.lower() or "gone" in clean.lower() or "disappear" in clean.lower()):
        tgt = m_closed.group(1).strip()
        if tgt.lower().startswith("the "):
            tgt = tgt[4:].strip()
        if tgt:
            return VisualGoalSpec(
                criterion=VisualGoalCriterion.WINDOW_ABSENT,
                target=tgt,
            )

    # 6. Window open / present: e.g. "is notepad open", "verify chrome is launched"
    m_open = _RE_WINDOW_OPEN.match(clean)
    if m_open and ("open" in clean.lower() or "launch" in clean.lower() or "running" in clean.lower() or "visible" in clean.lower()):
        tgt = m_open.group(1).strip()
        if tgt.lower().startswith("the "):
            tgt = tgt[4:].strip()
        if tgt:
            return VisualGoalSpec(
                criterion=VisualGoalCriterion.WINDOW_PRESENT,
                target=tgt,
            )

    # Fallback to general semantic criterion
    return VisualGoalSpec(
        criterion=VisualGoalCriterion.CUSTOM_SEMANTIC,
        target=clean,
    )


class VisualVerificationEngine:
    """Deterministic-first visual task outcome and goal-state verification engine.

    Coordinates window metadata, local OCR, structural UIScene parsing,
    control affordances, and visual deltas to evaluate whether a goal
    has been satisfied with calibrated confidence and an auditable evidence chain.
    """

    def __init__(
        self,
        *,
        ocr_provider: Optional[OCRProvider] = None,
        scene_parser: Optional[VisualSceneParser] = None,
        affordance_engine: Optional[VisualAffordanceEngine] = None,
        delta_engine: Optional[VisualDeltaEngine] = None,
        ai_provider: Optional[Any] = None,
        preprocessor: Optional[ImagePreprocessor] = None,
        prompt_builder: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize the verification engine with injected dependencies."""
        self._logger = logger or get_logger("VISION.VERIFICATION")
        self._ocr_provider = ocr_provider
        self._ai_provider = ai_provider
        self._preprocessor = preprocessor or default_preprocessor
        self._prompt_builder = prompt_builder
        self._scene_parser = scene_parser or VisualSceneParser(ocr_provider=ocr_provider)
        self._affordance_engine = affordance_engine or VisualAffordanceEngine(
            ai_provider=ai_provider,
            preprocessor=self._preprocessor,
            prompt_builder=self._prompt_builder,
            logger_instance=self._logger,
        )
        self._delta_engine = delta_engine or VisualDeltaEngine(ocr_provider=ocr_provider)

    def verify_goal(
        self,
        spec: Union[VisualGoalSpec, Dict[str, Any], str],
        observation: Optional[ScreenObservation],
        prior_observation: Optional[ScreenObservation] = None,
    ) -> VisualVerificationResult:
        """Verify whether an expected visual goal spec is achieved in the observation.

        Args:
            spec: VisualGoalSpec instance, dictionary, or string condition to parse.
            observation: The fresh ScreenObservation to inspect.
            prior_observation: Optional baseline observation for change/delta criteria.

        Returns:
            VisualVerificationResult with calibrated outcome, confidence, and evidence chain.
        """
        verif_id = f"verif-{uuid.uuid4().hex[:8]}"

        # Normalize spec
        if isinstance(spec, VisualGoalSpec):
            goal_spec = spec
        elif isinstance(spec, dict):
            goal_spec = VisualGoalSpec.from_dict(spec)
        else:
            goal_spec = parse_visual_goal(str(spec or ""))

        # 1. Security Check: Handle blocked or empty captures immediately
        if observation is None or observation.is_empty or observation.metadata.get("blocked"):
            blocked_reason = (
                observation.metadata.get("block_reason")
                if observation
                else "Screen observation was blocked or unavailable."
            )
            item = VisualEvidenceItem(
                source="security_policy",
                description=f"Capture blocked by vision privacy policy: {blocked_reason}",
                confidence=1.0,
            )
            return VisualVerificationResult(
                verification_id=verif_id,
                observation_id=observation.observation_id if observation else "",
                outcome=VisualOutcomeType.BLOCKED,
                goal_spec=goal_spec,
                confidence=1.0,
                evidence_chain=(item,),
                explanation="Verification blocked by visual security policy.",
                evaluation_source="security_policy",
                metadata={"blocked": True, "reason": blocked_reason},
            )

        obs_id = observation.observation_id

        # Dispatch to deterministic criterion evaluators
        crit = goal_spec.criterion

        if crit == VisualGoalCriterion.WINDOW_PRESENT:
            return self._verify_window_present(verif_id, obs_id, goal_spec, observation)

        if crit == VisualGoalCriterion.WINDOW_ABSENT:
            return self._verify_window_absent(verif_id, obs_id, goal_spec, observation)

        if crit == VisualGoalCriterion.TEXT_PRESENT:
            return self._verify_text_present(verif_id, obs_id, goal_spec, observation)

        if crit == VisualGoalCriterion.TEXT_ABSENT:
            return self._verify_text_absent(verif_id, obs_id, goal_spec, observation)

        if crit == VisualGoalCriterion.ELEMENT_STATE:
            return self._verify_element_state(verif_id, obs_id, goal_spec, observation)

        if crit == VisualGoalCriterion.CONTAINER_PRESENT:
            return self._verify_container_present(verif_id, obs_id, goal_spec, observation)

        if crit == VisualGoalCriterion.VISUAL_DELTA:
            return self._verify_visual_delta(verif_id, obs_id, goal_spec, observation, prior_observation)

        # CUSTOM_SEMANTIC fallback
        return self._verify_custom_semantic(verif_id, obs_id, goal_spec, observation, prior_observation=prior_observation)

    # -----------------------------------------------------------------------
    # Deterministic Evaluation Branches
    # -----------------------------------------------------------------------

    def _verify_window_present(
        self,
        verif_id: str,
        obs_id: str,
        spec: VisualGoalSpec,
        observation: ScreenObservation,
    ) -> VisualVerificationResult:
        """Verify target window is active or visible using window metadata and OCR evidence."""
        target = spec.target.strip().lower()
        if not target:
            item = VisualEvidenceItem(
                source="goal_validation",
                description="No target window name provided for WINDOW_PRESENT verification.",
                confidence=1.0,
            )
            return VisualVerificationResult(
                verification_id=verif_id,
                observation_id=obs_id,
                outcome=VisualOutcomeType.UNCERTAIN,
                goal_spec=spec,
                confidence=0.0,
                evidence_chain=(item,),
                explanation="Uncertain. No target window name was specified.",
                evaluation_source="deterministic_window",
            )

        meta = observation.metadata or {}
        cap_meta = observation.capture.metadata if observation.capture else {}
        win_title = str(meta.get("window_title") or cap_meta.get("window_title") or "").strip()
        proc_name = str(meta.get("process_name") or cap_meta.get("process_name") or "").strip()

        evidence: List[VisualEvidenceItem] = []

        # 1. Check window title match
        if win_title and target in win_title.lower():
            evidence.append(
                VisualEvidenceItem(
                    source="window_title",
                    description=f"Active window title '{win_title}' contains target '{spec.target}'.",
                    confidence=0.95 if target == win_title.lower() else 0.88,
                )
            )
            return VisualVerificationResult(
                verification_id=verif_id,
                observation_id=obs_id,
                outcome=VisualOutcomeType.VERIFIED,
                goal_spec=spec,
                confidence=evidence[0].confidence,
                evidence_chain=tuple(evidence),
                explanation=f"Verified. {spec.target} is open and active.",
                evaluation_source="deterministic_window",
            )

        # 2. Check process name match
        if proc_name and target in proc_name.lower():
            evidence.append(
                VisualEvidenceItem(
                    source="process_name",
                    description=f"Active process '{proc_name}' matches target '{spec.target}'.",
                    confidence=0.85,
                )
            )
            return VisualVerificationResult(
                verification_id=verif_id,
                observation_id=obs_id,
                outcome=VisualOutcomeType.VERIFIED,
                goal_spec=spec,
                confidence=0.85,
                evidence_chain=tuple(evidence),
                explanation=f"Verified. Process {spec.target} is active.",
                evaluation_source="deterministic_window",
            )

        # 3. Check OCR text if available
        ocr_res = self._get_or_run_ocr(observation)
        if ocr_res and ocr_res.text:
            ocr_text = ocr_res.text.lower()
            if target in ocr_text:
                evidence.append(
                    VisualEvidenceItem(
                        source="ocr_text",
                        description=f"Target text '{spec.target}' detected on screen via OCR.",
                        confidence=0.80,
                    )
                )
                return VisualVerificationResult(
                    verification_id=verif_id,
                    observation_id=obs_id,
                    outcome=VisualOutcomeType.VERIFIED,
                    goal_spec=spec,
                    confidence=0.80,
                    evidence_chain=tuple(evidence),
                    explanation=f"Verified. {spec.target} text is present on the screen.",
                    evaluation_source="deterministic_ocr",
                )

        # 4. Optional VLM fallback if multimodal AI provider is available
        if (
            self._ai_provider is not None
            and getattr(self._ai_provider, "supports_multimodal", False)
            and observation.capture
            and not observation.capture.is_empty
        ):
            return self._verify_custom_semantic(verif_id, obs_id, spec, observation, evidence_prefix=evidence)

        # If window is present but belongs to an entirely different application
        if win_title and target not in win_title.lower():
            evidence.append(
                VisualEvidenceItem(
                    source="window_title",
                    description=f"Active window is '{win_title}', target '{spec.target}' not found.",
                    confidence=0.85,
                )
            )
            return VisualVerificationResult(
                verification_id=verif_id,
                observation_id=obs_id,
                outcome=VisualOutcomeType.NOT_VERIFIED,
                goal_spec=spec,
                confidence=0.85,
                evidence_chain=tuple(evidence),
                explanation=f"Not verified. {spec.target} is not the active window.",
                evaluation_source="deterministic_window",
            )

        item = VisualEvidenceItem(
            source="window_metadata",
            description="Insufficient window metadata or OCR evidence to confirm presence.",
            confidence=0.5,
        )
        return VisualVerificationResult(
            verification_id=verif_id,
            observation_id=obs_id,
            outcome=VisualOutcomeType.UNCERTAIN,
            goal_spec=spec,
            confidence=0.4,
            evidence_chain=(item,),
            explanation=f"Uncertain. Could not confirm whether {spec.target} is open.",
            evaluation_source="deterministic_window",
        )

    def _verify_window_absent(
        self,
        verif_id: str,
        obs_id: str,
        spec: VisualGoalSpec,
        observation: ScreenObservation,
    ) -> VisualVerificationResult:
        """Verify target window is closed or no longer visible."""
        target = spec.target.strip().lower()
        if not target:
            item = VisualEvidenceItem(
                source="goal_validation",
                description="No target window name provided for WINDOW_ABSENT verification.",
                confidence=1.0,
            )
            return VisualVerificationResult(
                verification_id=verif_id,
                observation_id=obs_id,
                outcome=VisualOutcomeType.UNCERTAIN,
                goal_spec=spec,
                confidence=0.0,
                evidence_chain=(item,),
                explanation="Uncertain. No target window name was specified.",
                evaluation_source="deterministic_window",
            )

        meta = observation.metadata or {}
        cap_meta = observation.capture.metadata if observation.capture else {}
        win_title = str(meta.get("window_title") or cap_meta.get("window_title") or "").strip()
        proc_name = str(meta.get("process_name") or cap_meta.get("process_name") or "").strip()

        evidence: List[VisualEvidenceItem] = []

        # If target IS still in title or process -> NOT_VERIFIED (it is still open)
        if (win_title and target in win_title.lower()) or (proc_name and target in proc_name.lower()):
            evidence.append(
                VisualEvidenceItem(
                    source="window_title",
                    description=f"Window '{win_title}' still contains target '{spec.target}'.",
                    confidence=0.90,
                )
            )
            return VisualVerificationResult(
                verification_id=verif_id,
                observation_id=obs_id,
                outcome=VisualOutcomeType.NOT_VERIFIED,
                goal_spec=spec,
                confidence=0.90,
                evidence_chain=tuple(evidence),
                explanation=f"Not verified. {spec.target} is still open.",
                evaluation_source="deterministic_window",
            )

        # Check OCR to see if window content is still visible
        ocr_res = self._get_or_run_ocr(observation)
        if ocr_res and ocr_res.text and target in ocr_res.text.lower():
            evidence.append(
                VisualEvidenceItem(
                    source="ocr_text",
                    description=f"Target text '{spec.target}' is still detectable via OCR.",
                    confidence=0.75,
                )
            )
            return VisualVerificationResult(
                verification_id=verif_id,
                observation_id=obs_id,
                outcome=VisualOutcomeType.NOT_VERIFIED,
                goal_spec=spec,
                confidence=0.75,
                evidence_chain=tuple(evidence),
                explanation=f"Not verified. {spec.target} content is still visible on screen.",
                evaluation_source="deterministic_ocr",
            )

        # Target is not active title, process, or visible OCR
        evidence.append(
            VisualEvidenceItem(
                source="window_metadata",
                description=f"Active window '{win_title}' is not '{spec.target}'.",
                confidence=0.85,
            )
        )
        return VisualVerificationResult(
            verification_id=verif_id,
            observation_id=obs_id,
            outcome=VisualOutcomeType.VERIFIED,
            goal_spec=spec,
            confidence=0.85,
            evidence_chain=tuple(evidence),
            explanation=f"Verified. {spec.target} is no longer open.",
            evaluation_source="deterministic_window",
        )

    def _verify_text_present(
        self,
        verif_id: str,
        obs_id: str,
        spec: VisualGoalSpec,
        observation: ScreenObservation,
    ) -> VisualVerificationResult:
        """Verify target text is visible on the screen."""
        target_text = (spec.expected_text or spec.target or "").strip().lower()
        if not target_text:
            item = VisualEvidenceItem(
                source="goal_validation",
                description="No target text provided for TEXT_PRESENT verification.",
                confidence=1.0,
            )
            return VisualVerificationResult(
                verification_id=verif_id,
                observation_id=obs_id,
                outcome=VisualOutcomeType.UNCERTAIN,
                goal_spec=spec,
                confidence=0.0,
                evidence_chain=(item,),
                explanation="Uncertain. No target text was specified.",
                evaluation_source="deterministic_ocr",
            )

        ocr_res = self._get_or_run_ocr(observation)
        if ocr_res is None or not ocr_res.text:
            item = VisualEvidenceItem(
                source="ocr",
                description="OCR extraction yielded no detectable text.",
                confidence=0.8,
            )
            return VisualVerificationResult(
                verification_id=verif_id,
                observation_id=obs_id,
                outcome=VisualOutcomeType.NOT_VERIFIED,
                goal_spec=spec,
                confidence=0.8,
                evidence_chain=(item,),
                explanation=f"Not verified. Expected text '{spec.target}' was not detected.",
                evaluation_source="deterministic_ocr",
            )

        found_match = target_text in ocr_res.text.lower()
        if found_match:
            item = VisualEvidenceItem(
                source="ocr_text",
                description=f"Detected matching text for '{spec.target}' on screen.",
                confidence=0.90,
            )
            return VisualVerificationResult(
                verification_id=verif_id,
                observation_id=obs_id,
                outcome=VisualOutcomeType.VERIFIED,
                goal_spec=spec,
                confidence=0.90,
                evidence_chain=(item,),
                explanation=f"Verified. Text '{spec.target}' is visible on screen.",
                evaluation_source="deterministic_ocr",
            )
        else:
            item = VisualEvidenceItem(
                source="ocr_text",
                description=f"OCR scanned {len(ocr_res.blocks)} text blocks; target '{spec.target}' not present.",
                confidence=0.85,
            )
            return VisualVerificationResult(
                verification_id=verif_id,
                observation_id=obs_id,
                outcome=VisualOutcomeType.NOT_VERIFIED,
                goal_spec=spec,
                confidence=0.85,
                evidence_chain=(item,),
                explanation=f"Not verified. Expected text '{spec.target}' was not detected.",
                evaluation_source="deterministic_ocr",
            )

    def _verify_text_absent(
        self,
        verif_id: str,
        obs_id: str,
        spec: VisualGoalSpec,
        observation: ScreenObservation,
    ) -> VisualVerificationResult:
        """Verify target text (such as an error message or alert) is absent."""
        target_text = (spec.expected_text or spec.target or "").strip().lower()
        if not target_text:
            item = VisualEvidenceItem(
                source="goal_validation",
                description="No target text provided for TEXT_ABSENT verification.",
                confidence=1.0,
            )
            return VisualVerificationResult(
                verification_id=verif_id,
                observation_id=obs_id,
                outcome=VisualOutcomeType.UNCERTAIN,
                goal_spec=spec,
                confidence=0.0,
                evidence_chain=(item,),
                explanation="Uncertain. No target text was specified.",
                evaluation_source="deterministic_ocr",
            )

        ocr_res = self._get_or_run_ocr(observation)
        if ocr_res is None or not ocr_res.text:
            item = VisualEvidenceItem(
                source="ocr",
                description="No text detected on screen; target is absent.",
                confidence=0.85,
            )
            return VisualVerificationResult(
                verification_id=verif_id,
                observation_id=obs_id,
                outcome=VisualOutcomeType.VERIFIED,
                goal_spec=spec,
                confidence=0.85,
                evidence_chain=(item,),
                explanation=f"Verified. Text '{spec.target}' is not present.",
                evaluation_source="deterministic_ocr",
            )

        found_match = target_text in ocr_res.text.lower()
        if found_match:
            item = VisualEvidenceItem(
                source="ocr_text",
                description=f"Text '{spec.target}' is still detectable on screen.",
                confidence=0.90,
            )
            return VisualVerificationResult(
                verification_id=verif_id,
                observation_id=obs_id,
                outcome=VisualOutcomeType.NOT_VERIFIED,
                goal_spec=spec,
                confidence=0.90,
                evidence_chain=(item,),
                explanation=f"Not verified. Text '{spec.target}' is still present on screen.",
                evaluation_source="deterministic_ocr",
            )
        else:
            item = VisualEvidenceItem(
                source="ocr_text",
                description=f"OCR confirmed target '{spec.target}' is absent across {len(ocr_res.blocks)} blocks.",
                confidence=0.85,
            )
            return VisualVerificationResult(
                verification_id=verif_id,
                observation_id=obs_id,
                outcome=VisualOutcomeType.VERIFIED,
                goal_spec=spec,
                confidence=0.85,
                evidence_chain=(item,),
                explanation=f"Verified. Text '{spec.target}' is not present.",
                evaluation_source="deterministic_ocr",
            )

    def _verify_element_state(
        self,
        verif_id: str,
        obs_id: str,
        spec: VisualGoalSpec,
        observation: ScreenObservation,
    ) -> VisualVerificationResult:
        """Verify an interactive UI control matches expected state (e.g. ENABLED, CHECKED, EMPTY)."""
        target = spec.target.strip()
        expected = spec.expected_state
        if not target or expected is None:
            item = VisualEvidenceItem(
                source="goal_validation",
                description="Target element name and expected state are required for ELEMENT_STATE.",
                confidence=1.0,
            )
            return VisualVerificationResult(
                verification_id=verif_id,
                observation_id=obs_id,
                outcome=VisualOutcomeType.UNCERTAIN,
                goal_spec=spec,
                confidence=0.0,
                evidence_chain=(item,),
                explanation="Uncertain. Missing element target or expected state.",
                evaluation_source="deterministic_affordance",
            )

        # Use VisualAffordanceEngine to inspect the element state
        scene = self._scene_parser.parse_scene(observation)
        matched_el = None
        affordance_res = None

        if hasattr(self._affordance_engine, "inspect_element_state"):
            try:
                affordance_res = self._affordance_engine.inspect_element_state(observation, target)
                if affordance_res:
                    matched_el = affordance_res.element
            except Exception:
                pass

        if affordance_res is None:
            if hasattr(self._affordance_engine, "_resolve_target_element"):
                matched_el, _ = self._affordance_engine._resolve_target_element(scene, target.lower())
            if matched_el is None and scene.interactive_elements:
                for el in scene.interactive_elements:
                    if target.lower() in (el.name or "").lower():
                        matched_el = el
                        break
            if matched_el is not None:
                ocr_res = self._get_or_run_ocr(observation)
                affordance_res = self._affordance_engine.inspect_element_affordance(
                    matched_el,
                    capture=observation.capture if observation else None,
                    ocr_blocks=ocr_res.blocks if ocr_res else None,
                )

        if not affordance_res or affordance_res.element is None:
            item = VisualEvidenceItem(
                source="scene_parsing",
                description=f"Element '{target}' was not located among {len(scene.interactive_elements)} controls.",
                confidence=0.8,
            )
            return VisualVerificationResult(
                verification_id=verif_id,
                observation_id=obs_id,
                outcome=VisualOutcomeType.NOT_VERIFIED,
                goal_spec=spec,
                confidence=0.8,
                evidence_chain=(item,),
                explanation=f"Not verified. Control '{target}' was not found on screen.",
                evaluation_source="deterministic_affordance",
            )

        detected_state = affordance_res.detected_state
        item = VisualEvidenceItem(
            source="control_affordance",
            description=f"Control '{target}' detected in state '{detected_state.value}' (evidence: {affordance_res.evidence}).",
            confidence=affordance_res.confidence,
        )

        if detected_state == ControlVisualState.UNCERTAIN:
            return VisualVerificationResult(
                verification_id=verif_id,
                observation_id=obs_id,
                outcome=VisualOutcomeType.UNCERTAIN,
                goal_spec=spec,
                confidence=affordance_res.confidence,
                evidence_chain=(item,),
                explanation=f"Uncertain. Visual state of control '{target}' is ambiguous ({affordance_res.evidence}).",
                evaluation_source="deterministic_affordance",
            )

        if detected_state == expected:
            return VisualVerificationResult(
                verification_id=verif_id,
                observation_id=obs_id,
                outcome=VisualOutcomeType.VERIFIED,
                goal_spec=spec,
                confidence=affordance_res.confidence,
                evidence_chain=(item,),
                explanation=f"Verified. Control '{target}' is {expected.value}.",
                evaluation_source="deterministic_affordance",
            )
        else:
            return VisualVerificationResult(
                verification_id=verif_id,
                observation_id=obs_id,
                outcome=VisualOutcomeType.NOT_VERIFIED,
                goal_spec=spec,
                confidence=affordance_res.confidence,
                evidence_chain=(item,),
                explanation=f"Not verified. Control '{target}' is {detected_state.value}, expected {expected.value}.",
                evaluation_source="deterministic_affordance",
            )

    def _verify_container_present(
        self,
        verif_id: str,
        obs_id: str,
        spec: VisualGoalSpec,
        observation: ScreenObservation,
    ) -> VisualVerificationResult:
        """Verify a specific UI container (dialog, form, sidebar) is present on screen."""
        expected_type = spec.container_type or UIContainerType.DIALOG
        target_name = spec.target.strip().lower()

        scene = self._scene_parser.parse_scene(observation)
        matching_containers = [
            c for c in scene.containers
            if c.container_type == expected_type
        ]

        # Further filter by label/target if target specified and not just generic container name
        if target_name and target_name not in ("dialog", "form", "sidebar", "toolbar", "header"):
            specific_matches = [
                c for c in matching_containers
                if c.label and target_name in c.label.lower()
            ]
            if specific_matches:
                matching_containers = specific_matches

        if matching_containers:
            best_match = matching_containers[0]
            label_text = f" '{best_match.label}'" if best_match.label else ""
            item = VisualEvidenceItem(
                source="scene_containers",
                description=f"Detected {expected_type.value} container{label_text} at bounds {best_match.bounds}.",
                confidence=best_match.confidence,
            )
            return VisualVerificationResult(
                verification_id=verif_id,
                observation_id=obs_id,
                outcome=VisualOutcomeType.VERIFIED,
                goal_spec=spec,
                confidence=best_match.confidence,
                evidence_chain=(item,),
                explanation=f"Verified. The {expected_type.value}{label_text} is visible.",
                evaluation_source="deterministic_scene",
            )
        else:
            item = VisualEvidenceItem(
                source="scene_containers",
                description=f"No {expected_type.value} container detected among {len(scene.containers)} parsed containers.",
                confidence=0.85,
            )
            return VisualVerificationResult(
                verification_id=verif_id,
                observation_id=obs_id,
                outcome=VisualOutcomeType.NOT_VERIFIED,
                goal_spec=spec,
                confidence=0.85,
                evidence_chain=(item,),
                explanation=f"Not verified. The {expected_type.value} was not detected on screen.",
                evaluation_source="deterministic_scene",
            )

    def _verify_visual_delta(
        self,
        verif_id: str,
        obs_id: str,
        spec: VisualGoalSpec,
        observation: ScreenObservation,
        prior_observation: Optional[ScreenObservation],
    ) -> VisualVerificationResult:
        """Verify that an expected visual change occurred between prior and current observations."""
        if prior_observation is None:
            item = VisualEvidenceItem(
                source="delta_engine",
                description="Prior observation baseline was not provided for VISUAL_DELTA verification.",
                confidence=1.0,
            )
            return VisualVerificationResult(
                verification_id=verif_id,
                observation_id=obs_id,
                outcome=VisualOutcomeType.UNCERTAIN,
                goal_spec=spec,
                confidence=0.0,
                evidence_chain=(item,),
                explanation="Uncertain. Cannot verify visual delta without a baseline observation.",
                evaluation_source="deterministic_delta",
            )

        if hasattr(self._delta_engine, "compare_observations"):
            delta_res: VisualDeltaResult = self._delta_engine.compare_observations(prior_observation, observation)
        elif hasattr(self._delta_engine, "compute_delta"):
            delta_res = self._delta_engine.compute_delta(prior_observation, observation)
        else:
            delta_res = self._delta_engine(prior_observation, observation)
        target = spec.target.strip().lower()

        if not delta_res.meaningful_change_detected:
            item = VisualEvidenceItem(
                source="delta_engine",
                description="Perceptual diff and OCR analysis confirmed no meaningful visual changes.",
                confidence=delta_res.confidence,
            )
            return VisualVerificationResult(
                verification_id=verif_id,
                observation_id=obs_id,
                outcome=VisualOutcomeType.NOT_VERIFIED,
                goal_spec=spec,
                confidence=delta_res.confidence,
                evidence_chain=(item,),
                explanation="Not verified. No visual changes were detected.",
                evaluation_source="deterministic_delta",
            )

        # Meaningful change occurred
        evidence_desc = delta_res.explanation or "Visual delta detected."
        item = VisualEvidenceItem(
            source="delta_engine",
            description=evidence_desc,
            confidence=delta_res.confidence,
        )

        # If specific target was specified (e.g. element name or text)
        if target and target != "screen_change":
            found_in_added = any(target in t.lower() for t in (delta_res.added_texts or ()))
            found_in_elem = any(
                target in ((c.after_element.name if c.after_element else (c.before_element.name if c.before_element else "")) or "").lower()
                for c in (delta_res.element_changes or ())
            )
            if found_in_added or found_in_elem:
                return VisualVerificationResult(
                    verification_id=verif_id,
                    observation_id=obs_id,
                    outcome=VisualOutcomeType.VERIFIED,
                    goal_spec=spec,
                    confidence=delta_res.confidence,
                    evidence_chain=(item,),
                    explanation=f"Verified. Visual change involving '{spec.target}' detected.",
                    evaluation_source="deterministic_delta",
                )
            else:
                return VisualVerificationResult(
                    verification_id=verif_id,
                    observation_id=obs_id,
                    outcome=VisualOutcomeType.UNCERTAIN,
                    goal_spec=spec,
                    confidence=0.6,
                    evidence_chain=(item,),
                    explanation=f"Uncertain. Visual changes occurred, but could not isolate '{spec.target}'.",
                    evaluation_source="deterministic_delta",
                )

        return VisualVerificationResult(
            verification_id=verif_id,
            observation_id=obs_id,
            outcome=VisualOutcomeType.VERIFIED,
            goal_spec=spec,
            confidence=delta_res.confidence,
            evidence_chain=(item,),
            explanation="Verified. Visual changes were detected on screen.",
            evaluation_source="deterministic_delta",
        )

    def _verify_custom_semantic(
        self,
        verif_id: str,
        obs_id: str,
        spec: VisualGoalSpec,
        observation: ScreenObservation,
        evidence_prefix: Optional[Sequence[VisualEvidenceItem]] = None,
        prior_observation: Optional[ScreenObservation] = None,
    ) -> VisualVerificationResult:
        """Verify custom natural-language condition using multimodal VLM fallback when available."""
        if spec.criterion == VisualGoalCriterion.WINDOW_PRESENT:
            condition = f"{spec.target} is open"
        elif spec.criterion == VisualGoalCriterion.WINDOW_ABSENT:
            condition = f"{spec.target} is closed"
        else:
            condition = (spec.expected_text or spec.target).strip()
        evidence: List[VisualEvidenceItem] = list(evidence_prefix or [])

        is_click = (
            (spec.metadata and spec.metadata.get("action") == "click")
            or "click" in str(spec.metadata or "").lower()
        )

        # 1. Deterministic evaluation
        ocr_res = self._get_or_run_ocr(observation)
        prior_ocr = self._get_or_run_ocr(prior_observation) if prior_observation else None

        if is_click:
            # For deterministic click verification:
            # The continued presence of the clicked button's label alone is NOT evidence of success.
            target_label = (spec.target or "").strip().lower()

            # a) Check if clicked element disappeared
            if prior_ocr and prior_ocr.text and target_label:
                if target_label in prior_ocr.text.lower() and (not ocr_res or target_label not in ocr_res.text.lower()):
                    evidence.append(
                        VisualEvidenceItem(
                            source="deterministic_transition",
                            description=f"Clicked element '{spec.target}' disappeared from screen.",
                            confidence=0.88,
                        )
                    )

            # b) Check if window title or process changed
            if prior_observation is not None and observation is not None:
                p_meta = getattr(prior_observation, "metadata", {}) or {}
                c_meta = getattr(observation, "metadata", {}) or {}
                p_title = (p_meta.get("window_title") or "").strip().lower()
                c_title = (c_meta.get("window_title") or "").strip().lower()
                if p_title and c_title and p_title != c_title:
                    evidence.append(
                        VisualEvidenceItem(
                            source="deterministic_window_change",
                            description=f"Active window title changed post-click ('{p_title}' -> '{c_title}').",
                            confidence=0.90,
                        )
                    )

            # c) Check if modal/dialog closed
            if prior_observation is not None and observation is not None:
                p_meta = getattr(prior_observation, "metadata", {}) or {}
                c_meta = getattr(observation, "metadata", {}) or {}
                if p_meta.get("is_modal") and not c_meta.get("is_modal"):
                    evidence.append(
                        VisualEvidenceItem(
                            source="deterministic_modal_dismissal",
                            description="Modal dialog was dismissed post-click.",
                            confidence=0.90,
                        )
                    )

            # d) Check if visual delta occurred between T0 (prior) and T1 (post)
            if prior_observation is not None and observation is not None:
                try:
                    delta = self._delta_engine.detect_delta(prior_observation, observation)
                    if delta and delta.has_changes and delta.delta_percentage > 0.001:
                        evidence.append(
                            VisualEvidenceItem(
                                source="deterministic_delta",
                                description=f"Observable UI change detected ({delta.delta_percentage * 100:.2f}% screen delta).",
                                confidence=0.85,
                            )
                        )
                except Exception as exc:
                    self._logger.debug("Delta detection error during click verification: %s", exc)

            # e) Check explicitly specified expected outcome text (distinct from target label)
            if spec.expected_text and spec.expected_text.strip().lower() != target_label:
                exp_clean = spec.expected_text.strip()
                if ocr_res and ocr_res.text and exp_clean.lower() in ocr_res.text.lower():
                    evidence.append(
                        VisualEvidenceItem(
                            source="deterministic_expected_text",
                            description=f"Expected text '{exp_clean}' appeared post-click.",
                            confidence=0.88,
                        )
                    )
        else:
            # General non-click text matching: verifying expected text is present
            if ocr_res and ocr_res.text:
                if condition.lower() in ocr_res.text.lower():
                    evidence.append(
                        VisualEvidenceItem(
                            source="deterministic_ocr",
                            description=f"Supporting text match for '{condition}' found via OCR.",
                            confidence=0.80,
                        )
                    )

        # 2. Check if multimodal AI provider is available
        ai_p = self._ai_provider
        if ai_p is not None and getattr(ai_p, "supports_multimodal", False) and observation.capture and not observation.capture.is_empty:
            try:
                image_part = self._preprocessor.to_image_part(observation.capture)
                prompt = (
                    "You are J.A.R.V.I.S Visual Verification Engine. Inspect the provided screenshot and determine "
                    f"whether the following condition is true:\n\n"
                    f"Condition: {condition}\n\n"
                    "Instructions:\n"
                    "1. Assess whether visible evidence confirms or refutes this condition.\n"
                    "2. Your response MUST begin with exactly one of these three verdicts:\n"
                    "   - VERIFIED: <concise explanation of why the condition is met>\n"
                    "   - NOT VERIFIED: <concise explanation of why the condition is not met>\n"
                    "   - UNCERTAIN: <concise explanation of why visible evidence is insufficient>\n"
                    "3. Be concise, direct, and factual. Base your judgment strictly on visible evidence."
                )
                if self._prompt_builder is not None and hasattr(self._prompt_builder, "build_payload"):
                    payload = self._prompt_builder.build_payload(query=prompt, images=[image_part])
                else:
                    payload = {"contents": [{"parts": [{"text": prompt}, image_part]}]}
                response = ai_p.generate(payload)
                raw_text = getattr(response, "content", str(response)).strip()

                upper = raw_text.upper()
                if upper.startswith("VERIFIED:"):
                    reason = raw_text.split(":", 1)[1].strip()
                    evidence.append(
                        VisualEvidenceItem(
                            source="multimodal_vlm",
                            description=f"VLM confirmed condition: {reason}",
                            confidence=0.90,
                        )
                    )
                    return VisualVerificationResult(
                        verification_id=verif_id,
                        observation_id=obs_id,
                        outcome=VisualOutcomeType.VERIFIED,
                        goal_spec=spec,
                        confidence=0.90,
                        evidence_chain=tuple(evidence),
                        explanation=reason,
                        evaluation_source="multimodal_vlm",
                    )
                elif upper.startswith("NOT VERIFIED:"):
                    reason = raw_text.split(":", 1)[1].strip()
                    evidence.append(
                        VisualEvidenceItem(
                            source="multimodal_vlm",
                            description=f"VLM refuted condition: {reason}",
                            confidence=0.88,
                        )
                    )
                    return VisualVerificationResult(
                        verification_id=verif_id,
                        observation_id=obs_id,
                        outcome=VisualOutcomeType.NOT_VERIFIED,
                        goal_spec=spec,
                        confidence=0.88,
                        evidence_chain=tuple(evidence),
                        explanation=reason,
                        evaluation_source="multimodal_vlm",
                    )
                else:
                    reason = raw_text.split(":", 1)[1].strip() if ":" in raw_text else raw_text
                    evidence.append(
                        VisualEvidenceItem(
                            source="multimodal_vlm",
                            description=f"VLM reported uncertainty: {reason}",
                            confidence=0.5,
                        )
                    )
                    return VisualVerificationResult(
                        verification_id=verif_id,
                        observation_id=obs_id,
                        outcome=VisualOutcomeType.UNCERTAIN,
                        goal_spec=spec,
                        confidence=0.5,
                        evidence_chain=tuple(evidence),
                        explanation=reason or "Visual evidence is insufficient to verify condition.",
                        evaluation_source="multimodal_vlm",
                    )
            except Exception as exc:
                self._logger.warning("Multimodal VLM verification encountered error: %s", exc)
                evidence.append(
                    VisualEvidenceItem(
                        source="multimodal_vlm",
                        description=f"VLM verification attempt failed: {str(exc)}",
                        confidence=0.2,
                    )
                )

        # 3. If VLM unavailable or failed, evaluate deterministic evidence
        if evidence:
            valid_sources = {
                "deterministic_ocr",
                "deterministic_transition",
                "deterministic_window_change",
                "deterministic_modal_dismissal",
                "deterministic_delta",
                "deterministic_expected_text",
            }
            matching = [e for e in evidence if e.source in valid_sources]
            if matching:
                best_evidence = max(matching, key=lambda e: e.confidence)
                return VisualVerificationResult(
                    verification_id=verif_id,
                    observation_id=obs_id,
                    outcome=VisualOutcomeType.VERIFIED,
                    goal_spec=spec,
                    confidence=best_evidence.confidence,
                    evidence_chain=tuple(evidence),
                    explanation=f"Verified. {best_evidence.description}",
                    evaluation_source=best_evidence.source,
                )

        # Fallback when evidence is insufficient and VLM is unavailable
        if is_click and prior_observation is not None:
            explanation = f"Uncertain. Click on '{spec.target}' produced no observable UI transition or state change."
        else:
            explanation = "Uncertain. Visible evidence is insufficient to confirm condition."

        evidence.append(
            VisualEvidenceItem(
                source="verification_engine",
                description="Deterministic evidence is inconclusive and VLM is unavailable.",
                confidence=0.4,
            )
        )
        return VisualVerificationResult(
            verification_id=verif_id,
            observation_id=obs_id,
            outcome=VisualOutcomeType.UNCERTAIN,
            goal_spec=spec,
            confidence=0.4,
            evidence_chain=tuple(evidence),
            explanation=explanation,
            evaluation_source="deterministic",
        )

    # -----------------------------------------------------------------------
    # Helper: Safe OCR Access
    # -----------------------------------------------------------------------

    def _get_or_run_ocr(self, observation: ScreenObservation) -> Optional[OCRResult]:
        """Extract OCRResult from observation or execute local OCR provider."""
        if getattr(observation, "ocr_result", None) is not None:
            return observation.ocr_result
        if observation and observation.metadata and "ocr_result" in observation.metadata:
            return observation.metadata["ocr_result"]
        if self._ocr_provider is not None and observation and observation.capture and not observation.capture.is_empty:
            try:
                return self._ocr_provider.extract_text(observation.capture)
            except Exception as exc:
                self._logger.debug("OCR extraction during verification failed: %s", exc)
        return None

    def verify_goal_with_reobservation(
        self,
        goal_spec: VisualGoalSpec,
        capture_fn: Optional[Callable[[], Optional[ScreenObservation]]],
        prior_observation: Optional[ScreenObservation] = None,
    ) -> VisualVerificationResult:
        """Perform a single, safe re-observation capture when initial verification is UNCERTAIN.

        Guarantees:
        - NEVER re-dispatches physical input.
        - Capped at exactly ONE re-observation capture (no retry loops).
        - If capture_fn is None or returns None -> returns NOT_VERIFIED (fail closed).
        - If re-observation verification is VERIFIED -> returns VERIFIED.
        - If re-observation verification is UNCERTAIN -> returns NOT_VERIFIED (fail closed).
        - If re-observation verification is NOT_VERIFIED -> returns NOT_VERIFIED.
        """
        if capture_fn is None:
            return VisualVerificationResult(
                verification_id=f"reobs_{uuid.uuid4().hex[:8]}",
                observation_id="",
                outcome=VisualOutcomeType.NOT_VERIFIED,
                goal_spec=goal_spec,
                confidence=0.0,
                evidence_chain=(),
                explanation="Re-observation skipped: no screen capture function provided.",
                evaluation_source="reobservation_guard",
            )

        try:
            obs = capture_fn()
        except Exception as exc:
            self._logger.warning("Re-observation screen capture failed: %s", exc)
            return VisualVerificationResult(
                verification_id=f"reobs_{uuid.uuid4().hex[:8]}",
                observation_id="",
                outcome=VisualOutcomeType.NOT_VERIFIED,
                goal_spec=goal_spec,
                confidence=0.0,
                evidence_chain=(),
                explanation=f"Re-observation capture error: {str(exc)}",
                evaluation_source="reobservation_guard",
            )

        if obs is None:
            return VisualVerificationResult(
                verification_id=f"reobs_{uuid.uuid4().hex[:8]}",
                observation_id="",
                outcome=VisualOutcomeType.NOT_VERIFIED,
                goal_spec=goal_spec,
                confidence=0.0,
                evidence_chain=(),
                explanation="Re-observation capture returned empty observation.",
                evaluation_source="reobservation_guard",
            )

        # Run verify_goal on fresh observation
        reobs_result = self.verify_goal(
            goal_spec,
            obs,
            prior_observation=prior_observation,
        )

        if reobs_result.outcome == VisualOutcomeType.VERIFIED:
            return reobs_result
        else:
            # UNCERTAIN or NOT_VERIFIED on re-observation fails closed to NOT_VERIFIED
            return VisualVerificationResult(
                verification_id=reobs_result.verification_id,
                observation_id=reobs_result.observation_id,
                outcome=VisualOutcomeType.NOT_VERIFIED,
                goal_spec=goal_spec,
                confidence=reobs_result.confidence,
                evidence_chain=reobs_result.evidence_chain,
                explanation=f"Re-observation inconclusive: {reobs_result.explanation}",
                evaluation_source=reobs_result.evaluation_source,
            )
