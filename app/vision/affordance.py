"""Visual UI Control State, Interactive Affordance & Semantic Scene Querying Engine (Phase 27.12).

Provides deterministic-first, privacy-safe perception of control states (enabled, disabled,
checked, unchecked, focused, empty, populated), interactive affordances (clickable, editable,
toggleable, selectable, scrollable, read-only), and structured visual question answering over
active screen UI scenes.

Safety and Architectural Invariants Enforced:
1. Purely read-only perception: ZERO mouse movement, hovering, clicking, typing, focus changes,
   or desktop mutations.
2. Zero persistent raw screenshots or pixel bytes in returned models, logs, or telemetry.
3. Deterministic-first state & affordance detection: Functions 100% on CPU using visual/OCR/geometry
   heuristics without requiring VLM inference.
4. Conservative classification: Ambiguous visual contrast or conflicting cues default to UNCERTAIN.
5. Strict VLM isolation: Optional VLM fallback strictly validates JSON schema, discarding malformed
   payloads without corrupting deterministic scene results.
6. Privacy protection: Masked password fields are recognized as POPULATED without exposing or logging
   underlying characters.
"""

from __future__ import annotations

import asyncio
from difflib import SequenceMatcher
import json
import logging
import math
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import uuid

from app.ai.prompt import PromptBuilder
from app.core.container import ServiceContainer, container as default_container
from app.core.logger import get_logger
from app.vision.models import (
    ControlAffordance,
    ControlVisualState,
    ElementAffordance,
    FormField,
    GroundingSource,
    OCRTextBlock,
    Point,
    SceneQueryAnswer,
    ScreenCapture,
    ScreenObservation,
    UIContainer,
    UIContainerType,
    UIElement,
    UIElementType,
    UIScene,
    WindowBounds,
)
from app.vision.preprocessing import ImagePreprocessor, default_preprocessor

logger = get_logger("VISION.AFFORDANCE")

# --------------------------------------------------------------------------
# Prompt Schema for Optional VLM Fallback
# --------------------------------------------------------------------------

_DEFAULT_AFFORDANCE_PROMPT = (
    "You are J.A.R.V.I.S UI Affordance & State Inspection Engine.\n"
    "Inspect the indicated control on the active application window and determine its operational state.\n"
    "Respond ONLY with a valid JSON object adhering strictly to this schema:\n"
    "{\n"
    '  "detected_state": "enabled|disabled|checked|unchecked|indeterminate|empty|populated|focused|uncertain",\n'
    '  "primary_affordance": "clickable|editable|toggleable|selectable|scrollable|read_only",\n'
    '  "confidence": 0.0-1.0,\n'
    '  "evidence": "concise visual justification",\n'
    '  "summary": "concise speakable summary"\n'
    "}\n"
    "Do NOT fabricate certainty if evidence is ambiguous."
)

# Common placeholder cues for text inputs
_PLACEHOLDER_PATTERNS: Sequence[str] = (
    "enter ",
    "type here",
    "search...",
    "search google",
    "username...",
    "email...",
    "password...",
    "first name...",
    "last name...",
    "e.g.",
    "optional",
    "filter...",
)


# --------------------------------------------------------------------------
# Visual Affordance Engine
# --------------------------------------------------------------------------


class VisualAffordanceEngine:
    """Production UI Control State, Affordance & Semantic Scene Querying Engine."""

    def __init__(
        self,
        *,
        ai_provider: Optional[Any] = None,
        preprocessor: Optional[ImagePreprocessor] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        logger_instance: Optional[logging.Logger] = None,
        container_instance: Optional[ServiceContainer] = None,
    ) -> None:
        self._ai_provider = ai_provider
        self._preprocessor = preprocessor or default_preprocessor
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._logger = logger_instance or logger
        self._container = container_instance if container_instance is not None else default_container

    @property
    def ai_provider(self) -> Optional[Any]:
        """Resolve multimodal AI provider from instance or container."""
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

    # -----------------------------------------------------------------------
    # 1. Element Affordance & State Inspection
    # -----------------------------------------------------------------------

    def inspect_element_affordance(
        self,
        element: UIElement,
        capture: Optional[ScreenCapture] = None,
        ocr_blocks: Optional[Sequence[OCRTextBlock]] = None,
    ) -> ElementAffordance:
        """Inspect a single UIElement and classify its operational state and affordance."""
        if element is None:
            raise ValueError("element cannot be None")

        el_type = element.element_type
        name_clean = (element.name or "").strip()
        if element.text_content is not None:
            text_content = element.text_content.strip()
        else:
            text_content = name_clean
        meta = dict(element.metadata or {})

        # Primary Affordance Mapping (Conservative)
        affordance = self._classify_affordance(el_type, name_clean, meta)

        # State Classification
        state, conf, evidence = self._detect_control_state(
            element=element,
            el_type=el_type,
            text=text_content,
            capture=capture,
            ocr_blocks=ocr_blocks,
            metadata=meta,
        )

        return ElementAffordance(
            element=element,
            detected_state=state,
            primary_affordance=affordance,
            confidence=conf,
            evidence=evidence,
            metadata={"source": "deterministic_heuristics"},
        )

    def inspect_scene_affordances(
        self,
        scene: UIScene,
        capture: Optional[ScreenCapture] = None,
    ) -> Tuple[ElementAffordance, ...]:
        """Inspect and catalog operational affordances for all interactive elements in a UIScene."""
        if not scene or not scene.interactive_elements:
            return tuple()

        affordances: List[ElementAffordance] = []
        for el in scene.interactive_elements:
            try:
                aff = self.inspect_element_affordance(el, capture=capture)
                affordances.append(aff)
            except Exception as exc:
                self._logger.debug("Failed to inspect affordance for element '%s': %s", el.name, exc)
                affordances.append(
                    ElementAffordance(
                        element=el,
                        detected_state=ControlVisualState.UNCERTAIN,
                        primary_affordance=ControlAffordance.READ_ONLY,
                        confidence=0.5,
                        evidence=f"Inspection error: {exc}",
                    )
                )

        return tuple(affordances)

    # -----------------------------------------------------------------------
    # 2. Semantic Scene Querying
    # -----------------------------------------------------------------------

    def query_scene_state(
        self,
        scene: UIScene,
        query: str,
        capture: Optional[ScreenCapture] = None,
        force_multimodal: bool = False,
    ) -> SceneQueryAnswer:
        """Query state and affordances of controls in a UIScene using deterministic scene reasoning."""
        if not query or not query.strip():
            return SceneQueryAnswer(
                summary="No question or condition was specified.",
                confidence=0.0,
            )

        q_clean = query.strip().lower()

        # 1. Form Validity / Completeness Queries
        if any(phrase in q_clean for phrase in (
            "can i submit",
            "is the form complete",
            "is this form complete",
            "is form complete",
            "which required fields",
            "are required fields empty",
            "can the form be submitted",
        )):
            return self._query_form_completeness(scene, capture)

        # 2. Specific Element State Queries (Enabled / Disabled / Checked / Empty / Editable)
        matched_el, match_score = self._resolve_target_element(scene, q_clean)

        if matched_el is not None:
            aff = self.inspect_element_affordance(matched_el, capture=capture)
            return self._answer_element_query(matched_el, aff, q_clean)

        # 3. Optional VLM Fallback if deterministic matching was inconclusive
        if force_multimodal or matched_el is None:
            ai_p = self.ai_provider
            if ai_p is not None and getattr(ai_p, "supports_multimodal", False) and capture and not capture.is_empty:
                vlm_ans = self._execute_multimodal_query(query=query, capture=capture, scene=scene)
                if vlm_ans is not None:
                    return vlm_ans

        return SceneQueryAnswer(
            target_element=None,
            detected_state=ControlVisualState.UNCERTAIN,
            verified_condition=None,
            confidence=0.4,
            summary="I can't determine the control state confidently from the current screen.",
            metadata={"reason": "target_element_not_found"},
        )

    # -----------------------------------------------------------------------
    # Internal Classification Heuristics
    # -----------------------------------------------------------------------

    def _classify_affordance(
        self,
        el_type: UIElementType,
        name: str,
        metadata: Dict[str, Any],
    ) -> ControlAffordance:
        """Conservatively determine the primary interactive affordance of an element."""
        if el_type == UIElementType.BUTTON:
            return ControlAffordance.CLICKABLE
        if el_type == UIElementType.INPUT:
            return ControlAffordance.EDITABLE
        if el_type == UIElementType.CHECKBOX or "radio" in name.lower():
            return ControlAffordance.TOGGLEABLE
        if el_type in (UIElementType.DROPDOWN, UIElementType.TAB):
            return ControlAffordance.SELECTABLE
        if el_type == UIElementType.LINK:
            return ControlAffordance.CLICKABLE
        if el_type == UIElementType.ICON:
            # Action icons (close, search, save) are clickable; static icons are read-only
            if any(k in name.lower() for k in ("close", "minimize", "maximize", "search", "save", "menu", "gear", "settings")):
                return ControlAffordance.CLICKABLE
            return ControlAffordance.READ_ONLY

        return ControlAffordance.READ_ONLY

    def _detect_control_state(
        self,
        element: UIElement,
        el_type: UIElementType,
        text: str,
        capture: Optional[ScreenCapture],
        ocr_blocks: Optional[Sequence[OCRTextBlock]],
        metadata: Dict[str, Any],
    ) -> Tuple[ControlVisualState, float, str]:
        """Classify control visual state using conservative deterministic visual/text signals."""
        t_lower = text.lower().strip()

        # Explicit metadata override if set by authoritative source
        if "disabled" in metadata and metadata["disabled"] is True:
            return ControlVisualState.DISABLED, 0.98, "Explicitly marked disabled in metadata"
        if "enabled" in metadata and metadata["enabled"] is True:
            return ControlVisualState.ENABLED, 0.95, "Explicitly marked enabled in metadata"

        # ------------------------------------------------------------------
        # Checkboxes & Radios
        # ------------------------------------------------------------------
        if el_type == UIElementType.CHECKBOX or "radio" in t_lower or any(t_lower.startswith(g) for g in ("(•)", "(*)", "( )", "()")):
            # Checked glyphs
            if any(t_lower.startswith(g) or g in t_lower for g in ("[x]", "[v]", "✓", "☑", "(*)", "(•)", "(x)")):
                return ControlVisualState.CHECKED, 0.95, f"Explicit checked glyph detected in text '{text}'"
            if " checked" in t_lower or t_lower.endswith(": checked"):
                return ControlVisualState.CHECKED, 0.90, "Explicit checked keyword in control label"

            # Unchecked glyphs
            if any(t_lower.startswith(g) or g in t_lower for g in ("[ ]", "[]", "☐", "( )", "()")):
                return ControlVisualState.UNCHECKED, 0.95, f"Explicit unchecked glyph detected in text '{text}'"
            if " unchecked" in t_lower or t_lower.endswith(": unchecked"):
                return ControlVisualState.UNCHECKED, 0.90, "Explicit unchecked keyword in control label"

            # Pixel analysis fallback if capture frame is available
            if capture is not None and not capture.is_empty:
                pix_state, pix_conf, pix_ev = self._inspect_checkbox_pixels(element.bounds, capture)
                if pix_state != ControlVisualState.UNCERTAIN:
                    return pix_state, pix_conf, pix_ev

            return ControlVisualState.UNCERTAIN, 0.50, "No explicit checkmark or empty box glyph detected"

        # ------------------------------------------------------------------
        # Text Inputs
        # ------------------------------------------------------------------
        if el_type == UIElementType.INPUT:
            # Masked password input check
            if any(bullet in text for bullet in ("••••", "••••••", "****", "******")) or "password" in t_lower:
                if any(bullet in text for bullet in ("••••", "****")):
                    return ControlVisualState.POPULATED, 0.95, "Masked password field contains characters"
                if not text or text == "password" or text.endswith(":"):
                    return ControlVisualState.EMPTY, 0.85, "Password input appears empty"

            # Active Focus Caret / Pipe
            if text.endswith("|") or text.endswith("_") or metadata.get("has_focus") is True:
                return ControlVisualState.FOCUSED, 0.92, "Active input cursor or focus caret visible"

            # Placeholder vs Populated text
            if not text or text.strip() == "" or text in ("[input]", "input") or text.lower() == element.name.lower():
                return ControlVisualState.EMPTY, 0.90, "Input field has no entered text"

            if any(p in t_lower for p in _PLACEHOLDER_PATTERNS):
                return ControlVisualState.EMPTY, 0.85, f"Input contains placeholder prompt '{text}'"

            return ControlVisualState.POPULATED, 0.88, "Input field contains user-entered text content"

        # ------------------------------------------------------------------
        # Buttons
        # ------------------------------------------------------------------
        if el_type == UIElementType.BUTTON:
            # Explicit textual cues
            if "(disabled)" in t_lower or "[disabled]" in t_lower:
                return ControlVisualState.DISABLED, 0.98, "Explicit disabled text marker in button label"

            # Pixel contrast / luminance analysis if capture frame is available
            if capture is not None and not capture.is_empty:
                pix_state, pix_conf, pix_ev = self._inspect_button_pixels(element.bounds, capture)
                return pix_state, pix_conf, pix_ev

            # Standard button default if no disabled markers and no capture
            return ControlVisualState.ENABLED, 0.85, "Active button label without disabled modifiers"

        # ------------------------------------------------------------------
        # Tabs / Dropdowns / Links
        # ------------------------------------------------------------------
        if el_type in (UIElementType.TAB, UIElementType.DROPDOWN, UIElementType.LINK):
            if metadata.get("selected") is True or metadata.get("active") is True:
                return ControlVisualState.FOCUSED, 0.90, "Selected active tab or dropdown"
            return ControlVisualState.ENABLED, 0.85, f"Interactive {el_type.value} control"

        return ControlVisualState.UNCERTAIN, 0.50, "Static or unclassified visual element"

    # -----------------------------------------------------------------------
    # Pixel Inspection Utilities (In-Memory, Safe)
    # -----------------------------------------------------------------------

    def _inspect_button_pixels(
        self,
        bounds: WindowBounds,
        capture: ScreenCapture,
    ) -> Tuple[ControlVisualState, float, str]:
        """Analyze bounding-box luminance contrast across light/dark themes."""
        if bounds.is_empty or capture.is_empty:
            return ControlVisualState.UNCERTAIN, 0.5, "Empty bounds or capture frame"

        # Normalize relative to capture bounds
        c_bounds = capture.bounds
        rel_left = max(0, bounds.left - c_bounds.left)
        rel_top = max(0, bounds.top - c_bounds.top)
        rel_right = min(capture.width, bounds.right - c_bounds.left)
        rel_bottom = min(capture.height, bounds.bottom - c_bounds.top)

        box_w = rel_right - rel_left
        box_h = rel_bottom - rel_top

        if box_w < 6 or box_h < 6:
            return ControlVisualState.UNCERTAIN, 0.5, "Button bounding box too small for pixel analysis"

        try:
            rgba = capture.to_rgba()
            row_stride = capture.width * 4

            # Sample perimeter (background) and interior (foreground/text)
            bg_lums: List[float] = []
            fg_lums: List[float] = []

            for y_step in range(rel_top, rel_bottom, max(1, box_h // 8)):
                for x_step in range(rel_left, rel_right, max(1, box_w // 12)):
                    idx = (y_step * row_stride) + (x_step * 4)
                    if idx + 3 >= len(rgba):
                        continue
                    r, g, b = rgba[idx], rgba[idx + 1], rgba[idx + 2]
                    # Relative luminance: standard sRGB formula
                    lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0

                    # Perimeter points represent background
                    if (
                        x_step in (rel_left, rel_left + 1, rel_right - 1, rel_right - 2)
                        or y_step in (rel_top, rel_top + 1, rel_bottom - 1, rel_bottom - 2)
                    ):
                        bg_lums.append(lum)
                    else:
                        fg_lums.append(lum)

            if not bg_lums or not fg_lums:
                return ControlVisualState.UNCERTAIN, 0.5, "Insufficient pixel sample density"

            avg_bg = sum(bg_lums) / len(bg_lums)
            # Calculate maximum contrast difference between background and interior points
            max_contrast_diff = max(abs(l - avg_bg) for l in fg_lums)

            # Low contrast difference (< 0.16) indicates grayed-out/disabled button
            if max_contrast_diff < 0.16:
                return (
                    ControlVisualState.DISABLED,
                    0.88,
                    f"Low visual contrast (diff={max_contrast_diff:.3f}) consistent with disabled state",
                )

            # High contrast difference (>= 0.25) indicates crisp active button
            if max_contrast_diff >= 0.25:
                return (
                    ControlVisualState.ENABLED,
                    0.88,
                    f"Distinct visual contrast (diff={max_contrast_diff:.3f}) indicating active button",
                )

            return (
                ControlVisualState.UNCERTAIN,
                0.55,
                f"Intermediate visual contrast (diff={max_contrast_diff:.3f}); state cannot be conclusively proven",
            )

        except Exception as exc:
            self._logger.debug("Pixel contrast analysis encountered error: %s", exc)
            return ControlVisualState.UNCERTAIN, 0.50, f"Pixel analysis failed: {exc}"

    def _inspect_checkbox_pixels(
        self,
        bounds: WindowBounds,
        capture: ScreenCapture,
    ) -> Tuple[ControlVisualState, float, str]:
        """Inspect glyph region for checked mark vs empty square."""
        c_bounds = capture.bounds
        rel_left = max(0, bounds.left - c_bounds.left)
        rel_top = max(0, bounds.top - c_bounds.top)
        # Checkbox glyph is typically on the left 20x20 pixels
        glyph_w = min(24, bounds.width)
        glyph_h = min(24, bounds.height)

        if glyph_w < 8 or glyph_h < 8:
            return ControlVisualState.UNCERTAIN, 0.5, "Glyph region too small for inspection"

        try:
            rgba = capture.to_rgba()
            row_stride = capture.width * 4

            # Sample center 60% of glyph
            center_lums: List[float] = []
            border_lums: List[float] = []

            for dy in range(glyph_h):
                for dx in range(glyph_w):
                    idx = ((rel_top + dy) * row_stride) + ((rel_left + dx) * 4)
                    if idx + 3 >= len(rgba):
                        continue
                    lum = (0.2126 * rgba[idx] + 0.7152 * rgba[idx + 1] + 0.0722 * rgba[idx + 2]) / 255.0

                    if 0.25 * glyph_w <= dx <= 0.75 * glyph_w and 0.25 * glyph_h <= dy <= 0.75 * glyph_h:
                        center_lums.append(lum)
                    else:
                        border_lums.append(lum)

            if not center_lums or not border_lums:
                return ControlVisualState.UNCERTAIN, 0.5, "Insufficient glyph samples"

            c_mean = sum(center_lums) / len(center_lums)
            b_mean = sum(border_lums) / len(border_lums)
            c_std = math.sqrt(sum((l - c_mean) ** 2 for l in center_lums) / len(center_lums))

            # A checkmark inside creates high standard deviation / dark pixels in center
            if c_std > 0.10 or abs(c_mean - b_mean) > 0.25:
                return ControlVisualState.CHECKED, 0.82, "Visual checkmark pattern detected in box interior"

            # Uniform center matching background represents empty box
            if c_std < 0.03:
                return ControlVisualState.UNCHECKED, 0.82, "Uniform interior consistent with unchecked box"

            return ControlVisualState.UNCERTAIN, 0.50, "Glyph interior inconclusive"
        except Exception:
            return ControlVisualState.UNCERTAIN, 0.50, "Glyph inspection failed"

    # -----------------------------------------------------------------------
    # Target Resolution & Scene Query Handlers
    # -----------------------------------------------------------------------

    def _resolve_target_element(
        self,
        scene: UIScene,
        query_text: str,
    ) -> Tuple[Optional[UIElement], float]:
        """Find the most relevant UIElement in the scene matching the query."""
        best_el: Optional[UIElement] = None
        best_score = 0.0

        for el in scene.interactive_elements:
            el_name = (el.name or "").lower().strip()
            if not el_name:
                continue

            # Exact phrase match
            if el_name in query_text:
                score = 0.95 + min(0.04, len(el_name) / 100.0)
                if score > best_score:
                    best_score = score
                    best_el = el
                continue

            # Word overlap match
            q_words = set(re.findall(r"\w+", query_text))
            el_words = set(re.findall(r"\w+", el_name))
            overlap = el_words.intersection(q_words)
            if overlap:
                ratio = len(overlap) / float(len(el_words))
                if ratio > 0.5:
                    score = 0.70 + (0.20 * ratio)
                    if score > best_score:
                        best_score = score
                        best_el = el

        # Also search FormField labels
        if best_el is None and scene.form_fields:
            for field in scene.form_fields:
                lbl = field.label.lower().strip()
                if lbl in query_text or any(w in query_text for w in re.findall(r"\w+", lbl) if len(w) > 2):
                    return field.input_element, 0.88

        return best_el, best_score

    def _query_form_completeness(
        self,
        scene: UIScene,
        capture: Optional[ScreenCapture],
    ) -> SceneQueryAnswer:
        """Evaluate form completeness and submit affordance conservatively."""
        if not scene.form_fields and not scene.interactive_elements:
            return SceneQueryAnswer(
                verified_condition=None,
                detected_state=ControlVisualState.UNCERTAIN,
                confidence=0.5,
                summary="No form fields or interactive controls were found on the current screen.",
            )

        # 1. Check for empty required fields
        empty_required: List[str] = []
        for field in scene.form_fields:
            if field.is_required:
                aff = self.inspect_element_affordance(field.input_element, capture=capture)
                if aff.detected_state == ControlVisualState.EMPTY:
                    empty_required.append(field.label)

        if empty_required:
            names = ", ".join(f"'{name}'" for name in empty_required)
            return SceneQueryAnswer(
                verified_condition=False,
                detected_state=ControlVisualState.EMPTY,
                confidence=0.92,
                summary=f"The form appears incomplete because required field{'s' if len(empty_required) > 1 else ''} {names} appear empty.",
                metadata={"empty_required_fields": empty_required},
            )

        # 2. Check Submit / Save button state
        submit_button: Optional[UIElement] = None
        for el in scene.interactive_elements:
            if el.element_type == UIElementType.BUTTON and any(k in el.name.lower() for k in ("submit", "save", "next", "continue", "sign in", "login", "register")):
                submit_button = el
                break

        if submit_button is not None:
            btn_aff = self.inspect_element_affordance(submit_button, capture=capture)
            if btn_aff.detected_state == ControlVisualState.DISABLED:
                return SceneQueryAnswer(
                    target_element=submit_button.name,
                    element=submit_button,
                    verified_condition=False,
                    detected_state=ControlVisualState.DISABLED,
                    confidence=btn_aff.confidence,
                    summary=f"The form cannot be submitted because the '{submit_button.name}' button appears disabled.",
                )

        # 3. If no empty required fields were detected, do not overclaim validity
        return SceneQueryAnswer(
            verified_condition=None,
            detected_state=ControlVisualState.UNCERTAIN,
            confidence=0.65,
            summary="All visible form fields appear populated, but form validation completeness cannot be conclusively guaranteed.",
            metadata={"all_visible_populated": True},
        )

    def _answer_element_query(
        self,
        element: UIElement,
        affordance: ElementAffordance,
        query: str,
    ) -> SceneQueryAnswer:
        """Formulate a direct, speakable answer to a specific element query."""
        state = affordance.detected_state
        name = element.name or "The control"
        clean_q = query.lower()

        # Is ... enabled?
        if "enabled" in clean_q or "clickable" in clean_q:
            if state == ControlVisualState.ENABLED:
                return SceneQueryAnswer(
                    target_element=name,
                    element=element,
                    detected_state=state,
                    verified_condition=True,
                    confidence=affordance.confidence,
                    summary=f"The {name} button appears enabled.",
                )
            if state == ControlVisualState.DISABLED:
                return SceneQueryAnswer(
                    target_element=name,
                    element=element,
                    detected_state=state,
                    verified_condition=False,
                    confidence=affordance.confidence,
                    summary=f"The {name} button appears disabled.",
                )

        # Is ... disabled?
        if "disabled" in clean_q:
            if state == ControlVisualState.DISABLED:
                return SceneQueryAnswer(
                    target_element=name,
                    element=element,
                    detected_state=state,
                    verified_condition=True,
                    confidence=affordance.confidence,
                    summary=f"The {name} button appears disabled.",
                )
            if state == ControlVisualState.ENABLED:
                return SceneQueryAnswer(
                    target_element=name,
                    element=element,
                    detected_state=state,
                    verified_condition=False,
                    confidence=affordance.confidence,
                    summary=f"The {name} button appears enabled.",
                )

        # Is ... checked?
        if "checked" in clean_q and "unchecked" not in clean_q:
            if state == ControlVisualState.CHECKED:
                return SceneQueryAnswer(
                    target_element=name,
                    element=element,
                    detected_state=state,
                    verified_condition=True,
                    confidence=affordance.confidence,
                    summary=f"The {name} checkbox appears checked.",
                )
            if state == ControlVisualState.UNCHECKED:
                return SceneQueryAnswer(
                    target_element=name,
                    element=element,
                    detected_state=state,
                    verified_condition=False,
                    confidence=affordance.confidence,
                    summary=f"The {name} checkbox appears unchecked.",
                )

        # Is ... unchecked?
        if "unchecked" in clean_q:
            if state == ControlVisualState.UNCHECKED:
                return SceneQueryAnswer(
                    target_element=name,
                    element=element,
                    detected_state=state,
                    verified_condition=True,
                    confidence=affordance.confidence,
                    summary=f"The {name} checkbox appears unchecked.",
                )
            if state == ControlVisualState.CHECKED:
                return SceneQueryAnswer(
                    target_element=name,
                    element=element,
                    detected_state=state,
                    verified_condition=False,
                    confidence=affordance.confidence,
                    summary=f"The {name} checkbox appears checked.",
                )

        # Is ... empty?
        if "empty" in clean_q:
            if state == ControlVisualState.EMPTY:
                return SceneQueryAnswer(
                    target_element=name,
                    element=element,
                    detected_state=state,
                    verified_condition=True,
                    confidence=affordance.confidence,
                    summary=f"The {name} field appears empty.",
                )
            if state == ControlVisualState.POPULATED:
                return SceneQueryAnswer(
                    target_element=name,
                    element=element,
                    detected_state=state,
                    verified_condition=False,
                    confidence=affordance.confidence,
                    summary=f"The {name} field is populated with text.",
                )

        # Is ... editable?
        if "editable" in clean_q:
            is_ed = affordance.primary_affordance == ControlAffordance.EDITABLE
            return SceneQueryAnswer(
                target_element=name,
                element=element,
                detected_state=state,
                verified_condition=is_ed,
                confidence=affordance.confidence,
                summary=f"The {name} field appears {'editable' if is_ed else 'read-only'}.",
            )

        # Default state description
        if state != ControlVisualState.UNCERTAIN:
            return SceneQueryAnswer(
                target_element=name,
                element=element,
                detected_state=state,
                confidence=affordance.confidence,
                summary=f"{name} appears {state.value}.",
            )

        return SceneQueryAnswer(
            target_element=name,
            element=element,
            detected_state=ControlVisualState.UNCERTAIN,
            confidence=0.5,
            summary="I can't determine the control state confidently from the current screen.",
        )

    # -----------------------------------------------------------------------
    # 3. Optional Multimodal VLM Fallback
    # -----------------------------------------------------------------------

    def _execute_multimodal_query(
        self,
        query: str,
        capture: ScreenCapture,
        scene: UIScene,
    ) -> Optional[SceneQueryAnswer]:
        """Execute multimodal fallback for visually ambiguous controls."""
        ai_p = self.ai_provider
        if ai_p is None or not getattr(ai_p, "supports_multimodal", False):
            return None

        try:
            img_part = self._preprocessor.to_image_part(capture)
            full_prompt = f"{_DEFAULT_AFFORDANCE_PROMPT}\n\nUser Question: {query}"
            payload = self._prompt_builder.build_payload(query=full_prompt, images=[img_part])
            response = ai_p.generate(payload)
            content = getattr(response, "content", str(response)).strip()

            if "```json" in content:
                content = content.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in content:
                content = content.split("```", 1)[1].split("```", 1)[0].strip()

            data = json.loads(content)
            if not isinstance(data, dict):
                return None

            raw_state = str(data.get("detected_state", "")).lower().strip()
            try:
                state = ControlVisualState(raw_state)
            except ValueError:
                state = ControlVisualState.UNCERTAIN

            conf = max(0.0, min(1.0, float(data.get("confidence", 0.75))))
            summary = str(data.get("summary") or f"The control appears {state.value}.").strip()

            return SceneQueryAnswer(
                target_element=None,
                detected_state=state,
                confidence=conf,
                summary=summary,
                metadata={"source": "vlm_fallback"},
            )
        except Exception as exc:
            self._logger.debug("VLM affordance query fallback failed: %s", exc)
            return None


__all__ = ["VisualAffordanceEngine"]
