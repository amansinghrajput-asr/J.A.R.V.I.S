"""Visual Delta and Change Detection Engine for J.A.R.V.I.S (Phase 27.10).

Provides deterministic, privacy-safe comparative analysis between two visual
screen observations (T0 = before, T1 = after) to answer questions such as:
- "What changed on my screen?"
- "What just changed?"
- "Did the button appear?"
- "Did the popup disappear?"
- "Is the error still there?"
- "Did the UI element move?"

Safety and Architectural Invariants Enforced:
1. Purely read-only visual perception and change analysis: ZERO mouse/keyboard automation,
   process mutation, or desktop state manipulation.
2. Zero persistent raw screenshots or pixel bytes in returned objects, logs, or telemetry.
3. Layered conservative change analysis:
   - Layer 1: Window Context Diff (HWND, process, title, coordinate translation).
   - Layer 2: Deterministic OCR + Grounded UIElement Diff (Fast CPU path).
   - Layer 3: Multimodal Semantic Diff fallback for non-textual/graphical changes.
4. Ambiguous element matches resolve to UNCERTAIN rather than guessing or hallucinating identity.
5. Window desktop translation does NOT produce false child-element move events.
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

from app.ai.models import ImagePart, UnsupportedModalityError
from app.ai.prompt import PromptBuilder
from app.core.container import ServiceContainer, container as default_container
from app.core.logger import get_logger
from app.vision.models import (
    GroundingSource,
    OCRResult,
    OCRTextBlock,
    Point,
    ScreenCapture,
    ScreenObservation,
    UIElement,
    UIElementChange,
    UIElementType,
    VisualDeltaResult,
    VisualDeltaType,
    WindowBounds,
)
from app.vision.ocr import OCRProvider
from app.vision.preprocessing import ImagePreprocessor, default_preprocessor

logger = get_logger("VISION.DELTA")

# --------------------------------------------------------------------------
# Named Matching & Comparison Thresholds
# --------------------------------------------------------------------------

MIN_ELEMENT_MATCH_SCORE: float = 0.45
ELEMENT_MATCH_AMBIGUITY_DELTA: float = 0.05
MOVE_DISTANCE_THRESHOLD: float = 15.0
RESIZE_RATIO_THRESHOLD: float = 0.15
TEXT_SIMILARITY_THRESHOLD: float = 0.70
IOU_MATCH_THRESHOLD: float = 0.50

_DEFAULT_DELTA_PROMPT = (
    "You are J.A.R.V.I.S Visual Change Detection Engine.\n"
    "Inspect the before and after screen states and describe the visual change.\n"
    "Target filter: '{target}'\n"
    "Respond ONLY with a valid JSON object adhering strictly to this schema:\n"
    "{{\n"
    '  "changed": true,\n'
    '  "change_type": "element_appeared|element_disappeared|element_moved|element_resized|text_changed|region_changed|no_meaningful_change|uncertain",\n'
    '  "confidence": 0.0-1.0,\n'
    '  "explanation": "concise explanation of what changed",\n'
    '  "box_2d": [ymin, xmin, ymax, xmax]\n'
    "}}\n"
    "Coordinates in box_2d MUST be normalized integers in the range [0, 1000] representing [ymin, xmin, ymax, xmax] or null.\n"
    'If no meaningful visual change is visible, respond with:\n'
    '{{"changed": false, "change_type": "no_meaningful_change", "confidence": 1.0, "explanation": "No meaningful changes were detected.", "box_2d": null}}'
)


# --------------------------------------------------------------------------
# Helper Geometric & Text Utility Functions
# --------------------------------------------------------------------------


def _compute_iou(b1: WindowBounds, b2: WindowBounds) -> float:
    """Compute spatial Intersection-over-Union between two WindowBounds."""
    if b1.is_empty or b2.is_empty:
        return 0.0

    inter_left = max(b1.left, b2.left)
    inter_top = max(b1.top, b2.top)
    inter_right = min(b1.right, b2.right)
    inter_bottom = min(b1.bottom, b2.bottom)

    if inter_right <= inter_left or inter_bottom <= inter_top:
        return 0.0

    inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)
    union_area = b1.area + b2.area - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def _compute_text_similarity(s1: Optional[str], s2: Optional[str]) -> float:
    """Compute normalized text similarity ratio between two strings."""
    t1 = (s1 or "").strip().lower()
    t2 = (s2 or "").strip().lower()
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    if t1 == t2:
        return 1.0
    return SequenceMatcher(None, t1, t2).ratio()


def _normalize_bounds_relative_to_window(
    bounds: WindowBounds,
    window_bounds: Optional[WindowBounds],
) -> WindowBounds:
    """Convert screen/desktop bounds to window-relative coordinates."""
    if window_bounds is None or window_bounds.is_empty:
        return bounds
    return WindowBounds(
        left=bounds.left - window_bounds.left,
        top=bounds.top - window_bounds.top,
        right=bounds.right - window_bounds.left,
        bottom=bounds.bottom - window_bounds.top,
    )


# --------------------------------------------------------------------------
# Visual Delta Engine
# --------------------------------------------------------------------------


class VisualDeltaEngine:
    """Production Visual Delta & Change Detection Engine for J.A.R.V.I.S."""

    def __init__(
        self,
        *,
        ocr_provider: Optional[OCRProvider] = None,
        ai_provider: Optional[Any] = None,
        preprocessor: Optional[ImagePreprocessor] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        logger_instance: Optional[logging.Logger] = None,
        container_instance: Optional[ServiceContainer] = None,
    ) -> None:
        self._ocr_provider = ocr_provider
        self._ai_provider = ai_provider
        self._preprocessor = preprocessor or default_preprocessor
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._logger = logger_instance or logger
        self._container = container_instance if container_instance is not None else default_container

    @property
    def ocr_provider(self) -> Optional[OCRProvider]:
        """Resolve OCR provider from instance or container."""
        if self._ocr_provider is not None:
            return self._ocr_provider
        if self._container is not None and self._container.exists("ocr_provider"):
            return self._container.resolve("ocr_provider")
        return None

    @property
    def ai_provider(self) -> Optional[Any]:
        """Resolve AI provider from instance or container."""
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
    # Main Comparison Interface
    # -----------------------------------------------------------------------

    def compare_observations(
        self,
        before: ScreenObservation,
        after: ScreenObservation,
        target_filter: Optional[str] = None,
        force_multimodal: bool = False,
    ) -> VisualDeltaResult:
        """Compare two visual observations (T0 and T1) and return structured VisualDeltaResult."""
        start_time = time.time()
        delta_id = str(uuid.uuid4())

        # Validation: Verify observations
        if not before or not after:
            return VisualDeltaResult(
                delta_id=delta_id,
                before_observation_id=before.observation_id if before else "",
                after_observation_id=after.observation_id if after else "",
                primary_change_type=VisualDeltaType.UNCERTAIN,
                confidence=0.0,
                explanation="Cannot compare invalid or missing screen observations.",
            )

        time_delta = max(0.0, float(after.timestamp - before.timestamp))

        # ------------------------------------------------------------------
        # Layer 1: Window Context Comparison
        # ------------------------------------------------------------------
        win_before_title = (before.metadata.get("window_title") or "").strip()
        win_after_title = (after.metadata.get("window_title") or "").strip()
        proc_before = (before.metadata.get("process_name") or "").strip().lower()
        proc_after = (after.metadata.get("process_name") or "").strip().lower()

        hwnd_before = before.metadata.get("hwnd")
        hwnd_after = after.metadata.get("hwnd")

        # Detect total active window/process change
        is_window_change = False
        if hwnd_before and hwnd_after and hwnd_before != hwnd_after:
            # Different window handle
            if proc_before != proc_after or (win_before_title and win_after_title and win_before_title != win_after_title):
                is_window_change = True
        elif proc_before and proc_after and proc_before != proc_after:
            is_window_change = True

        if is_window_change:
            desc = (
                f"Active window switched from '{win_before_title or proc_before}' "
                f"to '{win_after_title or proc_after}'."
            )
            return VisualDeltaResult(
                delta_id=delta_id,
                before_observation_id=before.observation_id,
                after_observation_id=after.observation_id,
                time_delta_seconds=time_delta,
                primary_change_type=VisualDeltaType.WINDOW_CHANGED,
                window_changed=True,
                meaningful_change_detected=True,
                confidence=1.0,
                explanation=desc,
                metadata={
                    "window_before": win_before_title,
                    "window_after": win_after_title,
                    "process_before": proc_before,
                    "process_after": proc_after,
                },
            )

        # ------------------------------------------------------------------
        # Layer 2: Deterministic OCR and Grounded UIElement Diff
        # ------------------------------------------------------------------
        win_bounds_before = before.bounds
        win_bounds_after = after.bounds

        # Calculate window displacement if window moved
        win_dx = (win_bounds_after.left - win_bounds_before.left) if (win_bounds_before and win_bounds_after) else 0
        win_dy = (win_bounds_after.top - win_bounds_before.top) if (win_bounds_before and win_bounds_after) else 0

        # Extract OCR text blocks for both observations
        ocr_blocks_before = self._extract_ocr_blocks(before)
        ocr_blocks_after = self._extract_ocr_blocks(after)

        # Compute text diffs
        added_texts, removed_texts, modified_texts = self._compute_ocr_text_diff(
            ocr_blocks_before,
            ocr_blocks_after,
            win_bounds_before,
            win_bounds_after,
        )

        # Extract/Build UIElements for both observations
        elements_before = self._extract_ui_elements(before, ocr_blocks_before)
        elements_after = self._extract_ui_elements(after, ocr_blocks_after)

        # Match UIElements across T0 and T1
        element_changes, is_ambiguous = self._match_and_classify_elements(
            elements_before,
            elements_after,
            win_bounds_before,
            win_bounds_after,
        )

        # Apply target filtering if requested
        if target_filter and target_filter.strip():
            tf_clean = target_filter.strip().lower()
            element_changes = [
                ec for ec in element_changes
                if (ec.before_element and tf_clean in ec.before_element.name.lower())
                or (ec.after_element and tf_clean in ec.after_element.name.lower())
                or (ec.before_element and ec.before_element.text_content and tf_clean in ec.before_element.text_content.lower())
                or (ec.after_element and ec.after_element.text_content and tf_clean in ec.after_element.text_content.lower())
            ]

        # ------------------------------------------------------------------
        # Layer 3: Multimodal Semantic Fallback (if required)
        # ------------------------------------------------------------------
        multimodal_result: Optional[Dict[str, Any]] = None
        meaningful_detected = any(
            c.change_type not in (VisualDeltaType.NO_MEANINGFUL_CHANGE, VisualDeltaType.UNCERTAIN)
            for c in element_changes
        ) or bool(added_texts or removed_texts or modified_texts)

        if (force_multimodal or (not meaningful_detected and target_filter and not is_ambiguous)):
            ai_p = self.ai_provider
            if ai_p is not None and getattr(ai_p, "supports_multimodal", False):
                try:
                    multimodal_result = self._execute_multimodal_diff(
                        before=before,
                        after=after,
                        target=target_filter or "screen content",
                    )
                except Exception as exc:
                    self._logger.warning("Multimodal delta fallback failed: %s", exc)

        # Determine primary change classification and explanation
        primary_type, explanation, conf = self._synthesize_verdict(
            element_changes=element_changes,
            added_texts=added_texts,
            removed_texts=removed_texts,
            modified_texts=modified_texts,
            is_ambiguous=is_ambiguous,
            multimodal_result=multimodal_result,
            target_filter=target_filter,
        )

        meaningful_change = (primary_type not in (VisualDeltaType.NO_MEANINGFUL_CHANGE, VisualDeltaType.UNCERTAIN))

        return VisualDeltaResult(
            delta_id=delta_id,
            before_observation_id=before.observation_id,
            after_observation_id=after.observation_id,
            time_delta_seconds=time_delta,
            primary_change_type=primary_type,
            element_changes=tuple(element_changes),
            added_texts=tuple(added_texts),
            removed_texts=tuple(removed_texts),
            modified_texts=tuple(modified_texts),
            window_changed=False,
            meaningful_change_detected=meaningful_change,
            confidence=conf,
            explanation=explanation,
            metadata={
                "ocr_added_count": len(added_texts),
                "ocr_removed_count": len(removed_texts),
                "ocr_modified_count": len(modified_texts),
                "element_change_count": len(element_changes),
                "window_translation": {"dx": win_dx, "dy": win_dy},
                "used_multimodal": multimodal_result is not None,
            },
        )

    async def compare_observations_async(
        self,
        before: ScreenObservation,
        after: ScreenObservation,
        target_filter: Optional[str] = None,
        force_multimodal: bool = False,
    ) -> VisualDeltaResult:
        """Asynchronously compare observations on a worker thread."""
        return await asyncio.to_thread(
            self.compare_observations,
            before,
            after,
            target_filter,
            force_multimodal,
        )

    # -----------------------------------------------------------------------
    # OCR Extraction and Text Block Diffing
    # -----------------------------------------------------------------------

    def _extract_ocr_blocks(self, observation: ScreenObservation) -> List[OCRTextBlock]:
        """Extract OCR blocks from observation or run OCR provider."""
        # 1. Check existing metadata
        if "ocr_result" in observation.metadata:
            res = observation.metadata["ocr_result"]
            if isinstance(res, OCRResult):
                return list(res.blocks)
            if isinstance(res, dict) and "blocks" in res:
                return [
                    OCRTextBlock(
                        text=b.get("text", ""),
                        bounds=WindowBounds(
                            left=b.get("bounds", {}).get("left", 0),
                            top=b.get("bounds", {}).get("top", 0),
                            right=b.get("bounds", {}).get("right", 0),
                            bottom=b.get("bounds", {}).get("bottom", 0),
                        ),
                        confidence=float(b.get("confidence", 1.0)),
                    )
                    for b in res["blocks"]
                ]

        if "ocr_blocks" in observation.metadata:
            raw = observation.metadata["ocr_blocks"]
            if isinstance(raw, list):
                return [
                    b if isinstance(b, OCRTextBlock)
                    else OCRTextBlock(
                        text=str(b.get("text", "")),
                        bounds=WindowBounds(
                            left=b.get("bounds", {}).get("left", 0),
                            top=b.get("bounds", {}).get("top", 0),
                            right=b.get("bounds", {}).get("right", 0),
                            bottom=b.get("bounds", {}).get("bottom", 0),
                        ) if isinstance(b.get("bounds"), dict) else (b.bounds if hasattr(b, "bounds") else WindowBounds(0, 0, 0, 0)),
                        confidence=float(b.get("confidence", 1.0) if isinstance(b, dict) else getattr(b, "confidence", 1.0)),
                    )
                    for b in raw
                ]

        # 2. Run OCR provider if capture is available
        ocr_p = self.ocr_provider
        if ocr_p is not None and observation.capture is not None and not observation.capture.is_empty:
            try:
                ocr_res = ocr_p.extract_text(observation.capture)
                return list(ocr_res.blocks)
            except Exception as exc:
                self._logger.debug("OCR extraction failed during delta analysis: %s", exc)

        return []

    def _compute_ocr_text_diff(
        self,
        blocks_before: List[OCRTextBlock],
        blocks_after: List[OCRTextBlock],
        win_bounds_before: Optional[WindowBounds],
        win_bounds_after: Optional[WindowBounds],
    ) -> Tuple[List[str], List[str], List[str]]:
        """Compute added, removed, and modified text blocks across observations."""
        added: List[str] = []
        removed: List[str] = []
        modified: List[str] = []

        if not blocks_before and not blocks_after:
            return added, removed, modified

        # Normalize coordinates relative to window bounds
        norm_b_before = [
            (b, _normalize_bounds_relative_to_window(b.bounds, win_bounds_before))
            for b in blocks_before
        ]
        norm_b_after = [
            (b, _normalize_bounds_relative_to_window(b.bounds, win_bounds_after))
            for b in blocks_after
        ]

        matched_after_indices: Set[int] = set()
        matched_before_indices: Set[int] = set()

        # Step 1: Match blocks with high spatial and textual overlap
        for idx_b, (blk_b, rel_b) in enumerate(norm_b_before):
            best_idx_a = -1
            best_score = 0.0

            for idx_a, (blk_a, rel_a) in enumerate(norm_b_after):
                if idx_a in matched_after_indices:
                    continue

                sim = _compute_text_similarity(blk_b.text, blk_a.text)
                iou = _compute_iou(rel_b, rel_a)
                score = 0.6 * sim + 0.4 * iou

                if score > best_score:
                    best_score = score
                    best_idx_a = idx_a

            if best_idx_a >= 0 and best_score >= MIN_ELEMENT_MATCH_SCORE:
                blk_a, rel_a = norm_b_after[best_idx_a]
                matched_after_indices.add(best_idx_a)
                matched_before_indices.add(idx_b)

                sim = _compute_text_similarity(blk_b.text, blk_a.text)
                if sim < 0.95 and blk_b.text.strip().lower() != blk_a.text.strip().lower():
                    # Text modified in-place
                    modified.append(f"'{blk_b.text}' -> '{blk_a.text}'")

        # Step 2: Identify removed blocks
        for idx_b, (blk_b, _) in enumerate(norm_b_before):
            if idx_b not in matched_before_indices:
                t = blk_b.text.strip()
                if t:
                    removed.append(t)

        # Step 3: Identify added blocks
        for idx_a, (blk_a, _) in enumerate(norm_b_after):
            if idx_a not in matched_after_indices:
                t = blk_a.text.strip()
                if t:
                    added.append(t)

        return added, removed, modified

    # -----------------------------------------------------------------------
    # UI Element Extraction and Multi-Factor Matching
    # -----------------------------------------------------------------------

    def _extract_ui_elements(
        self,
        observation: ScreenObservation,
        ocr_blocks: List[OCRTextBlock],
    ) -> List[UIElement]:
        """Extract existing grounded UIElements or synthesize them from OCR blocks."""
        # 1. Existing UIElements in metadata
        if "elements" in observation.metadata:
            raw = observation.metadata["elements"]
            if isinstance(raw, list):
                res: List[UIElement] = []
                for e in raw:
                    if isinstance(e, UIElement):
                        res.append(e)
                    elif isinstance(e, dict):
                        b = e.get("bounds", {})
                        wb = WindowBounds(
                            left=b.get("left", 0),
                            top=b.get("top", 0),
                            right=b.get("right", 0),
                            bottom=b.get("bottom", 0),
                        )
                        c = e.get("center", {})
                        pt = Point(x=c.get("x", wb.center.x), y=c.get("y", wb.center.y))
                        etype_str = e.get("element_type", "unknown")
                        try:
                            etype = UIElementType(etype_str)
                        except Exception:
                            etype = UIElementType.UNKNOWN
                        res.append(
                            UIElement(
                                name=e.get("name", "element"),
                                element_type=etype,
                                bounds=wb,
                                center=pt,
                                confidence=float(e.get("confidence", 1.0)),
                                text_content=e.get("text_content"),
                                metadata=dict(e.get("metadata", {})),
                            )
                        )
                if res:
                    return res

        # 2. Synthesize UIElements from OCR blocks
        elements: List[UIElement] = []
        for blk in ocr_blocks:
            t = blk.text.strip()
            if not t:
                continue
            # Infer basic type
            t_lower = t.lower()
            if any(k in t_lower for k in ("ok", "cancel", "submit", "apply", "close", "save", "button", "yes", "no")):
                etype = UIElementType.BUTTON
            elif any(k in t_lower for k in ("error", "warning", "failed", "exception")):
                etype = UIElementType.TEXT
            elif any(k in t_lower for k in ("search", "username", "password", "input", "enter")):
                etype = UIElementType.INPUT
            else:
                etype = UIElementType.TEXT

            elements.append(
                UIElement(
                    name=t,
                    element_type=etype,
                    bounds=blk.bounds,
                    center=blk.bounds.center,
                    confidence=blk.confidence,
                    source=GroundingSource.OCR_EXACT,
                    text_content=t,
                )
            )
        return elements

    def _match_and_classify_elements(
        self,
        elements_before: List[UIElement],
        elements_after: List[UIElement],
        win_bounds_before: Optional[WindowBounds],
        win_bounds_after: Optional[WindowBounds],
    ) -> Tuple[List[UIElementChange], bool]:
        """Match UIElements across observations and classify changes."""
        changes: List[UIElementChange] = []
        is_ambiguous = False

        if not elements_before and not elements_after:
            return changes, False

        # Normalize element bounds to window-relative coordinates
        norm_eb = [
            (e, _normalize_bounds_relative_to_window(e.bounds, win_bounds_before))
            for e in elements_before
        ]
        norm_ea = [
            (e, _normalize_bounds_relative_to_window(e.bounds, win_bounds_after))
            for e in elements_after
        ]

        # Calculate pairwise match score matrix
        # Score = 0.40 * text_sim + 0.30 * iou + 0.20 * centroid_dist_sim + 0.10 * type_match
        matched_a_indices: Set[int] = set()
        matched_b_indices: Set[int] = set()
        matches: List[Tuple[int, int, float, float, float]] = []  # (idx_b, idx_a, score, text_sim, iou)

        for idx_b, (el_b, rel_b) in enumerate(norm_eb):
            scored_candidates: List[Tuple[int, float, float, float]] = []

            for idx_a, (el_a, rel_a) in enumerate(norm_ea):
                text_sim = _compute_text_similarity(
                    el_b.text_content or el_b.name,
                    el_a.text_content or el_a.name,
                )
                iou = _compute_iou(rel_b, rel_a)

                # Centroid distance in window-relative space
                dx = rel_a.center.x - rel_b.center.x
                dy = rel_a.center.y - rel_b.center.y
                dist = math.hypot(dx, dy)
                centroid_sim = max(0.0, 1.0 - (dist / 500.0))

                # Type match
                if el_b.element_type == el_a.element_type:
                    type_score = 1.0
                elif el_b.element_type == UIElementType.UNKNOWN or el_a.element_type == UIElementType.UNKNOWN:
                    type_score = 0.5
                else:
                    type_score = 0.0

                score = (0.40 * text_sim) + (0.30 * iou) + (0.20 * centroid_sim) + (0.10 * type_score)
                if score >= MIN_ELEMENT_MATCH_SCORE:
                    scored_candidates.append((idx_a, score, text_sim, iou))

            # Sort candidates by descending score
            scored_candidates.sort(key=lambda x: x[1], reverse=True)

            if len(scored_candidates) >= 2:
                # Ambiguity check: if top 2 candidates are materially indistinguishable
                top1 = scored_candidates[0]
                top2 = scored_candidates[1]
                if abs(top1[1] - top2[1]) < ELEMENT_MATCH_AMBIGUITY_DELTA and top1[2] == top2[2]:
                    # Duplicate identical elements without distinct spatial separation
                    is_ambiguous = True

            if scored_candidates:
                best = scored_candidates[0]
                matches.append((idx_b, best[0], best[1], best[2], best[3]))

        # Sort all candidate pairs globally by score descending to resolve 1-to-1 assignments
        matches.sort(key=lambda x: x[2], reverse=True)

        assigned_b: Set[int] = set()
        assigned_a: Set[int] = set()

        for idx_b, idx_a, score, text_sim, iou in matches:
            if idx_b in assigned_b or idx_a in assigned_a:
                continue

            assigned_b.add(idx_b)
            assigned_a.add(idx_a)

            el_b, rel_b = norm_eb[idx_b]
            el_a, rel_a = norm_ea[idx_a]

            dx = rel_a.center.x - rel_b.center.x
            dy = rel_a.center.y - rel_b.center.y
            disp_dist = math.hypot(dx, dy)

            # Classify change between paired elements
            if text_sim < 0.95 and (el_b.text_content or el_b.name) != (el_a.text_content or el_a.name):
                change_type = VisualDeltaType.TEXT_CHANGED
            elif rel_b.area > 0 and abs(rel_a.area - rel_b.area) / max(1, rel_b.area) >= RESIZE_RATIO_THRESHOLD:
                change_type = VisualDeltaType.ELEMENT_RESIZED
            elif disp_dist >= MOVE_DISTANCE_THRESHOLD:
                change_type = VisualDeltaType.ELEMENT_MOVED
            else:
                change_type = VisualDeltaType.NO_MEANINGFUL_CHANGE

            changes.append(
                UIElementChange(
                    change_type=change_type,
                    before_element=el_b,
                    after_element=el_a,
                    displacement=Point(x=int(dx), y=int(dy)),
                    text_similarity=text_sim,
                    iou=iou,
                    confidence=score,
                )
            )

        # Unmatched in T0 -> ELEMENT_DISAPPEARED
        for idx_b, (el_b, _) in enumerate(norm_eb):
            if idx_b not in assigned_b:
                changes.append(
                    UIElementChange(
                        change_type=VisualDeltaType.ELEMENT_DISAPPEARED,
                        before_element=el_b,
                        after_element=None,
                        confidence=el_b.confidence,
                    )
                )

        # Unmatched in T1 -> ELEMENT_APPEARED
        for idx_a, (el_a, _) in enumerate(norm_ea):
            if idx_a not in assigned_a:
                changes.append(
                    UIElementChange(
                        change_type=VisualDeltaType.ELEMENT_APPEARED,
                        before_element=None,
                        after_element=el_a,
                        confidence=el_a.confidence,
                    )
                )

        return changes, is_ambiguous

    # -----------------------------------------------------------------------
    # Multimodal Fallback
    # -----------------------------------------------------------------------

    def _execute_multimodal_diff(
        self,
        before: ScreenObservation,
        after: ScreenObservation,
        target: str,
    ) -> Optional[Dict[str, Any]]:
        """Perform multimodal VLM semantic differential analysis."""
        ai_p = self.ai_provider
        if ai_p is None or not getattr(ai_p, "supports_multimodal", False):
            return None

        cap_after = after.capture
        if cap_after is None or cap_after.is_empty:
            return None

        img_part = self._preprocessor.to_image_part(cap_after)
        prompt = _DEFAULT_DELTA_PROMPT.format(target=target)

        payload = self._prompt_builder.build_payload(
            query=prompt,
            images=[img_part],
        )

        response = ai_p.generate(payload)
        content = getattr(response, "content", str(response)).strip()

        # Clean markdown code blocks
        if "```json" in content:
            content = content.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in content:
            content = content.split("```", 1)[1].split("```", 1)[0].strip()

        try:
            data = json.loads(content)
            if isinstance(data, dict):
                return data
        except Exception:
            self._logger.debug("Failed to parse multimodal JSON response: %s", content)

        return None

    # -----------------------------------------------------------------------
    # Verdict Synthesis
    # -----------------------------------------------------------------------

    def _synthesize_verdict(
        self,
        element_changes: List[UIElementChange],
        added_texts: List[str],
        removed_texts: List[str],
        modified_texts: List[str],
        is_ambiguous: bool,
        multimodal_result: Optional[Dict[str, Any]],
        target_filter: Optional[str],
    ) -> Tuple[VisualDeltaType, str, float]:
        """Synthesize overall delta classification, human-readable explanation, and confidence."""
        # 1. Check for ambiguity (Ambiguous matches MUST result in UNCERTAIN rather than guessing)
        if is_ambiguous:
            return (
                VisualDeltaType.UNCERTAIN,
                "Screen state changed, but multiple identical elements made exact matching ambiguous.",
                0.5,
            )

        # 2. If multimodal result exists and deterministic changes were inconclusive
        if multimodal_result is not None:
            ctype_raw = str(multimodal_result.get("change_type", "uncertain")).strip().lower()
            explanation = str(multimodal_result.get("explanation") or "").strip()
            conf = float(multimodal_result.get("confidence", 0.8))
            try:
                ctype = VisualDeltaType(ctype_raw)
                return ctype, explanation or f"Visual change detected: {ctype.value}.", conf
            except Exception:
                pass

        # 3. Analyze classified element changes
        appeared = [c for c in element_changes if c.change_type == VisualDeltaType.ELEMENT_APPEARED]
        disappeared = [c for c in element_changes if c.change_type == VisualDeltaType.ELEMENT_DISAPPEARED]
        moved = [c for c in element_changes if c.change_type == VisualDeltaType.ELEMENT_MOVED]
        text_changed = [c for c in element_changes if c.change_type == VisualDeltaType.TEXT_CHANGED]
        resized = [c for c in element_changes if c.change_type == VisualDeltaType.ELEMENT_RESIZED]

        # Prioritize specific change types
        if appeared:
            names = ", ".join(f"'{c.after_element.name}'" for c in appeared[:3] if c.after_element)
            return (
                VisualDeltaType.ELEMENT_APPEARED,
                f"Element(s) appeared on the screen: {names}.",
                0.95,
            )

        if disappeared:
            names = ", ".join(f"'{c.before_element.name}'" for c in disappeared[:3] if c.before_element)
            return (
                VisualDeltaType.ELEMENT_DISAPPEARED,
                f"Element(s) disappeared from the screen: {names}.",
                0.95,
            )

        if text_changed or modified_texts:
            details = []
            if text_changed:
                for c in text_changed[:2]:
                    if c.before_element and c.after_element:
                        details.append(f"'{c.before_element.name}' changed to '{c.after_element.name}'")
            elif modified_texts:
                details.extend(modified_texts[:2])

            desc = f"Text changed: {', '.join(details)}." if details else "Text content changed on the screen."
            return (
                VisualDeltaType.TEXT_CHANGED,
                desc,
                0.95,
            )

        if moved:
            c = moved[0]
            name = c.after_element.name if c.after_element else "Element"
            dx = c.displacement.x if c.displacement else 0
            dy = c.displacement.y if c.displacement else 0
            return (
                VisualDeltaType.ELEMENT_MOVED,
                f"'{name}' moved by ({dx:+d}px, {dy:+d}px).",
                0.90,
            )

        if resized:
            c = resized[0]
            name = c.after_element.name if c.after_element else "Element"
            return (
                VisualDeltaType.ELEMENT_RESIZED,
                f"'{name}' was resized.",
                0.90,
            )

        if added_texts:
            return (
                VisualDeltaType.ELEMENT_APPEARED,
                f"New text appeared on the screen: '{added_texts[0]}'.",
                0.90,
            )

        if removed_texts:
            return (
                VisualDeltaType.ELEMENT_DISAPPEARED,
                f"Text was removed from the screen: '{removed_texts[0]}'.",
                0.90,
            )

        return (
            VisualDeltaType.NO_MEANINGFUL_CHANGE,
            "No meaningful visual changes were detected.",
            1.0,
        )


__all__ = [
    "ELEMENT_MATCH_AMBIGUITY_DELTA",
    "IOU_MATCH_THRESHOLD",
    "MIN_ELEMENT_MATCH_SCORE",
    "MOVE_DISTANCE_THRESHOLD",
    "RESIZE_RATIO_THRESHOLD",
    "TEXT_SIMILARITY_THRESHOLD",
    "VisualDeltaEngine",
]
