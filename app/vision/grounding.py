"""Visual Grounding and UI Element Localization Engine for J.A.R.V.I.S (Phase 27.8).

Coordinates natural-language visual target localization:
1. OCR fast path for deterministic textual targets.
2. Multimodal spatial grounding for semantic, iconic, and visual targets.
3. Strict coordinate parsing, validation, and calibration.
4. Hybrid spatial fusion of OCR text blocks and multimodal bounding boxes.
5. Production of immutable UIElement and VisualGroundingResult models.

Safety Invariants Enforced:
- All visual observation acquisition remains strictly gated by SecureVisionManager.
- Model coordinates are untrusted input: strictly validated, normalized, and dimension-checked.
- Malformed, negative, out-of-range, or inverted model coordinates are rejected (is_found=False).
- Zero raw pixel data or screenshot buffers in returned objects, logs, or telemetry.
- Graceful degradation for text-only providers (Ollama default) to OCR matching.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
    UIElementType,
    VisualGroundingResult,
    WindowBounds,
)
from app.vision.ocr import OCRProvider

logger = get_logger("VISION.GROUNDING")

_DEFAULT_GROUNDING_PROMPT = (
    "Locate the following UI element on the screen: '{target}'\n"
    "Respond ONLY with a valid JSON object adhering strictly to this schema:\n"
    "{\n"
    '  "found": true,\n'
    '  "element_type": "button|input|icon|text|link|checkbox|dropdown|menu|tab|dialog|container|unknown",\n'
    '  "box_2d": [ymin, xmin, ymax, xmax],\n'
    '  "confidence": 0.0-1.0,\n'
    '  "label": "exact label or description"\n'
    "}\n"
    "Coordinates in box_2d MUST be normalized integers in the range [0, 1000] representing [ymin, xmin, ymax, xmax].\n"
    'If the target element is NOT visible or cannot be determined with confidence, respond ONLY with:\n'
    '{"found": false, "confidence": 0.0, "reason": "Element not visible"}'
)


class VisualGroundingEngine:
    """Production visual grounding engine mapping target queries to UIElement coordinates."""

    def __init__(
        self,
        *,
        ocr_provider: Optional[OCRProvider] = None,
        ai_provider: Optional[Any] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        logger_instance: Optional[logging.Logger] = None,
        container_instance: Optional[ServiceContainer] = None,
    ) -> None:
        """Initialize VisualGroundingEngine."""
        self._ocr_provider = ocr_provider
        self._ai_provider = ai_provider
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._logger = logger_instance or logger
        self._container = container_instance or default_container

    # --------------------------------------------------------------------------
    # Dependency Resolvers
    # --------------------------------------------------------------------------

    def _get_ocr_provider(self) -> Optional[OCRProvider]:
        if self._ocr_provider is not None:
            return self._ocr_provider
        if self._container is not None and self._container.exists("ocr_provider"):
            return self._container.resolve("ocr_provider")
        return None

    def _get_ai_provider(self) -> Optional[Any]:
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

    # --------------------------------------------------------------------------
    # Public Entry Point
    # --------------------------------------------------------------------------

    async def locate_element_async(
        self,
        target: str,
        observation: ScreenObservation,
        *,
        ocr_result: Optional[OCRResult] = None,
        element_type_hint: Optional[Union[str, UIElementType]] = None,
        force_multimodal: bool = False,
    ) -> VisualGroundingResult:
        """Asynchronously ground a target UI element on the given ScreenObservation.

        Args:
            target: Natural-language description or label of element to find.
            observation: Valid, authorized ScreenObservation.
            ocr_result: Optional pre-computed OCRResult to avoid duplicate OCR.
            element_type_hint: Optional hint regarding expected element type.
            force_multimodal: If True, bypasses OCR fast path to force multimodal reasoning.

        Returns:
            VisualGroundingResult containing resolved UIElement or not-found status.
        """
        start_time = time.perf_counter()
        clean_target = str(target or "").strip()
        obs_id = observation.id if observation else ""

        if not clean_target:
            return VisualGroundingResult(
                target=target,
                element=None,
                is_found=False,
                confidence=0.0,
                observation_id=obs_id,
                duration=time.perf_counter() - start_time,
                summary="Target element description cannot be empty.",
            )

        if observation is None or not observation.is_valid or observation.capture is None:
            return VisualGroundingResult(
                target=clean_target,
                element=None,
                is_found=False,
                confidence=0.0,
                observation_id=obs_id,
                duration=time.perf_counter() - start_time,
                summary="Cannot ground element: no valid screen capture observation available.",
            )

        capture = observation.capture

        # 1. OCR Extraction (if not pre-computed and OCR provider available)
        active_ocr = ocr_result
        ocr_prov = self._get_ocr_provider()
        if active_ocr is None and ocr_prov is not None:
            try:
                active_ocr = await ocr_prov.extract_text_async(capture)
            except Exception as ocr_exc:
                self._logger.debug("OCR extraction failed during grounding: %s", ocr_exc)
                active_ocr = None

        # 2. OCR Fast Path for textual targets
        if not force_multimodal and active_ocr is not None and not active_ocr.is_empty:
            ocr_match = self._try_ocr_grounding(clean_target, active_ocr, element_type_hint)
            if ocr_match is not None:
                # If unambiguous high confidence match found
                if ocr_match.is_found:
                    dur = time.perf_counter() - start_time
                    summary = self._format_summary(clean_target, ocr_match.element, True, capture.bounds)
                    return VisualGroundingResult(
                        target=clean_target,
                        element=ocr_match.element,
                        is_found=True,
                        confidence=ocr_match.confidence,
                        observation_id=obs_id,
                        duration=dur,
                        summary=summary,
                        metadata={"source": "ocr_fast_path"},
                    )
                # If multiple ambiguous matches were found, we will try multimodal or report ambiguity
                if ocr_match.metadata.get("ambiguous", False):
                    self._logger.debug("OCR matched multiple ambiguous candidates for '%s'. Attempting multimodal resolution.", clean_target)

        # 3. Multimodal Spatial Path
        ai_prov = self._get_ai_provider()
        if ai_prov is not None:
            try:
                mm_res = await self._ground_multimodal(clean_target, capture, ai_prov, element_type_hint)
                dur = time.perf_counter() - start_time

                if mm_res.is_found and mm_res.element is not None:
                    # 4. Attempt Spatial Fusion with OCR text blocks if available
                    final_element = mm_res.element
                    final_conf = mm_res.confidence
                    if active_ocr is not None and not active_ocr.is_empty:
                        fused = self._fuse_multimodal_and_ocr(mm_res.element, active_ocr)
                        if fused is not None:
                            final_element = fused
                            final_conf = fused.confidence

                    summary = self._format_summary(clean_target, final_element, True, capture.bounds)
                    return VisualGroundingResult(
                        target=clean_target,
                        element=final_element,
                        is_found=True,
                        confidence=final_conf,
                        observation_id=obs_id,
                        duration=dur,
                        summary=summary,
                        metadata=mm_res.metadata,
                    )
                else:
                    summary = self._format_summary(clean_target, None, False, capture.bounds)
                    return VisualGroundingResult(
                        target=clean_target,
                        element=None,
                        is_found=False,
                        confidence=mm_res.confidence,
                        observation_id=obs_id,
                        duration=dur,
                        summary=summary,
                        metadata=mm_res.metadata,
                    )

            except UnsupportedModalityError:
                self._logger.info("AI provider does not support multimodal vision. Falling back to OCR text analysis.")
            except Exception as mm_exc:
                self._logger.warning("Multimodal grounding failed for target '%s': %s", clean_target, mm_exc)

        # 5. Fallback if multimodal not supported or unavailable
        dur = time.perf_counter() - start_time
        summary = f"I could not locate '{clean_target}' on the active window."
        return VisualGroundingResult(
            target=clean_target,
            element=None,
            is_found=False,
            confidence=0.0,
            observation_id=obs_id,
            duration=dur,
            summary=summary,
            metadata={"fallback": "not_found"},
        )

    # --------------------------------------------------------------------------
    # Step 3: OCR Fast Path
    # --------------------------------------------------------------------------

    def _try_ocr_grounding(
        self,
        target: str,
        ocr_result: OCRResult,
        element_type_hint: Optional[Union[str, UIElementType]] = None,
    ) -> Optional[VisualGroundingResult]:
        """Attempt deterministic OCR text-block matching for the target."""
        clean_target = target.strip()
        norm_target = self._normalize_text(clean_target)
        if not norm_target:
            return None

        # Filter blocks that have non-empty bounds
        valid_blocks = [b for b in ocr_result.blocks if b.bounds and not b.bounds.is_empty]
        if not valid_blocks:
            return None

        # 1. Exact normalized match
        exact_matches: List[OCRTextBlock] = []
        for block in valid_blocks:
            norm_b = self._normalize_text(block.text)
            if norm_b == norm_target:
                exact_matches.append(block)

        if len(exact_matches) == 1:
            match = exact_matches[0]
            el_type = self._infer_element_type(clean_target, element_type_hint)
            el = UIElement(
                name=match.text,
                element_type=el_type,
                bounds=match.bounds,  # type: ignore[arg-type]
                center=match.bounds.center,  # type: ignore[union-attr]
                confidence=1.0,
                source=GroundingSource.OCR_EXACT,
                text_content=match.text,
                metadata={"match_type": "exact_ocr"},
            )
            return VisualGroundingResult(
                target=clean_target,
                element=el,
                is_found=True,
                confidence=1.0,
                summary=f"Found '{match.text}' via OCR.",
            )

        if len(exact_matches) > 1:
            # Ambiguous multiple identical matches
            return VisualGroundingResult(
                target=clean_target,
                element=None,
                is_found=False,
                confidence=0.4,
                summary=f"Found multiple ({len(exact_matches)}) exact OCR matches for '{clean_target}'.",
                metadata={"ambiguous": True, "match_count": len(exact_matches)},
            )

        # 2. Token / Substring containment match (e.g. target="submit", block="submit order")
        containment_matches: List[OCRTextBlock] = []
        for block in valid_blocks:
            norm_b = self._normalize_text(block.text)
            # Check whole word token matching
            target_tokens = set(norm_target.split())
            block_tokens = set(norm_b.split())
            if target_tokens.issubset(block_tokens) or (len(norm_target) >= 4 and norm_target in norm_b):
                containment_matches.append(block)

        if len(containment_matches) == 1:
            match = containment_matches[0]
            el_type = self._infer_element_type(clean_target, element_type_hint)
            el = UIElement(
                name=match.text,
                element_type=el_type,
                bounds=match.bounds,  # type: ignore[arg-type]
                center=match.bounds.center,  # type: ignore[union-attr]
                confidence=0.92,
                source=GroundingSource.OCR_EXACT,
                text_content=match.text,
                metadata={"match_type": "token_ocr"},
            )
            return VisualGroundingResult(
                target=clean_target,
                element=el,
                is_found=True,
                confidence=0.92,
                summary=f"Found '{match.text}' containing target '{clean_target}' via OCR.",
            )

        if len(containment_matches) > 1:
            return VisualGroundingResult(
                target=clean_target,
                element=None,
                is_found=False,
                confidence=0.3,
                summary=f"Found multiple partial OCR matches for '{clean_target}'.",
                metadata={"ambiguous": True, "match_count": len(containment_matches)},
            )

        return None

    # --------------------------------------------------------------------------
    # Step 4: Multimodal Grounding
    # --------------------------------------------------------------------------

    async def _ground_multimodal(
        self,
        target: str,
        capture: ScreenCapture,
        ai_provider: Any,
        element_type_hint: Optional[Union[str, UIElementType]] = None,
    ) -> VisualGroundingResult:
        """Query multimodal AI provider to spatially locate target."""
        if not getattr(ai_provider, "supports_multimodal", False):
            raise UnsupportedModalityError(f"Provider {type(ai_provider).__name__} does not support multimodal vision.")

        png_bytes = capture.to_png_bytes()
        if not png_bytes:
            return VisualGroundingResult(
                target=target,
                element=None,
                is_found=False,
                confidence=0.0,
                summary="Empty screen capture buffer.",
            )

        prompt = _DEFAULT_GROUNDING_PROMPT.replace("{target}", target)
        image_part = ImagePart(data=png_bytes, mime_type="image/png")

        raw_response: str = ""
        try:
            if hasattr(ai_provider, "generate_content_async"):
                raw_response = await ai_provider.generate_content_async(prompt, images=[image_part])
            elif hasattr(ai_provider, "generate_content"):
                import asyncio
                raw_response = await asyncio.to_thread(ai_provider.generate_content, prompt, images=[image_part])
            else:
                raise RuntimeError("AI provider has no recognized generate_content method.")
        except UnsupportedModalityError:
            raise
        except Exception as exc:
            self._logger.warning("Error during multimodal content generation: %s", exc)
            return VisualGroundingResult(
                target=target,
                element=None,
                is_found=False,
                confidence=0.0,
                summary=f"Multimodal vision generation failed: {exc}",
            )

        return self._parse_multimodal_grounding_response(raw_response, target, capture, element_type_hint)

    # --------------------------------------------------------------------------
    # Step 5 & 6: Coordinate Contract, Validation, & Calibration
    # --------------------------------------------------------------------------

    def _parse_multimodal_grounding_response(
        self,
        raw_response: str,
        target: str,
        capture: ScreenCapture,
        element_type_hint: Optional[Union[str, UIElementType]] = None,
    ) -> VisualGroundingResult:
        """Parse structured JSON and validate bounding box coordinates defensively."""
        if not raw_response or not raw_response.strip():
            return VisualGroundingResult(
                target=target,
                element=None,
                is_found=False,
                confidence=0.0,
                summary="Empty response received from vision model.",
            )

        data = self._extract_json_payload(raw_response)
        if not data or not isinstance(data, dict):
            return VisualGroundingResult(
                target=target,
                element=None,
                is_found=False,
                confidence=0.0,
                summary="Malformed vision model response; could not parse JSON.",
            )

        is_found = bool(data.get("found", False))
        conf = float(max(0.0, min(1.0, float(data.get("confidence", 0.0) or 0.0))))

        if not is_found or conf < 0.35:
            return VisualGroundingResult(
                target=target,
                element=None,
                is_found=False,
                confidence=conf,
                summary=f"Element '{target}' was not found with sufficient confidence.",
            )

        # Extract coordinate box
        box_data = data.get("box_2d") or data.get("bbox") or data.get("bounds") or data.get("coordinates")
        if box_data is None:
            # Check for direct dictionary keys in data
            if any(k in data for k in ("left", "x", "ymin")):
                box_data = data

        bounds = self._parse_and_validate_box(box_data, capture.width, capture.height)
        if bounds is None:
            return VisualGroundingResult(
                target=target,
                element=None,
                is_found=False,
                confidence=0.0,
                summary="Model returned invalid or out-of-bounds coordinates.",
                metadata={"raw_box": str(box_data)},
            )

        raw_type = str(data.get("element_type", "")).lower()
        el_type = self._parse_element_type(raw_type) if raw_type else self._infer_element_type(target, element_type_hint)
        label = str(data.get("label") or target).strip()

        element = UIElement(
            name=label,
            element_type=el_type,
            bounds=bounds,
            center=bounds.center,
            confidence=conf,
            source=GroundingSource.MULTIMODAL_SEMANTIC,
            text_content=label if el_type in (UIElementType.TEXT, UIElementType.BUTTON) else None,
            metadata={"raw_model_confidence": conf},
        )

        return VisualGroundingResult(
            target=target,
            element=element,
            is_found=True,
            confidence=conf,
            summary=f"Located '{label}' via multimodal spatial reasoning.",
            metadata={"source": "multimodal"},
        )

    def _parse_and_validate_box(
        self,
        box_data: Any,
        frame_width: int,
        frame_height: int,
    ) -> Optional[WindowBounds]:
        """Strict coordinate validation and calibration from model space to pixel space."""
        if box_data is None or frame_width <= 0 or frame_height <= 0:
            return None

        ymin: Optional[float] = None
        xmin: Optional[float] = None
        ymax: Optional[float] = None
        xmax: Optional[float] = None

        # Format 1: List / Tuple [ymin, xmin, ymax, xmax] (standard Gemini 2D box)
        if isinstance(box_data, (list, tuple)) and len(box_data) == 4:
            try:
                coords = [float(c) for c in box_data]
                if any(math.isnan(c) or math.isinf(c) for c in coords):
                    return None

                # Check if coordinates are in [ymin, xmin, ymax, xmax] or [xmin, ymin, xmax, ymax]
                # Standard convention for multimodal vision prompts is [ymin, xmin, ymax, xmax]
                ymin, xmin, ymax, xmax = coords[0], coords[1], coords[2], coords[3]
            except (ValueError, TypeError):
                return None

        # Format 2: Dict format {ymin, xmin, ymax, xmax} or {left, top, right, bottom} or {x, y, width, height}
        elif isinstance(box_data, dict):
            try:
                if all(k in box_data for k in ("ymin", "xmin", "ymax", "xmax")):
                    ymin = float(box_data["ymin"])
                    xmin = float(box_data["xmin"])
                    ymax = float(box_data["ymax"])
                    xmax = float(box_data["xmax"])
                elif all(k in box_data for k in ("top", "left", "bottom", "right")):
                    ymin = float(box_data["top"])
                    xmin = float(box_data["left"])
                    ymax = float(box_data["bottom"])
                    xmax = float(box_data["right"])
                elif all(k in box_data for k in ("x", "y", "width", "height")):
                    xmin = float(box_data["x"])
                    ymin = float(box_data["y"])
                    xmax = xmin + float(box_data["width"])
                    ymax = ymin + float(box_data["height"])
                else:
                    return None
            except (ValueError, TypeError):
                return None

        if ymin is None or xmin is None or ymax is None or xmax is None:
            return None

        # Check for NaN / Infinity
        if any(math.isnan(v) or math.isinf(v) for v in (ymin, xmin, ymax, xmax)):
            return None

        # Strict Safety: Reject negative values or inverted coordinates
        if ymin < 0.0 or xmin < 0.0 or ymax < 0.0 or xmax < 0.0:
            return None
        if xmax <= xmin or ymax <= ymin:
            return None

        # Determine normalization scale (0..1000 vs 0..1 vs absolute pixels)
        max_val = max(ymin, xmin, ymax, xmax)

        if max_val <= 1.01:
            # 0.0 .. 1.0 range
            norm_xmin = xmin
            norm_ymin = ymin
            norm_xmax = xmax
            norm_ymax = ymax
        elif max_val <= 1000.0:
            # 0 .. 1000 range
            norm_xmin = xmin / 1000.0
            norm_ymin = ymin / 1000.0
            norm_xmax = xmax / 1000.0
            norm_ymax = ymax / 1000.0
        elif max_val <= max(frame_width, frame_height) * 1.05:
            # Already in absolute pixel space
            norm_xmin = xmin / float(frame_width)
            norm_ymin = ymin / float(frame_height)
            norm_xmax = xmax / float(frame_width)
            norm_ymax = ymax / float(frame_height)
        else:
            # Out of any recognized coordinate range -> reject
            return None

        # Clamping normalized bounds strictly to [0.0, 1.0]
        norm_xmin = max(0.0, min(1.0, norm_xmin))
        norm_ymin = max(0.0, min(1.0, norm_ymin))
        norm_xmax = max(0.0, min(1.0, norm_xmax))
        norm_ymax = max(0.0, min(1.0, norm_ymax))

        # Convert to pixel dimensions
        px_left = int(round(norm_xmin * frame_width))
        px_top = int(round(norm_ymin * frame_height))
        px_right = int(round(norm_xmax * frame_width))
        px_bottom = int(round(norm_ymax * frame_height))

        # Minimum dimensions check (at least 4x4 pixels)
        if (px_right - px_left) < 4 or (px_bottom - px_top) < 4:
            return None

        return WindowBounds(
            left=px_left,
            top=px_top,
            right=px_right,
            bottom=px_bottom,
        )

    # --------------------------------------------------------------------------
    # Step 7: Spatial Fusion
    # --------------------------------------------------------------------------

    def _fuse_multimodal_and_ocr(
        self,
        mm_element: UIElement,
        ocr_result: OCRResult,
    ) -> Optional[UIElement]:
        """Perform spatial intersection and text evidence fusion between Multimodal and OCR."""
        if mm_element is None or not mm_element.bounds or ocr_result.is_empty:
            return None

        mm_bounds = mm_element.bounds
        overlapping_blocks: List[OCRTextBlock] = []

        for block in ocr_result.blocks:
            if block.bounds and block.bounds.intersects(mm_bounds):
                inter = block.bounds.intersection(mm_bounds)
                if inter and inter.area >= 0.2 * min(block.bounds.area, mm_bounds.area):
                    overlapping_blocks.append(block)

        if not overlapping_blocks:
            return None

        # Consolidate overlapping text
        recognized_texts = [b.text.strip() for b in overlapping_blocks if b.text.strip()]
        combined_text = " ".join(recognized_texts)

        # Check if multimodal label is confirmed by OCR text
        norm_label = self._normalize_text(mm_element.name)
        norm_ocr = self._normalize_text(combined_text)

        confidence_boost = 0.95
        if norm_label and norm_label in norm_ocr:
            confidence_boost = 0.98

        # Encompassing bounding box that covers both visual container and OCR text
        fused_left = min(mm_bounds.left, min(b.bounds.left for b in overlapping_blocks if b.bounds))
        fused_top = min(mm_bounds.top, min(b.bounds.top for b in overlapping_blocks if b.bounds))
        fused_right = max(mm_bounds.right, max(b.bounds.right for b in overlapping_blocks if b.bounds))
        fused_bottom = max(mm_bounds.bottom, max(b.bounds.bottom for b in overlapping_blocks if b.bounds))

        fused_bounds = WindowBounds(
            left=fused_left,
            top=fused_top,
            right=fused_right,
            bottom=fused_bottom,
        )

        return UIElement(
            name=combined_text or mm_element.name,
            element_type=mm_element.element_type,
            bounds=fused_bounds,
            center=fused_bounds.center,
            confidence=confidence_boost,
            source=GroundingSource.HYBRID_FUSED,
            text_content=combined_text or mm_element.text_content,
            metadata={
                "multimodal_label": mm_element.name,
                "ocr_evidence": combined_text,
                "fused_block_count": len(overlapping_blocks),
            },
        )

    # --------------------------------------------------------------------------
    # Formatting & Utility Helpers
    # --------------------------------------------------------------------------

    def _format_summary(
        self,
        target: str,
        element: Optional[UIElement],
        is_found: bool,
        container_bounds: Optional[WindowBounds] = None,
    ) -> str:
        """Format a human-readable conversational summary without revealing raw pixels."""
        if not is_found or element is None:
            return f"I could not find the '{target}' on the active window."

        center = element.center
        loc_desc = "active window"
        if container_bounds and not container_bounds.is_empty:
            w_mid = container_bounds.width / 2.0
            h_mid = container_bounds.height / 2.0
            x_rel = center.x - container_bounds.left if container_bounds.left > 0 else center.x
            y_rel = center.y - container_bounds.top if container_bounds.top > 0 else center.y

            horiz = "left" if x_rel < w_mid * 0.75 else ("right" if x_rel > w_mid * 1.25 else "center")
            vert = "top" if y_rel < h_mid * 0.75 else ("bottom" if y_rel > h_mid * 1.25 else "middle")
            if horiz == "center" and vert == "middle":
                loc_desc = "center of the window"
            else:
                loc_desc = f"{vert}-{horiz} of the window"

        el_name = element.name or target
        type_str = element.element_type.value if element.element_type != UIElementType.UNKNOWN else "element"
        return f"Found '{el_name}' ({type_str}) near the {loc_desc} at ({center.x}, {center.y})."

    def _normalize_text(self, text: str) -> str:
        """Normalize string: lowercase, strip, collapse whitespace, strip punctuation."""
        if not text:
            return ""
        clean = re.sub(r"[^\w\s]", "", text.lower())
        return " ".join(clean.split())

    def _infer_element_type(
        self,
        target: str,
        hint: Optional[Union[str, UIElementType]] = None,
    ) -> UIElementType:
        """Infer UIElementType from target text or explicit hint."""
        if hint is not None:
            return self._parse_element_type(hint)

        low = target.lower()
        if "button" in low or "btn" in low:
            return UIElementType.BUTTON
        if "icon" in low:
            return UIElementType.ICON
        if "input" in low or "field" in low or "search bar" in low or "box" in low:
            return UIElementType.INPUT
        if "link" in low or "url" in low:
            return UIElementType.LINK
        if "checkbox" in low or "check box" in low:
            return UIElementType.CHECKBOX
        if "dropdown" in low or "select" in low:
            return UIElementType.DROPDOWN
        if "menu" in low:
            return UIElementType.MENU
        if "tab" in low:
            return UIElementType.TAB
        if "dialog" in low or "modal" in low or "popup" in low:
            return UIElementType.DIALOG
        return UIElementType.BUTTON if len(target.split()) <= 2 else UIElementType.TEXT

    def _parse_element_type(self, type_str: Union[str, UIElementType]) -> UIElementType:
        """Parse string to UIElementType enum value."""
        if isinstance(type_str, UIElementType):
            return type_str
        clean = str(type_str or "").strip().lower()
        for member in UIElementType:
            if member.value == clean:
                return member
        return UIElementType.UNKNOWN

    def _extract_json_payload(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON object from text or markdown code fence."""
        if not text:
            return None
        # Try raw json load
        try:
            return json.loads(text.strip())
        except Exception:
            pass

        # Try markdown code fence regex
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass

        # Try finding outer braces
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                pass

        return None


__all__ = [
    "VisualGroundingEngine",
]
