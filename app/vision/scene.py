"""Visual UI Scene Parsing and Interactive Element Mapping Engine (Phase 27.11).

Provides deterministic-first, privacy-safe structural analysis of active window/screen
observations to parse UI layouts into semantic containers (Header, Sidebar, Form, etc.),
catalog interactive widgets, and bind label-to-input form fields.

Safety and Architectural Invariants Enforced:
1. Purely read-only visual perception: ZERO mouse/keyboard automation, clicks, typing,
   or desktop state manipulation.
2. Zero persistent raw screenshots or pixel bytes in returned models, logs, or telemetry.
3. Deterministic-first parsing: Functions 100% on CPU without requiring VLM inference.
4. Conservative classification: Ambiguous visual regions default to CONTENT_AREA or UNCERTAIN.
5. Strict VLM isolation: Optional VLM fallback strictly validates JSON and coordinates,
   gracefully falling back to deterministic results on any error or malformed payload.
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
    FormField,
    GroundingSource,
    OCRResult,
    OCRTextBlock,
    Point,
    ScreenCapture,
    ScreenObservation,
    UIContainer,
    UIContainerType,
    UIElement,
    UIElementType,
    UIScene,
    WindowBounds,
)
from app.vision.ocr import OCRProvider
from app.vision.preprocessing import ImagePreprocessor, default_preprocessor

logger = get_logger("VISION.SCENE")

# --------------------------------------------------------------------------
# Named Constants & Thresholds
# --------------------------------------------------------------------------

MAX_LABEL_INPUT_RIGHT_DISTANCE: float = 250.0
MAX_LABEL_INPUT_BELOW_DISTANCE: float = 120.0
LABEL_ALIGNMENT_TOLERANCE_Y: float = 35.0
LABEL_ALIGNMENT_TOLERANCE_X: float = 60.0
CONTAINER_MIN_PADDING: int = 5

_DEFAULT_SCENE_PROMPT = (
    "You are J.A.R.V.I.S Visual Scene Parsing Engine.\n"
    "Analyze the active application window and describe its structural layout and controls.\n"
    "Respond ONLY with a valid JSON object adhering strictly to this schema:\n"
    "{{\n"
    '  "summary": "concise overview of layout and controls",\n'
    '  "containers": [\n'
    "    {{\n"
    '      "type": "header|sidebar|toolbar|form|dialog|table|tab_panel|status_bar|content_area|uncertain",\n'
    '      "label": "container title or null",\n'
    '      "box_2d": [ymin, xmin, ymax, xmax],\n'
    '      "confidence": 0.0-1.0\n'
    "    }}\n"
    "  ],\n"
    '  "interactive_elements": [\n'
    "    {{\n"
    '      "name": "control name",\n'
    '      "type": "button|input|checkbox|radio|dropdown|link|tab|icon|slider|unknown",\n'
    '      "box_2d": [ymin, xmin, ymax, xmax],\n'
    '      "confidence": 0.0-1.0\n'
    "    }}\n"
    "  ]\n"
    "}}\n"
    "Coordinates in box_2d MUST be normalized integers [ymin, xmin, ymax, xmax] in range [0, 1000]."
)


# --------------------------------------------------------------------------
# Geometric Utility Functions
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


def _normalize_bounds(bounds: WindowBounds, win_bounds: Optional[WindowBounds]) -> WindowBounds:
    """Normalize coordinate bounds relative to active window origin."""
    if not win_bounds or win_bounds.is_empty:
        return bounds
    return WindowBounds(
        left=max(0, bounds.left - win_bounds.left),
        top=max(0, bounds.top - win_bounds.top),
        right=max(0, bounds.right - win_bounds.left),
        bottom=max(0, bounds.bottom - win_bounds.top),
    )


# --------------------------------------------------------------------------
# Visual Scene Parser
# --------------------------------------------------------------------------


class VisualSceneParser:
    """Production Visual UI Scene Parsing & Interactive Element Mapping Engine."""

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
    # Main Parsing Interface
    # -----------------------------------------------------------------------

    def parse_scene(
        self,
        observation: ScreenObservation,
        container_filter: Optional[str] = None,
        force_multimodal: bool = False,
    ) -> UIScene:
        """Parse active screen observation into a structured UIScene."""
        start_time = time.time()
        scene_id = str(uuid.uuid4())

        if not observation:
            return UIScene(
                scene_id=scene_id,
                observation_id="",
                window_title="",
                confidence=0.0,
                summary="Cannot parse invalid or missing screen observation.",
            )

        win_title = str(observation.metadata.get("window_title") or "").strip()
        proc_name = observation.metadata.get("process_name")
        win_bounds = observation.bounds or WindowBounds(0, 0, 1000, 800)

        # ------------------------------------------------------------------
        # Phase 1: Extract OCR Text Blocks and UIElements
        # ------------------------------------------------------------------
        ocr_blocks = self._extract_ocr_blocks(observation)
        elements = self._extract_ui_elements(observation, ocr_blocks, win_bounds)

        # ------------------------------------------------------------------
        # Phase 2: Widget Classification & Filtering
        # ------------------------------------------------------------------
        interactive_elements = [
            e for e in elements
            if e.element_type not in (UIElementType.UNKNOWN, UIElementType.CONTAINER, UIElementType.TEXT)
        ]

        # ------------------------------------------------------------------
        # Phase 3: Form Field Association (Label -> Input)
        # ------------------------------------------------------------------
        form_fields = self._associate_form_fields(elements, ocr_blocks, win_bounds)

        # ------------------------------------------------------------------
        # Phase 4: Container Inference
        # ------------------------------------------------------------------
        containers = self._infer_containers(
            elements=elements,
            ocr_blocks=ocr_blocks,
            form_fields=form_fields,
            win_bounds=win_bounds,
            win_title=win_title,
        )

        # ------------------------------------------------------------------
        # Phase 5: Optional VLM Fallback
        # ------------------------------------------------------------------
        multimodal_data: Optional[Dict[str, Any]] = None
        if force_multimodal or (not containers and interactive_elements):
            ai_p = self.ai_provider
            if ai_p is not None and getattr(ai_p, "supports_multimodal", False):
                try:
                    multimodal_data = self._execute_multimodal_scene_parse(
                        observation=observation,
                        win_bounds=win_bounds,
                    )
                except Exception as exc:
                    self._logger.debug("VLM scene enhancement fallback failed: %s", exc)

        # Merge VLM enhancements if valid
        if multimodal_data is not None:
            containers, interactive_elements = self._merge_vlm_enhancements(
                containers=containers,
                interactive_elements=interactive_elements,
                vlm_data=multimodal_data,
                win_bounds=win_bounds,
            )

        # Apply container filter if requested
        if container_filter and container_filter.strip():
            c_type_filter = container_filter.strip().lower()
            containers = [
                c for c in containers
                if c_type_filter in c.container_type.value.lower()
                or (c.label and c_type_filter in c.label.lower())
            ]

        # ------------------------------------------------------------------
        # Phase 6: Synthesize Natural-Language Summary
        # ------------------------------------------------------------------
        summary = self._synthesize_scene_summary(
            win_title=win_title,
            containers=containers,
            interactive_elements=interactive_elements,
            form_fields=form_fields,
        )

        overall_conf = 1.0
        if not interactive_elements and not containers:
            overall_conf = 0.5

        return UIScene(
            scene_id=scene_id,
            observation_id=observation.observation_id,
            window_title=win_title,
            process_name=proc_name,
            window_bounds=win_bounds,
            containers=tuple(containers),
            interactive_elements=tuple(interactive_elements),
            form_fields=tuple(form_fields),
            summary=summary,
            confidence=overall_conf,
            metadata={
                "container_count": len(containers),
                "interactive_count": len(interactive_elements),
                "form_field_count": len(form_fields),
                "ocr_block_count": len(ocr_blocks),
                "duration_seconds": max(0.0, time.time() - start_time),
                "used_vlm": multimodal_data is not None,
            },
        )

    async def parse_scene_async(
        self,
        observation: ScreenObservation,
        container_filter: Optional[str] = None,
        force_multimodal: bool = False,
    ) -> UIScene:
        """Asynchronously parse active screen observation on a worker thread."""
        return await asyncio.to_thread(
            self.parse_scene,
            observation,
            container_filter,
            force_multimodal,
        )

    # -----------------------------------------------------------------------
    # OCR and Element Extraction
    # -----------------------------------------------------------------------

    def _extract_ocr_blocks(self, observation: ScreenObservation) -> List[OCRTextBlock]:
        """Extract valid OCR blocks from metadata or run OCRProvider."""
        blocks: List[OCRTextBlock] = []

        if "ocr_result" in observation.metadata:
            res = observation.metadata["ocr_result"]
            if isinstance(res, OCRResult):
                blocks = list(res.blocks)
            elif isinstance(res, dict) and "blocks" in res:
                for b in res["blocks"]:
                    b_bounds = b.get("bounds", {})
                    blocks.append(
                        OCRTextBlock(
                            text=str(b.get("text", "")),
                            bounds=WindowBounds(
                                left=int(b_bounds.get("left", 0)),
                                top=int(b_bounds.get("top", 0)),
                                right=int(b_bounds.get("right", 0)),
                                bottom=int(b_bounds.get("bottom", 0)),
                            ),
                            confidence=float(b.get("confidence", 1.0)),
                        )
                    )

        if not blocks and "ocr_blocks" in observation.metadata:
            raw = observation.metadata["ocr_blocks"]
            if isinstance(raw, list):
                for b in raw:
                    if isinstance(b, OCRTextBlock):
                        blocks.append(b)
                    elif isinstance(b, dict):
                        b_bounds = b.get("bounds", {})
                        blocks.append(
                            OCRTextBlock(
                                text=str(b.get("text", "")),
                                bounds=WindowBounds(
                                    left=int(b_bounds.get("left", 0)),
                                    top=int(b_bounds.get("top", 0)),
                                    right=int(b_bounds.get("right", 0)),
                                    bottom=int(b_bounds.get("bottom", 0)),
                                ),
                                confidence=float(b.get("confidence", 1.0)),
                            )
                        )

        if not blocks:
            ocr_p = self.ocr_provider
            if ocr_p is not None and observation.capture is not None and not observation.capture.is_empty:
                try:
                    ocr_res = ocr_p.extract_text(observation.capture)
                    blocks = list(ocr_res.blocks)
                except Exception as exc:
                    self._logger.debug("OCR extraction failed in scene parser: %s", exc)

        # Filter out malformed bounds
        valid_blocks = [b for b in blocks if b.bounds and not b.bounds.is_empty and b.text.strip()]
        return valid_blocks

    def _extract_ui_elements(
        self,
        observation: ScreenObservation,
        ocr_blocks: List[OCRTextBlock],
        win_bounds: WindowBounds,
    ) -> List[UIElement]:
        """Extract existing UIElements or synthesize them from OCR blocks."""
        elements: List[UIElement] = []

        if "elements" in observation.metadata:
            raw = observation.metadata["elements"]
            if isinstance(raw, list):
                for e in raw:
                    if isinstance(e, UIElement):
                        elements.append(e)
                    elif isinstance(e, dict):
                        b = e.get("bounds", {})
                        wb = WindowBounds(
                            left=int(b.get("left", 0)),
                            top=int(b.get("top", 0)),
                            right=int(b.get("right", 0)),
                            bottom=int(b.get("bottom", 0)),
                        )
                        c = e.get("center", {})
                        pt = Point(x=int(c.get("x", wb.center.x)), y=int(c.get("y", wb.center.y)))
                        etype_str = str(e.get("element_type", "unknown"))
                        try:
                            etype = UIElementType(etype_str)
                        except Exception:
                            etype = UIElementType.UNKNOWN

                        elements.append(
                            UIElement(
                                name=str(e.get("name", "element")),
                                element_type=etype,
                                bounds=wb,
                                center=pt,
                                confidence=float(e.get("confidence", 1.0)),
                                text_content=e.get("text_content"),
                                metadata=dict(e.get("metadata", {})),
                            )
                        )
                if elements:
                    return elements

        # Synthesize from OCR blocks
        for blk in ocr_blocks:
            t = blk.text.strip()
            if not t:
                continue

            etype = self._classify_element_type(t, blk.bounds, win_bounds)
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

    def _classify_element_type(
        self,
        text: str,
        bounds: WindowBounds,
        win_bounds: WindowBounds,
    ) -> UIElementType:
        """Conservatively classify UIElementType using text cues and geometry."""
        t_clean = text.strip()
        t_lower = t_clean.lower()

        # Buttons: distinct action words with reasonable button-like width
        button_keywords = ("ok", "cancel", "submit", "apply", "save", "close", "delete", "yes", "no", "login", "search", "send")
        if any(t_lower == k or t_lower.startswith(f"{k} ") or t_lower.endswith(f" {k}") for k in button_keywords):
            if bounds.width <= 300 and bounds.height <= 80:
                return UIElementType.BUTTON

        # Checkboxes / Radios
        if t_clean.startswith("[ ]") or t_clean.startswith("[x]") or t_clean.startswith("( )") or t_clean.startswith("(*)"):
            return UIElementType.CHECKBOX

        # Inputs
        if any(k in t_lower for k in ("enter ", "type here", "search...", "username", "password")) and not t_clean.endswith(":"):
            return UIElementType.INPUT

        # Tabs
        if bounds.top <= (win_bounds.height * 0.20) and any(k in t_lower for k in ("tab", "general", "settings", "view", "edit", "file")):
            if bounds.width <= 200:
                return UIElementType.TAB

        # Links
        if t_lower.startswith("http://") or t_lower.startswith("https://") or t_lower.startswith("www."):
            return UIElementType.LINK

        return UIElementType.TEXT

    # -----------------------------------------------------------------------
    # Form Field Binding (Label -> Input)
    # -----------------------------------------------------------------------

    def _associate_form_fields(
        self,
        elements: List[UIElement],
        ocr_blocks: List[OCRTextBlock],
        win_bounds: WindowBounds,
    ) -> List[FormField]:
        """Pair descriptor labels with adjacent interactive inputs deterministically."""
        form_fields: List[FormField] = []

        # Identify candidate labels (e.g. text ending in ':' or explicit field prompts)
        candidate_labels: List[Tuple[str, WindowBounds, bool]] = []
        for blk in ocr_blocks:
            t = blk.text.strip()
            if not t:
                continue
            is_req = False
            if t.endswith("*") or " (required)" in t.lower():
                is_req = True
                t = t.rstrip("*").replace(" (required)", "").strip()

            if t.endswith(":") or any(t.lower() == k for k in ("username", "password", "email", "first name", "last name", "address", "phone", "port", "host", "search")):
                candidate_labels.append((t, blk.bounds, is_req))

        # Identify candidate inputs
        candidate_inputs = [
            e for e in elements
            if e.element_type in (UIElementType.INPUT, UIElementType.CHECKBOX, UIElementType.DROPDOWN, UIElementType.UNKNOWN)
            or "input" in e.name.lower()
        ]

        assigned_inputs: Set[int] = set()

        for label_text, label_bounds, is_req in candidate_labels:
            best_input: Optional[Tuple[int, UIElement, float]] = None

            for idx_in, inp in enumerate(candidate_inputs):
                if idx_in in assigned_inputs:
                    continue

                dx = inp.bounds.left - label_bounds.right
                dy = inp.bounds.top - label_bounds.top

                # Check Right-Side Adjacency
                is_right_adjacent = (
                    0 <= dx <= MAX_LABEL_INPUT_RIGHT_DISTANCE
                    and abs(label_bounds.center.y - inp.bounds.center.y) <= LABEL_ALIGNMENT_TOLERANCE_Y
                )

                # Check Below-Label Adjacency
                dy_below = inp.bounds.top - label_bounds.bottom
                dx_below = abs(inp.bounds.left - label_bounds.left)
                is_below_adjacent = (
                    0 <= dy_below <= MAX_LABEL_INPUT_BELOW_DISTANCE
                    and dx_below <= LABEL_ALIGNMENT_TOLERANCE_X
                )

                if is_right_adjacent or is_below_adjacent:
                    dist = math.hypot(dx if is_right_adjacent else dx_below, dy if is_right_adjacent else dy_below)
                    score = max(0.1, 1.0 - (dist / 300.0))
                    if best_input is None or score > best_input[2]:
                        best_input = (idx_in, inp, score)

            if best_input is not None:
                idx_sel, inp_sel, conf = best_input
                assigned_inputs.add(idx_sel)
                form_fields.append(
                    FormField(
                        field_id=str(uuid.uuid4()),
                        label=label_text,
                        label_bounds=label_bounds,
                        input_element=inp_sel,
                        is_required=is_req,
                        confidence=conf,
                    )
                )

        return form_fields

    # -----------------------------------------------------------------------
    # Semantic Container Inference
    # -----------------------------------------------------------------------

    def _infer_containers(
        self,
        elements: List[UIElement],
        ocr_blocks: List[OCRTextBlock],
        form_fields: List[FormField],
        win_bounds: WindowBounds,
        win_title: str,
    ) -> List[UIContainer]:
        """Segment window elements into semantic structural containers."""
        containers: List[UIContainer] = []
        if not elements and not ocr_blocks:
            return containers

        w_h = max(1, win_bounds.height)
        w_w = max(1, win_bounds.width)

        # 1. Header Container (Top horizontal band)
        header_elements = [
            e for e in elements
            if e.bounds.top <= (win_bounds.top + (w_h * 0.15))
        ]
        if header_elements:
            left_e = min(e.bounds.left for e in header_elements)
            top_e = min(e.bounds.top for e in header_elements)
            right_e = max(e.bounds.right for e in header_elements)
            bottom_e = max(e.bounds.bottom for e in header_elements)
            containers.append(
                UIContainer(
                    container_id=str(uuid.uuid4()),
                    container_type=UIContainerType.HEADER,
                    bounds=WindowBounds(left_e, top_e, right_e, bottom_e),
                    elements=tuple(header_elements),
                    label=win_title or "Header",
                    confidence=0.95,
                )
            )

        # 2. Status Bar / Footer Container (Bottom horizontal band)
        footer_elements = [
            e for e in elements
            if e.bounds.bottom >= (win_bounds.top + (w_h * 0.85))
        ]
        if footer_elements:
            left_e = min(e.bounds.left for e in footer_elements)
            top_e = min(e.bounds.top for e in footer_elements)
            right_e = max(e.bounds.right for e in footer_elements)
            bottom_e = max(e.bounds.bottom for e in footer_elements)
            containers.append(
                UIContainer(
                    container_id=str(uuid.uuid4()),
                    container_type=UIContainerType.STATUS_BAR,
                    bounds=WindowBounds(left_e, top_e, right_e, bottom_e),
                    elements=tuple(footer_elements),
                    label="Status Bar",
                    confidence=0.90,
                )
            )

        # 3. Sidebar Container (Left vertical band)
        sidebar_elements = [
            e for e in elements
            if e.bounds.left <= (win_bounds.left + (w_w * 0.30))
            and (win_bounds.top + (w_h * 0.15)) < e.bounds.top < (win_bounds.top + (w_h * 0.85))
        ]
        if len(sidebar_elements) >= 2:
            left_e = min(e.bounds.left for e in sidebar_elements)
            top_e = min(e.bounds.top for e in sidebar_elements)
            right_e = max(e.bounds.right for e in sidebar_elements)
            bottom_e = max(e.bounds.bottom for e in sidebar_elements)
            containers.append(
                UIContainer(
                    container_id=str(uuid.uuid4()),
                    container_type=UIContainerType.SIDEBAR,
                    bounds=WindowBounds(left_e, top_e, right_e, bottom_e),
                    elements=tuple(sidebar_elements),
                    label="Navigation Sidebar",
                    confidence=0.88,
                )
            )

        # 4. Form Container (Bound Form Fields cluster)
        if form_fields:
            form_elements: List[UIElement] = [f.input_element for f in form_fields]
            left_f = min(f.label_bounds.left for f in form_fields)
            top_f = min(f.label_bounds.top for f in form_fields)
            right_f = max(max(f.label_bounds.right, f.input_element.bounds.right) for f in form_fields)
            bottom_f = max(max(f.label_bounds.bottom, f.input_element.bounds.bottom) for f in form_fields)
            containers.append(
                UIContainer(
                    container_id=str(uuid.uuid4()),
                    container_type=UIContainerType.FORM,
                    bounds=WindowBounds(left_f, top_f, right_f, bottom_f),
                    elements=tuple(form_elements),
                    label="Input Form",
                    confidence=0.92,
                )
            )

        # 5. Dialog / Modal Detection (Isolated popup with OK/Cancel)
        dialog_buttons = [
            e for e in elements
            if e.element_type == UIElementType.BUTTON and any(k in e.name.lower() for k in ("ok", "cancel", "dismiss", "close", "apply"))
        ]
        if len(dialog_buttons) >= 2 and any(k in win_title.lower() for k in ("dialog", "alert", "confirm", "modal", "prompt")):
            left_d = min(e.bounds.left for e in elements)
            top_d = min(e.bounds.top for e in elements)
            right_d = max(e.bounds.right for e in elements)
            bottom_d = max(e.bounds.bottom for e in elements)
            containers.append(
                UIContainer(
                    container_id=str(uuid.uuid4()),
                    container_type=UIContainerType.DIALOG,
                    bounds=WindowBounds(left_d, top_d, right_d, bottom_d),
                    elements=tuple(elements),
                    label=win_title or "Dialog",
                    confidence=0.95,
                )
            )

        # 6. Tab Panel Container
        tab_elements = [e for e in elements if e.element_type == UIElementType.TAB]
        if len(tab_elements) >= 2:
            left_t = min(e.bounds.left for e in tab_elements)
            top_t = min(e.bounds.top for e in tab_elements)
            right_t = max(e.bounds.right for e in tab_elements)
            bottom_t = max(e.bounds.bottom for e in tab_elements)
            containers.append(
                UIContainer(
                    container_id=str(uuid.uuid4()),
                    container_type=UIContainerType.TAB_PANEL,
                    bounds=WindowBounds(left_t, top_t, right_t, bottom_t),
                    elements=tuple(tab_elements),
                    label="Tab Navigation",
                    confidence=0.90,
                )
            )

        # 7. Toolbar Container (Action icons/buttons row near top)
        toolbar_buttons = [
            e for e in elements
            if e.element_type in (UIElementType.BUTTON, UIElementType.ICON)
            and (win_bounds.top + (w_h * 0.08)) <= e.bounds.top <= (win_bounds.top + (w_h * 0.25))
        ]
        if len(toolbar_buttons) >= 3:
            left_tb = min(e.bounds.left for e in toolbar_buttons)
            top_tb = min(e.bounds.top for e in toolbar_buttons)
            right_tb = max(e.bounds.right for e in toolbar_buttons)
            bottom_tb = max(e.bounds.bottom for e in toolbar_buttons)
            containers.append(
                UIContainer(
                    container_id=str(uuid.uuid4()),
                    container_type=UIContainerType.TOOLBAR,
                    bounds=WindowBounds(left_tb, top_tb, right_tb, bottom_tb),
                    elements=tuple(toolbar_buttons),
                    label="Toolbar",
                    confidence=0.88,
                )
            )

        # 8. Content Area (Fallback for central unassigned elements)
        assigned_element_ids = {e.name for c in containers for e in c.elements}
        remaining_elements = [e for e in elements if e.name not in assigned_element_ids]
        if remaining_elements:
            left_c = min(e.bounds.left for e in remaining_elements)
            top_c = min(e.bounds.top for e in remaining_elements)
            right_c = max(e.bounds.right for e in remaining_elements)
            bottom_c = max(e.bounds.bottom for e in remaining_elements)
            containers.append(
                UIContainer(
                    container_id=str(uuid.uuid4()),
                    container_type=UIContainerType.CONTENT_AREA,
                    bounds=WindowBounds(left_c, top_c, right_c, bottom_c),
                    elements=tuple(remaining_elements),
                    label="Main Content",
                    confidence=0.75,
                )
            )

        return containers

    # -----------------------------------------------------------------------
    # Multimodal Scene Fallback & Enhancement
    # -----------------------------------------------------------------------

    def _execute_multimodal_scene_parse(
        self,
        observation: ScreenObservation,
        win_bounds: WindowBounds,
    ) -> Optional[Dict[str, Any]]:
        """Perform multimodal VLM scene layout parsing."""
        ai_p = self.ai_provider
        if ai_p is None or not getattr(ai_p, "supports_multimodal", False):
            return None

        cap = observation.capture
        if cap is None or cap.is_empty:
            return None

        img_part = self._preprocessor.to_image_part(cap)
        payload = self._prompt_builder.build_payload(
            query=_DEFAULT_SCENE_PROMPT,
            images=[img_part],
        )

        response = ai_p.generate(payload)
        content = getattr(response, "content", str(response)).strip()

        # Clean markdown wrappers
        if "```json" in content:
            content = content.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in content:
            content = content.split("```", 1)[1].split("```", 1)[0].strip()

        try:
            data = json.loads(content)
            if isinstance(data, dict):
                return data
        except Exception:
            self._logger.debug("Failed to parse multimodal scene JSON response: %s", content)

        return None

    def _merge_vlm_enhancements(
        self,
        containers: List[UIContainer],
        interactive_elements: List[UIElement],
        vlm_data: Dict[str, Any],
        win_bounds: WindowBounds,
    ) -> Tuple[List[UIContainer], List[UIElement]]:
        """Merge verified VLM container and element detections into parsed scene."""
        w_w = max(1, win_bounds.width)
        w_h = max(1, win_bounds.height)

        # 1. Parse VLM containers
        vlm_containers = vlm_data.get("containers", [])
        if isinstance(vlm_containers, list):
            for vc in vlm_containers:
                if not isinstance(vc, dict):
                    continue
                ctype_str = str(vc.get("type", "content_area")).strip().lower()
                try:
                    ctype = UIContainerType(ctype_str)
                except Exception:
                    ctype = UIContainerType.UNCERTAIN

                box = vc.get("box_2d")
                if isinstance(box, list) and len(box) == 4:
                    ymin, xmin, ymax, xmax = [max(0, min(1000, int(v))) for v in box]
                    wb = WindowBounds(
                        left=win_bounds.left + int((xmin / 1000.0) * w_w),
                        top=win_bounds.top + int((ymin / 1000.0) * w_h),
                        right=win_bounds.left + int((xmax / 1000.0) * w_w),
                        bottom=win_bounds.top + int((ymax / 1000.0) * w_h),
                    )
                    containers.append(
                        UIContainer(
                            container_id=str(uuid.uuid4()),
                            container_type=ctype,
                            bounds=wb,
                            label=vc.get("label"),
                            confidence=float(vc.get("confidence", 0.85)),
                            metadata={"source": "vlm"},
                        )
                    )

        # 2. Parse VLM interactive elements
        vlm_elements = vlm_data.get("interactive_elements", [])
        if isinstance(vlm_elements, list):
            for ve in vlm_elements:
                if not isinstance(ve, dict):
                    continue
                etype_str = str(ve.get("type", "unknown")).strip().lower()
                try:
                    etype = UIElementType(etype_str)
                except Exception:
                    etype = UIElementType.UNKNOWN

                box = ve.get("box_2d")
                if isinstance(box, list) and len(box) == 4:
                    ymin, xmin, ymax, xmax = [max(0, min(1000, int(v))) for v in box]
                    wb = WindowBounds(
                        left=win_bounds.left + int((xmin / 1000.0) * w_w),
                        top=win_bounds.top + int((ymin / 1000.0) * w_h),
                        right=win_bounds.left + int((xmax / 1000.0) * w_w),
                        bottom=win_bounds.top + int((ymax / 1000.0) * w_h),
                    )
                    interactive_elements.append(
                        UIElement(
                            name=str(ve.get("name", "element")),
                            element_type=etype,
                            bounds=wb,
                            center=wb.center,
                            confidence=float(ve.get("confidence", 0.85)),
                            source=GroundingSource.MULTIMODAL_SEMANTIC,
                        )
                    )

        return containers, interactive_elements

    # -----------------------------------------------------------------------
    # Summary Synthesis
    # -----------------------------------------------------------------------

    def _synthesize_scene_summary(
        self,
        win_title: str,
        containers: List[UIContainer],
        interactive_elements: List[UIElement],
        form_fields: List[FormField],
    ) -> str:
        """Synthesize natural-language overview of the UI layout and controls."""
        if not containers and not interactive_elements and not form_fields:
            return f"The window '{win_title or 'Application'}' contains no distinct interactive controls."

        parts: List[str] = []
        app_name = f"'{win_title}'" if win_title else "The application window"

        # Controls summary
        button_count = sum(1 for e in interactive_elements if e.element_type == UIElementType.BUTTON)
        input_count = sum(1 for e in interactive_elements if e.element_type == UIElementType.INPUT)
        tab_count = sum(1 for e in interactive_elements if e.element_type == UIElementType.TAB)
        link_count = sum(1 for e in interactive_elements if e.element_type == UIElementType.LINK)

        ctrl_desc = []
        if button_count:
            ctrl_desc.append(f"{button_count} button{'s' if button_count > 1 else ''}")
        if input_count:
            ctrl_desc.append(f"{input_count} input field{'s' if input_count > 1 else ''}")
        if tab_count:
            ctrl_desc.append(f"{tab_count} tab{'s' if tab_count > 1 else ''}")
        if link_count:
            ctrl_desc.append(f"{link_count} link{'s' if link_count > 1 else ''}")

        if ctrl_desc:
            parts.append(f"Contains {len(interactive_elements)} interactive control{'s' if len(interactive_elements) > 1 else ''} ({', '.join(ctrl_desc)})")

        if form_fields:
            field_labels = ", ".join(f"'{f.label}'" for f in form_fields[:3])
            parts.append(f"{len(form_fields)} form field{'s' if len(form_fields) > 1 else ''} ({field_labels})")

        container_types = list({c.container_type.value for c in containers})
        if container_types:
            parts.append(f"Layout regions: {', '.join(container_types)}")

        return f"{app_name}: {'; '.join(parts)}."


__all__ = [
    "MAX_LABEL_INPUT_BELOW_DISTANCE",
    "MAX_LABEL_INPUT_RIGHT_DISTANCE",
    "VisualSceneParser",
]
