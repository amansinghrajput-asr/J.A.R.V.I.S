"""Phase 27.9 Relative Visual Grounding & Spatial Relationship Reasoning Tests.

Comprehensive deterministic unit and integration test suite covering:
A. SpatialRelation Model & Parsing
B. LEFT_OF Geometry & Scoring
C. RIGHT_OF Geometry & Scoring
D. ABOVE Geometry & Scoring
E. BELOW Geometry & Scoring
F. NEAR Geometry & Scoring
G. INSIDE Geometry & Scoring
H. Perpendicular Overlap & Alignment Scoring
I. Distance Decay Scoring
J. Ambiguity Handling (multiple equivalent candidates -> UNCERTAIN)
K. Ambiguous Anchor Handling (ambiguous reference -> UNCERTAIN)
L. Malformed Geometry Rejection
M. OCR Anchor + OCR Target Resolution
N. OCR Anchor + Multimodal Target Resolution
O. Multimodal Fallback
P. OCR + Multimodal Candidate Spatial Fusion
Q. VisionSkills Command Parsing (_RE_RELATIONAL_LOCATE)
R. VisionSkills End-to-End Execution Flow
S. Security Enforcement on Sensitive Windows
T. Ephemeral Observation Lifecycle & Reuse
U. Voice-Safe Relational Result Formatting
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
from unittest import mock

import pytest

from app.ai.models import UnsupportedModalityError
from app.ai.planner.planner import KNOWN_ACTIONS
from app.core.container import ServiceContainer
from app.skills.base import SkillExecutionError
from app.skills.system.base_system_skill import BaseSystemSkill
from app.skills.system.vision_skills import VisionSkills
from app.vision.grounding import (
    AMBIGUITY_SCORE_DELTA_THRESHOLD,
    MIN_RELATION_SCORE_THRESHOLD,
    VisualGroundingEngine,
)
from app.vision.models import (
    GroundingSource,
    OCRResult,
    OCRTextBlock,
    Point,
    ScreenCapture,
    ScreenObservation,
    SpatialRelation,
    UIElement,
    UIElementType,
    VisualGroundingResult,
    WindowBounds,
)
from app.vision.ocr import MockOCRProvider
from app.vision.security import (
    EphemeralBufferManager,
    SecureVisionManager,
    VisionSecurityPolicy,
)


# ---------------------------------------------------------------------------
# Test Helpers & Fixtures
# ---------------------------------------------------------------------------


def _create_synthetic_capture(
    width: int = 1000,
    height: int = 800,
    window_title: str = "Test App Window",
    process_name: str = "testapp.exe",
    hwnd: int = 12345,
    bounds: Optional[WindowBounds] = None,
) -> ScreenCapture:
    """Create synthetic in-memory ScreenCapture for testing."""
    raw = bytes([128] * (width * height * 4))
    b = bounds or WindowBounds(left=100, top=100, right=100 + width, bottom=100 + height)
    return ScreenCapture(
        raw_data=raw,
        width=width,
        height=height,
        channels=4,
        pixel_format="BGRA",
        bounds=b,
        timestamp=time.time(),
        source="window",
        metadata={
            "window_title": window_title,
            "process_name": process_name,
            "hwnd": hwnd,
        },
    )


def _create_synthetic_observation(
    width: int = 1000,
    height: int = 800,
    window_title: str = "Test Application",
    process_name: str = "testapp.exe",
    hwnd: int = 12345,
    ttl_seconds: float = 30.0,
    capture: Optional[ScreenCapture] = None,
) -> ScreenObservation:
    """Create valid authorized ScreenObservation."""
    cap = capture or _create_synthetic_capture(
        width=width,
        height=height,
        window_title=window_title,
        process_name=process_name,
        hwnd=hwnd,
    )
    from app.vision.models import CaptureAuthorization, CaptureCategory, CaptureDecision

    auth = CaptureAuthorization(
        decision=CaptureDecision.ALLOW,
        category=CaptureCategory.SAFE,
        reason="Safe test window",
    )
    return ScreenObservation(
        capture=cap,
        authorization=auth,
        source="window",
        timestamp=time.time(),
        expires_at=time.time() + ttl_seconds,
        metadata={
            "window_title": window_title,
            "process_name": process_name,
            "hwnd": hwnd,
        },
    )


class MockMultimodalAIProvider:
    """Deterministic Mock AI Provider for multimodal vision responses."""

    def __init__(self, canned_response: Optional[Dict[str, Any]] = None) -> None:
        self.canned_response = canned_response or {"found": False, "confidence": 0.0}
        self.call_count = 0
        self.supports_multimodal = True
        self.last_prompt = ""

    async def generate_content_async(self, prompt: str, images: Optional[List[Any]] = None) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        return json.dumps(self.canned_response)

    def generate_content(self, prompt: str, images: Optional[List[Any]] = None) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        return json.dumps(self.canned_response)


# ---------------------------------------------------------------------------
# Group A: SpatialRelation Model & Parsing
# ---------------------------------------------------------------------------


def test_spatial_relation_model_values_and_parsing():
    """Verify SpatialRelation enum values and parser string mapping."""
    engine = VisualGroundingEngine()

    assert SpatialRelation.LEFT_OF.value == "left_of"
    assert SpatialRelation.RIGHT_OF.value == "right_of"
    assert SpatialRelation.ABOVE.value == "above"
    assert SpatialRelation.BELOW.value == "below"
    assert SpatialRelation.NEAR.value == "near"
    assert SpatialRelation.INSIDE.value == "inside"

    assert engine._parse_spatial_relation("left_of") == SpatialRelation.LEFT_OF
    assert engine._parse_spatial_relation("to the left of") == SpatialRelation.LEFT_OF
    assert engine._parse_spatial_relation("right_of") == SpatialRelation.RIGHT_OF
    assert engine._parse_spatial_relation("to the right of") == SpatialRelation.RIGHT_OF
    assert engine._parse_spatial_relation("above") == SpatialRelation.ABOVE
    assert engine._parse_spatial_relation("on top of") == SpatialRelation.ABOVE
    assert engine._parse_spatial_relation("below") == SpatialRelation.BELOW
    assert engine._parse_spatial_relation("under") == SpatialRelation.BELOW
    assert engine._parse_spatial_relation("near") == SpatialRelation.NEAR
    assert engine._parse_spatial_relation("next to") == SpatialRelation.NEAR
    assert engine._parse_spatial_relation("beside") == SpatialRelation.NEAR
    assert engine._parse_spatial_relation("inside") == SpatialRelation.INSIDE
    assert engine._parse_spatial_relation("within") == SpatialRelation.INSIDE
    assert engine._parse_spatial_relation("invalid_relation") is None


# ---------------------------------------------------------------------------
# Group B–G: Directional & Topological Spatial Geometry Scoring
# ---------------------------------------------------------------------------


def test_left_of_geometry_scoring():
    """Verify LEFT_OF assigns high scores to left-aligned elements and 0.0 to right/overlapping ones."""
    engine = VisualGroundingEngine()
    anchor_bounds = WindowBounds(left=300, top=200, right=500, bottom=240)
    anchor = UIElement(
        name="Anchor",
        element_type=UIElementType.TEXT,
        bounds=anchor_bounds,
        center=anchor_bounds.center,
        confidence=1.0,
        source=GroundingSource.OCR_EXACT,
    )

    # Valid candidate strictly to the left, perfectly vertically aligned
    cand_left_bounds = WindowBounds(left=250, top=205, right=280, bottom=235)
    cand_left = UIElement(
        name="Checkbox",
        element_type=UIElementType.CHECKBOX,
        bounds=cand_left_bounds,
        center=cand_left_bounds.center,
        confidence=1.0,
        source=GroundingSource.OCR_EXACT,
    )
    score_left = engine._score_spatial_relation(cand_left, anchor, SpatialRelation.LEFT_OF)
    assert score_left > 0.85

    # Invalid candidate to the right
    cand_right_bounds = WindowBounds(left=520, top=205, right=560, bottom=235)
    cand_right = UIElement(
        name="RightBtn",
        element_type=UIElementType.BUTTON,
        bounds=cand_right_bounds,
        center=cand_right_bounds.center,
        confidence=1.0,
        source=GroundingSource.OCR_EXACT,
    )
    score_right = engine._score_spatial_relation(cand_right, anchor, SpatialRelation.LEFT_OF)
    assert score_right == 0.0


def test_right_of_geometry_scoring():
    """Verify RIGHT_OF assigns high scores to right-aligned elements and 0.0 to left ones."""
    engine = VisualGroundingEngine()
    anchor_bounds = WindowBounds(left=200, top=100, right=350, bottom=140)
    anchor = UIElement(
        name="SearchInput",
        element_type=UIElementType.INPUT,
        bounds=anchor_bounds,
        center=anchor_bounds.center,
        confidence=1.0,
        source=GroundingSource.OCR_EXACT,
    )

    # Valid candidate to the right
    cand_right_bounds = WindowBounds(left=360, top=105, right=400, bottom=135)
    cand_right = UIElement(
        name="SearchIcon",
        element_type=UIElementType.ICON,
        bounds=cand_right_bounds,
        center=cand_right_bounds.center,
        confidence=1.0,
        source=GroundingSource.OCR_EXACT,
    )
    score_right = engine._score_spatial_relation(cand_right, anchor, SpatialRelation.RIGHT_OF)
    assert score_right > 0.85

    # Candidate to the left
    cand_left_bounds = WindowBounds(left=150, top=105, right=190, bottom=135)
    cand_left = UIElement(
        name="LeftIcon",
        element_type=UIElementType.ICON,
        bounds=cand_left_bounds,
        center=cand_left_bounds.center,
        confidence=1.0,
        source=GroundingSource.OCR_EXACT,
    )
    score_left = engine._score_spatial_relation(cand_left, anchor, SpatialRelation.RIGHT_OF)
    assert score_left == 0.0


def test_above_and_below_geometry_scoring():
    """Verify ABOVE and BELOW directional geometry and horizontal perpendicular overlap."""
    engine = VisualGroundingEngine()
    anchor_bounds = WindowBounds(left=200, top=300, right=400, bottom=340)
    anchor = UIElement(
        name="PasswordLabel",
        element_type=UIElementType.TEXT,
        bounds=anchor_bounds,
        center=anchor_bounds.center,
        confidence=1.0,
        source=GroundingSource.OCR_EXACT,
    )

    # Candidate strictly below with high horizontal overlap
    cand_below_bounds = WindowBounds(left=200, top=350, right=400, bottom=390)
    cand_below = UIElement(
        name="PasswordInput",
        element_type=UIElementType.INPUT,
        bounds=cand_below_bounds,
        center=cand_below_bounds.center,
        confidence=1.0,
        source=GroundingSource.OCR_EXACT,
    )
    score_below = engine._score_spatial_relation(cand_below, anchor, SpatialRelation.BELOW)
    assert score_below > 0.85
    assert engine._score_spatial_relation(cand_below, anchor, SpatialRelation.ABOVE) == 0.0

    # Candidate strictly above
    cand_above_bounds = WindowBounds(left=200, top=240, right=400, bottom=280)
    cand_above = UIElement(
        name="UsernameInput",
        element_type=UIElementType.INPUT,
        bounds=cand_above_bounds,
        center=cand_above_bounds.center,
        confidence=1.0,
        source=GroundingSource.OCR_EXACT,
    )
    score_above = engine._score_spatial_relation(cand_above, anchor, SpatialRelation.ABOVE)
    assert score_above > 0.85
    assert engine._score_spatial_relation(cand_above, anchor, SpatialRelation.BELOW) == 0.0


def test_inside_and_near_geometry_scoring():
    """Verify INSIDE containment and NEAR proximity distance metrics."""
    engine = VisualGroundingEngine()
    dialog_bounds = WindowBounds(left=100, top=100, right=600, bottom=500)
    dialog = UIElement(
        name="ConfirmDialog",
        element_type=UIElementType.DIALOG,
        bounds=dialog_bounds,
        center=dialog_bounds.center,
        confidence=1.0,
        source=GroundingSource.OCR_EXACT,
    )

    # Candidate fully inside the dialog
    btn_inside_bounds = WindowBounds(left=450, top=420, right=550, bottom=460)
    btn_inside = UIElement(
        name="OKButton",
        element_type=UIElementType.BUTTON,
        bounds=btn_inside_bounds,
        center=btn_inside_bounds.center,
        confidence=1.0,
        source=GroundingSource.OCR_EXACT,
    )
    score_inside = engine._score_spatial_relation(btn_inside, dialog, SpatialRelation.INSIDE)
    assert score_inside >= 0.95

    # Candidate completely outside the dialog
    btn_outside_bounds = WindowBounds(left=700, top=100, right=800, bottom=140)
    btn_outside = UIElement(
        name="OutsideBtn",
        element_type=UIElementType.BUTTON,
        bounds=btn_outside_bounds,
        center=btn_outside_bounds.center,
        confidence=1.0,
        source=GroundingSource.OCR_EXACT,
    )
    score_outside = engine._score_spatial_relation(btn_outside, dialog, SpatialRelation.INSIDE)
    assert score_outside == 0.0

    # NEAR proximity check
    btn_near_bounds = WindowBounds(left=610, top=110, right=680, bottom=140)
    btn_near = UIElement(
        name="NearBtn",
        element_type=UIElementType.BUTTON,
        bounds=btn_near_bounds,
        center=btn_near_bounds.center,
        confidence=1.0,
        source=GroundingSource.OCR_EXACT,
    )
    score_near = engine._score_spatial_relation(btn_near, dialog, SpatialRelation.NEAR)
    assert score_near > 0.60


# ---------------------------------------------------------------------------
# Group H–I: Overlap vs. Misaligned & Distance Decay
# ---------------------------------------------------------------------------


def test_perpendicular_overlap_ranks_aligned_above_misaligned():
    """Verify candidate with perpendicular overlap beats misaligned candidate at same distance."""
    engine = VisualGroundingEngine()
    anchor_bounds = WindowBounds(left=400, top=300, right=600, bottom=340)
    anchor = UIElement(name="Label", element_type=UIElementType.TEXT, bounds=anchor_bounds, center=anchor_bounds.center, confidence=1.0, source=GroundingSource.OCR_EXACT)

    # Candidate 1: 50px left, perfectly vertically aligned (Y: 300..340)
    cand_aligned_bounds = WindowBounds(left=310, top=300, right=350, bottom=340)
    cand_aligned = UIElement(name="C1", element_type=UIElementType.BUTTON, bounds=cand_aligned_bounds, center=cand_aligned_bounds.center, confidence=1.0, source=GroundingSource.OCR_EXACT)

    # Candidate 2: 50px left, but vertically shifted far away (Y: 600..640)
    cand_misaligned_bounds = WindowBounds(left=310, top=600, right=350, bottom=640)
    cand_misaligned = UIElement(name="C2", element_type=UIElementType.BUTTON, bounds=cand_misaligned_bounds, center=cand_misaligned_bounds.center, confidence=1.0, source=GroundingSource.OCR_EXACT)

    score_aligned = engine._score_spatial_relation(cand_aligned, anchor, SpatialRelation.LEFT_OF)
    score_misaligned = engine._score_spatial_relation(cand_misaligned, anchor, SpatialRelation.LEFT_OF)

    assert score_aligned > score_misaligned
    assert score_aligned - score_misaligned > 0.30


def test_distance_decay_scoring():
    """Verify nearer candidate in the same direction beats farther candidate."""
    engine = VisualGroundingEngine()
    anchor_bounds = WindowBounds(left=500, top=300, right=600, bottom=340)
    anchor = UIElement(name="Anchor", element_type=UIElementType.TEXT, bounds=anchor_bounds, center=anchor_bounds.center, confidence=1.0, source=GroundingSource.OCR_EXACT)

    # Nearby candidate (20px left)
    cand_near_bounds = WindowBounds(left=440, top=300, right=480, bottom=340)
    cand_near = UIElement(name="Near", element_type=UIElementType.BUTTON, bounds=cand_near_bounds, center=cand_near_bounds.center, confidence=1.0, source=GroundingSource.OCR_EXACT)

    # Far candidate (300px left)
    cand_far_bounds = WindowBounds(left=100, top=300, right=140, bottom=340)
    cand_far = UIElement(name="Far", element_type=UIElementType.BUTTON, bounds=cand_far_bounds, center=cand_far_bounds.center, confidence=1.0, source=GroundingSource.OCR_EXACT)

    score_near = engine._score_spatial_relation(cand_near, anchor, SpatialRelation.LEFT_OF)
    score_far = engine._score_spatial_relation(cand_far, anchor, SpatialRelation.LEFT_OF)

    assert score_near > score_far
    assert score_near > 0.80
    assert score_far < 0.50


# ---------------------------------------------------------------------------
# Group J–L: Ambiguity & Malformed Geometry Rejection
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_ambiguous_candidates_returns_uncertain():
    """Verify two equidistant, equally aligned candidates return is_found=False with ambiguous metadata."""
    engine = VisualGroundingEngine()
    obs = _create_synthetic_observation()

    # Anchor at center
    b_anchor = WindowBounds(left=400, top=300, right=600, bottom=340)
    # Candidate 1 and Candidate 2 placed symmetrically on the left with identical vertical positions
    b_c1 = WindowBounds(left=300, top=280, right=340, bottom=320)
    b_c2 = WindowBounds(left=300, top=320, right=340, bottom=360)

    blocks = (
        OCRTextBlock(text="Submit Form", bounds=b_anchor),
        OCRTextBlock(text="Option", bounds=b_c1),
        OCRTextBlock(text="Option", bounds=b_c2),
    )
    ocr_res = OCRResult(text="text", blocks=blocks)

    result = await engine.locate_relative_element_async(
        target="Option",
        relation="left_of",
        reference_target="Submit Form",
        observation=obs,
        ocr_result=ocr_res,
    )

    assert result.is_found is False
    assert result.metadata.get("ambiguous") is True
    assert "ambiguous" in result.summary.lower()


@pytest.mark.anyio
async def test_ambiguous_anchor_returns_uncertain():
    """Verify ambiguous reference anchor matching multiple exact blocks aborts relative grounding cleanly."""
    engine = VisualGroundingEngine()
    obs = _create_synthetic_observation()

    # Two identical anchors
    b_a1 = WindowBounds(left=100, top=100, right=200, bottom=140)
    b_a2 = WindowBounds(left=400, top=400, right=500, bottom=440)
    b_c = WindowBounds(left=50, top=100, right=90, bottom=140)

    blocks = (
        OCRTextBlock(text="Save", bounds=b_a1),
        OCRTextBlock(text="Save", bounds=b_a2),
        OCRTextBlock(text="Icon", bounds=b_c),
    )
    ocr_res = OCRResult(text="Save Save Icon", blocks=blocks)

    result = await engine.locate_relative_element_async(
        target="Icon",
        relation="left_of",
        reference_target="Save",
        observation=obs,
        ocr_result=ocr_res,
    )

    assert result.is_found is False
    assert result.metadata.get("anchor_found") is False


def test_malformed_geometry_rejection():
    """Verify empty or invalid bounds yield 0.0 score."""
    engine = VisualGroundingEngine()
    valid_box = WindowBounds(left=100, top=100, right=200, bottom=200)
    empty_box = WindowBounds(left=0, top=0, right=0, bottom=0)

    el_valid = UIElement(name="Valid", element_type=UIElementType.TEXT, bounds=valid_box, center=valid_box.center, confidence=1.0, source=GroundingSource.OCR_EXACT)
    el_empty = UIElement(name="Empty", element_type=UIElementType.TEXT, bounds=empty_box, center=empty_box.center, confidence=1.0, source=GroundingSource.OCR_EXACT)

    assert engine._score_spatial_relation(el_empty, el_valid, SpatialRelation.LEFT_OF) == 0.0
    assert engine._score_spatial_relation(el_valid, el_empty, SpatialRelation.LEFT_OF) == 0.0


# ---------------------------------------------------------------------------
# Group M–P: End-to-End Relative Grounding (OCR, MM, Fusion)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_relative_locate_ocr_anchor_and_ocr_target():
    """Verify locating target button relative to anchor text using pure OCR path."""
    engine = VisualGroundingEngine()
    obs = _create_synthetic_observation()

    b_anchor = WindowBounds(left=400, top=200, right=550, bottom=240)
    b_target = WindowBounds(left=340, top=205, right=380, bottom=235)

    blocks = (
        OCRTextBlock(text="Remember Me", bounds=b_anchor),
        OCRTextBlock(text="Checkbox", bounds=b_target),
    )
    ocr_res = OCRResult(text="Remember Me Checkbox", blocks=blocks)

    result = await engine.locate_relative_element_async(
        target="Checkbox",
        relation="left_of",
        reference_target="Remember Me",
        observation=obs,
        ocr_result=ocr_res,
    )

    assert result.is_found is True
    assert result.element is not None
    assert result.element.name == "Checkbox"
    assert result.element.bounds == b_target
    assert result.metadata["relation"] == "left_of"
    assert result.metadata["reference_target"] == "Remember Me"
    assert "left of 'Remember Me'" in result.summary


@pytest.mark.anyio
async def test_relative_locate_multimodal_target_with_ocr_anchor():
    """Verify locating non-textual icon/checkbox via Multimodal when anchor is found via OCR."""
    canned_mm = {
        "found": True,
        "element_type": "checkbox",
        "box_2d": [200, 320, 240, 360],  # ymin=200, xmin=320, ymax=240, xmax=360 in 0..1000
        "confidence": 0.94,
        "label": "Remember Me Checkbox",
    }
    mock_ai = MockMultimodalAIProvider(canned_response=canned_mm)
    engine = VisualGroundingEngine(ai_provider=mock_ai)
    obs = _create_synthetic_observation()

    # Anchor at pixel space (400, 160) to (600, 192) in 1000x800 capture
    b_anchor = WindowBounds(left=400, top=160, right=600, bottom=192)
    blocks = (OCRTextBlock(text="Remember Me", bounds=b_anchor),)
    ocr_res = OCRResult(text="Remember Me", blocks=blocks)

    result = await engine.locate_relative_element_async(
        target="checkbox",
        relation="left_of",
        reference_target="Remember Me",
        observation=obs,
        ocr_result=ocr_res,
    )

    assert result.is_found is True
    assert result.element is not None
    assert result.element.element_type == UIElementType.CHECKBOX
    assert result.element.source == GroundingSource.MULTIMODAL_SEMANTIC
    assert result.confidence > 0.80


@pytest.mark.anyio
async def test_relative_locate_spatial_fusion_with_ocr_candidate():
    """Verify multimodal candidate box overlapping an OCR text block is fused into HYBRID_FUSED."""
    canned_mm = {
        "found": True,
        "element_type": "button",
        "box_2d": [200, 200, 240, 300],  # 0..1000 normalized -> [160, 200, 192, 300] in 1000x800
        "confidence": 0.90,
        "label": "Cancel",
    }
    mock_ai = MockMultimodalAIProvider(canned_response=canned_mm)
    engine = VisualGroundingEngine(ai_provider=mock_ai)
    obs = _create_synthetic_observation()

    b_anchor = WindowBounds(left=400, top=160, right=500, bottom=192)
    b_cancel_text = WindowBounds(left=210, top=165, right=290, bottom=185)

    blocks = (
        OCRTextBlock(text="OK", bounds=b_anchor),
        OCRTextBlock(text="Cancel", bounds=b_cancel_text),
    )
    ocr_res = OCRResult(text="OK Cancel", blocks=blocks)

    result = await engine.locate_relative_element_async(
        target="button",
        relation="left_of",
        reference_target="OK",
        observation=obs,
        ocr_result=ocr_res,
        force_multimodal=True,
    )

    assert result.is_found is True
    assert result.element is not None
    assert result.element.source == GroundingSource.HYBRID_FUSED
    assert "Cancel" in result.element.name


# ---------------------------------------------------------------------------
# Group Q–U: VisionSkills, Security, Lifecycle, & Voice Formatting
# ---------------------------------------------------------------------------


def test_vision_skills_can_handle_and_parse_relational_queries():
    """Verify natural language patterns for relational locate_element commands."""
    skill = VisionSkills()

    queries = [
        ("find the checkbox to the left of Remember Me", "checkbox", "left_of", "Remember Me"),
        ("where is the field below Username?", "field", "below", "Username"),
        ("find the icon to the right of Search", "icon", "right_of", "Search"),
        ("locate the button above Cancel", "button", "above", "Cancel"),
        ("find the close icon inside the dialog", "close icon", "inside", "dialog"),
        ("which control is near Submit?", "control", "near", "Submit"),
        ("find the button next to Cancel", "button", "near", "Cancel"),
    ]

    for q, expected_tgt, expected_rel, expected_ref in queries:
        assert skill.can_handle(q) is True, f"Failed can_handle for '{q}'"
        op, target, params, _ = skill.parse_command(q)
        assert op == "locate_element"
        assert target == "active_window"
        assert params["target"].lower() == expected_tgt.lower()
        assert params["relation"] == expected_rel
        assert params["reference_target"].lower() == expected_ref.lower()


def test_vision_skills_relative_locate_execution_flow():
    """Verify end-to-end execution of VisionSkills.execute for relational localization."""
    obs = _create_synthetic_observation()
    blocks = (
        OCRTextBlock(text="Remember Me", bounds=WindowBounds(left=400, top=200, right=550, bottom=240)),
        OCRTextBlock(text="Checkbox", bounds=WindowBounds(left=340, top=205, right=380, bottom=235)),
    )
    sec_mgr = mock.MagicMock()
    sec_mgr.capture_active_window.return_value = obs
    sec_mgr.get_latest_observation.return_value = obs

    skill = VisionSkills(
        secure_vision_manager=sec_mgr,
        ocr_provider=MockOCRProvider(canned_blocks=blocks),
    )

    res = skill.execute({
        "operation": "locate_element",
        "parameters": {
            "target": "Checkbox",
            "relation": "left_of",
            "reference_target": "Remember Me",
        },
    })

    assert res.success is True
    assert res.data["is_found"] is True
    assert res.data["relation"] == "left_of"
    assert res.data["reference_target"] == "Remember Me"
    assert res.data["element"]["bounds"]["left"] == 340

    # Test BaseSystemSkill voice message format
    voice_msg = res.message
    assert "left of 'Remember Me'" in voice_msg


def test_relative_locate_blocked_on_sensitive_window():
    """Verify sensitive window blocks relative element localization without revealing coordinates."""
    sec_policy = VisionSecurityPolicy()
    buffer_mgr = EphemeralBufferManager()
    sec_mgr = SecureVisionManager(policy=sec_policy, buffer_manager=buffer_mgr, auto_register_in_container=False)

    mock_cap = _create_synthetic_capture(window_title="Bitwarden - Password Vault", process_name="bitwarden.exe")
    sec_mgr._engine = mock.MagicMock()
    sec_mgr._engine.get_active_window_handle.return_value = 9999
    sec_mgr._engine.capture_active_window.return_value = mock_cap
    sec_mgr._resolve_window_metadata = mock.MagicMock(
        return_value=("Bitwarden - Password Vault", "bitwarden.exe")
    )

    skill = VisionSkills(secure_vision_manager=sec_mgr)

    with pytest.raises(Exception) as exc_info:
        skill.execute({
            "operation": "locate_element",
            "parameters": {
                "target": "checkbox",
                "relation": "left_of",
                "reference_target": "Master Password",
            },
        })

    assert "blocked" in str(exc_info.value).lower() or "security" in str(exc_info.value).lower() or "password" in str(exc_info.value).lower()
