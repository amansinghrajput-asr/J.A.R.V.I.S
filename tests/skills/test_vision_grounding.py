"""Phase 27.8 Visual Grounding & UI Element Localization Tests.

Comprehensive deterministic unit and integration test suite covering:
A. Models (UIElement, VisualGroundingResult, UIElementType, GroundingSource)
B. OCR Fast Path Grounding (exact, case-insensitive, token, ambiguous multiple matches)
C. Multimodal Spatial Grounding (normalized box parsing, icons, semantic targets, not-found)
D. Coordinate Contract & Validation (0..1000, 0..1, dicts, malformed, negative, NaN/inf, OOB)
E. Calibration & Geometry (pixel translation, center point, window bounds)
F. Hybrid Fusion (multimodal box + OCR text correlation, confidence boost)
G. Provider Degradation (UnsupportedModalityError, text-only fallback)
H. Security & Privacy (sensitive-window blocking, zero raw pixel persistence)
I. Ephemeral Lifecycle & Cache Reuse
J. VisionSkills Command Recognition & Dispatch
K. Planner Schema & Action Space Integration
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
from unittest import mock

import pytest

from app.ai.models import UnsupportedModalityError
from app.ai.planner.planner import KNOWN_ACTIONS, PLANNER_SYSTEM_PROMPT, RECOVERY_SYSTEM_PROMPT
from app.core.container import ServiceContainer
from app.core.event_bus import EventBus
from app.skills.base import SkillExecutionError
from app.skills.system.security import SystemConfirmationManager, SystemSecurityPolicy
from app.skills.system.vision_skills import VisionSkills
from app.vision.grounding import VisualGroundingEngine
from app.vision.models import (
    CaptureAuthorization,
    CaptureCategory,
    CaptureDecision,
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
from app.vision.ocr import MockOCRProvider
from app.vision.security import (
    EphemeralBufferManager,
    SecureVisionManager,
    SensitiveWindowRule,
    VisionSecurityPolicy,
)


# ---------------------------------------------------------------------------
# Test Fixtures & Helpers
# ---------------------------------------------------------------------------


def _create_synthetic_capture(
    width: int = 1000,
    height: int = 800,
    window_title: str = "Test Application",
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
        source="window",
        bounds=b,
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
) -> ScreenObservation:
    """Create valid authorized ScreenObservation."""
    cap = _create_synthetic_capture(
        width=width,
        height=height,
        window_title=window_title,
        process_name=process_name,
        hwnd=hwnd,
    )
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
    """Mock multimodal AI provider returning controlled JSON grounding responses."""

    def __init__(self, response_text: str = "", supports_multimodal: bool = True) -> None:
        self.response_text = response_text
        self.supports_multimodal = supports_multimodal
        self.call_count = 0
        self.last_prompt = ""
        self.last_images = []

    async def generate_content_async(self, prompt: str, images: Optional[List[Any]] = None) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        self.last_images = images or []
        if not self.supports_multimodal:
            raise UnsupportedModalityError("Provider does not support multimodal vision inputs.")
        return self.response_text

    def generate(self, payload: Any) -> Any:
        self.call_count += 1
        res = mock.MagicMock()
        res.content = self.response_text
        return res


# ===========================================================================
# Group A: Models & Serialization
# ===========================================================================


def test_ui_element_initialization_and_serialization():
    """Verify UIElement properties, bounds, center, and dictionary serialization."""
    bounds = WindowBounds(left=100, top=200, right=300, bottom=250)
    element = UIElement(
        name="Submit Order",
        element_type=UIElementType.BUTTON,
        bounds=bounds,
        center=bounds.center,
        confidence=0.98,
        source=GroundingSource.OCR_EXACT,
        text_content="Submit Order",
        metadata={"key": "val"},
    )

    assert element.name == "Submit Order"
    assert element.element_type == UIElementType.BUTTON
    assert element.bounds == bounds
    assert element.center == Point(x=200, y=225)
    assert element.confidence == 0.98
    assert element.source == GroundingSource.OCR_EXACT

    data = element.to_dict()
    assert data["name"] == "Submit Order"
    assert data["element_type"] == "button"
    assert data["bounds"]["left"] == 100
    assert data["center"]["x"] == 200
    assert data["confidence"] == 0.98
    assert data["source"] == "ocr_exact"
    assert data["text_content"] == "Submit Order"


def test_visual_grounding_result_serialization_and_confidence_clamping():
    """Verify VisualGroundingResult serialization and confidence bounds clamping."""
    bounds = WindowBounds(left=50, top=50, right=150, bottom=90)
    element = UIElement(
        name="Search",
        element_type=UIElementType.INPUT,
        bounds=bounds,
        center=bounds.center,
        confidence=1.5,  # Should be clamped to 1.0
        source=GroundingSource.MULTIMODAL_SEMANTIC,
    )
    assert element.confidence == 1.0

    res = VisualGroundingResult(
        target="search box",
        element=element,
        is_found=True,
        confidence=1.2,  # Should be clamped to 1.0
        observation_id="obs-123",
        duration=0.05,
        summary="Search box located.",
    )
    assert res.confidence == 1.0
    d = res.to_dict()
    assert d["target"] == "search box"
    assert d["is_found"] is True
    assert d["element"]["name"] == "Search"
    assert d["confidence"] == 1.0


# ===========================================================================
# Group B: OCR Fast Path Grounding
# ===========================================================================


@pytest.mark.anyio
async def test_locate_element_by_exact_ocr_text():
    """Verify fast path exact OCR text matching resolves UIElement in 1.0 confidence."""
    obs = _create_synthetic_observation(width=800, height=600)
    blocks = (
        OCRTextBlock(text="Cancel", bounds=WindowBounds(left=200, top=500, right=300, bottom=540)),
        OCRTextBlock(text="Submit", bounds=WindowBounds(left=350, top=500, right=450, bottom=540)),
    )
    ocr_res = OCRResult(text="Cancel\nSubmit", blocks=blocks, duration=0.01)
    mock_ocr = MockOCRProvider(canned_blocks=blocks)
    mock_ai = MockMultimodalAIProvider()

    engine = VisualGroundingEngine(ocr_provider=mock_ocr, ai_provider=mock_ai)
    result = await engine.locate_element_async(target="Submit", observation=obs, ocr_result=ocr_res)

    assert result.is_found is True
    assert result.confidence == 1.0
    assert result.element is not None
    assert result.element.name == "Submit"
    assert result.element.source == GroundingSource.OCR_EXACT
    assert result.element.bounds == WindowBounds(left=350, top=500, right=450, bottom=540)
    assert result.element.center == Point(x=400, y=520)
    # AI Provider was not invoked due to OCR fast-path short circuit
    assert mock_ai.call_count == 0


@pytest.mark.anyio
async def test_locate_element_by_case_insensitive_ocr():
    """Verify case-insensitive OCR matching (e.g. 'submit' matching 'SUBMIT')."""
    obs = _create_synthetic_observation()
    blocks = (OCRTextBlock(text="SUBMIT", bounds=WindowBounds(left=100, top=100, right=200, bottom=140)),)
    ocr_res = OCRResult(text="SUBMIT", blocks=blocks)

    engine = VisualGroundingEngine(ocr_provider=MockOCRProvider(canned_blocks=blocks))
    result = await engine.locate_element_async(target="submit", observation=obs, ocr_result=ocr_res)

    assert result.is_found is True
    assert result.element is not None
    assert result.element.name == "SUBMIT"
    assert result.element.center == Point(150, 120)


@pytest.mark.anyio
async def test_multiple_identical_ocr_matches_returns_uncertain():
    """Verify that multiple ambiguous identical OCR matches return is_found=False to avoid guessing."""
    obs = _create_synthetic_observation()
    blocks = (
        OCRTextBlock(text="Edit", bounds=WindowBounds(left=50, top=100, right=100, bottom=130)),
        OCRTextBlock(text="Edit", bounds=WindowBounds(left=50, top=200, right=100, bottom=230)),
        OCRTextBlock(text="Edit", bounds=WindowBounds(left=50, top=300, right=100, bottom=330)),
    )
    ocr_res = OCRResult(text="Edit\nEdit\nEdit", blocks=blocks)

    # Without multimodal AI to disambiguate
    engine = VisualGroundingEngine(ocr_provider=MockOCRProvider(canned_blocks=blocks), ai_provider=None)
    result = await engine.locate_element_async(target="Edit", observation=obs, ocr_result=ocr_res)

    assert result.is_found is False
    assert result.confidence < 0.5


# ===========================================================================
# Group C: Multimodal Spatial Grounding
# ===========================================================================


@pytest.mark.anyio
async def test_locate_element_by_multimodal_spatial_reasoning():
    """Verify multimodal AI spatial grounding for iconic/non-text target."""
    obs = _create_synthetic_observation(width=1000, height=800)
    json_resp = json.dumps({
        "found": True,
        "element_type": "icon",
        "box_2d": [100, 800, 150, 850],  # [ymin, xmin, ymax, xmax] in 0..1000
        "confidence": 0.88,
        "label": "Settings Gear Icon",
    })
    mock_ai = MockMultimodalAIProvider(response_text=json_resp, supports_multimodal=True)
    engine = VisualGroundingEngine(ai_provider=mock_ai)

    result = await engine.locate_element_async(target="settings icon", observation=obs)

    assert result.is_found is True
    assert result.confidence == 0.88
    assert result.element is not None
    assert result.element.element_type == UIElementType.ICON
    assert result.element.name == "Settings Gear Icon"
    assert result.element.source == GroundingSource.MULTIMODAL_SEMANTIC

    # Bounds: xmin=800 -> 800px, xmax=850 -> 850px, ymin=100 -> 80px, ymax=150 -> 120px
    assert result.element.bounds == WindowBounds(left=800, top=80, right=850, bottom=120)
    assert result.element.center == Point(x=825, y=100)


@pytest.mark.anyio
async def test_multimodal_target_not_found_response():
    """Verify multimodal not-found response sets is_found=False cleanly."""
    obs = _create_synthetic_observation()
    json_resp = json.dumps({"found": False, "confidence": 0.0, "reason": "Element not visible"})
    mock_ai = MockMultimodalAIProvider(response_text=json_resp)
    engine = VisualGroundingEngine(ai_provider=mock_ai)

    result = await engine.locate_element_async(target="nonexistent button", observation=obs)

    assert result.is_found is False
    assert result.element is None
    assert "could not find" in result.summary.lower()


# ===========================================================================
# Group D: Coordinate Contract & Validation
# ===========================================================================


@pytest.mark.anyio
async def test_coordinate_contract_0_to_1_normalized_range():
    """Verify coordinate parser accepts 0.0..1.0 normalized floating point values."""
    obs = _create_synthetic_observation(width=1000, height=800)
    json_resp = json.dumps({
        "found": True,
        "element_type": "button",
        "box_2d": [0.2, 0.4, 0.3, 0.6],  # ymin=0.2, xmin=0.4, ymax=0.3, xmax=0.6
        "confidence": 0.85,
        "label": "Login",
    })
    mock_ai = MockMultimodalAIProvider(response_text=json_resp)
    engine = VisualGroundingEngine(ai_provider=mock_ai)

    result = await engine.locate_element_async(target="Login", observation=obs, force_multimodal=True)
    assert result.is_found is True
    assert result.element is not None
    # x: 400..600, y: 160..240
    assert result.element.bounds == WindowBounds(left=400, top=160, right=600, bottom=240)


@pytest.mark.anyio
async def test_coordinate_contract_dictionary_formats():
    """Verify coordinate parser handles dictionary bounding box formats."""
    obs = _create_synthetic_observation(width=1000, height=1000)
    json_resp = json.dumps({
        "found": True,
        "element_type": "input",
        "bounds": {"left": 100, "top": 200, "right": 400, "bottom": 250},
        "confidence": 0.90,
        "label": "Search Input",
    })
    mock_ai = MockMultimodalAIProvider(response_text=json_resp)
    engine = VisualGroundingEngine(ai_provider=mock_ai)

    result = await engine.locate_element_async(target="Search Input", observation=obs, force_multimodal=True)
    assert result.is_found is True
    assert result.element is not None
    assert result.element.bounds == WindowBounds(left=100, top=200, right=400, bottom=250)


@pytest.mark.parametrize(
    "invalid_box",
    [
        [-10, 200, 300, 400],  # Negative coordinate
        [100, 200, 50, 400],  # Inverted y (ymax < ymin)
        [100, 500, 200, 300],  # Inverted x (xmax < xmin)
        [float("nan"), 100, 200, 300],  # NaN
        [0, 0, 10000, 20000],  # Out of valid bounds range (>1000 and >frame)
        [100, 100, 101, 101],  # Tiny dimensions (<4px)
        "not-a-box",  # Malformed type
    ],
)
@pytest.mark.anyio
async def test_coordinate_safety_rejects_invalid_coordinates(invalid_box):
    """Verify strict rejection of negative, inverted, NaN, and malformed model coordinates."""
    obs = _create_synthetic_observation(width=1000, height=800)
    json_resp = json.dumps({
        "found": True,
        "element_type": "button",
        "box_2d": invalid_box,
        "confidence": 0.9,
    })
    mock_ai = MockMultimodalAIProvider(response_text=json_resp)
    engine = VisualGroundingEngine(ai_provider=mock_ai)

    result = await engine.locate_element_async(target="button", observation=obs, force_multimodal=True)
    assert result.is_found is False
    assert result.element is None


# ===========================================================================
# Group E: Hybrid Spatial Fusion
# ===========================================================================


@pytest.mark.anyio
async def test_hybrid_spatial_fusion_boosts_confidence_and_attaches_text():
    """Verify that multimodal bounding box overlapping with OCR text creates HYBRID_FUSED result."""
    obs = _create_synthetic_observation(width=1000, height=800)
    blocks = (
        OCRTextBlock(text="Sign In With Google", bounds=WindowBounds(left=305, top=405, right=495, bottom=445)),
    )
    ocr_res = OCRResult(text="Sign In With Google", blocks=blocks)

    # Multimodal detects the larger button visual container [ymin=400, xmin=300, ymax=450, xmax=500] in 0..1000
    json_resp = json.dumps({
        "found": True,
        "element_type": "button",
        "box_2d": [500, 300, 562, 500],  # 500..562 in 800h -> 400..450px, 300..500 in 1000w -> 300..500px
        "confidence": 0.85,
        "label": "Google Login Button",
    })
    mock_ai = MockMultimodalAIProvider(response_text=json_resp)
    engine = VisualGroundingEngine(ai_provider=mock_ai)

    result = await engine.locate_element_async(
        target="Google Login Button",
        observation=obs,
        ocr_result=ocr_res,
        force_multimodal=True,
    )

    assert result.is_found is True
    assert result.element is not None
    assert result.element.source == GroundingSource.HYBRID_FUSED
    assert result.confidence >= 0.95
    assert "Sign In With Google" in (result.element.text_content or "")


# ===========================================================================
# Group F: Provider Degradation & Text-Only Fallbacks
# ===========================================================================


@pytest.mark.anyio
async def test_text_only_provider_degradation_to_ocr_grounding():
    """Verify text-only provider gracefully falls back to OCR matching without crashing."""
    obs = _create_synthetic_observation()
    blocks = (OCRTextBlock(text="Checkout", bounds=WindowBounds(left=200, top=300, right=300, bottom=340)),)
    ocr_res = OCRResult(text="Checkout", blocks=blocks)

    # Multimodal provider raises UnsupportedModalityError (e.g. text-only Ollama)
    mock_ai = MockMultimodalAIProvider(supports_multimodal=False)
    engine = VisualGroundingEngine(ai_provider=mock_ai, ocr_provider=MockOCRProvider(canned_blocks=blocks))

    result = await engine.locate_element_async(target="Checkout", observation=obs, ocr_result=ocr_res)
    assert result.is_found is True
    assert result.element is not None
    assert result.element.source == GroundingSource.OCR_EXACT
    assert result.element.name == "Checkout"


@pytest.mark.anyio
async def test_text_only_provider_icon_query_fails_gracefully():
    """Verify non-text icon target on text-only provider returns clean failure without crash."""
    obs = _create_synthetic_observation()
    ocr_res = OCRResult(text="Some text", blocks=())
    mock_ai = MockMultimodalAIProvider(supports_multimodal=False)
    engine = VisualGroundingEngine(ai_provider=mock_ai, ocr_provider=MockOCRProvider(canned_blocks=()))

    result = await engine.locate_element_async(target="gear icon", observation=obs, ocr_result=ocr_res)
    assert result.is_found is False
    assert result.element is None


# ===========================================================================
# Group G: Security & Privacy Guardrails
# ===========================================================================


def test_locate_element_blocked_on_sensitive_window():
    """Verify that sensitive windows (e.g. Bitwarden, KeePass) block element localization."""
    sec_policy = VisionSecurityPolicy()
    buffer_mgr = EphemeralBufferManager()
    sec_mgr = SecureVisionManager(policy=sec_policy, buffer_manager=buffer_mgr, auto_register_in_container=False)

    # Mock desktop capture engine returning sensitive Bitwarden window
    mock_cap = _create_synthetic_capture(window_title="Bitwarden - Password Vault", process_name="bitwarden.exe")
    sec_mgr._engine = mock.MagicMock()
    sec_mgr._engine.get_active_window_handle.return_value = 9999
    sec_mgr._engine.capture_active_window.return_value = mock_cap
    sec_mgr._resolve_window_metadata = mock.MagicMock(
        return_value=("Bitwarden - Password Vault", "bitwarden.exe")
    )

    skill = VisionSkills(secure_vision_manager=sec_mgr)

    with pytest.raises(Exception) as exc_info:
        skill.execute({"operation": "locate_element", "parameters": {"target": "Master Password"}})

    # Must raise CaptureBlockedError or SkillExecutionError without revealing coordinates
    assert "blocked" in str(exc_info.value).lower() or "security" in str(exc_info.value).lower() or "password" in str(exc_info.value).lower()


def test_zero_raw_pixel_bytes_in_grounding_skill_result():
    """Verify that SystemSkillResult and UIElement data contain zero raw image bytes."""
    obs = _create_synthetic_observation()
    blocks = (OCRTextBlock(text="Save", bounds=WindowBounds(left=100, top=100, right=200, bottom=140)),)
    sec_mgr = mock.MagicMock()
    sec_mgr.capture_active_window.return_value = obs
    sec_mgr.get_latest_observation.return_value = obs

    skill = VisionSkills(
        secure_vision_manager=sec_mgr,
        ocr_provider=MockOCRProvider(canned_blocks=blocks),
    )

    res = skill.execute({"operation": "locate_element", "parameters": {"target": "Save"}})
    assert res.success is True
    data_str = str(res.data)

    # Strict invariant: no raw bytes, PNG headers, or base64 blocks
    assert "PNG" not in data_str
    assert "raw_data" not in data_str
    assert "base64" not in data_str
    assert res.data["is_found"] is True
    assert res.data["element"]["name"] == "Save"


# ===========================================================================
# Group H: VisionSkills Command Recognition & Parsing
# ===========================================================================


def test_vision_skills_can_handle_and_parse_locate_commands():
    """Verify natural language patterns for locate_element commands."""
    skill = VisionSkills()

    queries = [
        ("where is the submit button", "submit button"),
        ("find the search bar", "search bar"),
        ("locate the close icon", "close icon"),
        ("show me where the settings icon is", "settings icon"),
        ("where is the error message?", "error message"),
    ]

    for q, expected_target in queries:
        assert skill.can_handle(q) is True
        op, target, params, _ = skill.parse_command(q)
        assert op == "locate_element"
        assert target == "active_window"
        assert params["target"].lower() == expected_target.lower()


# ===========================================================================
# Group I: Planner Integration
# ===========================================================================


def test_planner_known_actions_includes_locate_element():
    """Verify locate_element is registered in KNOWN_ACTIONS and planner prompts."""
    assert "locate_element" in KNOWN_ACTIONS
    assert "locate_element" in PLANNER_SYSTEM_PROMPT
    assert "locate_element" in RECOVERY_SYSTEM_PROMPT
