"""Unit tests for Phase 23.5 Live AI Cognition Telemetry and Dynamic Execution.

Verifies:
1. RecentExecutionCard dynamic updates and empty state fallback
2. AICoreBadge live provider, model, mode, latency, and health display
3. Missing telemetry gracefully falls back to "N/A"
4. Provider switching through existing ProviderRouter without modifying .env
5. UIBridge signals deliver execution and telemetry updates non-blockingly
6. Confirmation architecture invariants preserved
"""

import os
import pytest
from unittest.mock import MagicMock
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

os.environ["QT_QPA_PLATFORM"] = "offscreen"


@pytest.fixture(scope="session")
def qapp():
    """Session-level QApplication instance for offscreen GUI testing."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_recent_execution_card_empty_state(qapp):
    """Verify RecentExecutionCard defaults to clean 'No recent executions'."""
    from app.ui.components.center_panel import RecentExecutionCard

    card = RecentExecutionCard()
    try:
        assert card._badge_lbl.text() == "0"
        assert card._empty_label.isVisible() or not card._empty_label.isHidden()
        assert card._empty_label.text() == "No recent executions"
        assert card.skill_title.text() == "Idle"
    finally:
        card.close()


def test_recent_execution_card_dynamic_records(qapp):
    """Verify RecentExecutionCard renders dynamic execution steps correctly."""
    from app.ui.components.center_panel import RecentExecutionCard

    card = RecentExecutionCard()
    try:
        records = [
            {
                "task_id": "step-1",
                "action": "Parse Intent",
                "target": "Voice Audio",
                "status": "completed",
                "duration": 0.12,
            },
            {
                "task_id": "step-2",
                "action": "Web Search",
                "target": "Weather API",
                "status": "completed",
                "duration": 0.85,
            },
            {
                "task_id": "step-3",
                "action": "File Write",
                "target": "output.txt",
                "status": "failed",
                "error": "Permission denied",
                "duration": 0.05,
            },
        ]
        card.update_executions(records)

        assert card._badge_lbl.text() == "2 / 3"
        # Must have 3 step widgets
        assert card._steps_layout.count() == 3
        assert card.skill_title.text() == "File Write"

        # Test resetting back to empty
        card.update_executions([])
        assert card._badge_lbl.text() == "0"
        assert card._empty_label.text() == "No recent executions"
    finally:
        card.close()


def test_ai_core_badge_live_telemetry(qapp):
    """Verify AICoreBadge updates live provider, mode, model, and latency."""
    from app.ui.components.center_panel import AICoreBadge

    badge = AICoreBadge()
    try:
        # Default state
        assert badge._provider_badge.text() == "N/A"
        assert badge._mode_lbl.text() == "N/A"
        assert badge._latency_lbl.text() == "N/A"

        # Update with Gemini Cloud data
        gemini_data = {
            "active_provider": "gemini",
            "active_model": "gemini-2.5-flash",
            "mode": "CLOUD",
            "reasoning_latency_ms": 145.2,
            "health": "ONLINE",
        }
        badge.update_telemetry(gemini_data)

        assert badge._provider_badge.text() == "GEMINI"
        assert badge._mode_lbl.text() == "CLOUD"
        assert "gemini" in badge._model_lbl.text().lower()
        assert badge._latency_lbl.text() == "145 ms"
        assert badge._health_lbl.text() == "ONLINE"

        # Update with Ollama Local data
        ollama_data = {
            "active_provider": "ollama",
            "active_model": "qwen2.5:3b",
            "mode": "LOCAL",
            "reasoning_latency_ms": 42.0,
            "health": "HEALTHY",
        }
        badge.update_telemetry(ollama_data)

        assert badge._provider_badge.text() == "OLLAMA"
        assert badge._mode_lbl.text() == "LOCAL"
        assert "qwen" in badge._model_lbl.text().lower()
        assert badge._latency_lbl.text() == "42 ms"
    finally:
        badge.close()


def test_ai_core_badge_missing_telemetry_fallback(qapp):
    """Verify AICoreBadge displays N/A when telemetry values are missing."""
    from app.ui.components.center_panel import AICoreBadge

    badge = AICoreBadge()
    try:
        badge.update_telemetry({})
        assert badge._provider_badge.text() == "N/A"
        assert badge._mode_lbl.text() == "N/A"
        assert badge._model_lbl.text() == "N/A"
        assert badge._latency_lbl.text() == "N/A"
    finally:
        badge.close()


def test_provider_router_switching_integration():
    """Verify switching providers interacts with the canonical ProviderRouter safely."""
    from app.ai.provider_router import provider_router

    # Verify existing registered providers include gemini and ollama
    providers = provider_router.get_registered_providers()
    assert "gemini" in providers
    assert "ollama" in providers

    initial_provider = provider_router.active_provider_name

    try:
        # Switch to Ollama
        provider_router.set_active_provider("ollama")
        assert provider_router.active_provider_name == "ollama"

        # Switch to Gemini
        provider_router.set_active_provider("gemini")
        assert provider_router.active_provider_name == "gemini"
    finally:
        # Restore initial provider
        provider_router.set_active_provider(initial_provider)


def test_ui_bridge_cognition_signals(qapp):
    """Verify UIBridge execution history and cognition telemetry signals."""
    from app.core.presentation import PresentationAdapter
    from app.core.state import AssistantSnapshot, AssistantState
    from app.ui.bridge import UIBridge

    adapter = MagicMock(spec=PresentationAdapter)
    adapter.get_snapshot.return_value = AssistantSnapshot(state=AssistantState.IDLE)
    adapter.drain_events.return_value = []

    bridge = UIBridge(adapter)
    try:
        history_events = []
        bridge.execution_history_updated.connect(history_events.append)

        # Manually add execution record
        rec = {
            "task_id": "test-101",
            "action": "Inspect Hardware",
            "target": "SystemMonitor",
            "status": "completed",
            "duration": 0.08,
            "timestamp": 1720000050.0,
        }
        bridge.add_execution_record(rec)

        assert len(history_events) == 1
        assert history_events[0][0]["task_id"] == "test-101"
        assert bridge.get_execution_history()[0]["action"] == "Inspect Hardware"
    finally:
        bridge.close()


def test_confirmation_architecture_untouched(qapp):
    """Verify confirmation architecture still routes through safe security gateway."""
    from app.core.presentation import PresentationAdapter
    from app.core.state import AssistantSnapshot, AssistantState, PendingConfirmation
    from app.ui.bridge import UIBridge
    from app.ui.main_window import JarvisMainWindow

    adapter = MagicMock(spec=PresentationAdapter)
    adapter.get_snapshot.return_value = AssistantSnapshot(state=AssistantState.IDLE)
    adapter.drain_events.return_value = []
    adapter.resolve_confirmation.return_value = True

    bridge = UIBridge(adapter)
    win = JarvisMainWindow(presentation_adapter=adapter, bridge=bridge)

    try:
        # Simulate AWAITING_CONFIRMATION snapshot
        conf = PendingConfirmation(
            confirmation_id="conf-phase23-5",
            operation="system.hardware_inspection",
            target="all",
            risk_level="HIGH",
            description="Read-only diagnostic validation",
        )
        snap = AssistantSnapshot(
            state=AssistantState.AWAITING_CONFIRMATION,
            pending_confirmation=conf,
        )

        win._on_snapshot_updated(snap)

        # Confirm card is shown in center panel stack
        assert win.center_panel.lower_stack.currentIndex() == 1

        # Simulate operator confirmation approval
        win._on_confirmation_confirmed("conf-phase23-5")
        adapter.resolve_confirmation.assert_called_with(
            "conf-phase23-5",
            approved=True,
            decided_by="gui_operator",
            reason="Authorized by operator via HUD ConfirmationCard",
        )
    finally:
        win.close()
        bridge.close()
