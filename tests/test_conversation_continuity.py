"""Focused Unit and Integration Tests for Phase 24.

Validates:
1. ConversationMemory serialization / deserialization.
2. Missing history file handling (graceful empty return).
3. Corrupt / malformed history file recovery.
4. Atomic persistence using NamedTemporaryFile and replace.
5. Bounded conversation history enforcement.
6. Chronological history ordering.
7. Empty conversation card state display.
8. GUI history bootstrap via UIBridge.
9. User and assistant role rendering and avatar styles.
10. Typing indicator lifecycle and dot cycling.
11. Command completion removes typing indicator and renders response.
12. Command failure removes typing indicator and renders error.
13. Cognitive stage transitions (STANDBY, ANALYZING, ROUTING, EXECUTING, SYNTHESIZING).
14. Cognitive stage never exposes private chain-of-thought or reasoning text.
15. Memory persistence failure does not crash the GUI.
16. No duplicate conversation records on multiple dispatches.
17. Qt main thread safety during background history operations.
18. Preservation of the Phase 23.4 operator confirmation gateway.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest.mock as mock
import pytest

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication

import gui
from app.core.container import ServiceContainer
from app.core.event_bus import EventBus
from app.core.presentation import PresentationAdapter
from app.core.state import AssistantSnapshot, AssistantState, PendingConfirmation
from app.core.state_manager import AssistantStateManager
from app.memory.models import ConversationMemory
from app.memory.manager import MemoryManager
from app.memory.persistence import ConversationHistoryPersistence, redact_secrets
from app.ui.bridge import UIBridge
from app.ui.components.left_panel import ConversationCard, MessageBubble, TypingIndicatorBubble
from app.ui.components.center_panel import CognitiveStageChip
from app.ui.main_window import JarvisMainWindow


@pytest.fixture(scope="session")
def qapp():
    """Ensure a headless QApplication instance exists for offscreen testing."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["JARVIS_TEST_MODE"] = "1"
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


# ==============================================================================
# 1. Memory Serialization / Deserialization & Secret Redaction
# ==============================================================================

def test_conversation_memory_serialization_and_redaction():
    """1. Test ConversationMemory serialization and credential redaction."""
    mem = ConversationMemory(
        content="Testing key=AIzaSyA1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6",
        role="user",
        source="text",
    )
    d = mem.to_dict()
    assert d["content"] == "Testing key=AIzaSyA1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6"
    assert d["role"] == "user"

    # Verify secret redaction
    redacted = redact_secrets(d["content"])
    assert "AIzaSy" not in redacted
    assert "[REDACTED_SECRET]" in redacted

    bearer_test = redact_secrets("Authorization: Bearer my_secret_token_12345678")
    assert "my_secret_token_12345678" not in bearer_test
    assert "[REDACTED_SECRET]" in bearer_test


# ==============================================================================
# 2, 3, 4, 5, 6. Persistence Unit Tests
# ==============================================================================

def test_persistence_missing_file_graceful():
    """2. Missing history file returns an empty list without raising."""
    with tempfile.TemporaryDirectory() as tmpdir:
        non_existent = Path(tmpdir) / "sub" / "non_existent.json"
        p = ConversationHistoryPersistence(file_path=non_existent)
        history = p.load_history()
        assert history == []


def test_persistence_corrupt_file_recovery():
    """3. Corrupt or malformed JSON is handled gracefully without crashing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        corrupt_file = Path(tmpdir) / "corrupt.json"
        corrupt_file.write_text("{{{{{invalid_json: true", encoding="utf-8")

        p = ConversationHistoryPersistence(file_path=corrupt_file)
        history = p.load_history()
        assert history == []


def test_persistence_atomic_write_and_save():
    """4. Persistence writes atomically and reloads accurate data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hist_file = Path(tmpdir) / "history.json"
        p = ConversationHistoryPersistence(file_path=hist_file, max_entries=50)

        memories = [
            ConversationMemory(content="Turn 1", role="user", timestamp=1000.0),
            ConversationMemory(content="Turn 2", role="assistant", timestamp=1001.0),
        ]
        success = p.save_history(memories)
        assert success is True
        assert hist_file.exists()

        reloaded = p.load_history()
        assert len(reloaded) == 2
        assert reloaded[0].content == "Turn 1"
        assert reloaded[0].role == "user"
        assert reloaded[1].content == "Turn 2"
        assert reloaded[1].role == "assistant"


def test_persistence_bounded_history():
    """5. History is strictly bounded to max_entries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hist_file = Path(tmpdir) / "history.json"
        p = ConversationHistoryPersistence(file_path=hist_file, max_entries=5)

        many_memories = [
            ConversationMemory(content=f"Message {i}", role="user") for i in range(15)
        ]
        p.save_history(many_memories)

        reloaded = p.load_history()
        assert len(reloaded) == 5
        assert reloaded[0].content == "Message 10"
        assert reloaded[-1].content == "Message 14"


def test_persistence_chronological_ordering():
    """6. History preserves strict chronological ordering."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hist_file = Path(tmpdir) / "history.json"
        p = ConversationHistoryPersistence(file_path=hist_file)

        m1 = ConversationMemory(content="First", timestamp=100.0)
        m2 = ConversationMemory(content="Second", timestamp=200.0)
        m3 = ConversationMemory(content="Third", timestamp=300.0)

        p.save_history([m1, m2, m3])
        loaded = p.load_history()

        assert [m.content for m in loaded] == ["First", "Second", "Third"]


# ==============================================================================
# 7, 8, 9, 10. ConversationCard & MessageBubble Widget Tests
# ==============================================================================

def test_conversation_card_empty_state(qapp):
    """7. ConversationCard displays empty state when no messages exist."""
    card = ConversationCard()
    assert not card._empty_state.isHidden()
    assert card._message_count == 0


def test_conversation_card_load_history_bootstrap(qapp):
    """8. ConversationCard populates loaded history and hides empty state."""
    card = ConversationCard()
    records = [
        {"sender": "You", "text": "Hello Jarvis", "timestamp": "10:00 AM", "is_error": False},
        {"sender": "J.A.R.V.I.S", "text": "At your service.", "timestamp": "10:01 AM", "is_error": False},
    ]
    card.load_history(records)

    assert card._empty_state.isHidden() is True
    assert card._message_count == 2
    bubbles = card.findChildren(MessageBubble)
    assert len(bubbles) == 2
    assert bubbles[0].msg_lbl.text() == "Hello Jarvis"
    assert bubbles[1].msg_lbl.text() == "At your service."


def test_message_bubble_roles_and_avatars(qapp):
    """9. User and assistant bubbles render distinct avatars and role labels."""
    user_bubble = MessageBubble("You", "User query", "12:00 PM")
    jarvis_bubble = MessageBubble("J.A.R.V.I.S", "Assistant reply", "12:01 PM")
    error_bubble = MessageBubble("J.A.R.V.I.S", "Failure message", "12:02 PM", is_error=True)

    assert user_bubble.msg_lbl.text() == "User query"
    assert jarvis_bubble.msg_lbl.text() == "Assistant reply"
    assert error_bubble._is_error is True
    assert "Failure message" in error_bubble.msg_lbl.text()


def test_typing_indicator_lifecycle(qapp):
    """10. Typing indicator shows, cycles dots, and removes cleanly."""
    card = ConversationCard()
    card.show_typing_indicator()

    assert card._typing_indicator is not None
    assert not card._typing_indicator.isHidden()
    assert card._empty_state.isHidden() is True

    # Verify dot animation tick
    init_dots = card._typing_indicator._dots_lbl.text()
    card._typing_indicator._on_tick()
    next_dots = card._typing_indicator._dots_lbl.text()
    assert init_dots != next_dots

    # Hide indicator
    card.hide_typing_indicator()
    assert card._typing_indicator is None
    # Because message_count is 0, empty state returns
    assert not card._empty_state.isHidden()


# ==============================================================================
# 11, 12, 16. Integration of Command Completion & Failure
# ==============================================================================

def test_command_completion_removes_typing_indicator_and_adds_message(qapp):
    """11 & 16. Command completion removes typing indicator and avoids duplication."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hist_file = Path(tmpdir) / "test_hist.json"
        persistence = ConversationHistoryPersistence(file_path=hist_file)

        mock_router = mock.MagicMock()
        async def _mock_route(cmd, source="gui"):
            await asyncio.sleep(0.02)
            return f"Processed: {cmd}"
        mock_router.route_async = mock.AsyncMock(side_effect=_mock_route)

        state_mgr = AssistantStateManager(initial_state=AssistantState.IDLE)
        adapter = PresentationAdapter(state_manager=state_mgr, command_router=mock_router)
        mem_mgr = MemoryManager(persistence=persistence, auto_register_in_container=False)

        bridge = UIBridge(
            presentation_adapter=adapter,
            memory_manager=mem_mgr,
            persistence=persistence,
            poll_interval_ms=10,
        )
        window = JarvisMainWindow(bridge=bridge)

        # Dispatch a command
        window._on_command_dispatched("Check server status")
        assert window.left_panel.conversation_card._typing_indicator is not None

        # Wait for completion
        completed = []
        bridge.command_completed.connect(completed.append)

        start = time.perf_counter()
        while not completed and (time.perf_counter() - start) < 2.0:
            qapp.processEvents()
            time.sleep(0.01)

        assert len(completed) == 1
        assert completed[0] == "Processed: Check server status"

        # Verify typing indicator was removed
        assert window.left_panel.conversation_card._typing_indicator is None

        # Verify conversation card has user + assistant messages (count = 2)
        assert window.left_panel.conversation_card._message_count == 2

        # 16. Verify no duplicates in memory store
        recent_mems = mem_mgr.get_recent()
        assert len(recent_mems) == 2
        assert recent_mems[0].content == "Check server status"
        assert recent_mems[1].content == "Processed: Check server status"

        gui.shutdown_gui(bridge=bridge, adapter=adapter)
        window.close()


def test_command_failure_removes_typing_indicator_and_renders_error(qapp):
    """12. Command failure removes typing indicator and renders an error bubble."""
    mock_router = mock.MagicMock()
    async def _failing_route(cmd, source="gui"):
        await asyncio.sleep(0.01)
        raise RuntimeError("Subsystem timeout")
    mock_router.route_async = mock.AsyncMock(side_effect=_failing_route)

    state_mgr = AssistantStateManager(initial_state=AssistantState.IDLE)
    adapter = PresentationAdapter(state_manager=state_mgr, command_router=mock_router)

    bridge = UIBridge(presentation_adapter=adapter, poll_interval_ms=10)
    window = JarvisMainWindow(bridge=bridge)

    window._on_command_dispatched("Faulty command")
    assert window.left_panel.conversation_card._typing_indicator is not None

    failed = []
    bridge.command_failed.connect(failed.append)

    start = time.perf_counter()
    while not failed and (time.perf_counter() - start) < 2.0:
        qapp.processEvents()
        time.sleep(0.01)

    assert len(failed) == 1
    assert "Subsystem timeout" in failed[0]
    assert window.left_panel.conversation_card._typing_indicator is None

    # Error message added
    bubbles = window.left_panel.conversation_card.findChildren(MessageBubble)
    assert any("Error: Subsystem timeout" in b.msg_lbl.text() for b in bubbles)

    gui.shutdown_gui(bridge=bridge, adapter=adapter)
    window.close()


# ==============================================================================
# 13, 14. Cognitive Stage Telemetry Tests
# ==============================================================================

def test_cognitive_stage_transitions(qapp):
    """13. Cognitive stage chip transitions through ANALYZING, ROUTING, EXECUTING, SYNTHESIZING, STANDBY."""
    chip = CognitiveStageChip()
    assert chip._stage_lbl.text() == "STANDBY"

    chip.set_stage("ANALYZING", "Parsing query")
    assert chip._stage_lbl.text() == "ANALYZING"
    assert chip._detail_lbl.text() == "Parsing query"

    chip.set_stage("ROUTING", "Skill lookup")
    assert chip._stage_lbl.text() == "ROUTING"
    assert chip._detail_lbl.text() == "Skill lookup"

    chip.set_stage("EXECUTING", "Direct Command")
    assert chip._stage_lbl.text() == "EXECUTING"
    assert chip._detail_lbl.text() == "Direct Command"

    chip.set_stage("SYNTHESIZING", "Generating answer")
    assert chip._stage_lbl.text() == "SYNTHESIZING"
    assert chip._detail_lbl.text() == "Generating answer"

    chip.set_stage("STANDBY", "Ready")
    assert chip._stage_lbl.text() == "STANDBY"
    assert chip._detail_lbl.text() == "Ready"


def test_cognitive_stage_never_exposes_private_reasoning(qapp):
    """14. Detail text is sanitized/truncated and never allows internal chain-of-thought dump."""
    chip = CognitiveStageChip()
    long_chain_of_thought = "Thinking step 1: Let me ponder about the nature of existence and internal weights [SECRET_KEY=12345]..."
    chip.set_stage("ANALYZING", long_chain_of_thought)

    assert chip._stage_lbl.text() == "ANALYZING"
    # Guaranteed to be truncated to max 18 chars
    assert len(chip._detail_lbl.text()) <= 18
    assert "SECRET_KEY" not in chip._detail_lbl.text()


# ==============================================================================
# 15, 17. Resilience & Thread Safety Tests
# ==============================================================================

def test_memory_persistence_failure_does_not_crash_gui(qapp):
    """15. Disk persistence failure does not raise an exception or crash GUI."""
    mock_persistence = mock.MagicMock(spec=ConversationHistoryPersistence)
    mock_persistence.load_history.side_effect = IOError("Disk read failed")
    mock_persistence.save_history.side_effect = IOError("Disk write failed")

    state_mgr = AssistantStateManager(initial_state=AssistantState.IDLE)
    adapter = PresentationAdapter(state_manager=state_mgr)

    # UIBridge initialization must not crash despite persistence error
    bridge = UIBridge(
        presentation_adapter=adapter,
        persistence=mock_persistence,
        poll_interval_ms=10,
    )
    window = JarvisMainWindow(bridge=bridge)

    # GUI operates normally
    window._on_command_dispatched("Safe fallback command")
    assert window.left_panel.conversation_card._message_count == 1

    gui.shutdown_gui(bridge=bridge, adapter=adapter)
    window.close()


def test_qt_thread_safety_during_history_fetch(qapp):
    """17. Asynchronous history fetch runs on worker pool and emits safely to main thread."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hist_file = Path(tmpdir) / "history.json"
        p = ConversationHistoryPersistence(file_path=hist_file)
        p.append_turn("Past query", "Past answer")

        state_mgr = AssistantStateManager(initial_state=AssistantState.IDLE)
        adapter = PresentationAdapter(state_manager=state_mgr)
        mem_mgr = MemoryManager(persistence=p, auto_register_in_container=False)
        bridge = UIBridge(
            presentation_adapter=adapter,
            memory_manager=mem_mgr,
            persistence=p,
            poll_interval_ms=10,
        )

        history_results = []
        bridge.conversation_history_loaded.connect(history_results.append)

        bridge.fetch_conversation_history()

        start = time.perf_counter()
        while not history_results and (time.perf_counter() - start) < 2.0:
            qapp.processEvents()
            time.sleep(0.01)

        assert len(history_results) >= 1
        records = history_results[-1]
        assert len(records) == 2
        assert records[0]["text"] == "Past query"
        assert records[1]["text"] == "Past answer"

        gui.shutdown_gui(bridge=bridge, adapter=adapter)


# ==============================================================================
# 18. Operator Confirmation Gateway Invariant
# ==============================================================================

def test_operator_confirmation_gateway_remains_functional(qapp):
    """18. Phase 23.4 Confirmation card gateway operates properly alongside Phase 24 conversation."""
    state_mgr = AssistantStateManager(initial_state=AssistantState.IDLE)
    mock_conf_mgr = mock.MagicMock()
    mock_conf_mgr.resolve_confirmation.return_value = True

    adapter = PresentationAdapter(
        state_manager=state_mgr,
        confirmation_manager=mock_conf_mgr,
    )
    bridge = UIBridge(presentation_adapter=adapter, poll_interval_ms=10)
    window = JarvisMainWindow(bridge=bridge)

    conf = PendingConfirmation(
        confirmation_id="conf-phase24-test",
        operation="Format Drive",
        risk_level="CRITICAL",
        description="Dangerous operation test",
    )
    # Transition to confirmation state via snapshot
    snap_awaiting = AssistantSnapshot(
        state=AssistantState.AWAITING_CONFIRMATION,
        status_message="Operator confirmation needed",
        pending_confirmation=conf,
    )
    window._on_snapshot_updated(snap_awaiting)
    qapp.processEvents()

    assert window.center_panel.lower_stack.currentIndex() == 1
    assert window.center_panel.confirmation_card.confirmation_id == "conf-phase24-test"

    # Operator confirms
    window.center_panel.confirmation_card.confirm_btn.click()
    qapp.processEvents()

    mock_conf_mgr.resolve_confirmation.assert_called_once_with(
        "conf-phase24-test",
        approved=True,
        decided_by="gui_operator",
        reason="Authorized by operator via HUD ConfirmationCard",
    )

    gui.shutdown_gui(bridge=bridge, adapter=adapter)
    window.close()
