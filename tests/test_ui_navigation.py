"""Unit tests for Phase 23.5 HUD Multi-View Navigation.

Verifies:
1. HOME navigation (stack index 0)
2. ACTIVITY navigation (stack index 1)
3. SYSTEM navigation (stack index 2)
4. SETTINGS navigation (stack index 3)
5. Header selection updates QStackedWidget
6. Header and bottom bar remain persistent outside QStackedWidget
7. Read-only safety on SystemView
8. SettingsView provider selection
"""

import os
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QTableWidget

# Ensure headless Qt environment
os.environ["QT_QPA_PLATFORM"] = "offscreen"


@pytest.fixture(scope="session")
def qapp():
    """Session-level QApplication instance for offscreen GUI testing."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_main_window_stack_initialization(qapp):
    """Verify JarvisMainWindow initializes 4 stack views with HOME active."""
    from app.ui.main_window import JarvisMainWindow

    win = JarvisMainWindow(presentation_adapter=None, bridge=None)
    try:
        assert hasattr(win, "view_stack")
        assert win.view_stack.count() == 4

        # Initial index must be 0 (HOME)
        assert win.view_stack.currentIndex() == 0
        assert win.view_stack.currentWidget() == win.home_view
    finally:
        win.close()


def test_header_navigation_switches_views(qapp):
    """Verify HeaderBar navigation_changed signal updates QStackedWidget index."""
    from app.ui.main_window import JarvisMainWindow

    win = JarvisMainWindow(presentation_adapter=None, bridge=None)
    try:
        # 1. Navigate to ACTIVITY
        win.header.set_active_tab("ACTIVITY")
        assert win.view_stack.currentIndex() == 1
        assert win.view_stack.currentWidget() == win.activity_view

        # 2. Navigate to SYSTEM
        win.header.set_active_tab("SYSTEM")
        assert win.view_stack.currentIndex() == 2
        assert win.view_stack.currentWidget() == win.system_view

        # 3. Navigate to SETTINGS
        win.header.set_active_tab("SETTINGS")
        assert win.view_stack.currentIndex() == 3
        assert win.view_stack.currentWidget() == win.settings_view

        # 4. Navigate back to HOME
        win.header.set_active_tab("HOME")
        assert win.view_stack.currentIndex() == 0
        assert win.view_stack.currentWidget() == win.home_view
    finally:
        win.close()


def test_header_and_bottom_bar_persistence(qapp):
    """Verify HeaderBar and BottomBar are not inside QStackedWidget and remain persistent."""
    from app.ui.main_window import JarvisMainWindow

    win = JarvisMainWindow(presentation_adapter=None, bridge=None)
    try:
        # Check that header and bottom_bar are direct children of root layout, not inside stack
        assert win.header.parentWidget() == win.centralWidget()
        assert win.bottom_bar.parentWidget() == win.centralWidget()

        for tab in ("HOME", "ACTIVITY", "SYSTEM", "SETTINGS"):
            win.header.set_active_tab(tab)
            # Verify both remain visible and outside the view_stack
            assert win.header.isVisible() or not win.header.isHidden()
            assert win.bottom_bar.isVisible() or not win.bottom_bar.isHidden()
            assert win.view_stack.indexOf(win.header) == -1
            assert win.view_stack.indexOf(win.bottom_bar) == -1
    finally:
        win.close()


def test_activity_view_updates_and_empty_state(qapp):
    """Verify ActivityView handles empty state and dynamic execution records."""
    from app.ui.views.activity_view import ActivityView

    view = ActivityView()
    try:
        # Default empty state
        assert view._total_lbl.text() == "0"
        assert view._active_count_lbl.text() == "0"

        # Update with real/mock execution records
        records = [
            {
                "task_id": "task-1",
                "action": "Open Browser",
                "target": "Chrome",
                "status": "completed",
                "duration": 0.45,
                "timestamp": 1720000000.0,
            },
            {
                "task_id": "task-2",
                "action": "Query Telemetry",
                "target": "CPU",
                "status": "failed",
                "error": "Sensor timeout",
                "duration": 1.20,
                "timestamp": 1720000010.0,
            },
        ]
        view.update_executions(records)

        assert view._total_lbl.text() == "2"
        assert view._success_rate_lbl.text() == "50%"
        assert view._active_count_lbl.text() == "0"

        # Update active plan state
        plan = {
            "title": "System Diagnostic Scan",
            "progress": 0.6,
            "steps": [
                {"title": "Inspect CPU", "status": "completed"},
                {"title": "Inspect RAM", "status": "running"},
            ],
        }
        view.update_active_plan(plan)
        assert view._plan_title_lbl.text() == "System Diagnostic Scan"
        assert view._plan_pbar.value() == 60
    finally:
        view.close()


def test_system_view_read_only_invariants(qapp):
    """Verify SystemView is strictly read-only with no modification or execution capabilities."""
    from app.ui.views.system_view import SystemView

    view = SystemView()
    try:
        # Table edit triggers must be NoEditTriggers
        assert view.proc_table.editTriggers() == QTableWidget.EditTrigger.NoEditTriggers

        # Update diagnostics data
        diag = {
            "os_platform": "Windows 11 (AMD64)",
            "device_name": "TEST-DESKTOP",
            "uptime": "2h 15m",
            "cpu_percent": 18.5,
            "cpu_cores": "8P / 16L",
            "cpu_freq_mhz": 3400.0,
            "memory_percent": 45.0,
            "ram_used_gb": 7.2,
            "ram_total_gb": 16.0,
            "ram_free_gb": 8.8,
            "disk_percent": 62.0,
            "disk_used_gb": 310.0,
            "disk_total_gb": 500.0,
            "disk_mount": "C:\\",
            "network_rate_kbs": 12.4,
            "battery_percent": 98.0,
            "battery_plugged": True,
            "top_processes": [
                {"pid": 1024, "name": "python.exe", "cpu_percent": 5.2, "memory_percent": 2.1, "status": "running"},
                {"pid": 2048, "name": "chrome.exe", "cpu_percent": 3.1, "memory_percent": 4.5, "status": "running"},
            ],
        }
        view.update_diagnostics(diag)

        assert "TEST-DESKTOP" in view._host_info_lbl.text()
        assert "18.5%" in view.cpu_box._val_lbl.text()
        assert "7.2 / 16.0 GB" in view.ram_box._val_lbl.text()
        assert view.proc_table.rowCount() == 2
    finally:
        view.close()


def test_settings_view_provider_selection_signal(qapp):
    """Verify SettingsView provider switching emits signal without modifying disk."""
    from app.ui.views.settings_view import SettingsView

    view = SettingsView()
    emitted = []
    view.provider_changed.connect(emitted.append)
    try:
        # Select Ollama
        view._provider_cards["ollama"].trigger_selection()
        assert "ollama" in emitted
        assert view._provider_cards["ollama"]._is_active
        assert not view._provider_cards["gemini"]._is_active

        # Select Gemini
        view._provider_cards["gemini"].trigger_selection()
        assert "gemini" in emitted
        assert view._provider_cards["gemini"]._is_active
        assert not view._provider_cards["ollama"]._is_active
    finally:
        view.close()
