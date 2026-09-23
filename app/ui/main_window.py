"""Main Window Assembly for J.A.R.V.I.S Desktop HUD.

Assembles HeaderBar, LeftPanel, CenterPanel, RightPanel, and BottomBar
into a responsive, high-performance PySide6 desktop HUD matching the
locked jarvis_ui_reference.png reference design.
Communicates strictly through UIBridge and PresentationAdapter.
"""

from __future__ import annotations

import datetime
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.core.presentation import PresentationAdapter
from app.core.state import AssistantSnapshot, AssistantState
from app.ui.bridge import UIBridge
from app.ui.components.bottom_bar import BottomBar
from app.ui.components.center_panel import CenterPanel
from app.ui.components.header import HeaderBar
from app.ui.components.left_panel import LeftPanel
from app.ui.components.right_panel import RightPanel
from app.ui.styles import JarvisTheme, ensure_fonts_loaded
from app.ui.views import ActivityView, SettingsView, SystemView


class JarvisMainWindow(QMainWindow):
    """Full J.A.R.V.I.S Sci-Fi Desktop HUD Window with Multi-View Navigation."""

    def __init__(
        self,
        presentation_adapter: Optional[PresentationAdapter] = None,
        bridge: Optional[UIBridge] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        """Initialize JarvisMainWindow.

        Args:
            presentation_adapter: Optional PresentationAdapter boundary.
            bridge: Optional pre-configured UIBridge.
            parent: Optional parent QWidget.
        """
        super().__init__(parent)
        ensure_fonts_loaded()

        self.setWindowTitle("J.A.R.V.I.S — Tactical AI Assistant")
        self.setMinimumSize(1200, 780)
        self.resize(1440, 900)

        # Main background styling
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {JarvisTheme.BG_MAIN};
            }}
            QWidget {{
                color: {JarvisTheme.TEXT_PRIMARY};
                font-family: {JarvisTheme.FONT_FAMILY};
            }}
        """)

        # Central widget and root vertical layout
        central_widget = QWidget(self)
        central_widget.setObjectName("JarvisRootWidget")
        central_widget.setStyleSheet(f"QWidget#JarvisRootWidget {{ background-color: {JarvisTheme.BG_MAIN}; }}")
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 1. Top Header Bar (Persistent across all views)
        self.header = HeaderBar(self)
        root_layout.addWidget(self.header, 0)

        # 2. Main Content Area: QStackedWidget Multi-View Architecture
        self.view_stack = QStackedWidget(central_widget)
        self.view_stack.setStyleSheet("background: transparent;")

        # INDEX 0: HOME VIEW (Existing 3-column HUD)
        self.home_view = QWidget(self.view_stack)
        self.home_view.setStyleSheet("background: transparent;")
        home_layout = QHBoxLayout(self.home_view)
        home_layout.setContentsMargins(16, 12, 16, 8)
        home_layout.setSpacing(14)

        # Left Column: Conversation + Quick Actions (Fixed ~290px)
        self.left_panel = LeftPanel(self.home_view)
        home_layout.addWidget(self.left_panel, 0)

        # Center Column: ArcReactorCore + Waveform + State Pills + Recent Execution
        self.center_panel = CenterPanel(self.home_view)
        home_layout.addWidget(self.center_panel, 1)

        # Right Column: Current Task + System Status + System Info
        self.right_panel = RightPanel(self.home_view)
        home_layout.addWidget(self.right_panel, 0)

        self.view_stack.addWidget(self.home_view)  # Index 0: HOME

        # INDEX 1: ACTIVITY VIEW
        self.activity_view = ActivityView(self.view_stack)
        self.view_stack.addWidget(self.activity_view)  # Index 1: ACTIVITY

        # INDEX 2: SYSTEM DIAGNOSTICS VIEW
        self.system_view = SystemView(self.view_stack)
        self.view_stack.addWidget(self.system_view)  # Index 2: SYSTEM

        # INDEX 3: SETTINGS VIEW
        self.settings_view = SettingsView(self.view_stack)
        self.view_stack.addWidget(self.settings_view)  # Index 3: SETTINGS

        self.view_stack.setCurrentIndex(0)
        root_layout.addWidget(self.view_stack, 1)

        # 3. Bottom Command Bar (Persistent across all views)
        self.bottom_bar = BottomBar(self)
        root_layout.addWidget(self.bottom_bar, 0)

        # Setup UIBridge if provided or created
        self._bridge = bridge
        if self._bridge is None and presentation_adapter is not None:
            self._bridge = UIBridge(presentation_adapter, parent=self)

        if self._bridge is not None:
            self._connect_bridge(self._bridge)

        # Connect navigation signal from HeaderBar
        self.header.navigation_changed.connect(self._on_navigation_changed)

        # Connect user interaction signals from subcomponents to bridge
        self.left_panel.action_dispatched.connect(self._on_command_dispatched)
        self.bottom_bar.command_submitted.connect(self._on_command_dispatched)
        self.header.voice_toggled.connect(self._on_voice_toggled)
        self.bottom_bar.voice_toggled.connect(self._on_voice_toggled)
        self.center_panel.state_pills.state_selected.connect(self._on_state_selected)
        self.center_panel.confirmation_confirmed.connect(self._on_confirmation_confirmed)
        self.center_panel.confirmation_cancelled.connect(self._on_confirmation_cancelled)
        self.settings_view.provider_changed.connect(self._on_provider_changed)

    @property
    def bridge(self) -> Optional[UIBridge]:
        """Return connected UIBridge."""
        return self._bridge

    def _connect_bridge(self, bridge: UIBridge) -> None:
        """Connect UIBridge signals to UI component update slots."""
        bridge.state_changed.connect(self._on_state_changed)
        bridge.amplitude_updated.connect(self._on_amplitude_updated)
        bridge.snapshot_updated.connect(self._on_snapshot_updated)
        bridge.command_completed.connect(self._on_command_completed)
        bridge.command_failed.connect(self._on_command_failed)
        bridge.telemetry_updated.connect(self._on_telemetry_updated)
        bridge.plan_updated.connect(self._on_plan_updated)
        bridge.ai_telemetry_updated.connect(self._on_ai_telemetry_updated)
        bridge.execution_history_updated.connect(self._on_execution_history_updated)
        bridge.system_diagnostics_updated.connect(self._on_system_diagnostics_updated)
        bridge.conversation_history_loaded.connect(self._on_conversation_history_loaded)
        bridge.cognitive_stage_changed.connect(self._on_cognitive_stage_changed)

    def _on_conversation_history_loaded(self, records: list) -> None:
        """Populate conversation card with loaded historical turns."""
        self.left_panel.conversation_card.load_history(records)

    def _on_cognitive_stage_changed(self, stage: str, detail: str) -> None:
        """Update center panel cognitive stage chip."""
        self.center_panel.set_cognitive_stage(stage, detail=detail)

    def _on_navigation_changed(self, tab_name: str) -> None:
        """Switch views instantaneously on Qt GUI main thread."""
        target_map = {
            "HOME": 0,
            "ACTIVITY": 1,
            "SYSTEM": 2,
            "SETTINGS": 3,
        }
        idx = target_map.get((tab_name or "HOME").upper(), 0)
        self.view_stack.setCurrentIndex(idx)
        self.bottom_bar.set_status_message(f"Active View: {tab_name.upper()}")

        # Refresh telemetry if navigating to live data screens
        if self._bridge is not None:
            if idx == 2:  # SYSTEM
                self._bridge.fetch_system_diagnostics()
            elif idx == 3:  # SETTINGS
                self._bridge.fetch_ai_telemetry()

    def _on_provider_changed(self, provider_id: str) -> None:
        """Handle AI provider switch via safe existing ProviderRouter infrastructure."""
        if self._bridge is not None:
            self._bridge.set_ai_provider(provider_id)
        self.bottom_bar.set_status_message(f"AI Provider switched to: {provider_id.upper()}")

    def _on_ai_telemetry_updated(self, telemetry: dict) -> None:
        """Update center panel AI badge and settings view from live cognition telemetry."""
        self.center_panel.ai_badge.update_telemetry(telemetry)
        self.settings_view.update_ai_telemetry(telemetry)

    def _on_execution_history_updated(self, records: list) -> None:
        """Update recent execution card and activity view from real execution records."""
        self.center_panel.recent_execution.update_executions(records)
        self.activity_view.update_executions(records)

    def _on_system_diagnostics_updated(self, data: dict) -> None:
        """Update system diagnostics view from comprehensive host telemetry."""
        self.system_view.update_diagnostics(data)

    def _on_state_changed(self, state_name: str) -> None:
        """Handle state change from backend."""
        self.center_panel.set_state(state_name)
        self.bottom_bar.set_status_message(f"Assistant State: {state_name}")
        is_error = (state_name == "ERROR")
        self.header.set_system_status(not is_error, "ERROR" if is_error else "SYSTEM ONLINE")

        if state_name in ("THINKING", "PLANNING", "EXECUTING"):
            self.left_panel.conversation_card.show_typing_indicator()
        elif state_name in ("IDLE", "ERROR", "SPEAKING"):
            self.left_panel.conversation_card.hide_typing_indicator()

    def _on_amplitude_updated(self, amplitude: float) -> None:
        """Handle audio telemetry update."""
        self.center_panel.set_amplitude(amplitude)

    def _on_telemetry_updated(self, metrics: dict) -> None:
        """Update system status dials and system info card from live telemetry."""
        self.right_panel.system_status_card.update_telemetry(metrics)
        self.right_panel.system_info_card.update_info(metrics)

    def _on_plan_updated(self, plan_info: object) -> None:
        """Update current task card and activity view from active planner notifications."""
        if isinstance(plan_info, dict):
            title = str(plan_info.get("title", ""))
            prog = float(plan_info.get("progress", 0.0))
            steps = plan_info.get("steps")
            self.right_panel.current_task_card.update_task_state(title, prog, steps)
            self.activity_view.update_active_plan(plan_info)

    def _on_confirmation_confirmed(self, confirmation_id: str) -> None:
        """Resolve pending confirmation with approval through safe existing bridge API."""
        if self._bridge is not None:
            self._bridge.resolve_confirmation(
                confirmation_id,
                approved=True,
                decided_by="gui_operator",
                reason="Authorized by operator via HUD ConfirmationCard",
            )

    def _on_confirmation_cancelled(self, confirmation_id: str) -> None:
        """Resolve pending confirmation with rejection through safe existing bridge API."""
        if self._bridge is not None:
            self._bridge.resolve_confirmation(
                confirmation_id,
                approved=False,
                decided_by="gui_operator",
                reason="Rejected by operator via HUD ConfirmationCard",
            )

    def _on_snapshot_updated(self, snapshot: AssistantSnapshot) -> None:
        """Handle full snapshot update."""
        # 1. Check operator confirmation gateway
        if snapshot.state == AssistantState.AWAITING_CONFIRMATION and snapshot.pending_confirmation is not None:
            self.center_panel.show_confirmation(snapshot.pending_confirmation)
        else:
            self.center_panel.hide_confirmation()

        # 2. Update task progress if active
        if snapshot.current_command:
            self.right_panel.current_task_card.set_task(
                snapshot.current_command,
                snapshot.task_progress,
            )

        if snapshot.status_message:
            self.bottom_bar.set_status_message(snapshot.status_message)

    def _on_command_dispatched(self, command_text: str) -> None:
        """Handle command submission from quick actions or bottom command input."""
        now_str = datetime.datetime.now().strftime("%I:%M %p")
        self.left_panel.conversation_card.add_message("You", command_text, now_str, progressive=False)
        self.left_panel.conversation_card.show_typing_indicator()
        self.center_panel.set_cognitive_stage("ANALYZING", "Parsing command")

        if self._bridge is not None:
            self._bridge.submit_command(command_text)
        else:
            # Standalone visual fallback
            self.center_panel.set_state("EXECUTING")
            self.bottom_bar.set_status_message(f"Executing: {command_text}...")

    def _on_voice_toggled(self) -> None:
        """Handle microphone toggle."""
        if self._bridge is not None:
            self._bridge.start_voice_interaction()
        else:
            current = self.center_panel.arc_reactor.state
            new_state = "IDLE" if current == "LISTENING" else "LISTENING"
            self.center_panel.set_state(new_state)

    def _on_command_completed(self, result: object) -> None:
        """Handle successful command execution."""
        now_str = datetime.datetime.now().strftime("%I:%M %p")
        if result is None:
            res_text = "Operation completed successfully."
        elif hasattr(result, "to_user_message") and callable(result.to_user_message):
            res_text = result.to_user_message()
        elif hasattr(result, "message") and isinstance(result.message, str):
            res_text = result.message
        else:
            res_text = str(result)
        self.left_panel.conversation_card.hide_typing_indicator()
        self.left_panel.conversation_card.add_message("J.A.R.V.I.S", res_text, now_str, progressive=True)
        self.center_panel.set_state("IDLE")
        self.center_panel.set_cognitive_stage("STANDBY", "Ready")

    def _on_command_failed(self, error_message: str) -> None:
        """Handle command execution error."""
        now_str = datetime.datetime.now().strftime("%I:%M %p")
        self.left_panel.conversation_card.hide_typing_indicator()
        self.left_panel.conversation_card.add_message(
            "J.A.R.V.I.S",
            f"Error: {error_message}",
            now_str,
            is_error=True,
            progressive=False,
        )
        self.center_panel.set_state("ERROR")
        self.center_panel.set_cognitive_stage("STANDBY", "Error")
        self.bottom_bar.set_status_message(f"Error: {error_message}")

    def _on_state_selected(self, state_name: str) -> None:
        """Handle manual state selection from state indicator pills."""
        self.center_panel.set_state(state_name)
        self.bottom_bar.set_status_message(f"Selected State: {state_name}")

    def closeEvent(self, event: object) -> None:
        """Clean up bridge resources upon window close."""
        if self._bridge is not None:
            self._bridge.close()
        super().closeEvent(event)
