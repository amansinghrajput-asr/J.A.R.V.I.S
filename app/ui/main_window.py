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


class JarvisMainWindow(QMainWindow):
    """Full J.A.R.V.I.S Sci-Fi Desktop HUD Window."""

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

        # 1. Top Header Bar
        self.header = HeaderBar(self)
        root_layout.addWidget(self.header, 0)

        # 2. Main Middle 3-Column Layout
        middle_container = QWidget(self)
        middle_container.setStyleSheet("background: transparent;")
        middle_layout = QHBoxLayout(middle_container)
        middle_layout.setContentsMargins(16, 12, 16, 8)
        middle_layout.setSpacing(14)

        # Left Column: Conversation + Quick Actions (Fixed ~290px)
        self.left_panel = LeftPanel(middle_container)
        middle_layout.addWidget(self.left_panel, 0)

        # Center Column: ArcReactorCore + Waveform + State Pills + Stepper (Expanding)
        self.center_panel = CenterPanel(middle_container)
        middle_layout.addWidget(self.center_panel, 1)

        # Right Column: Current Task + System Status + Settings (Fixed ~300px)
        self.right_panel = RightPanel(middle_container)
        middle_layout.addWidget(self.right_panel, 0)

        root_layout.addWidget(middle_container, 1)

        # 3. Bottom Command Bar
        self.bottom_bar = BottomBar(self)
        root_layout.addWidget(self.bottom_bar, 0)

        # Setup UIBridge if provided or created
        self._bridge = bridge
        if self._bridge is None and presentation_adapter is not None:
            self._bridge = UIBridge(presentation_adapter, parent=self)

        if self._bridge is not None:
            self._connect_bridge(self._bridge)

        # Connect user interaction signals from subcomponents to bridge
        self.left_panel.action_dispatched.connect(self._on_command_dispatched)
        self.bottom_bar.command_submitted.connect(self._on_command_dispatched)
        self.header.voice_toggled.connect(self._on_voice_toggled)
        self.bottom_bar.voice_toggled.connect(self._on_voice_toggled)
        self.center_panel.state_pills.state_selected.connect(self._on_state_selected)

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

    def _on_state_changed(self, state_name: str) -> None:
        """Handle state change from backend."""
        self.center_panel.set_state(state_name)
        self.bottom_bar.set_status_message(f"Assistant State: {state_name}")
        is_error = (state_name == "ERROR")
        self.header.set_system_status(not is_error, "ERROR" if is_error else "SYSTEM ONLINE")

    def _on_amplitude_updated(self, amplitude: float) -> None:
        """Handle audio telemetry update."""
        self.center_panel.set_amplitude(amplitude)

    def _on_snapshot_updated(self, snapshot: AssistantSnapshot) -> None:
        """Handle full snapshot update."""
        # Update task progress if active
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
        self.left_panel.conversation_card.add_message("You", command_text, now_str)

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
        res_text = str(result) if result is not None else "Operation completed successfully."
        self.left_panel.conversation_card.add_message("J.A.R.V.I.S", res_text, now_str)
        self.center_panel.set_state("IDLE")

    def _on_command_failed(self, error_message: str) -> None:
        """Handle command execution error."""
        now_str = datetime.datetime.now().strftime("%I:%M %p")
        self.left_panel.conversation_card.add_message(
            "J.A.R.V.I.S",
            f"Error: {error_message}",
            now_str,
        )
        self.center_panel.set_state("ERROR")
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
