"""Center Content Panel for J.A.R.V.I.S HUD.

Houses the iconic ArcReactorCore, WaveformVisualizer, state selector pills,
AI Core telemetry badge, and the Recent Execution stepper card matching
jarvis_ui_reference.png.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.ui.components.arc_reactor import ArcReactorCore
from app.ui.components.glass_panel import GlassPanel
from app.ui.components.waveform import WaveformVisualizer
from app.ui.styles import JarvisTheme, ensure_fonts_loaded, get_state_color


class StateIndicatorPills(QWidget):
    """Vertical list of state indicator pills on the left of the Arc Reactor."""

    state_selected = Signal(str)

    STATES = ["IDLE", "LISTENING", "THINKING", "EXECUTING", "SPEAKING", "ERROR"]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize StateIndicatorPills."""
        super().__init__(parent)
        ensure_fonts_loaded()

        self._active_state = "IDLE"
        self._buttons: dict[str, QPushButton] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        icons = {
            "IDLE": "◉",
            "LISTENING": "🎙",
            "THINKING": "⚙",
            "EXECUTING": "⚡",
            "SPEAKING": "🔊",
            "ERROR": "⊗",
        }

        for st in self.STATES:
            btn = QPushButton(f"{icons.get(st, '•')}  {st}")
            btn.setFixedSize(110, 32)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            font = QFont(JarvisTheme.FONT_FAMILY, 8, QFont.Weight.Bold)
            font.setFamilies(JarvisTheme.FONT_FAMILIES)
            font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.0)
            btn.setFont(font)
            btn.clicked.connect(lambda checked=False, s=st: self.set_state(s, emit=True))
            self._buttons[st] = btn
            layout.addWidget(btn)

        self._update_styles()

    def set_state(self, state_name: str, emit: bool = False) -> None:
        """Update active state visually and optionally emit signal."""
        upper = (state_name or "IDLE").upper()
        if upper in self._buttons:
            self._active_state = upper
            self._update_styles()
            if emit:
                self.state_selected.emit(upper)

    def _update_styles(self) -> None:
        """Update buttons stylesheet according to active state."""
        for st, btn in self._buttons.items():
            if st == self._active_state:
                color = get_state_color(st)
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: #0c2b4d;
                        color: {color};
                        border: 1.5px solid {color};
                        border-radius: 16px;
                        text-align: left;
                        padding-left: 12px;
                    }}
                """)
            else:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: #06152b;
                        color: {JarvisTheme.TEXT_MUTED};
                        border: 1px solid #0a213e;
                        border-radius: 16px;
                        text-align: left;
                        padding-left: 12px;
                    }}
                    QPushButton:hover {{
                        color: {JarvisTheme.TEXT_PRIMARY};
                        border: 1px solid #143d70;
                    }}
                """)


class AICoreBadge(GlassPanel):
    """Status card displaying AI Core version and active engine subsystems."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize AICoreBadge."""
        super().__init__(parent, border_radius=10)
        self.setFixedWidth(135)
        self.setStyleSheet(f"""
            QFrame#GlassPanel {{
                background-color: #040e1f;
                border: 1px solid {JarvisTheme.BG_CARD_BORDER};
                border-radius: 10px;
            }}
        """)
        self.content_layout.setContentsMargins(10, 8, 10, 8)
        self.content_layout.setSpacing(5)

        # Header: AI CORE v2.4.8
        header_layout = QHBoxLayout()
        header_layout.setSpacing(4)
        title_lbl = QLabel("AI CORE")
        title_font = QFont(JarvisTheme.FONT_FAMILY, 8, QFont.Weight.Bold)
        title_font.setFamilies(JarvisTheme.FONT_FAMILIES)
        title_lbl.setFont(title_font)
        title_lbl.setStyleSheet(f"color: {JarvisTheme.CYAN_PRIMARY}; background: transparent; border: none;")
        header_layout.addWidget(title_lbl)

        ver_lbl = QLabel("v2.4.8")
        ver_font = QFont(JarvisTheme.FONT_FAMILY, 7)
        ver_lbl.setFont(ver_font)
        ver_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
        header_layout.addWidget(ver_lbl)
        header_layout.addStretch(1)
        self.content_layout.addLayout(header_layout)

        # Subsystems rows
        subsystems = ["Voice Engine", "AI Model", "Task Planner", "System Skills"]
        for sub in subsystems:
            row = QHBoxLayout()
            row.setSpacing(6)
            dot = QLabel("●")
            dot.setStyleSheet("color: #22c55e; font-size: 8px; background: transparent; border: none;")
            row.addWidget(dot)

            lbl = QLabel(sub)
            lbl_font = QFont(JarvisTheme.FONT_FAMILY, 7, QFont.Weight.Normal)
            lbl_font.setFamilies(JarvisTheme.FONT_FAMILIES)
            lbl.setFont(lbl_font)
            lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_PRIMARY}; background: transparent; border: none;")
            row.addWidget(lbl)
            row.addStretch(1)
            self.content_layout.addLayout(row)


class StepItem(QWidget):
    """Single execution step row in the Recent Execution card."""

    def __init__(
        self,
        step_num: int,
        title: str,
        time_ago: str,
        is_done: bool = False,
        is_active: bool = False,
        progress: float = 0.0,
        parent: Optional[QWidget] = None,
    ) -> None:
        """Initialize StepItem."""
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(3)

        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        # Step circle number/check
        badge = QLabel("✓" if is_done else str(step_num))
        badge.setFixedSize(18, 18)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if is_done:
            badge.setStyleSheet("""
                QLabel {
                    background-color: #042416;
                    color: #22c55e;
                    border: 1px solid #14532d;
                    border-radius: 9px;
                    font-size: 9px;
                    font-weight: bold;
                }
            """)
        elif is_active:
            badge.setStyleSheet(f"""
                QLabel {{
                    background-color: #0c2b4d;
                    color: {JarvisTheme.CYAN_PRIMARY};
                    border: 1px solid {JarvisTheme.CYAN_PRIMARY};
                    border-radius: 9px;
                    font-size: 9px;
                    font-weight: bold;
                }}
            """)
        else:
            badge.setStyleSheet(f"""
                QLabel {{
                    background-color: #06152b;
                    color: {JarvisTheme.TEXT_MUTED};
                    border: 1px solid #0a213e;
                    border-radius: 9px;
                    font-size: 9px;
                }}
            """)
        top_row.addWidget(badge)

        title_lbl = QLabel(title)
        title_font = QFont(JarvisTheme.FONT_FAMILY, 8, QFont.Weight.Medium if is_active else QFont.Weight.Normal)
        title_font.setFamilies(JarvisTheme.FONT_FAMILIES)
        title_lbl.setFont(title_font)
        title_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_PRIMARY if (is_done or is_active) else JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
        top_row.addWidget(title_lbl)

        top_row.addStretch(1)

        time_lbl = QLabel(time_ago)
        time_font = QFont(JarvisTheme.FONT_FAMILY, 7)
        time_lbl.setFont(time_font)
        time_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
        top_row.addWidget(time_lbl)

        layout.addLayout(top_row)

        if is_active and progress > 0.0:
            prog_row = QHBoxLayout()
            prog_row.setContentsMargins(26, 0, 0, 0)
            prog_row.setSpacing(6)

            pbar = QProgressBar()
            pbar.setFixedHeight(4)
            pbar.setRange(0, 100)
            pbar.setValue(int(progress * 100))
            pbar.setTextVisible(False)
            pbar.setStyleSheet(f"""
                QProgressBar {{
                    background-color: #0b2246;
                    border: none;
                    border-radius: 2px;
                }}
                QProgressBar::chunk {{
                    background-color: {JarvisTheme.CYAN_PRIMARY};
                    border-radius: 2px;
                }}
            """)
            prog_row.addWidget(pbar, 1)

            pct_lbl = QLabel(f"{int(progress * 100)}%")
            pct_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 7, QFont.Weight.Bold))
            pct_lbl.setStyleSheet(f"color: {JarvisTheme.CYAN_PRIMARY}; background: transparent; border: none;")
            prog_row.addWidget(pct_lbl)

            layout.addLayout(prog_row)


class RecentExecutionCard(GlassPanel):
    """Recent task execution breakdown card matching reference."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize RecentExecutionCard."""
        super().__init__(parent)

        badge_lbl = QLabel("2 / 4")
        badge_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 8, QFont.Weight.Bold))
        badge_lbl.setStyleSheet(f"color: {JarvisTheme.CYAN_PRIMARY}; background: transparent; border: none;")
        self.add_card_header("RECENT EXECUTION", icon_text="⏱", right_badge=badge_lbl)

        # Content split horizontally: Left steps list, Right current skill
        h_split = QHBoxLayout()
        h_split.setContentsMargins(0, 0, 0, 0)
        h_split.setSpacing(16)

        # Left: 4 Steps
        steps_col = QVBoxLayout()
        steps_col.setSpacing(4)
        steps_col.addWidget(StepItem(1, "Understand the request", "2 mins ago", is_done=True, parent=self))
        steps_col.addWidget(StepItem(2, "Open Chrome", "3 mins ago", is_done=True, parent=self))
        steps_col.addWidget(StepItem(3, "Search for weather", "1 hour ago", is_active=True, progress=0.75, parent=self))
        steps_col.addWidget(StepItem(4, "Show results", "", is_done=False, parent=self))
        h_split.addLayout(steps_col, 3)

        # Right: Current Skill Sub-panel
        skill_panel = QFrame()
        skill_panel.setStyleSheet(f"""
            QFrame {{
                background-color: #040e1f;
                border: 1px solid {JarvisTheme.BG_CARD_BORDER};
                border-radius: 8px;
            }}
        """)
        skill_layout = QVBoxLayout(skill_panel)
        skill_layout.setContentsMargins(10, 8, 10, 8)
        skill_layout.setSpacing(4)

        skill_tag = QLabel("Current Skill")
        skill_tag.setFont(QFont(JarvisTheme.FONT_FAMILY, 7))
        skill_tag.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
        skill_layout.addWidget(skill_tag)

        skill_row = QHBoxLayout()
        skill_row.setSpacing(6)
        skill_icon = QLabel("🌐")
        skill_icon.setStyleSheet(f"color: {JarvisTheme.CYAN_PRIMARY}; font-size: 14px; background: transparent; border: none;")
        skill_row.addWidget(skill_icon)

        skill_title = QLabel("Web Search")
        skill_title.setFont(QFont(JarvisTheme.FONT_FAMILY, 9, QFont.Weight.Bold))
        skill_title.setStyleSheet(f"color: {JarvisTheme.TEXT_PRIMARY}; background: transparent; border: none;")
        skill_row.addWidget(skill_title)
        skill_row.addStretch(1)
        skill_layout.addLayout(skill_row)

        skill_desc = QLabel("Fetching weather data\nfrom your location...")
        skill_desc.setFont(QFont(JarvisTheme.FONT_FAMILY, 7))
        skill_desc.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
        skill_layout.addWidget(skill_desc)
        skill_layout.addStretch(1)

        h_split.addWidget(skill_panel, 2)

        self.content_layout.addLayout(h_split)


class CenterPanel(QWidget):
    """Central interactive container displaying ArcReactorCore, waveform, and status."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize CenterPanel."""
        super().__init__(parent)
        ensure_fonts_loaded()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(8)

        # Upper Area: Left State Pills, Center ArcReactorCore, Right AI Core Badge
        upper_layout = QHBoxLayout()
        upper_layout.setContentsMargins(0, 0, 0, 0)
        upper_layout.setSpacing(8)

        # Left Column: State Pills
        self.state_pills = StateIndicatorPills(self)
        upper_layout.addWidget(self.state_pills, 0, alignment=Qt.AlignmentFlag.AlignVCenter)

        # Middle Column: ArcReactorCore prominently centered
        reactor_container = QVBoxLayout()
        reactor_container.setContentsMargins(0, 0, 0, 0)
        reactor_container.setSpacing(2)

        # Top Listening Pill
        top_pill_layout = QHBoxLayout()
        self._top_status_pill = QLabel("ıll  Listening...  🎙")
        self._top_status_pill.setFont(QFont(JarvisTheme.FONT_FAMILY, 8, QFont.Weight.Medium))
        self._top_status_pill.setStyleSheet(f"""
            QLabel {{
                color: {JarvisTheme.CYAN_PRIMARY};
                background-color: #061933;
                border: 1px solid #0f3964;
                border-radius: 12px;
                padding: 4px 14px;
            }}
        """)
        top_pill_layout.addStretch(1)
        top_pill_layout.addWidget(self._top_status_pill)
        top_pill_layout.addStretch(1)
        reactor_container.addLayout(top_pill_layout)

        # The iconic ArcReactorCore
        self.arc_reactor = ArcReactorCore(self)
        reactor_container.addWidget(self.arc_reactor, 1, alignment=Qt.AlignmentFlag.AlignCenter)

        # Directly beneath: WaveformVisualizer
        self.waveform = WaveformVisualizer(self, num_bars=48)
        reactor_container.addWidget(self.waveform, 0)

        upper_layout.addLayout(reactor_container, 1)

        # Right Column: AI Core Telemetry Badge
        ai_badge_col = QVBoxLayout()
        ai_badge_col.setContentsMargins(0, 20, 0, 0)
        self.ai_badge = AICoreBadge(self)
        ai_badge_col.addWidget(self.ai_badge, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        ai_badge_col.addStretch(1)
        upper_layout.addLayout(ai_badge_col, 0)

        layout.addLayout(upper_layout, 3)

        # Lower Area: Recent Execution Card
        self.recent_execution = RecentExecutionCard(self)
        layout.addWidget(self.recent_execution, 1)

    def set_state(self, state_name: str) -> None:
        """Update operational state across reactor, waveform, and pills."""
        upper = (state_name or "IDLE").upper()
        self.arc_reactor.set_state(upper)
        self.waveform.set_state(upper)
        self.state_pills.set_state(upper)
        self._top_status_pill.setText(f"ıll  {upper.capitalize()}...  🎙")

    def set_amplitude(self, amp: float) -> None:
        """Update microphone amplitude across reactor and waveform."""
        self.arc_reactor.set_amplitude(amp)
        self.waveform.set_amplitude(amp)
