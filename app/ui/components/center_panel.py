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
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.core.state import PendingConfirmation
from app.ui.components.arc_reactor import ArcReactorCore
from app.ui.components.confirmation_dialog import ConfirmationCard
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
    """Status card displaying live AI Core provider, model, latency, and health telemetry."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize AICoreBadge."""
        super().__init__(parent, border_radius=10)
        self.setFixedWidth(145)
        self.setStyleSheet(f"""
            QFrame#GlassPanel {{
                background-color: #040e1f;
                border: 1px solid {JarvisTheme.BG_CARD_BORDER};
                border-radius: 10px;
            }}
        """)
        self.content_layout.setContentsMargins(10, 8, 10, 8)
        self.content_layout.setSpacing(4)

        # Header: AI CORE + Provider Name
        header_layout = QHBoxLayout()
        header_layout.setSpacing(4)
        title_lbl = QLabel("AI CORE")
        title_font = QFont(JarvisTheme.FONT_FAMILY, 8, QFont.Weight.Bold)
        title_font.setFamilies(JarvisTheme.FONT_FAMILIES)
        title_lbl.setFont(title_font)
        title_lbl.setStyleSheet(f"color: {JarvisTheme.CYAN_PRIMARY}; background: transparent; border: none;")
        header_layout.addWidget(title_lbl)

        self._provider_badge = QLabel("N/A")
        ver_font = QFont(JarvisTheme.FONT_FAMILY, 7, QFont.Weight.Bold)
        self._provider_badge.setFont(ver_font)
        self._provider_badge.setStyleSheet("color: #38bdf8; background: #07203b; border-radius: 4px; padding: 1px 4px; border: none;")
        header_layout.addWidget(self._provider_badge)
        header_layout.addStretch(1)
        self.content_layout.addLayout(header_layout)

        # Live telemetry rows: Mode, Model, Latency, Health
        self._mode_lbl = QLabel("N/A")
        self._model_lbl = QLabel("N/A")
        self._latency_lbl = QLabel("N/A")
        self._health_dot = QLabel("●")
        self._health_lbl = QLabel("ONLINE")

        rows = [
            ("Target", self._mode_lbl),
            ("Model", self._model_lbl),
            ("Latency", self._latency_lbl),
        ]

        for label_text, widget in rows:
            row = QHBoxLayout()
            row.setSpacing(6)
            lbl = QLabel(label_text)
            lbl_font = QFont(JarvisTheme.FONT_FAMILY, 7)
            lbl.setFont(lbl_font)
            lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
            row.addWidget(lbl)
            row.addStretch(1)

            widget.setFont(QFont(JarvisTheme.FONT_FAMILY, 7, QFont.Weight.Bold))
            widget.setStyleSheet(f"color: {JarvisTheme.TEXT_PRIMARY}; background: transparent; border: none;")
            row.addWidget(widget)
            self.content_layout.addLayout(row)

        # Health Row
        health_row = QHBoxLayout()
        health_row.setSpacing(4)
        self._health_dot.setStyleSheet("color: #22c55e; font-size: 8px; background: transparent; border: none;")
        health_row.addWidget(self._health_dot)

        self._health_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 7, QFont.Weight.Bold))
        self._health_lbl.setStyleSheet("color: #22c55e; background: transparent; border: none;")
        health_row.addWidget(self._health_lbl)
        health_row.addStretch(1)
        self.content_layout.addLayout(health_row)

    def update_telemetry(self, data: dict[str, Any]) -> None:
        """Update live AI telemetry metrics dynamically.

        Args:
            data: Telemetry dictionary with provider, model, mode, latency, health.
        """
        provider = str(data.get("active_provider") or data.get("provider") or "N/A").upper()
        self._provider_badge.setText(provider)

        mode = str(data.get("mode") or ("CLOUD" if provider == "GEMINI" else ("LOCAL" if provider == "OLLAMA" else "N/A"))).upper()
        self._mode_lbl.setText(mode)

        model = str(data.get("active_model") or data.get("model") or "N/A")
        # Truncate model name if too long for compact badge
        if "/" in model and model != "N/A":
            short_model = model.split("/")[-1]
        else:
            short_model = model

        if len(short_model) > 12:
            short_model = short_model[:10] + ".."
        self._model_lbl.setText(short_model)

        latency = data.get("reasoning_latency_ms")
        if latency is not None and isinstance(latency, (int, float)):
            self._latency_lbl.setText(f"{latency:.0f} ms")
        elif "latency" in data and data["latency"]:
            self._latency_lbl.setText(str(data["latency"]))
        else:
            self._latency_lbl.setText("N/A")

        health = str(data.get("health") or data.get("status") or "ONLINE").upper()
        if "ONLINE" in health or "HEALTHY" in health:
            self._health_dot.setStyleSheet("color: #22c55e; font-size: 8px; background: transparent; border: none;")
            self._health_lbl.setText(health)
            self._health_lbl.setStyleSheet("color: #22c55e; background: transparent; border: none;")
        else:
            self._health_dot.setStyleSheet("color: #eab308; font-size: 8px; background: transparent; border: none;")
            self._health_lbl.setText(health)
            self._health_lbl.setStyleSheet("color: #eab308; background: transparent; border: none;")


class CognitiveStageChip(GlassPanel):
    """Real-time tactical indicator of the assistant's cognitive execution phase."""

    STAGES = {
        "STANDBY": ("#22c55e", "●", "STANDBY"),
        "IDLE": ("#22c55e", "●", "STANDBY"),
        "ANALYZING": ("#38bdf8", "⚙", "ANALYZING"),
        "ROUTING": ("#818cf8", "⤹", "ROUTING"),
        "EXECUTING": ("#f59e0b", "⚡", "EXECUTING"),
        "SYNTHESIZING": ("#06b6d4", "✦", "SYNTHESIZING"),
        "ERROR": ("#ef4444", "⊗", "ERROR"),
    }

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize CognitiveStageChip."""
        super().__init__(parent, border_radius=10)
        self.setFixedWidth(145)
        self.setStyleSheet(f"""
            QFrame#GlassPanel {{
                background-color: #040e1f;
                border: 1px solid {JarvisTheme.BG_CARD_BORDER};
                border-radius: 10px;
            }}
        """)
        self.content_layout.setContentsMargins(10, 6, 10, 6)
        self.content_layout.setSpacing(3)

        # Header row: COGNITION
        header_row = QHBoxLayout()
        header_row.setSpacing(4)
        title_lbl = QLabel("COGNITION")
        title_font = QFont(JarvisTheme.FONT_FAMILY, 7, QFont.Weight.Bold)
        title_font.setFamilies(JarvisTheme.FONT_FAMILIES)
        title_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.8)
        title_lbl.setFont(title_font)
        title_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
        header_row.addWidget(title_lbl)
        header_row.addStretch(1)
        self.content_layout.addLayout(header_row)

        # Stage row: Icon + Stage text
        stage_row = QHBoxLayout()
        stage_row.setSpacing(6)
        self._icon_lbl = QLabel("●")
        self._icon_lbl.setStyleSheet("color: #22c55e; font-size: 9px; background: transparent; border: none;")
        stage_row.addWidget(self._icon_lbl)

        self._stage_lbl = QLabel("STANDBY")
        stage_font = QFont(JarvisTheme.FONT_FAMILY, 8, QFont.Weight.Bold)
        stage_font.setFamilies(JarvisTheme.FONT_FAMILIES)
        self._stage_lbl.setFont(stage_font)
        self._stage_lbl.setStyleSheet("color: #22c55e; background: transparent; border: none;")
        stage_row.addWidget(self._stage_lbl)
        stage_row.addStretch(1)
        self.content_layout.addLayout(stage_row)

        # Detail subtitle label
        self._detail_lbl = QLabel("Ready")
        self._detail_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 7))
        self._detail_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
        self.content_layout.addWidget(self._detail_lbl)

    def set_stage(self, stage_name: str, detail: Optional[str] = None) -> None:
        """Update active cognitive stage and detail label."""
        upper = (stage_name or "STANDBY").upper()
        color, icon, display_name = self.STAGES.get(upper, (JarvisTheme.CYAN_PRIMARY, "●", upper))

        self._icon_lbl.setText(icon)
        self._icon_lbl.setStyleSheet(f"color: {color}; font-size: 9px; background: transparent; border: none;")
        self._stage_lbl.setText(display_name)
        self._stage_lbl.setStyleSheet(f"color: {color}; background: transparent; border: none;")

        if detail:
            clean_detail = str(detail).strip()
            if len(clean_detail) > 18:
                clean_detail = clean_detail[:16] + ".."
            self._detail_lbl.setText(clean_detail)
        elif upper in ("IDLE", "STANDBY"):
            self._detail_lbl.setText("Ready")
        else:
            self._detail_lbl.setText("Active")


class StepItem(QWidget):
    """Single execution step row in the Recent Execution card."""

    def __init__(
        self,
        step_num: int,
        title: str,
        time_ago: str,
        is_done: bool = False,
        is_active: bool = False,
        is_failed: bool = False,
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

        # Step circle number/check/cross
        badge = QLabel()
        badge.setFixedSize(18, 18)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if is_failed:
            badge.setText("✕")
            badge.setStyleSheet("""
                QLabel {
                    background-color: #2b0b0b;
                    color: #ef4444;
                    border: 1px solid #7f1d1d;
                    border-radius: 9px;
                    font-size: 9px;
                    font-weight: bold;
                }
            """)
        elif is_done:
            badge.setText("✓")
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
            badge.setText(str(step_num))
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
            badge.setText(str(step_num))
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
    """Recent task execution breakdown card connecting to real backend execution records."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize RecentExecutionCard."""
        super().__init__(parent)

        self._badge_lbl = QLabel("0")
        self._badge_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 8, QFont.Weight.Bold))
        self._badge_lbl.setStyleSheet(f"color: {JarvisTheme.CYAN_PRIMARY}; background: transparent; border: none;")
        self.add_card_header("RECENT EXECUTION", icon_text="⏱", right_badge=self._badge_lbl)

        # Content split horizontally: Left steps list, Right current skill
        h_split = QHBoxLayout()
        h_split.setContentsMargins(0, 0, 0, 0)
        h_split.setSpacing(16)

        # Left: Steps Column Container
        self._steps_container = QWidget()
        self._steps_container.setStyleSheet("background: transparent;")
        self._steps_layout = QVBoxLayout(self._steps_container)
        self._steps_layout.setContentsMargins(0, 0, 0, 0)
        self._steps_layout.setSpacing(4)

        self._empty_label = QLabel("No recent executions")
        self._empty_label.setFont(QFont(JarvisTheme.FONT_FAMILY, 8))
        self._empty_label.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self._steps_layout.addWidget(self._empty_label)

        h_split.addWidget(self._steps_container, 3)

        # Right: Current Skill Sub-panel
        self.skill_panel = QFrame()
        self.skill_panel.setStyleSheet(f"""
            QFrame {{
                background-color: #040e1f;
                border: 1px solid {JarvisTheme.BG_CARD_BORDER};
                border-radius: 8px;
            }}
        """)
        skill_layout = QVBoxLayout(self.skill_panel)
        skill_layout.setContentsMargins(10, 8, 10, 8)
        skill_layout.setSpacing(4)

        skill_tag = QLabel("Current Skill")
        skill_tag.setFont(QFont(JarvisTheme.FONT_FAMILY, 7))
        skill_tag.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
        skill_layout.addWidget(skill_tag)

        skill_row = QHBoxLayout()
        skill_row.setSpacing(6)
        self.skill_icon = QLabel("⚡")
        self.skill_icon.setStyleSheet(f"color: {JarvisTheme.CYAN_PRIMARY}; font-size: 14px; background: transparent; border: none;")
        skill_row.addWidget(self.skill_icon)

        self.skill_title = QLabel("Idle")
        self.skill_title.setFont(QFont(JarvisTheme.FONT_FAMILY, 9, QFont.Weight.Bold))
        self.skill_title.setStyleSheet(f"color: {JarvisTheme.TEXT_PRIMARY}; background: transparent; border: none;")
        skill_row.addWidget(self.skill_title)
        skill_row.addStretch(1)
        skill_layout.addLayout(skill_row)

        self.skill_desc = QLabel("Awaiting next operator command...")
        self.skill_desc.setFont(QFont(JarvisTheme.FONT_FAMILY, 7))
        self.skill_desc.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
        skill_layout.addWidget(self.skill_desc)
        skill_layout.addStretch(1)

        h_split.addWidget(self.skill_panel, 2)

        self.content_layout.addLayout(h_split)

    def update_executions(self, records: list[dict[str, Any]]) -> None:
        """Update recent execution steps dynamically with real execution records.

        Args:
            records: List of execution record dictionaries.
        """
        # Clear existing step widgets
        while self._steps_layout.count():
            item = self._steps_layout.takeAt(0)
            if item.widget():
                w = item.widget()
                w.setParent(None)
                w.deleteLater()

        if not records:
            self._empty_label = QLabel("No recent executions")
            self._empty_label.setFont(QFont(JarvisTheme.FONT_FAMILY, 8))
            self._empty_label.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
            self._steps_layout.addWidget(self._empty_label)
            self._badge_lbl.setText("0")
            self.skill_title.setText("Idle")
            self.skill_desc.setText("Awaiting next operator command...")
            self.skill_icon.setText("⚡")
            return

        total = len(records)
        completed = sum(1 for r in records if str(r.get("status", "")).lower() in ("completed", "done", "success"))
        self._badge_lbl.setText(f"{completed} / {total}")

        # Show up to 4 most recent records
        display_records = list(records)[-4:]
        for idx, rec in enumerate(display_records, start=1):
            title = str(rec.get("action") or rec.get("title") or rec.get("task_id") or "Task")
            target = rec.get("target")
            if target:
                title = f"{title} [{target}]"

            status = str(rec.get("status", "completed")).lower()
            is_done = status in ("completed", "done", "success")
            is_active = status in ("running", "executing", "in_progress")
            is_failed = status in ("failed", "error")

            duration = rec.get("duration", 0.0)
            dur_str = f"{float(duration):.2f}s" if isinstance(duration, (int, float)) and duration > 0 else "< 0.1s"

            step_widget = StepItem(
                step_num=idx,
                title=title,
                time_ago=dur_str,
                is_done=is_done,
                is_active=is_active,
                is_failed=is_failed,
                progress=1.0 if is_done else (0.5 if is_active else 0.0),
                parent=self,
            )
            self._steps_layout.addWidget(step_widget)

        # Update right skill sub-panel with the most recent execution
        latest = records[-1]
        action_name = str(latest.get("action") or latest.get("title") or "Task Execution")
        self.skill_title.setText(action_name[:18])
        target_info = latest.get("target") or latest.get("status") or "Completed"
        self.skill_desc.setText(f"Target: {target_info}\nStatus: {latest.get('status', 'completed')}")
        self.skill_icon.setText("🌐" if "web" in action_name.lower() or "search" in action_name.lower() else "⚙")


class CenterPanel(QWidget):
    """Central interactive container displaying ArcReactorCore, waveform, and status."""

    confirmation_confirmed = Signal(str)
    confirmation_cancelled = Signal(str)

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

        # Right Column: AI Core Telemetry Badge & Cognition Stage
        ai_badge_col = QVBoxLayout()
        ai_badge_col.setContentsMargins(0, 16, 0, 0)
        ai_badge_col.setSpacing(8)
        self.ai_badge = AICoreBadge(self)
        ai_badge_col.addWidget(self.ai_badge, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)

        self.cognitive_chip = CognitiveStageChip(self)
        ai_badge_col.addWidget(self.cognitive_chip, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)

        ai_badge_col.addStretch(1)
        upper_layout.addLayout(ai_badge_col, 0)

        layout.addLayout(upper_layout, 3)

        # Lower Area: QStackedWidget switching between RecentExecutionCard and ConfirmationCard
        self.lower_stack = QStackedWidget(self)
        self.recent_execution = RecentExecutionCard(self.lower_stack)
        self.confirmation_card = ConfirmationCard(self.lower_stack)
        self.lower_stack.addWidget(self.recent_execution)  # index 0: standard view
        self.lower_stack.addWidget(self.confirmation_card)  # index 1: confirmation view
        self.lower_stack.setCurrentIndex(0)
        layout.addWidget(self.lower_stack, 1)

        # Connect and forward confirmation signals
        self.confirmation_card.confirmed.connect(self.confirmation_confirmed.emit)
        self.confirmation_card.cancelled.connect(self.confirmation_cancelled.emit)

    def show_confirmation(self, conf: PendingConfirmation) -> None:
        """Display ConfirmationCard populated with PendingConfirmation data."""
        self.confirmation_card.load_confirmation(conf)
        self.lower_stack.setCurrentIndex(1)

    def hide_confirmation(self) -> None:
        """Hide ConfirmationCard and return to RecentExecutionCard."""
        self.lower_stack.setCurrentIndex(0)

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

    def set_cognitive_stage(self, stage_name: str, detail: Optional[str] = None) -> None:
        """Update active cognitive stage indicator."""
        self.cognitive_chip.set_stage(stage_name, detail=detail)
