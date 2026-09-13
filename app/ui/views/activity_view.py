"""Activity View for J.A.R.V.I.S Multi-View Tactical Command Center.

Displays real-time and historical execution logs, plan lifecycle telemetry,
task statuses, durations, and success/failure states.
"""

from __future__ import annotations

import datetime
from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.ui.components.glass_panel import GlassPanel
from app.ui.styles import JarvisTheme, ensure_fonts_loaded


class ExecutionHistoryItem(QFrame):
    """Single execution record card in the activity timeline."""

    def __init__(self, record: dict[str, Any], parent: Optional[QWidget] = None) -> None:
        """Initialize ExecutionHistoryItem with record metadata."""
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: #06152b;
                border: 1px solid #0d274c;
                border-radius: 8px;
            }}
            QFrame:hover {{
                border: 1px solid {JarvisTheme.CYAN_PRIMARY};
                background-color: #081a36;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        # Header row: Status Icon + Action Title + Target + Duration/Timestamp
        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        status = str(record.get("status", "completed")).lower()
        is_failed = status in ("failed", "error")
        is_running = status in ("running", "executing", "in_progress")
        is_completed = status in ("completed", "success", "done")

        # Status badge
        status_lbl = QLabel()
        status_lbl.setFixedSize(20, 20)
        status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if is_failed:
            status_lbl.setText("✕")
            status_lbl.setStyleSheet("""
                QLabel {
                    background-color: #2b0b0b;
                    color: #ef4444;
                    border: 1px solid #7f1d1d;
                    border-radius: 10px;
                    font-size: 9px;
                    font-weight: bold;
                }
            """)
        elif is_running:
            status_lbl.setText("◉")
            status_lbl.setStyleSheet(f"""
                QLabel {{
                    background-color: #0c2b4d;
                    color: {JarvisTheme.CYAN_PRIMARY};
                    border: 1px solid {JarvisTheme.CYAN_PRIMARY};
                    border-radius: 10px;
                    font-size: 9px;
                    font-weight: bold;
                }}
            """)
        else:
            status_lbl.setText("✓")
            status_lbl.setStyleSheet("""
                QLabel {
                    background-color: #042416;
                    color: #22c55e;
                    border: 1px solid #14532d;
                    border-radius: 10px;
                    font-size: 9px;
                    font-weight: bold;
                }
            """)
        top_row.addWidget(status_lbl)

        # Action / Title
        action = str(record.get("action") or record.get("title") or record.get("task_id") or "Unknown Task")
        action_lbl = QLabel(action)
        action_font = QFont(JarvisTheme.FONT_FAMILY, 9, QFont.Weight.Bold)
        action_font.setFamilies(JarvisTheme.FONT_FAMILIES)
        action_lbl.setFont(action_font)
        action_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_PRIMARY}; background: transparent; border: none;")
        top_row.addWidget(action_lbl)

        # Target (if present)
        target = record.get("target")
        if target:
            target_lbl = QLabel(f"[{str(target)}]")
            target_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 8))
            target_lbl.setStyleSheet(f"color: {JarvisTheme.CYAN_DIM}; background: transparent; border: none;")
            top_row.addWidget(target_lbl)

        top_row.addStretch(1)

        # Status text pill
        pill = QLabel(status.upper())
        pill_font = QFont(JarvisTheme.FONT_FAMILY, 7, QFont.Weight.Bold)
        pill_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.8)
        pill.setFont(pill_font)
        if is_failed:
            pill.setStyleSheet("color: #ef4444; background: #250a0a; border: 1px solid #7f1d1d; border-radius: 6px; padding: 2px 8px;")
        elif is_running:
            pill.setStyleSheet(f"color: {JarvisTheme.CYAN_PRIMARY}; background: #0c2b4d; border: 1px solid {JarvisTheme.CYAN_PRIMARY}; border-radius: 6px; padding: 2px 8px;")
        else:
            pill.setStyleSheet("color: #22c55e; background: #042416; border: 1px solid #14532d; border-radius: 6px; padding: 2px 8px;")
        top_row.addWidget(pill)

        # Duration badge
        duration_val = record.get("duration", 0.0)
        dur_text = f"{float(duration_val):.2f}s" if isinstance(duration_val, (int, float)) and duration_val > 0 else "< 0.1s"
        dur_lbl = QLabel(dur_text)
        dur_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 7, QFont.Weight.Medium))
        dur_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
        top_row.addWidget(dur_lbl)

        # Timestamp badge
        ts = record.get("timestamp")
        if ts:
            if isinstance(ts, (int, float)):
                ts_str = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            else:
                ts_str = str(ts)
            ts_lbl = QLabel(ts_str)
            ts_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 7))
            ts_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
            top_row.addWidget(ts_lbl)

        layout.addLayout(top_row)

        # Error details box (if failed)
        err = record.get("error")
        if err and is_failed:
            err_box = QLabel(f"Error: {str(err)}")
            err_box.setFont(QFont(JarvisTheme.FONT_FAMILY, 7))
            err_box.setWordWrap(True)
            err_box.setStyleSheet("""
                color: #f87171;
                background-color: #1a0505;
                border: 1px solid #5a1212;
                border-radius: 4px;
                padding: 4px 8px;
            """)
            layout.addWidget(err_box)


class ActivityView(QWidget):
    """Full-screen Activity & Execution telemetry view."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize ActivityView."""
        super().__init__(parent)
        ensure_fonts_loaded()

        self.setStyleSheet("background: transparent;")
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 16, 20, 16)
        main_layout.setSpacing(14)

        # 1. Header Banner
        header_card = GlassPanel(self, border_radius=12)
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(16, 12, 16, 12)
        header_layout.setSpacing(16)

        title_col = QVBoxLayout()
        title_col.setSpacing(4)
        title_lbl = QLabel("ACTIVITY & EXECUTION TELEMETRY")
        title_font = QFont(JarvisTheme.FONT_FAMILY, 14, QFont.Weight.Bold)
        title_font.setFamilies(JarvisTheme.FONT_FAMILIES)
        title_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.2)
        title_lbl.setFont(title_font)
        title_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_PRIMARY}; background: transparent; border: none;")
        title_col.addWidget(title_lbl)

        sub_lbl = QLabel("Live tactical execution logs, cognitive planner tasks, and outcome history.")
        sub_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 8))
        sub_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
        title_col.addWidget(sub_lbl)
        header_layout.addLayout(title_col, 1)

        # Metric Badges
        self._total_lbl = QLabel("0")
        self._success_rate_lbl = QLabel("100%")
        self._active_count_lbl = QLabel("0")

        for label_text, widget_val, color in (
            ("TOTAL EXECUTIONS", self._total_lbl, JarvisTheme.CYAN_PRIMARY),
            ("SUCCESS RATE", self._success_rate_lbl, "#22c55e"),
            ("ACTIVE TASKS", self._active_count_lbl, "#a855f7"),
        ):
            stat_card = QFrame()
            stat_card.setStyleSheet("""
                QFrame {
                    background-color: #06152b;
                    border: 1px solid #0d274c;
                    border-radius: 8px;
                    padding: 4px 12px;
                }
            """)
            stat_layout = QVBoxLayout(stat_card)
            stat_layout.setContentsMargins(8, 4, 8, 4)
            stat_layout.setSpacing(2)

            val_font = QFont(JarvisTheme.FONT_FAMILY, 12, QFont.Weight.Bold)
            widget_val.setFont(val_font)
            widget_val.setStyleSheet(f"color: {color}; background: transparent; border: none;")
            widget_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            stat_layout.addWidget(widget_val)

            lbl = QLabel(label_text)
            lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 6, QFont.Weight.Bold))
            lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            stat_layout.addWidget(lbl)

            header_layout.addWidget(stat_card)

        header_card.content_layout.addLayout(header_layout)
        main_layout.addWidget(header_card, 0)

        # 2. Main Content Split: Left Recent Executions Scroll, Right Active Plan Card
        content_split = QHBoxLayout()
        content_split.setSpacing(16)

        # Left: Timeline of Recent Executions
        timeline_panel = GlassPanel(self, border_radius=12)
        timeline_panel.add_card_header("EXECUTION TIMELINE", icon_text="⏱")

        # Scroll Area for Execution Items
        self._scroll_area = QScrollArea(timeline_panel)
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background: #040e1f;
                width: 6px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical {
                background: #0e3260;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical:hover {
                background: #00d2ff;
            }
        """)

        self._items_container = QWidget()
        self._items_container.setStyleSheet("background: transparent;")
        self._items_layout = QVBoxLayout(self._items_container)
        self._items_layout.setContentsMargins(0, 4, 8, 4)
        self._items_layout.setSpacing(8)
        self._items_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._empty_label = QLabel("No execution activity recorded")
        self._empty_label.setFont(QFont(JarvisTheme.FONT_FAMILY, 9, QFont.Weight.Medium))
        self._empty_label.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; padding: 40px; border: none;")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._items_layout.addWidget(self._empty_label)

        self._scroll_area.setWidget(self._items_container)
        timeline_panel.content_layout.addWidget(self._scroll_area)
        content_split.addWidget(timeline_panel, 3)

        # Right: Active Plan & Cognitive Status
        plan_panel = GlassPanel(self, border_radius=12)
        plan_panel.setFixedWidth(360)
        plan_panel.add_card_header("ACTIVE PLAN STATE", icon_text="🧠")

        plan_content = QVBoxLayout()
        plan_content.setContentsMargins(4, 4, 4, 4)
        plan_content.setSpacing(10)

        self._plan_title_lbl = QLabel("No active plan currently running.")
        self._plan_title_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 9, QFont.Weight.Bold))
        self._plan_title_lbl.setWordWrap(True)
        self._plan_title_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_PRIMARY}; background: transparent; border: none;")
        plan_content.addWidget(self._plan_title_lbl)

        # Progress bar
        self._plan_pbar = QProgressBar()
        self._plan_pbar.setFixedHeight(6)
        self._plan_pbar.setRange(0, 100)
        self._plan_pbar.setValue(0)
        self._plan_pbar.setTextVisible(False)
        self._plan_pbar.setStyleSheet(f"""
            QProgressBar {{
                background-color: #0b2246;
                border: none;
                border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background-color: {JarvisTheme.CYAN_PRIMARY};
                border-radius: 3px;
            }}
        """)
        plan_content.addWidget(self._plan_pbar)

        self._plan_pct_lbl = QLabel("0% Complete")
        self._plan_pct_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 7, QFont.Weight.Medium))
        self._plan_pct_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
        plan_content.addWidget(self._plan_pct_lbl)

        # Checklist for plan tasks
        self._plan_steps_container = QWidget()
        self._plan_steps_container.setStyleSheet("background: transparent;")
        self._plan_steps_layout = QVBoxLayout(self._plan_steps_container)
        self._plan_steps_layout.setContentsMargins(0, 4, 0, 4)
        self._plan_steps_layout.setSpacing(6)
        plan_content.addWidget(self._plan_steps_container)

        plan_content.addStretch(1)
        plan_panel.content_layout.addLayout(plan_content)
        content_split.addWidget(plan_panel, 0)

        main_layout.addLayout(content_split, 1)

    def update_executions(self, records: list[dict[str, Any]]) -> None:
        """Update recent executions list dynamically.

        Args:
            records: List of execution record dictionaries.
        """
        # Clear existing items
        while self._items_layout.count():
            item = self._items_layout.takeAt(0)
            if item.widget():
                w = item.widget()
                w.setParent(None)
                w.deleteLater()

        if not records:
            self._empty_label = QLabel("No execution activity recorded")
            self._empty_label.setFont(QFont(JarvisTheme.FONT_FAMILY, 9, QFont.Weight.Medium))
            self._empty_label.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; padding: 40px; border: none;")
            self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._items_layout.addWidget(self._empty_label)

            self._total_lbl.setText("0")
            self._success_rate_lbl.setText("100%")
            self._active_count_lbl.setText("0")
            return

        total = len(records)
        failed = sum(1 for r in records if str(r.get("status", "")).lower() in ("failed", "error"))
        running = sum(1 for r in records if str(r.get("status", "")).lower() in ("running", "executing", "in_progress"))
        success_pct = int(((total - failed) / total) * 100) if total > 0 else 100

        self._total_lbl.setText(str(total))
        self._success_rate_lbl.setText(f"{success_pct}%")
        self._active_count_lbl.setText(str(running))

        for rec in reversed(records):
            item_widget = ExecutionHistoryItem(rec, self._items_container)
            self._items_layout.addWidget(item_widget)

    def update_active_plan(self, plan_info: dict[str, Any]) -> None:
        """Update active plan status and sub-steps in the side card.

        Args:
            plan_info: Active plan info dictionary containing title, progress, steps.
        """
        title = str(plan_info.get("title", "")).strip()
        progress = float(plan_info.get("progress", 0.0))
        steps = plan_info.get("steps") or []

        if title:
            self._plan_title_lbl.setText(title)
        else:
            self._plan_title_lbl.setText("No active plan currently running.")

        pct = int(progress * 100)
        self._plan_pbar.setValue(pct)
        self._plan_pct_lbl.setText(f"{pct}% Complete")

        # Rebuild steps checklist
        while self._plan_steps_layout.count():
            item = self._plan_steps_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for s in steps:
            st_name = str(s.get("title") or s.get("action") or "Task")
            st_status = str(s.get("status", "pending")).lower()
            is_done = st_status in ("completed", "done", "success")
            is_active = st_status in ("running", "executing")
            is_failed = st_status in ("failed", "error")

            step_row = QHBoxLayout()
            step_row.setSpacing(6)

            dot = QLabel("✓" if is_done else ("✕" if is_failed else ("◉" if is_active else "○")))
            dot.setFixedSize(14, 14)
            dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
            if is_failed:
                dot.setStyleSheet("color: #ef4444; font-size: 8px; font-weight: bold; background: transparent; border: none;")
            elif is_done:
                dot.setStyleSheet("color: #22c55e; font-size: 8px; font-weight: bold; background: transparent; border: none;")
            elif is_active:
                dot.setStyleSheet(f"color: {JarvisTheme.CYAN_PRIMARY}; font-size: 8px; font-weight: bold; background: transparent; border: none;")
            else:
                dot.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; font-size: 8px; background: transparent; border: none;")
            step_row.addWidget(dot)

            lbl = QLabel(st_name)
            lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 8))
            lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_PRIMARY if (is_done or is_active) else JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
            step_row.addWidget(lbl)
            step_row.addStretch(1)

            step_widget = QWidget()
            step_widget.setLayout(step_row)
            self._plan_steps_layout.addWidget(step_widget)
