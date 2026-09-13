"""Right Panel Component for J.A.R.V.I.S HUD.

Houses Current Task progress card, System Status dials (CPU/RAM/GPU),
Voice & AI Settings card, and System Info widget matching jarvis_ui_reference.png.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.ui.components.dials import CircularGauge
from app.ui.components.glass_panel import GlassPanel
from app.ui.styles import JarvisTheme, ensure_fonts_loaded


class TaskCheckItem(QWidget):
    """Checklist item with status circle, label, and optional skill tag."""

    def __init__(
        self,
        title: str,
        is_done: bool = False,
        is_active: bool = False,
        tag: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        """Initialize TaskCheckItem."""
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(8)

        # Status Icon
        icon_lbl = QLabel()
        icon_lbl.setFixedSize(16, 16)
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if is_done:
            icon_lbl.setText("✓")
            icon_lbl.setStyleSheet("""
                QLabel {
                    background-color: #042416;
                    color: #22c55e;
                    border: 1px solid #14532d;
                    border-radius: 8px;
                    font-size: 8px;
                    font-weight: bold;
                }
            """)
        elif is_active:
            icon_lbl.setText("◉")
            icon_lbl.setStyleSheet(f"""
                QLabel {{
                    background-color: #0c2b4d;
                    color: {JarvisTheme.CYAN_PRIMARY};
                    border: 1.5px solid {JarvisTheme.CYAN_PRIMARY};
                    border-radius: 8px;
                    font-size: 9px;
                }}
            """)
        else:
            icon_lbl.setText("○")
            icon_lbl.setStyleSheet(f"""
                QLabel {{
                    background-color: #06152b;
                    color: {JarvisTheme.TEXT_MUTED};
                    border: 1px solid #0a213e;
                    border-radius: 8px;
                    font-size: 8px;
                }}
            """)
        layout.addWidget(icon_lbl)

        title_lbl = QLabel(title)
        title_font = QFont(JarvisTheme.FONT_FAMILY, 8, QFont.Weight.Medium if is_active else QFont.Weight.Normal)
        title_font.setFamilies(JarvisTheme.FONT_FAMILIES)
        title_lbl.setFont(title_font)
        title_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_PRIMARY if (is_done or is_active) else JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
        layout.addWidget(title_lbl)

        layout.addStretch(1)

        if tag:
            tag_lbl = QLabel(tag)
            tag_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 7))
            tag_lbl.setStyleSheet(f"""
                QLabel {{
                    color: {JarvisTheme.CYAN_PRIMARY};
                    background-color: #061e38;
                    border: 1px solid #0f3d6c;
                    border-radius: 4px;
                    padding: 1px 6px;
                }}
            """)
            layout.addWidget(tag_lbl)


class CurrentTaskCard(GlassPanel):
    """Current Task overview panel with progress bar and checklist."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize CurrentTaskCard."""
        super().__init__(parent)

        badge_lbl = QLabel("2 / 4")
        badge_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 8, QFont.Weight.Bold))
        badge_lbl.setStyleSheet(f"color: {JarvisTheme.CYAN_PRIMARY}; background: transparent; border: none;")
        self.add_card_header("CURRENT TASK", icon_text="🎯", right_badge=badge_lbl)

        # Task Title
        self._title_lbl = QLabel("Open Chrome and search for today's weather")
        title_font = QFont(JarvisTheme.FONT_FAMILY, 9, QFont.Weight.Bold)
        title_font.setFamilies(JarvisTheme.FONT_FAMILIES)
        self._title_lbl.setFont(title_font)
        self._title_lbl.setWordWrap(True)
        self._title_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_PRIMARY}; background: transparent; border: none;")
        self.content_layout.addWidget(self._title_lbl)

        # Glowing Progress Bar Row
        prog_row = QHBoxLayout()
        prog_row.setSpacing(8)

        self._pbar = QProgressBar()
        self._pbar.setFixedHeight(5)
        self._pbar.setRange(0, 100)
        self._pbar.setValue(75)
        self._pbar.setTextVisible(False)
        self._pbar.setStyleSheet(f"""
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
        prog_row.addWidget(self._pbar, 1)

        self._pct_lbl = QLabel("75%")
        self._pct_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 8, QFont.Weight.Bold))
        self._pct_lbl.setStyleSheet(f"color: {JarvisTheme.CYAN_PRIMARY}; background: transparent; border: none;")
        prog_row.addWidget(self._pct_lbl)
        self.content_layout.addLayout(prog_row)

        # Checklist items
        self.content_layout.addWidget(TaskCheckItem("Understand the request", is_done=True, parent=self))
        self.content_layout.addWidget(TaskCheckItem("Open Chrome", is_done=True, parent=self))
        self.content_layout.addWidget(TaskCheckItem("Search for weather", is_active=True, parent=self))
        self.content_layout.addWidget(TaskCheckItem("Show results", is_done=False, tag="Web Skill", parent=self))

    def set_task(self, title: str, progress: float) -> None:
        """Update current task display."""
        self._title_lbl.setText(title)
        val = int(max(0.0, min(1.0, progress)) * 100)
        self._pbar.setValue(val)
        self._pct_lbl.setText(f"{val}%")


class SystemStatusCard(GlassPanel):
    """System Status panel displaying CPU, RAM, and GPU dials and telemetry."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize SystemStatusCard."""
        super().__init__(parent)

        self.add_card_header("SYSTEM STATUS", icon_text="⚙")

        # 3 Circular Dials
        dials_row = QHBoxLayout()
        dials_row.setContentsMargins(0, 2, 0, 4)
        dials_row.setSpacing(6)

        self.cpu_dial = CircularGauge("CPU", 32.0, self, accent_color=JarvisTheme.CYAN_PRIMARY)
        self.ram_dial = CircularGauge("RAM", 48.0, self, accent_color="#00b4d8")
        self.gpu_dial = CircularGauge("GPU", 12.0, self, accent_color="#818cf8")

        dials_row.addWidget(self.cpu_dial)
        dials_row.addWidget(self.ram_dial)
        dials_row.addWidget(self.gpu_dial)
        self.content_layout.addLayout(dials_row)

        # Status rows
        metrics = [
            ("📡", "Network", "↑ 2.4 Mbps  ↓ 1.8 Mbps"),
            ("🖥", "Active App", "Google Chrome"),
            ("🛡", "System", "● All systems normal"),
        ]

        for icon, label, val in metrics:
            row = QHBoxLayout()
            row.setSpacing(8)

            ic = QLabel(icon)
            ic.setStyleSheet(f"color: {JarvisTheme.CYAN_PRIMARY}; font-size: 11px; background: transparent; border: none;")
            row.addWidget(ic)

            lbl = QLabel(label)
            lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 8))
            lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
            row.addWidget(lbl)

            row.addStretch(1)

            val_lbl = QLabel(val)
            val_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 8, QFont.Weight.Medium))
            color_css = "color: #22c55e;" if "All systems normal" in val else f"color: {JarvisTheme.TEXT_PRIMARY};"
            val_lbl.setStyleSheet(f"{color_css} background: transparent; border: none;")
            row.addWidget(val_lbl)

            self.content_layout.addLayout(row)


class Sparkline(QWidget):
    """Mini glowing cyan waveform / curve line widget."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize Sparkline."""
        super().__init__(parent)
        self.setFixedHeight(18)

    def paintEvent(self, event: object) -> None:
        """Draw smooth sine-wave line."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        w = float(self.width())
        h = float(self.height())

        path = QPainterPath()
        path.moveTo(0, h / 2.0)
        # 3 oscillations
        for i in range(1, 40):
            x = (w / 39.0) * i
            import math
            y = (h / 2.0) + math.sin(i * 0.4) * (h * 0.35)
            path.lineTo(x, y)

        pen = QPen(QColor(JarvisTheme.CYAN_PRIMARY))
        pen.setWidthF(1.5)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)


class VoiceSettingsCard(GlassPanel):
    """Voice & AI Settings summary card."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize VoiceSettingsCard."""
        super().__init__(parent)
        self.content_layout.setContentsMargins(10, 10, 10, 10)
        self.content_layout.setSpacing(6)

        hdr = self.add_card_header("VOICE & AI SETTINGS", icon_text="🎙")
        # Ensure title font fits in compact card
        for i in range(hdr.count()):
            item = hdr.itemAt(i)
            if item and item.widget() and isinstance(item.widget(), QLabel) and "VOICE" in item.widget().text():
                f = QFont(JarvisTheme.FONT_FAMILY, 7, QFont.Weight.Bold)
                f.setFamilies(JarvisTheme.FONT_FAMILIES)
                f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.8)
                item.widget().setFont(f)

        settings = [
            ("🛡", "Voice Provider", "Google (Gemini)"),
            ("🗣", "Voice Profile", "Default"),
            ("🌐", "Language", "English (US)"),
            ("🎤", "Wake Word", "JARVIS"),
        ]

        for icon, label, val in settings:
            row = QHBoxLayout()
            row.setSpacing(4)

            ic = QLabel(icon)
            ic.setStyleSheet(f"color: {JarvisTheme.CYAN_PRIMARY}; font-size: 9px; background: transparent; border: none;")
            row.addWidget(ic)

            lbl = QLabel(label)
            lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 7))
            lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
            row.addWidget(lbl)

            row.addStretch(1)

            val_lbl = QLabel(val)
            val_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 7, QFont.Weight.Medium))
            val_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_PRIMARY}; background: transparent; border: none;")
            row.addWidget(val_lbl)

            chev = QLabel("›")
            chev.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; font-weight: bold; font-size: 9px; background: transparent; border: none;")
            row.addWidget(chev)

            self.content_layout.addLayout(row)

        # Edit Settings Button
        edit_btn = QPushButton("Edit Settings ›")
        edit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        edit_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {JarvisTheme.CYAN_PRIMARY};
                border: none;
                font-size: 8px;
                text-align: right;
                padding-top: 2px;
            }}
            QPushButton:hover {{
                color: {JarvisTheme.CYAN_BRIGHT};
            }}
        """)
        self.content_layout.addWidget(edit_btn)


class SystemInfoCard(GlassPanel):
    """System Info badge with OS info and sparkline."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize SystemInfoCard."""
        super().__init__(parent)
        self.content_layout.setContentsMargins(10, 10, 10, 10)
        self.content_layout.setSpacing(6)

        hdr = self.add_card_header("SYSTEM INFO", icon_text="📊")
        for i in range(hdr.count()):
            item = hdr.itemAt(i)
            if item and item.widget() and isinstance(item.widget(), QLabel) and "SYSTEM" in item.widget().text():
                f = QFont(JarvisTheme.FONT_FAMILY, 7, QFont.Weight.Bold)
                f.setFamilies(JarvisTheme.FONT_FAMILIES)
                f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.8)
                item.widget().setFont(f)

        info_row = QHBoxLayout()
        info_row.setSpacing(8)

        win_icon = QLabel("🪟")
        win_icon.setStyleSheet(f"color: {JarvisTheme.CYAN_PRIMARY}; font-size: 14px; background: transparent; border: none;")
        info_row.addWidget(win_icon)

        os_col = QVBoxLayout()
        os_col.setSpacing(1)
        os_title = QLabel("Windows 11")
        os_title.setFont(QFont(JarvisTheme.FONT_FAMILY, 8, QFont.Weight.Bold))
        os_title.setStyleSheet(f"color: {JarvisTheme.TEXT_PRIMARY}; background: transparent; border: none;")
        os_col.addWidget(os_title)

        os_sub = QLabel("HP Victus")
        os_sub.setFont(QFont(JarvisTheme.FONT_FAMILY, 7))
        os_sub.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
        os_col.addWidget(os_sub)
        info_row.addLayout(os_col)
        info_row.addStretch(1)

        self.content_layout.addLayout(info_row)

        # Sparkline
        self.sparkline = Sparkline(self)
        self.content_layout.addWidget(self.sparkline)

        # Uptime
        uptime_row = QHBoxLayout()
        lbl = QLabel("Uptime")
        lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 7))
        lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
        uptime_row.addWidget(lbl)

        uptime_row.addStretch(1)

        val = QLabel("2d 4h 32m")
        val.setFont(QFont(JarvisTheme.FONT_FAMILY, 7, QFont.Weight.Medium))
        val.setStyleSheet(f"color: {JarvisTheme.TEXT_PRIMARY}; background: transparent; border: none;")
        uptime_row.addWidget(val)

        self.content_layout.addLayout(uptime_row)


class RightPanel(QWidget):
    """Container for Right HUD column (Task + System Status + Settings)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize RightPanel."""
        super().__init__(parent)
        ensure_fonts_loaded()

        self.setFixedWidth(340)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.current_task_card = CurrentTaskCard(self)
        layout.addWidget(self.current_task_card, 0)

        self.system_status_card = SystemStatusCard(self)
        layout.addWidget(self.system_status_card, 0)

        # Horizontal split for bottom settings & system info
        bottom_h = QHBoxLayout()
        bottom_h.setContentsMargins(0, 0, 0, 0)
        bottom_h.setSpacing(8)

        self.voice_settings_card = VoiceSettingsCard(self)
        bottom_h.addWidget(self.voice_settings_card, 3)

        self.system_info_card = SystemInfoCard(self)
        bottom_h.addWidget(self.system_info_card, 2)

        layout.addLayout(bottom_h, 1)
