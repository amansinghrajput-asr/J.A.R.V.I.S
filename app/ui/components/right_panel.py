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
        is_failed: bool = False,
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
        if is_failed:
            icon_lbl.setText("✕")
            icon_lbl.setStyleSheet("""
                QLabel {
                    background-color: #2b0b0b;
                    color: #ef4444;
                    border: 1px solid #7f1d1d;
                    border-radius: 8px;
                    font-size: 8px;
                    font-weight: bold;
                }
            """)
        elif is_done:
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
        if is_failed:
            text_color = "#ef4444"
        elif is_done or is_active:
            text_color = JarvisTheme.TEXT_PRIMARY
        else:
            text_color = JarvisTheme.TEXT_MUTED
        title_lbl.setStyleSheet(f"color: {text_color}; background: transparent; border: none;")
        layout.addWidget(title_lbl)

        layout.addStretch(1)

        if tag:
            tag_lbl = QLabel(tag)
            tag_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 7))
            tag_border = "#7f1d1d" if is_failed else "#0f3d6c"
            tag_bg = "#2b0b0b" if is_failed else "#061e38"
            tag_color = "#ef4444" if is_failed else JarvisTheme.CYAN_PRIMARY
            tag_lbl.setStyleSheet(f"""
                QLabel {{
                    color: {tag_color};
                    background-color: {tag_bg};
                    border: 1px solid {tag_border};
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

        self._badge_lbl = QLabel("0 / 0")
        self._badge_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 8, QFont.Weight.Bold))
        self._badge_lbl.setStyleSheet(f"color: {JarvisTheme.CYAN_PRIMARY}; background: transparent; border: none;")
        self.add_card_header("CURRENT TASK", icon_text="🎯", right_badge=self._badge_lbl)

        # Task Title
        self._title_lbl = QLabel("Standby — Ready for commands")
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
        self._pbar.setValue(0)
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

        self._pct_lbl = QLabel("0%")
        self._pct_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 8, QFont.Weight.Bold))
        self._pct_lbl.setStyleSheet(f"color: {JarvisTheme.CYAN_PRIMARY}; background: transparent; border: none;")
        prog_row.addWidget(self._pct_lbl)
        self.content_layout.addLayout(prog_row)

        # Dynamic Checklist container
        self._tasks_container = QWidget(self)
        self._tasks_container.setStyleSheet("background: transparent; border: none;")
        self._tasks_layout = QVBoxLayout(self._tasks_container)
        self._tasks_layout.setContentsMargins(0, 0, 0, 0)
        self._tasks_layout.setSpacing(4)
        self.content_layout.addWidget(self._tasks_container)

        # Initial standby state
        self.update_task_state("Standby — Ready for commands", 0.0)

    def set_task(self, title: str, progress: float) -> None:
        """Update current task display."""
        self.update_task_state(title, progress)

    def update_task_state(
        self,
        title: str = "",
        progress: float = 0.0,
        steps: Optional[list[dict]] = None,
    ) -> None:
        """Update current task title, progress bar, and dynamic checklist steps.

        Supports:
        - Standby / no active task
        - Single-command execution
        - Multi-step planner execution
        """
        # Clear existing items safely
        while self._tasks_layout.count():
            item = self._tasks_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

        # 1. Update Title & Progress
        display_title = title.strip() if title and title.strip() else "Standby — Ready for commands"
        self._title_lbl.setText(display_title)

        val = int(max(0.0, min(1.0, progress)) * 100)
        self._pbar.setValue(val)
        self._pct_lbl.setText(f"{val}%")

        # 2. Render Steps
        if steps:
            total = len(steps)
            completed = sum(1 for s in steps if s.get("status") == "completed" or s.get("is_done"))
            self._badge_lbl.setText(f"{completed} / {total}")

            for s in steps:
                st_title = str(s.get("title") or s.get("name") or s.get("action") or "Task")
                status = str(s.get("status", "")).lower()
                is_done = bool(s.get("is_done") or status in ("completed", "done", "success"))
                is_active = bool(s.get("is_active") or status in ("running", "active", "executing"))
                is_failed = bool(s.get("is_failed") or status in ("failed", "error"))
                tag = str(s.get("tag") or s.get("strategy") or "")

                self._tasks_layout.addWidget(
                    TaskCheckItem(
                        st_title,
                        is_done=is_done,
                        is_active=is_active,
                        is_failed=is_failed,
                        tag=tag,
                        parent=self._tasks_container,
                    )
                )
        elif title and title.strip() and title.strip() != "Standby — Ready for commands":
            # Single command in progress
            self._badge_lbl.setText("1 / 1")
            is_done = (progress >= 1.0)
            self._tasks_layout.addWidget(
                TaskCheckItem(
                    display_title,
                    is_done=is_done,
                    is_active=not is_done,
                    tag="Direct",
                    parent=self._tasks_container,
                )
            )
        else:
            # Standby state
            self._badge_lbl.setText("0 / 0")
            self._tasks_layout.addWidget(
                TaskCheckItem(
                    "Awaiting next instruction...",
                    is_done=False,
                    is_active=False,
                    parent=self._tasks_container,
                )
            )


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
        self.gpu_dial = CircularGauge("GPU", 0.0, self, accent_color="#818cf8")
        self.gpu_dial.set_display_text("N/A")

        dials_row.addWidget(self.cpu_dial)
        dials_row.addWidget(self.ram_dial)
        dials_row.addWidget(self.gpu_dial)
        self.content_layout.addLayout(dials_row)

        # Status rows with dynamic label references
        # 1. Network row
        net_row = QHBoxLayout()
        net_row.setSpacing(8)
        net_ic = QLabel("📡")
        net_ic.setStyleSheet(f"color: {JarvisTheme.CYAN_PRIMARY}; font-size: 11px; background: transparent; border: none;")
        net_row.addWidget(net_ic)

        net_title = QLabel("Network")
        net_title.setFont(QFont(JarvisTheme.FONT_FAMILY, 8))
        net_title.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
        net_row.addWidget(net_title)
        net_row.addStretch(1)

        self._net_val_lbl = QLabel("↑ 0.0 KB/s  ↓ 0.0 KB/s")
        self._net_val_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 8, QFont.Weight.Medium))
        self._net_val_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_PRIMARY}; background: transparent; border: none;")
        net_row.addWidget(self._net_val_lbl)
        self.content_layout.addLayout(net_row)

        # 2. Active App row
        app_row = QHBoxLayout()
        app_row.setSpacing(8)
        app_ic = QLabel("🖥")
        app_ic.setStyleSheet(f"color: {JarvisTheme.CYAN_PRIMARY}; font-size: 11px; background: transparent; border: none;")
        app_row.addWidget(app_ic)

        app_title = QLabel("Active Host")
        app_title.setFont(QFont(JarvisTheme.FONT_FAMILY, 8))
        app_title.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
        app_row.addWidget(app_title)
        app_row.addStretch(1)

        self._active_app_val_lbl = QLabel("Localhost")
        self._active_app_val_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 8, QFont.Weight.Medium))
        self._active_app_val_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_PRIMARY}; background: transparent; border: none;")
        app_row.addWidget(self._active_app_val_lbl)
        self.content_layout.addLayout(app_row)

        # 3. System Health row
        sys_row = QHBoxLayout()
        sys_row.setSpacing(8)
        sys_ic = QLabel("🛡")
        sys_ic.setStyleSheet(f"color: {JarvisTheme.CYAN_PRIMARY}; font-size: 11px; background: transparent; border: none;")
        sys_row.addWidget(sys_ic)

        sys_title = QLabel("System")
        sys_title.setFont(QFont(JarvisTheme.FONT_FAMILY, 8))
        sys_title.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
        sys_row.addWidget(sys_title)
        sys_row.addStretch(1)

        self._system_status_val_lbl = QLabel("● All systems normal")
        self._system_status_val_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 8, QFont.Weight.Medium))
        self._system_status_val_lbl.setStyleSheet("color: #22c55e; background: transparent; border: none;")
        sys_row.addWidget(self._system_status_val_lbl)
        self.content_layout.addLayout(sys_row)

    def update_telemetry(self, metrics: dict) -> None:
        """Update circular dials and network/system telemetry rows from live metrics."""
        if not isinstance(metrics, dict):
            return

        # 1. CPU
        if "cpu_percent" in metrics:
            try:
                self.cpu_dial.set_value(float(metrics["cpu_percent"]))
            except Exception:
                pass

        # 2. RAM
        if "memory_percent" in metrics:
            try:
                self.ram_dial.set_value(float(metrics["memory_percent"]))
            except Exception:
                pass

        # 3. GPU
        gpu_avail = metrics.get("gpu_available", False)
        if gpu_avail and "gpu_percent" in metrics and metrics["gpu_percent"] is not None:
            try:
                self.gpu_dial.set_value(float(metrics["gpu_percent"]))
            except Exception:
                self.gpu_dial.set_display_text("N/A")
        else:
            self.gpu_dial.set_display_text("N/A")

        # 4. Network throughput
        if "network_summary" in metrics:
            self._net_val_lbl.setText(str(metrics["network_summary"]))

        # 5. Active Host / App
        if "active_host" in metrics:
            self._active_app_val_lbl.setText(str(metrics["active_host"]))
        elif "active_app" in metrics:
            self._active_app_val_lbl.setText(str(metrics["active_app"]))

        # 6. Status summary
        if "system_status" in metrics:
            self._system_status_val_lbl.setText(str(metrics["system_status"]))


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
        self._os_title_lbl = QLabel("Windows")
        self._os_title_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 8, QFont.Weight.Bold))
        self._os_title_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_PRIMARY}; background: transparent; border: none;")
        os_col.addWidget(self._os_title_lbl)

        self._os_sub_lbl = QLabel("PC Host")
        self._os_sub_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 7))
        self._os_sub_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
        os_col.addWidget(self._os_sub_lbl)
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

        self._uptime_val_lbl = QLabel("0d 0h 0m")
        self._uptime_val_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 7, QFont.Weight.Medium))
        self._uptime_val_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_PRIMARY}; background: transparent; border: none;")
        uptime_row.addWidget(self._uptime_val_lbl)

        self.content_layout.addLayout(uptime_row)

    def update_info(self, info: dict) -> None:
        """Update OS details, machine hostname, and live uptime string."""
        if not isinstance(info, dict):
            return

        if "os_name" in info:
            self._os_title_lbl.setText(str(info["os_name"]))
        elif "platform" in info:
            self._os_title_lbl.setText(str(info["platform"]).split()[0])

        if "device_name" in info:
            self._os_sub_lbl.setText(str(info["device_name"]))
        elif "hostname" in info:
            self._os_sub_lbl.setText(str(info["hostname"]))

        if "uptime" in info:
            self._uptime_val_lbl.setText(str(info["uptime"]))
        elif "uptime_seconds" in info:
            try:
                secs = int(info["uptime_seconds"])
                days = secs // 86400
                hours = (secs % 86400) // 3600
                mins = (secs % 3600) // 60
                self._uptime_val_lbl.setText(f"{days}d {hours}h {mins}m")
            except Exception:
                pass


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
