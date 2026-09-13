"""System Diagnostics View for J.A.R.V.I.S Multi-View Command Center.

Displays real-time, strictly read-only diagnostics including CPU, RAM, GPU,
storage mounts, battery status, network throughput, and top processes using
safe SystemMonitor infrastructure.
"""

from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.ui.components.glass_panel import GlassPanel
from app.ui.styles import JarvisTheme, ensure_fonts_loaded


class MetricStatBox(QFrame):
    """Compact metric indicator tile with title, primary value, and subtitle."""

    def __init__(
        self,
        title: str,
        value: str = "--",
        subtitle: str = "",
        val_color: str = JarvisTheme.CYAN_PRIMARY,
        parent: Optional[QWidget] = None,
    ) -> None:
        """Initialize MetricStatBox."""
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: #06152b;
                border: 1px solid #0d274c;
                border-radius: 8px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)

        self._title_lbl = QLabel(title.upper())
        self._title_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 7, QFont.Weight.Bold))
        self._title_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
        layout.addWidget(self._title_lbl)

        self._val_lbl = QLabel(value)
        self._val_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 14, QFont.Weight.Bold))
        self._val_lbl.setStyleSheet(f"color: {val_color}; background: transparent; border: none;")
        layout.addWidget(self._val_lbl)

        self._sub_lbl = QLabel(subtitle)
        self._sub_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 7))
        self._sub_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
        layout.addWidget(self._sub_lbl)

    def set_value(self, value: str, subtitle: Optional[str] = None) -> None:
        """Update displayed value and optional subtitle."""
        self._val_lbl.setText(value)
        if subtitle is not None:
            self._sub_lbl.setText(subtitle)


class SystemView(QWidget):
    """Full-screen Read-Only System Diagnostics & Hardware Telemetry dashboard."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize SystemView."""
        super().__init__(parent)
        ensure_fonts_loaded()

        self.setStyleSheet("background: transparent;")
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 16, 20, 16)
        main_layout.setSpacing(14)

        # 1. Header Card
        header_card = GlassPanel(self, border_radius=12)
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(16, 12, 16, 12)
        header_layout.setSpacing(16)

        title_col = QVBoxLayout()
        title_col.setSpacing(4)
        title_lbl = QLabel("SYSTEM DIAGNOSTICS & TELEMETRY")
        title_font = QFont(JarvisTheme.FONT_FAMILY, 14, QFont.Weight.Bold)
        title_font.setFamilies(JarvisTheme.FONT_FAMILIES)
        title_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.2)
        title_lbl.setFont(title_font)
        title_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_PRIMARY}; background: transparent; border: none;")
        title_col.addWidget(title_lbl)

        self._host_info_lbl = QLabel("Host Architecture: Loading...")
        self._host_info_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 8))
        self._host_info_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
        title_col.addWidget(self._host_info_lbl)
        header_layout.addLayout(title_col, 1)

        # Read-only security seal
        security_badge = QLabel("🛡 READ-ONLY MODE ENFORCED")
        badge_font = QFont(JarvisTheme.FONT_FAMILY, 8, QFont.Weight.Bold)
        badge_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.0)
        security_badge.setFont(badge_font)
        security_badge.setStyleSheet("""
            QLabel {
                color: #22c55e;
                background-color: #042116;
                border: 1px solid #14532d;
                border-radius: 8px;
                padding: 6px 14px;
            }
        """)
        header_layout.addWidget(security_badge)

        header_card.content_layout.addLayout(header_layout)
        main_layout.addWidget(header_card, 0)

        # 2. Main Hardware Grid: CPU, RAM, GPU, Storage, Network, Battery
        grid_card = GlassPanel(self, border_radius=12)
        grid_card.add_card_header("HARDWARE SUBSYSTEMS", icon_text="⚙")

        grid_layout = QGridLayout()
        grid_layout.setSpacing(12)

        # CPU Box
        self.cpu_box = MetricStatBox("CPU Utilization", "0%", "Cores: N/A | Freq: N/A")
        grid_layout.addWidget(self.cpu_box, 0, 0)

        # RAM Box
        self.ram_box = MetricStatBox("Memory (RAM)", "0.0 GB", "Free: 0.0 GB (0%)", val_color="#38bdf8")
        grid_layout.addWidget(self.ram_box, 0, 1)

        # GPU Box
        self.gpu_box = MetricStatBox("Graphics (GPU)", "N/A", "Hardware Acceleration", val_color="#a855f7")
        grid_layout.addWidget(self.gpu_box, 0, 2)

        # Storage Box
        self.disk_box = MetricStatBox("Primary Disk", "0.0 GB", "Mount: C:\\ | Free: 0.0 GB", val_color="#fb923c")
        grid_layout.addWidget(self.disk_box, 1, 0)

        # Network Box
        self.net_box = MetricStatBox("Network Traffic", "0.0 KB/s", "Sent: 0 MB | Recv: 0 MB", val_color="#34d399")
        grid_layout.addWidget(self.net_box, 1, 1)

        # Battery Box
        self.battery_box = MetricStatBox("Power / Battery", "AC Power", "Plugged in (100%)", val_color="#eab308")
        grid_layout.addWidget(self.battery_box, 1, 2)

        grid_card.content_layout.addLayout(grid_layout)
        main_layout.addWidget(grid_card, 0)

        # 3. Bottom Table: Top Active Processes (Read-Only)
        proc_card = GlassPanel(self, border_radius=12)
        proc_card.add_card_header("TOP PROCESSES (READ-ONLY TELEMETRY)", icon_text="📊")

        self.proc_table = QTableWidget(proc_card)
        self.proc_table.setColumnCount(5)
        self.proc_table.setHorizontalHeaderLabels(["PID", "PROCESS NAME", "CPU %", "RAM %", "STATUS"])
        self.proc_table.horizontalHeader().setStretchLastSection(True)
        self.proc_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.proc_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.proc_table.verticalHeader().setVisible(False)
        self.proc_table.setShowGrid(False)
        self.proc_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: transparent;
                border: none;
                color: {JarvisTheme.TEXT_PRIMARY};
                font-family: {JarvisTheme.FONT_FAMILY};
                font-size: 11px;
            }}
            QHeaderView::section {{
                background-color: #06152b;
                color: {JarvisTheme.TEXT_MUTED};
                border: none;
                border-bottom: 1px solid #0d274c;
                padding: 6px 10px;
                font-weight: bold;
                font-size: 10px;
            }}
            QTableWidget::item {{
                padding: 5px 10px;
                border-bottom: 1px solid #081a33;
            }}
        """)
        proc_card.content_layout.addWidget(self.proc_table)
        main_layout.addWidget(proc_card, 1)

    def update_diagnostics(self, data: dict[str, Any]) -> None:
        """Update system metrics and process table from diagnostics data.

        Args:
            data: Structured telemetry and system metrics dictionary.
        """
        # Host platform
        platform_str = str(data.get("os_platform") or data.get("platform") or "Local Host")
        dev_name = str(data.get("device_name") or "Localhost")
        uptime = str(data.get("uptime") or "N/A")
        self._host_info_lbl.setText(f"{platform_str} | Host: {dev_name} | Uptime: {uptime}")

        # CPU
        cpu_pct = float(data.get("cpu_percent", 0.0))
        cpu_cores = data.get("cpu_cores", "N/A")
        cpu_freq = data.get("cpu_freq_mhz")
        freq_str = f"{cpu_freq:.0f} MHz" if isinstance(cpu_freq, (int, float)) else "N/A"
        self.cpu_box.set_value(f"{cpu_pct:.1f}%", f"Cores: {cpu_cores} | Freq: {freq_str}")

        # RAM
        ram_pct = float(data.get("memory_percent", 0.0))
        used_ram = data.get("ram_used_gb", 0.0)
        total_ram = data.get("ram_total_gb", 0.0)
        free_ram = data.get("ram_free_gb", 0.0)
        self.ram_box.set_value(f"{used_ram:.1f} / {total_ram:.1f} GB", f"Free: {free_ram:.1f} GB ({ram_pct:.0f}%)")

        # GPU
        gpu_pct = data.get("gpu_percent")
        gpu_avail = data.get("gpu_available", False)
        if gpu_avail and gpu_pct is not None:
            self.gpu_box.set_value(f"{float(gpu_pct):.0f}%", "Hardware Acceleration Active")
        else:
            self.gpu_box.set_value("N/A", "Hardware Acceleration Unavailable")

        # Disk
        disk_pct = float(data.get("disk_percent", 0.0))
        disk_used = data.get("disk_used_gb", 0.0)
        disk_total = data.get("disk_total_gb", 0.0)
        disk_mount = str(data.get("disk_mount", "C:\\"))
        self.disk_box.set_value(f"{disk_used:.1f} / {disk_total:.1f} GB", f"Mount: {disk_mount} ({disk_pct:.0f}%)")

        # Network
        net_rate = data.get("network_rate_kbs", 0.0)
        net_sent = data.get("network_sent_mb", 0.0)
        net_recv = data.get("network_recv_mb", 0.0)
        self.net_box.set_value(f"{float(net_rate):.1f} KB/s", f"Sent: {net_sent:.1f} MB | Recv: {net_recv:.1f} MB")

        # Battery
        bat_pct = data.get("battery_percent")
        bat_plugged = data.get("battery_plugged", True)
        if bat_pct is not None:
            status = "Plugged in" if bat_plugged else "Discharging"
            self.battery_box.set_value(f"{float(bat_pct):.0f}%", f"Power Source: {status}")
        else:
            self.battery_box.set_value("AC Power", "Desktop Workstation (Plugged)")

        # Top processes table
        procs = data.get("top_processes", [])
        self.proc_table.setRowCount(len(procs))
        for row, p in enumerate(procs):
            pid = str(p.get("pid", ""))
            name = str(p.get("name", ""))
            c_pct = f"{float(p.get('cpu_percent', 0.0)):.1f}%"
            m_pct = f"{float(p.get('memory_percent', 0.0)):.1f}%"
            st = str(p.get("status", "running")).upper()

            for col, val in enumerate([pid, name, c_pct, m_pct, st]):
                item = QTableWidgetItem(val)
                if col in (0, 2, 3):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.proc_table.setItem(row, col, item)
