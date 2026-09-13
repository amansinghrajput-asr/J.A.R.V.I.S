"""Bottom Command Bar Component for J.A.R.V.I.S HUD.

Implements the status display, text command input line, send button,
center glowing slider track, and audio controls matching jarvis_ui_reference.png.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

from app.ui.styles import JarvisTheme, ensure_fonts_loaded


class CenterTrackSlider(QWidget):
    """Futuristic glowing blue horizontal track with center luminous node."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize CenterTrackSlider."""
        super().__init__(parent)
        self.setFixedHeight(32)

    def paintEvent(self, event: object) -> None:
        """Draw futuristic segmented track with glowing central node."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        w = float(self.width())
        h = float(self.height())
        cy = h / 2.0
        cx = w / 2.0

        # Segmented track line
        pen = QPen(QColor("#0d2e53"))
        pen.setWidthF(1.5)
        painter.setPen(pen)

        # Left wing
        painter.drawLine(QPointF(20.0, cy), QPointF(cx - 30.0, cy))
        # Right wing
        painter.drawLine(QPointF(cx + 30.0, cy), QPointF(w - 20.0, cy))

        # Angled decorative notches near center
        pen_notch = QPen(QColor(JarvisTheme.CYAN_DIM))
        pen_notch.setWidthF(1.2)
        painter.setPen(pen_notch)
        painter.drawLine(QPointF(cx - 30.0, cy), QPointF(cx - 20.0, cy - 6.0))
        painter.drawLine(QPointF(cx - 20.0, cy - 6.0), QPointF(cx - 10.0, cy))
        painter.drawLine(QPointF(cx + 30.0, cy), QPointF(cx + 20.0, cy - 6.0))
        painter.drawLine(QPointF(cx + 20.0, cy - 6.0), QPointF(cx + 10.0, cy))

        # Central glowing orb
        glow_brush = QColor(JarvisTheme.CYAN_PRIMARY)
        painter.setPen(QPen(QColor(JarvisTheme.WHITE_GLOW), 1.5))
        painter.setBrush(glow_brush)
        painter.drawEllipse(QPointF(cx, cy), 5.0, 5.0)


class LevelMeter(QWidget):
    """Mini horizontal audio level meter bars."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize LevelMeter."""
        super().__init__(parent)
        self.setFixedSize(36, 14)

    def paintEvent(self, event: object) -> None:
        """Draw 5 vertical volume level bars."""
        painter = QPainter(self)
        painter.setPen(Qt.PenStyle.NoPen)

        for i in range(5):
            x = i * 7
            h = 4 + i * 2
            y = 14 - h
            col = QColor(JarvisTheme.CYAN_PRIMARY if i < 3 else JarvisTheme.CYAN_DIM)
            painter.setBrush(col)
            painter.drawRect(x, y, 4, h)


class BottomBar(QFrame):
    """Bottom interaction area with status, command input, and controls."""

    command_submitted = Signal(str)
    voice_toggled = Signal()
    cinematic_toggled = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize BottomBar."""
        super().__init__(parent)
        ensure_fonts_loaded()

        self.setFixedHeight(54)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {JarvisTheme.BG_MAIN};
                border: none;
                border-top: 1px solid #091a32;
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 8, 18, 8)
        layout.setSpacing(14)

        # 1. Left: J.A.R.V.I.S status badge & state message
        left_h = QHBoxLayout()
        left_h.setSpacing(8)

        status_badge = QLabel("⚡")
        status_badge.setFixedSize(26, 26)
        status_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_badge.setStyleSheet(f"""
            QLabel {{
                background-color: #0c2b4d;
                color: {JarvisTheme.CYAN_PRIMARY};
                border: 1px solid {JarvisTheme.CYAN_PRIMARY};
                border-radius: 6px;
                font-size: 11px;
            }}
        """)
        left_h.addWidget(status_badge)

        name_lbl = QLabel("J.A.R.V.I.S")
        name_font = QFont(JarvisTheme.FONT_FAMILY, 9, QFont.Weight.Bold)
        name_font.setFamilies(JarvisTheme.FONT_FAMILIES)
        name_lbl.setFont(name_font)
        name_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_PRIMARY}; background: transparent;")
        left_h.addWidget(name_lbl)

        self._status_lbl = QLabel("Ready to assist.")
        self._status_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 8))
        self._status_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; padding-left: 6px;")
        left_h.addWidget(self._status_lbl)

        layout.addLayout(left_h)

        # 2. Center: Command input line edit with Send button
        center_h = QHBoxLayout()
        center_h.setSpacing(6)

        self._input_edit = QLineEdit()
        self._input_edit.setPlaceholderText("Enter command or query (e.g. 'open chrome')...")
        self._input_edit.setFont(QFont(JarvisTheme.FONT_FAMILY, 9))
        self._input_edit.setFixedHeight(32)
        self._input_edit.setMinimumWidth(260)
        self._input_edit.setStyleSheet(f"""
            QLineEdit {{
                background-color: #06152b;
                color: {JarvisTheme.TEXT_PRIMARY};
                border: 1px solid #0d2e53;
                border-radius: 8px;
                padding-left: 10px;
                padding-right: 10px;
            }}
            QLineEdit:focus {{
                border: 1px solid {JarvisTheme.CYAN_PRIMARY};
                background-color: #081c38;
            }}
        """)
        self._input_edit.returnPressed.connect(self._on_send)
        center_h.addWidget(self._input_edit, 1)

        send_btn = QPushButton("Send ›")
        send_btn.setFont(QFont(JarvisTheme.FONT_FAMILY, 8, QFont.Weight.Bold))
        send_btn.setFixedHeight(32)
        send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        send_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #0b2f56;
                color: {JarvisTheme.CYAN_PRIMARY};
                border: 1px solid {JarvisTheme.CYAN_PRIMARY};
                border-radius: 8px;
                padding-left: 12px;
                padding-right: 12px;
            }}
            QPushButton:hover {{
                background-color: #0e3d6f;
                color: {JarvisTheme.WHITE_GLOW};
            }}
            QPushButton:pressed {{
                background-color: #08213e;
            }}
        """)
        send_btn.clicked.connect(self._on_send)
        center_h.addWidget(send_btn)

        layout.addLayout(center_h, 1)

        # 3. Right: Cinematic Mode + Volume controls
        right_h = QHBoxLayout()
        right_h.setSpacing(10)

        cinematic_btn = QPushButton("⚙ Cinematic Mode")
        cinematic_btn.setFont(QFont(JarvisTheme.FONT_FAMILY, 8))
        cinematic_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        cinematic_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {JarvisTheme.TEXT_MUTED};
                border: 1px solid #0d2e53;
                border-radius: 8px;
                padding: 4px 10px;
            }}
            QPushButton:hover {{
                color: {JarvisTheme.TEXT_PRIMARY};
                border: 1px solid {JarvisTheme.CYAN_DIM};
            }}
        """)
        cinematic_btn.clicked.connect(self.cinematic_toggled.emit)
        right_h.addWidget(cinematic_btn)

        vol_lbl = QLabel("🔊")
        vol_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; font-size: 11px; background: transparent;")
        right_h.addWidget(vol_lbl)

        meter = LevelMeter(self)
        right_h.addWidget(meter)

        layout.addLayout(right_h)

    def _on_send(self) -> None:
        """Handle submit button or enter key."""
        text = self._input_edit.text().strip()
        if text:
            self._input_edit.clear()
            self.command_submitted.emit(text)

    def set_status_message(self, message: str) -> None:
        """Update the bottom status text."""
        self._status_lbl.setText(message)
