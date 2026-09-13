"""Circular Radial Gauge Components for J.A.R.V.I.S HUD.

Implements antialiased circular arc dials for CPU, RAM, and GPU telemetry
matching the reference design in jarvis_ui_reference.png.
"""

from __future__ import annotations

import math
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from app.ui.styles import JarvisTheme, ensure_fonts_loaded


class CircularGauge(QWidget):
    """Futuristic circular arc dial widget displaying telemetry percentage."""

    def __init__(
        self,
        label: str = "CPU",
        value: float = 32.0,
        parent: Optional[QWidget] = None,
        *,
        accent_color: str = JarvisTheme.CYAN_PRIMARY,
        track_color: str = "#0b2648",
    ) -> None:
        """Initialize CircularGauge.

        Args:
            label: Telemetry label (e.g., 'CPU', 'RAM', 'GPU').
            value: Initial percentage value (0.0 - 100.0).
            parent: Parent QWidget.
            accent_color: Hex color for illuminated progress arc.
            track_color: Hex color for unlit background track.
        """
        super().__init__(parent)
        ensure_fonts_loaded()

        self._label = label
        self._value = max(0.0, min(100.0, float(value)))
        self._accent_color = QColor(accent_color)
        self._track_color = QColor(track_color)

        self.setMinimumSize(70, 70)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Preferred)

    @property
    def value(self) -> float:
        """Return current percentage value."""
        return self._value

    def set_value(self, val: float) -> None:
        """Update percentage value and trigger repaint."""
        clamped = max(0.0, min(100.0, float(val)))
        if abs(clamped - self._value) > 0.1:
            self._value = clamped
            self.update()

    def set_label(self, label: str) -> None:
        """Update gauge label."""
        self._label = label
        self.update()

    def paintEvent(self, event: object) -> None:
        """Draw antialiased circular arc dial, label, and percentage value."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        w = float(self.width())
        h = float(self.height())
        size = min(w, h)
        center = QPointF(w / 2.0, h / 2.0)
        radius = (size - 14.0) / 2.0

        arc_rect = QRectF(center.x() - radius, center.y() - radius, radius * 2.0, radius * 2.0)

        # Arc configuration: 240-degree sweep, centered open at the bottom
        # Qt angles are in 16ths of a degree. 0 is at 3 o'clock.
        start_angle_deg = 210.0
        total_span_deg = -240.0

        # 1. Background unlit track
        track_pen = QPen(self._track_color)
        track_pen.setWidthF(4.0)
        track_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(track_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(arc_rect, int(start_angle_deg * 16), int(total_span_deg * 16))

        # 2. Illuminated active progress arc
        progress_frac = self._value / 100.0
        active_span_deg = total_span_deg * progress_frac

        glow_color = QColor(self._accent_color)
        glow_color.setAlpha(240)
        active_pen = QPen(glow_color)
        active_pen.setWidthF(4.5)
        active_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(active_pen)
        painter.drawArc(arc_rect, int(start_angle_deg * 16), int(active_span_deg * 16))

        # 3. Label text (top center inside dial)
        label_font = QFont(JarvisTheme.FONT_FAMILY, max(6, int(radius * 0.22)), QFont.Weight.Bold)
        label_font.setFamilies(JarvisTheme.FONT_FAMILIES)
        painter.setFont(label_font)
        painter.setPen(QColor(JarvisTheme.TEXT_MUTED))
        lbl_rect = QRectF(center.x() - radius, center.y() - radius * 0.55, radius * 2.0, radius * 0.45)
        painter.drawText(lbl_rect, Qt.AlignmentFlag.AlignCenter, self._label.upper())

        # 4. Percentage value text (middle center)
        val_font = QFont(JarvisTheme.FONT_FAMILY, max(8, int(radius * 0.38)), QFont.Weight.Bold)
        val_font.setFamilies(JarvisTheme.FONT_FAMILIES)
        painter.setFont(val_font)
        painter.setPen(QColor(JarvisTheme.TEXT_PRIMARY))
        val_rect = QRectF(center.x() - radius, center.y() - radius * 0.15, radius * 2.0, radius * 0.55)
        painter.drawText(val_rect, Qt.AlignmentFlag.AlignCenter, f"{int(round(self._value))}%")
