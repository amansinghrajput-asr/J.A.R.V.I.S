"""Central Animated J.A.R.V.I.S. Arc Reactor Core for Phase 23.2.

Renders the multi-ring concentric glowing sci-fi reactor core matching
the approved jarvis_ui_reference.png design. Features:
- Rotating segmented rings (clockwise and counter-clockwise)
- Smooth breathing/pulsing radius and alpha glow modulation
- Dynamic audio-reactive scale expansion driven by mic RMS amplitude
- State-driven reactive color palettes (Idle, Listening, Thinking, Executing, Speaking, Error)
- Antialiased high-precision vector QPainter rendering
"""

from __future__ import annotations

import math
import time
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from app.ui.styles import JarvisTheme, ensure_fonts_loaded, get_state_color


class ArcReactorCore(QWidget):
    """Custom animated vector widget rendering the iconic J.A.R.V.I.S reactor core."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize ArcReactorCore widget with 60 FPS animation timer."""
        super().__init__(parent)
        ensure_fonts_loaded()

        self.setMinimumSize(JarvisTheme.CORE_MIN_SIZE, JarvisTheme.CORE_MIN_SIZE)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)

        # Operational State
        self._state: str = "IDLE"
        self._status_text: str = "IDLE"
        self._primary_color: QColor = QColor(JarvisTheme.CYAN_PRIMARY)

        # Audio Telemetry
        self._amplitude: float = 0.0
        self._smoothed_amplitude: float = 0.0

        # Animation Coordinates & Timers
        self._start_time = time.perf_counter()
        self._outer_rotation_deg: float = 0.0
        self._inner_rotation_deg: float = 0.0
        self._pulse_phase: float = 0.0

        # 60 FPS Animation Refresh Timer (~16.6 ms)
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(16)
        self._anim_timer.timeout.connect(self._on_animation_tick)
        self._anim_timer.start()

    # --------------------------------------------------------------------------
    # Public Slots & Telemetry Setters
    # --------------------------------------------------------------------------

    def set_state(self, state_name: str) -> None:
        """Update core visual state and reactive theme colors."""
        clean_state = (state_name or "IDLE").upper()
        if clean_state != self._state:
            self._state = clean_state
            self._status_text = clean_state
            hex_color = get_state_color(clean_state)
            self._primary_color = QColor(hex_color)
            self.update()

    @property
    def state(self) -> str:
        """Return current operational state name."""
        return self._state

    def set_amplitude(self, amplitude: float) -> None:
        """Update audio-reactive amplitude level (clamped 0.0 - 1.0)."""
        self._amplitude = max(0.0, min(1.0, float(amplitude)))

    # --------------------------------------------------------------------------
    # Animation Tick Logic
    # --------------------------------------------------------------------------

    def _on_animation_tick(self) -> None:
        """Update angles, breathing cycles, and smooth amplitude every frame."""
        now = time.perf_counter()
        dt = 0.016  # approx delta

        # Rotation speeds based on state
        rot_mult = 1.0
        if self._state == "THINKING":
            rot_mult = 3.2
        elif self._state in ("PLANNING", "EXECUTING"):
            rot_mult = 2.0
        elif self._state == "LISTENING":
            rot_mult = 1.5

        self._outer_rotation_deg = (self._outer_rotation_deg + 0.5 * rot_mult) % 360.0
        self._inner_rotation_deg = (self._inner_rotation_deg - 0.7 * rot_mult) % 360.0

        # Sine wave breathing: slow 0.5 Hz expansion/contraction
        breath_speed = 3.0 if self._state == "LISTENING" else 1.8
        self._pulse_phase = (now * breath_speed) % (2 * math.pi)

        # Smooth audio amplitude tracking for fluid organic expansion
        self._smoothed_amplitude += (self._amplitude - self._smoothed_amplitude) * 0.25

        self.update()

    # --------------------------------------------------------------------------
    # Paint Event (Precision Sci-Fi Rendering)
    # --------------------------------------------------------------------------

    def paintEvent(self, event: Any) -> None:
        """Draw concentric animated sci-fi rings matching the reference image."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        width = self.width()
        height = self.height()
        center = QPointF(width / 2.0, height / 2.0)
        base_radius = min(width, height) / 2.0 - 24.0

        if base_radius <= 20:
            return

        # Modulations
        breath_scale = 1.0 + 0.03 * math.sin(self._pulse_phase)
        audio_boost = self._smoothed_amplitude * 0.12  # up to 12% expansion on loud audio
        effective_radius = base_radius * (breath_scale + audio_boost)

        # 1. Background Radial Atmosphere Glow
        self._draw_ambient_glow(painter, center, effective_radius)

        # 2. Concentric Rings System
        self._draw_outer_segmented_ring(painter, center, effective_radius * 0.96)
        self._draw_middle_tech_ring(painter, center, effective_radius * 0.82)
        self._draw_inner_glow_ring(painter, center, effective_radius * 0.68)
        self._draw_core_border_ring(painter, center, effective_radius * 0.56)

        # 3. Particle Sparkles / Glow Flares
        self._draw_orbiting_nodes(painter, center, effective_radius * 0.75)

        # 4. Center Typography ("J.A.R.V.I.S" + State)
        self._draw_center_typography(painter, center, effective_radius * 0.50)

    # --------------------------------------------------------------------------
    # Ring Drawing Helpers
    # --------------------------------------------------------------------------

    def _draw_ambient_glow(self, painter: QPainter, center: QPointF, radius: float) -> None:
        """Render soft radial atmosphere glow behind rings."""
        glow = QRadialGradient(center, radius * 1.2)
        c_glow = QColor(self._primary_color)
        c_glow.setAlpha(45)
        glow.setColorAt(0.0, c_glow)
        c_glow.setAlpha(15)
        glow.setColorAt(0.6, c_glow)
        glow.setColorAt(1.0, QColor(0, 0, 0, 0))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(glow))
        painter.drawEllipse(center, radius * 1.2, radius * 1.2)

    def _draw_outer_segmented_ring(self, painter: QPainter, center: QPointF, radius: float) -> None:
        """Render outermost segmented dashed ring rotating clockwise."""
        painter.save()
        painter.translate(center)
        painter.rotate(self._outer_rotation_deg)

        pen = QPen(self._primary_color)
        pen.setWidthF(2.0)
        pen.setDashPattern([12, 10, 4, 8, 20, 14])
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        rect = QRectF(-radius, -radius, radius * 2, radius * 2)
        painter.drawEllipse(rect)

        # 4 prominent tick markers at cardinal angles
        tick_pen = QPen(QColor(JarvisTheme.CYAN_BRIGHT))
        tick_pen.setWidthF(3.0)
        painter.setPen(tick_pen)
        for angle in (0, 90, 180, 270):
            rad = math.radians(angle)
            x1 = (radius - 8) * math.cos(rad)
            y1 = (radius - 8) * math.sin(rad)
            x2 = (radius + 6) * math.cos(rad)
            y2 = (radius + 6) * math.sin(rad)
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        painter.restore()

    def _draw_middle_tech_ring(self, painter: QPainter, center: QPointF, radius: float) -> None:
        """Render middle counter-rotating arc segments with bright cyan accents."""
        painter.save()
        painter.translate(center)
        painter.rotate(self._inner_rotation_deg)

        rect = QRectF(-radius, -radius, radius * 2, radius * 2)

        # Base faint thin circle
        faint_pen = QPen(QColor(self._primary_color.red(), self._primary_color.green(), self._primary_color.blue(), 50))
        faint_pen.setWidthF(1.0)
        painter.setPen(faint_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(rect)

        # Bold segmented curved arcs
        arc_pen = QPen(self._primary_color)
        arc_pen.setWidthF(3.5)
        arc_pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(arc_pen)

        # Draw 3 balanced arc segments
        span_angle = 70 * 16
        for start_deg in (0, 120, 240):
            painter.drawArc(rect, int(start_deg * 16), span_angle)

        painter.restore()

    def _draw_inner_glow_ring(self, painter: QPainter, center: QPointF, radius: float) -> None:
        """Render high-intensity glowing energy ring surrounding the center."""
        # Multi-pass bloom simulation
        glow_colors = [
            (QColor(self._primary_color.red(), self._primary_color.green(), self._primary_color.blue(), 30), 8.0),
            (QColor(self._primary_color.red(), self._primary_color.green(), self._primary_color.blue(), 90), 4.5),
            (QColor(JarvisTheme.CYAN_BRIGHT), 2.0),
        ]

        rect = QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2)
        for col, width in glow_colors:
            pen = QPen(col)
            pen.setWidthF(width)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(rect)

    def _draw_core_border_ring(self, painter: QPainter, center: QPointF, radius: float) -> None:
        """Render fine precision circular boundary around central dark circle."""
        pen = QPen(QColor(self._primary_color.red(), self._primary_color.green(), self._primary_color.blue(), 180))
        pen.setWidthF(1.5)
        painter.setPen(pen)

        # Center dark vignette backing
        dark_brush = QBrush(QColor(JarvisTheme.BG_MAIN))
        painter.setBrush(dark_brush)
        painter.drawEllipse(center, radius, radius)

    def _draw_orbiting_nodes(self, painter: QPainter, center: QPointF, radius: float) -> None:
        """Draw luminous orbit sparkles around the mid-ring."""
        painter.save()
        painter.translate(center)
        painter.rotate(self._outer_rotation_deg * 1.5)

        flare_brush = QBrush(QColor(JarvisTheme.WHITE_GLOW))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(flare_brush)

        # 3 sparkling node dots
        for angle in (35, 155, 275):
            rad = math.radians(angle)
            x = radius * math.cos(rad)
            y = radius * math.sin(rad)
            painter.drawEllipse(QPointF(x, y), 3.0, 3.0)

        painter.restore()

    def _draw_center_typography(self, painter: QPainter, center: QPointF, core_radius: float) -> None:
        """Render centered J.A.R.V.I.S title and dynamic operational state subtitle."""
        painter.save()

        # Title: "J . A . R . V . I . S"
        title_font = QFont(JarvisTheme.FONT_FAMILY, max(10, int(core_radius * 0.28)), QFont.Weight.Bold)
        title_font.setFamilies(JarvisTheme.FONT_FAMILIES)
        title_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 4.0)
        painter.setFont(title_font)
        painter.setPen(QColor(JarvisTheme.TEXT_PRIMARY))

        title_text = "J.A.R.V.I.S"
        rect_title = QRectF(center.x() - core_radius * 1.6, center.y() - core_radius * 0.45, core_radius * 3.2, core_radius * 0.5)
        painter.drawText(rect_title, Qt.AlignmentFlag.AlignCenter, title_text)

        # Subtitle: State String (e.g. "LISTENING", "IDLE", "THINKING")
        sub_font = QFont(JarvisTheme.FONT_FAMILY, max(7, int(core_radius * 0.16)), QFont.Weight.DemiBold)
        sub_font.setFamilies(JarvisTheme.FONT_FAMILIES)
        sub_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 3.0)
        painter.setFont(sub_font)
        painter.setPen(self._primary_color)

        rect_sub = QRectF(center.x() - core_radius * 1.6, center.y() + core_radius * 0.05, core_radius * 3.2, core_radius * 0.4)
        painter.drawText(rect_sub, Qt.AlignmentFlag.AlignCenter, self._status_text)

        painter.restore()

    def closeEvent(self, event: Any) -> None:
        """Clean up animation timer upon widget destruction."""
        if self._anim_timer.isActive():
            self._anim_timer.stop()
        super().closeEvent(event)
