"""Audio-Reactive Waveform Visualizer for J.A.R.V.I.S. Phase 23.2.

Renders the horizontal audio frequency/amplitude bars and status caption
positioned directly below the Arc Reactor core in jarvis_ui_reference.png.
Driven dynamically by real microphone RMS telemetry from the Presentation Bridge.
"""

from __future__ import annotations

import math
import random
import time
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from app.ui.styles import JarvisTheme, ensure_fonts_loaded, get_state_color


class WaveformVisualizer(QWidget):
    """Animated horizontal audio equalizer bar visualizer."""

    def __init__(self, parent: Optional[QWidget] = None, num_bars: int = 48) -> None:
        """Initialize WaveformVisualizer with specified number of equalizer bars."""
        super().__init__(parent)
        ensure_fonts_loaded()

        self._num_bars = max(16, num_bars)
        self.setMinimumHeight(70)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # State & Colors
        self._state: str = "IDLE"
        self._status_text: str = "Listening..."
        self._primary_color = QColor(JarvisTheme.CYAN_PRIMARY)

        # Telemetry
        self._target_amplitude: float = 0.0
        self._current_amplitude: float = 0.0

        # Bar heights array and phase offsets
        self._bar_heights: list[float] = [0.08] * self._num_bars
        self._phase_offsets: list[float] = [random.uniform(0, math.pi * 2) for _ in range(self._num_bars)]

        # 60 FPS animation timer
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

    def set_amplitude(self, amp: float) -> None:
        """Update microphone amplitude level (0.0 to 1.0)."""
        self._target_amplitude = max(0.0, min(1.0, float(amp)))

    def set_state(self, state_name: str, status_text: Optional[str] = None) -> None:
        """Update visual state and caption text."""
        clean = (state_name or "IDLE").upper()
        self._state = clean
        self._primary_color = QColor(get_state_color(clean))

        if status_text is not None:
            self._status_text = status_text
        elif clean == "LISTENING":
            self._status_text = "Listening..."
        elif clean == "TRANSCRIBING":
            self._status_text = "Transcribing speech..."
        elif clean == "THINKING":
            self._status_text = "Analyzing intent..."
        elif clean == "SPEAKING":
            self._status_text = "Speaking response..."
        else:
            self._status_text = "Ready"
        self.update()

    def _on_tick(self) -> None:
        """Interpolate bar heights with gentle wave modulation and mic volume."""
        now = time.perf_counter()
        # Smooth overall amplitude
        self._current_amplitude += (self._target_amplitude - self._current_amplitude) * 0.3

        # Center bell-curve weight factor (bars in the center rise higher)
        mid = (self._num_bars - 1) / 2.0

        is_active = self._state in ("LISTENING", "SPEAKING")
        base_floor = 0.08 if is_active else 0.04
        amp_scale = max(self._current_amplitude, 0.15 if is_active else 0.0)

        for i in range(self._num_bars):
            dist = abs(i - mid) / mid
            envelope = math.exp(-2.2 * (dist ** 2))  # Gaussian bell envelope

            # Wave motion
            wave = 0.5 + 0.5 * math.sin(now * 6.0 + self._phase_offsets[i])
            target_h = base_floor + envelope * amp_scale * wave * 0.88

            # Smooth interpolation
            self._bar_heights[i] += (target_h - self._bar_heights[i]) * 0.35

        self.update()

    def paintEvent(self, event: Any) -> None:
        """Render vertical bars and centered caption text."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        w = self.width()
        h = self.height()
        bars_area_height = h - 24.0
        center_y = bars_area_height / 2.0

        total_bars = self._num_bars
        bar_width = max(2.0, min(5.0, (w * 0.75) / (total_bars * 1.5)))
        gap = bar_width * 0.7
        total_width = total_bars * (bar_width + gap) - gap
        start_x = (w - total_width) / 2.0

        # Draw Equalizer Bars
        painter.setPen(Qt.PenStyle.NoPen)
        for i in range(total_bars):
            bx = start_x + i * (bar_width + gap)
            bar_norm = min(1.0, max(0.04, self._bar_heights[i]))
            bar_pixel_h = bar_norm * (bars_area_height * 0.85)

            # Symmetrical bars extending up and down from centerline
            by = center_y - (bar_pixel_h / 2.0)
            bar_rect = QRectF(bx, by, bar_width, bar_pixel_h)

            # Gradient alpha: center bright cyan, top/bottom slight glow
            alpha = int(120 + 135 * bar_norm)
            col = QColor(self._primary_color)
            col.setAlpha(min(255, alpha))
            painter.setBrush(QBrush(col))
            painter.drawRoundedRect(bar_rect, bar_width / 2.0, bar_width / 2.0)

        # Draw Centered Subtitle Caption (e.g., "Listening...")
        caption_font = QFont(JarvisTheme.FONT_FAMILY, 10, QFont.Weight.Medium)
        caption_font.setFamilies(JarvisTheme.FONT_FAMILIES)
        painter.setFont(caption_font)
        painter.setPen(QColor(JarvisTheme.TEXT_MUTED if self._state == "IDLE" else JarvisTheme.CYAN_PRIMARY))
        text_rect = QRectF(0, h - 20.0, w, 20.0)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, self._status_text)

    def closeEvent(self, event: Any) -> None:
        """Stop animation timer on widget close."""
        if self._timer.isActive():
            self._timer.stop()
        super().closeEvent(event)
