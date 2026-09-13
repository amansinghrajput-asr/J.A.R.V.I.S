"""Temporary Visual Preview for J.A.R.V.I.S. Core and Waveform Visualizer.

Phase 23.2.3:
Displays ONLY the animated ArcReactorCore and WaveformVisualizer centered on
the approved deep navy/black Sci-Fi background.
Cycles automatically through: IDLE -> LISTENING -> THINKING -> EXECUTING -> SPEAKING
and simulates dynamic audio amplitude variation for visual inspection.
Can also capture a screenshot to brain artifacts for verification in headless mode.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys
import time

# Ensure repository root is on sys.path when executed directly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from app.ui.components.arc_reactor import ArcReactorCore
from app.ui.components.waveform import WaveformVisualizer
from app.ui.styles import JarvisTheme


class JarvisCorePreviewWindow(QMainWindow):
    """Isolated preview window for visual validation of core animations."""

    STATES = ["IDLE", "LISTENING", "THINKING", "EXECUTING", "SPEAKING"]

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("J.A.R.V.I.S. — Core Animation Preview (Phase 23.2.3)")
        self.resize(900, 720)
        self.setMinimumSize(700, 600)

        # Apply exact deep dark background matching jarvis_ui_reference.png
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(JarvisTheme.BG_MAIN))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        # Central Layout Container
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(40, 30, 40, 30)
        main_layout.setSpacing(15)

        # Minimal Top Telemetry Pill (State & Amplitude Display)
        self._pill_label = QLabel("STATE: IDLE | SIMULATED MIC RMS: 0.00", self)
        self._pill_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont(JarvisTheme.FONT_FAMILY_MONO, 11, QFont.Weight.Medium)
        self._pill_label.setFont(font)
        self._pill_label.setStyleSheet(
            f"color: {JarvisTheme.TEXT_MUTED}; "
            f"background-color: {JarvisTheme.BG_CARD}; "
            f"border: 1px solid {JarvisTheme.BG_CARD_BORDER}; "
            f"border-radius: 14px; "
            f"padding: 6px 16px;"
        )
        main_layout.addWidget(self._pill_label, 0, Qt.AlignmentFlag.AlignCenter)

        # Central Arc Reactor Core Widget
        self._core = ArcReactorCore(self)
        main_layout.addWidget(self._core, 1)

        # Waveform Visualizer directly beneath the Arc Reactor Core
        self._waveform = WaveformVisualizer(self, num_bars=48)
        main_layout.addWidget(self._waveform, 0)

        # Animation Simulation Parameters
        self._start_time = time.perf_counter()
        self._state_index = 0
        self._last_state_switch = time.perf_counter()
        self._state_switch_interval = 4.0  # Cycle state every 4 seconds

        # 60 FPS Telemetry Simulation Timer (~16ms)
        self._sim_timer = QTimer(self)
        self._sim_timer.setInterval(16)
        self._sim_timer.timeout.connect(self._on_sim_tick)
        self._sim_timer.start()

    def _on_sim_tick(self) -> None:
        """Simulate realistic continuous amplitude modulation and state progression."""
        now = time.perf_counter()
        elapsed = now - self._start_time

        # 1. State Cycling (every 4 seconds)
        if now - self._last_state_switch >= self._state_switch_interval:
            self._state_index = (self._state_index + 1) % len(self.STATES)
            current_state = self.STATES[self._state_index]
            self._core.set_state(current_state)
            self._waveform.set_state(current_state)
            self._last_state_switch = now

        current_state = self.STATES[self._state_index]

        # 2. Simulated Audio Amplitude Modulation (0.0 to ~0.85 smoothly)
        # Combination of two low-frequency sine waves creating organic speech envelope
        if current_state in ("LISTENING", "SPEAKING"):
            base_wave = 0.5 * (1.0 + math.sin(elapsed * 4.5))
            flutter = 0.25 * (1.0 + math.sin(elapsed * 11.0))
            sim_amp = max(0.0, min(0.95, (base_wave * 0.7 + flutter * 0.3) * 0.85))
        elif current_state == "THINKING":
            # Subtle ambient brainwave vibration
            sim_amp = 0.08 + 0.05 * math.sin(elapsed * 8.0)
        else:
            # Idle gentle background noise
            sim_amp = 0.02 + 0.02 * math.sin(elapsed * 2.0)

        # Update widgets with simulated amplitude
        self._core.set_amplitude(sim_amp)
        self._waveform.set_amplitude(sim_amp)

        # Update top status display
        self._pill_label.setText(
            f"STATE: {current_state:<10} | SIMULATED MIC RMS: {sim_amp:.2f}"
        )


def capture_preview_screenshot(output_path: str, duration_sec: float = 2.0) -> None:
    """Run preview in offscreen headless mode and save a high-res screenshot."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance() or QApplication(sys.argv)
    window = JarvisCorePreviewWindow()
    window.resize(900, 720)
    window.show()

    # Step forward in time through a few simulation ticks
    start = time.perf_counter()
    while time.perf_counter() - start < duration_sec:
        app.processEvents()
        time.sleep(0.016)

    # Capture window screenshot
    pixmap = window.grab()
    pixmap.save(output_path, "PNG")
    window.close()


def main() -> int:
    """Launch the interactive preview or screenshot generator."""
    if "--screenshot" in sys.argv:
        target = (
            sys.argv[sys.argv.index("--screenshot") + 1]
            if len(sys.argv) > sys.argv.index("--screenshot") + 1
            else "preview.png"
        )
        duration = 2.0
        if "--duration" in sys.argv and len(sys.argv) > sys.argv.index("--duration") + 1:
            duration = float(sys.argv[sys.argv.index("--duration") + 1])
        capture_preview_screenshot(target, duration_sec=duration)
        print(f"Screenshot saved to: {target}")
        return 0

    app = QApplication.instance() or QApplication(sys.argv)
    window = JarvisCorePreviewWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
