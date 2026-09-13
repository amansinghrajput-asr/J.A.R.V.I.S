"""UI Components Package for J.A.R.V.I.S. Phase 23.2.

Contains modular HUD widgets matching the approved Sci-Fi reference design.
"""

from __future__ import annotations

from app.ui.components.arc_reactor import ArcReactorCore
from app.ui.components.bottom_bar import BottomBar
from app.ui.components.center_panel import CenterPanel
from app.ui.components.dials import CircularGauge
from app.ui.components.glass_panel import GlassPanel
from app.ui.components.header import HeaderBar
from app.ui.components.left_panel import LeftPanel
from app.ui.components.right_panel import RightPanel
from app.ui.components.waveform import WaveformVisualizer

__all__ = [
    "ArcReactorCore",
    "BottomBar",
    "CenterPanel",
    "CircularGauge",
    "GlassPanel",
    "HeaderBar",
    "LeftPanel",
    "RightPanel",
    "WaveformVisualizer",
]
