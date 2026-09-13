"""UI Package for J.A.R.V.I.S. Phase 23.2.

Provides the PySide6 Sci-Fi Desktop HUD implementation.
Isolated from the core backend services, communicating exclusively
through the PresentationAdapter boundary.
"""

from __future__ import annotations

from app.ui.bridge import UIBridge
from app.ui.main_window import JarvisMainWindow
from app.ui.styles import JarvisTheme

__all__ = [
    "JarvisMainWindow",
    "JarvisTheme",
    "UIBridge",
]
