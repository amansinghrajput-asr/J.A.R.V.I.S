"""Visual Style System and Theming for J.A.R.V.I.S. Phase 23.2.

Defines exact color palettes, glow effects, typography specs, and styling
constants directly extracted from the approved jarvis_ui_reference.png design.
"""

from __future__ import annotations

from typing import Final


class JarvisTheme:
    """Color palette and visual constants for J.A.R.V.I.S Sci-Fi HUD."""

    # Backgrounds & Glassmorphism
    BG_MAIN: Final[str] = "#030812"              # Deep dark space navy/black
    BG_CARD: Final[str] = "#061226"              # Semi-transparent dark blue glass
    BG_CARD_BORDER: Final[str] = "#0b2648"       # Subtle card border
    BG_CARD_BORDER_GLOW: Final[str] = "#124b82"  # Highlighted card border
    BG_HOVER: Final[str] = "#0b2246"             # Interactive item hover

    # Core Sci-Fi Glows & Accents
    CYAN_PRIMARY: Final[str] = "#00d4ff"         # Primary glowing cyan
    CYAN_BRIGHT: Final[str] = "#5ce1e6"          # Bright core white-cyan
    CYAN_DIM: Final[str] = "#007a99"             # Dim peripheral cyan
    BLUE_ELECTRIC: Final[str] = "#0088ff"        # Electric arc reactor blue
    BLUE_DEEP: Final[str] = "#004488"            # Deep outer ring blue
    WHITE_GLOW: Final[str] = "#ffffff"           # Center text and hot spots

    # State Indicator Colors
    STATE_IDLE: Final[str] = "#00b4d8"           # Calm cyan
    STATE_LISTENING: Final[str] = "#00f0ff"      # Vibrant glowing cyan
    STATE_THINKING: Final[str] = "#38bdf8"       # Bright azure
    STATE_PLANNING: Final[str] = "#818cf8"       # Indigo/violet
    STATE_EXECUTING: Final[str] = "#38bdf8"      # Pulsing cyan
    STATE_SPEAKING: Final[str] = "#22d3ee"       # Resonant cyan
    STATE_CONFIRMATION: Final[str] = "#f59e0b"   # Amber warning
    STATE_ERROR: Final[str] = "#ef4444"          # Crimson alert

    # Text Colors
    TEXT_PRIMARY: Final[str] = "#e2f1ff"         # Crisp readable white-blue
    TEXT_MUTED: Final[str] = "#6282a5"           # Subdued secondary text
    TEXT_ACCENT: Final[str] = "#00d4ff"          # Glowing label text

    # Typography
    FONT_FAMILY: Final[str] = "Segoe UI"
    FONT_FAMILIES: Final[list[str]] = ["Segoe UI", "Segoe UI Emoji", "Arial", "Helvetica", "sans-serif"]
    FONT_FAMILY_MONO: Final[str] = "Consolas"

    # Core Arc Reactor Dimensions
    CORE_MIN_SIZE: Final[int] = 300
    CORE_DEFAULT_SIZE: Final[int] = 420


def ensure_fonts_loaded() -> None:
    """Ensure fonts are available in Qt font database (crucial for headless/offscreen validation)."""
    try:
        import os
        from PySide6.QtGui import QFontDatabase

        if not QFontDatabase.families():
            candidates = [
                r"C:\Windows\Fonts\segoeui.ttf",
                r"C:\Windows\Fonts\segoeuib.ttf",
                r"C:\Windows\Fonts\seguisb.ttf",
                r"C:\Windows\Fonts\seguiemj.ttf",
                r"C:\Windows\Fonts\arial.ttf",
            ]
            for font_path in candidates:
                if os.path.exists(font_path):
                    QFontDatabase.addApplicationFont(font_path)
    except Exception:
        pass


def get_state_color(state_name: str) -> str:
    """Return the theme color hex associated with a given assistant state string."""
    name = (state_name or "").upper()
    if name == "LISTENING":
        return JarvisTheme.STATE_LISTENING
    if name == "THINKING":
        return JarvisTheme.STATE_THINKING
    if name in ("PLANNING", "EXECUTING"):
        return JarvisTheme.STATE_EXECUTING
    if name == "SPEAKING":
        return JarvisTheme.STATE_SPEAKING
    if name == "AWAITING_CONFIRMATION":
        return JarvisTheme.STATE_CONFIRMATION
    if name == "ERROR":
        return JarvisTheme.STATE_ERROR
    return JarvisTheme.STATE_IDLE
