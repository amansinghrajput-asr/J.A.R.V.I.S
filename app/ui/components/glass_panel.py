"""Reusable Glassmorphic Panel Component for J.A.R.V.I.S HUD.

Implements the dark navy translucent glass card with glowing borders,
rounded corners, and optional standardized card header matching the
approved jarvis_ui_reference.png design.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.ui.styles import JarvisTheme, ensure_fonts_loaded


class GlassPanel(QFrame):
    """Semi-transparent dark blue glass container with subtle cyan border highlight."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        border_color: str = JarvisTheme.BG_CARD_BORDER,
        bg_color: str = JarvisTheme.BG_CARD,
        border_radius: int = 12,
        has_glow: bool = False,
    ) -> None:
        """Initialize GlassPanel.

        Args:
            parent: Parent QWidget.
            border_color: Hex color for panel border.
            bg_color: Hex color for card glass background.
            border_radius: Corner radius in pixels.
            has_glow: Whether to draw a subtle bright outer glow.
        """
        super().__init__(parent)
        ensure_fonts_loaded()

        self._border_color = QColor(border_color)
        self._bg_color = QColor(bg_color)
        self._border_radius = border_radius
        self._has_glow = has_glow

        self.setObjectName("GlassPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Preferred)

        # Base stylesheet for glassmorphism
        border_css = f"border: 1px solid {JarvisTheme.BG_CARD_BORDER_GLOW};" if has_glow else f"border: 1px solid {border_color};"
        self.setStyleSheet(f"""
            QFrame#GlassPanel {{
                background-color: {bg_color};
                {border_css}
                border-radius: {border_radius}px;
            }}
        """)

        # Main content layout
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(14, 12, 14, 12)
        self._layout.setSpacing(10)

    @property
    def content_layout(self) -> QVBoxLayout:
        """Return the inner content layout for adding child widgets."""
        return self._layout

    def add_card_header(
        self,
        title: str,
        icon_text: str = "",
        right_badge: Optional[QWidget] = None,
    ) -> QHBoxLayout:
        """Add a standardized futuristic card header with icon, title, and right badge.

        Args:
            title: Uppercase section title (e.g., 'CONVERSATION', 'CURRENT TASK').
            icon_text: Optional unicode/glyph icon.
            right_badge: Optional custom widget on the right (e.g. status dot or '2 / 4').

        Returns:
            The header layout.
        """
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 4)
        header_layout.setSpacing(8)

        if icon_text:
            icon_lbl = QLabel(icon_text)
            icon_lbl.setStyleSheet(f"color: {JarvisTheme.CYAN_PRIMARY}; font-size: 13px; background: transparent; border: none;")
            header_layout.addWidget(icon_lbl)

        title_lbl = QLabel(title.upper())
        font = QFont(JarvisTheme.FONT_FAMILY, 10, QFont.Weight.Bold)
        font.setFamilies(JarvisTheme.FONT_FAMILIES)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.5)
        title_lbl.setFont(font)
        title_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_PRIMARY}; background: transparent; border: none;")
        header_layout.addWidget(title_lbl)

        header_layout.addStretch(1)

        if right_badge is not None:
            header_layout.addWidget(right_badge)

        self._layout.addLayout(header_layout)
        return header_layout
