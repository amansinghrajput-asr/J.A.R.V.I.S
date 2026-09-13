"""Header Bar Component for J.A.R.V.I.S HUD.

Implements the top bar featuring the glowing J.A.R.V.I.S emblem,
navigation tabs (Home, Activity, System, Settings), live system status badge,
control buttons, and user profile chip matching jarvis_ui_reference.png.
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from app.ui.styles import JarvisTheme, ensure_fonts_loaded


class BrandEmblem(QWidget):
    """Circular glowing J.A.R.V.I.S emblem with cyan neon ring."""

    def __init__(self, parent: Optional[QWidget] = None, size: int = 38) -> None:
        """Initialize BrandEmblem."""
        super().__init__(parent)
        self.setFixedSize(size, size)

    def paintEvent(self, event: object) -> None:
        """Draw circular neon badge with letter J."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        w = float(self.width())
        h = float(self.height())
        center = QPointF(w / 2.0, h / 2.0)
        radius = (w - 4.0) / 2.0

        # Outer glowing ring
        pen = QPen(QColor(JarvisTheme.CYAN_PRIMARY))
        pen.setWidthF(1.8)
        painter.setPen(pen)
        painter.setBrush(QColor("#06162e"))
        painter.drawEllipse(center, radius, radius)

        # Concentric thin inner ring
        inner_pen = QPen(QColor(JarvisTheme.CYAN_DIM))
        inner_pen.setWidthF(1.0)
        painter.setPen(inner_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(center, radius - 3.0, radius - 3.0)

        # Centered 'J' letter
        font = QFont(JarvisTheme.FONT_FAMILY, 14, QFont.Weight.Bold)
        font.setFamilies(JarvisTheme.FONT_FAMILIES)
        painter.setFont(font)
        painter.setPen(QColor(JarvisTheme.WHITE_GLOW))
        painter.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, "J")


class HeaderBar(QFrame):
    """Top navigation and telemetry header bar."""

    navigation_changed = Signal(str)
    voice_toggled = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize HeaderBar."""
        super().__init__(parent)
        ensure_fonts_loaded()

        self.setFixedHeight(56)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {JarvisTheme.BG_MAIN};
                border: none;
                border-bottom: 1px solid #091a32;
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 8, 18, 8)
        layout.setSpacing(16)

        # 1. Left Branding: Emblem + Titles
        brand_layout = QHBoxLayout()
        brand_layout.setSpacing(10)
        self._emblem = BrandEmblem(self, size=38)
        brand_layout.addWidget(self._emblem)

        titles_layout = QHBoxLayout()
        titles_layout.setSpacing(6)
        
        title_lbl = QLabel("J . A . R . V . I . S")
        title_font = QFont(JarvisTheme.FONT_FAMILY, 13, QFont.Weight.Bold)
        title_font.setFamilies(JarvisTheme.FONT_FAMILIES)
        title_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 3.0)
        title_lbl.setFont(title_font)
        title_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_PRIMARY}; background: transparent;")
        titles_layout.addWidget(title_lbl)

        sub_lbl = QLabel("TACTICAL AI ASSISTANT")
        sub_font = QFont(JarvisTheme.FONT_FAMILY, 8, QFont.Weight.DemiBold)
        sub_font.setFamilies(JarvisTheme.FONT_FAMILIES)
        sub_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.2)
        sub_lbl.setFont(sub_font)
        sub_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; padding-left: 8px;")
        titles_layout.addWidget(sub_lbl)

        brand_layout.addLayout(titles_layout)
        layout.addLayout(brand_layout)

        layout.addStretch(1)

        # 2. Center Navigation Tabs: HOME, ACTIVITY, SYSTEM, SETTINGS
        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(6)
        self._nav_buttons: dict[str, QPushButton] = {}
        for tab_name, icon in (("HOME", "⌂"), ("ACTIVITY", "▤"), ("SYSTEM", "⚙"), ("SETTINGS", "⛭")):
            btn = QPushButton(f"{icon}  {tab_name}")
            btn_font = QFont(JarvisTheme.FONT_FAMILY, 9, QFont.Weight.Bold)
            btn_font.setFamilies(JarvisTheme.FONT_FAMILIES)
            btn_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.0)
            btn.setFont(btn_font)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            
            is_active = (tab_name == "HOME")
            if is_active:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: #0b2f56;
                        color: {JarvisTheme.CYAN_PRIMARY};
                        border: 1px solid {JarvisTheme.CYAN_PRIMARY};
                        border-radius: 14px;
                        padding: 5px 16px;
                    }}
                """)
            else:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: transparent;
                        color: {JarvisTheme.TEXT_MUTED};
                        border: 1px solid transparent;
                        border-radius: 14px;
                        padding: 5px 16px;
                    }}
                    QPushButton:hover {{
                        background-color: #081a33;
                        color: {JarvisTheme.TEXT_PRIMARY};
                    }}
                """)
            btn.clicked.connect(lambda checked=False, t=tab_name: self._on_nav_clicked(t))
            self._nav_buttons[tab_name] = btn
            nav_layout.addWidget(btn)

        layout.addLayout(nav_layout)

        layout.addStretch(1)

        # 3. Right Controls & Profile
        right_layout = QHBoxLayout()
        right_layout.setSpacing(12)

        # System Online status badge
        self._online_badge = QLabel("● SYSTEM ONLINE")
        badge_font = QFont(JarvisTheme.FONT_FAMILY, 8, QFont.Weight.Bold)
        badge_font.setFamilies(JarvisTheme.FONT_FAMILIES)
        badge_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.0)
        self._online_badge.setFont(badge_font)
        self._online_badge.setStyleSheet("""
            QLabel {
                color: #22c55e;
                background-color: #042116;
                border: 1px solid #14532d;
                border-radius: 10px;
                padding: 4px 10px;
            }
        """)
        right_layout.addWidget(self._online_badge)

        # Mic Icon Button
        self._mic_btn = QPushButton("🎙")
        self._mic_btn.setFixedSize(30, 30)
        self._mic_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._mic_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #081a33;
                color: {JarvisTheme.CYAN_PRIMARY};
                border: 1px solid {JarvisTheme.BG_CARD_BORDER};
                border-radius: 15px;
                font-size: 13px;
            }}
            QPushButton:hover {{
                border: 1px solid {JarvisTheme.CYAN_PRIMARY};
            }}
        """)
        self._mic_btn.clicked.connect(self.voice_toggled.emit)
        right_layout.addWidget(self._mic_btn)

        # Bell notification icon button
        bell_btn = QPushButton("🔔")
        bell_btn.setFixedSize(30, 30)
        bell_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #081a33;
                color: {JarvisTheme.TEXT_MUTED};
                border: 1px solid {JarvisTheme.BG_CARD_BORDER};
                border-radius: 15px;
                font-size: 12px;
            }}
        """)
        right_layout.addWidget(bell_btn)

        # Menu icon button
        menu_btn = QPushButton("☰")
        menu_btn.setFixedSize(30, 30)
        menu_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #081a33;
                color: {JarvisTheme.TEXT_MUTED};
                border: 1px solid {JarvisTheme.BG_CARD_BORDER};
                border-radius: 15px;
                font-size: 12px;
            }}
        """)
        right_layout.addWidget(menu_btn)

        # User profile badge
        user_badge = QFrame()
        user_badge.setStyleSheet(f"""
            QFrame {{
                background-color: #081a33;
                border: 1px solid {JarvisTheme.BG_CARD_BORDER};
                border-radius: 15px;
            }}
        """)
        user_layout = QHBoxLayout(user_badge)
        user_layout.setContentsMargins(8, 2, 4, 2)
        user_layout.setSpacing(6)

        user_text = QLabel("User Aman")
        user_text.setFont(QFont(JarvisTheme.FONT_FAMILY, 8, QFont.Weight.Medium))
        user_text.setStyleSheet(f"color: {JarvisTheme.TEXT_PRIMARY}; border: none; background: transparent;")
        user_layout.addWidget(user_text)

        avatar_lbl = QLabel("👤")
        avatar_lbl.setFixedSize(24, 24)
        avatar_lbl.setStyleSheet(f"""
            QLabel {{
                background-color: #0f3460;
                color: {JarvisTheme.CYAN_PRIMARY};
                border: 1px solid {JarvisTheme.CYAN_DIM};
                border-radius: 12px;
                font-size: 11px;
            }}
        """)
        avatar_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        user_layout.addWidget(avatar_lbl)

        right_layout.addWidget(user_badge)

        layout.addLayout(right_layout)

    def _on_nav_clicked(self, active_tab: str) -> None:
        """Update active navigation tab styling and emit signal."""
        for tab_name, btn in self._nav_buttons.items():
            if tab_name == active_tab:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: #0b2f56;
                        color: {JarvisTheme.CYAN_PRIMARY};
                        border: 1px solid {JarvisTheme.CYAN_PRIMARY};
                        border-radius: 14px;
                        padding: 5px 16px;
                    }}
                """)
            else:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: transparent;
                        color: {JarvisTheme.TEXT_MUTED};
                        border: 1px solid transparent;
                        border-radius: 14px;
                        padding: 5px 16px;
                    }}
                    QPushButton:hover {{
                        background-color: #081a33;
                        color: {JarvisTheme.TEXT_PRIMARY};
                    }}
                """)
        self.navigation_changed.emit(active_tab)

    def set_system_status(self, is_online: bool, text: str = "SYSTEM ONLINE") -> None:
        """Update the system online status pill."""
        if is_online:
            self._online_badge.setText(f"● {text.upper()}")
            self._online_badge.setStyleSheet("""
                QLabel {
                    color: #22c55e;
                    background-color: #042116;
                    border: 1px solid #14532d;
                    border-radius: 10px;
                    padding: 4px 10px;
                }
            """)
        else:
            self._online_badge.setText(f"● {text.upper()}")
            self._online_badge.setStyleSheet("""
                QLabel {
                    color: #ef4444;
                    background-color: #2b0c0c;
                    border: 1px solid #7f1d1d;
                    border-radius: 10px;
                    padding: 4px 10px;
                }
            """)
