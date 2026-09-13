"""Left Panel Component for J.A.R.V.I.S HUD.

Houses the Conversation history card and the Quick Actions card matching
jarvis_ui_reference.png. Dispatches commands strictly via signals.
"""

from __future__ import annotations

import datetime
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.ui.components.glass_panel import GlassPanel
from app.ui.styles import JarvisTheme, ensure_fonts_loaded


class MessageBubble(QFrame):
    """Individual conversation bubble for User or J.A.R.V.I.S."""

    def __init__(
        self,
        sender: str,
        text: str,
        timestamp: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        """Initialize MessageBubble."""
        super().__init__(parent)
        is_jarvis = (sender.upper() == "J.A.R.V.I.S")
        time_str = timestamp or datetime.datetime.now().strftime("%I:%M %p")

        self.setStyleSheet("""
            QFrame {
                background: transparent;
                border: none;
            }
        """)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(2, 4, 2, 4)
        main_layout.setSpacing(10)

        # Avatar
        avatar = QLabel("J" if is_jarvis else "👤")
        avatar.setFixedSize(26, 26)
        avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if is_jarvis:
            avatar.setStyleSheet(f"""
                QLabel {{
                    background-color: #0d3862;
                    color: {JarvisTheme.CYAN_PRIMARY};
                    border: 1px solid {JarvisTheme.CYAN_PRIMARY};
                    border-radius: 13px;
                    font-weight: bold;
                    font-size: 11px;
                }}
            """)
        else:
            avatar.setStyleSheet(f"""
                QLabel {{
                    background-color: #0b2246;
                    color: {JarvisTheme.TEXT_MUTED};
                    border: 1px solid {JarvisTheme.BG_CARD_BORDER};
                    border-radius: 13px;
                    font-size: 11px;
                }}
            """)
        main_layout.addWidget(avatar, alignment=Qt.AlignmentFlag.AlignTop)

        # Content layout
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(2)

        # Header: Name + Timestamp
        meta_layout = QHBoxLayout()
        meta_layout.setSpacing(6)

        name_lbl = QLabel(sender)
        name_font = QFont(JarvisTheme.FONT_FAMILY, 9, QFont.Weight.Bold)
        name_font.setFamilies(JarvisTheme.FONT_FAMILIES)
        name_lbl.setFont(name_font)
        name_lbl.setStyleSheet(f"color: {JarvisTheme.CYAN_PRIMARY if is_jarvis else JarvisTheme.TEXT_PRIMARY};")
        meta_layout.addWidget(name_lbl)

        time_lbl = QLabel(time_str)
        time_font = QFont(JarvisTheme.FONT_FAMILY, 7)
        time_font.setFamilies(JarvisTheme.FONT_FAMILIES)
        time_lbl.setFont(time_font)
        time_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED};")
        meta_layout.addWidget(time_lbl)
        meta_layout.addStretch(1)

        content_layout.addLayout(meta_layout)

        # Message Text
        msg_lbl = QLabel(text)
        msg_lbl.setWordWrap(True)
        msg_font = QFont(JarvisTheme.FONT_FAMILY, 8, QFont.Weight.Normal)
        msg_font.setFamilies(JarvisTheme.FONT_FAMILIES)
        msg_lbl.setFont(msg_font)
        msg_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_PRIMARY if not is_jarvis else JarvisTheme.TEXT_PRIMARY}; line-height: 120%;")
        content_layout.addWidget(msg_lbl)

        main_layout.addLayout(content_layout)


class ConversationCard(GlassPanel):
    """Conversation history display panel with live updates."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize ConversationCard."""
        super().__init__(parent)

        # Header with "● Live" badge
        live_badge = QFrame()
        live_layout = QHBoxLayout(live_badge)
        live_layout.setContentsMargins(6, 2, 6, 2)
        live_layout.setSpacing(4)
        live_badge.setStyleSheet("""
            QFrame {
                background-color: #042116;
                border: 1px solid #14532d;
                border-radius: 8px;
            }
        """)
        live_lbl = QLabel("● Live")
        live_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 7, QFont.Weight.Bold))
        live_lbl.setStyleSheet("color: #22c55e; border: none; background: transparent;")
        live_layout.addWidget(live_lbl)
        wave_icon = QLabel("ıll")
        wave_icon.setFont(QFont(JarvisTheme.FONT_FAMILY, 7))
        wave_icon.setStyleSheet(f"color: {JarvisTheme.CYAN_PRIMARY}; border: none; background: transparent;")
        live_layout.addWidget(wave_icon)

        self.add_card_header("CONVERSATION", icon_text="💬", right_badge=live_badge)

        # Scroll area for conversation messages
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background: #040e1d;
                width: 5px;
                margin: 0px;
                border-radius: 2px;
            }
            QScrollBar::handle:vertical {
                background: #0d3862;
                min-height: 20px;
                border-radius: 2px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        self._container = QWidget()
        self._container.setStyleSheet("background: transparent;")
        self._messages_layout = QVBoxLayout(self._container)
        self._messages_layout.setContentsMargins(0, 0, 0, 0)
        self._messages_layout.setSpacing(8)
        self._messages_layout.addStretch(1)

        self._scroll.setWidget(self._container)
        self.content_layout.addWidget(self._scroll, 1)

        # Seed initial reference conversation messages
        self.add_message("You", "Open Chrome and search for today's weather.", "10:24 AM")
        self.add_message(
            "J.A.R.V.I.S",
            "Opening Chrome...\nSearching for today's weather in your location.",
            "10:24 AM",
        )
        self.add_message("You", "That's perfect, thanks!", "10:25 AM")

    def add_message(self, sender: str, text: str, timestamp: Optional[str] = None) -> None:
        """Add a new message bubble to the conversation display."""
        bubble = MessageBubble(sender, text, timestamp=timestamp, parent=self._container)
        # Insert before bottom stretch
        count = self._messages_layout.count()
        self._messages_layout.insertWidget(max(0, count - 1), bubble)
        # Scroll to bottom
        QScrollArea.ensureWidgetVisible(self._scroll, bubble)


class ActionButton(QPushButton):
    """Futuristic horizontal action row with icon, title, and right chevron."""

    def __init__(
        self,
        icon_symbol: str,
        title: str,
        command: str,
        parent: Optional[QWidget] = None,
        *,
        icon_color: str = JarvisTheme.CYAN_PRIMARY,
    ) -> None:
        """Initialize ActionButton."""
        super().__init__(parent)
        self._command = command

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(40)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: #061833;
                border: 1px solid {JarvisTheme.BG_CARD_BORDER};
                border-radius: 8px;
                text-align: left;
                padding-left: 10px;
                padding-right: 12px;
            }}
            QPushButton:hover {{
                background-color: #092448;
                border: 1px solid {JarvisTheme.CYAN_PRIMARY};
            }}
            QPushButton:pressed {{
                background-color: #0b2f5c;
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 10, 0)
        layout.setSpacing(10)

        icon_lbl = QLabel(icon_symbol)
        icon_lbl.setStyleSheet(f"color: {icon_color}; font-size: 14px; background: transparent; border: none;")
        layout.addWidget(icon_lbl)

        title_lbl = QLabel(title)
        title_font = QFont(JarvisTheme.FONT_FAMILY, 9, QFont.Weight.Medium)
        title_font.setFamilies(JarvisTheme.FONT_FAMILIES)
        title_lbl.setFont(title_font)
        title_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_PRIMARY}; background: transparent; border: none;")
        layout.addWidget(title_lbl)

        layout.addStretch(1)

        chevron = QLabel("›")
        chevron.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; font-size: 14px; font-weight: bold; background: transparent; border: none;")
        layout.addWidget(chevron)

    @property
    def command(self) -> str:
        """Return the command associated with this quick action."""
        return self._command


class QuickActionsCard(GlassPanel):
    """Quick Action button cards panel matching reference."""

    action_triggered = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize QuickActionsCard."""
        super().__init__(parent)

        self.add_card_header("QUICK ACTIONS", icon_text="⚡")

        actions = [
            ("🌐", "Open Chrome", "open chrome", "#38bdf8"),
            ("💻", "Open VS Code", "open vs code", "#00d4ff"),
            ("⏻", "Shutdown PC", "shutdown pc", "#ef4444"),
            ("📷", "Take Screenshot", "take screenshot", "#22d3ee"),
        ]

        for icon, title, cmd, color in actions:
            btn = ActionButton(icon, title, cmd, self, icon_color=color)
            btn.clicked.connect(lambda checked=False, c=cmd: self.action_triggered.emit(c))
            self.content_layout.addWidget(btn)


class LeftPanel(QWidget):
    """Container for Left HUD column (Conversation + Quick Actions)."""

    action_dispatched = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize LeftPanel."""
        super().__init__(parent)
        ensure_fonts_loaded()

        self.setFixedWidth(290)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.conversation_card = ConversationCard(self)
        layout.addWidget(self.conversation_card, 3)

        self.quick_actions_card = QuickActionsCard(self)
        self.quick_actions_card.action_triggered.connect(self.action_dispatched.emit)
        layout.addWidget(self.quick_actions_card, 2)
