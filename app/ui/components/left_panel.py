"""Left Panel Component for J.A.R.V.I.S HUD.

Houses the Conversation history card and the Quick Actions card matching
jarvis_ui_reference.png. Dispatches commands strictly via signals.
"""

from __future__ import annotations

import datetime
import os
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
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


class TypingIndicatorBubble(QFrame):
    """Pulsing three-dot thinking/typing indicator shown during AI reasoning."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize TypingIndicatorBubble."""
        super().__init__(parent)
        self.setStyleSheet("QFrame { background: transparent; border: none; }")

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(2, 4, 2, 4)
        main_layout.setSpacing(10)

        # Avatar
        avatar = QLabel("J")
        avatar.setFixedSize(26, 26)
        avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
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
        main_layout.addWidget(avatar, alignment=Qt.AlignmentFlag.AlignTop)

        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(2)

        # Name label
        name_lbl = QLabel("J.A.R.V.I.S")
        name_font = QFont(JarvisTheme.FONT_FAMILY, 9, QFont.Weight.Bold)
        name_font.setFamilies(JarvisTheme.FONT_FAMILIES)
        name_lbl.setFont(name_font)
        name_lbl.setStyleSheet(f"color: {JarvisTheme.CYAN_PRIMARY};")
        content_layout.addWidget(name_lbl)

        # Animated dots label
        self._dots_lbl = QLabel("●  ○  ○")
        self._dots_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 8, QFont.Weight.Bold))
        self._dots_lbl.setStyleSheet(f"color: {JarvisTheme.CYAN_PRIMARY}; line-height: 120%;")
        content_layout.addWidget(self._dots_lbl)

        main_layout.addLayout(content_layout)

        # 250ms animation timer
        self._dot_state = 0
        self._timer = QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

    def _on_tick(self) -> None:
        """Cycle three-dot thinking pattern."""
        patterns = ["●  ○  ○", "○  ●  ○", "○  ○  ●"]
        self._dot_state = (self._dot_state + 1) % len(patterns)
        self._dots_lbl.setText(patterns[self._dot_state])

    def stop(self) -> None:
        """Safely stop the animation timer."""
        if self._timer.isActive():
            self._timer.stop()


class MessageBubble(QFrame):
    """Individual conversation bubble for User or J.A.R.V.I.S with progressive typewriter."""

    def __init__(
        self,
        sender: str,
        text: str,
        timestamp: Optional[str] = None,
        is_error: bool = False,
        progressive: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        """Initialize MessageBubble."""
        super().__init__(parent)
        is_jarvis = (sender.upper() == "J.A.R.V.I.S")
        time_str = timestamp or datetime.datetime.now().strftime("%I:%M %p")
        self._full_text = text
        self._is_error = is_error

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
        if is_error:
            avatar = QLabel("⊗" if is_jarvis else "👤")
            avatar_style = """
                QLabel {
                    background-color: #2b0b0b;
                    color: #ef4444;
                    border: 1px solid #7f1d1d;
                    border-radius: 13px;
                    font-weight: bold;
                    font-size: 11px;
                }
            """
        elif is_jarvis:
            avatar = QLabel("J")
            avatar_style = f"""
                QLabel {{
                    background-color: #0d3862;
                    color: {JarvisTheme.CYAN_PRIMARY};
                    border: 1px solid {JarvisTheme.CYAN_PRIMARY};
                    border-radius: 13px;
                    font-weight: bold;
                    font-size: 11px;
                }}
            """
        else:
            avatar = QLabel("👤")
            avatar_style = f"""
                QLabel {{
                    background-color: #0b2246;
                    color: {JarvisTheme.TEXT_MUTED};
                    border: 1px solid {JarvisTheme.BG_CARD_BORDER};
                    border-radius: 13px;
                    font-size: 11px;
                }}
            """
        avatar.setFixedSize(26, 26)
        avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        avatar.setStyleSheet(avatar_style)
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
        if is_error:
            name_color = "#ef4444"
        elif is_jarvis:
            name_color = JarvisTheme.CYAN_PRIMARY
        else:
            name_color = JarvisTheme.TEXT_PRIMARY
        name_lbl.setStyleSheet(f"color: {name_color};")
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
        self.msg_lbl = QLabel()
        self.msg_lbl.setWordWrap(True)
        msg_font = QFont(JarvisTheme.FONT_FAMILY, 8, QFont.Weight.Normal)
        msg_font.setFamilies(JarvisTheme.FONT_FAMILIES)
        self.msg_lbl.setFont(msg_font)

        if is_error:
            text_color = "#f87171"
        else:
            text_color = JarvisTheme.TEXT_PRIMARY
        self.msg_lbl.setStyleSheet(f"color: {text_color}; line-height: 120%;")
        content_layout.addWidget(self.msg_lbl)

        main_layout.addLayout(content_layout)

        # Check for headless test environment
        is_test_env = (os.environ.get("QT_QPA_PLATFORM") == "offscreen" or os.environ.get("JARVIS_TEST_MODE") == "1")

        self._type_index = 0
        self._type_timer: Optional[QTimer] = None

        if progressive and is_jarvis and not is_test_env and len(text) > 0:
            # Start smooth progressive typewriter
            self.msg_lbl.setText("")
            self._type_timer = QTimer(self)
            self._type_timer.setInterval(16)
            self._type_timer.timeout.connect(self._on_type_tick)
            self._type_timer.start()
        else:
            self.msg_lbl.setText(text)

    def _on_type_tick(self) -> None:
        """Incrementally type out response text."""
        step = max(3, len(self._full_text) // 30)
        self._type_index += step
        if self._type_index >= len(self._full_text):
            self.msg_lbl.setText(self._full_text)
            if self._type_timer is not None:
                self._type_timer.stop()
        else:
            self.msg_lbl.setText(self._full_text[:self._type_index])

    def complete_immediately(self) -> None:
        """Instantly finalize the full message text."""
        if self._type_timer is not None and self._type_timer.isActive():
            self._type_timer.stop()
        self.msg_lbl.setText(self._full_text)

    def text(self) -> str:
        """Return the visible message text."""
        return self.msg_lbl.text()


class ConversationCard(GlassPanel):
    """Conversation history display panel with live updates and persistence."""

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

        # Futuristic empty state
        self._empty_state = QFrame(self._container)
        self._empty_state.setStyleSheet("background: transparent; border: none;")
        empty_layout = QVBoxLayout(self._empty_state)
        empty_layout.setContentsMargins(12, 36, 12, 36)
        empty_layout.setSpacing(6)
        empty_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        empty_icon = QLabel("💬")
        empty_icon.setFont(QFont(JarvisTheme.FONT_FAMILY, 20))
        empty_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_icon.setStyleSheet(f"color: {JarvisTheme.CYAN_DIM}; background: transparent; border: none;")
        empty_layout.addWidget(empty_icon)

        empty_title = QLabel("NO CONVERSATION HISTORY")
        empty_title.setFont(QFont(JarvisTheme.FONT_FAMILY, 8, QFont.Weight.Bold))
        empty_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_title.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
        empty_layout.addWidget(empty_title)

        empty_sub = QLabel("Speak or enter a command below to begin.")
        empty_sub.setFont(QFont(JarvisTheme.FONT_FAMILY, 7))
        empty_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_sub.setStyleSheet("color: #1e3a5f; background: transparent; border: none;")
        empty_layout.addWidget(empty_sub)

        self._messages_layout.addWidget(self._empty_state)
        self._messages_layout.addStretch(1)

        self._scroll.setWidget(self._container)
        self.content_layout.addWidget(self._scroll, 1)

        self._typing_indicator: Optional[TypingIndicatorBubble] = None
        self._message_count: int = 0

    def add_message(
        self,
        sender: str,
        text: str,
        timestamp: Optional[str] = None,
        *,
        is_error: bool = False,
        progressive: bool = True,
    ) -> MessageBubble:
        """Add a new message bubble to the conversation display."""
        # If assistant response arrived, hide thinking indicator
        if sender.upper() == "J.A.R.V.I.S":
            self.hide_typing_indicator()

        self._empty_state.setVisible(False)

        bubble = MessageBubble(
            sender,
            text,
            timestamp=timestamp,
            is_error=is_error,
            progressive=progressive,
            parent=self._container,
        )
        # Insert before bottom stretch
        count = self._messages_layout.count()
        self._messages_layout.insertWidget(max(0, count - 1), bubble)
        self._message_count += 1

        # Scroll to bottom
        QScrollArea.ensureWidgetVisible(self._scroll, bubble)
        return bubble

    def show_typing_indicator(self) -> None:
        """Display the animated thinking/typing indicator."""
        if self._typing_indicator is None:
            self._empty_state.setVisible(False)
            self._typing_indicator = TypingIndicatorBubble(parent=self._container)
            count = self._messages_layout.count()
            self._messages_layout.insertWidget(max(0, count - 1), self._typing_indicator)
            QScrollArea.ensureWidgetVisible(self._scroll, self._typing_indicator)

    def hide_typing_indicator(self) -> None:
        """Hide and delete the thinking/typing indicator."""
        if self._typing_indicator is not None:
            self._typing_indicator.stop()
            self._messages_layout.removeWidget(self._typing_indicator)
            self._typing_indicator.setParent(None)
            self._typing_indicator.deleteLater()
            self._typing_indicator = None

        if self._message_count == 0:
            self._empty_state.setVisible(True)

    def load_history(self, records: list[dict[str, Any]]) -> None:
        """Populate conversation history from loaded memory records."""
        # Clear existing bubbles (skip empty_state and stretch)
        self.clear_conversation()

        if not records:
            self._empty_state.setVisible(True)
            return

        self._empty_state.setVisible(False)
        for rec in records:
            sender = str(rec.get("sender", "You"))
            text = str(rec.get("text", ""))
            ts = rec.get("timestamp")
            is_err = bool(rec.get("is_error", False))
            self.add_message(sender, text, timestamp=ts, is_error=is_err, progressive=False)

    def clear_conversation(self) -> None:
        """Clear all conversation messages and reset to empty state."""
        self.hide_typing_indicator()

        # Remove all widgets except empty state and bottom stretch
        while self._messages_layout.count() > 2:
            item = self._messages_layout.takeAt(1)
            if item.widget():
                w = item.widget()
                if w != self._empty_state:
                    w.setParent(None)
                    w.deleteLater()

        self._message_count = 0
        self._empty_state.setVisible(True)


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
