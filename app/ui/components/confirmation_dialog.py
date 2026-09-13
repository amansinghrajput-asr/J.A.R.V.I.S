"""Operator Confirmation Gateway Component for J.A.R.V.I.S HUD.

Displays an authoritative, futuristic warning card when the assistant enters
AssistantState.AWAITING_CONFIRMATION. Requires explicit operator approval
before destructive or restricted system skills can proceed.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.core.state import PendingConfirmation
from app.ui.components.glass_panel import GlassPanel
from app.ui.styles import JarvisTheme, ensure_fonts_loaded


class ConfirmationCard(GlassPanel):
    """Futuristic operator confirmation panel matching the J.A.R.V.I.S sci-fi aesthetic.

    Emits confirmed(confirmation_id) or cancelled(confirmation_id) once per decision,
    immediately locking the action buttons to prevent double-dispatch.
    """

    confirmed = Signal(str)
    cancelled = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize ConfirmationCard."""
        super().__init__(parent, border_radius=12)
        ensure_fonts_loaded()

        self._confirmation_id: str = ""
        self._resolved: bool = False

        # Amber warning border style override
        self.setObjectName("ConfirmationCardPanel")
        self.setStyleSheet(f"""
            QFrame#ConfirmationCardPanel {{
                background-color: #0d192e;
                border: 2px solid {JarvisTheme.STATE_CONFIRMATION};
                border-radius: 12px;
            }}
        """)
        self.content_layout.setContentsMargins(14, 12, 14, 12)
        self.content_layout.setSpacing(8)

        # 1. Header Banner
        hdr_layout = QHBoxLayout()
        hdr_layout.setSpacing(8)

        icon_lbl = QLabel("⚠️")
        icon_lbl.setStyleSheet(f"font-size: 16px; color: {JarvisTheme.STATE_CONFIRMATION}; background: transparent; border: none;")
        hdr_layout.addWidget(icon_lbl)

        title_lbl = QLabel("OPERATOR CONFIRMATION REQUIRED")
        title_font = QFont(JarvisTheme.FONT_FAMILY, 9, QFont.Weight.Bold)
        title_font.setFamilies(JarvisTheme.FONT_FAMILIES)
        title_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.0)
        title_lbl.setFont(title_font)
        title_lbl.setStyleSheet(f"color: {JarvisTheme.STATE_CONFIRMATION}; background: transparent; border: none;")
        hdr_layout.addWidget(title_lbl)

        hdr_layout.addStretch(1)

        self._risk_badge = QLabel("RISK: HIGH")
        risk_font = QFont(JarvisTheme.FONT_FAMILY, 7, QFont.Weight.Bold)
        self._risk_badge.setFont(risk_font)
        self._risk_badge.setStyleSheet("""
            QLabel {
                color: #ef4444;
                background-color: #2b0b0b;
                border: 1px solid #7f1d1d;
                border-radius: 4px;
                padding: 2px 8px;
            }
        """)
        hdr_layout.addWidget(self._risk_badge)
        self.content_layout.addLayout(hdr_layout)

        # 2. Description Label
        self._desc_lbl = QLabel("Action requires explicit operator authorization.")
        desc_font = QFont(JarvisTheme.FONT_FAMILY, 8, QFont.Weight.Medium)
        desc_font.setFamilies(JarvisTheme.FONT_FAMILIES)
        self._desc_lbl.setFont(desc_font)
        self._desc_lbl.setWordWrap(True)
        self._desc_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_PRIMARY}; background: transparent; border: none;")
        self.content_layout.addWidget(self._desc_lbl)

        # 3. Details Container (Operation, Target, Token ID)
        details_frame = QFrame()
        details_frame.setStyleSheet("""
            QFrame {
                background-color: #071324;
                border: 1px solid #0d274c;
                border-radius: 6px;
            }
        """)
        details_layout = QVBoxLayout(details_frame)
        details_layout.setContentsMargins(10, 8, 10, 8)
        details_layout.setSpacing(4)

        # Row: Operation & Target
        op_row = QHBoxLayout()
        op_row.setSpacing(6)

        op_title = QLabel("OPERATION:")
        op_title.setFont(QFont(JarvisTheme.FONT_FAMILY, 7, QFont.Weight.Bold))
        op_title.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
        op_row.addWidget(op_title)

        self._op_lbl = QLabel("SYSTEM_CONTROL")
        self._op_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 8, QFont.Weight.Bold))
        self._op_lbl.setStyleSheet(f"color: {JarvisTheme.CYAN_PRIMARY}; background: transparent; border: none;")
        op_row.addWidget(self._op_lbl)

        op_row.addSpacing(12)

        target_title = QLabel("TARGET:")
        target_title.setFont(QFont(JarvisTheme.FONT_FAMILY, 7, QFont.Weight.Bold))
        target_title.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
        op_row.addWidget(target_title)

        self._target_lbl = QLabel("localhost")
        self._target_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 8))
        self._target_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_PRIMARY}; background: transparent; border: none;")
        op_row.addWidget(self._target_lbl)

        op_row.addStretch(1)
        details_layout.addLayout(op_row)

        # Row: Token ID
        token_row = QHBoxLayout()
        token_row.setSpacing(6)

        token_title = QLabel("TOKEN ID:")
        token_title.setFont(QFont(JarvisTheme.FONT_FAMILY, 7, QFont.Weight.Bold))
        token_title.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
        token_row.addWidget(token_title)

        self._token_lbl = QLabel("---")
        self._token_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 7))
        self._token_lbl.setStyleSheet("color: #94a3b8; font-family: monospace; background: transparent; border: none;")
        token_row.addWidget(self._token_lbl)

        token_row.addStretch(1)
        details_layout.addLayout(token_row)

        self.content_layout.addWidget(details_frame)

        # 4. Action Buttons Row (CONFIRM vs CANCEL)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self.cancel_btn = QPushButton("✕  CANCEL / ABORT")
        self.cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cancel_btn.setFixedHeight(34)
        btn_font = QFont(JarvisTheme.FONT_FAMILY, 8, QFont.Weight.Bold)
        btn_font.setFamilies(JarvisTheme.FONT_FAMILIES)
        btn_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.8)
        self.cancel_btn.setFont(btn_font)
        self.cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #0b1f3b;
                color: {JarvisTheme.TEXT_PRIMARY};
                border: 1px solid #143d70;
                border-radius: 6px;
                padding: 0 16px;
            }}
            QPushButton:hover {{
                background-color: #11305c;
                border: 1px solid {JarvisTheme.CYAN_PRIMARY};
                color: {JarvisTheme.CYAN_PRIMARY};
            }}
            QPushButton:disabled {{
                background-color: #061224;
                color: #475569;
                border: 1px solid #0f2444;
            }}
        """)
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)
        btn_row.addWidget(self.cancel_btn, 1)

        self.confirm_btn = QPushButton("⚡  CONFIRM ACTION")
        self.confirm_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.confirm_btn.setFixedHeight(34)
        self.confirm_btn.setFont(btn_font)
        self.confirm_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #2b1704;
                color: {JarvisTheme.STATE_CONFIRMATION};
                border: 1.5px solid {JarvisTheme.STATE_CONFIRMATION};
                border-radius: 6px;
                padding: 0 16px;
            }}
            QPushButton:hover {{
                background-color: #452405;
                color: #fbbf24;
                border: 1.5px solid #fbbf24;
            }}
            QPushButton:disabled {{
                background-color: #1a0e02;
                color: #78350f;
                border: 1px solid #451a03;
            }}
        """)
        self.confirm_btn.clicked.connect(self._on_confirm_clicked)
        btn_row.addWidget(self.confirm_btn, 1)

        self.content_layout.addLayout(btn_row)

    @property
    def confirmation_id(self) -> str:
        """Return currently loaded confirmation ID."""
        return self._confirmation_id

    def load_confirmation(self, conf: PendingConfirmation) -> None:
        """Populate widget with PendingConfirmation details and reset button states."""
        self._confirmation_id = conf.confirmation_id
        self._resolved = False

        self._desc_lbl.setText(conf.description or f"Action '{conf.operation}' requires explicit operator confirmation.")
        self._op_lbl.setText(conf.operation.upper())
        self._target_lbl.setText(conf.target or "Local System")
        self._token_lbl.setText(conf.confirmation_id)

        risk = (conf.risk_level or "HIGH").upper()
        self._risk_badge.setText(f"RISK: {risk}")
        if risk in ("CRITICAL", "HIGH"):
            self._risk_badge.setStyleSheet("""
                QLabel {
                    color: #ef4444;
                    background-color: #2b0b0b;
                    border: 1px solid #7f1d1d;
                    border-radius: 4px;
                    padding: 2px 8px;
                }
            """)
        else:
            self._risk_badge.setStyleSheet("""
                QLabel {
                    color: #f59e0b;
                    background-color: #261704;
                    border: 1px solid #78350f;
                    border-radius: 4px;
                    padding: 2px 8px;
                }
            """)

        # Re-enable buttons for new confirmation
        self.confirm_btn.setEnabled(True)
        self.cancel_btn.setEnabled(True)

    def _on_confirm_clicked(self) -> None:
        """Handle confirmation approval."""
        if self._resolved or not self._confirmation_id:
            return
        self._resolved = True
        self.confirm_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        self.confirmed.emit(self._confirmation_id)

    def _on_cancel_clicked(self) -> None:
        """Handle confirmation rejection."""
        if self._resolved or not self._confirmation_id:
            return
        self._resolved = True
        self.confirm_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        self.cancelled.emit(self._confirmation_id)
