"""Settings View for J.A.R.V.I.S Multi-View Command Center.

Exposes runtime AI Provider switching (Gemini vs Ollama) using the canonical
ProviderRouter, voice/TTS settings, and read-only subsystem policy statuses.
"""

from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from app.ui.components.glass_panel import GlassPanel
from app.ui.styles import JarvisTheme, ensure_fonts_loaded


class ProviderSelectionCard(QFrame):
    """Interactive AI provider selector tile for Gemini or Ollama."""

    selected = Signal(str)

    def __init__(
        self,
        provider_id: str,
        display_name: str,
        mode_desc: str,
        icon: str = "⚡",
        parent: Optional[QWidget] = None,
    ) -> None:
        """Initialize ProviderSelectionCard."""
        super().__init__(parent)
        self.provider_id = provider_id.lower()
        self._is_active = False

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(100)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)

        # Header row: Icon + Title + Active check
        header_row = QHBoxLayout()
        header_row.setSpacing(8)

        ic_lbl = QLabel(icon)
        ic_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 14))
        ic_lbl.setStyleSheet(f"color: {JarvisTheme.CYAN_PRIMARY}; background: transparent; border: none;")
        header_row.addWidget(ic_lbl)

        self._title_lbl = QLabel(display_name)
        title_font = QFont(JarvisTheme.FONT_FAMILY, 11, QFont.Weight.Bold)
        title_font.setFamilies(JarvisTheme.FONT_FAMILIES)
        self._title_lbl.setFont(title_font)
        self._title_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_PRIMARY}; background: transparent; border: none;")
        header_row.addWidget(self._title_lbl)

        header_row.addStretch(1)

        self._check_badge = QLabel("ACTIVE")
        badge_font = QFont(JarvisTheme.FONT_FAMILY, 7, QFont.Weight.Bold)
        badge_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.8)
        self._check_badge.setFont(badge_font)
        self._check_badge.setVisible(False)
        header_row.addWidget(self._check_badge)

        layout.addLayout(header_row)

        desc_lbl = QLabel(mode_desc)
        desc_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 8))
        desc_lbl.setWordWrap(True)
        desc_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
        layout.addWidget(desc_lbl)

        self._update_style()

    def set_active(self, active: bool) -> None:
        """Update active selection state and redraw styling."""
        self._is_active = active
        self._check_badge.setVisible(active)
        self._update_style()

    def _update_style(self) -> None:
        """Update tile stylesheet based on active state."""
        if self._is_active:
            self.setStyleSheet(f"""
                QFrame {{
                    background-color: #0c2b4d;
                    border: 1.5px solid {JarvisTheme.CYAN_PRIMARY};
                    border-radius: 10px;
                }}
            """)
            self._check_badge.setStyleSheet(f"""
                color: {JarvisTheme.CYAN_PRIMARY};
                background-color: #061933;
                border: 1px solid {JarvisTheme.CYAN_PRIMARY};
                border-radius: 6px;
                padding: 2px 8px;
            """)
        else:
            self.setStyleSheet(f"""
                QFrame {{
                    background-color: #06152b;
                    border: 1px solid #0d274c;
                    border-radius: 10px;
                }}
                QFrame:hover {{
                    border: 1px solid #1a4980;
                    background-color: #081a36;
                }}
            """)

    def trigger_selection(self) -> None:
        """Trigger selection of this provider programmatically or on click."""
        self.selected.emit(self.provider_id)

    def mousePressEvent(self, event: object) -> None:
        """Handle click to select provider."""
        self.trigger_selection()
        if event is not None:
            super().mousePressEvent(event)


class SettingsView(QWidget):
    """Full-screen Configuration & Provider Switching view."""

    provider_changed = Signal(str)
    volume_changed = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize SettingsView."""
        super().__init__(parent)
        ensure_fonts_loaded()

        self._active_provider = "gemini"
        self._provider_cards: dict[str, ProviderSelectionCard] = {}

        self.setStyleSheet("background: transparent;")
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 16, 20, 16)
        main_layout.setSpacing(14)

        # 1. Header Card
        header_card = GlassPanel(self, border_radius=12)
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(16, 12, 16, 12)
        header_layout.setSpacing(16)

        title_col = QVBoxLayout()
        title_col.setSpacing(4)
        title_lbl = QLabel("SYSTEM & AI CONFIGURATION")
        title_font = QFont(JarvisTheme.FONT_FAMILY, 14, QFont.Weight.Bold)
        title_font.setFamilies(JarvisTheme.FONT_FAMILIES)
        title_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.2)
        title_lbl.setFont(title_font)
        title_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_PRIMARY}; background: transparent; border: none;")
        title_col.addWidget(title_lbl)

        sub_lbl = QLabel("Cognitive AI Provider Routing, Model Execution Mode, and TTS Runtime Parameters.")
        sub_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 8))
        sub_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
        title_col.addWidget(sub_lbl)
        header_layout.addLayout(title_col, 1)

        # Safety badge
        safety_badge = QLabel("⚡ CANONICAL PROVIDER ROUTER")
        s_font = QFont(JarvisTheme.FONT_FAMILY, 8, QFont.Weight.Bold)
        s_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.8)
        safety_badge.setFont(s_font)
        safety_badge.setStyleSheet(f"""
            QLabel {{
                color: {JarvisTheme.CYAN_PRIMARY};
                background-color: #061933;
                border: 1px solid #0f3964;
                border-radius: 8px;
                padding: 6px 14px;
            }}
        """)
        header_layout.addWidget(safety_badge)

        header_card.content_layout.addLayout(header_layout)
        main_layout.addWidget(header_card, 0)

        # 2. Main Content Split: Left Provider Routing, Right Audio & Policy Status
        content_split = QHBoxLayout()
        content_split.setSpacing(16)

        # Left Column: AI Provider Selection & Status
        provider_panel = GlassPanel(self, border_radius=12)
        provider_panel.add_card_header("AI COGNITIVE PROVIDER ROUTING", icon_text="🧠")

        prov_layout = QVBoxLayout()
        prov_layout.setContentsMargins(4, 8, 4, 8)
        prov_layout.setSpacing(12)

        # Selection Cards: Gemini vs Ollama
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(12)

        # Gemini Card
        gemini_card = ProviderSelectionCard(
            provider_id="gemini",
            display_name="Google Gemini",
            mode_desc="Cloud LLM • High-speed reasoning • Multi-modal support • Google Cloud API",
            icon="🌐",
            parent=self,
        )
        gemini_card.selected.connect(self._on_provider_selected)
        cards_layout.addWidget(gemini_card)
        self._provider_cards["gemini"] = gemini_card

        # Ollama Card
        ollama_card = ProviderSelectionCard(
            provider_id="ollama",
            display_name="Ollama Local",
            mode_desc="Local LLM • Zero cloud latency • Complete offline privacy • Local Daemon",
            icon="🖥",
            parent=self,
        )
        ollama_card.selected.connect(self._on_provider_selected)
        cards_layout.addWidget(ollama_card)
        self._provider_cards["ollama"] = ollama_card

        prov_layout.addLayout(cards_layout)

        # Model Telemetry Detail Card
        detail_frame = QFrame()
        detail_frame.setStyleSheet("""
            QFrame {
                background-color: #06152b;
                border: 1px solid #0d274c;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        detail_layout = QGridLayout(detail_frame)
        detail_layout.setSpacing(10)

        # Active Model
        self._model_val = QLabel("gemini-2.5-flash")
        self._mode_val = QLabel("CLOUD")
        self._latency_val = QLabel("N/A")
        self._health_val = QLabel("● HEALTHY")

        for row, (lbl_name, val_widget) in enumerate([
            ("Active Model", self._model_val),
            ("Execution Target", self._mode_val),
            ("Reasoning Latency", self._latency_val),
            ("Provider Health", self._health_val),
        ]):
            lbl = QLabel(lbl_name)
            lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 8, QFont.Weight.Medium))
            lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
            detail_layout.addWidget(lbl, row, 0)

            val_widget.setFont(QFont(JarvisTheme.FONT_FAMILY, 8, QFont.Weight.Bold))
            val_widget.setStyleSheet(f"color: {JarvisTheme.TEXT_PRIMARY}; background: transparent; border: none;")
            detail_layout.addWidget(val_widget, row, 1)

        prov_layout.addWidget(detail_frame)
        prov_layout.addStretch(1)

        provider_panel.content_layout.addLayout(prov_layout)
        content_split.addWidget(provider_panel, 3)

        # Right Column: Voice/TTS Settings & Locked Feature Status
        side_panel = GlassPanel(self, border_radius=12)
        side_panel.setFixedWidth(380)
        side_panel.add_card_header("AUDIO & RUNTIME POLICIES", icon_text="🎙")

        side_layout = QVBoxLayout()
        side_layout.setContentsMargins(4, 8, 4, 8)
        side_layout.setSpacing(14)

        # Voice / Speech Section
        voice_box = QFrame()
        voice_box.setStyleSheet("""
            QFrame {
                background-color: #06152b;
                border: 1px solid #0d274c;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        vb_layout = QVBoxLayout(voice_box)
        vb_layout.setSpacing(8)

        v_title = QLabel("Text-To-Speech (TTS) Parameters")
        v_title.setFont(QFont(JarvisTheme.FONT_FAMILY, 8, QFont.Weight.Bold))
        v_title.setStyleSheet(f"color: {JarvisTheme.CYAN_PRIMARY}; background: transparent; border: none;")
        vb_layout.addWidget(v_title)

        # TTS Volume Slider
        vol_row = QHBoxLayout()
        vol_row.setSpacing(8)
        vol_lbl = QLabel("Volume")
        vol_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 8))
        vol_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
        vol_row.addWidget(vol_lbl)

        self._vol_slider = QSlider(Qt.Orientation.Horizontal)
        self._vol_slider.setRange(0, 100)
        self._vol_slider.setValue(80)
        self._vol_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 4px;
                background: #0b2246;
                border-radius: 2px;
            }}
            QSlider::sub-page:horizontal {{
                background: {JarvisTheme.CYAN_PRIMARY};
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {JarvisTheme.CYAN_PRIMARY};
                border: 1px solid #ffffff;
                width: 12px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 6px;
            }}
        """)
        self._vol_slider.valueChanged.connect(self._on_volume_slider_changed)
        vol_row.addWidget(self._vol_slider, 1)

        self._vol_num_lbl = QLabel("80%")
        self._vol_num_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 8, QFont.Weight.Bold))
        self._vol_num_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_PRIMARY}; background: transparent; border: none;")
        vol_row.addWidget(self._vol_num_lbl)
        vb_layout.addLayout(vol_row)

        side_layout.addWidget(voice_box)

        # Subsystems & Feature Flags (Unavailable in Phase 23.5 per contract)
        features_box = QFrame()
        features_box.setStyleSheet("""
            QFrame {
                background-color: #06152b;
                border: 1px solid #0d274c;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        fb_layout = QVBoxLayout(features_box)
        fb_layout.setSpacing(8)

        f_title = QLabel("Architectural Feature Boundaries")
        f_title.setFont(QFont(JarvisTheme.FONT_FAMILY, 8, QFont.Weight.Bold))
        f_title.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
        fb_layout.addWidget(f_title)

        for feature_name, status_str in [
            ("Vision Stream Analysis", "UNAVAILABLE (Phase 23.5 Scope)"),
            ("Microphone Streaming Ingest", "UNAVAILABLE (Phase 23.5 Scope)"),
            ("Cinematic Mode", "UNAVAILABLE (Phase 23.5 Scope)"),
            ("Conversation Persistence", "UNAVAILABLE (Phase 23.5 Scope)"),
        ]:
            row = QHBoxLayout()
            row.setSpacing(6)
            name_lbl = QLabel(feature_name)
            name_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 7))
            name_lbl.setStyleSheet(f"color: {JarvisTheme.TEXT_MUTED}; background: transparent; border: none;")
            row.addWidget(name_lbl)
            row.addStretch(1)

            st_lbl = QLabel(status_str)
            st_lbl.setFont(QFont(JarvisTheme.FONT_FAMILY, 6, QFont.Weight.Bold))
            st_lbl.setStyleSheet("color: #94a3b8; background: #0b1c36; border-radius: 4px; padding: 2px 6px; border: none;")
            row.addWidget(st_lbl)
            fb_layout.addLayout(row)

        side_layout.addWidget(features_box)
        side_layout.addStretch(1)

        side_panel.content_layout.addLayout(side_layout)
        content_split.addWidget(side_panel, 0)

        main_layout.addLayout(content_split, 1)

        # Set initial visual state
        self.set_active_provider("gemini")

    def _on_provider_selected(self, provider_id: str) -> None:
        """Handle user selection of provider tile and emit signal."""
        self.set_active_provider(provider_id)
        self.provider_changed.emit(provider_id)

    def _on_volume_slider_changed(self, value: int) -> None:
        """Handle volume slider change."""
        self._vol_num_lbl.setText(f"{value}%")
        self.volume_changed.emit(value)

    def set_active_provider(self, provider_name: str) -> None:
        """Visually highlight selected provider card."""
        clean = (provider_name or "gemini").lower()
        self._active_provider = clean
        for pid, card in self._provider_cards.items():
            card.set_active(pid == clean)

        if clean == "ollama":
            self._mode_val.setText("LOCAL")
            self._model_val.setText("qwen2.5:3b")
        else:
            self._mode_val.setText("CLOUD")
            self._model_val.setText("gemini-2.5-flash")

    def update_ai_telemetry(self, data: dict[str, Any]) -> None:
        """Update telemetry labels from live provider status.

        Args:
            data: Structured telemetry dictionary with provider, model, mode, latency, health.
        """
        provider = str(data.get("active_provider") or data.get("provider") or "").lower()
        if provider:
            self.set_active_provider(provider)

        model = str(data.get("active_model") or data.get("model") or "N/A")
        self._model_val.setText(model)

        mode = str(data.get("mode") or ("LOCAL" if provider == "ollama" else "CLOUD"))
        self._mode_val.setText(mode.upper())

        latency = data.get("reasoning_latency_ms")
        if latency is not None and isinstance(latency, (int, float)):
            self._latency_val.setText(f"{latency:.0f} ms")
        elif "latency" in data and data["latency"]:
            self._latency_val.setText(str(data["latency"]))
        else:
            self._latency_val.setText("N/A")

        health = str(data.get("health") or data.get("status") or "HEALTHY").upper()
        if "HEALTHY" in health or "ONLINE" in health:
            self._health_val.setText("● HEALTHY")
            self._health_val.setStyleSheet("color: #22c55e; background: transparent; border: none;")
        else:
            self._health_val.setText(f"● {health}")
            self._health_val.setStyleSheet("color: #eab308; background: transparent; border: none;")
