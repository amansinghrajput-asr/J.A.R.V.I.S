"""Intent data structures for the J.A.R.V.I.S Command Router subsystem.

Defines the standardized Intent dataclass representing parsed user intents,
confidence scores, extracted parameters, tracking IDs, contextual telemetry,
and source channels.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Union


@dataclass
class Intent:
    """Represents a parsed user intent and its contextual execution payload.

    Attributes:
        raw_command: The unmodified original command text received from the user.
        normalized_command: The sanitized, lowercased, and preprocessed command text.
        intent_name: Categorical identifier for the recognized intent (e.g., 'open_app', 'unknown').
        confidence: Numeric confidence score between 0.0 and 1.0.
        parameters: Extracted key-value argument mapping for the command.
        timestamp: Unix epoch timestamp marking intent creation.
        id: Unique UUID tracking identifier across logs, events, AI, memory, and UI.
        source: Input channel source ('voice', 'text', 'cli', 'api', 'automation'). Default: 'text'.
        context: Contextual dictionary holding conversation_id, memory references, screen state, etc.
    """

    raw_command: str
    normalized_command: str = ""
    intent_name: str = "unknown"
    confidence: float = 1.0
    parameters: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = "text"
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and sanitize intent fields upon instantiation."""
        # 1. UUID Tracking Identifier
        if not self.id:
            self.id = str(uuid.uuid4())
        elif isinstance(self.id, uuid.UUID):
            self.id = str(self.id)
        else:
            self.id = str(self.id).strip()

        # 2. Raw & Normalized Command Strings
        if not isinstance(self.raw_command, str):
            self.raw_command = str(self.raw_command) if self.raw_command is not None else ""

        if not isinstance(self.normalized_command, str) or not self.normalized_command.strip():
            self.normalized_command = self.raw_command.strip().lower()
        else:
            self.normalized_command = self.normalized_command.strip().lower()

        # 3. Intent Name
        if not isinstance(self.intent_name, str) or not self.intent_name.strip():
            self.intent_name = "unknown"
        else:
            self.intent_name = self.intent_name.strip()

        # 4. Confidence bounds [0.0, 1.0]
        try:
            conf = float(self.confidence)
            self.confidence = max(0.0, min(1.0, conf))
        except (ValueError, TypeError):
            self.confidence = 0.0

        # 5. Parameters Dictionary
        if self.parameters is None or not isinstance(self.parameters, dict):
            self.parameters = {}
        else:
            self.parameters = dict(self.parameters)

        # 6. Timestamp
        if not isinstance(self.timestamp, (int, float)):
            self.timestamp = time.time()

        # 7. Source Channel
        if not isinstance(self.source, str) or not self.source.strip():
            self.source = "text"
        else:
            self.source = self.source.strip().lower()

        # 8. Context Dictionary
        if self.context is None or not isinstance(self.context, dict):
            self.context = {}
        else:
            self.context = dict(self.context)

    def to_dict(self) -> dict[str, Any]:
        """Convert the Intent instance to a JSON-serializable dictionary.

        Returns:
            Dictionary containing all intent attributes.
        """
        return {
            "id": self.id,
            "raw_command": self.raw_command,
            "normalized_command": self.normalized_command,
            "intent_name": self.intent_name,
            "confidence": self.confidence,
            "parameters": dict(self.parameters),
            "timestamp": self.timestamp,
            "source": self.source,
            "context": dict(self.context),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Intent:
        """Construct an Intent instance from a dictionary payload.

        Args:
            data: Dictionary containing intent attributes.

        Returns:
            Validated Intent instance.
        """
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict for Intent.from_dict, got {type(data).__name__}")

        return cls(
            id=str(data.get("id", uuid.uuid4())),
            raw_command=str(data.get("raw_command", "")),
            normalized_command=str(data.get("normalized_command", "")),
            intent_name=str(data.get("intent_name", "unknown")),
            confidence=float(data.get("confidence", 1.0)),
            parameters=dict(data.get("parameters", {})),
            timestamp=float(data.get("timestamp", time.time())),
            source=str(data.get("source", "text")),
            context=dict(data.get("context", {})),
        )

    def __repr__(self) -> str:
        """Developer-friendly string representation."""
        return (
            f"<Intent(id={self.id[:8]}..., intent_name={self.intent_name!r}, "
            f"source={self.source!r}, confidence={self.confidence:.2f}, "
            f"normalized={self.normalized_command!r})>"
        )
