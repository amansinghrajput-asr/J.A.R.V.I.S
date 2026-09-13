"""Assistant State Model and Presentation Events for J.A.R.V.I.S. Phase 23.1.

Defines the unified state machine enumeration, deeply immutable snapshots,
pending confirmation tokens, and UI-agnostic presentation events providing
a stable boundary for headless clients and the future PySide6 HUD.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Optional


class AssistantState(str, Enum):
    """Unified operational states of the J.A.R.V.I.S assistant.

    Inherits from str to provide clean JSON serialization and string comparisons.
    """

    INITIALIZING = "INITIALIZING"
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    TRANSCRIBING = "TRANSCRIBING"
    THINKING = "THINKING"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    SPEAKING = "SPEAKING"
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
    ERROR = "ERROR"


@dataclass(frozen=True)
class PendingConfirmation:
    """Deeply immutable description of a pending operator confirmation request.

    Exposes only presentation-safe primitive fields. No raw internal skill or policy
    objects are leaked.
    """

    confirmation_id: str
    operation: str
    target: Optional[str] = None
    risk_level: str = "HIGH"
    description: str = ""
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """Validate primitive types."""
        if not isinstance(self.confirmation_id, str) or not self.confirmation_id.strip():
            raise ValueError("confirmation_id must be a non-empty string.")
        if not isinstance(self.operation, str) or not self.operation.strip():
            raise ValueError("operation must be a non-empty string.")


@dataclass(frozen=True)
class AssistantSnapshot:
    """Deeply immutable point-in-time snapshot of assistant state.

    Thread-safe to pass directly across thread boundaries between backend services
    and presentation/GUI layers.
    """

    state: AssistantState
    status_message: str = ""
    current_command: Optional[str] = None
    last_response: Optional[str] = None
    last_error: Optional[str] = None
    execution_id: Optional[str] = None
    task_id: Optional[str] = None
    plan_id: Optional[str] = None
    task_progress: float = 0.0
    pending_confirmation: Optional[PendingConfirmation] = None
    transcript: tuple[str, ...] = field(default_factory=tuple)
    mic_amplitude: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """Ensure deep immutability and valid bounds."""
        if not isinstance(self.state, AssistantState):
            raise TypeError(f"state must be an AssistantState instance, got {type(self.state)}")

        # Enforce tuple for transcript
        if not isinstance(self.transcript, tuple):
            object.__setattr__(self, "transcript", tuple(self.transcript))

        # Clamp task progress to [0.0, 1.0]
        progress = max(0.0, min(1.0, float(self.task_progress)))
        object.__setattr__(self, "task_progress", progress)

        # Clamp mic_amplitude to [0.0, 1.0]
        amp = max(0.0, min(1.0, float(self.mic_amplitude)))
        object.__setattr__(self, "mic_amplitude", amp)

        # Pending confirmation type check
        if self.pending_confirmation is not None and not isinstance(
            self.pending_confirmation, PendingConfirmation
        ):
            raise TypeError("pending_confirmation must be a PendingConfirmation instance or None.")


@dataclass(frozen=True)
class PresentationEvent:
    """UI-agnostic presentation event carrying state transitions or telemetry.

    Never exposes internal AI providers, LLM prompts, raw skill implementations,
    or internal Planner components.
    """

    event_type: str
    snapshot: AssistantSnapshot
    payload: tuple[tuple[str, Any], ...] = field(default_factory=tuple)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """Ensure payload is an immutable tuple of key-value pairs."""
        if not isinstance(self.payload, tuple):
            if isinstance(self.payload, dict):
                object.__setattr__(self, "payload", tuple(sorted(self.payload.items())))
            else:
                object.__setattr__(self, "payload", tuple(self.payload))

    def get_payload_dict(self) -> dict[str, Any]:
        """Convert payload tuple back to dict for convenient read-only inspection."""
        return dict(self.payload)
