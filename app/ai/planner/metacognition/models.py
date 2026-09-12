"""Data models and specifications for Phase 20 Metacognition & Tool Synthesis.

Provides type-safe status enumerations, verification result structures,
and synthesized tool records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Callable, Dict, Optional


class ToolSynthesisStatus(str, Enum):
    """Lifecycle status of a dynamic tool synthesis attempt."""

    PENDING = "PENDING"
    AST_VALIDATION_FAILED = "AST_VALIDATION_FAILED"
    SANDBOX_TEST_FAILED = "SANDBOX_TEST_FAILED"
    VERIFIED_AND_PROMOTED = "VERIFIED_AND_PROMOTED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class SandboxVerificationResult:
    """Outcome of executing generated tool code and tests inside an isolated sandbox."""

    success: bool
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    execution_time_ms: float = 0.0
    memory_peak_mb: float = 0.0
    error: Optional[str] = None


@dataclass
class SynthesizedToolResult:
    """Detailed record and artifacts of a dynamic tool synthesis operation."""

    action_name: str
    code: str
    test_code: str
    status: ToolSynthesisStatus
    verification: Optional[SandboxVerificationResult] = None
    created_at: float = field(default_factory=time.time)
    iteration_count: int = 1
    error: Optional[str] = None
    callable_tool: Optional[Callable[..., Any]] = None


@dataclass(frozen=True)
class KnowledgeTriplet:
    """Semantic relation representing an entity-predicate-entity statement with confidence and expiry."""

    subject: str
    predicate: str
    object_: str
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None

    def is_expired(self, current_time: Optional[float] = None) -> bool:
        """Check whether the triplet has passed its expiration deadline."""
        if self.expires_at is None:
            return False
        now = current_time if current_time is not None else time.time()
        return now >= self.expires_at


@dataclass
class CausalDiagnosis:
    """Metacognitive diagnostic finding from inspecting a failure or suboptimal execution."""

    trajectory_id: str
    failed_task_id: str
    root_cause_type: str
    description: str
    invariant_constraint: str
    recommended_action: str
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)


@dataclass
class DistilledSkillMetadata:
    """Metadata describing a compiled macro-skill derived from repeated swarm trajectories."""

    skill_name: str
    source_subplan_signature: str
    compiled_code: str
    input_parameters: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    historical_avg_latency_ms: float = 0.0
    compiled_latency_ms: float = 0.0
    speedup_multiplier: float = 1.0
    verified: bool = False
    quarantined: bool = False
    version: str = "1.0.0"
    created_at: float = field(default_factory=time.time)


