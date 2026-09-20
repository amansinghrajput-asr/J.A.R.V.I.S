"""Data models for the J.A.R.V.I.S Planner Subsystem.

Defines TaskStatus enumeration, Task model, Plan container,
and ExecutionResult structures.

Phase 27.24 additions:
    VisualDispatchState, SemanticActionKey, normalize_semantic_target,
    make_semantic_action_key — cross-wave visual action idempotency.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Final, FrozenSet, List, Optional, Set

if TYPE_CHECKING:
    from app.vision.models import VisualGoalSpec


GROUNDED_PARAM_KEYS: Final[tuple[str, ...]] = (
    "action_target",
    "target_obj",
    "target_point",
    "point",
    "bounds",
    "coordinates",
    "window_handle",
    "hwnd",
    "grounded_at",
    "observation_id",
    "observation",
    "scene",
    "current_scene",
    "situation",
    "current_situation",
    "current_window_info",
    "confirmation_token",
    "confirmation_id",
    "screenshot",
    "screenshots",
    "ocr",
    "raw_ocr",
    "ocr_result",
    "password",
    "passwords",
    "input_text",
)


# ---------------------------------------------------------------------------
# Phase 27.24 — Visual Action Transaction Safety & Idempotency
# ---------------------------------------------------------------------------

# Process-local HMAC key for input_text discriminators.
# Generated once per process start. Never logged, exported, serialized, or
# persisted. Ephemeral: a process restart resets all ExecutionMemory anyway.
_INPUT_DISCRIMINATOR_KEY: bytes = secrets.token_bytes(32)

# Visual actions that carry input_text and therefore need input discriminators.
_TYPE_ACTIONS: Final[FrozenSet[str]] = frozenset({
    "visual_type",
    "visual_clear_and_type",
})


def _compute_input_discriminator(input_text: str) -> str:
    """Return a process-local HMAC-SHA256 token for the given input value.

    Properties:
    - Same input_text -> same token within one process lifetime.
    - Different input_text -> different token (collision prob ~2^-64).
    - Token reveals nothing about the input_text value (one-way).
    - Not usable after process restart (key is ephemeral).
    - Token format: 'disc:<16 hex chars>'.

    INVARIANT: Never call with raw passwords/secrets directly in log output.
    """
    digest = hmac.new(
        _INPUT_DISCRIMINATOR_KEY,
        input_text.encode("utf-8", errors="replace"),
        hashlib.sha256,
    ).hexdigest()[:16]
    return f"disc:{digest}"


class VisualDispatchState(str, Enum):
    """Observed dispatch lifecycle state for a semantic visual action key.

    Derived exclusively from TaskExecutionRecord.visual_status and
    TaskExecutionRecord.preflight_status fields. Never inferred from
    task.status alone.

    CRITICAL INVARIANT:
    - COMPLETED status alone does NOT imply VERIFIED.
    - visual_status must be exactly 'SUCCESS' to imply VERIFIED.
    - Unknown visual_status values must produce UNKNOWN -> fail closed.
    """

    NOT_DISPATCHED        = "NOT_DISPATCHED"         # No prior record for this key
    VERIFIED              = "VERIFIED"               # Physical input sent; post-condition confirmed
    DISPATCHED_UNVERIFIED = "DISPATCHED_UNVERIFIED"  # Physical input sent; no verification configured
    UNCERTAIN             = "UNCERTAIN"              # Physical input sent; verification inconclusive
    FAILED_BEFORE         = "FAILED_BEFORE"          # Failure before physical input was sent
    FAILED_AFTER          = "FAILED_AFTER"           # Failure after physical input was sent
    UNKNOWN               = "UNKNOWN"                # Record exists but state undeterminable safely


@dataclass(frozen=True)
class SemanticActionKey:
    """Canonical semantic identity for a visual action across execution waves.

    INVARIANTS — this dataclass MUST NEVER contain:
      coordinates, HWNDs, window bounds, screenshots, OCR content,
      confirmation tokens, passwords, raw input_text, or physical observations.

    Fields:
      action:             normalized lowercase visual action name.
      target_normalized:  conservative canonical target string.
      input_discriminator: Optional process-local HMAC token for type actions.
                           None for all non-type actions and empty-input calls.
                           NEVER the plaintext input value.
    """

    action: str
    target_normalized: str
    input_discriminator: Optional[str] = None


# Statuses proven to mean physical input was never dispatched (traced from
# VisualActionAdapter.execute_target() and InteractionSkills.execute()).
_FAILED_BEFORE_STATUSES: Final[FrozenSet[str]] = frozenset({
    "GROUNDING_FAILED",
    "PRECONDITION_FAILED",
    "PREFLIGHT_STALE_COORDINATES",
    "PREFLIGHT_WINDOW_MISMATCH",
    "PREFLIGHT_MODAL_CHANGED",
    "SECURITY_BLOCKED",
    "CONFIRMATION_REQUIRED",
    "SENSITIVE_PROTECTED",
    "DISABLED_CONTROL",
    "TARGET_NOT_FOUND",
})

# Statuses proven to mean physical input was dispatched but failed afterward.
_FAILED_AFTER_STATUSES: Final[FrozenSet[str]] = frozenset({
    "INPUT_DISPATCH_ERROR",
    "VERIFICATION_FAILED",
})

# The ONE status that proves dispatch succeeded AND post-condition was confirmed.
_VERIFIED_STATUS: Final[str] = "SUCCESS"


def _derive_dispatch_state(record: "TaskExecutionRecord") -> VisualDispatchState:  # noqa: F821
    """Derive VisualDispatchState from a TaskExecutionRecord.

    CRITICAL INVARIANTS:
    - task.status == COMPLETED alone does NOT imply VERIFIED.
    - visual_status must be exactly 'SUCCESS' to imply VERIFIED.
    - Any unexpected visual_status value -> UNKNOWN -> caller must fail closed.
    - None visual_status on COMPLETED task -> DISPATCHED_UNVERIFIED.
    - preflight_status field is NOT used for state derivation; visual_status
      is the authoritative signal (proven from VisualActionAdapter source).
    """
    vs = (record.visual_status or "").strip().upper()

    # 1. Proven VERIFIED: COMPLETED + visual_status == 'SUCCESS'
    if record.status == TaskStatus.COMPLETED and vs == _VERIFIED_STATUS:
        return VisualDispatchState.VERIFIED

    # 2. COMPLETED + no visual_status -> dispatched but not verified
    if record.status == TaskStatus.COMPLETED and vs == "":
        return VisualDispatchState.DISPATCHED_UNVERIFIED

    # 3. COMPLETED + unexpected visual_status -> fail closed
    if record.status == TaskStatus.COMPLETED:
        return VisualDispatchState.UNKNOWN

    # 4. UNCERTAIN
    if vs == "VERIFICATION_UNCERTAIN":
        return VisualDispatchState.UNCERTAIN

    # 5. FAILED_AFTER: physics happened but failed afterward
    if vs in _FAILED_AFTER_STATUSES:
        return VisualDispatchState.FAILED_AFTER

    # 6. FAILED_BEFORE: physics never happened (proven from source)
    if vs in _FAILED_BEFORE_STATUSES:
        return VisualDispatchState.FAILED_BEFORE

    # 7. Any other case -> fail closed
    return VisualDispatchState.UNKNOWN


def normalize_semantic_target(action: str, target: Optional[str]) -> str:
    """Produce a conservative canonical semantic string for an action target.

    Rules (only provably safe transformations applied):
    1. Non-visual actions -> return '' (no fence identity).
    2. None or empty target -> return '' (empty-target; fence policy in executor).
    3. Strip, lowercase, collapse internal whitespace.
    4. Remove exactly ONE leading article ('a ', 'an ', 'the ') if present.
    5. Strip again.
    6. Truncate at last word boundary to 150 chars max.
    7. Run through sanitize_sensitive_data(redact_coordinates=True).

    WHAT IS NEVER REMOVED:
      button, field, icon, checkbox, input, link, control, box, tab, menu,
      panel, dialog — these preserve meaningful control-type distinctions.
    """
    if not action.startswith("visual_"):
        return ""
    if not target:
        return ""
    norm = str(target).strip().lower()
    # Collapse internal whitespace
    norm = re.sub(r"\s+", " ", norm).strip()
    # Remove exactly one leading article
    for art in ("the ", "an ", "a "):
        if norm.startswith(art):
            norm = norm[len(art):].strip()
            break
    # Truncate at last word boundary
    if len(norm) > 150:
        truncated = norm[:150]
        last_space = truncated.rfind(" ")
        norm = truncated[:last_space] if last_space > 0 else truncated
    # Strip sensitive data / coordinates
    from app.ai.planner.memory import sanitize_sensitive_data
    norm = sanitize_sensitive_data(norm, redact_coordinates=True)
    return norm


def make_semantic_action_key(
    action: str,
    target: Optional[str],
    input_text: Optional[str] = None,
) -> SemanticActionKey:
    """Construct a SemanticActionKey safely. Never stores raw input_text.

    For visual_type / visual_clear_and_type:
      - A non-empty input_text -> HMAC discriminator computed and stored.
      - Empty string or None  -> discriminator=None (fence not applied for
        empty-target ambiguous case; see executor fence policy).
    For all other actions:
      - discriminator is always None.
    """
    norm = normalize_semantic_target(action, target)
    discriminator: Optional[str] = None
    if action in _TYPE_ACTIONS and input_text is not None and input_text != "":
        discriminator = _compute_input_discriminator(input_text)
    return SemanticActionKey(
        action=action,
        target_normalized=norm,
        input_discriminator=discriminator,
    )


# Re-export for use in memory.py / executor.py without circular import
# (models.py has no imports from memory.py at module level; normalize_semantic_target
# imports sanitize_sensitive_data lazily inside its body)

def purge_physical_state(params: Dict[str, Any]) -> None:
    """Purge all physical visual state and sensitive tokens from parameters dictionary.

    Enforces that physical state is strictly JIT and never persisted or propagated
    across task boundaries or execution waves.
    """
    if not isinstance(params, dict):
        return
    for k in GROUNDED_PARAM_KEYS:
        params.pop(k, None)
    if "target" in params:
        raw_t = params["target"]
        if isinstance(raw_t, dict) or hasattr(raw_t, "target_point"):
            params.pop("target", None)


class TaskStatus(str, Enum):
    """Execution lifecycle status for planned tasks."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    CANCELLED = "CANCELLED"


class PlanningStrategy(str, Enum):
    """Planning strategy modes for task decomposition."""

    RULE_BASED = "RULE_BASED"
    LLM = "LLM"
    HYBRID = "HYBRID"
    MULTI_AGENT = "MULTI_AGENT"


@dataclass
class Task:
    """Represents a single atomic planned action.

    Attributes:
        action: Identifier for the action to execute (e.g. 'open_app', 'web_search').
        target: Optional primary target or operand of the action (e.g. 'chrome', 'weather').
        parameters: Optional dictionary of arguments or metadata.
        status: Current task lifecycle status.
        id: Unique identifier for tracking task execution.
        dependencies: Task IDs that must complete before this task executes.
        assigned_agent: Optional identifier of the specialized agent assigned to this task.
        expected_visual_goal: Optional expected visual goal or post-condition.
    """

    action: str
    target: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    dependencies: List[str] = field(default_factory=list)
    assigned_agent: Optional[str] = None
    expected_visual_goal: Optional[Any] = None
    error: Optional[str] = None
    result: Optional[Any] = None
    workflow_context: Optional[WorkflowContext] = None
    # Phase 27.24: transient HMAC discriminator for visual_type actions.
    # Computed from input_text BEFORE purge_physical_state() runs.
    # Never serialized, never forwarded to skills, never in params dict.
    _input_discriminator: Optional[str] = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize Task to a dictionary."""
        d = {
            "id": self.id,
            "action": self.action,
            "target": self.target,
            "parameters": dict(self.parameters),
            "status": self.status.value,
            "dependencies": list(self.dependencies),
        }
        if self.assigned_agent is not None:
            d["assigned_agent"] = self.assigned_agent
        if self.expected_visual_goal is not None:
            d["expected_visual_goal"] = (
                self.expected_visual_goal.to_dict()
                if hasattr(self.expected_visual_goal, "to_dict")
                else self.expected_visual_goal
            )
        if self.workflow_context is not None:
            d["workflow_context"] = self.workflow_context.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Task:
        """Deserialize dictionary to Task."""
        status_val = data.get("status", TaskStatus.PENDING.value)
        try:
            status = TaskStatus(status_val)
        except Exception:
            status = TaskStatus.PENDING

        evg_raw = data.get("expected_visual_goal")
        expected_visual_goal = None
        if isinstance(evg_raw, dict):
            try:
                from app.vision.models import VisualGoalSpec
                expected_visual_goal = VisualGoalSpec.from_dict(evg_raw)
            except Exception:
                expected_visual_goal = None
        elif evg_raw is not None:
            expected_visual_goal = evg_raw

        wf_raw = data.get("workflow_context")
        wf_ctx = None
        if isinstance(wf_raw, dict):
            try:
                wf_ctx = WorkflowContext.from_dict(wf_raw)
            except Exception:
                wf_ctx = None

        return cls(
            id=str(data.get("id", str(uuid.uuid4()))),
            action=str(data.get("action", "")),
            target=data.get("target"),
            parameters=dict(data.get("parameters", {})),
            status=status,
            dependencies=list(data.get("dependencies", [])),
            assigned_agent=data.get("assigned_agent"),
            expected_visual_goal=expected_visual_goal,
            workflow_context=wf_ctx,
        )


@dataclass(frozen=True)
class WorkflowContext:
    """Immutable semantic context maintained across a multi-step workflow.

    CRITICAL INVARIANT:
    Strictly zero physical visual state. HWNDs, coordinates, bounds,
    VisualActionTarget objects, raw UI elements, screenshots, OCR objects,
    confirmation tokens, passwords, and input_text must NEVER be added to
    or serialized by WorkflowContext.
    """

    workflow_id: str
    objective: str
    expected_app: Optional[str] = None
    expected_process: Optional[str] = None
    current_semantic_step: int = 1
    previous_action: Optional[str] = None
    previous_verification_outcome: Optional[str] = None
    verification_summary: Optional[str] = None
    is_valid: bool = True
    created_at: float = field(default_factory=time.monotonic)
    ttl_seconds: float = 60.0

    def is_expired(self, current_time: Optional[float] = None) -> bool:
        """Check if workflow context has exceeded its time-to-live."""
        now = current_time if current_time is not None else time.monotonic()
        return (now - self.created_at) > self.ttl_seconds

    def with_step_outcome(
        self,
        action: Optional[str] = None,
        outcome: Optional[str] = None,
        summary: Optional[str] = None,
        expected_app: Optional[str] = None,
        expected_process: Optional[str] = None,
        current_time: Optional[float] = None,
    ) -> WorkflowContext:
        """Return a new WorkflowContext advanced to the next semantic step."""
        from app.ai.planner.memory import sanitize_sensitive_data

        clean_sum = (
            sanitize_sensitive_data(summary, redact_coordinates=True)
            if summary
            else None
        )
        if clean_sum:
            import re
            clean_sum = re.sub(
                r"(?i)\b(?:password|secret|credential|api_key|token)\s*[:=\s]\s*['\"]?[^\s,'\"]+",
                "[REDACTED]",
                clean_sum,
            )
        if clean_sum and len(clean_sum) > 200:
            clean_sum = clean_sum[:197] + "..."

        now = current_time if current_time is not None else time.monotonic()
        return WorkflowContext(
            workflow_id=self.workflow_id,
            objective=self.objective,
            expected_app=expected_app or self.expected_app,
            expected_process=expected_process or self.expected_process,
            current_semantic_step=self.current_semantic_step + 1,
            previous_action=action or self.previous_action,
            previous_verification_outcome=outcome or self.previous_verification_outcome,
            verification_summary=clean_sum or self.verification_summary,
            is_valid=self.is_valid,
            created_at=now,
            ttl_seconds=self.ttl_seconds,
        )

    def invalidate(self, reason: Optional[str] = None) -> WorkflowContext:
        """Return an invalidated copy of the WorkflowContext."""
        return WorkflowContext(
            workflow_id=self.workflow_id,
            objective=self.objective,
            expected_app=self.expected_app,
            expected_process=self.expected_process,
            current_semantic_step=self.current_semantic_step,
            previous_action=self.previous_action,
            previous_verification_outcome=(
                f"INVALIDATED: {reason}" if reason else "INVALIDATED"
            ),
            verification_summary=None,
            is_valid=False,
            created_at=self.created_at,
            ttl_seconds=self.ttl_seconds,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize semantic context to dictionary (strictly zero physical state)."""
        return {
            "workflow_id": self.workflow_id,
            "objective": self.objective,
            "expected_app": self.expected_app,
            "expected_process": self.expected_process,
            "current_semantic_step": self.current_semantic_step,
            "previous_action": self.previous_action,
            "previous_verification_outcome": self.previous_verification_outcome,
            "verification_summary": self.verification_summary,
            "is_valid": self.is_valid,
            "created_at": self.created_at,
            "ttl_seconds": self.ttl_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkflowContext:
        """Deserialize dictionary into WorkflowContext."""
        return cls(
            workflow_id=str(data.get("workflow_id", "")),
            objective=str(data.get("objective", "")),
            expected_app=data.get("expected_app"),
            expected_process=data.get("expected_process"),
            current_semantic_step=int(data.get("current_semantic_step", 1)),
            previous_action=data.get("previous_action"),
            previous_verification_outcome=data.get("previous_verification_outcome"),
            verification_summary=data.get("verification_summary"),
            is_valid=bool(data.get("is_valid", True)),
            created_at=float(data.get("created_at", time.monotonic())),
            ttl_seconds=float(data.get("ttl_seconds", 60.0)),
        )


@dataclass
class Plan:
    """Represents an ordered sequence of tasks created to satisfy a user query.

    Attributes:
        query: The original user query.
        tasks: Ordered list of tasks to execute.
        id: Unique identifier for the plan.
        created_at: Unix timestamp marking plan creation.
        strategy: Planning strategy utilized to generate this plan.
        metadata: Optional metadata dictionary associated with plan generation.
        workflow_context: Optional semantic workflow context for multi-step continuity.
    """

    query: str
    tasks: List[Task] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    strategy: PlanningStrategy = PlanningStrategy.RULE_BASED
    metadata: Dict[str, Any] = field(default_factory=dict)
    workflow_context: Optional[WorkflowContext] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize Plan to a dictionary."""
        d = {
            "id": self.id,
            "query": self.query,
            "tasks": [t.to_dict() for t in self.tasks],
            "strategy": self.strategy.value if hasattr(self.strategy, "value") else str(self.strategy),
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }
        if self.workflow_context is not None:
            d["workflow_context"] = self.workflow_context.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Plan:
        """Deserialize dictionary to Plan."""
        strategy_val = data.get("strategy", PlanningStrategy.RULE_BASED.value)
        try:
            strategy = PlanningStrategy(strategy_val)
        except Exception:
            strategy = PlanningStrategy.RULE_BASED
        tasks = [Task.from_dict(t) for t in data.get("tasks", [])]
        wf_raw = data.get("workflow_context")
        wf_ctx = None
        if isinstance(wf_raw, dict):
            try:
                wf_ctx = WorkflowContext.from_dict(wf_raw)
            except Exception:
                wf_ctx = None
        return cls(
            id=str(data.get("id", str(uuid.uuid4()))),
            query=str(data.get("query", "")),
            tasks=tasks,
            strategy=strategy,
            created_at=float(data.get("created_at", time.time())),
            metadata=dict(data.get("metadata", {})),
            workflow_context=wf_ctx,
        )

    def is_empty(self) -> bool:
        """Return True if plan contains no tasks."""
        return len(self.tasks) == 0


@dataclass
class ExecutionResult:
    """Outcome of executing a plan.

    Attributes:
        success: Whether all tasks completed successfully.
        completed_tasks: List of tasks that executed successfully.
        failed_tasks: List of tasks that failed.
        output: Human-readable output summary or return value.
        skipped_tasks: List of tasks skipped due to dependency failures.
        execution_order: Order of task IDs executed during plan processing.
        dependency_failures: Mapping of skipped task ID to failed dependency task IDs.
    """

    success: bool
    completed_tasks: List[Task] = field(default_factory=list)
    failed_tasks: List[Task] = field(default_factory=list)
    output: Optional[str] = None
    skipped_tasks: List[Task] = field(default_factory=list)
    execution_order: List[str] = field(default_factory=list)
    dependency_failures: Dict[str, List[str]] = field(default_factory=dict)
    execution_duration: float = 0.0
    task_outputs: Dict[str, str] = field(default_factory=dict)
    task_results: Dict[str, Any] = field(default_factory=dict)

    @property
    def completed_task_ids(self) -> Set[str]:
        """Return a set of task IDs that completed successfully."""
        return {t.id for t in self.completed_tasks}

    @property
    def failed_task_ids(self) -> Set[str]:
        """Return a set of task IDs that failed."""
        return {t.id for t in self.failed_tasks}

    @property
    def skipped_task_ids(self) -> Set[str]:
        """Return a set of task IDs that were skipped."""
        return {t.id for t in self.skipped_tasks}

    def merge(self, other: ExecutionResult) -> ExecutionResult:
        """Immutably merge this ExecutionResult with another (recovery) ExecutionResult.

        Merge Contract:
        1. Immutability: Returns a new ExecutionResult instance without mutating either
           `self` or `other`.
        2. Completed Tasks: Preserves all completed tasks from `self`, then appends completed
           tasks from `other` whose task IDs have not already been recorded, preventing duplicates.
        3. Execution Order: Preserves execution order from `self`, then appends task IDs from
           `other` that have not yet appeared, maintaining deterministic execution history.
        4. Dependency Failures: Merges failure mapping across both results. If a task ID exists
           in both, unique failed dependency IDs are unioned without duplicates.
        5. Execution Duration: Total duration is accumulated (self.execution_duration + other.execution_duration).
        6. Authoritative Final Outcome: `other` represents the recovery execution and provides the
           authoritative final values for `failed_tasks`, `skipped_tasks`, and `success`.
        7. Sensible Output: Uses `other.output` if non-empty; falls back to `self.output` otherwise.

        Args:
            other: The subsequent (e.g. recovery) ExecutionResult to merge.

        Returns:
            A new merged ExecutionResult instance.
        """
        # 1. Merge completed tasks preserving order without duplicate task IDs
        seen_completed: Set[str] = set()
        merged_completed: List[Task] = []
        for t in self.completed_tasks:
            if t.id not in seen_completed:
                seen_completed.add(t.id)
                merged_completed.append(t)
        for t in other.completed_tasks:
            if t.id not in seen_completed:
                seen_completed.add(t.id)
                merged_completed.append(t)

        # 2. Merge execution order deterministically without duplicate task IDs
        seen_order: Set[str] = set()
        merged_order: List[str] = []
        for tid in self.execution_order:
            if tid not in seen_order:
                seen_order.add(tid)
                merged_order.append(tid)
        for tid in other.execution_order:
            if tid not in seen_order:
                seen_order.add(tid)
                merged_order.append(tid)

        # 3. Merge dependency failures
        merged_deps: Dict[str, List[str]] = {k: list(v) for k, v in self.dependency_failures.items()}
        for k, v in other.dependency_failures.items():
            if k in merged_deps:
                seen_deps = set(merged_deps[k])
                for dep in v:
                    if dep not in seen_deps:
                        merged_deps[k].append(dep)
                        seen_deps.add(dep)
            else:
                merged_deps[k] = list(v)

        # 4. Resolve output sensibly
        resolved_output = (
            other.output
            if (other.output is not None and str(other.output).strip())
            else self.output
        )

        merged_outputs = dict(self.task_outputs)
        merged_outputs.update(other.task_outputs)
        merged_results = dict(self.task_results)
        merged_results.update(other.task_results)

        return ExecutionResult(
            success=other.success,
            completed_tasks=merged_completed,
            failed_tasks=list(other.failed_tasks),
            skipped_tasks=list(other.skipped_tasks),
            execution_order=merged_order,
            dependency_failures=merged_deps,
            execution_duration=self.execution_duration + other.execution_duration,
            output=resolved_output,
            task_outputs=merged_outputs,
            task_results=merged_results,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ExecutionResult to a dictionary."""
        return {
            "success": self.success,
            "completed_tasks": [t.to_dict() for t in self.completed_tasks],
            "failed_tasks": [t.to_dict() for t in self.failed_tasks],
            "skipped_tasks": [t.to_dict() for t in self.skipped_tasks],
            "execution_order": list(self.execution_order),
            "dependency_failures": {k: list(v) for k, v in self.dependency_failures.items()},
            "execution_duration": self.execution_duration,
            "output": self.output,
        }
