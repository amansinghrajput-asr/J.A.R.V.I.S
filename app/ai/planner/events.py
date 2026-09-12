"""Event System for the J.A.R.V.I.S. Planner Subsystem.

Defines immutable, strongly-typed planner events and a decoupled, thread-safe,
asynchronous-compatible PlannerEventBus with subscriber error isolation.
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
import logging
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar

logger = logging.getLogger("app.ai.planner.events")


# ---------------------------------------------------------------------------
# Base Event
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlannerEvent:
    """Base immutable event for all planner lifecycle notifications.

    Attributes:
        schema_version: Event schema version for serialization compatibility.
        sequence_id: Monotonically increasing identifier assigned by EventBus.
        timestamp: Unix epoch timestamp when the event was created.
        execution_id: Unique correlation ID identifying the execution flow.
        metadata: Extensible contextual key-value pairs.
    """

    schema_version: int = 1
    sequence_id: int = 0
    timestamp: float = field(default_factory=time.time)
    execution_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to a serializable dictionary."""
        data = {
            "event_type": self.__class__.__name__,
            "schema_version": self.schema_version,
            "sequence_id": self.sequence_id,
            "timestamp": self.timestamp,
            "execution_id": self.execution_id,
            "metadata": dict(self.metadata),
        }
        for k, v in self.__dict__.items():
            if k not in data:
                data[k] = v
        return data


TEvent = TypeVar("TEvent", bound=PlannerEvent)


# ---------------------------------------------------------------------------
# Plan Lifecycle Events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlanStarted(PlannerEvent):
    """Emitted when initial plan decomposition begins or execution starts."""

    plan_id: str = ""
    query: str = ""
    task_count: int = 0
    strategy: str = ""


@dataclass(frozen=True)
class PlanCompleted(PlannerEvent):
    """Emitted when plan execution completes successfully or reaches terminal state."""

    plan_id: str = ""
    success: bool = True
    completed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    duration: float = 0.0


@dataclass(frozen=True)
class PlanFailed(PlannerEvent):
    """Emitted when an entire plan fails or encounters fatal validation error."""

    plan_id: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Task Lifecycle Events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskStarted(PlannerEvent):
    """Emitted immediately before an individual task handler executes."""

    plan_id: str = ""
    task_id: str = ""
    action: str = ""
    target: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class TaskCompleted(PlannerEvent):
    """Emitted when an individual task handler finishes successfully."""

    plan_id: str = ""
    task_id: str = ""
    action: str = ""
    result: str = ""
    duration: float = 0.0


@dataclass(frozen=True)
class TaskFailed(PlannerEvent):
    """Emitted when an individual task handler encounters an error."""

    plan_id: str = ""
    task_id: str = ""
    action: str = ""
    error: str = ""
    duration: float = 0.0


@dataclass(frozen=True)
class TaskRetried(PlannerEvent):
    """Emitted when a failed task is scheduled for re-execution or retried."""

    plan_id: str = ""
    task_id: str = ""
    action: str = ""
    attempt: int = 1
    reason: str = ""


# ---------------------------------------------------------------------------
# Recovery Lifecycle Events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RecoveryStarted(PlannerEvent):
    """Emitted when automatic recovery replanning is initiated."""

    query: str = ""
    attempt: int = 1
    failed_task_ids: List[str] = field(default_factory=list)
    skipped_task_ids: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class RecoveryCompleted(PlannerEvent):
    """Emitted when a recovery wave completes execution."""

    query: str = ""
    attempt: int = 1
    success: bool = True
    new_plan_id: Optional[str] = None
    task_count: int = 0
    duration: float = 0.0


@dataclass(frozen=True)
class RecoveryFailed(PlannerEvent):
    """Emitted when recovery replanning is rejected or fails to produce progress."""

    query: str = ""
    attempt: int = 1
    reason: str = ""
    duration: float = 0.0


# ---------------------------------------------------------------------------
# Execution Control & Lifecycle Events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlanPaused(PlannerEvent):
    """Emitted when plan execution is temporarily suspended."""

    plan_id: str = ""
    reason: str = ""


@dataclass(frozen=True)
class PlanResumed(PlannerEvent):
    """Emitted when suspended plan execution is resumed."""

    plan_id: str = ""
    reason: str = ""


@dataclass(frozen=True)
class PlanCancelled(PlannerEvent):
    """Emitted when plan execution is aborted via cancellation."""

    plan_id: str = ""
    reason: str = ""


@dataclass(frozen=True)
class TaskTimeout(PlannerEvent):
    """Emitted when a task execution exceeds its allocated deadline."""

    plan_id: str = ""
    task_id: str = ""
    timeout_seconds: float = 0.0
    policy: str = "SKIP"


@dataclass(frozen=True)
class RecoveryAborted(PlannerEvent):
    """Emitted when replanning loop is explicitly terminated before exhausting retries."""

    query: str = ""
    attempt: int = 1
    reason: str = ""


# ---------------------------------------------------------------------------
# Multi-Agent Coordination Events (Phase 18)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentRegistered(PlannerEvent):
    """Emitted when a new agent registers with the AgentRegistry."""

    agent_id: str = ""
    name: str = ""
    role: str = ""
    capabilities: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class AgentDeregistered(PlannerEvent):
    """Emitted when an agent is unregistered from the AgentRegistry."""

    agent_id: str = ""
    reason: str = ""


@dataclass(frozen=True)
class AgentStatusChanged(PlannerEvent):
    """Emitted when an agent's lifecycle state transitions."""

    agent_id: str = ""
    old_status: str = ""
    new_status: str = ""


@dataclass(frozen=True)
class AgentMessageSent(PlannerEvent):
    """Emitted when a message is dispatched across the AgentCommunicationBus."""

    message_id: str = ""
    sender: str = ""
    recipient: str = ""
    message_type: str = ""
    correlation_id: str = ""


@dataclass(frozen=True)
class TaskDelegated(PlannerEvent):
    """Emitted when a coordinator delegates a task to a specialized agent."""

    plan_id: str = ""
    task_id: str = ""
    agent_id: str = ""
    action: str = ""


@dataclass(frozen=True)
class TaskCompletedByAgent(PlannerEvent):
    """Emitted when an agent finishes processing its assigned task."""

    plan_id: str = ""
    task_id: str = ""
    agent_id: str = ""
    action: str = ""
    duration: float = 0.0
    success: bool = True


@dataclass(frozen=True)
class AgentCritiqueSubmitted(PlannerEvent):
    """Emitted when the reflection pipeline or CriticAgent produces a critique."""

    plan_id: str = ""
    task_id: str = ""
    critic_id: str = ""
    passed: bool = True
    feedback: str = ""
    score: float = 1.0


@dataclass(frozen=True)
class ConflictDetected(PlannerEvent):
    """Emitted when divergent agent outputs or conflicting decisions occur."""

    plan_id: str = ""
    conflict_type: str = ""
    parties: List[str] = field(default_factory=list)
    details: str = ""


@dataclass(frozen=True)
class ConflictResolved(PlannerEvent):
    """Emitted when a conflict between agents is reconciled."""

    plan_id: str = ""
    conflict_type: str = ""
    resolution_strategy: str = ""
    outcome: str = ""


# ---------------------------------------------------------------------------
# Phase 19 Swarm Lifecycle Events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SwarmCreated(PlannerEvent):
    """Emitted when a new SubSwarm is instantiated."""

    swarm_id: str = ""
    parent_swarm_id: Optional[str] = None
    depth: int = 0


@dataclass(frozen=True)
class SwarmTerminated(PlannerEvent):
    """Emitted when a SubSwarm is terminated."""

    swarm_id: str = ""


@dataclass(frozen=True)
class SubSwarmSpawned(PlannerEvent):
    """Emitted when a parent swarm spawns a child sub-swarm."""

    parent_swarm_id: str = ""
    child_swarm_id: str = ""
    depth: int = 1


@dataclass(frozen=True)
class SpeculativeRaceStarted(PlannerEvent):
    """Emitted when speculative parallel execution of candidate agents begins."""

    task_id: str = ""
    candidates: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class SpeculativeRaceCompleted(PlannerEvent):
    """Emitted when a speculative race concludes with a winning branch."""

    task_id: str = ""
    winning_agent_id: str = ""
    duration_ms: float = 0.0


@dataclass(frozen=True)
class ConsensusReached(PlannerEvent):
    """Emitted when swarm consensus is achieved across proposals."""

    strategy: str = ""
    winning_id: str = ""
    winning_score: float = 0.0
    total_votes: int = 0


@dataclass(frozen=True)
class SwarmHealthChecked(PlannerEvent):
    """Emitted when autonomous supervisor completes a health inspection."""

    healthy: bool = True
    total_swarms: int = 0
    stuck_agents: List[str] = field(default_factory=list)
    faulted_agents: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class SwarmHealed(PlannerEvent):
    """Emitted when autonomous supervisor executes self-healing remediation."""

    actions_count: int = 0
    actions_summary: str = ""


@dataclass(frozen=True)
class InterventionRequested(PlannerEvent):
    """Emitted when an action requires human-in-the-loop approval."""

    request_id: str = ""
    action: str = ""
    risk_level: str = "MEDIUM"


@dataclass(frozen=True)
class InterventionResolved(PlannerEvent):
    """Emitted when a human-in-the-loop approval decision is recorded."""

    request_id: str = ""
    approved: bool = False
    decided_by: str = ""


@dataclass(frozen=True)
class PolicyViolationDetected(PlannerEvent):
    """Emitted when an agent action violates a security policy or boundary."""

    action: str = ""
    reason: str = ""
    agent_id: str = ""


# ---------------------------------------------------------------------------
# Phase 20 Metacognition & Tool Synthesis Lifecycle Events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolSynthesisStarted(PlannerEvent):
    """Emitted when dynamic tool synthesis commences."""

    action_name: str = ""
    target: str = ""


@dataclass(frozen=True)
class ToolSynthesisVerified(PlannerEvent):
    """Emitted when a synthesized tool passes verification and is promoted."""

    action_name: str = ""
    execution_time_ms: float = 0.0
    attempts: int = 1


@dataclass(frozen=True)
class ToolSynthesisRejected(PlannerEvent):
    """Emitted when dynamic tool synthesis fails verification and is rejected."""

    action_name: str = ""
    reason: str = ""
    attempts: int = 1


@dataclass(frozen=True)
class CausalDiagnosisGenerated(PlannerEvent):
    """Emitted when causal reflection generates a diagnosis for a failure."""

    trajectory_id: str = ""
    root_cause: str = ""
    invariant: str = ""


@dataclass(frozen=True)
class SkillDistillationCompleted(PlannerEvent):
    """Emitted when a multi-agent subplan is distilled into a verified macro skill."""

    skill_name: str = ""
    speedup_ratio: float = 1.0


@dataclass(frozen=True)
class KnowledgeGraphUpdated(PlannerEvent):
    """Emitted when entities, relations, or invariants are added or updated in the knowledge graph."""

    subject: str = ""
    predicate: str = ""
    target: str = ""
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Phase 21 Autonomous Learning & Skill Evolution Lifecycle Events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SkillEvaluated(PlannerEvent):
    """Emitted when a skill performance metric is calculated and quality scored."""

    skill_name: str = ""
    score: float = 0.0
    status: str = ""
    recommendation: str = ""


@dataclass(frozen=True)
class SkillDegraded(PlannerEvent):
    """Emitted when repeated failures cause a skill to transition to DEGRADED."""

    skill_name: str = ""
    reason: str = ""
    consecutive_failures: int = 0
    score: float = 0.0


@dataclass(frozen=True)
class SkillQuarantined(PlannerEvent):
    """Emitted when severe or persistent failures cause a skill to be QUARANTINED."""

    skill_name: str = ""
    reason: str = ""


@dataclass(frozen=True)
class SkillImprovementStarted(PlannerEvent):
    """Emitted when causal reflection initiates autonomous improvement of a skill."""

    skill_name: str = ""
    reason: str = ""
    prescription: str = ""


@dataclass(frozen=True)
class SkillImprovementCompleted(PlannerEvent):
    """Emitted when candidate skill code generation and sandbox verification finishes."""

    skill_name: str = ""
    version: str = ""
    verified: bool = False
    error: Optional[str] = None


@dataclass(frozen=True)
class SkillVersionPromoted(PlannerEvent):
    """Emitted when a verified candidate skill version is promoted to active execution."""

    skill_name: str = ""
    version: str = ""
    previous_version: Optional[str] = None


@dataclass(frozen=True)
class SkillRollback(PlannerEvent):
    """Emitted when a degraded or failing skill version is rolled back to a stable version."""

    skill_name: str = ""
    restored_version: str = ""
    reason: str = ""


@dataclass(frozen=True)
class SkillRetired(PlannerEvent):
    """Emitted when an obsolete or inactive skill is safely retired."""

    skill_name: str = ""
    reason: str = ""


# ---------------------------------------------------------------------------
# Phase 22: System Skill Observability Events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SystemSkillStarted(PlannerEvent):
    """Emitted when a real system/PC skill operation begins execution."""

    skill_name: str = ""
    operation: str = ""
    target: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SystemSkillCompleted(PlannerEvent):
    """Emitted when a system skill operation completes successfully."""

    skill_name: str = ""
    operation: str = ""
    target: Optional[str] = None
    duration: float = 0.0
    success: bool = True
    result: Any = None


@dataclass(frozen=True)
class SystemSkillFailed(PlannerEvent):
    """Emitted when a system skill operation encounters an execution error."""

    skill_name: str = ""
    operation: str = ""
    target: Optional[str] = None
    error: str = ""
    duration: float = 0.0


@dataclass(frozen=True)
class SystemSkillConfirmationRequired(PlannerEvent):
    """Emitted when a system operation requires interactive operator confirmation."""

    skill_name: str = ""
    operation: str = ""
    target: Optional[str] = None
    confirmation_id: str = ""
    risk_level: str = "HIGH"


@dataclass(frozen=True)
class SystemSkillPolicyRejected(PlannerEvent):
    """Emitted when a system operation is blocked by security policy."""

    skill_name: str = ""
    operation: str = ""
    target: Optional[str] = None
    reason: str = ""
    tier: str = "RESTRICTED"


# ---------------------------------------------------------------------------
# Planner Event Bus
# ---------------------------------------------------------------------------


class PlannerEventBus:
    """Thread-safe, subscriber-isolated event bus for planner observability.

    Supports:
    - Strongly-typed subscriptions (e.g. subscribe(TaskCompleted, callback))
    - Wildcard subscriptions (e.g. subscribe(PlannerEvent, callback))
    - Monotonic atomic sequence numbering
    - Exception isolation (subscriber failures never interrupt execution)
    - Synchronous and asynchronous dispatching
    - Scoped instances without global state
    """

    def __init__(self) -> None:
        """Initialize a new PlannerEventBus instance."""
        self._lock = threading.RLock()
        self._counter = itertools.count(1)
        self._subscribers: Dict[Type[PlannerEvent], List[Callable[[Any], Any]]] = {}
        self._wildcard_subscribers: List[Callable[[PlannerEvent], Any]] = []

    def next_sequence_id(self) -> int:
        """Return the next monotonically increasing sequence ID."""
        with self._lock:
            return next(self._counter)

    def subscribe(
        self,
        event_type: Type[TEvent],
        handler: Callable[[TEvent], Any],
    ) -> Callable[[], bool]:
        """Subscribe a handler callback to a specific event type or wildcard.

        If event_type is PlannerEvent, handler receives all events.
        Otherwise, handler only receives instances of event_type.

        Args:
            event_type: Subclass of PlannerEvent to observe.
            handler: Callable taking the event as its single argument.

        Returns:
            A callable unsubscription function `() -> bool`.
        """
        if not callable(handler):
            raise TypeError("Subscriber handler must be a callable.")

        with self._lock:
            if event_type is PlannerEvent:
                if handler not in self._wildcard_subscribers:
                    self._wildcard_subscribers.append(handler)
            else:
                sub_list = self._subscribers.setdefault(event_type, [])
                if handler not in sub_list:
                    sub_list.append(handler)

        def _unsubscribe() -> bool:
            return self.unsubscribe(event_type, handler)

        return _unsubscribe

    def unsubscribe(
        self,
        event_type: Type[TEvent],
        handler: Callable[[TEvent], Any],
    ) -> bool:
        """Remove a previously subscribed handler.

        Args:
            event_type: Target event type class.
            handler: The handler callback to remove.

        Returns:
            True if handler was found and removed, False otherwise.
        """
        with self._lock:
            if event_type is PlannerEvent:
                if handler in self._wildcard_subscribers:
                    self._wildcard_subscribers.remove(handler)
                    return True
                return False
            else:
                sub_list = self._subscribers.get(event_type)
                if sub_list and handler in sub_list:
                    sub_list.remove(handler)
                    if not sub_list:
                        del self._subscribers[event_type]
                    return True
                return False

    def publish(self, event: PlannerEvent) -> None:
        """Synchronously publish an event to all matching subscribers.

        Assigns a sequence_id if not already assigned. Subscriber errors are
        caught and logged to protect the caller.

        Args:
            event: PlannerEvent instance to broadcast.
        """
        assigned_event = self._prepare_event(event)
        handlers = self._gather_handlers(type(assigned_event))

        for handler in handlers:
            try:
                res = handler(assigned_event)
                if inspect.isawaitable(res):
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = None
                    if loop and loop.is_running():
                        asyncio.create_task(res)
                    else:
                        asyncio.run(res)
            except Exception as exc:
                logger.error(
                    "Error in PlannerEventBus subscriber %s handling %s: %s",
                    getattr(handler, "__name__", str(handler)),
                    type(assigned_event).__name__,
                    exc,
                    exc_info=True,
                )

    async def publish_async(self, event: PlannerEvent) -> None:
        """Asynchronously publish an event to all matching subscribers.

        Args:
            event: PlannerEvent instance to broadcast.
        """
        assigned_event = self._prepare_event(event)
        handlers = self._gather_handlers(type(assigned_event))

        for handler in handlers:
            try:
                res = handler(assigned_event)
                if inspect.isawaitable(res):
                    await res
            except Exception as exc:
                logger.error(
                    "Error in PlannerEventBus async subscriber %s handling %s: %s",
                    getattr(handler, "__name__", str(handler)),
                    type(assigned_event).__name__,
                    exc,
                    exc_info=True,
                )

    def _prepare_event(self, event: PlannerEvent) -> PlannerEvent:
        """Ensure the event has a valid monotonic sequence_id and timestamp."""
        if event.sequence_id <= 0:
            seq = self.next_sequence_id()
            return replace(event, sequence_id=seq)
        return event

    def _gather_handlers(self, event_cls: Type[PlannerEvent]) -> List[Callable[[Any], Any]]:
        """Collect all matching handlers for the given event class in thread-safe manner."""
        with self._lock:
            handlers: List[Callable[[Any], Any]] = []
            # Specific subscribers for this class and any matching base classes
            for registered_cls, sub_list in self._subscribers.items():
                if issubclass(event_cls, registered_cls):
                    handlers.extend(sub_list)
            # Wildcard subscribers
            handlers.extend(self._wildcard_subscribers)
            return list(handlers)
