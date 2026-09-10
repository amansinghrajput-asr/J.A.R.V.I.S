"""Planner Package for J.A.R.V.I.S.

Provides goal decomposition, task sequencing, plan execution,
execution memory, heuristics, recovery foundations, and event-driven observability.
"""

from __future__ import annotations

from app.ai.planner.control import (
    CancellationToken,
    ExecutionCancelledError,
    ExecutionController,
    ExecutionPausedError,
)
from app.ai.planner.events import (
    PlanCancelled,
    PlanCompleted,
    PlanFailed,
    PlannerEvent,
    PlannerEventBus,
    PlanPaused,
    PlanResumed,
    PlanStarted,
    RecoveryAborted,
    RecoveryCompleted,
    RecoveryFailed,
    RecoveryStarted,
    TaskCompleted,
    TaskFailed,
    TaskRetried,
    TaskStarted,
    TaskTimeout,
)
from app.ai.planner.executor import Executor, executor
from app.ai.planner.failure_classifier import FailureClassifier
from app.ai.planner.heuristics import RecoveryDecision, evaluate_recovery_viability
from app.ai.planner.logging_subscriber import StructuredLoggingSubscriber
from app.ai.planner.memory import (
    ExecutionMemory,
    ExecutionMetrics,
    FailureCategory,
    TaskExecutionRecord,
)
from app.ai.planner.memory_summary import MemorySummaryBuilder
from app.ai.planner.metrics_collector import PlannerMetricsCollector
from app.ai.planner.models import (
    ExecutionResult,
    Plan,
    PlanningStrategy,
    Task,
    TaskStatus,
)
from app.ai.planner.multi_agent import (
    AgentCapability,
    AgentCommunicationBus,
    AgentManifest,
    AgentMessage,
    AgentMessageType,
    AgentRegistry,
    AgentRole,
    AgentSelector,
    AgentStatus,
    ArtifactStore,
    BaseAgent,
    ConflictResolution,
    ConflictResolver,
    CriticAgent,
    CriticFeedback,
    DelegationRequest,
    DelegationResponse,
    MultiAgentCoordinator,
    ReflectionPipeline,
    ResultAggregator,
    SharedAgentMemory,
    WorkerAgent,
)
from app.ai.planner.persistence import (
    PersistedExecutionState,
    load,
    resume,
    save,
)
from app.ai.planner.planner import Planner, planner
from app.ai.planner.replay import PlannerReplayEngine, ReplaySnapshot
from app.ai.planner.timeline import ExecutionTimeline
from app.ai.planner.timeouts import (
    TaskTimeoutError,
    TimeoutConfig,
    TimeoutManager,
    TimeoutPolicy,
)

__all__ = [
    "AgentCapability",
    "AgentCommunicationBus",
    "AgentManifest",
    "AgentMessage",
    "AgentMessageType",
    "AgentRegistry",
    "AgentRole",
    "AgentSelector",
    "AgentStatus",
    "ArtifactStore",
    "BaseAgent",
    "CancellationToken",
    "ConflictResolution",
    "ConflictResolver",
    "CriticAgent",
    "CriticFeedback",
    "DelegationRequest",
    "DelegationResponse",
    "ExecutionCancelledError",
    "ExecutionController",
    "ExecutionMemory",
    "ExecutionMetrics",
    "ExecutionPausedError",
    "ExecutionResult",
    "ExecutionTimeline",
    "Executor",
    "FailureCategory",
    "FailureClassifier",
    "MemorySummaryBuilder",
    "MultiAgentCoordinator",
    "PersistedExecutionState",
    "Plan",
    "PlanCancelled",
    "PlanCompleted",
    "PlanFailed",
    "PlanPaused",
    "PlanResumed",
    "PlanStarted",
    "Planner",
    "PlannerEvent",
    "PlannerEventBus",
    "PlannerMetricsCollector",
    "PlannerReplayEngine",
    "PlanningStrategy",
    "RecoveryAborted",
    "RecoveryCompleted",
    "RecoveryDecision",
    "RecoveryFailed",
    "RecoveryStarted",
    "ReflectionPipeline",
    "ReplaySnapshot",
    "ResultAggregator",
    "SharedAgentMemory",
    "StructuredLoggingSubscriber",
    "Task",
    "TaskCompleted",
    "TaskFailed",
    "TaskRetried",
    "TaskStarted",
    "TaskStatus",
    "TaskTimeout",
    "TaskTimeoutError",
    "TimeoutConfig",
    "TimeoutManager",
    "TimeoutPolicy",
    "WorkerAgent",
    "evaluate_recovery_viability",
    "executor",
    "load",
    "planner",
    "resume",
    "save",
]
