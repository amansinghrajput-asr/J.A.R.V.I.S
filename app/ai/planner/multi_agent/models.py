"""Data Models for Phase 18.0 Multi-Agent Planner Architecture.

Defines agent roles, lifecycle states, capabilities, manifests,
inter-agent communication message contracts, and delegation structures.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class AgentRole(str, Enum):
    """Functional role of an agent in the multi-agent system."""

    COORDINATOR = "COORDINATOR"
    WORKER = "WORKER"
    CRITIC = "CRITIC"
    RESEARCHER = "RESEARCHER"
    EXECUTOR = "EXECUTOR"
    CUSTOM = "CUSTOM"


class AgentStatus(str, Enum):
    """Lifecycle operational state of an agent."""

    CREATED = "CREATED"
    IDLE = "IDLE"
    BUSY = "BUSY"
    PAUSED = "PAUSED"
    FAILED = "FAILED"
    TERMINATED = "TERMINATED"


class AgentMessageType(str, Enum):
    """Categorization of inter-agent messages."""

    DELEGATE = "DELEGATE"
    INFORM = "INFORM"
    REQUEST = "REQUEST"
    RESPONSE = "RESPONSE"
    CRITIQUE = "CRITIQUE"
    ERROR = "ERROR"
    BROADCAST = "BROADCAST"


@dataclass(frozen=True)
class AgentCapability:
    """Declared capability of an agent.

    Attributes:
        name: Identifier of the capability.
        description: Human-readable summary of capability.
        supported_actions: Set of task action strings this capability satisfies.
        domain_tags: Set of thematic domain tags (e.g. {'web', 'code', 'research'}).
        resource_cost: Relative execution cost weight (default: 1.0).
        max_concurrency: Maximum parallel tasks for this specific capability.
        confidence_score: Base self-reported competency score [0.0, 1.0].
    """

    name: str
    description: str = ""
    supported_actions: Set[str] = field(default_factory=set)
    domain_tags: Set[str] = field(default_factory=set)
    resource_cost: float = 1.0
    max_concurrency: int = 4
    confidence_score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert capability to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "supported_actions": sorted(self.supported_actions),
            "domain_tags": sorted(self.domain_tags),
            "resource_cost": self.resource_cost,
            "max_concurrency": self.max_concurrency,
            "confidence_score": self.confidence_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentCapability:
        """Create capability from dictionary."""
        return cls(
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            supported_actions=set(data.get("supported_actions", [])),
            domain_tags=set(data.get("domain_tags", [])),
            resource_cost=float(data.get("resource_cost", 1.0)),
            max_concurrency=int(data.get("max_concurrency", 4)),
            confidence_score=float(data.get("confidence_score", 1.0)),
        )


@dataclass
class AgentManifest:
    """Registration and identity specification of an agent.

    Attributes:
        agent_id: Unique string identifier for the agent.
        name: Human-readable agent display name.
        role: Primary functional role in planning and execution.
        capabilities: Collection of capabilities the agent advertises.
        system_prompt: Core directive or behavioral prompt governing the agent.
        max_concurrency: Global maximum concurrent tasks for this agent.
        timeout_seconds: Optional default execution timeout per task.
        allowed_actions: Set of permitted action types for security sandboxing.
        metadata: Custom arbitrary attributes.
    """

    agent_id: str
    name: str = ""
    role: AgentRole = AgentRole.WORKER
    capabilities: List[AgentCapability] = field(default_factory=list)
    system_prompt: str = ""
    max_concurrency: int = 4
    timeout_seconds: Optional[float] = None
    allowed_actions: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to dictionary."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role.value if hasattr(self.role, "value") else str(self.role),
            "capabilities": [c.to_dict() for c in self.capabilities],
            "system_prompt": self.system_prompt,
            "max_concurrency": self.max_concurrency,
            "timeout_seconds": self.timeout_seconds,
            "allowed_actions": sorted(self.allowed_actions),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentManifest:
        """Create manifest from dictionary."""
        role_val = data.get("role", AgentRole.WORKER.value)
        try:
            role = AgentRole(role_val)
        except Exception:
            role = AgentRole.WORKER

        caps = [AgentCapability.from_dict(c) for c in data.get("capabilities", [])]
        timeout = data.get("timeout_seconds")

        return cls(
            agent_id=str(data.get("agent_id", "")),
            name=str(data.get("name", "")),
            role=role,
            capabilities=caps,
            system_prompt=str(data.get("system_prompt", "")),
            max_concurrency=int(data.get("max_concurrency", 4)),
            timeout_seconds=float(timeout) if timeout is not None else None,
            allowed_actions=set(data.get("allowed_actions", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class AgentMessage:
    """Inter-agent communication payload.

    Attributes:
        id: Unique message identifier.
        sender: Sending agent ID.
        recipient: Target agent ID ('coordinator', specific agent_id, or 'broadcast').
        message_type: Nature of the message.
        content: Structured payload data.
        correlation_id: Conversation or task correlation ID.
        timestamp: Unix epoch time when sent.
        metadata: Extensible contextual headers.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    recipient: str = ""
    message_type: AgentMessageType = AgentMessageType.INFORM
    content: Dict[str, Any] = field(default_factory=dict)
    correlation_id: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "id": self.id,
            "sender": self.sender,
            "recipient": self.recipient,
            "message_type": self.message_type.value if hasattr(self.message_type, "value") else str(self.message_type),
            "content": dict(self.content),
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentMessage:
        """Create message from dictionary."""
        m_type_val = data.get("message_type", AgentMessageType.INFORM.value)
        try:
            m_type = AgentMessageType(m_type_val)
        except Exception:
            m_type = AgentMessageType.INFORM

        return cls(
            id=str(data.get("id", str(uuid.uuid4()))),
            sender=str(data.get("sender", "")),
            recipient=str(data.get("recipient", "")),
            message_type=m_type,
            content=dict(data.get("content", {})),
            correlation_id=str(data.get("correlation_id", "")),
            timestamp=float(data.get("timestamp", time.time())),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class DelegationRequest:
    """Contract defining a delegated task sent from a coordinator to an agent."""

    task_id: str
    action: str
    target: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    delegated_by: str = "coordinator"
    timeout_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary."""
        return {
            "task_id": self.task_id,
            "action": self.action,
            "target": self.target,
            "parameters": dict(self.parameters),
            "context": dict(self.context),
            "delegated_by": self.delegated_by,
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DelegationRequest:
        """Create request from dictionary."""
        t_out = data.get("timeout_seconds")
        return cls(
            task_id=str(data.get("task_id", "")),
            action=str(data.get("action", "")),
            target=data.get("target"),
            parameters=dict(data.get("parameters", {})),
            context=dict(data.get("context", {})),
            delegated_by=str(data.get("delegated_by", "coordinator")),
            timeout_seconds=float(t_out) if t_out is not None else None,
        )


@dataclass
class DelegationResponse:
    """Outcome returned by an agent after executing a delegated task."""

    task_id: str
    agent_id: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "artifacts": dict(self.artifacts),
            "duration": self.duration,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DelegationResponse:
        """Create response from dictionary."""
        return cls(
            task_id=str(data.get("task_id", "")),
            agent_id=str(data.get("agent_id", "")),
            success=bool(data.get("success", False)),
            output=data.get("output"),
            error=data.get("error"),
            artifacts=dict(data.get("artifacts", {})),
            duration=float(data.get("duration", 0.0)),
            confidence=float(data.get("confidence", 1.0)),
        )


@dataclass(frozen=True)
class CriticFeedback:
    """Critique evaluated by a CriticAgent or ReflectionPipeline."""

    task_id: str
    critic_id: str
    passed: bool
    score: float
    critique: str
    suggestions: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert feedback to dictionary."""
        return {
            "task_id": self.task_id,
            "critic_id": self.critic_id,
            "passed": self.passed,
            "score": self.score,
            "critique": self.critique,
            "suggestions": list(self.suggestions),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CriticFeedback:
        """Create feedback from dictionary."""
        return cls(
            task_id=str(data.get("task_id", "")),
            critic_id=str(data.get("critic_id", "")),
            passed=bool(data.get("passed", False)),
            score=float(data.get("score", 0.0)),
            critique=str(data.get("critique", "")),
            suggestions=list(data.get("suggestions", [])),
            timestamp=float(data.get("timestamp", time.time())),
        )


@dataclass(frozen=True)
class ConflictResolution:
    """Record of a resolved conflict between agents."""

    conflict_id: str
    conflict_type: str
    parties: List[str]
    strategy: str
    resolved: bool
    winner_agent_id: Optional[str] = None
    resolution_data: Any = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert resolution record to dictionary."""
        return {
            "conflict_id": self.conflict_id,
            "conflict_type": self.conflict_type,
            "parties": list(self.parties),
            "strategy": self.strategy,
            "resolved": self.resolved,
            "winner_agent_id": self.winner_agent_id,
            "resolution_data": self.resolution_data,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConflictResolution:
        """Create resolution record from dictionary."""
        return cls(
            conflict_id=str(data.get("conflict_id", "")),
            conflict_type=str(data.get("conflict_type", "")),
            parties=list(data.get("parties", [])),
            strategy=str(data.get("strategy", "")),
            resolved=bool(data.get("resolved", False)),
            winner_agent_id=data.get("winner_agent_id"),
            resolution_data=data.get("resolution_data"),
            timestamp=float(data.get("timestamp", time.time())),
        )
