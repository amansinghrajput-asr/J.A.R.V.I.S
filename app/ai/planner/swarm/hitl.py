"""Human-in-the-Loop (HITL) Gateway & Human Proxy Agent for Phase 19.5.

Provides interactive approval workflows, guidance injection, risk assessment,
and human-proxy delegation without global state.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from app.ai.planner.events import InterventionRequested, InterventionResolved, PlannerEventBus
from app.ai.planner.multi_agent.base import BaseAgent
from app.ai.planner.multi_agent.models import (
    AgentCapability,
    AgentManifest,
    AgentRole,
    DelegationRequest,
    DelegationResponse,
)

logger = logging.getLogger("app.ai.planner.swarm.hitl")


class RiskLevel(str, Enum):
    """Risk severity classifications for agent operations."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class ApprovalDecision:
    """Outcome of a human-in-the-loop review decision.

    Attributes:
        request_id: Identifier of the approval request.
        approved: Whether the operation was authorized.
        decided_by: Entity who made the decision (e.g. 'human_operator', 'policy_engine').
        reason: Justification or comment accompanying the decision.
        modifications: Optional parameter overrides injected by reviewer.
        timestamp: Unix epoch timestamp when decision was reached.
    """

    request_id: str
    approved: bool
    decided_by: str = "human_operator"
    reason: str = ""
    modifications: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to dictionary."""
        return {
            "request_id": self.request_id,
            "approved": self.approved,
            "decided_by": self.decided_by,
            "reason": self.reason,
            "modifications": dict(self.modifications),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ApprovalDecision:
        """Create decision from dictionary."""
        return cls(
            request_id=str(data.get("request_id", "")),
            approved=bool(data.get("approved", False)),
            decided_by=str(data.get("decided_by", "human_operator")),
            reason=str(data.get("reason", "")),
            modifications=dict(data.get("modifications", {})),
            timestamp=float(data.get("timestamp", time.time())),
        )


class InterventionGateway:
    """Thread-safe and async-compatible gateway managing human approvals and guidance."""

    def __init__(
        self,
        event_bus: Optional[PlannerEventBus] = None,
        default_timeout: float = 300.0,
    ) -> None:
        """Initialize InterventionGateway.

        Args:
            event_bus: Optional PlannerEventBus for emitting intervention events.
            default_timeout: Default timeout in seconds for awaiting approval.
        """
        self._lock = threading.RLock()
        self._event_bus = event_bus
        self._default_timeout = default_timeout
        self._pending_requests: Dict[str, Dict[str, Any]] = {}
        self._resolved_decisions: Dict[str, ApprovalDecision] = {}
        self._injected_guidance: Dict[str, List[str]] = {}
        self._approval_handler: Optional[Callable[[Dict[str, Any]], ApprovalDecision]] = None

    def register_approval_handler(
        self,
        handler: Callable[[Dict[str, Any]], ApprovalDecision],
    ) -> None:
        """Register a callback handler for resolving approval requests automatically or via UI."""
        with self._lock:
            self._approval_handler = handler

    def inject_guidance(self, swarm_id: str, guidance: str) -> None:
        """Inject operator guidance or directive into a sub-swarm.

        Args:
            swarm_id: Target swarm identifier.
            guidance: Human guidance instruction text.
        """
        with self._lock:
            if swarm_id not in self._injected_guidance:
                self._injected_guidance[swarm_id] = []
            self._injected_guidance[swarm_id].append(guidance)
            logger.info("Injected guidance to swarm '%s': %s", swarm_id, guidance)

    def get_guidance(self, swarm_id: str) -> List[str]:
        """Retrieve all guidance directives for a swarm."""
        with self._lock:
            return list(self._injected_guidance.get(swarm_id, []))

    def request_approval(
        self,
        action: str,
        target: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        risk_level: RiskLevel = RiskLevel.MEDIUM,
        timeout: Optional[float] = None,
    ) -> ApprovalDecision:
        """Synchronously request human authorization for an action.

        Args:
            action: Action being attempted.
            target: Primary target or resource.
            parameters: Action arguments.
            risk_level: Risk classification.
            timeout: Maximum wait time in seconds.

        Returns:
            ApprovalDecision indicating whether action is approved.
        """
        req_id = f"hitl_{uuid.uuid4().hex[:8]}"
        req_data = {
            "request_id": req_id,
            "action": action,
            "target": target,
            "parameters": dict(parameters or {}),
            "risk_level": risk_level.value,
            "timestamp": time.time(),
        }

        with self._lock:
            self._pending_requests[req_id] = req_data

        if self._event_bus is not None:
            self._event_bus.publish(
                InterventionRequested(
                    request_id=req_id,
                    action=action,
                    risk_level=risk_level.value,
                )
            )

        # If approval handler registered, invoke it
        decision: Optional[ApprovalDecision] = None
        handler = self._approval_handler
        if handler is not None:
            try:
                decision = handler(req_data)
            except Exception as exc:
                logger.error("Approval handler error for '%s': %s", req_id, exc)
                decision = ApprovalDecision(
                    request_id=req_id,
                    approved=False,
                    reason=f"Approval handler error: {exc}",
                )

        if decision is None:
            # By default: LOW risk auto-approves; HIGH/CRITICAL defaults to rejected without handler
            if risk_level == RiskLevel.LOW:
                decision = ApprovalDecision(
                    request_id=req_id,
                    approved=True,
                    decided_by="policy_auto_approve",
                    reason="Low risk operation auto-approved.",
                )
            else:
                decision = ApprovalDecision(
                    request_id=req_id,
                    approved=False,
                    decided_by="hitl_gateway_default",
                    reason=f"{risk_level.value} risk requires interactive approval handler.",
                )

        with self._lock:
            self._pending_requests.pop(req_id, None)
            self._resolved_decisions[req_id] = decision

        if self._event_bus is not None:
            self._event_bus.publish(
                InterventionResolved(
                    request_id=req_id,
                    approved=decision.approved,
                    decided_by=decision.decided_by,
                )
            )

        return decision

    async def request_approval_async(
        self,
        action: str,
        target: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        risk_level: RiskLevel = RiskLevel.MEDIUM,
        timeout: Optional[float] = None,
    ) -> ApprovalDecision:
        """Asynchronously request human authorization for an action.

        Args:
            action: Action being attempted.
            target: Primary target or resource.
            parameters: Action arguments.
            risk_level: Risk classification.
            timeout: Maximum wait time in seconds.

        Returns:
            ApprovalDecision indicating whether action is approved.
        """
        # Run synchronous request in executor or delegate directly
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.request_approval(
                action=action,
                target=target,
                parameters=parameters,
                risk_level=risk_level,
                timeout=timeout,
            ),
        )


class HumanProxyAgent(BaseAgent):
    """Agent acting as a surrogate for human operator decisions within a swarm."""

    def __init__(
        self,
        gateway: InterventionGateway,
        agent_id: str = "human_proxy",
        name: str = "Human Proxy Operator",
    ) -> None:
        """Initialize HumanProxyAgent with injected gateway.

        Args:
            gateway: InterventionGateway instance.
            agent_id: Agent identifier.
            name: Agent display name.
        """
        manifest = AgentManifest(
            agent_id=agent_id,
            name=name,
            role=AgentRole.CUSTOM,
            capabilities=[
                AgentCapability(
                    name="human_intervention",
                    supported_actions={"request_approval", "human_input", "operator_decision"},
                )
            ],
            system_prompt="Forward high-stakes operations to the human operator for authorization.",
        )
        super().__init__(manifest=manifest)
        self.gateway = gateway

    def execute_delegation(self, request: DelegationRequest) -> DelegationResponse:
        """Synchronously request human authorization via gateway."""
        start = time.perf_counter()
        risk_val = request.parameters.get("risk_level", "MEDIUM")
        try:
            risk_level = RiskLevel(risk_val)
        except Exception:
            risk_level = RiskLevel.MEDIUM

        decision = self.gateway.request_approval(
            action=request.action,
            target=request.target,
            parameters=request.parameters,
            risk_level=risk_level,
        )

        elapsed = time.perf_counter() - start
        return DelegationResponse(
            task_id=request.task_id,
            agent_id=self.agent_id,
            success=decision.approved,
            output=decision.to_dict(),
            error=None if decision.approved else (decision.reason or "Denied by human operator."),
            duration=elapsed,
            confidence=1.0 if decision.approved else 0.0,
        )

    async def execute_delegation_async(self, request: DelegationRequest) -> DelegationResponse:
        """Asynchronously request human authorization via gateway."""
        start = time.perf_counter()
        risk_val = request.parameters.get("risk_level", "MEDIUM")
        try:
            risk_level = RiskLevel(risk_val)
        except Exception:
            risk_level = RiskLevel.MEDIUM

        decision = await self.gateway.request_approval_async(
            action=request.action,
            target=request.target,
            parameters=request.parameters,
            risk_level=risk_level,
        )

        elapsed = time.perf_counter() - start
        return DelegationResponse(
            task_id=request.task_id,
            agent_id=self.agent_id,
            success=decision.approved,
            output=decision.to_dict(),
            error=None if decision.approved else (decision.reason or "Denied by human operator."),
            duration=elapsed,
            confidence=1.0 if decision.approved else 0.0,
        )
