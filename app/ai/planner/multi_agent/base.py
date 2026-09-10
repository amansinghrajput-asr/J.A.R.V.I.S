"""Base Agent Definitions and Standard Implementations for Phase 18.0.

Provides BaseAgent abstract interface, WorkerAgent, and CriticAgent.
"""

from __future__ import annotations

import abc
import asyncio
import inspect
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from app.ai.planner.events import AgentStatusChanged, PlannerEventBus, TaskCompletedByAgent
from app.ai.planner.models import Task
from app.ai.planner.multi_agent.models import (
    AgentCapability,
    AgentManifest,
    AgentMessage,
    AgentMessageType,
    AgentRole,
    AgentStatus,
    CriticFeedback,
    DelegationRequest,
    DelegationResponse,
)
from app.ai.planner.multi_agent.protocol import AgentCommunicationBus

logger = logging.getLogger("app.ai.planner.multi_agent.base")


class BaseAgent(abc.ABC):
    """Abstract base class for all autonomous agents in the multi-agent system."""

    def __init__(
        self,
        manifest: AgentManifest,
        communication_bus: Optional[AgentCommunicationBus] = None,
        event_bus: Optional[PlannerEventBus] = None,
    ) -> None:
        """Initialize base agent.

        Args:
            manifest: Static specification and metadata for this agent.
            communication_bus: Optional communication bus for inter-agent messaging.
            event_bus: Optional PlannerEventBus for observability events.
        """
        self._manifest = manifest
        self._communication_bus = communication_bus
        self._event_bus = event_bus
        self._status: AgentStatus = AgentStatus.CREATED
        self._inbox: List[AgentMessage] = []

    @property
    def agent_id(self) -> str:
        """Agent unique identifier."""
        return self._manifest.agent_id

    @property
    def name(self) -> str:
        """Agent display name."""
        return self._manifest.name

    @property
    def role(self) -> AgentRole:
        """Agent role."""
        return self._manifest.role

    @property
    def status(self) -> AgentStatus:
        """Current operational status."""
        return self._status

    @property
    def manifest(self) -> AgentManifest:
        """Agent specification manifest."""
        return self._manifest

    def set_status(self, new_status: AgentStatus) -> None:
        """Transition agent lifecycle status and emit event."""
        if self._status != new_status:
            old_status = self._status
            self._status = new_status
            logger.debug("Agent '%s' status: %s -> %s", self.agent_id, old_status.value, new_status.value)
            if self._event_bus is not None:
                self._event_bus.publish(
                    AgentStatusChanged(
                        agent_id=self.agent_id,
                        old_status=old_status.value,
                        new_status=new_status.value,
                    )
                )

    def initialize(self) -> None:
        """Prepare agent for operation and connect to bus."""
        if self._communication_bus is not None:
            self._communication_bus.register_agent(self.agent_id)
        self.set_status(AgentStatus.IDLE)

    def shutdown(self) -> None:
        """Terminate agent and clean up resources."""
        if self._communication_bus is not None:
            self._communication_bus.unregister_agent(self.agent_id)
        self.set_status(AgentStatus.TERMINATED)

    def send_message(
        self,
        recipient: str,
        message_type: AgentMessageType,
        content: Dict[str, Any],
        correlation_id: str = "",
    ) -> bool:
        """Send a message to another agent or coordinator."""
        if self._communication_bus is None:
            return False
        msg = AgentMessage(
            sender=self.agent_id,
            recipient=recipient,
            message_type=message_type,
            content=content,
            correlation_id=correlation_id,
        )
        return self._communication_bus.send(msg)

    async def send_message_async(
        self,
        recipient: str,
        message_type: AgentMessageType,
        content: Dict[str, Any],
        correlation_id: str = "",
    ) -> bool:
        """Send a message asynchronously to another agent or coordinator."""
        if self._communication_bus is None:
            return False
        msg = AgentMessage(
            sender=self.agent_id,
            recipient=recipient,
            message_type=message_type,
            content=content,
            correlation_id=correlation_id,
        )
        return await self._communication_bus.send_async(msg)

    def receive_message(self, message: AgentMessage) -> None:
        """Receive incoming message from mailbox."""
        self._inbox.append(message)
        logger.debug("Agent '%s' received message from '%s': %s", self.agent_id, message.sender, message.message_type)

    @abc.abstractmethod
    def execute_delegation(self, request: DelegationRequest) -> DelegationResponse:
        """Execute a delegated task synchronously."""
        raise NotImplementedError

    async def execute_delegation_async(self, request: DelegationRequest) -> DelegationResponse:
        """Execute a delegated task asynchronously. Default implementation delegates to thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.execute_delegation, request)


class WorkerAgent(BaseAgent):
    """General-purpose worker agent capable of invoking action callables or tools."""

    def __init__(
        self,
        manifest: AgentManifest,
        handlers: Optional[Dict[str, Callable[[Any], Any]]] = None,
        communication_bus: Optional[AgentCommunicationBus] = None,
        event_bus: Optional[PlannerEventBus] = None,
    ) -> None:
        """Initialize worker agent.

        Args:
            manifest: Specification of the worker agent.
            handlers: Mapping of supported action names to executable functions.
            communication_bus: Inter-agent communication bus.
            event_bus: PlannerEventBus for event publishing.
        """
        super().__init__(manifest, communication_bus=communication_bus, event_bus=event_bus)
        self._handlers: Dict[str, Callable[[Any], Any]] = dict(handlers or {})

    def register_handler(self, action: str, handler: Callable[[Any], Any]) -> None:
        """Register an action handler on this worker."""
        self._handlers[action.lower()] = handler

    def execute_delegation(self, request: DelegationRequest) -> DelegationResponse:
        """Synchronously execute delegated task."""
        self.set_status(AgentStatus.BUSY)
        t_start = time.perf_counter()
        action_key = request.action.lower()

        # Security check: action must be permitted if allowed_actions is configured
        if self._manifest.allowed_actions and request.action not in self._manifest.allowed_actions:
            self.set_status(AgentStatus.IDLE)
            err = f"Action '{request.action}' not permitted for agent '{self.agent_id}'."
            return DelegationResponse(
                task_id=request.task_id,
                agent_id=self.agent_id,
                success=False,
                error=err,
                duration=time.perf_counter() - t_start,
                confidence=0.0,
            )

        handler = self._handlers.get(action_key)
        if handler is None:
            self.set_status(AgentStatus.IDLE)
            err = f"No handler registered on agent '{self.agent_id}' for action '{request.action}'."
            return DelegationResponse(
                task_id=request.task_id,
                agent_id=self.agent_id,
                success=False,
                error=err,
                duration=time.perf_counter() - t_start,
                confidence=0.0,
            )

        try:
            # Build mock or actual task object for handler compatibility
            task_obj = Task(
                id=request.task_id,
                action=request.action,
                target=request.target,
                parameters=request.parameters,
            )

            sig = inspect.signature(handler)
            if len(sig.parameters) == 0:
                res = handler()
            elif len(sig.parameters) == 1:
                res = handler(task_obj)
            else:
                res = handler(task_obj, request.context)

            if inspect.isawaitable(res):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop and loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        res = pool.submit(asyncio.run, res).result()
                else:
                    res = asyncio.run(res)

            duration = time.perf_counter() - t_start
            self.set_status(AgentStatus.IDLE)

            if self._event_bus is not None:
                self._event_bus.publish(
                    TaskCompletedByAgent(
                        task_id=request.task_id,
                        agent_id=self.agent_id,
                        action=request.action,
                        duration=duration,
                        success=True,
                    )
                )

            return DelegationResponse(
                task_id=request.task_id,
                agent_id=self.agent_id,
                success=True,
                output=res,
                duration=duration,
                confidence=1.0,
            )

        except Exception as exc:
            duration = time.perf_counter() - t_start
            self.set_status(AgentStatus.FAILED)
            logger.error("Agent '%s' failed task '%s': %s", self.agent_id, request.task_id, exc)

            if self._event_bus is not None:
                self._event_bus.publish(
                    TaskCompletedByAgent(
                        task_id=request.task_id,
                        agent_id=self.agent_id,
                        action=request.action,
                        duration=duration,
                        success=False,
                    )
                )
            self.set_status(AgentStatus.IDLE)
            return DelegationResponse(
                task_id=request.task_id,
                agent_id=self.agent_id,
                success=False,
                error=str(exc),
                duration=duration,
                confidence=0.0,
            )


class CriticAgent(BaseAgent):
    """Specialized agent performing reflection, plan critique, and artifact validation."""

    def __init__(
        self,
        manifest: Optional[AgentManifest] = None,
        critique_fn: Optional[Callable[[Any, Dict[str, Any]], CriticFeedback]] = None,
        communication_bus: Optional[AgentCommunicationBus] = None,
        event_bus: Optional[PlannerEventBus] = None,
    ) -> None:
        """Initialize CriticAgent."""
        if manifest is None:
            manifest = AgentManifest(
                agent_id="critic_default",
                name="Default Critic Agent",
                role=AgentRole.CRITIC,
                capabilities=[
                    AgentCapability(
                        name="critique",
                        description="Evaluates output quality, completeness, and safety.",
                        supported_actions={"critique", "validate", "review"},
                        domain_tags={"quality", "critique"},
                    )
                ],
            )
        super().__init__(manifest, communication_bus=communication_bus, event_bus=event_bus)
        self._critique_fn = critique_fn

    def critique(self, target_data: Any, context: Optional[Dict[str, Any]] = None) -> CriticFeedback:
        """Evaluate plan or output data and produce constructive critique."""
        ctx = context or {}
        task_id = str(ctx.get("task_id", ""))

        if self._critique_fn is not None:
            return self._critique_fn(target_data, ctx)

        # Default rule-based heuristic critique
        passed = True
        score = 1.0
        suggestions: List[str] = []
        critique_text = "Validation passed with no defects identified."

        if target_data is None:
            passed = False
            score = 0.0
            critique_text = "Target data is None."
            suggestions.append("Ensure task produces valid non-null output.")
        elif isinstance(target_data, str) and not target_data.strip():
            passed = False
            score = 0.2
            critique_text = "Target text output is empty."
            suggestions.append("Provide meaningful output content.")

        return CriticFeedback(
            task_id=task_id,
            critic_id=self.agent_id,
            passed=passed,
            score=score,
            critique=critique_text,
            suggestions=suggestions,
        )

    def execute_delegation(self, request: DelegationRequest) -> DelegationResponse:
        """Execute delegated critique task."""
        t_start = time.perf_counter()
        target = request.parameters.get("target_data")
        if target is None and request.target:
            target = request.target
        if target is None and "blackboard" in request.context:
            bb = request.context["blackboard"]
            keys = bb.list_keys() if hasattr(bb, "list_keys") else []
            if keys:
                target = bb.get(keys[-1])

        fb = self.critique(target if target is not None else "Valid output", request.context)
        duration = time.perf_counter() - t_start

        return DelegationResponse(
            task_id=request.task_id,
            agent_id=self.agent_id,
            success=fb.passed,
            output=fb.to_dict(),
            duration=duration,
            confidence=fb.score,
        )
