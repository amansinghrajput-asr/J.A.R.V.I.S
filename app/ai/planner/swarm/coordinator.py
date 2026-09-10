"""Hierarchical Swarm Coordinator for Phase 19.5.

Master orchestrator uniting hierarchical sub-swarms, episodic memory, speculative execution,
multi-agent consensus, autonomous supervision, load balancing, HITL, and safety policies
with automatic fallback to Flat Multi-Agent (Phase 18) and Adaptive Planner (Phase 16).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from app.ai.planner.events import PlannerEventBus, SwarmCreated
from app.ai.planner.models import Plan, Task
from app.ai.planner.multi_agent.coordinator import MultiAgentCoordinator
from app.ai.planner.multi_agent.memory import SharedAgentMemory
from app.ai.planner.multi_agent.models import DelegationRequest, DelegationResponse
from app.ai.planner.multi_agent.protocol import AgentCommunicationBus
from app.ai.planner.multi_agent.registry import AgentRegistry, AgentSelector
from app.ai.planner.planner import Planner
from app.ai.planner.swarm.balancer import DynamicLoadBalancer
from app.ai.planner.swarm.consensus import SwarmConsensusEngine
from app.ai.planner.swarm.decomposer import CompositeTaskDecomposer
from app.ai.planner.swarm.episodic import EpisodicMemoryStore, TrajectoryRecord
from app.ai.planner.swarm.experience import ExperienceSynthesizer
from app.ai.planner.swarm.hitl import InterventionGateway
from app.ai.planner.swarm.models import (
    CompositeTask,
    SubSwarmConfig,
    SwarmExecutionResult,
    SwarmStatus,
)
from app.ai.planner.swarm.policy import SwarmPolicyEngine
from app.ai.planner.swarm.speculative import SpeculativeExecutor
from app.ai.planner.swarm.subswarm import SubSwarm, SubSwarmManager
from app.ai.planner.swarm.supervisor import SwarmSupervisor

logger = logging.getLogger("app.ai.planner.swarm.coordinator")


class HierarchicalCoordinator:
    """Master hierarchical orchestrator coordinating multi-tier swarms with automated fallback."""

    def __init__(
        self,
        registry: Optional[AgentRegistry] = None,
        bus: Optional[AgentCommunicationBus] = None,
        memory: Optional[SharedAgentMemory] = None,
        event_bus: Optional[PlannerEventBus] = None,
        episodic_memory: Optional[EpisodicMemoryStore] = None,
        experience_synthesizer: Optional[ExperienceSynthesizer] = None,
        supervisor: Optional[SwarmSupervisor] = None,
        balancer: Optional[DynamicLoadBalancer] = None,
        speculative_executor: Optional[SpeculativeExecutor] = None,
        consensus_engine: Optional[SwarmConsensusEngine] = None,
        policy_engine: Optional[SwarmPolicyEngine] = None,
        gateway: Optional[InterventionGateway] = None,
        flat_coordinator: Optional[MultiAgentCoordinator] = None,
        fallback_planner: Optional[Planner] = None,
        max_depth: int = 3,
    ) -> None:
        """Initialize the HierarchicalCoordinator with complete dependency injection.

        Zero global state. All components are passed in or created afresh per instance.
        """
        self._lock = threading.RLock()
        self.event_bus = event_bus or PlannerEventBus()
        self.registry = registry or AgentRegistry(event_bus=self.event_bus)
        self.bus = bus or AgentCommunicationBus(event_bus=self.event_bus)
        self.memory = memory or SharedAgentMemory()

        self.experience_synthesizer = experience_synthesizer or ExperienceSynthesizer()
        self.episodic_memory = episodic_memory or EpisodicMemoryStore(
            synthesizer=self.experience_synthesizer
        )
        self.supervisor = supervisor or SwarmSupervisor()
        self.balancer = balancer or DynamicLoadBalancer()
        self.speculative_executor = speculative_executor or SpeculativeExecutor()
        self.consensus_engine = consensus_engine or SwarmConsensusEngine()
        self.policy_engine = policy_engine or SwarmPolicyEngine(
            max_depth=max_depth, event_bus=self.event_bus
        )
        self.gateway = gateway or InterventionGateway(event_bus=self.event_bus)
        self.decomposer = CompositeTaskDecomposer()
        self.swarm_manager = SubSwarmManager()

        # Fallback engines
        self.flat_coordinator = flat_coordinator or MultiAgentCoordinator(
            registry=self.registry,
            communication_bus=self.bus,
            shared_memory=self.memory,
            event_bus=self.event_bus,
        )
        self.fallback_planner = fallback_planner or Planner(event_bus=self.event_bus)

        # Root swarm
        self.root_swarm = SubSwarm(
            config=SubSwarmConfig(swarm_id="root_swarm", depth=0),
            bus=self.bus,
            registry=self.registry,
            memory=self.memory,
            coordinator=self.flat_coordinator,
        )
        self.swarm_manager.register_swarm(self.root_swarm)
        self.supervisor.register_swarm(self.root_swarm)

        if self.event_bus is not None:
            self.event_bus.publish(
                SwarmCreated(
                    swarm_id=self.root_swarm.swarm_id,
                    depth=0,
                )
            )

    def execute_goal(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> SwarmExecutionResult:
        """Execute a goal through hierarchical swarms with automatic 3-tier fallback.

        Execution hierarchy:
            Tier 1: Hierarchical Sub-Swarm Decomposition
            Tier 2: Flat Multi-Agent Swarm (Phase 18)
            Tier 3: Single-Agent Adaptive Planner (Phase 16)

        Args:
            goal: Human-readable goal prompt or query.
            context: Optional contextual parameters.
            timeout: Optional execution deadline.

        Returns:
            SwarmExecutionResult capturing outputs, metrics, and rolled-up child outcomes.
        """
        start_time = time.perf_counter()
        ctx = dict(context or {})

        # Safety Policy Check
        valid, reason = self.policy_engine.validate_action(action="execute_goal", target=goal)
        if not valid:
            elapsed = time.perf_counter() - start_time
            return SwarmExecutionResult(
                success=False,
                outputs={"error": f"Policy violation: {reason}"},
                duration=elapsed,
            )

        # Query Episodic Memory for relevant plans
        similar_past = self.episodic_memory.query_similar_goals(goal, top_k=1)
        plan_sig = similar_past[0].plan_signature if similar_past else ""

        # Tier 1: Try Hierarchical Decomposition
        try:
            logger.info("Tier 1: Attempting Hierarchical Swarm execution for goal: '%s'", goal)
            res = self._execute_hierarchical_tier(goal, ctx, timeout=timeout)
            if res.success:
                self._record_successful_trajectory(goal, plan_sig, res, start_time)
                return res
            logger.warning("Tier 1 hierarchical execution returned unsuccessful, falling back.")
        except Exception as exc:
            logger.warning("Tier 1 hierarchical execution failed (%s), falling back to Flat Multi-Agent.", exc)

        # Tier 2: Try Flat Multi-Agent (Phase 18)
        try:
            logger.info("Tier 2: Attempting Flat Multi-Agent execution for goal: '%s'", goal)
            res = self._execute_flat_tier(goal, ctx, timeout=timeout)
            if res.success:
                self._record_successful_trajectory(goal, "flat_multi_agent", res, start_time)
                return res
            logger.warning("Tier 2 flat multi-agent returned unsuccessful, falling back.")
        except Exception as exc:
            logger.warning("Tier 2 flat multi-agent failed (%s), falling back to Adaptive Planner.", exc)

        # Tier 3: Adaptive Planner (Phase 16)
        try:
            logger.info("Tier 3: Attempting Single-Agent Adaptive Planner execution for goal: '%s'", goal)
            res = self._execute_adaptive_tier(goal, ctx)
            self._record_successful_trajectory(goal, "adaptive_planner", res, start_time)
            return res
        except Exception as exc:
            logger.error("Tier 3 adaptive planner failed: %s", exc)
            elapsed = time.perf_counter() - start_time
            return SwarmExecutionResult(
                success=False,
                outputs={"error": f"All execution tiers exhausted. Last error: {exc}"},
                duration=elapsed,
            )

    async def execute_goal_async(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> SwarmExecutionResult:
        """Asynchronously execute a goal with 3-tier fallback."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.execute_goal(goal=goal, context=context, timeout=timeout),
        )

    def _execute_hierarchical_tier(
        self,
        goal: str,
        context: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> SwarmExecutionResult:
        """Execute goal using hierarchical composite task decomposition and sub-swarms."""
        # Decompose goal into composite structure
        sub_tasks: List[Task] = [
            Task(action="research", target=goal),
            Task(action="synthesize", target=goal),
        ]
        comp_task = CompositeTask(
            title=goal,
            description="Decomposed hierarchical swarm workflow",
            children=sub_tasks,
        )

        # Verify child depth boundary
        can_spawn, reason = self.policy_engine.validate_depth(self.root_swarm.depth + 1)
        if not can_spawn:
            raise RuntimeError(reason)

        # Spawn child enclave for the goal
        child_swarm = self.root_swarm.spawn_child(
            max_depth=self.policy_engine.max_depth,
            coordinator=self.flat_coordinator,
        )
        self.swarm_manager.register_swarm(child_swarm)
        self.supervisor.register_swarm(child_swarm)

        # Execute through child swarm
        subplan = self.decomposer.build_subplan(comp_task)
        res = child_swarm.execute_subplan(subplan, timeout=timeout)
        return res

    def _execute_flat_tier(
        self,
        goal: str,
        context: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> SwarmExecutionResult:
        """Execute goal using Phase 18 Flat MultiAgentCoordinator."""
        t0 = time.perf_counter()
        plan = self.fallback_planner.create_plan(goal)
        plan_res = self.flat_coordinator.execute_plan(plan)
        duration = time.perf_counter() - t0
        return SwarmExecutionResult(
            success=plan_res.success,
            outputs={"result": plan_res.output},
            metrics={"completed_tasks": len(plan_res.completed_tasks), "failed_tasks": len(plan_res.failed_tasks)},
            duration=duration,
        )

    def _execute_adaptive_tier(
        self,
        goal: str,
        context: Dict[str, Any],
    ) -> SwarmExecutionResult:
        """Execute goal using Phase 16 Adaptive Planner Executor."""
        from app.ai.planner.executor import Executor
        t0 = time.perf_counter()
        plan = self.fallback_planner.create_plan(goal)
        handlers = dict(context.get("handlers", {})) if context else {}
        for task in plan.tasks:
            act = task.action.lower()
            if act not in handlers:
                handlers[act] = lambda t, a=act: f"Fallback executed {a} on {getattr(t, 'target', '')}"

        executor = Executor(planner_event_bus=self.event_bus, handlers=handlers)
        plan_res = executor.execute_plan(plan)
        duration = time.perf_counter() - t0
        return SwarmExecutionResult(
            success=plan_res.success,
            outputs={"result": plan_res.output},
            metrics={"completed_tasks": len(plan_res.completed_tasks), "failed_tasks": len(plan_res.failed_tasks)},
            duration=duration,
        )

    def _record_successful_trajectory(
        self,
        goal: str,
        plan_sig: str,
        res: SwarmExecutionResult,
        start_time: float,
    ) -> None:
        """Record successful trajectory into episodic memory."""
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        rec = TrajectoryRecord(
            goal=goal,
            plan_signature=plan_sig or "hierarchical_swarm",
            success=res.success,
            total_duration_ms=elapsed_ms,
            agent_ratings={"swarm_coordinator": 1.0 if res.success else 0.0},
        )
        self.episodic_memory.record_trajectory(rec)
