"""Multi-Agent Coordinator and Result Aggregator for Phase 18.0.

Orchestrates multi-agent execution, task delegation, shared memory integration,
and parallel result aggregation.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set

from app.ai.planner.control import ExecutionController
from app.ai.planner.events import PlannerEventBus, TaskDelegated
from app.ai.planner.executor import Executor
from app.ai.planner.models import ExecutionResult, Plan, Task, TaskStatus
from app.ai.planner.multi_agent.base import BaseAgent
from app.ai.planner.multi_agent.conflict import ConflictResolver
from app.ai.planner.multi_agent.critic import ReflectionPipeline
from app.ai.planner.multi_agent.memory import SharedAgentMemory
from app.ai.planner.multi_agent.models import DelegationRequest, DelegationResponse
from app.ai.planner.multi_agent.protocol import AgentCommunicationBus
from app.ai.planner.multi_agent.registry import AgentRegistry, AgentSelector
from app.ai.planner.timeouts import TimeoutConfig

logger = logging.getLogger("app.ai.planner.multi_agent.coordinator")


class ResultAggregator:
    """Collates and synthesizes execution outputs from multiple agents."""

    @staticmethod
    def aggregate(
        completed_tasks: List[Task],
        task_output_map: Dict[str, str],
        shared_memory: Optional[SharedAgentMemory] = None,
    ) -> str:
        """Collate task outputs into a unified, formatted summary string.

        Args:
            completed_tasks: List of successfully completed tasks.
            task_output_map: Mapping of task IDs to their output strings.
            shared_memory: Optional shared memory blackboard.

        Returns:
            Formatted synthesis of task execution results.
        """
        lines: List[str] = []
        for t in completed_tasks:
            out = task_output_map.get(t.id, f"[{t.action}] Completed")
            agent_tag = f" (Agent: {t.parameters.get('assigned_agent')})" if t.parameters.get("assigned_agent") else ""
            lines.append(f"- {out}{agent_tag}")

        if not lines:
            return "No task outputs produced."
        return "\n".join(lines)


class MultiAgentCoordinator:
    """Coordinates specialized autonomous agents to execute complex planned workflows."""

    def __init__(
        self,
        registry: Optional[AgentRegistry] = None,
        communication_bus: Optional[AgentCommunicationBus] = None,
        shared_memory: Optional[SharedAgentMemory] = None,
        event_bus: Optional[PlannerEventBus] = None,
        reflection_pipeline: Optional[ReflectionPipeline] = None,
        conflict_resolver: Optional[ConflictResolver] = None,
        selector: Optional[AgentSelector] = None,
        aggregator: Optional[ResultAggregator] = None,
    ) -> None:
        """Initialize coordinator with collaborating subsystems."""
        self.event_bus = event_bus
        self.registry = registry or AgentRegistry(event_bus=event_bus)
        self.communication_bus = communication_bus or AgentCommunicationBus(event_bus=event_bus)
        self.shared_memory = shared_memory or SharedAgentMemory()
        self.reflection = reflection_pipeline or ReflectionPipeline(event_bus=event_bus)
        self.conflict_resolver = conflict_resolver or ConflictResolver(event_bus=event_bus)
        self.selector = selector or AgentSelector(self.registry)
        self.aggregator = aggregator or ResultAggregator()

    def register_agent(self, agent: BaseAgent) -> None:
        """Convenience method to register an agent and connect it to bus and registry."""
        agent.initialize()
        self.registry.register(agent)

    def select_agent_for_task(self, task: Task) -> Optional[BaseAgent]:
        """Choose the optimal agent for a task, respecting explicit assignments or capability scoring."""
        # 1. Explicitly assigned agent
        assigned_id = task.parameters.get("assigned_agent") or getattr(task, "assigned_agent", None)
        if assigned_id:
            ag = self.registry.get_agent(str(assigned_id))
            if ag is not None:
                return ag

        # 2. Dynamic selection
        domain = str(task.parameters.get("domain", "")) or None
        return self.selector.select_agent(action=task.action, target=task.target, domain=domain)

    def delegate_task(
        self,
        task: Task,
        agent: Optional[BaseAgent] = None,
        context: Optional[Dict[str, Any]] = None,
        plan_id: str = "",
    ) -> DelegationResponse:
        """Delegate an individual task to an assigned or selected agent.

        Args:
            task: Task to execute.
            agent: Optional target agent override.
            context: Contextual runtime dictionary.
            plan_id: Correlation plan ID.

        Returns:
            DelegationResponse detailing outcome and artifacts.
        """
        target_agent = agent or self.select_agent_for_task(task)
        if target_agent is None:
            err = f"No agent available capable of executing action '{task.action}'."
            return DelegationResponse(
                task_id=task.id,
                agent_id="none",
                success=False,
                error=err,
            )

        # Mark assignment
        task.parameters["assigned_agent"] = target_agent.agent_id
        if hasattr(task, "assigned_agent"):
            setattr(task, "assigned_agent", target_agent.agent_id)

        if self.event_bus is not None:
            self.event_bus.publish(
                TaskDelegated(
                    plan_id=plan_id,
                    task_id=task.id,
                    agent_id=target_agent.agent_id,
                    action=task.action,
                )
            )

        # Build execution context including relevant shared memory
        ctx = dict(context or {})
        ctx["blackboard"] = self.shared_memory
        ctx["plan_id"] = plan_id

        # Execute with optional reflection refinement
        success, output, critiques = self.reflection.refine_task(
            task=task,
            agent=target_agent,
            plan_id=plan_id,
            context=ctx,
        )

        # Record output in shared memory
        self.shared_memory.set_task_data(task.id, "output", output)
        self.shared_memory.set(f"task:{task.id}:output", output)

        return DelegationResponse(
            task_id=task.id,
            agent_id=target_agent.agent_id,
            success=success,
            output=output,
            error=critiques[-1].critique if (critiques and not success) else None,
        )

    def execute_plan(
        self,
        plan: Plan,
        executor: Optional[Executor] = None,
        controller: Optional[ExecutionController] = None,
        timeout_config: Optional[TimeoutConfig] = None,
    ) -> ExecutionResult:
        """Execute a plan using multi-agent task delegation integrated into Executor DAG waves.

        Args:
            plan: The planned task DAG.
            executor: Optional Executor instance. If None, creates a coordinated executor.
            controller: Optional ExecutionController for pause/resume/cancellation.
            timeout_config: Optional TimeoutConfig.

        Returns:
            ExecutionResult summary with aggregated outputs.
        """
        # 1. Pre-execution plan critique
        critique = self.reflection.critique_plan(plan)
        if not critique.passed:
            logger.warning("Pre-execution plan critique failed: %s", critique.critique)

        # 2. Build multi-agent action handlers adapter
        exec_instance = executor or Executor(
            planner_event_bus=self.event_bus,
            controller=controller,
            timeout_config=timeout_config,
        )

        # Route actions through coordinator delegation handler
        for t in plan.tasks:
            action_key = t.action.strip().lower()
            if not exec_instance.get_handler(action_key):
                # Dynamically bind handler that delegates to the coordinator
                def make_handler(act: str) -> Callable[[Task], Any]:
                    def _handler(task: Task) -> Any:
                        resp = self.delegate_task(task, plan_id=plan.id)
                        if not resp.success:
                            raise RuntimeError(resp.error or f"Agent '{resp.agent_id}' failed action '{act}'")
                        return resp.output
                    return _handler

                exec_instance.register_handler(action_key, make_handler(action_key))

        # 3. Execute plan using underlying DAG executor
        result = exec_instance.execute_plan(
            plan,
            controller=controller,
            timeout_config=timeout_config,
        )

        # 4. Synthesize final output with result aggregator
        if result.completed_tasks:
            task_out_map = {t.id: str(result.output) for t in result.completed_tasks}
            synth_output = self.aggregator.aggregate(
                result.completed_tasks,
                task_out_map,
                shared_memory=self.shared_memory,
            )
            result.output = synth_output

        return result
