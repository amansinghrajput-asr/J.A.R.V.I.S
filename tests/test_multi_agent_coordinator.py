"""Integration Tests for Multi-Agent Coordinator (Sprint 18.5)."""

import unittest
from typing import Any, Dict

from app.ai.planner.control import ExecutionController
from app.ai.planner.events import PlannerEventBus, TaskDelegated
from app.ai.planner.models import Plan, Task, TaskStatus
from app.ai.planner.multi_agent.base import CriticAgent, WorkerAgent
from app.ai.planner.multi_agent.coordinator import MultiAgentCoordinator, ResultAggregator
from app.ai.planner.multi_agent.models import AgentCapability, AgentManifest, AgentRole
from app.ai.planner.multi_agent.registry import AgentRegistry


class TestMultiAgentCoordinator(unittest.TestCase):
    """Test suite for MultiAgentCoordinator DAG execution and result aggregation."""

    def setUp(self) -> None:
        self.bus = PlannerEventBus()
        self.events = []
        self.bus.subscribe(TaskDelegated, self.events.append)
        self.coordinator = MultiAgentCoordinator(event_bus=self.bus)

    def test_task_delegation_dynamic_and_explicit(self) -> None:
        """Verify dynamic selection vs explicit assignment in delegation."""
        # Register researcher agent
        manifest_res = AgentManifest(
            agent_id="researcher_web",
            name="Web Researcher",
            role=AgentRole.RESEARCHER,
            capabilities=[AgentCapability(name="search", supported_actions={"web_search"})],
        )
        agent_res = WorkerAgent(manifest=manifest_res, handlers={"web_search": lambda t: f"Found {t.target}"})
        self.coordinator.register_agent(agent_res)

        # 1. Dynamic selection for web_search
        t1 = Task(action="web_search", target="Python 3.13")
        resp1 = self.coordinator.delegate_task(t1)
        self.assertTrue(resp1.success)
        self.assertEqual(resp1.agent_id, "researcher_web")
        self.assertEqual(resp1.output, "Found Python 3.13")
        self.assertEqual(t1.parameters.get("assigned_agent"), "researcher_web")

        # 2. Explicit assignment override
        manifest_custom = AgentManifest(
            agent_id="custom_specialist",
            name="Specialist",
            capabilities=[AgentCapability(name="search", supported_actions={"web_search"})],
        )
        agent_custom = WorkerAgent(manifest=manifest_custom, handlers={"web_search": lambda t: "Specialized finding"})
        self.coordinator.register_agent(agent_custom)

        t2 = Task(action="web_search", target="AI", assigned_agent="custom_specialist")
        resp2 = self.coordinator.delegate_task(t2)
        self.assertTrue(resp2.success)
        self.assertEqual(resp2.agent_id, "custom_specialist")
        self.assertEqual(resp2.output, "Specialized finding")

    def test_shared_memory_update_on_delegation(self) -> None:
        """Verify task output is recorded in shared blackboard on completion."""
        manifest = AgentManifest(
            agent_id="math_worker",
            capabilities=[AgentCapability(name="calc", supported_actions={"calculate"})],
        )
        agent = WorkerAgent(manifest=manifest, handlers={"calculate": lambda t: 100})
        self.coordinator.register_agent(agent)

        task = Task(id="t_calc_blackboard", action="calculate")
        resp = self.coordinator.delegate_task(task)
        self.assertTrue(resp.success)

        # Check blackboard
        stored_out = self.coordinator.shared_memory.get_task_data("t_calc_blackboard", "output")
        self.assertEqual(stored_out, 100)
        global_out = self.coordinator.shared_memory.get("task:t_calc_blackboard:output")
        self.assertEqual(global_out, 100)

    def test_multi_agent_dag_plan_execution(self) -> None:
        """Verify full multi-agent plan execution with wave dependencies."""
        # 1. Setup specialized agents
        researcher = WorkerAgent(
            manifest=AgentManifest(
                agent_id="agent_research",
                role=AgentRole.RESEARCHER,
                capabilities=[AgentCapability(name="fetch", supported_actions={"fetch_data"})],
            ),
            handlers={"fetch_data": lambda t: "Raw market data: AAPL 250"},
        )
        processor = WorkerAgent(
            manifest=AgentManifest(
                agent_id="agent_process",
                role=AgentRole.WORKER,
                capabilities=[AgentCapability(name="process", supported_actions={"process_data"})],
            ),
            handlers={"process_data": lambda t: "Processed: Bullish trend detected"},
        )
        critic = CriticAgent(
            manifest=AgentManifest(
                agent_id="agent_critic",
                role=AgentRole.CRITIC,
                capabilities=[AgentCapability(name="review", supported_actions={"review_report"})],
            )
        )

        self.coordinator.register_agent(researcher)
        self.coordinator.register_agent(processor)
        self.coordinator.register_agent(critic)

        # 2. Build 3-stage DAG Plan
        t1 = Task(id="t1", action="fetch_data")
        t2 = Task(id="t2", action="process_data", dependencies=["t1"])
        t3 = Task(id="t3", action="review_report", dependencies=["t2"])
        plan = Plan(query="Financial trend report", tasks=[t1, t2, t3])

        # 3. Execute
        result = self.coordinator.execute_plan(plan)

        self.assertTrue(result.success)
        self.assertEqual(len(result.completed_tasks), 3)
        self.assertEqual(t1.status, TaskStatus.COMPLETED)
        self.assertEqual(t2.status, TaskStatus.COMPLETED)
        self.assertEqual(t3.status, TaskStatus.COMPLETED)

        # 4. Verify aggregated output
        self.assertIn("fetch_data", result.output)
        self.assertIn("agent_research", result.output)
        self.assertIn("process_data", result.output)
        self.assertIn("agent_process", result.output)

    def test_multi_agent_execution_with_controller_cancellation(self) -> None:
        """Verify cancellation gracefully cancels multi-agent execution."""
        ctrl = ExecutionController()

        manifest = AgentManifest(
            agent_id="slow_agent",
            capabilities=[AgentCapability(name="slow", supported_actions={"slow_step"})],
        )

        def handle_step(t: Task) -> str:
            ctrl.cancel("Cancelled by user")
            return "done"

        agent = WorkerAgent(manifest=manifest, handlers={"slow_step": handle_step})
        self.coordinator.register_agent(agent)

        t1 = Task(id="t1", action="slow_step")
        t2 = Task(id="t2", action="slow_step", dependencies=["t1"])
        plan = Plan(query="cancelled plan", tasks=[t1, t2])

        result = self.coordinator.execute_plan(plan, controller=ctrl)

        self.assertFalse(result.success)
        self.assertEqual(t1.status, TaskStatus.COMPLETED)
        self.assertEqual(t2.status, TaskStatus.CANCELLED)
        self.assertEqual(len(result.completed_tasks), 1)


if __name__ == "__main__":
    unittest.main()
