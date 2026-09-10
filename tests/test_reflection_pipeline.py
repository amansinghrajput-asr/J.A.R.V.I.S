"""Unit Tests for Reflection and Critique Pipeline (Sprint 18.4)."""

import unittest
from typing import Any, Dict, List

from app.ai.planner.events import AgentCritiqueSubmitted, PlannerEvent, PlannerEventBus
from app.ai.planner.models import Plan, Task
from app.ai.planner.multi_agent.base import CriticAgent, WorkerAgent
from app.ai.planner.multi_agent.critic import ReflectionPipeline
from app.ai.planner.multi_agent.models import AgentCapability, AgentManifest, CriticFeedback


class TestReflectionPipeline(unittest.TestCase):
    """Test suite for ReflectionPipeline and self-critique loops."""

    def setUp(self) -> None:
        self.bus = PlannerEventBus()
        self.events: List[PlannerEvent] = []
        self.bus.subscribe(PlannerEvent, self.events.append)
        self.pipeline = ReflectionPipeline(event_bus=self.bus)

    def test_critique_plan_valid_and_invalid(self) -> None:
        """Verify pre-execution plan validation."""
        # Empty plan
        empty_plan = Plan(query="empty")
        fb_empty = self.pipeline.critique_plan(empty_plan)
        self.assertFalse(fb_empty.passed)
        self.assertEqual(fb_empty.score, 0.0)

        # Plan with invalid dependency
        bad_plan = Plan(query="bad deps")
        bad_plan.tasks = [Task(id="t1", action="step1", dependencies=["non_existent_id"])]
        fb_bad = self.pipeline.critique_plan(bad_plan)
        self.assertFalse(fb_bad.passed)
        self.assertTrue(any("non-existent" in s for s in fb_bad.suggestions))

        # Valid plan
        good_plan = Plan(query="good plan")
        t1 = Task(id="t1", action="step1")
        t2 = Task(id="t2", action="step2", dependencies=["t1"])
        good_plan.tasks = [t1, t2]
        fb_good = self.pipeline.critique_plan(good_plan)
        self.assertTrue(fb_good.passed)
        self.assertEqual(fb_good.score, 1.0)

        # Check events
        crit_events = [e for e in self.events if isinstance(e, AgentCritiqueSubmitted)]
        self.assertEqual(len(crit_events), 3)

    def test_critique_task_result(self) -> None:
        """Verify task output evaluation."""
        task = Task(id="t_calc", action="calculate")

        fb_none = self.pipeline.critique_task_result(task, None)
        self.assertFalse(fb_none.passed)

        fb_empty = self.pipeline.critique_task_result(task, "   ")
        self.assertFalse(fb_empty.passed)

        fb_ok = self.pipeline.critique_task_result(task, "Result: 42")
        self.assertTrue(fb_ok.passed)

    def test_iterative_refinement_loop(self) -> None:
        """Verify refine_task iterates and incorporates critique feedback."""
        attempt_counter = 0

        def adaptive_handler(task: Task, context: Dict[str, Any]) -> str:
            nonlocal attempt_counter
            attempt_counter += 1
            if attempt_counter == 1:
                # First attempt returns empty or flawed output
                return ""
            # Second attempt receives critique suggestion and fixes it
            return "Refined and complete response"

        manifest = AgentManifest(
            agent_id="adaptive_worker",
            name="Adaptive Worker",
            capabilities=[AgentCapability(name="work", supported_actions={"work"})],
        )
        worker = WorkerAgent(manifest=manifest, handlers={"work": adaptive_handler})
        worker.initialize()

        task = Task(id="task_refine", action="work")
        success, final_output, critiques = self.pipeline.refine_task(task, worker, max_rounds=2)

        self.assertTrue(success)
        self.assertEqual(final_output, "Refined and complete response")
        self.assertEqual(len(critiques), 2)
        self.assertFalse(critiques[0].passed)
        self.assertTrue(critiques[1].passed)


if __name__ == "__main__":
    unittest.main()
