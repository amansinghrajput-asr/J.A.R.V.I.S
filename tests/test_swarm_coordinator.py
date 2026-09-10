"""Unit tests for Hierarchical Swarm Coordinator (Sprint 19.5).

Covers master goal orchestration, hierarchical sub-swarm delegation,
3-tier automated fallback (Hierarchical -> Flat Multi-Agent -> Adaptive Planner),
async execution, episodic learning integration, and security guardrail enforcement.
"""

import asyncio
import unittest

from app.ai.planner.events import PlannerEventBus
from app.ai.planner.multi_agent.base import WorkerAgent
from app.ai.planner.multi_agent.models import AgentCapability, AgentManifest, AgentRole
from app.ai.planner.swarm.coordinator import HierarchicalCoordinator
from app.ai.planner.swarm.episodic import EpisodicMemoryStore
from app.ai.planner.swarm.experience import ExperienceSynthesizer
from app.ai.planner.swarm.policy import SwarmPolicyEngine


def _create_worker(agent_id: str, actions: set[str]) -> WorkerAgent:
    manifest = AgentManifest(
        agent_id=agent_id,
        role=AgentRole.WORKER,
        capabilities=[
            AgentCapability(
                name=f"cap_{agent_id}",
                supported_actions=actions,
            )
        ],
    )
    handlers = {act: (lambda t, a=act: f"{a} executed") for act in actions}
    return WorkerAgent(manifest=manifest, handlers=handlers)


class TestSwarmCoordinator(unittest.TestCase):
    """Test suite for HierarchicalCoordinator master orchestration and fallbacks."""

    def test_tier_1_hierarchical_swarm_execution(self) -> None:
        """Verify successful goal execution through Hierarchical Sub-Swarm delegation."""
        event_bus = PlannerEventBus()
        synth = ExperienceSynthesizer()
        mem = EpisodicMemoryStore(synthesizer=synth)

        coordinator = HierarchicalCoordinator(
            event_bus=event_bus,
            episodic_memory=mem,
            experience_synthesizer=synth,
        )

        # Register workers in root coordinator registry
        w_research = _create_worker("worker_res", {"research"})
        w_synth = _create_worker("worker_syn", {"synthesize"})
        coordinator.registry.register(w_research)
        coordinator.registry.register(w_synth)

        res = coordinator.execute_goal("Research electric vehicles market trend")
        self.assertTrue(res.success)
        self.assertGreater(res.duration, 0.0)

        # Verify episodic memory recorded the trajectory
        similar = mem.query_similar_goals("electric vehicles")
        self.assertEqual(len(similar), 1)
        self.assertTrue(similar[0].success)

    def test_tier_2_flat_multi_agent_fallback(self) -> None:
        """Verify fallback to Flat Multi-Agent coordinator if hierarchical execution is disabled/fails."""
        coordinator = HierarchicalCoordinator()

        # Intentionally break Tier 1 by setting max_depth=0 so child sub-swarm cannot spawn
        coordinator.policy_engine.max_depth = 0

        w_open = _create_worker("worker_open", {"open_app"})
        coordinator.registry.register(w_open)

        res = coordinator.execute_goal("open chrome")
        self.assertTrue(res.success)
        self.assertIn("open_app", str(res.outputs))

    def test_tier_3_adaptive_planner_fallback(self) -> None:
        """Verify fallback to Adaptive Planner if no multi-agent worker is available."""
        coordinator = HierarchicalCoordinator()

        # Set max_depth=0 and clear registry of agents
        coordinator.policy_engine.max_depth = 0

        # With empty agent registry, Tier 2 will fail to resolve specialized agents, falling back to Tier 3
        res = coordinator.execute_goal("open chrome")
        self.assertTrue(res.success)
        self.assertIn("result", res.outputs)

    def test_async_goal_execution(self) -> None:
        """Verify asynchronous goal execution."""
        async def _run_async_test() -> None:
            coordinator = HierarchicalCoordinator()
            w_research = _create_worker("w_async_res", {"research"})
            w_synth = _create_worker("w_async_syn", {"synthesize"})
            coordinator.registry.register(w_research)
            coordinator.registry.register(w_synth)

            res = await coordinator.execute_goal_async("Research quantum computing advances")
            self.assertTrue(res.success)

        asyncio.run(_run_async_test())

    def test_policy_engine_blocks_dangerous_goal(self) -> None:
        """Verify policy engine immediately blocks forbidden actions before dispatch."""
        coordinator = HierarchicalCoordinator()

        res = coordinator.execute_goal("run rm -rf /root/system")
        self.assertFalse(res.success)
        self.assertIn("Policy violation", res.outputs.get("error", ""))


if __name__ == "__main__":
    unittest.main()
