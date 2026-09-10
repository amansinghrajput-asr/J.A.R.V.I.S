"""Unit Tests for Agent Registry and Dynamic Agent Selection (Sprint 18.2)."""

import threading
import unittest
from typing import List

from app.ai.planner.events import AgentDeregistered, AgentRegistered, PlannerEvent, PlannerEventBus
from app.ai.planner.multi_agent.base import WorkerAgent
from app.ai.planner.multi_agent.models import (
    AgentCapability,
    AgentManifest,
    AgentRole,
    AgentStatus,
)
from app.ai.planner.multi_agent.registry import AgentRegistry, AgentSelector


class TestAgentRegistry(unittest.TestCase):
    """Test suite for AgentRegistry and AgentSelector."""

    def setUp(self) -> None:
        self.bus = PlannerEventBus()
        self.events: List[PlannerEvent] = []
        self.bus.subscribe(PlannerEvent, self.events.append)
        self.registry = AgentRegistry(event_bus=self.bus)

    def _create_agent(self, aid: str, role: AgentRole, actions: List[str], domains: List[str], confidence: float = 1.0) -> WorkerAgent:
        cap = AgentCapability(
            name=f"cap_{aid}",
            supported_actions=set(actions),
            domain_tags=set(domains),
            confidence_score=confidence,
        )
        manifest = AgentManifest(
            agent_id=aid,
            name=f"Agent {aid}",
            role=role,
            capabilities=[cap],
        )
        return WorkerAgent(manifest=manifest, event_bus=self.bus)

    def test_registration_and_indexing(self) -> None:
        """Verify registration, duplicate checking, and indexing."""
        a1 = self._create_agent("search_1", AgentRole.RESEARCHER, ["web_search"], ["web", "research"])
        self.registry.register(a1)

        self.assertEqual(self.registry.count(), 1)
        self.assertEqual(self.registry.get_agent("search_1"), a1)

        # Duplicate check
        with self.assertRaises(ValueError):
            self.registry.register(a1)

        # Query indices
        by_action = self.registry.find_agents_for_action("web_search")
        self.assertEqual(len(by_action), 1)
        self.assertEqual(by_action[0].agent_id, "search_1")

        by_domain = self.registry.find_agents_by_domain("web")
        self.assertEqual(len(by_domain), 1)

        by_role = self.registry.list_agents(role=AgentRole.RESEARCHER)
        self.assertEqual(len(by_role), 1)

        # Check events
        reg_events = [e for e in self.events if isinstance(e, AgentRegistered)]
        self.assertEqual(len(reg_events), 1)
        self.assertEqual(reg_events[0].agent_id, "search_1")

    def test_unregistration(self) -> None:
        """Verify unregistering removes agent from registry and indices."""
        a1 = self._create_agent("calc_1", AgentRole.WORKER, ["calculate"], ["math"])
        self.registry.register(a1)
        self.assertEqual(self.registry.count(), 1)

        removed = self.registry.unregister("calc_1", reason="maintenance")
        self.assertTrue(removed)
        self.assertEqual(self.registry.count(), 0)
        self.assertIsNone(self.registry.get_agent("calc_1"))
        self.assertEqual(len(self.registry.find_agents_for_action("calculate")), 0)

        # Deregistered event
        dereg_events = [e for e in self.events if isinstance(e, AgentDeregistered)]
        self.assertEqual(len(dereg_events), 1)
        self.assertEqual(dereg_events[0].agent_id, "calc_1")

    def test_capabilities_summary(self) -> None:
        """Verify capabilities summary structure."""
        a1 = self._create_agent("a1", AgentRole.WORKER, ["action_a"], ["domain_x"])
        a2 = self._create_agent("a2", AgentRole.RESEARCHER, ["action_b"], ["domain_y"])
        self.registry.register(a1)
        self.registry.register(a2)

        summary = self.registry.get_capabilities_summary()
        self.assertEqual(summary["total_agents"], 2)
        self.assertIn("action_a", summary["supported_actions"])
        self.assertIn("domain_x", summary["domains"])

    def test_agent_selector_scoring(self) -> None:
        """Verify AgentSelector multi-criteria scoring chooses the optimal agent."""
        a_idle = self._create_agent("agent_idle", AgentRole.WORKER, ["scrape"], ["finance"], confidence=0.9)
        a_idle.set_status(AgentStatus.IDLE)

        a_busy = self._create_agent("agent_busy", AgentRole.WORKER, ["scrape"], ["finance"], confidence=0.95)
        a_busy.set_status(AgentStatus.BUSY)

        self.registry.register(a_idle)
        self.registry.register(a_busy)

        selector = AgentSelector(self.registry)

        # Idle agent should score higher due to availability weighting
        chosen = selector.select_agent("scrape", domain="finance")
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.agent_id, "agent_idle")

    def test_agent_selector_domain_and_role_affinity(self) -> None:
        """Verify selector chooses agent with matching domain and preferred role."""
        general_worker = self._create_agent("gen_worker", AgentRole.WORKER, ["analyze"], ["general"])
        data_researcher = self._create_agent("data_res", AgentRole.RESEARCHER, ["analyze"], ["data_science"])
        general_worker.set_status(AgentStatus.IDLE)
        data_researcher.set_status(AgentStatus.IDLE)

        self.registry.register(general_worker)
        self.registry.register(data_researcher)

        selector = AgentSelector(self.registry)

        # Request with domain "data_science"
        chosen = selector.select_agent("analyze", domain="data_science")
        self.assertEqual(chosen.agent_id, "data_res")

        # Request with preferred role RESEARCHER
        chosen_role = selector.select_agent("analyze", preferred_role=AgentRole.RESEARCHER)
        self.assertEqual(chosen_role.agent_id, "data_res")

    def test_concurrent_registry_thread_safety(self) -> None:
        """Verify thread-safety of registry under concurrent registrations."""
        agents = [self._create_agent(f"agent_{i}", AgentRole.WORKER, [f"action_{i % 5}"], ["test"]) for i in range(50)]

        def worker(ag_list: List[WorkerAgent]) -> None:
            for ag in ag_list:
                self.registry.register(ag)

        threads = [
            threading.Thread(target=worker, args=(agents[0:25],)),
            threading.Thread(target=worker, args=(agents[25:50],)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2.0)

        self.assertEqual(self.registry.count(), 50)


if __name__ == "__main__":
    unittest.main()
