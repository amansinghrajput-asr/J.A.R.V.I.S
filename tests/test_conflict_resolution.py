"""Unit Tests for Conflict Resolver (Sprint 18.4)."""

import unittest
from typing import List

from app.ai.planner.events import ConflictDetected, ConflictResolved, PlannerEvent, PlannerEventBus
from app.ai.planner.multi_agent.conflict import ConflictResolver
from app.ai.planner.multi_agent.models import AgentRole


class TestConflictResolution(unittest.TestCase):
    """Test suite for ConflictResolver strategies."""

    def setUp(self) -> None:
        self.bus = PlannerEventBus()
        self.events: List[PlannerEvent] = []
        self.bus.subscribe(PlannerEvent, self.events.append)
        self.resolver = ConflictResolver(event_bus=self.bus)

    def test_confidence_weighted_resolution(self) -> None:
        """Verify candidate with highest confidence is selected."""
        candidates = [
            {"agent_id": "agent_a", "output": "answer A", "confidence": 0.70},
            {"agent_id": "agent_b", "output": "answer B", "confidence": 0.95},
            {"agent_id": "agent_c", "output": "answer C", "confidence": 0.85},
        ]
        res = self.resolver.resolve("FACT_DISAGREEMENT", candidates, strategy="CONFIDENCE_WEIGHTED")
        self.assertTrue(res.resolved)
        self.assertEqual(res.winner_agent_id, "agent_b")
        self.assertEqual(res.resolution_data, "answer B")

        # Verify events
        detected = [e for e in self.events if isinstance(e, ConflictDetected)]
        resolved = [e for e in self.events if isinstance(e, ConflictResolved)]
        self.assertEqual(len(detected), 1)
        self.assertEqual(len(resolved), 1)

    def test_priority_resolution(self) -> None:
        """Verify candidate with highest role priority wins."""
        candidates = [
            {"agent_id": "worker_1", "role": AgentRole.WORKER, "output": "worker result", "confidence": 0.99},
            {"agent_id": "critic_1", "role": AgentRole.CRITIC, "output": "critic override", "confidence": 0.80},
        ]
        res = self.resolver.resolve("DECISION_OVERRIDE", candidates, strategy="PRIORITY")
        self.assertTrue(res.resolved)
        self.assertEqual(res.winner_agent_id, "critic_1")
        self.assertEqual(res.resolution_data, "critic override")

    def test_voting_consensus_resolution(self) -> None:
        """Verify majority voting consensus resolves conflicting votes."""
        candidates = [
            {"agent_id": "agent_1", "output": "YES", "confidence": 0.8},
            {"agent_id": "agent_2", "output": "NO", "confidence": 0.9},
            {"agent_id": "agent_3", "output": "YES", "confidence": 0.7},
        ]
        res = self.resolver.resolve("VOTE_CONCURRENCE", candidates, strategy="VOTING")
        self.assertTrue(res.resolved)
        self.assertEqual(res.resolution_data, "YES")

    def test_adjudication_strategy(self) -> None:
        """Verify custom adjudicator synthesis callable."""
        candidates = [
            {"agent_id": "researcher_1", "output": "Paris is sunny", "confidence": 0.8},
            {"agent_id": "researcher_2", "output": "Paris is 22C", "confidence": 0.85},
        ]

        def custom_adjudicator(cands: List[dict]) -> dict:
            return {
                "agent_id": "coordinator_synthesis",
                "output": "Paris is sunny and 22C",
            }

        res = self.resolver.resolve("SYNTHESIS_NEED", candidates, strategy="ADJUDICATION", adjudicator_fn=custom_adjudicator)
        self.assertTrue(res.resolved)
        self.assertEqual(res.winner_agent_id, "coordinator_synthesis")
        self.assertEqual(res.resolution_data, "Paris is sunny and 22C")

    def test_single_and_empty_candidates(self) -> None:
        """Verify edge cases with single or empty candidates."""
        res_empty = self.resolver.resolve("NO_DATA", [])
        self.assertFalse(res_empty.resolved)

        res_single = self.resolver.resolve("SOLO", [{"agent_id": "lonely", "output": "solo output"}])
        self.assertTrue(res_single.resolved)
        self.assertEqual(res_single.winner_agent_id, "lonely")


if __name__ == "__main__":
    unittest.main()
