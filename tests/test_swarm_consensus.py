"""Unit tests for Multi-Agent Consensus Engine (Sprint 19.3).

Covers weighted quorum, strict majority, borda count, tie-breaking,
invalid proposal rejection, and critic score/confidence aggregation.
"""

import unittest

from app.ai.planner.swarm.consensus import (
    ConsensusResult,
    ConsensusStrategy,
    SwarmConsensusEngine,
)


class MockCritic:
    """Mock critic providing deterministic evaluation scores."""

    def __init__(self, agent_id: str, scores: dict[str, float]) -> None:
        self.agent_id = agent_id
        self.scores = scores

    def evaluate(self, proposal: dict) -> float:
        pid = proposal.get("id") or proposal.get("proposal_id", "")
        return self.scores.get(pid, 0.5)


class TestSwarmConsensus(unittest.TestCase):
    """Test suite for SwarmConsensusEngine."""

    def test_weighted_quorum_reaches_consensus(self) -> None:
        """Verify weighted quorum selects highest score meeting threshold."""
        engine = SwarmConsensusEngine()

        proposals = [
            {"id": "prop_a", "content": "plan A", "confidence": 0.9, "weight": 2.0},
            {"id": "prop_b", "content": "plan B", "confidence": 0.4, "weight": 1.0},
        ]

        res = engine.reach_consensus(
            proposals, strategy=ConsensusStrategy.WEIGHTED_QUORUM, quorum_threshold=0.5
        )

        self.assertTrue(res.quorum_reached)
        self.assertIsNotNone(res.winner)
        self.assertEqual(res.winner["id"], "prop_a")
        self.assertGreater(res.winning_score, 0.5)

    def test_weighted_quorum_fails_when_below_threshold(self) -> None:
        """Verify quorum fails when no proposal reaches the threshold."""
        engine = SwarmConsensusEngine()

        proposals = [
            {"id": "prop_a", "confidence": 0.4, "weight": 1.0},
            {"id": "prop_b", "confidence": 0.3, "weight": 1.0},
        ]

        # Quorum threshold 0.8
        res = engine.reach_consensus(
            proposals, strategy=ConsensusStrategy.WEIGHTED_QUORUM, quorum_threshold=0.8
        )

        self.assertFalse(res.quorum_reached)
        self.assertIsNone(res.winner)

    def test_strict_majority_success_and_failure(self) -> None:
        """Verify strict majority requires > 50% of the cast votes."""
        engine = SwarmConsensusEngine()

        # 3 evaluators: 2 vote for prop_1, 1 votes for prop_2 -> Strict majority
        c1 = MockCritic("c1", {"p1": 0.9, "p2": 0.2})
        c2 = MockCritic("c2", {"p1": 0.8, "p2": 0.3})
        c3 = MockCritic("c3", {"p1": 0.1, "p2": 0.95})

        proposals = [{"id": "p1"}, {"id": "p2"}]

        res = engine.reach_consensus(
            proposals, evaluators=[c1, c2, c3], strategy=ConsensusStrategy.STRICT_MAJORITY
        )

        self.assertTrue(res.quorum_reached)
        self.assertEqual(res.winner["id"], "p1")
        self.assertEqual(res.winning_score, 2.0)

        # 3-way tie with 3 candidates: no candidate gets > 50%
        c_a = MockCritic("c_a", {"p1": 0.9, "p2": 0.1, "p3": 0.1})
        c_b = MockCritic("c_b", {"p1": 0.1, "p2": 0.9, "p3": 0.1})
        c_c = MockCritic("c_c", {"p1": 0.1, "p2": 0.1, "p3": 0.9})
        prop3 = [{"id": "p1"}, {"id": "p2"}, {"id": "p3"}]

        res_tie = engine.reach_consensus(
            prop3, evaluators=[c_a, c_b, c_c], strategy=ConsensusStrategy.STRICT_MAJORITY
        )
        self.assertFalse(res_tie.quorum_reached)
        self.assertIsNone(res_tie.winner)

    def test_borda_count_preference_ranking(self) -> None:
        """Verify Borda count assigns points according to candidate ranking."""
        engine = SwarmConsensusEngine()

        # 3 candidates: 1st place gets 2 pts, 2nd gets 1 pt, 3rd gets 0 pts
        # c1 ranks: A(0.9) > B(0.7) > C(0.2) -> A: 2, B: 1, C: 0
        # c2 ranks: B(0.95) > A(0.6) > C(0.1) -> B: 2, A: 1, C: 0
        # Totals: B = 3, A = 3, C = 0.
        c1 = MockCritic("c1", {"A": 0.9, "B": 0.7, "C": 0.2})
        c2 = MockCritic("c2", {"A": 0.6, "B": 0.95, "C": 0.1})

        proposals = [{"id": "A", "confidence": 0.9}, {"id": "B", "confidence": 0.95}, {"id": "C"}]

        res = engine.reach_consensus(
            proposals, evaluators=[c1, c2], strategy=ConsensusStrategy.BORDA_COUNT
        )

        self.assertTrue(res.quorum_reached)
        # B and A have 3 points each; B has confidence 0.95 > 0.9 -> B wins tie-breaker!
        self.assertEqual(res.winner["id"], "B")
        self.assertEqual(res.scores["B"], 3.0)
        self.assertEqual(res.scores["A"], 3.0)
        self.assertEqual(res.scores["C"], 0.0)

    def test_invalid_proposal_rejection(self) -> None:
        """Verify proposals failing schema or validator_fn are rejected."""
        engine = SwarmConsensusEngine()

        proposals = [
            {"id": "valid_1", "security_risk": "low", "confidence": 0.8},
            {"id": "invalid_unsafe", "security_risk": "critical", "confidence": 0.99},
            {"no_id_field": "bad_schema"},
        ]

        def _security_guard(p: dict) -> bool:
            return p.get("security_risk") != "critical"

        res = engine.reach_consensus(
            proposals,
            validator_fn=_security_guard,
            strategy=ConsensusStrategy.WEIGHTED_QUORUM,
            quorum_threshold=0.5,
        )

        self.assertTrue(res.quorum_reached)
        self.assertEqual(res.winner["id"], "valid_1")
        self.assertIn("invalid_unsafe", res.metadata["rejected_proposals"])
        self.assertIn("unknown_missing_id", res.metadata["rejected_proposals"])

    def test_deterministic_tie_breaking(self) -> None:
        """Verify identical scores and confidences break ties deterministically by ID."""
        engine = SwarmConsensusEngine()

        # Two identical proposals
        proposals = [
            {"id": "alpha_plan", "confidence": 0.9, "weight": 1.0},
            {"id": "beta_plan", "confidence": 0.9, "weight": 1.0},
        ]

        res = engine.reach_consensus(
            proposals, strategy=ConsensusStrategy.WEIGHTED_QUORUM, quorum_threshold=0.1
        )

        self.assertTrue(res.quorum_reached)
        # Deterministic lexicographical tie-break
        self.assertEqual(res.winner["id"], "alpha_plan")

    def test_consensus_result_serialization(self) -> None:
        """Verify ConsensusResult round-trip serialization."""
        res = ConsensusResult(
            winner={"id": "winner_p", "content": "win"},
            winning_score=0.88,
            strategy=ConsensusStrategy.WEIGHTED_QUORUM,
            total_votes=3,
            quorum_reached=True,
            scores={"winner_p": 0.88},
            metadata={"source": "test"},
        )

        d = res.to_dict()
        self.assertEqual(d["winner"]["id"], "winner_p")
        self.assertEqual(d["strategy"], "WEIGHTED_QUORUM")

        restored = ConsensusResult.from_dict(d)
        self.assertEqual(restored.winning_score, 0.88)
        self.assertEqual(restored.strategy, ConsensusStrategy.WEIGHTED_QUORUM)
        self.assertTrue(restored.quorum_reached)


if __name__ == "__main__":
    unittest.main()
