"""Unit tests for Episodic Swarm Memory & Experience Synthesis (Sprint 19.2).

Covers trajectory recording, deterministic similarity ranking, agent affinity boosting,
experience statistics aggregation, serialization, export/import knowledge, and thread safety.
"""

import concurrent.futures
import time
import unittest

from app.ai.planner.swarm.episodic import EpisodicMemoryStore, TrajectoryRecord
from app.ai.planner.swarm.experience import ExperienceSynthesizer


class TestSwarmEpisodic(unittest.TestCase):
    """Test suite for EpisodicMemoryStore and ExperienceSynthesizer."""

    def test_trajectory_record_serialization(self) -> None:
        """Verify TrajectoryRecord round-trip dictionary serialization."""
        rec = TrajectoryRecord(
            trajectory_id="traj_100",
            goal="Analyze quarterly sales and forecast growth",
            plan_signature="fetch->aggregate->model",
            success=True,
            total_duration_ms=450.5,
            timestamp=1700000000.0,
            agent_ratings={"analyst_1": 0.95, "critic_1": 0.90},
            recovery_events=[{"action": "aggregate", "agent_id": "analyst_1", "score": 0.95}],
            artifacts_produced=["report.pdf"],
        )

        d = rec.to_dict()
        self.assertEqual(d["trajectory_id"], "traj_100")
        self.assertEqual(d["goal"], "Analyze quarterly sales and forecast growth")
        self.assertTrue(d["success"])
        self.assertEqual(d["agent_ratings"]["analyst_1"], 0.95)

        restored = TrajectoryRecord.from_dict(d)
        self.assertEqual(restored.trajectory_id, rec.trajectory_id)
        self.assertEqual(restored.goal, rec.goal)
        self.assertEqual(restored.plan_signature, rec.plan_signature)
        self.assertEqual(restored.total_duration_ms, 450.5)
        self.assertEqual(restored.agent_ratings, rec.agent_ratings)
        self.assertEqual(restored.recovery_events, rec.recovery_events)
        self.assertEqual(restored.artifacts_produced, rec.artifacts_produced)

    def test_trajectory_recording_and_retrieval(self) -> None:
        """Verify recording trajectories and inverted token index updates."""
        store = EpisodicMemoryStore()
        rec1 = TrajectoryRecord(
            trajectory_id="t1",
            goal="Search web for Python async tutorials",
            plan_signature="search->summarize",
            success=True,
        )
        rec2 = TrajectoryRecord(
            trajectory_id="t2",
            goal="Compile C++ shared library on Linux",
            plan_signature="compile->link",
            success=True,
        )

        store.record_trajectory(rec1)
        store.record_trajectory(rec2)

        # Query Python
        results = store.query_similar_goals("Python tutorials for web", top_k=2)
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].trajectory_id, "t1")

        # Query C++
        results_cpp = store.query_similar_goals("Compile C++ code", top_k=2)
        self.assertGreaterEqual(len(results_cpp), 1)
        self.assertEqual(results_cpp[0].trajectory_id, "t2")

    def test_similarity_ranking_and_threshold(self) -> None:
        """Verify deterministic ranking and threshold filtering."""
        store = EpisodicMemoryStore()
        r_exact = TrajectoryRecord(
            trajectory_id="r_exact",
            goal="Deploy Docker container to Kubernetes cluster",
            timestamp=100.0,
        )
        r_partial = TrajectoryRecord(
            trajectory_id="r_partial",
            goal="Deploy web application to server",
            timestamp=200.0,
        )
        r_unrelated = TrajectoryRecord(
            trajectory_id="r_unrelated",
            goal="Generate audio file using neural TTS",
            timestamp=300.0,
        )

        store.record_trajectory(r_exact)
        store.record_trajectory(r_partial)
        store.record_trajectory(r_unrelated)

        matches = store.query_similar_goals(
            "Deploy Docker container to Kubernetes cluster",
            top_k=5,
            threshold=0.2,
        )
        self.assertGreaterEqual(len(matches), 1)
        # Exact match must be first
        self.assertEqual(matches[0].trajectory_id, "r_exact")
        # Unrelated must not be present
        self.assertNotIn("r_unrelated", [m.trajectory_id for m in matches])

    def test_agent_affinity_boosting(self) -> None:
        """Verify agent affinity boost calculation based on historical ratings."""
        store = EpisodicMemoryStore()

        # Agent with high ratings (0.9, 0.95)
        rec_high = TrajectoryRecord(
            trajectory_id="th",
            goal="Code optimization",
            agent_ratings={"coder_expert": 0.95},
            recovery_events=[{"action": "refactor", "agent_id": "coder_expert", "score": 0.95}],
        )
        # Agent with low ratings (0.3)
        rec_low = TrajectoryRecord(
            trajectory_id="tl",
            goal="Code review",
            agent_ratings={"coder_novice": 0.3},
            recovery_events=[{"action": "refactor", "agent_id": "coder_novice", "score": 0.3}],
        )

        store.record_trajectory(rec_high)
        store.record_trajectory(rec_low)

        boost_expert = store.get_agent_affinity_boost("coder_expert", "refactor")
        boost_novice = store.get_agent_affinity_boost("coder_novice", "refactor")
        boost_unknown = store.get_agent_affinity_boost("unknown_agent", "refactor")

        self.assertGreater(boost_expert, 0.0)
        self.assertEqual(boost_novice, 0.0)
        self.assertEqual(boost_unknown, 0.0)

    def test_experience_synthesizer_metrics(self) -> None:
        """Verify ExperienceSynthesizer tracks agent averages and goal summaries."""
        synth = ExperienceSynthesizer()
        store = EpisodicMemoryStore(synthesizer=synth)

        rec1 = TrajectoryRecord(
            trajectory_id="s1",
            goal="Scrape product pricing data",
            plan_signature="scrape->parse",
            success=True,
            total_duration_ms=100.0,
            agent_ratings={"worker_a": 0.9, "worker_b": 0.8},
        )
        rec2 = TrajectoryRecord(
            trajectory_id="s2",
            goal="Scrape stock pricing data",
            plan_signature="scrape->parse",
            success=False,
            total_duration_ms=150.0,
            agent_ratings={"worker_a": 0.7, "worker_b": 0.6},
        )

        store.record_trajectory(rec1)
        store.record_trajectory(rec2)

        # Check agent scores
        score_a = synth.get_agent_score("worker_a")
        score_b = synth.get_agent_score("worker_b")
        self.assertAlmostEqual(score_a, 0.8, places=4)
        self.assertAlmostEqual(score_b, 0.7, places=4)

        # Check goal summary
        summary = synth.summarize_goal_history("scrape")
        self.assertEqual(summary["matched_goals"], 2)
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["avg_duration_ms"], 125.0)
        self.assertEqual(summary["best_plan_signature"], "scrape->parse")

        # Check global statistics export
        stats = synth.export_statistics()
        self.assertEqual(stats["total_trajectories"], 2)
        self.assertEqual(stats["successful_trajectories"], 1)
        self.assertEqual(stats["success_rate"], 0.5)
        self.assertEqual(stats["total_duration_ms"], 250.0)
        self.assertIn("worker_a", stats["agent_scores"])

    def test_export_and_import_knowledge(self) -> None:
        """Verify round-trip export and import of knowledge store."""
        store1 = EpisodicMemoryStore()
        rec = TrajectoryRecord(
            trajectory_id="exp_1",
            goal="Generate weekly financial report",
            plan_signature="query->format->send",
            success=True,
            agent_ratings={"reporter": 0.92},
        )
        store1.record_trajectory(rec)

        exported = store1.export_knowledge()
        self.assertEqual(exported["record_count"], 1)

        store2 = EpisodicMemoryStore()
        store2.import_knowledge(exported)

        matches = store2.query_similar_goals("financial report")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].trajectory_id, "exp_1")

        boost = store2.get_agent_affinity_boost("reporter", "*")
        self.assertGreater(boost, 0.0)

    def test_thread_safety_concurrent_recording_and_queries(self) -> None:
        """Verify thread safety under heavy concurrent records and queries."""
        store = EpisodicMemoryStore(synthesizer=ExperienceSynthesizer())
        num_threads = 10
        ops_per_thread = 20

        def _worker(thread_idx: int) -> None:
            for i in range(ops_per_thread):
                rec = TrajectoryRecord(
                    trajectory_id=f"t_{thread_idx}_{i}",
                    goal=f"Goal from worker {thread_idx} iteration {i}",
                    plan_signature="stepA->stepB",
                    success=(i % 2 == 0),
                    total_duration_ms=float(i * 10),
                    agent_ratings={f"agent_{thread_idx}": 0.8},
                )
                store.record_trajectory(rec)
                # Concurrent read
                _ = store.query_similar_goals(f"worker {thread_idx}")
                _ = store.get_agent_affinity_boost(f"agent_{thread_idx}", "*")

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(_worker, tid) for tid in range(num_threads)]
            for f in concurrent.futures.as_completed(futures):
                f.result()

        exported = store.export_knowledge()
        self.assertEqual(exported["record_count"], num_threads * ops_per_thread)


if __name__ == "__main__":
    unittest.main()
