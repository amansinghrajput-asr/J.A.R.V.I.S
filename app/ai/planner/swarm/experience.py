"""Experience Synthesizer for Phase 19.2.

Aggregates execution metrics, calculates agent performance scores,
and summarizes goal trajectories across multi-agent swarm runs.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from app.ai.planner.swarm.episodic import TrajectoryRecord

logger = logging.getLogger("app.ai.planner.swarm.experience")


class ExperienceSynthesizer:
    """Thread-safe synthesizer aggregating cross-run swarm experience and metrics."""

    def __init__(self) -> None:
        """Initialize experience synthesizer."""
        self._lock = threading.RLock()
        self._agent_scores: Dict[str, List[float]] = {}
        self._goal_records: Dict[str, List[Dict[str, Any]]] = {}
        self._total_trajectories: int = 0
        self._successful_trajectories: int = 0
        self._total_duration_ms: float = 0.0

    def update_from_record(self, record: TrajectoryRecord) -> None:
        """Update aggregate experience statistics from a TrajectoryRecord.

        Args:
            record: TrajectoryRecord to incorporate.
        """
        with self._lock:
            self._total_trajectories += 1
            if record.success:
                self._successful_trajectories += 1
            self._total_duration_ms += float(record.total_duration_ms)

            # Update agent rating scores
            for agent_id, rating in record.agent_ratings.items():
                if agent_id not in self._agent_scores:
                    self._agent_scores[agent_id] = []
                self._agent_scores[agent_id].append(float(rating))

            # Index goal record summary
            normalized_goal = record.goal.strip().lower()
            if normalized_goal not in self._goal_records:
                self._goal_records[normalized_goal] = []

            self._goal_records[normalized_goal].append(
                {
                    "trajectory_id": record.trajectory_id,
                    "plan_signature": record.plan_signature,
                    "success": record.success,
                    "duration_ms": record.total_duration_ms,
                    "timestamp": record.timestamp,
                }
            )

    def get_agent_score(self, agent_id: str, default: float = 0.5) -> float:
        """Calculate the average historical performance score for an agent.

        Args:
            agent_id: Identifier of the target agent.
            default: Score to return if agent has no historical rating.

        Returns:
            Float rating between 0.0 and 1.0.
        """
        with self._lock:
            scores = self._agent_scores.get(agent_id)
            if not scores:
                return default
            return sum(scores) / len(scores)

    def summarize_goal_history(self, goal_prefix: str) -> Dict[str, Any]:
        """Summarize execution performance for goals starting with goal_prefix.

        Args:
            goal_prefix: Prefix or search term for goal matching.

        Returns:
            Dictionary containing match count, success rate, and duration metrics.
        """
        with self._lock:
            prefix_clean = goal_prefix.strip().lower()
            matching_records: List[Dict[str, Any]] = []

            for goal_key, records in self._goal_records.items():
                if prefix_clean in goal_key or goal_key.startswith(prefix_clean):
                    matching_records.extend(records)

            total = len(matching_records)
            if total == 0:
                return {
                    "matched_goals": 0,
                    "success_rate": 0.0,
                    "avg_duration_ms": 0.0,
                    "best_plan_signature": "",
                }

            successful = sum(1 for r in matching_records if r["success"])
            durations = [r["duration_ms"] for r in matching_records]
            avg_duration = sum(durations) / total

            # Find most frequent plan signature among successful executions
            sig_counts: Dict[str, int] = {}
            for r in matching_records:
                if r["success"] and r.get("plan_signature"):
                    sig = r["plan_signature"]
                    sig_counts[sig] = sig_counts.get(sig, 0) + 1

            best_sig = (
                max(sig_counts.items(), key=lambda x: x[1])[0]
                if sig_counts
                else ""
            )

            return {
                "matched_goals": total,
                "success_rate": round(successful / total, 4),
                "avg_duration_ms": round(avg_duration, 2),
                "best_plan_signature": best_sig,
            }

    def export_statistics(self) -> Dict[str, Any]:
        """Export comprehensive aggregate swarm experience statistics.

        Returns:
            Dictionary containing global trajectory counts, agent averages, and goal metrics.
        """
        with self._lock:
            success_rate = (
                self._successful_trajectories / self._total_trajectories
                if self._total_trajectories > 0
                else 0.0
            )
            avg_duration = (
                self._total_duration_ms / self._total_trajectories
                if self._total_trajectories > 0
                else 0.0
            )

            agent_averages = {
                agent_id: round(sum(scores) / len(scores), 4)
                for agent_id, scores in self._agent_scores.items()
                if scores
            }

            return {
                "total_trajectories": self._total_trajectories,
                "successful_trajectories": self._successful_trajectories,
                "success_rate": round(success_rate, 4),
                "total_duration_ms": round(self._total_duration_ms, 2),
                "avg_duration_ms": round(avg_duration, 2),
                "agent_scores": agent_averages,
                "tracked_goals_count": len(self._goal_records),
            }
