"""Multi-Agent Swarm Consensus Engine for Phase 19.3.

Supports Weighted Quorum, Strict Majority, and Borda Count voting strategies
with deterministic tie-breaking, invalid proposal rejection, and critic score integration.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger("app.ai.planner.swarm.consensus")


class ConsensusStrategy(str, Enum):
    """Voting and consensus aggregation strategies."""

    WEIGHTED_QUORUM = "WEIGHTED_QUORUM"
    STRICT_MAJORITY = "STRICT_MAJORITY"
    BORDA_COUNT = "BORDA_COUNT"


@dataclass
class ConsensusResult:
    """Outcome of a multi-agent swarm consensus voting process.

    Attributes:
        winner: Winning proposal dictionary or None if no consensus was reached.
        winning_score: Score or vote tally achieved by the winning proposal.
        strategy: Consensus strategy applied.
        total_votes: Total valid votes cast or evaluators participating.
        quorum_reached: Whether the required threshold/majority was satisfied.
        scores: Mapping of proposal identifiers to their aggregate scores.
        metadata: Diagnostic details including rejected proposals and tie-breakers.
    """

    winner: Optional[Dict[str, Any]] = None
    winning_score: float = 0.0
    strategy: ConsensusStrategy = ConsensusStrategy.WEIGHTED_QUORUM
    total_votes: int = 0
    quorum_reached: bool = False
    scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "winner": dict(self.winner) if self.winner else None,
            "winning_score": self.winning_score,
            "strategy": self.strategy.value if hasattr(self.strategy, "value") else str(self.strategy),
            "total_votes": self.total_votes,
            "quorum_reached": self.quorum_reached,
            "scores": dict(self.scores),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConsensusResult:
        """Create result from dictionary."""
        strat_str = data.get("strategy", ConsensusStrategy.WEIGHTED_QUORUM.value)
        try:
            strat = ConsensusStrategy(strat_str)
        except Exception:
            strat = ConsensusStrategy.WEIGHTED_QUORUM

        return cls(
            winner=dict(data["winner"]) if data.get("winner") else None,
            winning_score=float(data.get("winning_score", 0.0)),
            strategy=strat,
            total_votes=int(data.get("total_votes", 0)),
            quorum_reached=bool(data.get("quorum_reached", False)),
            scores=dict(data.get("scores", {})),
            metadata=dict(data.get("metadata", {})),
        )


class SwarmConsensusEngine:
    """Thread-safe engine orchestrating multi-agent consensus across candidate proposals."""

    def __init__(self) -> None:
        """Initialize SwarmConsensusEngine."""
        self._lock = threading.RLock()

    def _get_proposal_id(self, proposal: Dict[str, Any]) -> str:
        """Extract a unique string identifier from a proposal dictionary."""
        return str(proposal.get("id") or proposal.get("proposal_id") or "")

    def _validate_proposals(
        self,
        proposals: List[Dict[str, Any]],
        validator_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Filter out invalid, malformed, or rejected proposals."""
        valid: List[Dict[str, Any]] = []
        rejected_ids: List[str] = []

        for p in proposals:
            if not isinstance(p, dict):
                continue
            pid = self._get_proposal_id(p)
            if not pid:
                rejected_ids.append("unknown_missing_id")
                continue

            # Check custom validator function if supplied
            if validator_fn is not None:
                try:
                    if not validator_fn(p):
                        rejected_ids.append(pid)
                        continue
                except Exception as exc:
                    logger.warning("Validator rejected proposal '%s': %s", pid, exc)
                    rejected_ids.append(pid)
                    continue

            # Check if proposal is explicitly marked invalid
            if p.get("is_valid") is False:
                rejected_ids.append(pid)
                continue

            valid.append(p)

        return valid, rejected_ids

    def _evaluate_proposal_score(
        self,
        evaluator: Any,
        proposal: Dict[str, Any],
    ) -> float:
        """Calculate a score for a proposal from an evaluator (CriticAgent or callable)."""
        if hasattr(evaluator, "critique_plan"):
            feedback = evaluator.critique_plan(proposal)
            return float(getattr(feedback, "score", 0.5))
        if hasattr(evaluator, "evaluate"):
            return float(evaluator.evaluate(proposal))
        if callable(evaluator):
            try:
                res = evaluator(proposal)
                return float(res)
            except Exception:
                return 0.5
        return float(proposal.get("confidence", 1.0))

    def reach_consensus(
        self,
        proposals: List[Dict[str, Any]],
        evaluators: Optional[List[Any]] = None,
        strategy: ConsensusStrategy = ConsensusStrategy.WEIGHTED_QUORUM,
        quorum_threshold: float = 0.5,
        validator_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
        evaluator_weights: Optional[Dict[str, float]] = None,
    ) -> ConsensusResult:
        """Reach consensus among candidate proposals according to the specified strategy.

        Args:
            proposals: List of proposal dictionaries to evaluate.
            evaluators: Optional list of CriticAgents or evaluator callables.
            strategy: ConsensusStrategy (WEIGHTED_QUORUM, STRICT_MAJORITY, BORDA_COUNT).
            quorum_threshold: Minimum fraction of votes/scores required to reach quorum.
            validator_fn: Optional validation callback to reject malformed proposals.
            evaluator_weights: Optional map of evaluator ID to vote weight.

        Returns:
            ConsensusResult containing the winning proposal, score breakdown, and metadata.
        """
        with self._lock:
            valid_proposals, rejected = self._validate_proposals(proposals, validator_fn)

            if not valid_proposals:
                return ConsensusResult(
                    winner=None,
                    winning_score=0.0,
                    strategy=strategy,
                    total_votes=0,
                    quorum_reached=False,
                    scores={},
                    metadata={"rejected_proposals": rejected, "reason": "No valid proposals."},
                )

            proposal_map = {self._get_proposal_id(p): p for p in valid_proposals}

            if strategy == ConsensusStrategy.WEIGHTED_QUORUM:
                return self._resolve_weighted_quorum(
                    valid_proposals, proposal_map, evaluators, quorum_threshold, evaluator_weights, rejected
                )
            elif strategy == ConsensusStrategy.STRICT_MAJORITY:
                return self._resolve_strict_majority(
                    valid_proposals, proposal_map, evaluators, rejected
                )
            elif strategy == ConsensusStrategy.BORDA_COUNT:
                return self._resolve_borda_count(
                    valid_proposals, proposal_map, evaluators, evaluator_weights, rejected
                )

            # Fallback
            return self._resolve_weighted_quorum(
                valid_proposals, proposal_map, evaluators, quorum_threshold, evaluator_weights, rejected
            )

    def _resolve_weighted_quorum(
        self,
        proposals: List[Dict[str, Any]],
        proposal_map: Dict[str, Dict[str, Any]],
        evaluators: Optional[List[Any]],
        quorum_threshold: float,
        evaluator_weights: Optional[Dict[str, float]],
        rejected: List[str],
    ) -> ConsensusResult:
        """Resolve consensus using weighted score aggregation and quorum threshold."""
        scores: Dict[str, float] = {pid: 0.0 for pid in proposal_map}
        total_weight: float = 0.0

        if evaluators:
            for idx, evaluator in enumerate(evaluators):
                ev_id = getattr(evaluator, "agent_id", f"evaluator_{idx}")
                weight = float((evaluator_weights or {}).get(ev_id, 1.0))
                total_weight += weight

                for p in proposals:
                    pid = self._get_proposal_id(p)
                    eval_score = self._evaluate_proposal_score(evaluator, p)
                    scores[pid] += eval_score * weight
        else:
            # Aggregate based on proposal self-reported confidence and weight
            for p in proposals:
                pid = self._get_proposal_id(p)
                w = float(p.get("weight", 1.0))
                conf = float(p.get("confidence", 1.0))
                scores[pid] = conf * w
                total_weight += w

        # Normalize scores to fraction of total possible weight
        norm_scores: Dict[str, float] = {}
        for pid, score in scores.items():
            norm_scores[pid] = round(score / total_weight if total_weight > 0 else 0.0, 4)

        # Determine winner deterministically
        # Tie-break key: (score, confidence, -proposal_id)
        candidates_sorted = sorted(
            proposal_map.keys(),
            key=lambda pid: (
                norm_scores.get(pid, 0.0),
                float(proposal_map[pid].get("confidence", 0.0)),
                # Invert string comparison for reverse sorting
                [-ord(c) for c in pid],
            ),
            reverse=True,
        )

        top_id = candidates_sorted[0]
        top_score = norm_scores[top_id]
        quorum_reached = top_score >= quorum_threshold

        return ConsensusResult(
            winner=proposal_map[top_id] if quorum_reached else None,
            winning_score=top_score,
            strategy=ConsensusStrategy.WEIGHTED_QUORUM,
            total_votes=len(evaluators) if evaluators else len(proposals),
            quorum_reached=quorum_reached,
            scores=norm_scores,
            metadata={
                "quorum_threshold": quorum_threshold,
                "rejected_proposals": rejected,
                "total_weight": total_weight,
            },
        )

    def _resolve_strict_majority(
        self,
        proposals: List[Dict[str, Any]],
        proposal_map: Dict[str, Dict[str, Any]],
        evaluators: Optional[List[Any]],
        rejected: List[str],
    ) -> ConsensusResult:
        """Resolve consensus requiring strictly > 50% of the total vote."""
        votes: Dict[str, float] = {pid: 0.0 for pid in proposal_map}
        total_votes: float = 0.0

        if evaluators:
            total_votes = float(len(evaluators))
            for evaluator in evaluators:
                # Evaluator votes for their highest rated proposal
                best_pid: Optional[str] = None
                best_score = -1.0
                for p in proposals:
                    pid = self._get_proposal_id(p)
                    s = self._evaluate_proposal_score(evaluator, p)
                    if s > best_score:
                        best_score = s
                        best_pid = pid
                if best_pid is not None:
                    votes[best_pid] += 1.0
        else:
            total_votes = float(sum(p.get("votes", 1) for p in proposals))
            for p in proposals:
                pid = self._get_proposal_id(p)
                votes[pid] = float(p.get("votes", 1))

        # Check strict majority (> 50%)
        candidates_sorted = sorted(
            proposal_map.keys(),
            key=lambda pid: (
                votes.get(pid, 0.0),
                float(proposal_map[pid].get("confidence", 0.0)),
                [-ord(c) for c in pid],
            ),
            reverse=True,
        )

        top_id = candidates_sorted[0]
        top_vote = votes[top_id]
        quorum_reached = top_vote > (total_votes / 2.0)

        return ConsensusResult(
            winner=proposal_map[top_id] if quorum_reached else None,
            winning_score=top_vote,
            strategy=ConsensusStrategy.STRICT_MAJORITY,
            total_votes=int(total_votes),
            quorum_reached=quorum_reached,
            scores=votes,
            metadata={
                "majority_required": (total_votes / 2.0),
                "rejected_proposals": rejected,
            },
        )

    def _resolve_borda_count(
        self,
        proposals: List[Dict[str, Any]],
        proposal_map: Dict[str, Dict[str, Any]],
        evaluators: Optional[List[Any]],
        evaluator_weights: Optional[Dict[str, float]],
        rejected: List[str],
    ) -> ConsensusResult:
        """Resolve consensus using Borda Count preference ranking."""
        borda_scores: Dict[str, float] = {pid: 0.0 for pid in proposal_map}
        num_candidates = len(proposals)

        if num_candidates == 1:
            only_id = self._get_proposal_id(proposals[0])
            return ConsensusResult(
                winner=proposals[0],
                winning_score=1.0,
                strategy=ConsensusStrategy.BORDA_COUNT,
                total_votes=1,
                quorum_reached=True,
                scores={only_id: 1.0},
                metadata={"rejected_proposals": rejected},
            )

        evaluator_list = evaluators if evaluators else [None]
        total_evaluators = len(evaluator_list)

        for idx, evaluator in enumerate(evaluator_list):
            weight = 1.0
            if evaluator is not None:
                ev_id = getattr(evaluator, "agent_id", f"evaluator_{idx}")
                weight = float((evaluator_weights or {}).get(ev_id, 1.0))

            # Rank proposals for this evaluator
            def _score_key(p: Dict[str, Any]) -> float:
                if evaluator is not None:
                    return self._evaluate_proposal_score(evaluator, p)
                return float(p.get("confidence", 1.0))

            ranked = sorted(proposals, key=_score_key, reverse=True)

            # 1st gets (N-1)*w, 2nd gets (N-2)*w, ..., last gets 0
            for rank, p in enumerate(ranked):
                pid = self._get_proposal_id(p)
                points = (num_candidates - 1 - rank) * weight
                borda_scores[pid] += points

        candidates_sorted = sorted(
            proposal_map.keys(),
            key=lambda pid: (
                borda_scores.get(pid, 0.0),
                float(proposal_map[pid].get("confidence", 0.0)),
                [-ord(c) for c in pid],
            ),
            reverse=True,
        )

        top_id = candidates_sorted[0]
        top_score = borda_scores[top_id]

        return ConsensusResult(
            winner=proposal_map[top_id],
            winning_score=top_score,
            strategy=ConsensusStrategy.BORDA_COUNT,
            total_votes=total_evaluators,
            quorum_reached=True,
            scores=borda_scores,
            metadata={"num_candidates": num_candidates, "rejected_proposals": rejected},
        )
