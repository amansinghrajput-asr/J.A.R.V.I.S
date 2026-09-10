"""Conflict Detection & Resolution Subsystem for Phase 18.0.

Resolves divergent agent outputs, contradictory decisions, or conflicting plan steps
using deterministic consensus, confidence weighting, role priorities, or adjudication.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Callable, Dict, List, Optional

from app.ai.planner.events import ConflictDetected, ConflictResolved, PlannerEventBus
from app.ai.planner.multi_agent.models import AgentRole, ConflictResolution

logger = logging.getLogger("app.ai.planner.multi_agent.conflict")


class ConflictResolver:
    """Detects and reconciles conflicts between multiple agents."""

    ROLE_PRIORITIES: Dict[AgentRole, int] = {
        AgentRole.COORDINATOR: 100,
        AgentRole.CRITIC: 80,
        AgentRole.RESEARCHER: 60,
        AgentRole.WORKER: 50,
        AgentRole.EXECUTOR: 40,
        AgentRole.CUSTOM: 30,
    }

    def __init__(self, event_bus: Optional[PlannerEventBus] = None) -> None:
        self._event_bus = event_bus

    def resolve(
        self,
        conflict_type: str,
        candidates: List[Dict[str, Any]],
        strategy: str = "CONFIDENCE_WEIGHTED",
        adjudicator_fn: Optional[Callable[[List[Dict[str, Any]]], Dict[str, Any]]] = None,
        plan_id: str = "",
    ) -> ConflictResolution:
        """Resolve conflicting agent outputs.

        Args:
            conflict_type: Nature of conflict (e.g. 'OUTPUT_MISMATCH', 'PLAN_CONTRADICTION').
            candidates: List of candidate dicts with keys:
                        'agent_id', 'output', 'confidence' (float), optional 'role' (AgentRole).
            strategy: Strategy mode: 'CONFIDENCE_WEIGHTED', 'PRIORITY', 'VOTING', 'ADJUDICATION'.
            adjudicator_fn: Optional callable for explicit synthesis/adjudication.
            plan_id: Correlation plan ID.

        Returns:
            ConflictResolution describing the chosen candidate or synthesized outcome.
        """
        conf_id = str(uuid.uuid4())
        parties = [str(c.get("agent_id", "unknown")) for c in candidates]

        if self._event_bus is not None:
            self._event_bus.publish(
                ConflictDetected(
                    plan_id=plan_id,
                    conflict_type=conflict_type,
                    parties=parties,
                    details=f"Detected conflict between {len(candidates)} candidates using {strategy}",
                )
            )

        if not candidates:
            return ConflictResolution(
                conflict_id=conf_id,
                conflict_type=conflict_type,
                parties=[],
                strategy=strategy,
                resolved=False,
            )

        if len(candidates) == 1:
            c = candidates[0]
            return ConflictResolution(
                conflict_id=conf_id,
                conflict_type=conflict_type,
                parties=parties,
                strategy=strategy,
                resolved=True,
                winner_agent_id=c.get("agent_id"),
                resolution_data=c.get("output"),
            )

        winner_id: Optional[str] = None
        resolution_data: Any = None
        strat = strategy.upper()

        if strat == "CONFIDENCE_WEIGHTED":
            # Highest confidence wins
            best = max(candidates, key=lambda x: float(x.get("confidence", 0.0)))
            winner_id = best.get("agent_id")
            resolution_data = best.get("output")

        elif strat == "PRIORITY":
            # Highest role priority wins, ties broken by confidence
            def _priority_score(c: Dict[str, Any]) -> tuple[int, float]:
                role = c.get("role", AgentRole.WORKER)
                prio = self.ROLE_PRIORITIES.get(role, 50)
                conf = float(c.get("confidence", 0.0))
                return prio, conf

            best = max(candidates, key=_priority_score)
            winner_id = best.get("agent_id")
            resolution_data = best.get("output")

        elif strat == "VOTING":
            # Majority vote on output string representation
            votes: Dict[str, List[Dict[str, Any]]] = {}
            for c in candidates:
                val_key = str(c.get("output"))
                votes.setdefault(val_key, []).append(c)

            # Winner is value with most votes; ties broken by cumulative confidence
            best_val = max(votes.keys(), key=lambda k: (len(votes[k]), sum(float(x.get("confidence", 0.0)) for x in votes[k])))
            best = votes[best_val][0]
            winner_id = best.get("agent_id")
            resolution_data = best.get("output")

        elif strat == "ADJUDICATION" and adjudicator_fn is not None:
            try:
                adjudicated = adjudicator_fn(candidates)
                winner_id = adjudicated.get("agent_id", "adjudicator")
                resolution_data = adjudicated.get("output")
            except Exception as exc:
                logger.error("Adjudication failed: %s; falling back to confidence", exc)
                best = max(candidates, key=lambda x: float(x.get("confidence", 0.0)))
                winner_id = best.get("agent_id")
                resolution_data = best.get("output")
        else:
            # Default fallback: highest confidence
            best = max(candidates, key=lambda x: float(x.get("confidence", 0.0)))
            winner_id = best.get("agent_id")
            resolution_data = best.get("output")

        resolution = ConflictResolution(
            conflict_id=conf_id,
            conflict_type=conflict_type,
            parties=parties,
            strategy=strat,
            resolved=True,
            winner_agent_id=winner_id,
            resolution_data=resolution_data,
        )

        if self._event_bus is not None:
            self._event_bus.publish(
                ConflictResolved(
                    plan_id=plan_id,
                    conflict_type=conflict_type,
                    resolution_strategy=strat,
                    outcome=f"Winner: {winner_id}",
                )
            )

        return resolution
