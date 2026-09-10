"""Dynamic Load Balancer and Circuit Breaker for Phase 19.4.

Provides least-loaded task assignment, deterministic queue rebalancing,
task migration, and automated circuit breaking on failing agents.
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("app.ai.planner.swarm.balancer")


class BalancingPolicy(str, Enum):
    """Supported task distribution strategies."""

    LEAST_LOADED = "LEAST_LOADED"
    ROUND_ROBIN = "ROUND_ROBIN"


class DynamicLoadBalancer:
    """Thread-safe task load balancer with adaptive circuit breaking."""

    def __init__(
        self,
        policy: BalancingPolicy = BalancingPolicy.LEAST_LOADED,
        failure_threshold: int = 3,
        cooldown_seconds: float = 30.0,
    ) -> None:
        """Initialize DynamicLoadBalancer.

        Args:
            policy: BalancingPolicy strategy.
            failure_threshold: Number of failures before circuit breaker trips.
            cooldown_seconds: Cooldown duration before circuit breaker allows retry.
        """
        self._lock = threading.RLock()
        self.policy = policy
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds

        self._round_robin_idx = 0
        self._agent_failures: Dict[str, int] = {}
        self._circuit_broken_at: Dict[str, float] = {}
        self._agent_queues: Dict[str, List[Any]] = {}

    def _get_agent_id(self, candidate: Any) -> str:
        """Extract unique agent string ID from candidate."""
        return str(getattr(candidate, "agent_id", candidate))

    def apply_circuit_breaker(self, agent_id: str, failure_count: int = 1) -> bool:
        """Record task failures and trip circuit breaker if threshold reached.

        Args:
            agent_id: Identifier of the failing agent.
            failure_count: Number of failures to increment.

        Returns:
            True if the circuit breaker is now tripped (broken), False otherwise.
        """
        with self._lock:
            current = self._agent_failures.get(agent_id, 0) + failure_count
            self._agent_failures[agent_id] = current

            if current >= self.failure_threshold:
                self._circuit_broken_at[agent_id] = time.time()
                logger.warning(
                    "Circuit breaker TRIPPED for agent '%s' (%d consecutive failures)",
                    agent_id,
                    current,
                )
                return True
            return False

    def is_circuit_broken(self, agent_id: str) -> bool:
        """Check if an agent is currently circuit broken.

        Returns:
            True if circuit is active/broken, False if healthy or cooldown elapsed.
        """
        with self._lock:
            tripped_time = self._circuit_broken_at.get(agent_id)
            if tripped_time is None:
                return False

            # Check if cooldown has elapsed
            if (time.time() - tripped_time) >= self.cooldown_seconds:
                # Cooldown expired: half-open recovery
                del self._circuit_broken_at[agent_id]
                self._agent_failures[agent_id] = 0
                logger.info("Circuit breaker reset for agent '%s' after cooldown", agent_id)
                return False

            return True

    def reset_circuit_breaker(self, agent_id: str) -> None:
        """Explicitly reset circuit breaker and clear failure counters for an agent."""
        with self._lock:
            self._circuit_broken_at.pop(agent_id, None)
            self._agent_failures.pop(agent_id, None)

    def assign_task(
        self,
        task: Any,
        candidate_agents: List[Any],
        agent_loads: Optional[Dict[str, int]] = None,
    ) -> str:
        """Assign a task to the optimal candidate agent avoiding circuit-broken agents.

        Args:
            task: Task to assign.
            candidate_agents: List of candidate agents or string IDs.
            agent_loads: Optional explicit map of current queue depths.

        Returns:
            Selected agent identifier.

        Raises:
            ValueError: If candidate_agents is empty.
        """
        with self._lock:
            if not candidate_agents:
                raise ValueError("Cannot assign task with empty candidate list.")

            candidate_ids = [self._get_agent_id(c) for c in candidate_agents]

            # Filter out circuit-broken agents
            available_ids = [aid for aid in candidate_ids if not self.is_circuit_broken(aid)]

            # If all candidates are circuit broken, fallback to candidate with fewest failures
            if not available_ids:
                available_ids = sorted(
                    candidate_ids,
                    key=lambda aid: (self._agent_failures.get(aid, 0), aid),
                )

            # Assign based on policy
            if self.policy == BalancingPolicy.ROUND_ROBIN:
                selected = available_ids[self._round_robin_idx % len(available_ids)]
                self._round_robin_idx += 1
                return selected

            # Default: LEAST_LOADED
            loads: Dict[str, int] = {}
            for aid in available_ids:
                if agent_loads is not None and aid in agent_loads:
                    loads[aid] = agent_loads[aid]
                else:
                    loads[aid] = len(self._agent_queues.get(aid, []))

            # Deterministic selection: min load, then lexicographical order on tie
            selected = min(available_ids, key=lambda aid: (loads.get(aid, 0), aid))

            # Track in internal queue
            if selected not in self._agent_queues:
                self._agent_queues[selected] = []
            self._agent_queues[selected].append(task)

            return selected

    def rebalance(
        self,
        agent_queues: Dict[str, List[Any]],
        max_imbalance_threshold: int = 2,
    ) -> Dict[str, List[Any]]:
        """Redistribute tasks across agent queues to eliminate workload imbalances.

        Args:
            agent_queues: Dictionary mapping agent IDs to their pending task lists.
            max_imbalance_threshold: Max allowed task difference between busiest and lightest.

        Returns:
            New balanced dictionary of agent queues.
        """
        with self._lock:
            if len(agent_queues) <= 1:
                return {k: list(v) for k, v in agent_queues.items()}

            # Copy queues to prevent mutating inputs
            queues: Dict[str, List[Any]] = {k: list(v) for k, v in agent_queues.items()}
            healthy_agents = [aid for aid in sorted(queues.keys()) if not self.is_circuit_broken(aid)]

            if not healthy_agents:
                return queues

            # Loop while imbalance exceeds threshold
            while True:
                # Find maximum and minimum loaded agents among healthy
                max_agent = max(healthy_agents, key=lambda aid: len(queues[aid]))
                min_agent = min(healthy_agents, key=lambda aid: len(queues[aid]))

                diff = len(queues[max_agent]) - len(queues[min_agent])
                if diff <= max_imbalance_threshold:
                    break

                # Migrate one task from max_agent to min_agent
                task_to_move = queues[max_agent].pop()
                queues[min_agent].append(task_to_move)

            return queues

    def migrate_tasks(
        self,
        from_agent_id: str,
        to_agent_ids: List[str],
        max_tasks: Optional[int] = None,
    ) -> List[Any]:
        """Evacuate tasks from an overloaded or faulted agent to target agents.

        Args:
            from_agent_id: Identifier of the source agent.
            to_agent_ids: Target candidate agent identifiers.
            max_tasks: Maximum number of tasks to migrate (None for all).

        Returns:
            List of successfully migrated tasks.
        """
        with self._lock:
            source_queue = self._agent_queues.get(from_agent_id, [])
            if not source_queue or not to_agent_ids:
                return []

            target_candidates = [aid for aid in to_agent_ids if not self.is_circuit_broken(aid)]
            if not target_candidates:
                target_candidates = list(to_agent_ids)

            num_to_move = len(source_queue) if max_tasks is None else min(len(source_queue), max_tasks)
            migrated: List[Any] = []

            for i in range(num_to_move):
                task = source_queue.pop(0)
                # Assign to target using least loaded
                target_aid = min(
                    target_candidates,
                    key=lambda aid: (len(self._agent_queues.get(aid, [])), aid),
                )
                if target_aid not in self._agent_queues:
                    self._agent_queues[target_aid] = []
                self._agent_queues[target_aid].append(task)
                migrated.append(task)

            return migrated
