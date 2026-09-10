"""Autonomous Swarm Supervisor and Self-Healing Engine for Phase 19.4.

Provides heartbeat monitoring, agent timeout detection, stuck agent recovery,
automatic status reset, fault reporting, and load distribution analysis.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from app.ai.planner.multi_agent.models import AgentStatus
from app.ai.planner.swarm.models import SwarmStatus
from app.ai.planner.swarm.subswarm import SubSwarm

logger = logging.getLogger("app.ai.planner.swarm.supervisor")


@dataclass
class SwarmHealthReport:
    """Comprehensive health summary across managed sub-swarms and their agents."""

    healthy: bool
    total_swarms: int
    swarm_statuses: Dict[str, str] = field(default_factory=dict)
    stuck_agents: List[str] = field(default_factory=list)
    faulted_agents: List[str] = field(default_factory=list)
    active_agent_count: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert health report to dictionary."""
        return {
            "healthy": self.healthy,
            "total_swarms": self.total_swarms,
            "swarm_statuses": dict(self.swarm_statuses),
            "stuck_agents": list(self.stuck_agents),
            "faulted_agents": list(self.faulted_agents),
            "active_agent_count": self.active_agent_count,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SwarmHealthReport:
        """Create health report from dictionary."""
        return cls(
            healthy=bool(data.get("healthy", False)),
            total_swarms=int(data.get("total_swarms", 0)),
            swarm_statuses=dict(data.get("swarm_statuses", {})),
            stuck_agents=list(data.get("stuck_agents", [])),
            faulted_agents=list(data.get("faulted_agents", [])),
            active_agent_count=int(data.get("active_agent_count", 0)),
            details=dict(data.get("details", {})),
        )


class SwarmSupervisor:
    """Autonomous health watchdog and self-healing coordinator for hierarchical swarms."""

    def __init__(
        self,
        heartbeat_timeout: float = 30.0,
        stuck_threshold: float = 45.0,
    ) -> None:
        """Initialize SwarmSupervisor.

        Args:
            heartbeat_timeout: Maximum elapsed seconds between heartbeats before warning.
            stuck_threshold: Duration in seconds an agent can stay BUSY before being deemed stuck.
        """
        self._lock = threading.RLock()
        self._heartbeat_timeout = heartbeat_timeout
        self._stuck_threshold = stuck_threshold
        self._swarms: Dict[str, SubSwarm] = {}
        self._heartbeats: Dict[str, float] = {}
        self._agent_busy_since: Dict[str, float] = {}

    def register_swarm(self, swarm: SubSwarm) -> None:
        """Register a sub-swarm under supervision.

        Args:
            swarm: SubSwarm instance to monitor.
        """
        with self._lock:
            self._swarms[swarm.swarm_id] = swarm
            self._heartbeats[swarm.swarm_id] = time.time()
            logger.info("Supervisor registered swarm '%s'", swarm.swarm_id)

    def unregister_swarm(self, swarm_id: str) -> Optional[SubSwarm]:
        """Remove a sub-swarm from supervision.

        Args:
            swarm_id: Swarm identifier.

        Returns:
            The unregistered SubSwarm or None.
        """
        with self._lock:
            self._heartbeats.pop(swarm_id, None)
            return self._swarms.pop(swarm_id, None)

    def record_heartbeat(self, target_id: str) -> None:
        """Record a heartbeat timestamp for an agent or sub-swarm.

        Args:
            target_id: Identifier of the agent or swarm.
        """
        with self._lock:
            self._heartbeats[target_id] = time.time()

    def check_health(self) -> SwarmHealthReport:
        """Inspect all monitored swarms and agents to detect faults and timeouts.

        Returns:
            SwarmHealthReport summarizing system health.
        """
        with self._lock:
            now = time.time()
            swarm_statuses: Dict[str, str] = {}
            stuck_agents: List[str] = []
            faulted_agents: List[str] = []
            total_active_agents = 0
            swarm_healthy = True

            for s_id, swarm in self._swarms.items():
                s_status = swarm.status.value if hasattr(swarm.status, "value") else str(swarm.status)
                swarm_statuses[s_id] = s_status

                if swarm.status in (SwarmStatus.FAILED, SwarmStatus.TERMINATED):
                    swarm_healthy = False

                # Inspect agents in the swarm's registry
                if hasattr(swarm, "registry") and hasattr(swarm.registry, "list_agents"):
                    agents = swarm.registry.list_agents()
                    total_active_agents += len(agents)

                    for agent in agents:
                        a_id = agent.agent_id
                        status = agent.status

                        # Check if agent has reported heartbeat
                        last_hb = self._heartbeats.get(a_id, now)

                        if status == AgentStatus.FAILED:
                            faulted_agents.append(a_id)
                            swarm_healthy = False

                        elif status == AgentStatus.BUSY:
                            if a_id not in self._agent_busy_since:
                                self._agent_busy_since[a_id] = now
                            elapsed_busy = now - self._agent_busy_since[a_id]

                            # Deem stuck if busy past threshold or missed heartbeat
                            if elapsed_busy > self._stuck_threshold or (now - last_hb) > self._heartbeat_timeout:
                                stuck_agents.append(a_id)
                                swarm_healthy = False
                        else:
                            # Idle/ready
                            self._agent_busy_since.pop(a_id, None)

            overall_healthy = (
                swarm_healthy
                and len(stuck_agents) == 0
                and len(faulted_agents) == 0
            )

            return SwarmHealthReport(
                healthy=overall_healthy,
                total_swarms=len(self._swarms),
                swarm_statuses=swarm_statuses,
                stuck_agents=sorted(list(set(stuck_agents))),
                faulted_agents=sorted(list(set(faulted_agents))),
                active_agent_count=total_active_agents,
                details={"timestamp": now},
            )

    def auto_heal(self) -> List[Dict[str, Any]]:
        """Automatically remediate stuck agents, faulted states, and failed swarms.

        Returns:
            List of healing action event descriptions performed.
        """
        with self._lock:
            actions_taken: List[Dict[str, Any]] = []
            health = self.check_health()

            # 1. Remediate failed swarms
            for s_id, status_str in health.swarm_statuses.items():
                if status_str == SwarmStatus.FAILED.value:
                    swarm = self._swarms.get(s_id)
                    if swarm is not None:
                        swarm.status = SwarmStatus.READY
                        self.record_heartbeat(s_id)
                        action = {
                            "target_type": "swarm",
                            "target_id": s_id,
                            "action": "reset_swarm_status",
                            "from_status": status_str,
                            "to_status": SwarmStatus.READY.value,
                        }
                        actions_taken.append(action)
                        logger.warning("Auto-healed swarm '%s' from FAILED to READY", s_id)

            # 2. Remediate stuck agents
            for s_id, swarm in self._swarms.items():
                if not hasattr(swarm, "registry") or not hasattr(swarm.registry, "list_agents"):
                    continue

                for agent in swarm.registry.list_agents():
                    a_id = agent.agent_id

                    # If agent was flagged stuck
                    if a_id in health.stuck_agents:
                        agent.set_status(AgentStatus.IDLE)
                        self._agent_busy_since.pop(a_id, None)
                        self.record_heartbeat(a_id)
                        action = {
                            "target_type": "agent",
                            "target_id": a_id,
                            "swarm_id": s_id,
                            "action": "reset_stuck_agent",
                            "new_status": AgentStatus.IDLE.value,
                        }
                        actions_taken.append(action)
                        logger.warning("Auto-healed stuck agent '%s' -> IDLE", a_id)

                    # If agent was faulted in ERROR
                    elif a_id in health.faulted_agents:
                        agent.set_status(AgentStatus.IDLE)
                        self.record_heartbeat(a_id)
                        action = {
                            "target_type": "agent",
                            "target_id": a_id,
                            "swarm_id": s_id,
                            "action": "recover_faulted_agent",
                            "new_status": AgentStatus.IDLE.value,
                        }
                        actions_taken.append(action)
                        logger.warning("Auto-healed faulted agent '%s' -> IDLE", a_id)

            return actions_taken

    def get_load_distribution(self) -> Dict[str, float]:
        """Compute the current workload distribution across managed swarms and agents.

        Returns:
            Dictionary mapping agent_id or swarm_id to normalized load metric (0.0 - 1.0).
        """
        with self._lock:
            distribution: Dict[str, float] = {}

            for s_id, swarm in self._swarms.items():
                total_agents = 0
                busy_agents = 0

                if hasattr(swarm, "registry") and hasattr(swarm.registry, "list_agents"):
                    for agent in swarm.registry.list_agents():
                        total_agents += 1
                        is_busy = agent.status == AgentStatus.BUSY
                        if is_busy:
                            busy_agents += 1
                        distribution[agent.agent_id] = 1.0 if is_busy else 0.0

                swarm_load = (busy_agents / total_agents) if total_agents > 0 else 0.0
                distribution[s_id] = round(swarm_load, 4)

            return distribution
