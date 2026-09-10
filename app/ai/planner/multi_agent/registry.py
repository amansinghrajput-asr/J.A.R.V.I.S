"""Agent Registry & Dynamic Agent Selection for Phase 18.0.

Provides thread-safe agent registration, capability discovery,
and multi-criteria dynamic agent selection.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Set

from app.ai.planner.events import AgentDeregistered, AgentRegistered, PlannerEventBus
from app.ai.planner.multi_agent.base import BaseAgent
from app.ai.planner.multi_agent.models import AgentRole, AgentStatus

logger = logging.getLogger("app.ai.planner.multi_agent.registry")


class AgentRegistry:
    """Thread-safe registry managing agent instances, manifests, and capability indices."""

    def __init__(self, event_bus: Optional[PlannerEventBus] = None) -> None:
        """Initialize agent registry.

        Args:
            event_bus: Optional PlannerEventBus for registration lifecycle events.
        """
        self._lock = threading.RLock()
        self._event_bus = event_bus
        self._agents: Dict[str, BaseAgent] = {}
        # Inverted indices
        self._role_index: Dict[AgentRole, Set[str]] = {}
        self._action_index: Dict[str, Set[str]] = {}
        self._domain_index: Dict[str, Set[str]] = {}

    def register(self, agent: BaseAgent) -> None:
        """Register an agent in the registry and populate capability indices.

        Args:
            agent: Agent instance to register.

        Raises:
            ValueError: If an agent with the same ID is already registered.
        """
        if not isinstance(agent, BaseAgent):
            raise TypeError(f"Expected BaseAgent instance, got {type(agent).__name__}")

        with self._lock:
            if agent.agent_id in self._agents:
                raise ValueError(f"Agent with ID '{agent.agent_id}' is already registered.")

            self._agents[agent.agent_id] = agent

            # Index by role
            self._role_index.setdefault(agent.role, set()).add(agent.agent_id)

            # Index by capabilities
            cap_names: List[str] = []
            for cap in agent.manifest.capabilities:
                cap_names.append(cap.name)
                for action in cap.supported_actions:
                    self._action_index.setdefault(action.strip().lower(), set()).add(agent.agent_id)
                for domain in cap.domain_tags:
                    self._domain_index.setdefault(domain.strip().lower(), set()).add(agent.agent_id)

            # Also index allowed_actions if defined
            for action in agent.manifest.allowed_actions:
                self._action_index.setdefault(action.strip().lower(), set()).add(agent.agent_id)

            logger.info("Registered agent '%s' (role: %s)", agent.agent_id, agent.role.value)

        if self._event_bus is not None:
            self._event_bus.publish(
                AgentRegistered(
                    agent_id=agent.agent_id,
                    name=agent.name,
                    role=agent.role.value if hasattr(agent.role, "value") else str(agent.role),
                    capabilities=cap_names,
                )
            )

    def unregister(self, agent_id: str, reason: str = "") -> bool:
        """Unregister an agent and clean up indices.

        Returns:
            True if agent was removed, False if not found.
        """
        with self._lock:
            agent = self._agents.pop(agent_id, None)
            if agent is None:
                return False

            # Remove from role index
            if agent.role in self._role_index:
                self._role_index[agent.role].discard(agent_id)
                if not self._role_index[agent.role]:
                    del self._role_index[agent.role]

            # Remove from action index
            for s in self._action_index.values():
                s.discard(agent_id)

            # Remove from domain index
            for s in self._domain_index.values():
                s.discard(agent_id)

            logger.info("Unregistered agent '%s'", agent_id)

        if self._event_bus is not None:
            self._event_bus.publish(
                AgentDeregistered(
                    agent_id=agent_id,
                    reason=reason,
                )
            )

        return True

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Retrieve agent by unique ID."""
        with self._lock:
            return self._agents.get(agent_id)

    def list_agents(
        self,
        role: Optional[AgentRole] = None,
        status: Optional[AgentStatus] = None,
    ) -> List[BaseAgent]:
        """List registered agents with optional role and status filters."""
        with self._lock:
            agents = list(self._agents.values())

        if role is not None:
            agents = [a for a in agents if a.role == role]
        if status is not None:
            agents = [a for a in agents if a.status == status]

        return agents

    def find_agents_for_action(self, action: str) -> List[BaseAgent]:
        """Find all registered agents declaring support for a specific task action."""
        clean_action = action.strip().lower()
        with self._lock:
            agent_ids = self._action_index.get(clean_action, set())
            return [self._agents[aid] for aid in agent_ids if aid in self._agents]

    def find_agents_by_domain(self, domain: str) -> List[BaseAgent]:
        """Find all registered agents declaring interest in a domain tag."""
        clean_domain = domain.strip().lower()
        with self._lock:
            agent_ids = self._domain_index.get(clean_domain, set())
            return [self._agents[aid] for aid in agent_ids if aid in self._agents]

    def count(self) -> int:
        """Total number of registered agents."""
        with self._lock:
            return len(self._agents)

    def get_capabilities_summary(self) -> Dict[str, Any]:
        """Return structured summary of registered capabilities across all agents."""
        with self._lock:
            return {
                "total_agents": len(self._agents),
                "roles": {r.value: len(ids) for r, ids in self._role_index.items()},
                "supported_actions": {act: len(ids) for act, ids in self._action_index.items() if ids},
                "domains": {dom: len(ids) for dom, ids in self._domain_index.items() if ids},
            }


class AgentSelector:
    """Evaluates task requirements and selects the optimal specialized agent."""

    def __init__(
        self,
        registry: AgentRegistry,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialize selector.

        Args:
            registry: AgentRegistry to select agents from.
            weights: Optional custom scoring weights (action, domain, availability, confidence).
        """
        self._registry = registry
        self._weights = weights or {
            "action": 0.40,
            "domain": 0.20,
            "availability": 0.25,
            "confidence": 0.15,
        }

    def select_agent(
        self,
        action: str,
        target: Optional[str] = None,
        domain: Optional[str] = None,
        preferred_role: Optional[AgentRole] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[BaseAgent]:
        """Dynamically select the optimal agent for a given task.

        Args:
            action: Required task action name.
            target: Optional target operand.
            domain: Optional domain tag.
            preferred_role: Optional preferred AgentRole.
            context: Additional task context metadata.

        Returns:
            The highest scoring BaseAgent, or None if no candidate can perform the action.
        """
        candidates = self._registry.find_agents_for_action(action)
        if not candidates:
            # Fallback: check general WORKER agents if any declare wildcard or general capabilities
            workers = self._registry.list_agents(role=AgentRole.WORKER)
            candidates = [w for w in workers if not w.manifest.allowed_actions or action in w.manifest.allowed_actions]

        if not candidates:
            return None

        # Filter out offline or terminated agents
        active_candidates = [
            c for c in candidates
            if c.status not in (AgentStatus.FAILED, AgentStatus.TERMINATED)
        ]
        if not active_candidates:
            return None

        clean_domain = domain.strip().lower() if domain else None

        best_agent: Optional[BaseAgent] = None
        best_score = -1.0

        for agent in active_candidates:
            score = self._compute_score(agent, action, clean_domain, preferred_role)
            if score > best_score:
                best_score = score
                best_agent = agent

        return best_agent

    def _compute_score(
        self,
        agent: BaseAgent,
        action: str,
        domain: Optional[str],
        preferred_role: Optional[AgentRole],
    ) -> float:
        """Calculate normalized multi-criteria suitability score [0.0, 1.0]."""
        w = self._weights

        # 1. Action Match Score
        action_match = 1.0 if any(action.lower() in cap.supported_actions for cap in agent.manifest.capabilities) else 0.5

        # 2. Domain Match Score
        domain_match = 0.5
        if domain:
            has_domain = any(domain in cap.domain_tags for cap in agent.manifest.capabilities)
            domain_match = 1.0 if has_domain else 0.0

        # 3. Availability Score
        availability = 0.0
        if agent.status == AgentStatus.IDLE:
            availability = 1.0
        elif agent.status == AgentStatus.BUSY:
            availability = 0.5
        elif agent.status == AgentStatus.PAUSED:
            availability = 0.2

        # 4. Confidence Score
        confidence = 0.8
        for cap in agent.manifest.capabilities:
            if action.lower() in cap.supported_actions:
                confidence = max(confidence, cap.confidence_score)

        # 5. Role Alignment Bonus
        role_bonus = 0.1 if preferred_role and agent.role == preferred_role else 0.0

        total = (
            w.get("action", 0.4) * action_match
            + w.get("domain", 0.2) * domain_match
            + w.get("availability", 0.25) * availability
            + w.get("confidence", 0.15) * confidence
            + role_bonus
        )
        return total
