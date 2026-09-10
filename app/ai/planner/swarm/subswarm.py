"""SubSwarm and SubSwarmManager implementation for Phase 19.1.

Provides isolated, thread-safe, and async-compatible child swarm instantiation,
recursive depth enforcement, memory enclave isolation, and hierarchical result roll-up.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from app.ai.planner.multi_agent.memory import SharedAgentMemory
from app.ai.planner.multi_agent.protocol import AgentCommunicationBus
from app.ai.planner.multi_agent.registry import AgentRegistry
from app.ai.planner.swarm.models import (
    SubSwarmConfig,
    SwarmExecutionResult,
    SwarmHierarchyNode,
    SwarmStatus,
)

logger = logging.getLogger("app.ai.planner.swarm.subswarm")


class MaxRecursionDepthExceededError(Exception):
    """Raised when a sub-swarm attempts to spawn a child beyond the allowed maximum depth."""

    pass


class SubSwarm:
    """An autonomous, isolated sub-swarm enclave in a hierarchical swarm network.

    Every SubSwarm owns:
    - An isolated AgentCommunicationBus
    - An isolated AgentRegistry
    - An isolated SharedAgentMemory
    - An optional injected coordinator reference
    """

    def __init__(
        self,
        config: Optional[SubSwarmConfig] = None,
        bus: Optional[AgentCommunicationBus] = None,
        registry: Optional[AgentRegistry] = None,
        memory: Optional[SharedAgentMemory] = None,
        coordinator: Optional[Any] = None,
    ) -> None:
        """Initialize a SubSwarm enclave.

        All resources are injected or created afresh to guarantee zero shared mutable state.
        """
        self._lock = threading.RLock()
        self.config = config or SubSwarmConfig()
        self.bus = bus or AgentCommunicationBus()
        self.registry = registry or AgentRegistry()
        self.memory = memory or SharedAgentMemory()
        self.coordinator = coordinator
        self.status = SwarmStatus.READY
        self._children: Dict[str, SubSwarm] = {}

    @property
    def swarm_id(self) -> str:
        """Return swarm identifier."""
        return self.config.swarm_id

    @property
    def depth(self) -> int:
        """Return swarm hierarchy depth."""
        return self.config.depth

    def spawn_child(
        self,
        child_config: Optional[SubSwarmConfig] = None,
        max_depth: int = 3,
        coordinator: Optional[Any] = None,
    ) -> SubSwarm:
        """Spawn an isolated child sub-swarm with depth boundary checking.

        Args:
            child_config: Optional custom configuration for the child.
            max_depth: Maximum permitted recursion depth.
            coordinator: Optional coordinator reference for the child.

        Returns:
            Newly initialized isolated child SubSwarm.

        Raises:
            MaxRecursionDepthExceededError: If spawning exceeds max_depth.
        """
        with self._lock:
            target_depth = self.config.depth + 1
            if target_depth > max_depth:
                raise MaxRecursionDepthExceededError(
                    f"Cannot spawn child swarm at depth {target_depth}. "
                    f"Maximum recursion depth limit of {max_depth} reached."
                )

            if child_config is None:
                effective_config = SubSwarmConfig(
                    parent_swarm_id=self.config.swarm_id,
                    depth=target_depth,
                    max_concurrency=self.config.max_concurrency,
                    timeout_seconds=self.config.timeout_seconds,
                    isolated_memory=True,
                )
            else:
                effective_config = SubSwarmConfig(
                    swarm_id=child_config.swarm_id,
                    parent_swarm_id=self.config.swarm_id,
                    depth=target_depth,
                    max_concurrency=child_config.max_concurrency,
                    timeout_seconds=child_config.timeout_seconds,
                    isolated_memory=child_config.isolated_memory,
                )

            # Guarantee 100% memory and bus isolation
            child_bus = AgentCommunicationBus()
            child_registry = AgentRegistry()
            child_memory = SharedAgentMemory()

            child_swarm = SubSwarm(
                config=effective_config,
                bus=child_bus,
                registry=child_registry,
                memory=child_memory,
                coordinator=coordinator if coordinator is not None else self.coordinator,
            )

            self._children[child_swarm.swarm_id] = child_swarm
            logger.info(
                "Swarm '%s' spawned child swarm '%s' at depth %d",
                self.swarm_id,
                child_swarm.swarm_id,
                child_swarm.depth,
            )
            return child_swarm

    def get_child(self, child_id: str) -> Optional[SubSwarm]:
        """Retrieve direct child swarm by ID."""
        with self._lock:
            return self._children.get(child_id)

    def list_children(self) -> List[SubSwarm]:
        """Return list of all direct child swarms."""
        with self._lock:
            return list(self._children.values())

    def execute_subplan(
        self,
        plan_or_tasks: Any,
        timeout: Optional[float] = None,
    ) -> SwarmExecutionResult:
        """Execute a subplan within this swarm and aggregate child results.

        Args:
            plan_or_tasks: Target execution payload (tasks, dict, or runnable callable).
            timeout: Optional execution timeout override.

        Returns:
            SwarmExecutionResult containing outputs and rolled-up child results.
        """
        start_time = time.perf_counter()
        with self._lock:
            if self.status == SwarmStatus.TERMINATED:
                return SwarmExecutionResult(
                    success=False,
                    outputs={"error": f"Swarm '{self.swarm_id}' is terminated."},
                    duration=0.0,
                )
            self.status = SwarmStatus.RUNNING

        try:
            outputs: Dict[str, Any] = {}
            # Execute through coordinator if available
            if self.coordinator is not None:
                if hasattr(self.coordinator, "execute_subplan"):
                    outputs = self.coordinator.execute_subplan(plan_or_tasks, timeout=timeout)
                elif hasattr(self.coordinator, "execute_tasks"):
                    outputs = self.coordinator.execute_tasks(plan_or_tasks)
                elif callable(self.coordinator):
                    outputs = self.coordinator(plan_or_tasks)
            elif callable(plan_or_tasks):
                outputs = plan_or_tasks()
            elif isinstance(plan_or_tasks, dict):
                outputs = dict(plan_or_tasks)
            else:
                outputs = {"payload": str(plan_or_tasks)}

            # Roll up results from direct children
            child_results: List[SwarmExecutionResult] = []
            with self._lock:
                for child in self._children.values():
                    # If child already has execution state or can be queried
                    if child.status != SwarmStatus.TERMINATED:
                        child_res = child.execute_subplan(
                            plan_or_tasks={}, timeout=timeout
                        )
                        child_results.append(child_res)

            elapsed = time.perf_counter() - start_time
            with self._lock:
                self.status = SwarmStatus.READY

            return SwarmExecutionResult(
                success=True,
                outputs=outputs if isinstance(outputs, dict) else {"result": outputs},
                metrics={"depth": self.depth, "children_count": len(self._children)},
                duration=elapsed,
                child_results=child_results,
            )

        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            with self._lock:
                self.status = SwarmStatus.FAILED
            logger.error("Swarm '%s' execution failed: %s", self.swarm_id, exc)
            return SwarmExecutionResult(
                success=False,
                outputs={"error": str(exc)},
                metrics={"depth": self.depth, "children_count": len(self._children)},
                duration=elapsed,
                child_results=[],
            )

    async def execute_subplan_async(
        self,
        plan_or_tasks: Any,
        timeout: Optional[float] = None,
    ) -> SwarmExecutionResult:
        """Asynchronously execute a subplan within this swarm and aggregate child results.

        Args:
            plan_or_tasks: Target execution payload.
            timeout: Optional execution timeout override.

        Returns:
            SwarmExecutionResult containing outputs and rolled-up child results.
        """
        start_time = time.perf_counter()
        with self._lock:
            if self.status == SwarmStatus.TERMINATED:
                return SwarmExecutionResult(
                    success=False,
                    outputs={"error": f"Swarm '{self.swarm_id}' is terminated."},
                    duration=0.0,
                )
            self.status = SwarmStatus.RUNNING

        try:
            outputs: Dict[str, Any] = {}
            if self.coordinator is not None:
                if hasattr(self.coordinator, "execute_subplan_async"):
                    outputs = await self.coordinator.execute_subplan_async(
                        plan_or_tasks, timeout=timeout
                    )
                elif hasattr(self.coordinator, "execute_subplan"):
                    outputs = self.coordinator.execute_subplan(
                        plan_or_tasks, timeout=timeout
                    )
                elif callable(self.coordinator):
                    res = self.coordinator(plan_or_tasks)
                    outputs = await res if asyncio.iscoroutine(res) else res
            elif callable(plan_or_tasks):
                res = plan_or_tasks()
                outputs = await res if asyncio.iscoroutine(res) else res
            elif isinstance(plan_or_tasks, dict):
                outputs = dict(plan_or_tasks)
            else:
                outputs = {"payload": str(plan_or_tasks)}

            # Roll up results from direct children asynchronously
            child_results: List[SwarmExecutionResult] = []
            children_to_run: List[SubSwarm] = []
            with self._lock:
                children_to_run = [
                    c for c in self._children.values() if c.status != SwarmStatus.TERMINATED
                ]

            for child in children_to_run:
                c_res = await child.execute_subplan_async({}, timeout=timeout)
                child_results.append(c_res)

            elapsed = time.perf_counter() - start_time
            with self._lock:
                self.status = SwarmStatus.READY

            return SwarmExecutionResult(
                success=True,
                outputs=outputs if isinstance(outputs, dict) else {"result": outputs},
                metrics={"depth": self.depth, "children_count": len(self._children)},
                duration=elapsed,
                child_results=child_results,
            )

        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            with self._lock:
                self.status = SwarmStatus.FAILED
            logger.error("Async swarm '%s' execution failed: %s", self.swarm_id, exc)
            return SwarmExecutionResult(
                success=False,
                outputs={"error": str(exc)},
                metrics={"depth": self.depth, "children_count": len(self._children)},
                duration=elapsed,
                child_results=[],
            )

    def terminate(self) -> None:
        """Recursively terminate this swarm and all its descendant child swarms."""
        with self._lock:
            for child in list(self._children.values()):
                child.terminate()
            self.status = SwarmStatus.TERMINATED
            logger.info("Swarm '%s' terminated", self.swarm_id)


class SubSwarmManager:
    """Thread-safe registry and manager for hierarchical sub-swarms."""

    def __init__(self) -> None:
        """Initialize the SubSwarmManager."""
        self._lock = threading.RLock()
        self._swarms: Dict[str, SubSwarm] = {}

    def register_swarm(self, swarm: SubSwarm) -> None:
        """Register a sub-swarm in the manager.

        Args:
            swarm: SubSwarm instance to register.
        """
        with self._lock:
            self._swarms[swarm.swarm_id] = swarm

    def get_swarm(self, swarm_id: str) -> Optional[SubSwarm]:
        """Retrieve a registered sub-swarm by ID.

        Args:
            swarm_id: Identifier of the target swarm.

        Returns:
            SubSwarm instance or None if not registered.
        """
        with self._lock:
            return self._swarms.get(swarm_id)

    def unregister_swarm(self, swarm_id: str) -> Optional[SubSwarm]:
        """Remove and return a registered sub-swarm."""
        with self._lock:
            return self._swarms.pop(swarm_id, None)

    def get_hierarchy_tree(self, root_swarm_id: str) -> Optional[SwarmHierarchyNode]:
        """Construct the hierarchy metadata tree rooted at root_swarm_id.

        Args:
            root_swarm_id: Root swarm identifier.

        Returns:
            SwarmHierarchyNode with linked child IDs, or None if root not found.
        """
        with self._lock:
            root_swarm = self._swarms.get(root_swarm_id)
            if root_swarm is None:
                return None

            return SwarmHierarchyNode(
                swarm_id=root_swarm.swarm_id,
                parent=root_swarm.config.parent_swarm_id,
                children=[c.swarm_id for c in root_swarm.list_children()],
                depth=root_swarm.depth,
            )

    def terminate_all(self) -> None:
        """Terminate all registered swarms."""
        with self._lock:
            for swarm in list(self._swarms.values()):
                swarm.terminate()
            self._swarms.clear()
