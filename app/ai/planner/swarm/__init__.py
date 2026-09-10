"""Hierarchical Swarm Foundation Subsystem (Phase 19.1).

Exports schemas, models, isolated sub-swarm containers, and task decomposers.
"""

from app.ai.planner.swarm.decomposer import CompositeTaskDecomposer
from app.ai.planner.swarm.models import (
    CompositeTask,
    SubSwarmConfig,
    SwarmExecutionResult,
    SwarmHierarchyNode,
    SwarmStatus,
)
from app.ai.planner.swarm.subswarm import (
    MaxRecursionDepthExceededError,
    SubSwarm,
    SubSwarmManager,
)

__all__ = [
    "SwarmStatus",
    "SubSwarmConfig",
    "CompositeTask",
    "SwarmHierarchyNode",
    "SwarmExecutionResult",
    "MaxRecursionDepthExceededError",
    "SubSwarm",
    "SubSwarmManager",
    "CompositeTaskDecomposer",
]
