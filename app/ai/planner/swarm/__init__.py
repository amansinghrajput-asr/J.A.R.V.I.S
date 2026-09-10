"""Hierarchical Swarm Foundation Subsystem (Phase 19.1).

Exports schemas, models, isolated sub-swarm containers, and task decomposers.
"""

from app.ai.planner.swarm.balancer import BalancingPolicy, DynamicLoadBalancer
from app.ai.planner.swarm.consensus import (
    ConsensusResult,
    ConsensusStrategy,
    SwarmConsensusEngine,
)
from app.ai.planner.swarm.decomposer import CompositeTaskDecomposer
from app.ai.planner.swarm.episodic import EpisodicMemoryStore, TrajectoryRecord
from app.ai.planner.swarm.experience import ExperienceSynthesizer
from app.ai.planner.swarm.models import (
    CompositeTask,
    SubSwarmConfig,
    SwarmExecutionResult,
    SwarmHierarchyNode,
    SwarmStatus,
)
from app.ai.planner.swarm.speculative import SpeculativeExecutor
from app.ai.planner.swarm.subswarm import (
    MaxRecursionDepthExceededError,
    SubSwarm,
    SubSwarmManager,
)
from app.ai.planner.swarm.supervisor import SwarmHealthReport, SwarmSupervisor

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
    "TrajectoryRecord",
    "EpisodicMemoryStore",
    "ExperienceSynthesizer",
    "SpeculativeExecutor",
    "ConsensusStrategy",
    "ConsensusResult",
    "SwarmConsensusEngine",
    "SwarmHealthReport",
    "SwarmSupervisor",
    "BalancingPolicy",
    "DynamicLoadBalancer",
]
