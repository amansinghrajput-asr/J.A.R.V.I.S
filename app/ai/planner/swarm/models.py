"""Data models for Hierarchical Swarm Foundation (Phase 19.1).

Defines immutable lifecycle status, configuration, composite tasks,
hierarchy metadata, and execution result schemas for nested sub-swarms.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from app.ai.planner.models import Task


class SwarmStatus(str, Enum):
    """Lifecycle states for a SubSwarm instance."""

    CREATED = "CREATED"
    READY = "READY"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    FAILED = "FAILED"
    TERMINATED = "TERMINATED"


@dataclass(frozen=True)
class SubSwarmConfig:
    """Configuration descriptor for a SubSwarm.

    Attributes:
        swarm_id: Unique identifier for the sub-swarm.
        parent_swarm_id: Optional ID of the parent swarm (None for root).
        depth: Recursion depth level (0 for root swarm).
        max_concurrency: Maximum parallel task workers in this sub-swarm.
        timeout_seconds: Execution timeout budget for the sub-swarm.
        isolated_memory: Whether the sub-swarm maintains an isolated memory scope.
    """

    swarm_id: str = field(default_factory=lambda: f"swarm_{uuid.uuid4().hex[:8]}")
    parent_swarm_id: Optional[str] = None
    depth: int = 0
    max_concurrency: int = 4
    timeout_seconds: float = 60.0
    isolated_memory: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a dictionary."""
        return {
            "swarm_id": self.swarm_id,
            "parent_swarm_id": self.parent_swarm_id,
            "depth": self.depth,
            "max_concurrency": self.max_concurrency,
            "timeout_seconds": self.timeout_seconds,
            "isolated_memory": self.isolated_memory,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SubSwarmConfig:
        """Create configuration from dictionary."""
        return cls(
            swarm_id=data.get("swarm_id", f"swarm_{uuid.uuid4().hex[:8]}"),
            parent_swarm_id=data.get("parent_swarm_id"),
            depth=int(data.get("depth", 0)),
            max_concurrency=int(data.get("max_concurrency", 4)),
            timeout_seconds=float(data.get("timeout_seconds", 60.0)),
            isolated_memory=bool(data.get("isolated_memory", True)),
        )


@dataclass
class CompositeTask:
    """Represents a composite or recursive task in a hierarchical plan.

    Attributes:
        task_id: Unique task identifier.
        title: Human-readable task name.
        description: Detailed goal description.
        children: Child sub-tasks (either nested CompositeTask, atomic Task, or dict).
        metadata: Optional metadata attributes.
    """

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    children: List[Union[CompositeTask, Task, Dict[str, Any]]] = field(
        default_factory=list
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize composite task recursively to a dictionary."""
        child_dicts: List[Dict[str, Any]] = []
        for child in self.children:
            if isinstance(child, CompositeTask):
                child_dicts.append(child.to_dict())
            elif isinstance(child, Task):
                child_dicts.append(child.to_dict())
            elif isinstance(child, dict):
                child_dicts.append(dict(child))
            else:
                child_dicts.append({"value": str(child)})

        return {
            "task_id": self.task_id,
            "title": self.title,
            "description": self.description,
            "children": child_dicts,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CompositeTask:
        """Deserialize composite task recursively from a dictionary."""
        raw_children = data.get("children", [])
        parsed_children: List[Union[CompositeTask, Task, Dict[str, Any]]] = []
        for c in raw_children:
            if isinstance(c, dict):
                if "children" in c:
                    parsed_children.append(CompositeTask.from_dict(c))
                elif "action" in c:
                    parsed_children.append(Task.from_dict(c))
                else:
                    parsed_children.append(dict(c))
            else:
                parsed_children.append(c)

        return cls(
            task_id=data.get("task_id", str(uuid.uuid4())),
            title=data.get("title", ""),
            description=data.get("description", ""),
            children=parsed_children,
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class SwarmHierarchyNode:
    """Metadata node representing a swarm's position in the swarm hierarchy tree.

    Attributes:
        swarm_id: Identifier of this swarm.
        parent: Parent swarm identifier (None for root).
        children: List of child swarm identifiers.
        depth: Hierarchy depth level.
    """

    swarm_id: str
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    depth: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize hierarchy node to dictionary."""
        return {
            "swarm_id": self.swarm_id,
            "parent": self.parent,
            "children": list(self.children),
            "depth": self.depth,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SwarmHierarchyNode:
        """Deserialize hierarchy node from dictionary."""
        return cls(
            swarm_id=data["swarm_id"],
            parent=data.get("parent"),
            children=list(data.get("children", [])),
            depth=int(data.get("depth", 0)),
        )


@dataclass
class SwarmExecutionResult:
    """Aggregated execution result of a SubSwarm.

    Attributes:
        success: Whether the execution completed successfully.
        outputs: Key-value map of execution outputs.
        metrics: Performance and operational metrics.
        duration: Total execution duration in seconds.
        child_results: Aggregated results from child sub-swarms.
    """

    success: bool
    outputs: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    child_results: List[SwarmExecutionResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize execution result recursively to dictionary."""
        return {
            "success": self.success,
            "outputs": dict(self.outputs),
            "metrics": dict(self.metrics),
            "duration": self.duration,
            "child_results": [r.to_dict() for r in self.child_results],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SwarmExecutionResult:
        """Deserialize execution result recursively from dictionary."""
        children = [
            cls.from_dict(c) for c in data.get("child_results", [])
        ]
        return cls(
            success=bool(data.get("success", False)),
            outputs=dict(data.get("outputs", {})),
            metrics=dict(data.get("metrics", {})),
            duration=float(data.get("duration", 0.0)),
            child_results=children,
        )
