"""CompositeTaskDecomposer implementation for Phase 19.1.

Provides recursive, deterministic task decomposition and subplan structural generation
without execution or planner dependencies.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

from app.ai.planner.models import Task
from app.ai.planner.swarm.models import CompositeTask


class CompositeTaskDecomposer:
    """Decomposes composite tasks recursively into executable sub-task hierarchies."""

    def __init__(self) -> None:
        """Initialize task decomposer."""
        pass

    def is_composite(self, task: Any) -> bool:
        """Check if an object represents a composite task.

        Args:
            task: Task instance, CompositeTask instance, or dictionary.

        Returns:
            True if the task has children or composite markers, False otherwise.
        """
        if isinstance(task, CompositeTask):
            return True

        if isinstance(task, dict):
            if "children" in task and isinstance(task["children"], list):
                return len(task["children"]) > 0 or task.get("is_composite", False)
            return bool(task.get("is_composite", False))

        if hasattr(task, "children") and isinstance(getattr(task, "children"), list):
            return True

        return False

    def decompose(
        self,
        composite_task: CompositeTask,
        flatten: bool = False,
    ) -> List[Union[CompositeTask, Task, Dict[str, Any]]]:
        """Recursively decompose a composite task into its child components.

        Args:
            composite_task: Root composite task to decompose.
            flatten: If True, recursively flattens and returns only leaf atomic tasks.
                     If False, returns direct child components.

        Returns:
            Deterministic list of child tasks.
        """
        if not flatten:
            return list(composite_task.children)

        # Flatten leaf tasks recursively in deterministic order
        leaves: List[Union[CompositeTask, Task, Dict[str, Any]]] = []
        for child in composite_task.children:
            if isinstance(child, CompositeTask):
                if child.children:
                    leaves.extend(self.decompose(child, flatten=True))
                else:
                    leaves.append(child)
            elif isinstance(child, dict) and self.is_composite(child):
                child_comp = CompositeTask.from_dict(child)
                leaves.extend(self.decompose(child_comp, flatten=True))
            else:
                leaves.append(child)

        return leaves

    def build_subplan(
        self,
        composite_task: CompositeTask,
        sequential_dependencies: bool = False,
    ) -> Dict[str, Any]:
        """Construct a deterministic subplan representation from a composite task.

        Args:
            composite_task: Target composite task.
            sequential_dependencies: If True, automatically links children sequentially.

        Returns:
            Dictionary defining the subplan structure, task nodes, and dependency links.
        """
        subplan_tasks: List[Dict[str, Any]] = []
        prev_id: str = ""

        for idx, child in enumerate(composite_task.children):
            if isinstance(child, CompositeTask):
                child_dict = child.to_dict()
                task_id = child.task_id
            elif isinstance(child, Task):
                child_dict = child.to_dict()
                task_id = child.id
            elif isinstance(child, dict):
                child_dict = dict(child)
                task_id = str(child.get("id") or child.get("task_id", f"task_{idx}"))
            else:
                task_id = f"task_{idx}"
                child_dict = {"task_id": task_id, "action": str(child)}

            # Apply sequential dependency linkage if requested
            if sequential_dependencies and prev_id:
                deps = child_dict.get("dependencies", [])
                if prev_id not in deps:
                    child_dict["dependencies"] = list(deps) + [prev_id]

            subplan_tasks.append(child_dict)
            prev_id = task_id

        return {
            "composite_task_id": composite_task.task_id,
            "title": composite_task.title,
            "description": composite_task.description,
            "task_count": len(subplan_tasks),
            "tasks": subplan_tasks,
            "metadata": dict(composite_task.metadata),
        }
