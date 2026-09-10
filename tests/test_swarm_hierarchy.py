"""Unit tests for Hierarchical SubSwarm & Decomposer (Sprint 19.1).

Covers child spawning, recursion depth limits, memory isolation, parent-child linkage,
result aggregation, async execution, recursive termination, and task decomposition.
"""

import asyncio
import unittest

from app.ai.planner.models import Task
from app.ai.planner.swarm.decomposer import CompositeTaskDecomposer
from app.ai.planner.swarm.models import (
    CompositeTask,
    SubSwarmConfig,
    SwarmStatus,
)
from app.ai.planner.swarm.subswarm import (
    MaxRecursionDepthExceededError,
    SubSwarm,
    SubSwarmManager,
)


class TestSwarmHierarchy(unittest.TestCase):
    """Test suite for hierarchical sub-swarms and decomposition."""

    def test_child_spawning_and_linkage(self) -> None:
        """Verify root swarm spawns child and correctly links depth and parent ID."""
        root = SubSwarm(config=SubSwarmConfig(swarm_id="root_swarm", depth=0))
        child = root.spawn_child(max_depth=3)

        self.assertEqual(child.depth, 1)
        self.assertEqual(child.config.parent_swarm_id, "root_swarm")
        self.assertIn(child.swarm_id, [c.swarm_id for c in root.list_children()])
        self.assertEqual(root.get_child(child.swarm_id), child)

    def test_recursion_depth_enforcement(self) -> None:
        """Verify MaxRecursionDepthExceededError is raised when exceeding max_depth."""
        root = SubSwarm(config=SubSwarmConfig(swarm_id="root", depth=0))
        depth_1 = root.spawn_child(max_depth=2)
        depth_2 = depth_1.spawn_child(max_depth=2)

        self.assertEqual(depth_2.depth, 2)
        with self.assertRaises(MaxRecursionDepthExceededError):
            depth_2.spawn_child(max_depth=2)

    def test_isolated_memory_instances(self) -> None:
        """Verify child swarm has completely isolated memory from parent."""
        root = SubSwarm(config=SubSwarmConfig(swarm_id="root", depth=0))
        child = root.spawn_child()

        # Write to parent blackboard
        root.memory.set("shared_key", "parent_val")
        # Write to child blackboard
        child.memory.set("shared_key", "child_val")

        # Parent must retain parent_val
        self.assertEqual(root.memory.get("shared_key"), "parent_val")
        # Child must have child_val
        self.assertEqual(child.memory.get("shared_key"), "child_val")
        # Distinct memory instances
        self.assertIsNot(root.memory, child.memory)
        self.assertIsNot(root.bus, child.bus)
        self.assertIsNot(root.registry, child.registry)

    def test_execute_subplan_and_result_aggregation(self) -> None:
        """Verify parent rolls up execution outputs from child swarms."""
        root = SubSwarm(
            config=SubSwarmConfig(swarm_id="root"),
            coordinator=lambda payload: {"master_output": "data_analyzed"},
        )
        child = root.spawn_child(
            coordinator=lambda payload: {"child_output": "sub_analysis_complete"}
        )

        res = root.execute_subplan({"query": "analyze_market"})
        self.assertTrue(res.success)
        self.assertEqual(res.outputs.get("master_output"), "data_analyzed")
        self.assertEqual(len(res.child_results), 1)
        self.assertEqual(
            res.child_results[0].outputs.get("child_output"),
            "sub_analysis_complete",
        )
        self.assertEqual(root.status, SwarmStatus.READY)

    def test_execute_subplan_async(self) -> None:
        """Verify async execution of subplans with child aggregation."""
        async def _run_test() -> None:
            root = SubSwarm(
                config=SubSwarmConfig(swarm_id="root_async"),
                coordinator=lambda payload: {"async_master": "done"},
            )
            child = root.spawn_child(
                coordinator=lambda payload: {"async_child": "done"}
            )

            res = await root.execute_subplan_async({"query": "run_async"})
            self.assertTrue(res.success)
            self.assertEqual(res.outputs.get("async_master"), "done")
            self.assertEqual(len(res.child_results), 1)
            self.assertEqual(res.child_results[0].outputs.get("async_child"), "done")

        asyncio.run(_run_test())

    def test_recursive_termination(self) -> None:
        """Verify terminating parent recursively terminates all children."""
        root = SubSwarm(config=SubSwarmConfig(swarm_id="root"))
        child1 = root.spawn_child()
        child2 = child1.spawn_child(max_depth=3)

        self.assertEqual(root.status, SwarmStatus.READY)
        self.assertEqual(child1.status, SwarmStatus.READY)
        self.assertEqual(child2.status, SwarmStatus.READY)

        root.terminate()

        self.assertEqual(root.status, SwarmStatus.TERMINATED)
        self.assertEqual(child1.status, SwarmStatus.TERMINATED)
        self.assertEqual(child2.status, SwarmStatus.TERMINATED)

    def test_subswarm_manager_lifecycle(self) -> None:
        """Verify SubSwarmManager registers, tree inspection, and terminates all."""
        mgr = SubSwarmManager()
        root = SubSwarm(config=SubSwarmConfig(swarm_id="root_swarm", depth=0))
        child = root.spawn_child(max_depth=3)

        mgr.register_swarm(root)
        mgr.register_swarm(child)

        self.assertEqual(mgr.get_swarm("root_swarm"), root)
        self.assertEqual(mgr.get_swarm(child.swarm_id), child)

        tree = mgr.get_hierarchy_tree("root_swarm")
        self.assertIsNotNone(tree)
        self.assertEqual(tree.swarm_id, "root_swarm")
        self.assertEqual(tree.children, [child.swarm_id])

        mgr.terminate_all()
        self.assertEqual(root.status, SwarmStatus.TERMINATED)
        self.assertEqual(child.status, SwarmStatus.TERMINATED)
        self.assertIsNone(mgr.get_swarm("root_swarm"))

    def test_composite_task_decomposer(self) -> None:
        """Verify CompositeTaskDecomposer detection, flattening, and subplan creation."""
        decomposer = CompositeTaskDecomposer()

        t1 = Task(action="download", target="file.csv")
        t2 = Task(action="parse", target="file.csv")
        comp_inner = CompositeTask(
            task_id="comp_1",
            title="Parse File",
            children=[t2],
        )
        comp_root = CompositeTask(
            task_id="comp_root",
            title="ETL Pipeline",
            children=[t1, comp_inner],
        )

        self.assertTrue(decomposer.is_composite(comp_root))
        self.assertFalse(decomposer.is_composite(t1))

        # Direct children
        direct = decomposer.decompose(comp_root, flatten=False)
        self.assertEqual(len(direct), 2)

        # Flattened leaves
        leaves = decomposer.decompose(comp_root, flatten=True)
        self.assertEqual(len(leaves), 2)
        self.assertEqual(leaves[0].action, "download")
        self.assertEqual(leaves[1].action, "parse")

        # Subplan generation with sequential dependencies
        subplan = decomposer.build_subplan(comp_root, sequential_dependencies=True)
        self.assertEqual(subplan["composite_task_id"], "comp_root")
        self.assertEqual(subplan["task_count"], 2)
        self.assertEqual(len(subplan["tasks"]), 2)
        # Second task depends on first
        self.assertIn(t1.id, subplan["tasks"][1]["dependencies"])


if __name__ == "__main__":
    unittest.main()
