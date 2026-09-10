"""Unit tests for Hierarchical Swarm Models (Sprint 19.1).

Covers dataclass defaults, config creation, serialization, hierarchy metadata,
and immutability guarantees.
"""

import unittest
from dataclasses import FrozenInstanceError

from app.ai.planner.models import Task
from app.ai.planner.swarm.models import (
    CompositeTask,
    SubSwarmConfig,
    SwarmExecutionResult,
    SwarmHierarchyNode,
    SwarmStatus,
)


class TestSwarmModels(unittest.TestCase):
    """Test suite for swarm data models."""

    def test_swarm_status_enumeration(self) -> None:
        """Verify SwarmStatus values."""
        self.assertEqual(SwarmStatus.CREATED.value, "CREATED")
        self.assertEqual(SwarmStatus.READY.value, "READY")
        self.assertEqual(SwarmStatus.RUNNING.value, "RUNNING")
        self.assertEqual(SwarmStatus.PAUSED.value, "PAUSED")
        self.assertEqual(SwarmStatus.FAILED.value, "FAILED")
        self.assertEqual(SwarmStatus.TERMINATED.value, "TERMINATED")

    def test_subswarm_config_defaults_and_immutability(self) -> None:
        """Verify SubSwarmConfig default values and frozen immutability."""
        config = SubSwarmConfig()
        self.assertTrue(config.swarm_id.startswith("swarm_"))
        self.assertIsNone(config.parent_swarm_id)
        self.assertEqual(config.depth, 0)
        self.assertEqual(config.max_concurrency, 4)
        self.assertEqual(config.timeout_seconds, 60.0)
        self.assertTrue(config.isolated_memory)

        # Immutability check
        with self.assertRaises(FrozenInstanceError):
            config.depth = 2  # type: ignore[misc]

    def test_subswarm_config_serialization(self) -> None:
        """Verify round-trip dictionary serialization for SubSwarmConfig."""
        config = SubSwarmConfig(
            swarm_id="child_swarm_1",
            parent_swarm_id="root_swarm",
            depth=1,
            max_concurrency=8,
            timeout_seconds=45.5,
            isolated_memory=True,
        )
        d = config.to_dict()
        self.assertEqual(d["swarm_id"], "child_swarm_1")
        self.assertEqual(d["parent_swarm_id"], "root_swarm")
        self.assertEqual(d["depth"], 1)
        self.assertEqual(d["max_concurrency"], 8)
        self.assertEqual(d["timeout_seconds"], 45.5)
        self.assertTrue(d["isolated_memory"])

        restored = SubSwarmConfig.from_dict(d)
        self.assertEqual(restored, config)

    def test_composite_task_nesting_and_serialization(self) -> None:
        """Verify nested CompositeTask creation and recursive serialization."""
        atomic_task = Task(action="fetch_url", target="https://example.com")
        nested_child = CompositeTask(
            title="Sub-step",
            description="Process fetched data",
            children=[atomic_task],
            metadata={"priority": "high"},
        )
        root_composite = CompositeTask(
            title="Master Pipeline",
            description="Fetch and process",
            children=[nested_child],
            metadata={"owner": "research_team"},
        )

        self.assertEqual(len(root_composite.children), 1)
        first_child = root_composite.children[0]
        self.assertIsInstance(first_child, CompositeTask)
        self.assertEqual(len(first_child.children), 1)

        d = root_composite.to_dict()
        self.assertEqual(d["title"], "Master Pipeline")
        self.assertEqual(len(d["children"]), 1)
        self.assertEqual(d["children"][0]["title"], "Sub-step")
        self.assertEqual(d["children"][0]["children"][0]["action"], "fetch_url")

        restored = CompositeTask.from_dict(d)
        self.assertEqual(restored.title, "Master Pipeline")
        self.assertEqual(len(restored.children), 1)
        restored_child = restored.children[0]
        self.assertIsInstance(restored_child, CompositeTask)
        self.assertEqual(restored_child.title, "Sub-step")
        self.assertEqual(len(restored_child.children), 1)
        self.assertEqual(restored_child.children[0].action, "fetch_url")

    def test_swarm_hierarchy_node_serialization(self) -> None:
        """Verify SwarmHierarchyNode serialization and deserialization."""
        node = SwarmHierarchyNode(
            swarm_id="swarm_alpha",
            parent="swarm_root",
            children=["swarm_child_1", "swarm_child_2"],
            depth=1,
        )
        d = node.to_dict()
        self.assertEqual(d["swarm_id"], "swarm_alpha")
        self.assertEqual(d["parent"], "swarm_root")
        self.assertEqual(d["children"], ["swarm_child_1", "swarm_child_2"])
        self.assertEqual(d["depth"], 1)

        restored = SwarmHierarchyNode.from_dict(d)
        self.assertEqual(restored.swarm_id, node.swarm_id)
        self.assertEqual(restored.parent, node.parent)
        self.assertEqual(restored.children, node.children)
        self.assertEqual(restored.depth, node.depth)

    def test_swarm_execution_result_serialization(self) -> None:
        """Verify SwarmExecutionResult round-trip serialization with child results."""
        child_res = SwarmExecutionResult(
            success=True,
            outputs={"report": "child_done"},
            metrics={"cpu_time": 0.05},
            duration=0.1,
            child_results=[],
        )
        parent_res = SwarmExecutionResult(
            success=True,
            outputs={"final": "parent_done"},
            metrics={"subswarms": 1},
            duration=0.25,
            child_results=[child_res],
        )

        d = parent_res.to_dict()
        self.assertTrue(d["success"])
        self.assertEqual(len(d["child_results"]), 1)
        self.assertEqual(d["child_results"][0]["outputs"]["report"], "child_done")

        restored = SwarmExecutionResult.from_dict(d)
        self.assertTrue(restored.success)
        self.assertEqual(restored.outputs["final"], "parent_done")
        self.assertEqual(len(restored.child_results), 1)
        self.assertEqual(restored.child_results[0].outputs["report"], "child_done")
        self.assertEqual(restored.child_results[0].duration, 0.1)


if __name__ == "__main__":
    unittest.main()
