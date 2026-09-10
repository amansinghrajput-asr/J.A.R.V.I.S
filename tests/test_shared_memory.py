"""Unit Tests for Shared Execution Memory and Artifact Store (Sprint 18.3)."""

import threading
import unittest

from app.ai.planner.multi_agent.memory import ArtifactStore, SharedAgentMemory


class TestSharedAgentMemory(unittest.TestCase):
    """Test suite for SharedAgentMemory and ArtifactStore."""

    def test_global_store_crud(self) -> None:
        """Verify global blackboard get, set, delete, and version increments."""
        mem = SharedAgentMemory()
        self.assertFalse(mem.has("query_result"))
        self.assertEqual(mem.get("query_result", "none"), "none")

        v1 = mem.set("query_result", 42)
        self.assertEqual(v1, 1)
        self.assertTrue(mem.has("query_result"))
        self.assertEqual(mem.get("query_result"), 42)

        v2 = mem.set("query_result", 43)
        self.assertEqual(v2, 2)
        self.assertEqual(mem.get_version("query_result"), 2)

        deleted = mem.delete("query_result")
        self.assertTrue(deleted)
        self.assertFalse(mem.has("query_result"))

    def test_optimistic_concurrency_versioning(self) -> None:
        """Verify set_with_version detects version mismatches."""
        mem = SharedAgentMemory()
        mem.set("shared_counter", 100)
        curr_v = mem.get_version("shared_counter")
        self.assertEqual(curr_v, 1)

        # Successful update with correct expected version
        success, new_v = mem.set_with_version("shared_counter", 101, expected_version=1)
        self.assertTrue(success)
        self.assertEqual(new_v, 2)
        self.assertEqual(mem.get("shared_counter"), 101)

        # Stale update attempt with old version
        stale_success, v_stale = mem.set_with_version("shared_counter", 999, expected_version=1)
        self.assertFalse(stale_success)
        self.assertEqual(v_stale, 2)
        self.assertEqual(mem.get("shared_counter"), 101)

    def test_agent_and_task_scoped_storage(self) -> None:
        """Verify agent and task private namespaces remain isolated."""
        mem = SharedAgentMemory()

        mem.set_agent_data("agent_a", "scratchpad", "notes A")
        mem.set_agent_data("agent_b", "scratchpad", "notes B")

        self.assertEqual(mem.get_agent_data("agent_a", "scratchpad"), "notes A")
        self.assertEqual(mem.get_agent_data("agent_b", "scratchpad"), "notes B")
        self.assertIn("scratchpad", mem.list_agent_keys("agent_a"))

        mem.set_task_data("task_1", "raw_response", {"status": 200})
        self.assertEqual(mem.get_task_data("task_1", "raw_response")["status"], 200)

    def test_artifact_store_crud(self) -> None:
        """Verify putting, getting, and listing artifacts."""
        store = ArtifactStore()
        art_id = store.put_artifact(
            artifact_id="art_report_1",
            content="# Findings\nAll systems nominal.",
            created_by="agent_reporter",
            metadata={"format": "markdown"},
        )
        self.assertEqual(art_id, "art_report_1")

        item = store.get_artifact("art_report_1")
        self.assertIsNotNone(item)
        self.assertEqual(item["created_by"], "agent_reporter")
        self.assertIn("All systems nominal", item["content"])

        all_artifacts = store.list_artifacts()
        self.assertEqual(len(all_artifacts), 1)

    def test_serialization_and_deserialization(self) -> None:
        """Verify SharedAgentMemory to_dict and from_dict roundtrip."""
        mem = SharedAgentMemory()
        mem.set("config_key", "val_1")
        mem.set_agent_data("agent_x", "private_k", "private_v")
        mem.artifacts.put_artifact("art_1", "code content", created_by="coder")

        d = mem.to_dict()
        restored = SharedAgentMemory.from_dict(d)

        self.assertEqual(restored.get("config_key"), "val_1")
        self.assertEqual(restored.get_agent_data("agent_x", "private_k"), "private_v")
        art = restored.artifacts.get_artifact("art_1")
        self.assertIsNotNone(art)
        self.assertEqual(art["content"], "code content")

    def test_concurrent_access_thread_safety(self) -> None:
        """Verify thread safety during concurrent reads and writes."""
        mem = SharedAgentMemory()
        errors = []

        def worker(w_id: int) -> None:
            try:
                for i in range(100):
                    mem.set(f"key_{w_id}_{i}", i)
                    mem.get(f"key_{w_id}_{i}")
                    mem.set_agent_data(f"ag_{w_id}", "step", i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=3.0)

        self.assertEqual(len(errors), 0)


if __name__ == "__main__":
    unittest.main()
