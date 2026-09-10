"""Unit Tests for Planner Execution Persistence (Sprint 17.1).

Verifies PersistedExecutionState serialization, deserialization,
transient cache exclusion, dynamic cache reconstruction, and resume invariants.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import List

from app.ai.planner.executor import Executor
from app.ai.planner.memory import ExecutionMemory, FailureCategory, TaskExecutionRecord
from app.ai.planner.models import ExecutionResult, Plan, Task, TaskStatus
from app.ai.planner.persistence import (
    PersistedExecutionState,
    load,
    resume,
    save,
)


class TestPlannerPersistence(unittest.TestCase):
    """Test suite for execution persistence and resumption."""

    def setUp(self) -> None:
        """Create sample tasks, memory, and persisted state."""
        self.t1 = Task(id="t1", action="open_app", target="chrome", status=TaskStatus.COMPLETED)
        self.t2 = Task(id="t2", action="web_search", target="weather", status=TaskStatus.FAILED, dependencies=["t1"])
        self.t3 = Task(id="t3", action="calculate", target="2+2", status=TaskStatus.SKIPPED, dependencies=["t2"])

        record1 = TaskExecutionRecord(
            task_id="t1",
            action="open_app",
            target="chrome",
            status=TaskStatus.COMPLETED,
            wave=1,
            attempt=1,
            output="Opened chrome",
            duration=0.05,
        )
        record2 = TaskExecutionRecord(
            task_id="t2",
            action="web_search",
            target="weather",
            status=TaskStatus.FAILED,
            wave=1,
            attempt=1,
            error="Network error",
            failure_category=FailureCategory.PROVIDER_ERROR,
            duration=0.10,
        )

        self.memory = ExecutionMemory(
            records=[record1, record2],
            dependency_failures={"t3": ["t2"]},
        )

        # Trigger internal lazy cache computation
        _ = self.memory.completed_task_ids
        _ = self.memory.failed_task_ids
        self.assertTrue(len(self.memory._cache) > 0, "Cache should be populated before serialization.")

        self.state = PersistedExecutionState(
            execution_id="exec_123",
            plan_id="plan_abc",
            query="open chrome then search weather and calculate 2+2",
            memory=self.memory,
            dag_state={
                "task_map": {"t1": self.t1, "t2": self.t2, "t3": self.t3},
                "completed_ids": ["t1"],
                "failed_ids": ["t2"],
                "skipped_ids": ["t3"],
            },
            completed_tasks=[self.t1],
            failed_tasks=[self.t2],
            skipped_tasks=[self.t3],
            retry_history={"t1": 1, "t2": 1},
            recovery_attempts=0,
            metadata={"source": "test"},
        )

    def test_serialization_to_dict_and_json(self) -> None:
        """Verify serialization produces correct schema without transient caches."""
        data = self.state.to_dict()
        self.assertEqual(data["execution_id"], "exec_123")
        self.assertEqual(data["plan_id"], "plan_abc")
        self.assertEqual(data["query"], self.state.query)
        self.assertEqual(len(data["completed_tasks"]), 1)
        self.assertEqual(len(data["failed_tasks"]), 1)
        self.assertEqual(len(data["skipped_tasks"]), 1)
        self.assertNotIn("_cache", data["memory"], "Transient caches must not be serialized.")

        json_str = self.state.to_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["execution_id"], "exec_123")

    def test_deserialization_rebuilds_caches_dynamically(self) -> None:
        """Verify deserialized memory starts with an empty cache and rebuilds derived properties."""
        json_str = self.state.to_json()
        loaded_state = PersistedExecutionState.from_json(json_str)

        self.assertEqual(loaded_state.execution_id, "exec_123")
        self.assertEqual(loaded_state.plan_id, "plan_abc")
        self.assertEqual(len(loaded_state.completed_tasks), 1)
        self.assertEqual(loaded_state.completed_tasks[0].id, "t1")

        # Memory cache should be empty upon reload
        self.assertEqual(len(loaded_state.memory._cache), 0, "Cache must be empty immediately after reload.")

        # Accessing properties dynamically populates cache
        self.assertEqual(loaded_state.memory.completed_task_ids, {"t1"})
        self.assertEqual(loaded_state.memory.failed_task_ids, {"t2"})
        self.assertIn("completed_task_ids", loaded_state.memory._cache)

    def test_save_and_load_file(self) -> None:
        """Verify save to file and load from file path and stream."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "state.json"
            save(self.state, file_path)
            self.assertTrue(file_path.exists())

            # Load from Path
            loaded = load(file_path)
            self.assertEqual(loaded.execution_id, "exec_123")
            self.assertEqual(loaded.query, self.state.query)

            # Load from open file object
            with open(file_path, "r", encoding="utf-8") as f:
                loaded_from_stream = load(f)
            self.assertEqual(loaded_from_stream.execution_id, "exec_123")

            # Save to StringIO stream
            stream = io.StringIO()
            save(self.state, stream)
            stream.seek(0)
            loaded_from_sio = load(stream)
            self.assertEqual(loaded_from_sio.plan_id, "plan_abc")

    def test_save_returns_json_string_when_no_target(self) -> None:
        """Verify save() returns JSON string when target is None."""
        res = save(self.state)
        self.assertIsInstance(res, str)
        parsed = json.loads(res)
        self.assertEqual(parsed["execution_id"], "exec_123")

    def test_resume_guarantees_completed_tasks_never_rerun(self) -> None:
        """Completed tasks must NEVER execute again during resume."""
        executed_task_ids: List[str] = []

        def tracking_handler(task: Task) -> str:
            executed_task_ids.append(task.id)
            return f"Executed {task.action}"

        custom_handlers = {
            "open_app": tracking_handler,
            "web_search": tracking_handler,
            "calculate": tracking_handler,
        }
        executor = Executor(handlers=custom_handlers, auto_register_in_container=False)

        # In self.state, t1 is COMPLETED, t2 was FAILED, t3 was SKIPPED
        result = resume(self.state, executor=executor)

        # t1 must NOT be in executed_task_ids!
        self.assertNotIn("t1", executed_task_ids, "Completed task t1 must NEVER run again.")
        # t2 and t3 should run
        self.assertIn("t2", executed_task_ids)
        self.assertIn("t3", executed_task_ids)

        # Result should reflect completed t1, t2, and t3
        self.assertTrue(result.success)
        completed_ids = {t.id for t in result.completed_tasks}
        self.assertEqual(completed_ids, {"t1", "t2", "t3"})
        self.assertIn("t1", result.execution_order)
        self.assertIn("t2", result.execution_order)
        self.assertIn("t3", result.execution_order)

    def test_resume_when_all_tasks_already_completed(self) -> None:
        """When all tasks are already completed, resume returns success immediately."""
        all_done_state = PersistedExecutionState(
            execution_id="done_123",
            plan_id="done_plan",
            query="all done",
            memory=ExecutionMemory(
                records=[
                    TaskExecutionRecord(
                        task_id="t1",
                        action="open_app",
                        target="notepad",
                        status=TaskStatus.COMPLETED,
                    )
                ]
            ),
            completed_tasks=[self.t1],
            dag_state={"completed_ids": ["t1"]},
        )
        res = resume(all_done_state)
        self.assertTrue(res.success)
        self.assertEqual(len(res.completed_tasks), 1)
        self.assertEqual(res.completed_tasks[0].id, "t1")


if __name__ == "__main__":
    unittest.main()
