"""Comprehensive unit tests for Sprint 22.3: File & Folder Skills.

Covers:
1. Skill registration.
2. create_folder success.
3. create_folder existing target behavior.
4. create_file success.
5. create_file does not silently overwrite.
6. read_file success.
7. read_file missing file.
8. read_file size limit.
9. list_directory success.
10. list_directory bounded output.
11. rename success.
12. rename collision handling.
13. move requires confirmation.
14. move success with valid confirmation.
15. copy success.
16. copy overwrite protection.
17. delete requires confirmation.
18. delete succeeds with exact valid confirmation in temp fixture.
19. delete rejects missing/invalid confirmation.
20. delete rejects protected paths.
21. traversal is rejected.
22. Windows reserved names are rejected.
23. filesystem root deletion is rejected.
24. symlink/junction safety behavior.
25. open_in_explorer validation.
26. shell injection attempts are rejected.
27. protected Windows directory operations are rejected.
28. EventBus lifecycle behavior.
29. concurrent safe read/list operations.
30. architecture isolation / no global mutable state.
"""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from app.ai.planner.events import (
    PlannerEvent,
    PlannerEventBus,
    SystemSkillCompleted,
    SystemSkillConfirmationRequired,
    SystemSkillFailed,
    SystemSkillPolicyRejected,
    SystemSkillStarted,
)
from app.core.container import ServiceContainer
from app.skills.base import SkillExecutionError
from app.skills.manager import SkillManager
from app.skills.system.file_skills import (
    DEFAULT_MAX_DIR_ENTRIES,
    DEFAULT_MAX_READ_BYTES,
    FileSkills,
)
from app.skills.system.security import (
    ConfirmationRejectedError,
    ConfirmationRequiredError,
    SecurityPolicyViolationError,
    SystemConfirmationManager,
    SystemSafetyTier,
    SystemSecurityPolicy,
)


class TestFileSkills(unittest.TestCase):
    """Unit test suite for FileSkills."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name).resolve()

        self.bus = PlannerEventBus()
        self.events: list[PlannerEvent] = []
        self.bus.subscribe(PlannerEvent, lambda ev: self.events.append(ev))

        # Restrict policy to our temporary root directory
        self.policy = SystemSecurityPolicy(
            allowed_roots=[self.root_path],
            event_bus=self.bus,
        )
        self.confirmation_mgr = SystemConfirmationManager(event_bus=self.bus)

        self.skill = FileSkills(
            security_policy=self.policy,
            confirmation_manager=self.confirmation_mgr,
            event_bus=self.bus,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_01_skill_registration(self) -> None:
        """1. File skill registers properly into SkillManager and adheres to BaseSkill contract."""
        container = ServiceContainer()
        manager = SkillManager(container_instance=container, auto_register_in_container=False)
        manager.register(self.skill)

        registered = manager.get("file")
        self.assertIs(registered, self.skill)
        self.assertEqual(registered.name, "file")
        self.assertEqual(registered.priority, 60)
        self.assertTrue(self.skill.can_handle("create folder test"))
        self.assertTrue(self.skill.can_handle("read file test.txt"))
        self.assertTrue(self.skill.can_handle("list directory test"))
        self.assertTrue(self.skill.can_handle("delete file test.txt"))
        self.assertTrue(self.skill.can_handle("open test in explorer"))

    def test_02_create_folder_success(self) -> None:
        """2. create_folder successfully creates directory tree."""
        target = self.root_path / "sub" / "folder"
        res = self.skill.execute({"operation": "create_folder", "target": str(target)})
        self.assertTrue(res.success)
        self.assertTrue(res.data["created"])
        self.assertTrue(target.is_dir())

    def test_03_create_folder_existing_target(self) -> None:
        """3. create_folder handles existing directory without error."""
        target = self.root_path / "existing"
        target.mkdir(parents=True, exist_ok=True)

        res = self.skill.execute({"operation": "create_folder", "target": str(target)})
        self.assertTrue(res.success)
        self.assertFalse(res.data["created"])
        self.assertTrue(res.data["exists"])

    def test_04_create_file_success(self) -> None:
        """4. create_file successfully writes new file."""
        target = self.root_path / "doc.txt"
        res = self.skill.execute({
            "operation": "create_file",
            "target": str(target),
            "parameters": {"content": "Hello J.A.R.V.I.S."},
        })
        self.assertTrue(res.success)
        self.assertTrue(res.data["created"])
        self.assertEqual(res.data["bytes_written"], len("Hello J.A.R.V.I.S.".encode("utf-8")))
        self.assertEqual(target.read_text(encoding="utf-8"), "Hello J.A.R.V.I.S.")

    def test_05_create_file_does_not_silently_overwrite(self) -> None:
        """5. create_file rejects overwriting existing file without confirmation."""
        target = self.root_path / "existing.txt"
        target.write_text("initial content", encoding="utf-8")

        # Attempt to create without overwrite flag
        with self.assertRaises(SkillExecutionError) as ctx:
            self.skill.execute({
                "operation": "create_file",
                "target": str(target),
                "parameters": {"content": "new content"},
            })
        self.assertIn("already exists", str(ctx.exception))

        # Attempt to overwrite requires confirmation
        with self.assertRaises(ConfirmationRequiredError):
            self.skill.execute({
                "operation": "create_file",
                "target": str(target),
                "parameters": {"content": "new content", "overwrite": True},
            })

    def test_06_read_file_success(self) -> None:
        """6. read_file successfully reads content."""
        target = self.root_path / "read_sample.txt"
        target.write_text("Sample file data for J.A.R.V.I.S.", encoding="utf-8")

        res = self.skill.execute({"operation": "read_file", "target": str(target)})
        self.assertTrue(res.success)
        self.assertEqual(res.data["content"], "Sample file data for J.A.R.V.I.S.")
        self.assertEqual(res.data["size_bytes"], len("Sample file data for J.A.R.V.I.S.".encode("utf-8")))

    def test_07_read_file_missing_file(self) -> None:
        """7. read_file raises SkillExecutionError on nonexistent file."""
        target = self.root_path / "nonexistent.txt"
        with self.assertRaises(SkillExecutionError) as ctx:
            self.skill.execute({"operation": "read_file", "target": str(target)})
        self.assertIn("does not exist", str(ctx.exception))

    def test_08_read_file_size_limit(self) -> None:
        """8. read_file rejects files exceeding maximum read limit."""
        target = self.root_path / "large.bin"
        # Write 2048 bytes
        target.write_bytes(b"x" * 2048)

        # Set max_bytes to 1024
        with self.assertRaises(SkillExecutionError) as ctx:
            self.skill.execute({
                "operation": "read_file",
                "target": str(target),
                "parameters": {"max_bytes": 1024},
            })
        self.assertIn("exceeds maximum permitted read size", str(ctx.exception))

    def test_09_list_directory_success(self) -> None:
        """9. list_directory returns structured file and directory entries."""
        (self.root_path / "subfolder").mkdir()
        (self.root_path / "file1.txt").write_text("a")
        (self.root_path / "file2.log").write_text("bb")

        res = self.skill.execute({"operation": "list_directory", "target": str(self.root_path)})
        self.assertTrue(res.success)
        self.assertEqual(res.data["total_returned"], 3)
        names = {e["name"] for e in res.data["entries"]}
        self.assertEqual(names, {"subfolder", "file1.txt", "file2.log"})

    def test_10_list_directory_bounded_output(self) -> None:
        """10. list_directory bounds returned items by limit."""
        for i in range(15):
            (self.root_path / f"item_{i}.txt").write_text(str(i))

        res = self.skill.execute({
            "operation": "list_directory",
            "target": str(self.root_path),
            "parameters": {"limit": 5},
        })
        self.assertTrue(res.success)
        self.assertEqual(res.data["total_returned"], 5)
        self.assertTrue(res.data["has_more"])

    def test_11_rename_success(self) -> None:
        """11. rename_path successfully renames file."""
        src = self.root_path / "old.txt"
        dest = self.root_path / "new.txt"
        src.write_text("renamed content")

        res = self.skill.execute({
            "operation": "rename_path",
            "target": str(src),
            "parameters": {"destination": str(dest)},
        })
        self.assertTrue(res.success)
        self.assertTrue(dest.is_file())
        self.assertFalse(src.exists())

    def test_12_rename_collision_handling(self) -> None:
        """12. rename_path rejects overwriting destination when overwrite=False."""
        src = self.root_path / "src.txt"
        dest = self.root_path / "dest.txt"
        src.write_text("source")
        dest.write_text("dest")

        with self.assertRaises(SkillExecutionError) as ctx:
            self.skill.execute({
                "operation": "rename_path",
                "target": str(src),
                "parameters": {"destination": str(dest)},
            })
        self.assertIn("already exists", str(ctx.exception))

    def test_13_move_requires_confirmation(self) -> None:
        """13. move_path requires confirmation by security policy."""
        src = self.root_path / "moving.txt"
        dest = self.root_path / "moved.txt"
        src.write_text("content")

        with self.assertRaises(ConfirmationRequiredError):
            self.skill.execute({
                "operation": "move_path",
                "target": str(src),
                "parameters": {"destination": str(dest)},
            })

    def test_14_move_success_with_valid_confirmation(self) -> None:
        """14. move_path succeeds when provided valid confirmation token."""
        src = self.root_path / "moving_confirmed.txt"
        dest = self.root_path / "moved_confirmed.txt"
        src.write_text("move me")

        token = self.confirmation_mgr.request_confirmation(
            "move_path", target=str(src), parameters={"destination": str(dest)}
        )
        self.confirmation_mgr.resolve_confirmation(token, approved=True)

        res = self.skill.execute({
            "operation": "move_path",
            "target": str(src),
            "parameters": {"destination": str(dest)},
            "confirmation_id": token,
        })
        self.assertTrue(res.success)
        self.assertTrue(dest.is_file())
        self.assertFalse(src.exists())

    def test_15_copy_success(self) -> None:
        """15. copy_path copies file successfully."""
        src = self.root_path / "original.txt"
        dest = self.root_path / "copy.txt"
        src.write_text("copy content")

        res = self.skill.execute({
            "operation": "copy_path",
            "target": str(src),
            "parameters": {"destination": str(dest)},
        })
        self.assertTrue(res.success)
        self.assertTrue(src.exists())
        self.assertTrue(dest.exists())
        self.assertEqual(dest.read_text(), "copy content")

    def test_16_copy_overwrite_protection(self) -> None:
        """16. copy_path rejects overwriting existing destination without confirmation."""
        src = self.root_path / "copy_src.txt"
        dest = self.root_path / "copy_dest.txt"
        src.write_text("src")
        dest.write_text("dest")

        with self.assertRaises(SkillExecutionError) as ctx:
            self.skill.execute({
                "operation": "copy_path",
                "target": str(src),
                "parameters": {"destination": str(dest)},
            })
        self.assertIn("already exists", str(ctx.exception))

    def test_17_delete_requires_confirmation(self) -> None:
        """17. delete_path requires explicit confirmation token."""
        target = self.root_path / "to_delete.txt"
        target.write_text("delete me")

        with self.assertRaises(ConfirmationRequiredError) as ctx:
            self.skill.execute({"operation": "delete_path", "target": str(target)})

        token = ctx.exception.confirmation_id
        self.assertTrue(token.startswith("sys_conf_"))
        self.assertTrue(target.exists())

    def test_18_delete_succeeds_with_exact_valid_confirmation(self) -> None:
        """18. delete_path succeeds when provided exact authorized confirmation token."""
        target = self.root_path / "delete_valid.txt"
        target.write_text("goodbye")

        token = self.confirmation_mgr.request_confirmation("delete_path", target=str(target))
        self.confirmation_mgr.resolve_confirmation(token, approved=True)

        res = self.skill.execute({
            "operation": "delete_path",
            "target": str(target),
            "confirmation_id": token,
        })
        self.assertTrue(res.success)
        self.assertFalse(target.exists())

    def test_19_delete_rejects_missing_or_invalid_confirmation(self) -> None:
        """19. delete_path rejects bogus confirmation token."""
        target = self.root_path / "target.txt"
        target.write_text("safe")

        with self.assertRaises(ConfirmationRejectedError):
            self.skill.execute({
                "operation": "delete_path",
                "target": str(target),
                "confirmation_id": "invalid_fake_token",
            })
        self.assertTrue(target.exists())

    def test_20_delete_rejects_protected_paths(self) -> None:
        """20. delete_path rejects protected system directories."""
        protected = r"C:\Windows\System32\drivers"
        tier, reason = self.policy.validate_operation("delete_path", target=protected)
        self.assertEqual(tier, SystemSafetyTier.RESTRICTED)
        self.assertTrue(reason and len(reason) > 0)
        with self.assertRaises(SecurityPolicyViolationError):
            self.skill.execute({"operation": "delete_path", "target": protected})

    def test_21_traversal_rejected(self) -> None:
        """21. Path traversal escaping allowed root is rejected."""
        traversal_target = str(self.root_path / ".." / "outside.txt")
        tier, reason = self.policy.validate_operation("read_file", target=traversal_target)
        self.assertEqual(tier, SystemSafetyTier.RESTRICTED)
        self.assertTrue(reason and len(reason) > 0)
        with self.assertRaises(SecurityPolicyViolationError):
            self.skill.execute({"operation": "read_file", "target": traversal_target})

    def test_22_windows_reserved_names_rejected(self) -> None:
        """22. Windows reserved device names (CON, NUL, AUX, etc.) are rejected."""
        reserved_paths = [
            str(self.root_path / "CON"),
            str(self.root_path / "NUL.txt"),
            str(self.root_path / "COM1"),
            str(self.root_path / "AUX"),
        ]
        for rp in reserved_paths:
            with self.subTest(path=rp):
                tier, reason = self.policy.validate_operation("create_file", target=rp)
                self.assertEqual(tier, SystemSafetyTier.RESTRICTED)
                self.assertTrue(reason and len(reason) > 0)

    def test_23_filesystem_root_deletion_rejected(self) -> None:
        """23. Deletion of filesystem roots (e.g. C:\\) is strictly prohibited."""
        root_targets = ["C:\\", "c:/", "/", "\\"]
        for rt in root_targets:
            with self.subTest(root=rt):
                tier, reason = self.policy.validate_operation("delete_path", target=rt)
                self.assertEqual(tier, SystemSafetyTier.RESTRICTED)
                self.assertTrue(reason and len(reason) > 0)

    def test_24_symlink_safety_behavior(self) -> None:
        """24. Deleting a symlink unlinks the link itself, never deleting into target content."""
        target_file = self.root_path / "target_real.txt"
        target_file.write_text("real content to preserve")

        link_file = self.root_path / "link_to_real.txt"
        try:
            link_file.symlink_to(target_file)
        except (OSError, NotImplementedError):
            self.skipTest("Symlink creation not permitted in this OS environment.")

        # Confirm delete link
        token = self.confirmation_mgr.request_confirmation("delete_path", target=str(link_file))
        self.confirmation_mgr.resolve_confirmation(token, approved=True)

        res = self.skill.execute({
            "operation": "delete_path",
            "target": str(link_file),
            "confirmation_id": token,
        })
        self.assertTrue(res.success)
        self.assertEqual(res.data["type"], "symlink")
        self.assertFalse(link_file.exists())
        # Target real file must still exist and be untouched!
        self.assertTrue(target_file.exists())
        self.assertEqual(target_file.read_text(), "real content to preserve")

    @patch("subprocess.Popen")
    def test_25_open_in_explorer_validation(self, mock_popen: MagicMock) -> None:
        """25. open_in_explorer validates target and invokes explorer without shell."""
        target = self.root_path / "browse_me.txt"
        target.write_text("open in explorer")

        res = self.skill.execute({"operation": "open_in_explorer", "target": str(target)})
        self.assertTrue(res.success)
        self.assertTrue(res.data["opened"])

        if sys.platform == "win32":
            mock_popen.assert_called_once()
            args, kwargs = mock_popen.call_args
            self.assertIn("explorer.exe", args[0][0])
            self.assertFalse(kwargs.get("shell", True))

    def test_26_shell_injection_attempts_rejected(self) -> None:
        """26. Shell injection tokens in paths are rejected by security policy."""
        injection_paths = [
            str(self.root_path / "test.txt; rm -rf /"),
            str(self.root_path / "file.txt && whoami"),
            str(self.root_path / "test | dir"),
        ]
        for ip in injection_paths:
            with self.subTest(path=ip):
                tier, reason = self.policy.validate_operation("read_file", target=ip)
                self.assertEqual(tier, SystemSafetyTier.RESTRICTED)
                self.assertTrue(reason and len(reason) > 0)

    def test_27_protected_windows_directory_operations_rejected(self) -> None:
        """27. Operations targeting protected Windows directories are rejected."""
        protected_targets = [
            r"C:\Windows\System32\drivers\etc\hosts",
            r"C:\ProgramData\Microsoft\test.txt",
            r"C:\$Recycle.Bin\test",
        ]
        for pt in protected_targets:
            with self.subTest(path=pt):
                tier, reason = self.policy.validate_operation("read_file", target=pt)
                self.assertEqual(tier, SystemSafetyTier.RESTRICTED)
                self.assertTrue(reason and len(reason) > 0)

    def test_28_event_bus_lifecycle_behavior(self) -> None:
        """28. Filesystem operations publish lifecycle events on PlannerEventBus."""
        target = self.root_path / "event_file.txt"
        self.skill.execute({
            "operation": "create_file",
            "target": str(target),
            "parameters": {"content": "event test"},
        })

        event_types = [type(e) for e in self.events]
        self.assertIn(SystemSkillStarted, event_types)
        self.assertIn(SystemSkillCompleted, event_types)

    def test_29_concurrent_safe_read_and_list_operations(self) -> None:
        """29. Concurrent read_file and list_directory operations are thread-safe."""
        shared_file = self.root_path / "shared.txt"
        shared_file.write_text("concurrency test data")

        def reader(idx: int) -> bool:
            res_read = self.skill.execute({"operation": "read_file", "target": str(shared_file)})
            res_list = self.skill.execute({"operation": "list_directory", "target": str(self.root_path)})
            return res_read.success and res_list.success

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(reader, range(30)))

        self.assertTrue(all(results))

    def test_30_architecture_isolation_and_di(self) -> None:
        """30. Multiple FileSkills instances maintain independent boundaries without global state."""
        skill1 = FileSkills(max_read_bytes=2048)
        skill2 = FileSkills(max_read_bytes=4096)

        self.assertEqual(skill1.max_read_bytes, 2048)
        self.assertEqual(skill2.max_read_bytes, 4096)
        self.assertIsNot(skill1.security_policy, skill2.security_policy)


if __name__ == "__main__":
    unittest.main()
