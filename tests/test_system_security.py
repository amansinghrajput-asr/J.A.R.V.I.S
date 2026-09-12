"""Comprehensive unit tests for Sprint 22.1: System Skill Foundation & Security.

Verifies:
1. SAFE operation is allowed.
2. CONFIRMATION_REQUIRED operation is detected.
3. RESTRICTED operation is rejected.
4. Arbitrary shell execution is rejected.
5. Arbitrary PowerShell execution is rejected.
6. Critical process termination is rejected.
7. Protected Windows directory operations are rejected.
8. Path traversal is rejected.
9. Valid safe paths are accepted.
10. Confirmation approval allows the exact requested operation.
11. Confirmation rejection blocks execution.
12. Confirmation expiry blocks execution.
13. Approval for one operation cannot authorize a different operation.
14. Concurrent policy checks are safe.
15. Architecture isolation / no global mutable state.
16. BaseSystemSkill execution, telemetry, and event publishing.
"""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import tempfile
import time
import unittest

from app.ai.planner.events import (
    PlannerEvent,
    PlannerEventBus,
    SystemSkillCompleted,
    SystemSkillConfirmationRequired,
    SystemSkillFailed,
    SystemSkillPolicyRejected,
    SystemSkillStarted,
)
from app.ai.planner.swarm.hitl import ApprovalDecision, InterventionGateway, RiskLevel
from app.core.container import ServiceContainer
from app.skills.base import SkillExecutionError
from app.skills.system.base_system_skill import BaseSystemSkill, SystemSkillResult
from app.skills.system.security import (
    ConfirmationRejectedError,
    ConfirmationRequiredError,
    ConfirmationTimeoutError,
    SecurityPolicyViolationError,
    SystemConfirmationManager,
    SystemSafetyTier,
    SystemSecurityPolicy,
    is_critical_process,
    is_shell_command,
    validate_path,
)


class DummyConcreteSystemSkill(BaseSystemSkill):
    """Test concrete implementation of BaseSystemSkill."""

    name = "dummy_system"
    description = "Dummy system skill for testing base foundation."

    def can_handle(self, command: str) -> bool:
        op, _, _, _ = self.parse_command(command)
        return op in ("get_cpu_info", "delete_file", "raw_shell_test", "get_failing_data")

    def _execute_operation(self, operation: str, target: str | None, parameters: dict) -> any:
        if operation == "get_cpu_info":
            return {"usage": 15.5, "cores": 8}
        elif operation == "delete_file":
            return {"deleted": target}
        elif operation == "get_failing_data":
            raise RuntimeError("Operation encountered hardware fault")
        return {"status": "ok"}


class TestSystemSecurityPolicy(unittest.TestCase):
    """Unit tests for SystemSecurityPolicy and path/process guardrails."""

    def setUp(self) -> None:
        self.bus = PlannerEventBus()
        self.policy = SystemSecurityPolicy(event_bus=self.bus)

    def test_01_safe_operation_allowed(self) -> None:
        """1. SAFE operation is allowed without confirmation."""
        tier, reason = self.policy.validate_operation("get_cpu_info")
        self.assertEqual(tier, SystemSafetyTier.SAFE)
        self.assertIsNone(reason)

        tier, _ = self.policy.validate_operation("list_directory", target="C:\\Users\\test")
        self.assertEqual(tier, SystemSafetyTier.SAFE)

        tier, _ = self.policy.validate_operation("get_memory_info")
        self.assertEqual(tier, SystemSafetyTier.SAFE)

    def test_02_confirmation_required_operation_detected(self) -> None:
        """2. CONFIRMATION_REQUIRED operation is detected."""
        tier, reason = self.policy.validate_operation("delete_path", target="C:\\tmp\\foo.txt")
        self.assertEqual(tier, SystemSafetyTier.CONFIRMATION_REQUIRED)
        self.assertIn("requires user confirmation", reason)

        tier, reason = self.policy.validate_operation("terminate_process", target="notepad.exe")
        self.assertEqual(tier, SystemSafetyTier.CONFIRMATION_REQUIRED)

    def test_03_restricted_operation_rejected(self) -> None:
        """3. RESTRICTED operation is classified as RESTRICTED with policy explanation."""
        tier, reason = self.policy.validate_operation("shell", target="whoami")
        self.assertEqual(tier, SystemSafetyTier.RESTRICTED)
        self.assertTrue(reason and len(reason) > 0)

        tier, reason = self.policy.validate_operation("run_command", target="del /f /s /q c:")
        self.assertEqual(tier, SystemSafetyTier.RESTRICTED)
        self.assertTrue(reason and len(reason) > 0)

    def test_04_arbitrary_shell_execution_rejected(self) -> None:
        """4. Arbitrary shell execution is rejected as RESTRICTED."""
        shell_targets = [
            "cmd.exe /c dir",
            "cmd /k echo hello",
            "bash -c 'ls -la'",
            "sh test.sh",
            "cscript //nologo test.vbs",
        ]
        for cmd in shell_targets:
            with self.subTest(cmd=cmd):
                self.assertTrue(is_shell_command(cmd))
                tier, reason = self.policy.validate_operation("open_app", target=cmd)
                self.assertEqual(tier, SystemSafetyTier.RESTRICTED)
                self.assertTrue(reason and len(reason) > 0)

    def test_05_arbitrary_powershell_execution_rejected(self) -> None:
        """5. Arbitrary PowerShell execution is rejected as RESTRICTED."""
        ps_targets = [
            "powershell -encodedcommand dGVzdA==",
            "powershell.exe -ExecutionPolicy Bypass -File evil.ps1",
            "pwsh -Command Get-Process",
            "dir; Invoke-Expression evil",
            "calc.exe && notepad.exe",
            "test | evil",
        ]
        for cmd in ps_targets:
            with self.subTest(cmd=cmd):
                self.assertTrue(is_shell_command(cmd))
                tier, reason = self.policy.validate_operation("open_app", target=cmd)
                self.assertEqual(tier, SystemSafetyTier.RESTRICTED)
                self.assertTrue(reason and len(reason) > 0)

    def test_06_critical_process_termination_rejected(self) -> None:
        """6. Critical process termination is rejected as RESTRICTED."""
        critical_procs = [
            "explorer.exe",
            "explorer",
            "csrss.exe",
            "smss.exe",
            "wininit.exe",
            "services.exe",
            "lsass.exe",
            "dwm.exe",
            "svchost.exe",
            0,
            4,
            os.getpid(),
        ]
        for proc in critical_procs:
            with self.subTest(proc=proc):
                self.assertTrue(is_critical_process(proc))
                tier_close, reason_close = self.policy.validate_operation("close_app", target=str(proc))
                self.assertEqual(tier_close, SystemSafetyTier.RESTRICTED)
                self.assertTrue(reason_close and len(reason_close) > 0)

                tier_term, reason_term = self.policy.validate_operation("terminate_process", target=str(proc))
                self.assertEqual(tier_term, SystemSafetyTier.RESTRICTED)
                self.assertTrue(reason_term and len(reason_term) > 0)

    def test_07_protected_windows_directory_operations_rejected(self) -> None:
        """7. Protected Windows directory operations are rejected."""
        protected_paths = [
            r"C:\Windows\System32\cmd.exe",
            r"C:\Windows\explorer.exe",
            r"C:\ProgramData\Microsoft\Crypto",
            r"c:\windows\winsxs\manifest",
        ]
        for p in protected_paths:
            with self.subTest(path=p):
                with self.assertRaises(SecurityPolicyViolationError):
                    validate_path(p)
                tier, reason = self.policy.validate_operation("delete_file", target=p)
                self.assertEqual(tier, SystemSafetyTier.RESTRICTED)
                self.assertTrue(reason and len(reason) > 0)

    def test_08_path_traversal_rejected(self) -> None:
        """8. Path traversal is rejected when allowed_roots is specified."""
        with tempfile.TemporaryDirectory() as safe_root:
            allowed_root = Path(safe_root).resolve()
            traversal_attempt = str(allowed_root / "subdir" / ".." / ".." / "outside.txt")

            with self.assertRaises(SecurityPolicyViolationError):
                validate_path(traversal_attempt, allowed_roots=[allowed_root])

            # Also verify via policy configured with allowed_roots
            confined_policy = SystemSecurityPolicy(allowed_roots=[allowed_root])
            tier, reason = confined_policy.validate_operation("read_file", target=traversal_attempt)
            self.assertEqual(tier, SystemSafetyTier.RESTRICTED)
            self.assertTrue(reason and len(reason) > 0)

    def test_09_valid_safe_paths_accepted(self) -> None:
        """9. Valid safe paths are accepted."""
        with tempfile.TemporaryDirectory() as safe_dir:
            file_path = Path(safe_dir) / "document.txt"
            file_path.write_text("hello world")

            validated = validate_path(str(file_path), allowed_roots=[safe_dir])
            self.assertEqual(validated, file_path.resolve())

            tier, _ = self.policy.validate_operation("read_file", target=str(file_path))
            self.assertEqual(tier, SystemSafetyTier.SAFE)


class TestSystemConfirmationManager(unittest.TestCase):
    """Unit tests for SystemConfirmationManager and parameter cryptographic binding."""

    def setUp(self) -> None:
        self.bus = PlannerEventBus()
        self.mgr = SystemConfirmationManager(event_bus=self.bus, default_timeout=5.0)

    def test_10_confirmation_approval_allows_exact_operation(self) -> None:
        """10. Confirmation approval allows the exact requested operation."""
        conf_id = self.mgr.request_confirmation(
            operation="delete_file",
            target="report.pdf",
            parameters={"permanent": True},
        )
        self.assertFalse(self.mgr.is_approved(conf_id))

        # Approve
        resolved = self.mgr.resolve_confirmation(conf_id, approved=True, decided_by="test_user")
        self.assertTrue(resolved)
        self.assertTrue(self.mgr.is_approved(conf_id))

        # Verification and consumption succeeds
        consumed = self.mgr.verify_and_consume(
            confirmation_id=conf_id,
            operation="delete_file",
            target="report.pdf",
            parameters={"permanent": True},
        )
        self.assertTrue(consumed)

        # Re-consuming same token fails
        with self.assertRaises(ConfirmationRejectedError):
            self.mgr.verify_and_consume(
                confirmation_id=conf_id,
                operation="delete_file",
                target="report.pdf",
                parameters={"permanent": True},
            )

    def test_11_confirmation_rejection_blocks_execution(self) -> None:
        """11. Confirmation rejection blocks execution."""
        conf_id = self.mgr.request_confirmation(
            operation="delete_file",
            parameters={"permanent": True},
        )
        self.mgr.resolve_confirmation(conf_id, approved=False, reason="User aborted.")

        with self.assertRaises(ConfirmationRejectedError) as ctx:
            self.mgr.verify_and_consume(
                confirmation_id=conf_id,
                operation="delete_file",
                parameters={"permanent": True},
            )
        self.assertIn("User aborted", str(ctx.exception))

    def test_12_confirmation_expiry_blocks_execution(self) -> None:
        """12. Confirmation expiry blocks execution."""
        # Short timeout of 0.05 seconds
        conf_id = self.mgr.request_confirmation(
            operation="delete_file",
            timeout=0.05,
        )
        self.mgr.resolve_confirmation(conf_id, approved=True)

        # Wait for expiration
        time.sleep(0.08)

        with self.assertRaises(ConfirmationTimeoutError):
            self.mgr.verify_and_consume(
                confirmation_id=conf_id,
                operation="delete_file",
            )

    def test_13_approval_for_one_operation_cannot_authorize_different_operation(self) -> None:
        """13. Approval for one operation cannot authorize a different operation or altered arguments."""
        conf_id = self.mgr.request_confirmation(
            operation="delete_file",
            target="safe_file.txt",
            parameters={"force": False},
        )
        self.mgr.resolve_confirmation(conf_id, approved=True)

        # 1. Different operation name
        with self.assertRaises(ConfirmationRejectedError):
            self.mgr.verify_and_consume(
                confirmation_id=conf_id,
                operation="terminate_process",
                target="safe_file.txt",
                parameters={"force": False},
            )

        # 2. Different target
        with self.assertRaises(ConfirmationRejectedError):
            self.mgr.verify_and_consume(
                confirmation_id=conf_id,
                operation="delete_file",
                target="critical_file.sys",
                parameters={"force": False},
            )

        # 3. Altered parameters (e.g. force changed to True)
        with self.assertRaises(ConfirmationRejectedError):
            self.mgr.verify_and_consume(
                confirmation_id=conf_id,
                operation="delete_file",
                target="safe_file.txt",
                parameters={"force": True},
            )

    def test_14_concurrent_policy_and_confirmation_checks(self) -> None:
        """14. Concurrent policy checks and confirmation actions are thread-safe."""
        policy = SystemSecurityPolicy()
        mgr = SystemConfirmationManager()

        def worker(idx: int) -> bool:
            # Run policy check
            tier, _ = policy.validate_operation("get_cpu_info")
            if tier != SystemSafetyTier.SAFE:
                return False

            # Request, resolve, verify confirmation
            cid = mgr.request_confirmation(
                operation="delete_file",
                target=f"file_{idx}.tmp",
                parameters={"idx": idx},
            )
            mgr.resolve_confirmation(cid, approved=True)
            return mgr.verify_and_consume(
                confirmation_id=cid,
                operation="delete_file",
                target=f"file_{idx}.tmp",
                parameters={"idx": idx},
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(worker, range(50)))

        self.assertTrue(all(results))

    def test_15_architecture_isolation_and_container_wiring(self) -> None:
        """15. Architecture isolation / no global mutable state and container registration."""
        cont1 = ServiceContainer()
        cont2 = ServiceContainer()

        pol1 = SystemSecurityPolicy()
        pol2 = SystemSecurityPolicy()

        conf1 = SystemConfirmationManager()
        conf2 = SystemConfirmationManager()

        from app.skills.system import register_system_foundation
        register_system_foundation(cont1, security_policy=pol1, confirmation_manager=conf1)
        register_system_foundation(cont2, security_policy=pol2, confirmation_manager=conf2)

        self.assertIs(cont1.resolve("system_security_policy"), pol1)
        self.assertIs(cont2.resolve("system_security_policy"), pol2)
        self.assertIsNot(cont1.resolve("system_security_policy"), cont2.resolve("system_security_policy"))

    def test_16_hitl_gateway_integration(self) -> None:
        """16. Integration with InterventionGateway auto-approves or delegates correctly."""
        gateway = InterventionGateway()
        # Register a mock handler approving only target 'approved_target.txt'
        def handler(req: dict) -> ApprovalDecision:
            approved = req.get("target") == "approved_target.txt"
            return ApprovalDecision(
                request_id=req["request_id"],
                approved=approved,
                decided_by="hitl_test_handler",
            )
        gateway.register_approval_handler(handler)

        mgr = SystemConfirmationManager(hitl_gateway=gateway)

        # 1. Target that gets approved
        cid1 = mgr.request_confirmation("delete_file", target="approved_target.txt")
        self.assertTrue(mgr.is_approved(cid1))
        self.assertTrue(mgr.verify_and_consume(cid1, "delete_file", target="approved_target.txt"))

        # 2. Target that gets denied
        cid2 = mgr.request_confirmation("delete_file", target="denied_target.txt")
        self.assertFalse(mgr.is_approved(cid2))
        with self.assertRaises(ConfirmationRejectedError):
            mgr.verify_and_consume(cid2, "delete_file", target="denied_target.txt")


class TestBaseSystemSkillExecution(unittest.TestCase):
    """Unit tests for BaseSystemSkill execution, telemetry, and event broadcasting."""

    def setUp(self) -> None:
        self.bus = PlannerEventBus()
        self.events: list[PlannerEvent] = []
        self.bus.subscribe(PlannerEvent, lambda ev: self.events.append(ev))

        self.policy = SystemSecurityPolicy(event_bus=self.bus)
        self.confirmation_mgr = SystemConfirmationManager(event_bus=self.bus)

        self.skill = DummyConcreteSystemSkill(
            security_policy=self.policy,
            confirmation_manager=self.confirmation_mgr,
            event_bus=self.bus,
        )

    def test_execute_safe_operation_success(self) -> None:
        """Executing a SAFE operation succeeds and emits start/completion events."""
        result = self.skill.execute("get_cpu_info")
        self.assertIsInstance(result, SystemSkillResult)
        self.assertTrue(result.success)
        self.assertEqual(result.operation, "get_cpu_info")
        self.assertEqual(result.data["usage"], 15.5)
        self.assertGreaterEqual(result.duration_ms, 0.0)

        # Check emitted events
        event_types = [type(e) for e in self.events]
        self.assertIn(SystemSkillStarted, event_types)
        self.assertIn(SystemSkillCompleted, event_types)

    def test_execute_confirmation_required_flow(self) -> None:
        """Executing a CONFIRMATION_REQUIRED operation requires token and executes upon approval."""
        cmd = {"operation": "delete_file", "target": "old_backup.zip"}

        # 1. First execution raises ConfirmationRequiredError and issues token
        with self.assertRaises(ConfirmationRequiredError) as ctx:
            self.skill.execute(cmd)

        token = ctx.exception.confirmation_id
        self.assertTrue(token.startswith("sys_conf_"))

        # Emitted ConfirmationRequired event
        event_types = [type(e) for e in self.events]
        self.assertIn(SystemSkillConfirmationRequired, event_types)

        # 2. Operator approves token
        self.confirmation_mgr.resolve_confirmation(token, approved=True, decided_by="admin")

        # 3. Second execution with confirmation_id succeeds
        cmd_with_token = {
            "operation": "delete_file",
            "target": "old_backup.zip",
            "confirmation_id": token,
        }
        res = self.skill.execute(cmd_with_token)
        self.assertTrue(res.success)
        self.assertEqual(res.data, {"deleted": "old_backup.zip"})

    def test_execute_restricted_operation_raises_and_emits_rejected_event(self) -> None:
        """Executing a RESTRICTED operation raises SecurityPolicyViolationError and emits event."""
        cmd = {"operation": "shell", "target": "format C:"}

        with self.assertRaises(SecurityPolicyViolationError):
            self.skill.execute(cmd)

        event_types = [type(e) for e in self.events]
        self.assertIn(SystemSkillPolicyRejected, event_types)

    def test_execute_failing_operation_emits_failed_event(self) -> None:
        """Executing an operation that raises an internal error emits SystemSkillFailed."""
        with self.assertRaises(SkillExecutionError):
            self.skill.execute("get_failing_data")

        event_types = [type(e) for e in self.events]
        self.assertIn(SystemSkillStarted, event_types)
        self.assertIn(SystemSkillFailed, event_types)


if __name__ == "__main__":
    unittest.main()
