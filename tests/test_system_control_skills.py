"""Comprehensive unit and security tests for Phase 22.5 System Control Skills.

Covers:
- Volume inspection and bounds-clamped modification
- Display brightness inspection and graceful unsupported handling
- Workstation locking through direct Windows user32 API
- Cryptographically bound confirmation for sensitive operations (e.g. max volume)
- Safety tests preventing unauthorized, expired, or modified execution
- Lifecycle events, DI integration, and concurrent thread safety
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import MagicMock, call, patch

from app.ai.planner.events import (
    PlannerEventBus,
    SystemSkillCompleted,
    SystemSkillConfirmationRequired,
    SystemSkillStarted,
)
from app.core.container import ServiceContainer
from app.skills.base import SkillExecutionError
from app.skills.system.base_system_skill import SystemSkillResult
from app.skills.system.security import (
    ConfirmationRejectedError,
    ConfirmationRequiredError,
    ConfirmationTimeoutError,
    SecurityPolicyViolationError,
    SystemConfirmationManager,
    SystemSafetyTier,
    SystemSecurityPolicy,
)
from app.skills.system.system_control_skills import SystemControlSkills


class TestSystemControlSkills(unittest.TestCase):
    """Unit and security test suite for SystemControlSkills."""

    @classmethod
    def setUpClass(cls) -> None:
        """Enforce class-wide safety isolation against destructive OS calls."""
        super().setUpClass()
        cls._orig_lock = SystemControlSkills._call_lock_api

        # Safety stub replacing class-level lock method so no instance can execute OS lock commands
        SystemControlSkills._call_lock_api = MagicMock(return_value=True)  # type: ignore

    @classmethod
    def tearDownClass(cls) -> None:
        """Restore original OS call primitives."""
        SystemControlSkills._call_lock_api = cls._orig_lock
        super().tearDownClass()

    def setUp(self) -> None:
        """Create fresh isolated environment and mocks."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name).resolve()
        self.event_bus = PlannerEventBus()
        self.security_policy = SystemSecurityPolicy(
            allowed_roots=[self.temp_path],
            event_bus=self.event_bus,
        )
        self.confirmation_manager = SystemConfirmationManager(event_bus=self.event_bus)
        self.container = ServiceContainer()

        self.skill = SystemControlSkills(
            security_policy=self.security_policy,
            confirmation_manager=self.confirmation_manager,
            container=self.container,
            event_bus=self.event_bus,
        )

        # Mocks for OS primitives to ensure physical system state is never altered
        self.mock_lock_api = MagicMock(return_value=True)

        self.skill._call_lock_api = self.mock_lock_api

    def tearDown(self) -> None:
        """Clean up temporary directory and verify destructive OS hooks remained mocked."""
        # Safety invariant checks: destructive hooks must NEVER have been unmocked
        self.assertIs(
            self.skill._call_lock_api,
            self.mock_lock_api,
            "Safety violation: _call_lock_api was unmocked during test execution!",
        )
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    # ---------------------------------------------------------------------------
    # 1. Metadata and Registration
    # ---------------------------------------------------------------------------

    def test_01_skill_metadata_and_registration(self) -> None:
        """Verify skill metadata, priority, permissions, and tags."""
        self.assertEqual(self.skill.name, "system_control")
        self.assertEqual(self.skill.priority, 60)
        self.assertTrue(self.skill.enabled)
        self.assertIn("system:control", self.skill.permissions)
        self.assertIn("volume", self.skill.tags)
        self.assertIn("brightness", self.skill.tags)
        self.assertIn("lock", self.skill.tags)

    def test_02_can_handle_explicit_operations(self) -> None:
        """Verify can_handle accepts all structured operation names."""
        for op in (
            "get_volume",
            "set_volume",
            "get_brightness",
            "set_brightness",
            "lock_workstation",
        ):
            self.assertTrue(self.skill.can_handle({"operation": op}))
            self.assertTrue(self.skill.can_handle(op))

    def test_03_can_handle_natural_language_queries(self) -> None:
        """Verify can_handle and parse_command accurately route natural language queries."""
        queries = [
            ("What is the volume?", "get_volume", {}),
            ("Set volume to 50", "set_volume", {"level": 50}),
            ("What is the brightness?", "get_brightness", {}),
            ("Set brightness to 80", "set_brightness", {"level": 80}),
            ("Lock the workstation", "lock_workstation", {}),
        ]
        for query, expected_op, expected_params in queries:
            self.assertTrue(self.skill.can_handle(query), f"Failed to handle query: '{query}'")
            op, _, params, _ = self.skill.parse_command(query)
            self.assertEqual(op, expected_op, f"Op mismatch for '{query}'")
            for k, v in expected_params.items():
                self.assertEqual(params.get(k), v, f"Param {k} mismatch for '{query}'")

    def test_04_can_handle_rejects_unrelated(self) -> None:
        """Verify unrelated queries and commands are rejected."""
        unrelated = [
            "open notepad",
            "delete file secret.txt",
            "what is my cpu usage?",
            "write a poem",
            "",
            None,
        ]
        for item in unrelated:
            self.assertFalse(self.skill.can_handle(item), f"Should not handle: {item}")

    # ---------------------------------------------------------------------------
    # 2. Volume Controls
    # ---------------------------------------------------------------------------

    def test_05_get_volume_success(self) -> None:
        """Verify get_volume returns structured level and muted values."""
        res = self.skill.execute({"operation": "get_volume"})
        self.assertIsInstance(res, SystemSkillResult)
        self.assertTrue(res.success)
        data = res.data
        self.assertIn("level", data)
        self.assertIn("muted", data)
        self.assertIsInstance(data["level"], int)
        self.assertGreaterEqual(data["level"], 0)
        self.assertLessEqual(data["level"], 100)
        self.assertIsInstance(data["muted"], bool)

    def test_06_set_volume_success(self) -> None:
        """Verify set_volume returns structured previous and current level."""
        res = self.skill.execute({"operation": "set_volume", "parameters": {"level": 45}})
        self.assertTrue(res.success)
        data = res.data
        self.assertIn("previous_level", data)
        self.assertEqual(data["current_level"], 45)
        self.assertIn("muted", data)

    def test_07_set_volume_clamps_bounds(self) -> None:
        """Verify volume clamping to 0-100 range."""
        res_low = self.skill.set_volume(-15)
        self.assertEqual(res_low["current_level"], 0)

        res_high = self.skill.set_volume(150)
        self.assertEqual(res_high["current_level"], 100)

    def test_08_set_volume_non_numeric_rejected(self) -> None:
        """Verify non-numeric volume raises SkillExecutionError."""
        with self.assertRaises(SkillExecutionError):
            self.skill.set_volume("maximum")

    def test_09_set_volume_none_rejected(self) -> None:
        """Verify missing/None volume level raises SkillExecutionError."""
        with self.assertRaises(SkillExecutionError):
            self.skill.set_volume(None)

    def test_10_set_volume_level_100_requires_confirmation(self) -> None:
        """Verify policy requires confirmation for setting volume to maximum 100%."""
        tier, reason = self.security_policy.validate_operation("set_volume", parameters={"level": 100})
        self.assertEqual(tier, SystemSafetyTier.CONFIRMATION_REQUIRED)
        self.assertIn("maximum", reason.lower())

        with self.assertRaises(ConfirmationRequiredError):
            self.skill.execute({"operation": "set_volume", "parameters": {"level": 100}})

    # ---------------------------------------------------------------------------
    # 3. Brightness Controls
    # ---------------------------------------------------------------------------

    def test_11_get_brightness_supported(self) -> None:
        """Verify get_brightness structured response when supported."""
        mock_sbc = MagicMock()
        mock_sbc.get_brightness.return_value = [70]

        with patch.object(self.skill, "_get_sbc_module", return_value=mock_sbc):
            res = self.skill.execute({"operation": "get_brightness"})
            self.assertTrue(res.success)
            self.assertTrue(res.data["supported"])
            self.assertEqual(res.data["level"], 70)
            self.assertEqual(res.data["displays"], [70])

    def test_12_get_brightness_unsupported_graceful(self) -> None:
        """Verify graceful structured fallback when brightness is unsupported."""
        mock_sbc = MagicMock()
        mock_sbc.get_brightness.side_effect = Exception("No display found")

        with patch.object(self.skill, "_get_sbc_module", return_value=mock_sbc):
            res = self.skill.execute({"operation": "get_brightness"})
            self.assertTrue(res.success)
            self.assertFalse(res.data["supported"])
            self.assertIsNone(res.data["level"])
            self.assertIn("error", res.data)

    def test_13_set_brightness_supported(self) -> None:
        """Verify set_brightness sets level successfully."""
        mock_sbc = MagicMock()
        with patch.object(self.skill, "_get_sbc_module", return_value=mock_sbc):
            res = self.skill.execute({"operation": "set_brightness", "parameters": {"level": 60}})
            self.assertTrue(res.success)
            self.assertTrue(res.data["supported"])
            self.assertEqual(res.data["level"], 60)
            mock_sbc.set_brightness.assert_called_once_with(60)

    def test_14_set_brightness_clamped(self) -> None:
        """Verify set_brightness clamps values outside 0-100."""
        mock_sbc = MagicMock()
        with patch.object(self.skill, "_get_sbc_module", return_value=mock_sbc):
            res_low = self.skill.set_brightness(-20)
            self.assertEqual(res_low["level"], 0)
            mock_sbc.set_brightness.assert_called_with(0)

            res_high = self.skill.set_brightness(130)
            self.assertEqual(res_high["level"], 100)
            mock_sbc.set_brightness.assert_called_with(100)

    def test_15_set_brightness_non_numeric_rejected(self) -> None:
        """Verify non-numeric brightness level raises SkillExecutionError."""
        with self.assertRaises(SkillExecutionError):
            self.skill.set_brightness("dim")

    def test_16_set_brightness_unsupported_graceful(self) -> None:
        """Verify set_brightness graceful handling when hardware unsupported."""
        mock_sbc = MagicMock()
        mock_sbc.set_brightness.side_effect = Exception("DDC/CI not supported")

        with patch.object(self.skill, "_get_sbc_module", return_value=mock_sbc):
            res = self.skill.execute({"operation": "set_brightness", "parameters": {"level": 50}})
            self.assertTrue(res.success)
            self.assertFalse(res.data["supported"])
            self.assertIsNone(res.data["level"])
            self.assertIn("error", res.data)

    # ---------------------------------------------------------------------------
    # 4. Workstation Lock
    # ---------------------------------------------------------------------------

    def test_17_lock_workstation_calls_user32_api(self) -> None:
        """Verify lock_workstation triggers LockWorkStation API and returns structured result."""
        res = self.skill.execute({"operation": "lock_workstation"})
        self.assertTrue(res.success)
        self.assertTrue(res.data["locked"])
        self.assertEqual(res.data["method"], "LockWorkStation")
        self.mock_lock_api.assert_called_once()

    def test_18_lock_workstation_is_safe(self) -> None:
        """Verify lock_workstation is classified as SAFE."""
        tier, reason = self.security_policy.validate_operation("lock_workstation")
        self.assertEqual(tier, SystemSafetyTier.SAFE)

    # ---------------------------------------------------------------------------
    # 5. Strict Safety Boundaries & Confirmation Tampering
    # ---------------------------------------------------------------------------

    def test_26_invalid_confirmation_token_rejected(self) -> None:
        """Verify forged or nonexistent token raises ConfirmationRejectedError."""
        with self.assertRaises(ConfirmationRejectedError):
            self.skill.execute({
                "operation": "set_volume",
                "parameters": {"level": 100},
                "confirmation_id": "fake_token_12345",
            })

    def test_27_expired_confirmation_token_rejected(self) -> None:
        """Verify expired confirmation token is rejected."""
        conf_id = self.confirmation_manager.request_confirmation(
            operation="set_volume",
            parameters={"level": 100},
            timeout=0.01,
        )
        self.confirmation_manager.resolve_confirmation(conf_id, approved=True)
        time.sleep(0.02)

        with self.assertRaises(ConfirmationTimeoutError):
            self.skill.execute({
                "operation": "set_volume",
                "parameters": {"level": 100},
                "confirmation_id": conf_id,
            })

    def test_28_reused_confirmation_token_rejected(self) -> None:
        """Verify single-use token cannot be consumed a second time."""
        conf_id = self.confirmation_manager.request_confirmation(
            operation="set_volume",
            parameters={"level": 100},
        )
        self.confirmation_manager.resolve_confirmation(conf_id, approved=True)

        # First consumption succeeds
        res = self.skill.execute({
            "operation": "set_volume",
            "parameters": {"level": 100},
            "confirmation_id": conf_id,
        })
        self.assertTrue(res.success)

        # Second consumption MUST fail
        with self.assertRaises(ConfirmationRejectedError):
            self.skill.execute({
                "operation": "set_volume",
                "parameters": {"level": 100},
                "confirmation_id": conf_id,
            })

    def test_32_modified_parameters_invalidate_confirmation(self) -> None:
        """Verify modifying parameters after token issuance invalidates authorization."""
        token = self.confirmation_manager.request_confirmation(
            operation="set_volume",
            parameters={"level": 100},
        )
        self.confirmation_manager.resolve_confirmation(token, approved=True)

        # Attempt to consume with modified parameters
        with self.assertRaises(ConfirmationRejectedError):
            self.skill.execute({
                "operation": "set_volume",
                "parameters": {"level": 50},
                "confirmation_id": token,
            })

    def test_33_security_policy_classification_tiers(self) -> None:
        """Verify exact security policy classifications for all operations."""
        safe_ops = ["get_volume", "get_brightness", "lock_workstation"]
        for op in safe_ops:
            tier, _ = self.security_policy.validate_operation(op)
            self.assertEqual(tier, SystemSafetyTier.SAFE, f"'{op}' must be SAFE")

        tier_vol, _ = self.security_policy.validate_operation("set_volume", parameters={"level": 50})
        self.assertEqual(tier_vol, SystemSafetyTier.SAFE)

        tier_max_vol, _ = self.security_policy.validate_operation("set_volume", parameters={"level": 100})
        self.assertEqual(tier_max_vol, SystemSafetyTier.CONFIRMATION_REQUIRED)

    # ---------------------------------------------------------------------------
    # 7. Lifecycle Events, DI, and Concurrency
    # ---------------------------------------------------------------------------

    def test_35_lifecycle_events_emitted_for_safe_ops(self) -> None:
        """Verify execution publishes SystemSkillStarted and SystemSkillCompleted events."""
        events = []
        self.event_bus.subscribe(SystemSkillStarted, lambda e: events.append(e))
        self.event_bus.subscribe(SystemSkillCompleted, lambda e: events.append(e))

        res = self.skill.execute({"operation": "get_volume"})
        self.assertTrue(res.success)
        self.assertEqual(len(events), 2)
        self.assertIsInstance(events[0], SystemSkillStarted)
        self.assertIsInstance(events[1], SystemSkillCompleted)
        self.assertEqual(events[0].operation, "get_volume")
        self.assertEqual(events[1].operation, "get_volume")

    def test_36_lifecycle_events_emitted_for_confirmation(self) -> None:
        """Verify confirmation request publishes SystemSkillConfirmationRequired event."""
        conf_events = []
        self.event_bus.subscribe(SystemSkillConfirmationRequired, lambda e: conf_events.append(e))

        with self.assertRaises(ConfirmationRequiredError):
            self.skill.execute({"operation": "set_volume", "parameters": {"level": 100}})

        self.assertEqual(len(conf_events), 1)
        self.assertEqual(conf_events[0].operation, "set_volume")

    def test_37_dependency_injection_service_container(self) -> None:
        """Verify dependencies resolve cleanly through ServiceContainer."""
        container = ServiceContainer()
        custom_sec = SystemSecurityPolicy()
        container.register_singleton("system_security_policy", custom_sec)

        skill = SystemControlSkills(container=container)
        skill._call_lock_api = MagicMock(return_value=True)
        self.assertIs(skill.security_policy, custom_sec)

    def test_38_thread_safety_concurrent_safe_queries(self) -> None:
        """Verify concurrent execution of safe volume/brightness/lock queries."""
        ops = [
            "get_volume",
            "get_brightness",
            "lock_workstation",
        ]
        errors = []

        def worker(op_name: str) -> None:
            try:
                res = self.skill.execute({"operation": op_name})
                if not res.success:
                    errors.append(f"{op_name} failed: {res.error}")
            except Exception as e:
                errors.append(f"{op_name} raised: {e}")

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(worker, op) for op in ops * 4]
            for f in futures:
                f.result()

        self.assertEqual(len(errors), 0, f"Concurrent execution errors: {errors}")

    def test_39_hardware_agnostic_graceful_fallbacks(self) -> None:
        """Verify fallback values when audio or brightness hardware is absent."""
        with patch.object(self.skill, "_get_audio_endpoint", return_value=None), \
             patch.object(self.skill, "_get_sbc_module", return_value=None):
            vol = self.skill.get_volume()
            self.assertIn("level", vol)
            self.assertFalse(vol["muted"])

            bright = self.skill.get_brightness()
            self.assertFalse(bright["supported"])
            self.assertIsNone(bright["level"])

    def test_40_no_filesystem_or_system_state_mutation(self) -> None:
        """Verify safe operations produce zero filesystem changes."""
        sub = self.temp_path / "control_check"
        sub.mkdir()
        before_files = list(sub.glob("*"))

        self.skill.execute({"operation": "get_volume"})
        self.skill.execute({"operation": "get_brightness"})
        self.skill.execute({"operation": "lock_workstation"})

        after_files = list(sub.glob("*"))
        self.assertEqual(before_files, after_files)


if __name__ == "__main__":
    unittest.main()
