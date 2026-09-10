"""Unit tests for HITL Gateway, Human Proxy Agent, and Swarm Policy Engine (Sprint 19.5).

Covers synchronous and asynchronous approval workflows, risk levels, guidance injection,
HumanProxyAgent delegation, policy capability guardrails, depth limits, and action blocking.
"""

import asyncio
import unittest

from app.ai.planner.events import PlannerEventBus
from app.ai.planner.multi_agent.models import DelegationRequest
from app.ai.planner.swarm.hitl import (
    ApprovalDecision,
    HumanProxyAgent,
    InterventionGateway,
    RiskLevel,
)
from app.ai.planner.swarm.policy import SwarmPolicyEngine


class TestSwarmHITL(unittest.TestCase):
    """Test suite for InterventionGateway and HumanProxyAgent."""

    def test_low_risk_auto_approval(self) -> None:
        """Verify LOW risk requests auto-approve by default without interactive handler."""
        gateway = InterventionGateway()
        decision = gateway.request_approval(
            action="read_doc",
            target="readme.txt",
            risk_level=RiskLevel.LOW,
        )

        self.assertTrue(decision.approved)
        self.assertEqual(decision.decided_by, "policy_auto_approve")

    def test_high_risk_requires_handler(self) -> None:
        """Verify HIGH risk requests are denied if no handler authorizes them."""
        gateway = InterventionGateway()
        decision = gateway.request_approval(
            action="delete_partition",
            target="/dev/sda1",
            risk_level=RiskLevel.CRITICAL,
        )

        self.assertFalse(decision.approved)
        self.assertIn("requires interactive approval", decision.reason)

    def test_custom_approval_handler_resolution(self) -> None:
        """Verify registered custom approval handler resolves decisions."""
        gateway = InterventionGateway()

        def _operator_handler(req: dict) -> ApprovalDecision:
            if req["target"] == "approved_target":
                return ApprovalDecision(
                    request_id=req["request_id"],
                    approved=True,
                    decided_by="senior_operator",
                    reason="Authorized for maintenance window.",
                )
            return ApprovalDecision(
                request_id=req["request_id"],
                approved=False,
                decided_by="senior_operator",
                reason="Target unauthorized.",
            )

        gateway.register_approval_handler(_operator_handler)

        res_ok = gateway.request_approval(action="restart_service", target="approved_target")
        self.assertTrue(res_ok.approved)
        self.assertEqual(res_ok.decided_by, "senior_operator")

        res_deny = gateway.request_approval(action="restart_service", target="forbidden_target")
        self.assertFalse(res_deny.approved)

    def test_async_approval_workflow(self) -> None:
        """Verify asynchronous approval resolution."""
        async def _run_async_test() -> None:
            gateway = InterventionGateway()
            gateway.register_approval_handler(
                lambda req: ApprovalDecision(
                    request_id=req["request_id"],
                    approved=True,
                    decided_by="async_admin",
                )
            )

            decision = await gateway.request_approval_async(
                action="deploy_package",
                target="v2.0",
                risk_level=RiskLevel.HIGH,
            )
            self.assertTrue(decision.approved)
            self.assertEqual(decision.decided_by, "async_admin")

        asyncio.run(_run_async_test())

    def test_guidance_injection_and_retrieval(self) -> None:
        """Verify operator guidance injection into swarms."""
        gateway = InterventionGateway()
        gateway.inject_guidance("swarm_dev", "Prioritize unit test coverage over performance.")
        gateway.inject_guidance("swarm_dev", "Ensure PEP8 adherence.")

        guidance = gateway.get_guidance("swarm_dev")
        self.assertEqual(len(guidance), 2)
        self.assertIn("Prioritize unit test coverage over performance.", guidance)

    def test_human_proxy_agent_delegation(self) -> None:
        """Verify HumanProxyAgent bridges delegation requests to the gateway."""
        gateway = InterventionGateway()
        gateway.register_approval_handler(
            lambda req: ApprovalDecision(
                request_id=req["request_id"],
                approved=True,
                reason="Approved by proxy operator",
            )
        )
        proxy = HumanProxyAgent(gateway=gateway, agent_id="human_proxy")

        req = DelegationRequest(
            task_id="task_hitl",
            action="reboot_host",
            parameters={"risk_level": "HIGH"},
        )
        res = proxy.execute_delegation(req)

        self.assertTrue(res.success)
        self.assertEqual(res.agent_id, "human_proxy")
        self.assertIsNone(res.error)


class TestSwarmPolicyEngine(unittest.TestCase):
    """Test suite for SwarmPolicyEngine."""

    def test_denies_dangerous_commands(self) -> None:
        """Verify policy engine detects and denies dangerous shell patterns."""
        policy = SwarmPolicyEngine()

        valid, reason = policy.validate_action(action="run_command", target="rm -rf /")
        self.assertFalse(valid)
        self.assertIn("violates safety policy pattern", reason)

        valid_safe, reason_safe = policy.validate_action(action="run_command", target="echo hello")
        self.assertTrue(valid_safe)
        self.assertIsNone(reason_safe)

    def test_recursion_depth_policy(self) -> None:
        """Verify maximum recursion depth enforcement."""
        policy = SwarmPolicyEngine(max_depth=2)

        valid_1, _ = policy.validate_depth(current_depth=1)
        valid_2, _ = policy.validate_depth(current_depth=2)
        valid_3, reason_3 = policy.validate_depth(current_depth=3)

        self.assertTrue(valid_1)
        self.assertTrue(valid_2)
        self.assertFalse(valid_3)
        self.assertIn("Exceeded maximum swarm depth", reason_3)

    def test_strict_action_allowlist(self) -> None:
        """Verify allowlist blocking unpermitted actions."""
        policy = SwarmPolicyEngine(allowed_actions={"read", "query", "summarize"})

        v_ok, _ = policy.validate_action("read")
        self.assertTrue(v_ok)

        v_denied, reason = policy.validate_action("write_file")
        self.assertFalse(v_denied)
        self.assertIn("not in allowed actions policy", reason)


if __name__ == "__main__":
    unittest.main()
