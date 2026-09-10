"""Unit Tests for Multi-Agent Models and Specifications (Sprint 18.1)."""

import unittest
from app.ai.planner.multi_agent.models import (
    AgentCapability,
    AgentManifest,
    AgentMessage,
    AgentMessageType,
    AgentRole,
    AgentStatus,
    ConflictResolution,
    CriticFeedback,
    DelegationRequest,
    DelegationResponse,
)


class TestAgentModels(unittest.TestCase):
    """Test suite for Phase 18.0 multi-agent data structures."""

    def test_agent_role_and_status_enums(self) -> None:
        """Verify role and status enumeration values."""
        self.assertEqual(AgentRole.COORDINATOR.value, "COORDINATOR")
        self.assertEqual(AgentRole.WORKER.value, "WORKER")
        self.assertEqual(AgentRole.CRITIC.value, "CRITIC")
        self.assertEqual(AgentRole.RESEARCHER.value, "RESEARCHER")

        self.assertEqual(AgentStatus.CREATED.value, "CREATED")
        self.assertEqual(AgentStatus.IDLE.value, "IDLE")
        self.assertEqual(AgentStatus.BUSY.value, "BUSY")
        self.assertEqual(AgentStatus.PAUSED.value, "PAUSED")
        self.assertEqual(AgentStatus.FAILED.value, "FAILED")
        self.assertEqual(AgentStatus.TERMINATED.value, "TERMINATED")

    def test_agent_capability_serialization(self) -> None:
        """Verify AgentCapability to_dict and from_dict roundtrip."""
        cap = AgentCapability(
            name="web_searcher",
            description="Searches internet endpoints",
            supported_actions={"web_search", "scrape"},
            domain_tags={"web", "research"},
            resource_cost=1.5,
            max_concurrency=8,
            confidence_score=0.95,
        )
        d = cap.to_dict()
        self.assertEqual(d["name"], "web_searcher")
        self.assertEqual(d["resource_cost"], 1.5)
        self.assertIn("web_search", d["supported_actions"])

        restored = AgentCapability.from_dict(d)
        self.assertEqual(restored.name, cap.name)
        self.assertEqual(restored.supported_actions, cap.supported_actions)
        self.assertEqual(restored.confidence_score, 0.95)

    def test_agent_manifest_serialization(self) -> None:
        """Verify AgentManifest serialization and deserialization."""
        cap = AgentCapability(name="calc", supported_actions={"calculate"})
        manifest = AgentManifest(
            agent_id="agent_math_1",
            name="Math Specialist",
            role=AgentRole.WORKER,
            capabilities=[cap],
            system_prompt="Solve mathematical expressions accurately.",
            max_concurrency=2,
            timeout_seconds=5.0,
            allowed_actions={"calculate"},
            metadata={"version": "1.0"},
        )
        d = manifest.to_dict()
        self.assertEqual(d["agent_id"], "agent_math_1")
        self.assertEqual(d["role"], "WORKER")
        self.assertEqual(len(d["capabilities"]), 1)

        restored = AgentManifest.from_dict(d)
        self.assertEqual(restored.agent_id, manifest.agent_id)
        self.assertEqual(restored.role, AgentRole.WORKER)
        self.assertEqual(restored.timeout_seconds, 5.0)
        self.assertIn("calculate", restored.allowed_actions)

    def test_agent_message_serialization(self) -> None:
        """Verify AgentMessage serialization and defaults."""
        msg = AgentMessage(
            sender="agent_a",
            recipient="agent_b",
            message_type=AgentMessageType.REQUEST,
            content={"query": "weather in NYC"},
            correlation_id="corr_123",
        )
        self.assertTrue(msg.id)
        d = msg.to_dict()
        self.assertEqual(d["sender"], "agent_a")
        self.assertEqual(d["recipient"], "agent_b")
        self.assertEqual(d["message_type"], "REQUEST")
        self.assertEqual(d["correlation_id"], "corr_123")

        restored = AgentMessage.from_dict(d)
        self.assertEqual(restored.id, msg.id)
        self.assertEqual(restored.sender, msg.sender)
        self.assertEqual(restored.message_type, AgentMessageType.REQUEST)
        self.assertEqual(restored.content, {"query": "weather in NYC"})

    def test_delegation_request_and_response(self) -> None:
        """Verify DelegationRequest and DelegationResponse contracts."""
        req = DelegationRequest(
            task_id="t_1",
            action="summarize",
            target="report.txt",
            parameters={"max_words": 100},
            context={"session": "s_1"},
            delegated_by="coordinator_main",
            timeout_seconds=10.0,
        )
        req_dict = req.to_dict()
        req_restored = DelegationRequest.from_dict(req_dict)
        self.assertEqual(req_restored.task_id, "t_1")
        self.assertEqual(req_restored.timeout_seconds, 10.0)

        resp = DelegationResponse(
            task_id="t_1",
            agent_id="worker_summary",
            success=True,
            output="Summary text...",
            artifacts={"token_count": 42},
            duration=0.12,
            confidence=0.98,
        )
        resp_dict = resp.to_dict()
        resp_restored = DelegationResponse.from_dict(resp_dict)
        self.assertTrue(resp_restored.success)
        self.assertEqual(resp_restored.output, "Summary text...")
        self.assertEqual(resp_restored.confidence, 0.98)

    def test_critic_feedback_and_conflict_resolution(self) -> None:
        """Verify CriticFeedback and ConflictResolution models."""
        fb = CriticFeedback(
            task_id="t_crit",
            critic_id="critic_agent",
            passed=False,
            score=0.4,
            critique="Output lacks required precision",
            suggestions=["Increase decimal precision to 4 places"],
        )
        fb_dict = fb.to_dict()
        fb_restored = CriticFeedback.from_dict(fb_dict)
        self.assertFalse(fb_restored.passed)
        self.assertEqual(fb_restored.score, 0.4)
        self.assertEqual(len(fb_restored.suggestions), 1)

        conflict = ConflictResolution(
            conflict_id="conf_1",
            conflict_type="OUTPUT_MISMATCH",
            parties=["agent_1", "agent_2"],
            strategy="CONFIDENCE_WEIGHTED",
            resolved=True,
            winner_agent_id="agent_1",
            resolution_data={"final_value": 42},
        )
        c_dict = conflict.to_dict()
        c_restored = ConflictResolution.from_dict(c_dict)
        self.assertTrue(c_restored.resolved)
        self.assertEqual(c_restored.winner_agent_id, "agent_1")


if __name__ == "__main__":
    unittest.main()
