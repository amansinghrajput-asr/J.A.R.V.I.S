"""Unit Tests for Agent Communication Bus and Base Agents (Sprint 18.1)."""

import asyncio
import threading
import time
import unittest
from typing import List

from app.ai.planner.events import (
    AgentMessageSent,
    AgentStatusChanged,
    PlannerEvent,
    PlannerEventBus,
    TaskCompletedByAgent,
)
from app.ai.planner.multi_agent.base import CriticAgent, WorkerAgent
from app.ai.planner.multi_agent.models import (
    AgentCapability,
    AgentManifest,
    AgentMessage,
    AgentMessageType,
    AgentRole,
    AgentStatus,
    DelegationRequest,
)
from app.ai.planner.multi_agent.protocol import AgentCommunicationBus


class TestAgentCommunication(unittest.TestCase):
    """Test suite for AgentCommunicationBus, WorkerAgent, and CriticAgent."""

    def setUp(self) -> None:
        self.bus = PlannerEventBus()
        self.events: List[PlannerEvent] = []
        self.bus.subscribe(PlannerEvent, self.events.append)
        self.comm_bus = AgentCommunicationBus(event_bus=self.bus)

    def test_mailbox_registration(self) -> None:
        """Verify registering and unregistering agent mailboxes."""
        self.assertFalse(self.comm_bus.is_registered("agent_1"))
        self.comm_bus.register_agent("agent_1")
        self.assertTrue(self.comm_bus.is_registered("agent_1"))

        q = self.comm_bus.get_mailbox("agent_1")
        self.assertIsNotNone(q)

        self.comm_bus.unregister_agent("agent_1")
        self.assertFalse(self.comm_bus.is_registered("agent_1"))

    def test_point_to_point_delivery(self) -> None:
        """Verify delivering message from agent A to agent B."""
        self.comm_bus.register_agent("agent_a")
        self.comm_bus.register_agent("agent_b")

        msg = AgentMessage(
            sender="agent_a",
            recipient="agent_b",
            message_type=AgentMessageType.INFORM,
            content={"data": "hello B"},
        )
        delivered = self.comm_bus.send(msg)
        self.assertTrue(delivered)

        q_b = self.comm_bus.get_mailbox("agent_b")
        received = q_b.get_nowait()
        self.assertEqual(received.content["data"], "hello B")

        # Verify event emission
        msg_events = [e for e in self.events if isinstance(e, AgentMessageSent)]
        self.assertEqual(len(msg_events), 1)
        self.assertEqual(msg_events[0].sender, "agent_a")
        self.assertEqual(msg_events[0].recipient, "agent_b")

    def test_broadcast_delivery(self) -> None:
        """Verify broadcast reaches all agents except sender."""
        self.comm_bus.register_agent("sender")
        self.comm_bus.register_agent("rec_1")
        self.comm_bus.register_agent("rec_2")

        msg = AgentMessage(
            sender="sender",
            recipient="broadcast",
            message_type=AgentMessageType.BROADCAST,
            content={"announcement": "system reboot"},
        )
        delivered = self.comm_bus.send(msg)
        self.assertTrue(delivered)

        # rec_1 and rec_2 got it, sender did not
        q_1 = self.comm_bus.get_mailbox("rec_1")
        q_2 = self.comm_bus.get_mailbox("rec_2")
        q_s = self.comm_bus.get_mailbox("sender")

        self.assertEqual(q_1.get_nowait().content["announcement"], "system reboot")
        self.assertEqual(q_2.get_nowait().content["announcement"], "system reboot")
        self.assertTrue(q_s.empty())

    def test_send_and_wait_sync(self) -> None:
        """Verify synchronous request-response flow with correlation matching."""
        self.comm_bus.register_agent("client")
        self.comm_bus.register_agent("server")

        def server_loop() -> None:
            q = self.comm_bus.get_mailbox("server")
            req = q.get(timeout=1.0)
            # Reply
            reply = AgentMessage(
                sender="server",
                recipient="client",
                message_type=AgentMessageType.RESPONSE,
                content={"result": "OK 200"},
                correlation_id=req.correlation_id or req.id,
            )
            self.comm_bus.send(reply)

        t = threading.Thread(target=server_loop, daemon=True)
        t.start()

        req = AgentMessage(
            sender="client",
            recipient="server",
            message_type=AgentMessageType.REQUEST,
            content={"action": "ping"},
            correlation_id="corr_999",
        )
        response = self.comm_bus.send_and_wait(req, timeout=1.0)
        t.join(timeout=1.0)

        self.assertIsNotNone(response)
        self.assertEqual(response.content["result"], "OK 200")

    def test_send_and_wait_async(self) -> None:
        """Verify asynchronous request-response flow."""
        async def run_async_test() -> None:
            self.comm_bus.register_agent("async_client")
            self.comm_bus.register_agent("async_server")

            async def mock_server() -> None:
                aq = self.comm_bus.get_async_mailbox("async_server")
                req = await aq.get()
                reply = AgentMessage(
                    sender="async_server",
                    recipient="async_client",
                    message_type=AgentMessageType.RESPONSE,
                    content={"answer": 42},
                    correlation_id=req.id,
                )
                await self.comm_bus.send_async(reply)

            server_task = asyncio.create_task(mock_server())

            req = AgentMessage(
                sender="async_client",
                recipient="async_server",
                message_type=AgentMessageType.REQUEST,
                content={"question": "life"},
            )
            resp = await self.comm_bus.send_and_wait_async(req, timeout=1.0)
            await server_task

            self.assertIsNotNone(resp)
            self.assertEqual(resp.content["answer"], 42)

        asyncio.run(run_async_test())

    def test_worker_agent_execution(self) -> None:
        """Verify WorkerAgent delegation execution and status lifecycle."""
        manifest = AgentManifest(
            agent_id="worker_math",
            name="Math Worker",
            role=AgentRole.WORKER,
            capabilities=[AgentCapability(name="calc", supported_actions={"calculate"})],
        )
        worker = WorkerAgent(
            manifest=manifest,
            handlers={"calculate": lambda t: f"Result: {eval(t.target or '0')}"},
            communication_bus=self.comm_bus,
            event_bus=self.bus,
        )
        worker.initialize()
        self.assertEqual(worker.status, AgentStatus.IDLE)

        req = DelegationRequest(
            task_id="task_calc_1",
            action="calculate",
            target="10 * 5",
        )
        response = worker.execute_delegation(req)

        self.assertTrue(response.success)
        self.assertEqual(response.output, "Result: 50")
        self.assertEqual(worker.status, AgentStatus.IDLE)

        # Check events
        status_events = [e for e in self.events if isinstance(e, AgentStatusChanged)]
        self.assertTrue(len(status_events) > 0)
        comp_events = [e for e in self.events if isinstance(e, TaskCompletedByAgent)]
        self.assertEqual(len(comp_events), 1)
        self.assertEqual(comp_events[0].agent_id, "worker_math")

        worker.shutdown()
        self.assertEqual(worker.status, AgentStatus.TERMINATED)

    def test_critic_agent_reflection(self) -> None:
        """Verify CriticAgent evaluation and feedback output."""
        critic = CriticAgent(event_bus=self.bus)
        critic.initialize()

        # Valid data
        fb_good = critic.critique("Completed calculation successfully", context={"task_id": "t_1"})
        self.assertTrue(fb_good.passed)
        self.assertEqual(fb_good.score, 1.0)

        # Empty data
        fb_bad = critic.critique("", context={"task_id": "t_2"})
        self.assertFalse(fb_bad.passed)
        self.assertTrue(len(fb_bad.suggestions) > 0)


if __name__ == "__main__":
    unittest.main()
