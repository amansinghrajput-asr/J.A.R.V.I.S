"""Inter-Agent Communication Protocol & Bus for Phase 18.0.

Provides asynchronous and synchronous message delivery, mailbox routing,
broadcast channels, and request-response patterns with correlation tracking.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set

from app.ai.planner.events import AgentMessageSent, PlannerEventBus
from app.ai.planner.multi_agent.models import AgentMessage, AgentMessageType

logger = logging.getLogger("app.ai.planner.multi_agent.protocol")


class AgentCommunicationBus:
    """Thread-safe and async-compatible communication bus for autonomous agents.

    Supports:
    - Point-to-point message dispatching to agent mailboxes
    - Broadcast messaging to all active agents
    - Synchronous and asynchronous request-response patterns with timeout enforcement
    - Event mirroring to PlannerEventBus for system-wide observability
    """

    def __init__(
        self,
        event_bus: Optional[PlannerEventBus] = None,
        max_mailbox_size: int = 1000,
    ) -> None:
        """Initialize communication bus.

        Args:
            event_bus: Optional PlannerEventBus for event mirroring.
            max_mailbox_size: Maximum buffered messages per mailbox.
        """
        self._lock = threading.RLock()
        self._event_bus = event_bus
        self._max_mailbox_size = max_mailbox_size
        self._mailboxes: Dict[str, queue.Queue[AgentMessage]] = {}
        self._async_mailboxes: Dict[str, asyncio.Queue[AgentMessage]] = {}
        self._pending_sync_responses: Dict[str, Dict[str, Any]] = {}
        self._pending_async_responses: Dict[str, asyncio.Future[AgentMessage]] = {}
        self._message_history: List[AgentMessage] = []
        self._history_limit = 500

    def register_agent(self, agent_id: str) -> None:
        """Register an agent mailbox on the bus."""
        with self._lock:
            if agent_id not in self._mailboxes:
                self._mailboxes[agent_id] = queue.Queue(maxsize=self._max_mailbox_size)
            if agent_id not in self._async_mailboxes:
                self._async_mailboxes[agent_id] = asyncio.Queue(maxsize=self._max_mailbox_size)
            logger.debug("Registered mailbox for agent '%s'", agent_id)

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent mailbox and clean up pending futures."""
        with self._lock:
            self._mailboxes.pop(agent_id, None)
            self._async_mailboxes.pop(agent_id, None)
            logger.debug("Unregistered mailbox for agent '%s'", agent_id)

    def is_registered(self, agent_id: str) -> bool:
        """Check if an agent mailbox is registered."""
        with self._lock:
            return agent_id in self._mailboxes

    def get_mailbox(self, agent_id: str) -> Optional[queue.Queue[AgentMessage]]:
        """Retrieve synchronous mailbox queue for an agent."""
        with self._lock:
            return self._mailboxes.get(agent_id)

    def get_async_mailbox(self, agent_id: str) -> asyncio.Queue[AgentMessage]:
        """Retrieve or create an async mailbox queue for an agent."""
        with self._lock:
            if agent_id not in self._async_mailboxes:
                self._async_mailboxes[agent_id] = asyncio.Queue(maxsize=self._max_mailbox_size)
            return self._async_mailboxes[agent_id]

    def send(self, message: AgentMessage) -> bool:
        """Dispatch a message synchronously to recipient mailbox or broadcast channel.

        Returns:
            True if delivered to at least one recipient, False otherwise.
        """
        if not message.recipient:
            logger.warning("Dropped message %s: No recipient specified.", message.id)
            return False

        delivered = False
        with self._lock:
            # 1. Record history
            self._message_history.append(message)
            if len(self._message_history) > self._history_limit:
                self._message_history.pop(0)

            # 2. Check pending correlation responses (must be a reply/response, not the request)
            if (
                message.correlation_id
                and message.correlation_id in self._pending_sync_responses
                and message.message_type in (AgentMessageType.RESPONSE, AgentMessageType.ERROR)
            ):
                slot = self._pending_sync_responses[message.correlation_id]
                slot["response"] = message
                slot["event"].set()

            # 3. Deliver
            if message.recipient == "broadcast":
                for aid, q in self._mailboxes.items():
                    if aid != message.sender:
                        try:
                            q.put_nowait(message)
                            delivered = True
                        except queue.Full:
                            logger.warning("Mailbox for agent '%s' is full; dropped broadcast.", aid)
            else:
                target_q = self._mailboxes.get(message.recipient)
                if target_q is not None:
                    try:
                        target_q.put_nowait(message)
                        delivered = True
                    except queue.Full:
                        logger.warning("Mailbox for agent '%s' is full.", message.recipient)
                else:
                    logger.debug("Recipient '%s' not registered on communication bus.", message.recipient)

        # 4. Mirror to PlannerEventBus
        if self._event_bus is not None:
            self._event_bus.publish(
                AgentMessageSent(
                    message_id=message.id,
                    sender=message.sender,
                    recipient=message.recipient,
                    message_type=message.message_type.value if hasattr(message.message_type, "value") else str(message.message_type),
                    correlation_id=message.correlation_id,
                )
            )

        return delivered

    def send_and_wait(self, message: AgentMessage, timeout: float = 5.0) -> Optional[AgentMessage]:
        """Send a request message and block until matching correlation response arrives or timeout.

        Args:
            message: Request message to send.
            timeout: Maximum wait time in seconds.

        Returns:
            Matching AgentMessage response or None on timeout.
        """
        corr_id = message.correlation_id or message.id
        wait_event = threading.Event()
        slot = {"event": wait_event, "response": None}

        with self._lock:
            self._pending_sync_responses[corr_id] = slot

        # Send message
        delivered = self.send(message)
        if not delivered:
            with self._lock:
                self._pending_sync_responses.pop(corr_id, None)
            return None

        # Wait
        signaled = wait_event.wait(timeout=timeout)

        with self._lock:
            res_slot = self._pending_sync_responses.pop(corr_id, None)

        if signaled and res_slot and res_slot.get("response"):
            return res_slot["response"]
        return None

    async def send_async(self, message: AgentMessage) -> bool:
        """Dispatch a message asynchronously to recipient async mailbox or broadcast channel."""
        if not message.recipient:
            return False

        delivered = False
        with self._lock:
            self._message_history.append(message)
            if len(self._message_history) > self._history_limit:
                self._message_history.pop(0)

            # Check async future correlation
            if (
                message.correlation_id
                and message.correlation_id in self._pending_async_responses
                and message.message_type in (AgentMessageType.RESPONSE, AgentMessageType.ERROR)
            ):
                fut = self._pending_async_responses[message.correlation_id]
                if not fut.done():
                    try:
                        fut.get_loop().call_soon_threadsafe(fut.set_result, message)
                    except Exception:
                        fut.set_result(message)

            # Also feed sync mailboxes
            if message.recipient == "broadcast":
                for aid, q in self._mailboxes.items():
                    if aid != message.sender:
                        try:
                            q.put_nowait(message)
                            delivered = True
                        except queue.Full:
                            pass
            else:
                target_q = self._mailboxes.get(message.recipient)
                if target_q is not None:
                    try:
                        target_q.put_nowait(message)
                        delivered = True
                    except queue.Full:
                        pass

        # Also feed async mailboxes
        if message.recipient == "broadcast":
            for aid, aq in list(self._async_mailboxes.items()):
                if aid != message.sender:
                    try:
                        aq.put_nowait(message)
                        delivered = True
                    except asyncio.QueueFull:
                        pass
        else:
            target_aq = self._async_mailboxes.get(message.recipient)
            if target_aq is not None:
                try:
                    target_aq.put_nowait(message)
                    delivered = True
                except asyncio.QueueFull:
                    pass

        if self._event_bus is not None:
            evt = AgentMessageSent(
                message_id=message.id,
                sender=message.sender,
                recipient=message.recipient,
                message_type=message.message_type.value if hasattr(message.message_type, "value") else str(message.message_type),
                correlation_id=message.correlation_id,
            )
            if hasattr(self._event_bus, "publish_async") and callable(getattr(self._event_bus, "publish_async")):
                await self._event_bus.publish_async(evt)
            else:
                self._event_bus.publish(evt)

        return delivered

    async def send_and_wait_async(self, message: AgentMessage, timeout: float = 5.0) -> Optional[AgentMessage]:
        """Send a request message asynchronously and await matching response."""
        corr_id = message.correlation_id or message.id
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[AgentMessage] = loop.create_future()

        with self._lock:
            self._pending_async_responses[corr_id] = fut

        delivered = await self.send_async(message)
        if not delivered:
            with self._lock:
                self._pending_async_responses.pop(corr_id, None)
            return None

        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError):
            return None
        finally:
            with self._lock:
                self._pending_async_responses.pop(corr_id, None)

    def get_history(self, limit: int = 50) -> List[AgentMessage]:
        """Retrieve recent dispatched messages."""
        with self._lock:
            return list(self._message_history[-limit:])
