"""Regression tests for event loop lifecycle, loop-aware AsyncClient, and lifecycle cleanup.

Validates that:
A. Loop A -> request -> Loop A closes -> Loop B -> request completes without error.
B. Repeated sequential worker-loop operations do not accumulate stale clients.
C. Custom injected async client remains untouched across operations.
D. Sync provider remains unaffected.
E. Provider cleanup/close remains safe.
F. Existing UIBridge voice -> conversation lifecycle remains fixed.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from PySide6.QtWidgets import QApplication

from app.ai.provider import GeminiProvider
from app.core.presentation import PresentationAdapter
from app.core.state import AssistantState
from app.core.state_manager import AssistantStateManager
from app.ui.bridge import UIBridge


def _mock_gemini_response_payload(text: str = "Test response") -> dict[str, Any]:
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": text}],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 15,
            "totalTokenCount": 25,
        },
    }


class TestEventLoopLifecycle(unittest.TestCase):
    """Test suite for event-loop lifecycle, loop-aware async client, and cleanup."""

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        cls._qapp = QApplication.instance() or QApplication(sys.argv)

    def test_gemini_provider_multi_loop_behavior(self) -> None:
        """A. Loop A -> request -> Loop A closes -> Loop B -> request.

        Verify the second operation does NOT fail with 'Event loop is closed'.
        """
        provider = GeminiProvider(api_key="test_api_key")
        payload = {"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]}
        mock_data = _mock_gemini_response_payload("Loop 1 response")
        mock_data_2 = _mock_gemini_response_payload("Loop 2 response")

        # Loop 1
        loop1 = asyncio.new_event_loop()
        asyncio.set_event_loop(loop1)
        try:
            with patch.object(
                httpx.AsyncClient,
                "post",
                new=AsyncMock(return_value=httpx.Response(200, json=mock_data)),
            ):
                resp1 = loop1.run_until_complete(provider.generate_async(payload))
                self.assertEqual(resp1.content, "Loop 1 response")
        finally:
            loop1.close()
            asyncio.set_event_loop(None)

        # Loop 2 - same provider instance
        loop2 = asyncio.new_event_loop()
        asyncio.set_event_loop(loop2)
        try:
            with patch.object(
                httpx.AsyncClient,
                "post",
                new=AsyncMock(return_value=httpx.Response(200, json=mock_data_2)),
            ):
                resp2 = loop2.run_until_complete(provider.generate_async(payload))
                self.assertEqual(resp2.content, "Loop 2 response")
        finally:
            loop2.close()
            asyncio.set_event_loop(None)

    def test_sequential_worker_loops_do_not_accumulate_stale_clients(self) -> None:
        """B. Repeated sequential worker-loop operations do not accumulate stale clients.

        Runs multiple sequential transient loops and confirms that after each operation,
        the transient AsyncClient has been cleanly closed and reset on the provider,
        preventing an unbounded accumulation of abandoned clients.
        """
        provider = GeminiProvider(api_key="test_api_key")
        payload = {"contents": [{"role": "user", "parts": [{"text": "Query"}]}]}
        mock_data = _mock_gemini_response_payload("Sequential response")

        closed_clients = []
        original_aclose = httpx.AsyncClient.aclose

        async def tracking_aclose(client_self: httpx.AsyncClient) -> None:
            closed_clients.append(client_self)
            await original_aclose(client_self)

        with patch.object(
            httpx.AsyncClient,
            "post",
            new=AsyncMock(return_value=httpx.Response(200, json=mock_data)),
        ), patch.object(httpx.AsyncClient, "aclose", new=tracking_aclose):
            for i in range(5):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    resp = loop.run_until_complete(provider.generate_async(payload))
                    self.assertEqual(resp.content, "Sequential response")
                finally:
                    loop.close()
                    asyncio.set_event_loop(None)

                # Confirm the client was cleanly closed and discarded on its owning loop
                self.assertIsNone(provider._async_client)
                self.assertIsNone(provider._async_client_loop)
                self.assertEqual(len(closed_clients), i + 1)

    def test_custom_injected_async_client_remains_untouched(self) -> None:
        """C. Custom injected async client remains untouched.

        Existing injected/mock async clients must continue working across calls
        and must NOT be closed or discarded by generate_async().
        """
        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_client.aclose = AsyncMock()
        mock_resp = httpx.Response(200, json=_mock_gemini_response_payload("Mocked response"))
        mock_client.post = AsyncMock(return_value=mock_resp)

        provider = GeminiProvider(api_key="test_api_key", async_client=mock_client)
        payload = {"contents": [{"role": "user", "parts": [{"text": "Test"}]}]}

        # Call across two loops
        for _ in range(2):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                resp = loop.run_until_complete(provider.generate_async(payload))
                self.assertEqual(resp.content, "Mocked response")
            finally:
                loop.close()
                asyncio.set_event_loop(None)

        # Injected client was never closed and remains the active client
        mock_client.aclose.assert_not_called()
        self.assertIs(provider._async_client, mock_client)

    def test_sync_provider_remains_unaffected(self) -> None:
        """D. Sync provider remains unaffected.

        Existing synchronous provider behavior must remain unchanged and unaffected
        by any opening/closing of asyncio event loops.
        """
        provider = GeminiProvider(api_key="test_api_key")
        payload = {"contents": [{"role": "user", "parts": [{"text": "Sync call"}]}]}
        mock_data = _mock_gemini_response_payload("Sync response")

        # Synchronous execution
        with patch.object(
            httpx.Client,
            "post",
            return_value=httpx.Response(200, json=mock_data),
        ):
            resp = provider.generate(payload)
            self.assertEqual(resp.content, "Sync response")

        # Interleave with an async loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            with patch.object(
                httpx.AsyncClient,
                "post",
                new=AsyncMock(return_value=httpx.Response(200, json=mock_data)),
            ):
                async_resp = loop.run_until_complete(provider.generate_async(payload))
                self.assertEqual(async_resp.content, "Sync response")
        finally:
            loop.close()
            asyncio.set_event_loop(None)

        # Another synchronous call after loop closed
        with patch.object(
            httpx.Client,
            "post",
            return_value=httpx.Response(200, json=mock_data),
        ):
            resp2 = provider.generate(payload)
            self.assertEqual(resp2.content, "Sync response")

    def test_provider_cleanup_close_remains_safe(self) -> None:
        """E. Provider cleanup/close remains safe.

        Calling close() and aclose() handles uninitialized, open, and closed loops safely
        without raising unhandled exceptions or awaiting across closed loops.
        """
        provider = GeminiProvider(api_key="test_api_key")

        # 1. Close uninitialized provider
        provider.close()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(provider.aclose())
        finally:
            loop.close()
            asyncio.set_event_loop(None)

        # 2. Initialize sync client and close
        with patch.object(
            httpx.Client,
            "post",
            return_value=httpx.Response(200, json=_mock_gemini_response_payload("ok")),
        ):
            provider.generate({"contents": []})
            self.assertIsNotNone(provider._sync_client)
            provider.close()
            self.assertTrue(provider._sync_client.is_closed)

        # 3. aclose with a simulated closed loop should NOT raise RuntimeError
        provider._async_client = MagicMock(spec=httpx.AsyncClient)
        provider._async_client.is_closed = False
        closed_loop = MagicMock(spec=asyncio.AbstractEventLoop)
        closed_loop.is_closed.return_value = True
        provider._async_client_loop = closed_loop

        loop2 = asyncio.new_event_loop()
        asyncio.set_event_loop(loop2)
        try:
            # Must safely skip awaiting on closed loop and reset references
            loop2.run_until_complete(provider.aclose())
            self.assertIsNone(provider._async_client)
            self.assertIsNone(provider._async_client_loop)
        finally:
            loop2.close()
            asyncio.set_event_loop(None)

    def test_aclose_does_not_await_client_from_different_loop(self) -> None:
        """G. Proves that a client owned by Loop A is not awaited/closed from Loop B.

        When Loop A owns an internally managed AsyncClient, invoking aclose() from
        Loop B must safely skip awaiting client.aclose() on Loop B and clear
        provider references.
        """
        provider = GeminiProvider(api_key="test_api_key")

        loop_a = asyncio.new_event_loop()
        loop_b = asyncio.new_event_loop()

        try:
            # Create an internally managed mock client owned by Loop A (Loop A is open)
            mock_client = MagicMock(spec=httpx.AsyncClient)
            mock_client.is_closed = False
            mock_client.aclose = AsyncMock()

            provider._async_client = mock_client
            provider._async_client_loop = loop_a

            # Execute aclose on Loop B while Loop A is open but different
            asyncio.set_event_loop(loop_b)
            loop_b.run_until_complete(provider.aclose())

            # Loop B must NOT await aclose on the client owned by Loop A
            mock_client.aclose.assert_not_called()

            # Provider references must be cleared safely afterward
            self.assertIsNone(provider._async_client)
            self.assertIsNone(provider._async_client_loop)
        finally:
            loop_a.close()
            loop_b.close()
            asyncio.set_event_loop(None)

    def test_voice_to_conversation_lifecycle_in_uibridge(self) -> None:
        """F. Existing UIBridge voice -> conversation lifecycle remains fixed.

        Simulate successful voice interaction.
        Then submit a normal conversational command.
        Verify both complete successfully without event-loop-closed failure.
        """
        state_mgr = AssistantStateManager(initial_state=AssistantState.IDLE)
        adapter = PresentationAdapter(state_manager=state_mgr)
        adapter.start_voice_interaction = AsyncMock(return_value="Voice interaction result")
        adapter.submit_command = AsyncMock(return_value="Conversational response")
        bridge = UIBridge(presentation_adapter=adapter)

        completed_results = []
        bridge.command_completed.connect(completed_results.append)

        failed_errors = []
        bridge.command_failed.connect(failed_errors.append)

        try:
            # 1. Trigger voice interaction
            triggered = bridge.start_voice_interaction()
            self.assertTrue(triggered)

            t0 = time.time()
            while len(completed_results) < 1 and time.time() - t0 < 3.0:
                time.sleep(0.05)
                self._qapp.processEvents()

            self.assertEqual(len(completed_results), 1)
            self.assertEqual(completed_results[0], "Voice interaction result")
            self.assertEqual(len(failed_errors), 0)

            # 2. Subsequent text command
            bridge.submit_command("hello jarvis")
            t0 = time.time()
            while len(completed_results) < 2 and time.time() - t0 < 3.0:
                time.sleep(0.05)
                self._qapp.processEvents()

            self.assertEqual(len(completed_results), 2)
            self.assertEqual(completed_results[1], "Conversational response")
            self.assertEqual(len(failed_errors), 0)
        finally:
            bridge.close()


if __name__ == "__main__":
    unittest.main()
