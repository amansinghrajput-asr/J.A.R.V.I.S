"""Unit tests for the J.A.R.V.I.S AI Subsystem."""

from __future__ import annotations

import asyncio
import json
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import httpx

from app.ai.manager import AIManager, ai_manager
from app.ai.models import (
    AIAuthenticationError,
    AIConfigError,
    AIError,
    AIProviderError,
    AIRateLimitError,
    AIResponse,
    AITimeoutError,
    ChatMessage,
    GenerationConfig,
    PromptError,
    Role,
)
from app.ai.prompt import DEFAULT_SYSTEM_PROMPT, PromptBuilder
from app.ai.provider import GeminiProvider
from app.core.config import settings
from app.core.container import JarvisException, ServiceContainer
from app.core.event_bus import Event, EventBus
from app.memory.manager import MemoryManager
from app.memory.models import ConversationMemory


def _mock_gemini_response_payload(
    text: str = "Yes, Sir. How may I assist you?",
    prompt_tokens: int = 25,
    completion_tokens: int = 12,
    total_tokens: int = 37,
    finish_reason: str = "STOP",
) -> dict[str, Any]:
    """Generate a valid mock Google Gemini API response dictionary."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": text}],
                    "role": "model",
                },
                "finishReason": finish_reason,
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": completion_tokens,
            "totalTokenCount": total_tokens,
        },
    }


class TestAIModels(unittest.TestCase):
    """Test suite for AI data models, generation configuration, and exception hierarchy."""

    def test_exception_hierarchy(self) -> None:
        """Verify inheritance tree of AI exceptions."""
        self.assertTrue(issubclass(AIError, JarvisException))
        self.assertTrue(issubclass(AIConfigError, AIError))
        self.assertTrue(issubclass(AIProviderError, AIError))
        self.assertTrue(issubclass(AIRateLimitError, AIProviderError))
        self.assertTrue(issubclass(AIAuthenticationError, AIProviderError))
        self.assertTrue(issubclass(AITimeoutError, AIProviderError))
        self.assertTrue(issubclass(PromptError, AIError))

    def test_chat_message_model(self) -> None:
        """Verify ChatMessage initialization and dict export."""
        msg = ChatMessage(role=Role.USER.value, content="Hello Jarvis", metadata={"source": "test"})
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "Hello Jarvis")
        self.assertGreater(msg.timestamp, 0.0)
        d = msg.to_dict()
        self.assertEqual(d["role"], "user")
        self.assertEqual(d["content"], "Hello Jarvis")
        self.assertEqual(d["metadata"], {"source": "test"})

    def test_generation_config(self) -> None:
        """Verify GenerationConfig defaults and Gemini schema transformation."""
        cfg = GenerationConfig(
            temperature=0.4,
            top_p=0.9,
            top_k=20,
            max_output_tokens=1024,
            stop_sequences=["STOP"],
        )
        gemini_dict = cfg.to_gemini_dict()
        self.assertEqual(gemini_dict["temperature"], 0.4)
        self.assertEqual(gemini_dict["topP"], 0.9)
        self.assertEqual(gemini_dict["topK"], 20)
        self.assertEqual(gemini_dict["maxOutputTokens"], 1024)
        self.assertEqual(gemini_dict["stopSequences"], ["STOP"])

    def test_ai_response_model(self) -> None:
        """Verify AIResponse initialization and serialization."""
        resp = AIResponse(
            content="Operational, Sir.",
            model="gemini-2.5-flash",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            duration=0.34,
            finish_reason="STOP",
        )
        d = resp.to_dict()
        self.assertEqual(d["content"], "Operational, Sir.")
        self.assertEqual(d["model"], "gemini-2.5-flash")
        self.assertEqual(d["total_tokens"], 15)
        self.assertEqual(d["finish_reason"], "STOP")


class TestPromptBuilder(unittest.TestCase):
    """Test suite for PromptBuilder and J.A.R.V.I.S persona construction."""

    def setUp(self) -> None:
        self.builder = PromptBuilder()

    def test_default_system_prompt(self) -> None:
        """Verify default system persona is present."""
        self.assertIn("J.A.R.V.I.S", self.builder.system_prompt)
        self.assertIn("bilingual", self.builder.system_prompt.lower())

    def test_set_system_prompt(self) -> None:
        """Verify setting a custom system prompt."""
        self.builder.set_system_prompt("You are a specialized math assistant.")
        self.assertEqual(self.builder.system_prompt, "You are a specialized math assistant.")

        with self.assertRaises(PromptError):
            self.builder.set_system_prompt("")

    def test_format_history_role_mapping_and_alternation(self) -> None:
        """Verify history formatting maps assistant->model and merges consecutive roles."""
        memories = [
            ConversationMemory(content="Hi", role="user"),
            ConversationMemory(content="Hello Sir", role="assistant"),
            ConversationMemory(content="How are you?", role="user"),
            ConversationMemory(content="More user details", role="user"),  # consecutive user
            ConversationMemory(content="System instruction", role="system"),  # system skipped
        ]
        turns = self.builder.format_history(memories)
        self.assertEqual(len(turns), 3)
        self.assertEqual(turns[0]["role"], "user")
        self.assertEqual(turns[0]["parts"][0]["text"], "Hi")
        self.assertEqual(turns[1]["role"], "model")
        self.assertEqual(turns[1]["parts"][0]["text"], "Hello Sir")
        self.assertEqual(turns[2]["role"], "user")
        self.assertEqual(turns[2]["parts"][0]["text"], "How are you?\nMore user details")

    def test_build_payload_structure(self) -> None:
        """Verify build_payload generates compliant Gemini request dictionary."""
        payload = self.builder.build_payload(
            query="Status report",
            extra_context={"cpu": "12%", "battery": "98%"},
        )
        self.assertIn("systemInstruction", payload)
        self.assertIn("Active System Context", payload["systemInstruction"]["parts"][0]["text"])
        self.assertIn("cpu: 12%", payload["systemInstruction"]["parts"][0]["text"])

        contents = payload["contents"]
        self.assertEqual(len(contents), 1)
        self.assertEqual(contents[0]["role"], "user")
        self.assertEqual(contents[0]["parts"][0]["text"], "Status report")

    def test_build_payload_empty_query_raises(self) -> None:
        """Verify empty query raises PromptError."""
        with self.assertRaises(PromptError):
            self.builder.build_payload("")

    def test_history_limit_truncation(self) -> None:
        """Verify PromptBuilder only formats the most recent history_limit entries."""
        builder = PromptBuilder(history_limit=3)
        memories = [
            ConversationMemory(content=f"Message {i}", role="user" if i % 2 == 0 else "assistant")
            for i in range(10)
        ]
        payload = builder.build_payload("Current query", history=memories)
        # Should only include last 3 history turns + current query
        # Turns 7 (assistant -> model), 8 (user), 9 (assistant -> model), current query (user)
        # Total turns in contents: 4
        self.assertEqual(len(payload["contents"]), 4)
        self.assertIn("Message 7", payload["contents"][0]["parts"][0]["text"])
        self.assertIn("Message 9", payload["contents"][2]["parts"][0]["text"])
        self.assertEqual(payload["contents"][3]["parts"][0]["text"], "Current query")


class TestGeminiProvider(unittest.TestCase):
    """Test suite for GeminiProvider with mocked HTTP transport."""

    def test_missing_api_key_raises_config_error(self) -> None:
        """Verify calling generate without an API key raises AIConfigError."""
        provider = GeminiProvider(api_key="")
        with self.assertRaises(AIConfigError):
            provider.generate({"contents": [{"role": "user", "parts": [{"text": "hi"}]}]})

    def test_generate_sync_success(self) -> None:
        """Verify successful synchronous content generation."""
        mock_data = _mock_gemini_response_payload("Online and ready, Sir.")

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertIn("key=test_key", str(request.url))
            return httpx.Response(200, json=mock_data)

        transport = httpx.MockTransport(handler)
        mock_client = httpx.Client(transport=transport)
        provider = GeminiProvider(api_key="test_key", client=mock_client)

        resp = provider.generate({"contents": [{"role": "user", "parts": [{"text": "Status"}]}]})
        self.assertEqual(resp.content, "Online and ready, Sir.")
        self.assertEqual(resp.total_tokens, 37)
        self.assertEqual(resp.finish_reason, "STOP")
        self.assertGreater(resp.duration, 0.0)

    def test_generate_async_success(self) -> None:
        """Verify successful asynchronous content generation."""
        mock_data = _mock_gemini_response_payload("Async online, Sir.")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=mock_data)

        transport = httpx.MockTransport(handler)
        mock_async_client = httpx.AsyncClient(transport=transport)
        provider = GeminiProvider(api_key="test_key", async_client=mock_async_client)

        async def _run() -> AIResponse:
            return await provider.generate_async(
                {"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]}
            )

        resp = asyncio.run(_run())
        self.assertEqual(resp.content, "Async online, Sir.")

    def test_authentication_error(self) -> None:
        """Verify HTTP 401 raises AIAuthenticationError without retry."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(401, json={"error": {"message": "API key not valid"}})

        transport = httpx.MockTransport(handler)
        mock_client = httpx.Client(transport=transport)
        provider = GeminiProvider(api_key="invalid_key", client=mock_client, max_retries=3)

        with self.assertRaises(AIAuthenticationError):
            provider.generate({"contents": [{"role": "user", "parts": [{"text": "test"}]}]})
        # Auth error should not retry
        self.assertEqual(call_count, 1)

    def test_rate_limit_retry_and_eventual_error(self) -> None:
        """Verify HTTP 429 retries max_retries times and raises AIRateLimitError."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(429, text="Rate limit exceeded")

        transport = httpx.MockTransport(handler)
        mock_client = httpx.Client(transport=transport)
        provider = GeminiProvider(
            api_key="test_key",
            client=mock_client,
            max_retries=2,
            retry_delay=0.01,
        )

        with self.assertRaises(AIRateLimitError):
            provider.generate({"contents": [{"role": "user", "parts": [{"text": "test"}]}]})
        self.assertEqual(call_count, 2)

    def test_provider_model_from_settings(self) -> None:
        """Verify GeminiProvider resolves model, timeout, and retries from settings when omitted."""
        provider = GeminiProvider(api_key="test_key")
        self.assertEqual(provider.model, settings.ai.model)
        self.assertEqual(provider._timeout, settings.ai.timeout)
        self.assertEqual(provider._max_retries, settings.ai.max_retries)

    def test_provider_stream_generate_raises_not_implemented(self) -> None:
        """Verify streaming methods raise NotImplementedError."""
        provider = GeminiProvider(api_key="test_key")
        with self.assertRaises(NotImplementedError):
            provider.stream_generate({"contents": []})

        async def _test_async() -> None:
            await provider.stream_generate_async({"contents": []})

        with self.assertRaises(NotImplementedError):
            asyncio.run(_test_async())


class TestAIManager(unittest.TestCase):
    """Test suite for AIManager integration across Memory, Events, and Container."""

    def setUp(self) -> None:
        """Set up test container, bus, memory manager, and mock provider."""
        self.container = ServiceContainer()
        self.event_bus = EventBus()
        self.memory = MemoryManager(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            auto_register_in_container=False,
        )

        # Mock Gemini provider
        mock_data = _mock_gemini_response_payload("All systems nominal, Sir.")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=mock_data)

        transport = httpx.MockTransport(handler)
        mock_client = httpx.Client(transport=transport)
        mock_async_client = httpx.AsyncClient(transport=transport)
        self.provider = GeminiProvider(
            api_key="test_key",
            client=mock_client,
            async_client=mock_async_client,
        )

        self.manager = AIManager(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            memory_manager_instance=self.memory,
            provider_instance=self.provider,
            auto_register_in_container=True,
        )

    def test_service_container_registration(self) -> None:
        """Verify AIManager self-registers into ServiceContainer."""
        self.assertTrue(self.container.exists("ai_manager"))
        self.assertTrue(self.container.exists("ai"))
        self.assertIs(self.container.resolve("ai_manager"), self.manager)
        self.assertIs(self.container.resolve("ai"), self.manager)

    def test_generate_sync_with_memory_and_event(self) -> None:
        """Verify synchronous generate updates memory and publishes ai.generated event."""
        events: list[Event] = []
        self.event_bus.subscribe("ai.generated", lambda e: events.append(e))

        resp = self.manager.generate("Jarvis, check status", conversation_id="test_conv")
        self.assertEqual(resp.content, "All systems nominal, Sir.")

        # Check memory updated: 1 user query, 1 assistant response
        recent = self.memory.get_recent(conversation_id="test_conv")
        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[0].role, "user")
        self.assertEqual(recent[0].content, "Jarvis, check status")
        self.assertEqual(recent[1].role, "assistant")
        self.assertEqual(recent[1].content, "All systems nominal, Sir.")

        # Check event bus published ai.generated
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].name, "ai.generated")
        self.assertEqual(events[0].payload["query"], "Jarvis, check status")
        self.assertEqual(events[0].payload["response"], "All systems nominal, Sir.")
        self.assertEqual(events[0].payload["conversation_id"], "test_conv")

    def test_generate_async_with_memory_and_event(self) -> None:
        """Verify asynchronous generate updates memory and publishes ai.generated event."""
        events: list[Event] = []
        self.event_bus.subscribe("ai.generated", lambda e: events.append(e))

        async def _run() -> AIResponse:
            return await self.manager.generate_async("System diagnostics", conversation_id="async_conv")

        resp = asyncio.run(_run())
        self.assertEqual(resp.content, "All systems nominal, Sir.")

        recent = self.memory.get_recent(conversation_id="async_conv")
        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[0].content, "System diagnostics")
        self.assertEqual(recent[1].content, "All systems nominal, Sir.")

        self.assertEqual(len(events), 1)

    def test_chat_convenience_methods(self) -> None:
        """Verify chat() and chat_async() return string directly."""
        text_sync = self.manager.chat("Hello sync")
        self.assertEqual(text_sync, "All systems nominal, Sir.")

        async def _chat_async() -> str:
            return await self.manager.chat_async("Hello async")

        text_async = asyncio.run(_chat_async())
        self.assertEqual(text_async, "All systems nominal, Sir.")

    def test_generation_failure_publishes_event_and_raises(self) -> None:
        """Verify provider failure publishes ai.failed event and raises AIError."""
        failed_events: list[Event] = []
        self.event_bus.subscribe("ai.failed", lambda e: failed_events.append(e))

        def failing_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal Server Error")

        failing_client = httpx.Client(transport=httpx.MockTransport(failing_handler))
        failing_provider = GeminiProvider(
            api_key="test_key",
            client=failing_client,
            max_retries=1,
        )

        manager = AIManager(
            container_instance=self.container,
            event_bus_instance=self.event_bus,
            memory_manager_instance=self.memory,
            provider_instance=failing_provider,
            auto_register_in_container=False,
        )

        with self.assertRaises(AIError):
            manager.generate("Will fail")

        self.assertEqual(len(failed_events), 1)
        self.assertEqual(failed_events[0].name, "ai.failed")
        self.assertEqual(failed_events[0].payload["query"], "Will fail")

    def test_thread_safety_concurrent_generation(self) -> None:
        """Verify thread-safety during concurrent generation requests."""
        def _worker(idx: int) -> str:
            return self.manager.chat(f"Concurrent task {idx}", conversation_id=f"thread_{idx % 4}")

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_worker, i) for i in range(20)]
            results = [f.result() for f in as_completed(futures)]

        self.assertEqual(len(results), 20)
        for res in results:
            self.assertEqual(res, "All systems nominal, Sir.")

    def test_lifecycle_events_started_and_completed(self) -> None:
        """Verify ai.started and ai.completed events are emitted alongside ai.generated."""
        started_events: list[Event] = []
        completed_events: list[Event] = []
        generated_events: list[Event] = []

        self.event_bus.subscribe("ai.started", lambda e: started_events.append(e))
        self.event_bus.subscribe("ai.completed", lambda e: completed_events.append(e))
        self.event_bus.subscribe("ai.generated", lambda e: generated_events.append(e))

        resp = self.manager.generate("Lifecycle test", conversation_id="life_conv")
        self.assertEqual(resp.content, "All systems nominal, Sir.")

        self.assertEqual(len(started_events), 1)
        self.assertEqual(started_events[0].name, "ai.started")
        self.assertEqual(started_events[0].payload["query"], "Lifecycle test")
        self.assertEqual(started_events[0].payload["conversation_id"], "life_conv")

        self.assertEqual(len(completed_events), 1)
        self.assertEqual(completed_events[0].name, "ai.completed")
        self.assertEqual(completed_events[0].payload["response"], "All systems nominal, Sir.")

        self.assertEqual(len(generated_events), 1)
        self.assertEqual(generated_events[0].name, "ai.generated")

    def test_manager_stream_generate_raises_not_implemented(self) -> None:
        """Verify AIManager stream_generate and stream_generate_async raise NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.manager.stream_generate("Stream query")

        async def _test_async() -> None:
            await self.manager.stream_generate_async("Stream query")

        with self.assertRaises(NotImplementedError):
            asyncio.run(_test_async())


class TestAIConfigExtensions(unittest.TestCase):
    """Test suite for AI configuration extensions on Settings."""

    def test_settings_ai_extensions(self) -> None:
        """Verify settings.ai.model, settings.model, timeout, max_retries, and history_limit."""
        self.assertEqual(settings.ai.model, "gemini-2.5-flash")
        self.assertEqual(settings.model, "gemini-2.5-flash")
        self.assertEqual(settings.ai.timeout, 30.0)
        self.assertEqual(settings.ai.max_retries, 3)
        self.assertEqual(settings.ai.history_limit, 20)


if __name__ == "__main__":
    unittest.main()
