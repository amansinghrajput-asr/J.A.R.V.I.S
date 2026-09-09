"""AI Manager Subsystem for J.A.R.V.I.S.

Coordinates LLM generation, prompt construction, memory context retrieval,
service container dependency injection, and event lifecycle dispatching.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import logging
import threading
import time
from typing import Any, Final, Optional
from app.ai.intent_router import (
    IntentRouter,
    intent_router,
    IntentType,
)
from app.ai.models import (
    AIError,
    AIResponse,
    GenerationConfig,
)
from app.ai.planner import (
    ExecutionMemory,
    ExecutionMetrics,
    ExecutionResult,
    Executor,
    MemorySummaryBuilder,
    Plan,
    Planner,
    Task,
    executor,
    planner,
)
from app.ai.prompt import PromptBuilder
from app.ai.provider_router import ProviderRouter, provider_router
from app.core.config import Settings, settings
from app.core.container import ServiceContainer, container
from app.core.event_bus import EventBus, event_bus
from app.core.logger import get_logger
from app.memory.manager import MemoryManager, memory_manager


class AIManager:
    """Central coordinator for cognition, reasoning, and conversational intelligence.

    Integrates AI model providers with the short-term MemoryManager, system persona
    PromptBuilder, central EventBus, ServiceContainer, Task Planner, and Task Executor.

    Architecture & Recovery Orchestration:
    --------------------------------------
    1. Orchestration Layer Only:
       AIManager acts purely as an orchestration engine. Task decomposition, schema
       validation, and recovery planning are strictly delegated to `Planner`. Directed
       acyclic graph (DAG) scheduling, concurrency, and task execution are strictly
       delegated to `Executor`. AIManager coordinates the handoffs between them.

    2. Recovery Lifecycle:
       - Initial Execution: A query is planned by `Planner`. If a multi-step plan (>1 task)
         is created, it is dispatched to `Executor`.
       - Evaluation: `_should_replan()` evaluates the resulting `ExecutionResult`. If the
         execution failed, uncompleted tasks remain, and remaining replan attempts exist,
         recovery replanning is triggered.
       - Recovery Planning: `Planner.replan()` or `Planner.replan_async()` is invoked with
         the user query, execution outcome (completed/failed/skipped tasks), and original tasks.
         The Planner generates tasks *only* for the uncompleted work.
       - Recovery Execution: If a valid non-empty recovery plan is produced, `Executor.execute_plan()`
         or `Executor.execute_plan_async()` runs the recovery plan.
       - Immutable Result Merging: `ExecutionResult.merge()` produces a new merged result combining
         all completed tasks, execution order, and accumulated durations, while adopting the
         recovery's outcome as the authoritative final state.
       - Termination Policy: Replanning halts immediately if execution succeeds, the recovery
         plan is empty/invalid, `max_replans` is reached, the recovery plan is identical to a
         previous attempt, or recovery execution yields no forward progress.
       - Response Formatting: The final merged `ExecutionResult` is formatted uniformly via
         `_format_execution_result_response()` and returned with comprehensive metadata.
    """

    def __init__(
        self,
        config: Optional[Settings] = None,
        logger: Optional[logging.Logger] = None,
        container_instance: Optional[ServiceContainer] = None,
        event_bus_instance: Optional[EventBus] = None,
        memory_manager_instance: Optional[MemoryManager] = None,
        provider_instance: Optional[Any] = None,
        prompt_builder_instance: Optional[PromptBuilder] = None,
        provider_router_instance: Optional[ProviderRouter] = None,
        intent_router_instance: Optional[IntentRouter] = None,
        planner_instance: Optional[Planner] = None,
        executor_instance: Optional[Executor] = None,
        web_search_skill_instance: Optional[Any] = None,
        system_skill_instance: Optional[Any] = None,
        file_skill_instance: Optional[Any] = None,
        tool_manager_instance: Optional[Any] = None,
        auto_replan: bool = True,
        max_replans: int = 1,
        *,
        auto_register_in_container: bool = True,
    ) -> None:
        """Initialize the AIManager.

        Args:
            config: Optional Settings instance.
            logger: Optional Logger instance.
            container_instance: Optional ServiceContainer instance.
            event_bus_instance: Optional EventBus instance.
            memory_manager_instance: Optional MemoryManager instance.
            provider_instance: Optional provider instance.
            prompt_builder_instance: Optional PromptBuilder instance.
            provider_router_instance: Optional ProviderRouter instance.
            intent_router_instance: Optional IntentRouter instance.
            planner_instance: Optional Planner instance.
            web_search_skill_instance: Optional WebSearchSkill instance.
            system_skill_instance: Optional SystemSkill instance.
            file_skill_instance: Optional FileSkill instance.
            tool_manager_instance: Optional ToolManager instance.
            auto_replan: If True, automatically attempts recovery planning after task failures.
            max_replans: Maximum number of replanning attempts allowed per query.
            auto_register_in_container: If True, self-registers into ServiceContainer.
        """
        self._lock = threading.RLock()
        self._last_intent: Optional[Any] = None
        self._last_plan: Optional[Plan] = None
        self._auto_replan = auto_replan
        self._max_replans = max(0, max_replans)

        # 1. Dependency resolution: Service Container
        self._container = container_instance if container_instance is not None else container

        # 2. Dependency resolution: Logger
        self._logger = logger if logger is not None else get_logger("AI_MANAGER")

        # 3. Dependency resolution: Configuration
        if config is not None:
            self._config = config
        elif self._container.exists("settings"):
            self._config = self._container.resolve("settings")
        elif self._container.exists("config"):
            self._config = self._container.resolve("config")
        else:
            self._config = settings

        # 4. Dependency resolution: Event Bus
        if event_bus_instance is not None:
            self._event_bus = event_bus_instance
        elif self._container.exists("event_bus"):
            self._event_bus = self._container.resolve("event_bus")
        else:
            self._event_bus = event_bus

        # 5. Dependency resolution: Memory Manager
        if memory_manager_instance is not None:
            self._memory_manager: Optional[MemoryManager] = memory_manager_instance
        elif self._container.exists("memory_manager"):
            self._memory_manager = self._container.resolve("memory_manager")
        elif self._container.exists("memory"):
            self._memory_manager = self._container.resolve("memory")
        else:
            self._memory_manager = memory_manager

        # 6. Dependency resolution: Prompt Builder
        self._prompt_builder = (
            prompt_builder_instance if prompt_builder_instance is not None else PromptBuilder()
        )

        # 7. Dependency resolution: Provider Router
        if provider_router_instance is not None:
            self._provider_router = provider_router_instance
        elif self._container.exists("provider_router"):
            self._provider_router = self._container.resolve("provider_router")
        elif self._container.exists("ai_provider_router"):
            self._provider_router = self._container.resolve("ai_provider_router")
        else:
            self._provider_router = provider_router

        # 8. Dependency resolution: Intent Router
        if intent_router_instance is not None:
            self._intent_router = intent_router_instance
        elif self._container.exists("intent_router"):
            self._intent_router = self._container.resolve("intent_router")
        elif self._container.exists("ai_intent_router"):
            self._intent_router = self._container.resolve("ai_intent_router")
        else:
            self._intent_router = intent_router

        # 9. Dependency resolution: Planner
        if planner_instance is not None:
            self._planner = planner_instance
        elif self._container.exists("planner"):
            self._planner = self._container.resolve("planner")
        elif self._container.exists("ai_planner"):
            self._planner = self._container.resolve("ai_planner")
        else:
            self._planner = planner

        # 10. Dependency resolution: Executor
        if executor_instance is not None:
            self._executor = executor_instance
        elif self._container.exists("executor"):
            self._executor = self._container.resolve("executor")
        elif self._container.exists("plan_executor"):
            self._executor = self._container.resolve("plan_executor")
        else:
            self._executor = executor

        # 11. Dependency resolution: AI Provider
        if provider_instance is not None:
            self._provider = provider_instance
        else:
            self._provider = self._provider_router.resolve()

        # 11. Dependency resolution: Optional Skills & Tool Manager
        self._injected_web_search_skill = web_search_skill_instance
        self._injected_system_skill = system_skill_instance
        self._injected_file_skill = file_skill_instance
        self._injected_tool_manager = tool_manager_instance

        # 12. Self-registration in Service Container
        if auto_register_in_container:
            try:
                self._container.register_singleton("ai_manager", self, allow_override=True)
                self._container.register_singleton("ai", self, allow_override=True)
                self._logger.debug("Registered 'ai_manager' and 'ai' into Service Container.")
            except Exception as exc:
                self._logger.warning(f"Could not register AIManager into container: {exc}")

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------

    @property
    def config(self) -> Settings:
        """Return the active settings configuration."""
        return self._config

    @property
    def logger(self) -> logging.Logger:
        """Return the active logger."""
        return self._logger

    @property
    def container(self) -> ServiceContainer:
        """Return the service container."""
        return self._container

    @property
    def event_bus(self) -> EventBus:
        """Return the event bus."""
        return self._event_bus

    @property
    def memory_manager(self) -> Optional[MemoryManager]:
        """Return the active memory manager."""
        return self._memory_manager

    @property
    def provider_router(self) -> ProviderRouter:
        """Return the active provider router."""
        return self._provider_router

    @property
    def intent_router(self) -> IntentRouter:
        """Return the active intent router."""
        return self._intent_router

    @property
    def planner(self) -> Planner:
        """Return the active planner instance."""
        return self._planner

    @property
    def executor(self) -> Executor:
        """Return the active executor instance."""
        return self._executor

    @property
    def auto_replan(self) -> bool:
        """Return whether automatic replanning after task failure is enabled."""
        return self._auto_replan

    @property
    def max_replans(self) -> int:
        """Return the maximum number of replanning attempts permitted."""
        return self._max_replans

    @property
    def last_intent(self) -> Optional[Any]:
        """Return the last classified intent result."""
        return self._last_intent

    @property
    def last_plan(self) -> Optional[Plan]:
        """Return the last created plan."""
        return self._last_plan

    @property
    def provider(self):
        """Return the underlying AI provider."""
        return self._provider

    @property
    def prompt_builder(self) -> PromptBuilder:
        """Return the active prompt builder."""
        return self._prompt_builder

    # --------------------------------------------------------------------------
    # Optional Skill / Tool Resolution Helpers & Properties
    # --------------------------------------------------------------------------

    def _resolve_web_search_skill(self) -> Optional[Any]:
        """Resolve WebSearchSkill from injection, container, or skill manager."""
        if self._injected_web_search_skill is not None:
            return self._injected_web_search_skill
        for name in ("web_search_skill", "web_search", "search_skill"):
            if self._container.exists(name):
                return self._container.resolve(name)
        if self._container.exists("skill_manager"):
            try:
                sm = self._container.resolve("skill_manager")
                return sm.get_skill("web_search") or sm.get_skill("search")
            except Exception:
                pass
        return None

    def _resolve_system_skill(self) -> Optional[Any]:
        """Resolve SystemSkill from injection, container, or skill manager."""
        if self._injected_system_skill is not None:
            return self._injected_system_skill
        for name in ("system_skill", "system"):
            if self._container.exists(name):
                return self._container.resolve(name)
        if self._container.exists("skill_manager"):
            try:
                sm = self._container.resolve("skill_manager")
                return sm.get_skill("system")
            except Exception:
                pass
        return None

    def _resolve_file_skill(self) -> Optional[Any]:
        """Resolve FileSkill from injection, container, or skill manager."""
        if self._injected_file_skill is not None:
            return self._injected_file_skill
        for name in ("file_skill", "file"):
            if self._container.exists(name):
                return self._container.resolve(name)
        if self._container.exists("skill_manager"):
            try:
                sm = self._container.resolve("skill_manager")
                return sm.get_skill("file")
            except Exception:
                pass
        return None

    def _resolve_tool_manager(self) -> Optional[Any]:
        """Resolve ToolManager from injection, container, or tools registry."""
        if self._injected_tool_manager is not None:
            return self._injected_tool_manager
        for name in ("tool_manager", "tools", "tool"):
            if self._container.exists(name):
                return self._container.resolve(name)
        return None

    @property
    def web_search_skill(self) -> Optional[Any]:
        """Return the active web search skill if registered."""
        return self._resolve_web_search_skill()

    @property
    def system_skill(self) -> Optional[Any]:
        """Return the active system skill if registered."""
        return self._resolve_system_skill()

    @property
    def file_skill(self) -> Optional[Any]:
        """Return the active file skill if registered."""
        return self._resolve_file_skill()

    @property
    def tool_manager(self) -> Optional[Any]:
        """Return the active tool manager if registered."""
        return self._resolve_tool_manager()

    # --------------------------------------------------------------------------
    # Skill Execution & Wrapping Helpers
    # --------------------------------------------------------------------------

    @staticmethod
    def _format_task_description(task: Task) -> str:
        """Format an individual Task into a human-readable action description."""
        action = task.action
        target = task.target or ""

        if action == "open_app":
            formatted_target = target.capitalize() if target.islower() else target
            return f"Open {formatted_target}".strip()
        elif action == "web_search":
            return f"Search {target}".strip()
        elif action == "calculate":
            return f"Calculate {target}".strip()
        elif action == "summarize_file":
            return f"Summarize {target}".strip()
        elif action == "save_memory":
            return f"Remember {target}".strip()
        elif action == "clear_memory":
            return "Clear Memory"
        else:
            action_title = action.replace("_", " ").title()
            return f"{action_title} {target}".strip()

    def _format_plan_response(self, plan: Plan) -> str:
        """Format a multi-task Plan into structured response text."""
        lines = ["Plan:"]
        for idx, task in enumerate(plan.tasks, 1):
            desc = self._format_task_description(task)
            lines.append(f"{idx}. {desc}")
        lines.append("")
        lines.append("Execution engine will process this plan later.")
        return "\n".join(lines)

    def _format_execution_result_response(self, result: ExecutionResult) -> str:
        """Format an ExecutionResult into a readable structured text response."""
        sections: list[str] = []

        if result.completed_tasks:
            completed_lines = ["Completed:"]
            for task in result.completed_tasks:
                desc = self._format_task_description(task)
                completed_lines.append(f"✓ {desc}")
            sections.append("\n".join(completed_lines))

        if result.failed_tasks:
            failed_lines = ["Failed:"]
            for task in result.failed_tasks:
                desc = self._format_task_description(task)
                failed_lines.append(f"✗ {desc}")
            sections.append("\n".join(failed_lines))

        completed_count = len(result.completed_tasks)
        failed_count = len(result.failed_tasks)
        overall_lines = [
            "Overall:",
            f"{completed_count} completed",
            f"{failed_count} failed",
        ]
        sections.append("\n".join(overall_lines))

        return "\n\n".join(sections)

    def _should_replan(
        self,
        execution_result: ExecutionResult,
        attempt: int,
    ) -> bool:
        """Determine whether another replanning iteration should be attempted.

        Conditions for replanning:
        1. `auto_replan` configuration is enabled.
        2. Previous execution was not successful (`not execution_result.success`).
        3. Work remains to be completed (`failed_tasks` or `skipped_tasks` are non-empty).
        4. Current attempt count is strictly below `max_replans`.

        Args:
            execution_result: The latest execution outcome to inspect.
            attempt: Current 0-indexed attempt count.

        Returns:
            True if replanning should proceed; False otherwise.
        """
        if not self._auto_replan:
            return False
        if execution_result.success:
            return False
        if not (execution_result.failed_tasks or execution_result.skipped_tasks):
            return False
        if attempt >= self._max_replans:
            return False
        return True

    def _build_replan_response_metadata(
        self,
        original_plan: Plan,
        recovery_plan: Optional[Plan],
        execution_result: ExecutionResult,
        execution_results: list[ExecutionResult],
        replanned: bool,
        replan_attempts: int,
        execution_duration: float,
        memory: Optional[ExecutionMemory] = None,
    ) -> dict[str, Any]:
        """Construct unified metadata dictionary for plan execution responses.

        Field Specifications:
        - `plan`: (dict) Serialized original Plan dictionary for backwards compatibility.
        - `planning_strategy`: (str) String identifier of original planning strategy (e.g. 'rule_based', 'llm', 'hybrid').
        - `replanned`: (bool) True if at least one recovery replanning attempt was executed.
        - `replan_attempts`: (int) Total number of replanning attempts initiated.
        - `original_plan`: (dict) Serialized original Plan dictionary before recovery.
        - `recovery_plan`: (dict | None) Serialized recovery Plan dictionary from last attempt, or None if none created.
        - `execution_results`: (list[dict]) Serialized list of all ExecutionResult stages in sequential order.
        - `execution_result`: (dict) Final authoritative merged ExecutionResult dictionary.
        - `completed_count`: (int) Total count of completed tasks across execution history.
        - `failed_count`: (int) Total count of failed tasks in final merged result.
        - `execution_duration`: (float) Total duration in seconds encompassing initial and recovery execution.
        - `execution_summary`: (dict | None) Compact metadata summary derived from canonical ExecutionMemory.
        - `execution_metrics`: (dict | None) Observational metrics tracking waves, recoveries, and durations.

        Args:
            original_plan: The plan initially generated for the user query.
            recovery_plan: The recovery plan produced during the latest replanning attempt, or None.
            execution_result: The final merged ExecutionResult outcome.
            execution_results: Chronological list of all ExecutionResult instances produced.
            replanned: Whether any recovery replanning was executed.
            replan_attempts: Number of replanning attempts performed.
            execution_duration: Total accumulated wall-clock execution duration in seconds.
            memory: Optional canonical ExecutionMemory instance across all waves.

        Returns:
            Structured dictionary of metadata attached to the final AIResponse.
        """
        strategy_str = (
            original_plan.strategy.value
            if hasattr(original_plan.strategy, "value")
            else str(original_plan.strategy)
        )
        meta: dict[str, Any] = {
            "plan": original_plan.to_dict(),
            "planning_strategy": strategy_str,
            "replanned": replanned,
            "replan_attempts": replan_attempts,
            "original_plan": original_plan.to_dict(),
            "recovery_plan": recovery_plan.to_dict() if recovery_plan is not None else None,
            "execution_results": [res.to_dict() for res in execution_results],
            "execution_result": execution_result.to_dict(),
            "completed_count": len(execution_result.completed_tasks),
            "failed_count": len(execution_result.failed_tasks),
            "execution_duration": execution_duration,
        }
        if memory is not None:
            meta["execution_summary"] = MemorySummaryBuilder.build_metadata_summary(memory)
            meta["execution_metrics"] = memory.metrics.to_dict()
        return meta

    def _invoke_skill_sync(self, target: Any, query: str, method_names: tuple[str, ...]) -> Any:
        """Invoke a matching method on the target synchronously."""
        func = None
        for m in method_names:
            if hasattr(target, m):
                func = getattr(target, m)
                break
        if func is None and callable(target):
            func = target
        if func is None:
            raise AttributeError(f"No suitable method among {method_names} found on {target}")

        res = func(query)
        if inspect.isawaitable(res):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(asyncio.run, res).result()
            return asyncio.run(res)
        return res

    async def _invoke_skill_async(self, target: Any, query: str, method_names: tuple[str, ...]) -> Any:
        """Invoke a matching method on the target asynchronously."""
        func = None
        for m in method_names:
            if hasattr(target, m):
                func = getattr(target, m)
                break
        if func is None and callable(target):
            func = target
        if func is None:
            raise AttributeError(f"No suitable method among {method_names} found on {target}")

        if asyncio.iscoroutinefunction(func):
            return await func(query)
        res = func(query)
        if inspect.isawaitable(res):
            return await res
        return res

    def _wrap_skill_response(
        self,
        result: Any,
        *,
        query: str,
        model: str,
        conversation_id: str = "default",
        record_memory: bool = True,
        duration: float = 0.0,
    ) -> AIResponse:
        """Wrap skill result in an AIResponse, updating memory and emitting lifecycle events."""
        if isinstance(result, AIResponse):
            resp = result
        else:
            resp = AIResponse(
                content=str(result),
                model=model,
                duration=duration,
            )

        if record_memory and self._memory_manager is not None:
            with self._lock:
                self._memory_manager.add(
                    content=query,
                    role="user",
                    conversation_id=conversation_id,
                    source="user",
                )
                self._memory_manager.add(
                    content=resp.content,
                    role="assistant",
                    conversation_id=conversation_id,
                    source="ai",
                    metadata={
                        "model": resp.model,
                        "tokens": resp.total_tokens,
                        "duration": resp.duration,
                    },
                )

        self._event_bus.publish(
            "ai.completed",
            payload={
                "query": query,
                "response": resp.content,
                "model": resp.model,
                "duration": resp.duration,
                "tokens": resp.total_tokens,
                "conversation_id": conversation_id,
            },
            source="ai_manager",
        )
        self._event_bus.publish(
            "ai.generated",
            payload={
                "query": query,
                "response": resp.content,
                "model": resp.model,
                "duration": resp.duration,
                "tokens": resp.total_tokens,
                "conversation_id": conversation_id,
            },
            source="ai_manager",
        )
        return resp

    async def _wrap_skill_response_async(
        self,
        result: Any,
        *,
        query: str,
        model: str,
        conversation_id: str = "default",
        record_memory: bool = True,
        duration: float = 0.0,
    ) -> AIResponse:
        """Asynchronously wrap skill result in an AIResponse, updating memory and emitting lifecycle events."""
        if isinstance(result, AIResponse):
            resp = result
        else:
            resp = AIResponse(
                content=str(result),
                model=model,
                duration=duration,
            )

        if record_memory and self._memory_manager is not None:
            with self._lock:
                self._memory_manager.add(
                    content=query,
                    role="user",
                    conversation_id=conversation_id,
                    source="user",
                )
                self._memory_manager.add(
                    content=resp.content,
                    role="assistant",
                    conversation_id=conversation_id,
                    source="ai",
                    metadata={
                        "model": resp.model,
                        "tokens": resp.total_tokens,
                        "duration": resp.duration,
                    },
                )

        await self._event_bus.publish_async(
            "ai.completed",
            payload={
                "query": query,
                "response": resp.content,
                "model": resp.model,
                "duration": resp.duration,
                "tokens": resp.total_tokens,
                "conversation_id": conversation_id,
            },
            source="ai_manager",
        )
        await self._event_bus.publish_async(
            "ai.generated",
            payload={
                "query": query,
                "response": resp.content,
                "model": resp.model,
                "duration": resp.duration,
                "tokens": resp.total_tokens,
                "conversation_id": conversation_id,
            },
            source="ai_manager",
        )
        return resp

    def _dispatch_sync(
        self,
        intent: IntentType,
        query: str,
        conversation_id: str,
        record_memory: bool,
    ) -> Optional[AIResponse]:
        """Dispatch query based on IntentType synchronously. Returns AIResponse or None for LLM fallback."""
        start_time = time.perf_counter()

        # 1. SEARCH: If WebSearchSkill exists: return web_search_skill.search(query) else fallback
        if intent == IntentType.SEARCH:
            skill = self._resolve_web_search_skill()
            if skill is not None:
                try:
                    res = self._invoke_skill_sync(
                        skill, query, ("search", "execute", "handle")
                    )
                    duration = time.perf_counter() - start_time
                    return self._wrap_skill_response(
                        res,
                        query=query,
                        model="web_search_skill",
                        conversation_id=conversation_id,
                        record_memory=record_memory,
                        duration=duration,
                    )
                except Exception as exc:
                    self._logger.warning("WebSearchSkill dispatch failed; falling back to LLM: %s", exc)

        # 2. SYSTEM: If SystemSkill exists: return system_skill.execute(query) else fallback
        elif intent == IntentType.SYSTEM:
            skill = self._resolve_system_skill()
            if skill is not None:
                try:
                    res = self._invoke_skill_sync(
                        skill, query, ("execute", "handle")
                    )
                    duration = time.perf_counter() - start_time
                    return self._wrap_skill_response(
                        res,
                        query=query,
                        model="system_skill",
                        conversation_id=conversation_id,
                        record_memory=record_memory,
                        duration=duration,
                    )
                except Exception as exc:
                    self._logger.warning("SystemSkill dispatch failed; falling back to LLM: %s", exc)

        # 3. FILE: If FileSkill exists: return file_skill.handle(query) else fallback
        elif intent == IntentType.FILE:
            skill = self._resolve_file_skill()
            if skill is not None:
                try:
                    res = self._invoke_skill_sync(
                        skill, query, ("handle", "execute")
                    )
                    duration = time.perf_counter() - start_time
                    return self._wrap_skill_response(
                        res,
                        query=query,
                        model="file_skill",
                        conversation_id=conversation_id,
                        record_memory=record_memory,
                        duration=duration,
                    )
                except Exception as exc:
                    self._logger.warning("FileSkill dispatch failed; falling back to LLM: %s", exc)

        # 4. MEMORY: If MemoryManager already supports the command, execute memory operation, otherwise fallback
        elif intent == IntentType.MEMORY:
            if self._memory_manager is not None:
                try:
                    handled = False
                    res = None
                    if hasattr(self._memory_manager, "execute"):
                        res = self._invoke_skill_sync(self._memory_manager, query, ("execute",))
                        handled = True
                    elif hasattr(self._memory_manager, "handle"):
                        res = self._invoke_skill_sync(self._memory_manager, query, ("handle",))
                        handled = True
                    else:
                        clean_q = query.strip().lower()
                        if clean_q in ("forget that", "forget this", "forget everything", "clear memory"):
                            cleared = self._memory_manager.clear()
                            res = f"Memory cleared ({cleared} entries removed)."
                            handled = True
                    if handled and res is not None:
                        duration = time.perf_counter() - start_time
                        return self._wrap_skill_response(
                            res,
                            query=query,
                            model="memory_manager",
                            conversation_id=conversation_id,
                            record_memory=False,
                            duration=duration,
                        )
                except Exception as exc:
                    self._logger.warning("MemoryManager dispatch failed; falling back to LLM: %s", exc)

        # 5. TOOL: If ToolManager exists: execute tool else fallback
        elif intent == IntentType.TOOL:
            mgr = self._resolve_tool_manager()
            if mgr is not None:
                try:
                    res = self._invoke_skill_sync(
                        mgr, query, ("execute", "run")
                    )
                    duration = time.perf_counter() - start_time
                    return self._wrap_skill_response(
                        res,
                        query=query,
                        model="tool_manager",
                        conversation_id=conversation_id,
                        record_memory=record_memory,
                        duration=duration,
                    )
                except Exception as exc:
                    self._logger.warning("ToolManager dispatch failed; falling back to LLM: %s", exc)

        return None

    async def _dispatch_async(
        self,
        intent: IntentType,
        query: str,
        conversation_id: str,
        record_memory: bool,
    ) -> Optional[AIResponse]:
        """Dispatch query based on IntentType asynchronously. Returns AIResponse or None for LLM fallback."""
        start_time = time.perf_counter()

        # 1. SEARCH: If WebSearchSkill exists: return web_search_skill.search(query) else fallback
        if intent == IntentType.SEARCH:
            skill = self._resolve_web_search_skill()
            if skill is not None:
                try:
                    res = await self._invoke_skill_async(
                        skill, query, ("search", "search_async", "execute", "execute_async")
                    )
                    duration = time.perf_counter() - start_time
                    return await self._wrap_skill_response_async(
                        res,
                        query=query,
                        model="web_search_skill",
                        conversation_id=conversation_id,
                        record_memory=record_memory,
                        duration=duration,
                    )
                except Exception as exc:
                    self._logger.warning("WebSearchSkill async dispatch failed; falling back to LLM: %s", exc)

        # 2. SYSTEM: If SystemSkill exists: return system_skill.execute(query) else fallback
        elif intent == IntentType.SYSTEM:
            skill = self._resolve_system_skill()
            if skill is not None:
                try:
                    res = await self._invoke_skill_async(
                        skill, query, ("execute", "execute_async", "handle", "handle_async")
                    )
                    duration = time.perf_counter() - start_time
                    return await self._wrap_skill_response_async(
                        res,
                        query=query,
                        model="system_skill",
                        conversation_id=conversation_id,
                        record_memory=record_memory,
                        duration=duration,
                    )
                except Exception as exc:
                    self._logger.warning("SystemSkill async dispatch failed; falling back to LLM: %s", exc)

        # 3. FILE: If FileSkill exists: return file_skill.handle(query) else fallback
        elif intent == IntentType.FILE:
            skill = self._resolve_file_skill()
            if skill is not None:
                try:
                    res = await self._invoke_skill_async(
                        skill, query, ("handle", "handle_async", "execute", "execute_async")
                    )
                    duration = time.perf_counter() - start_time
                    return await self._wrap_skill_response_async(
                        res,
                        query=query,
                        model="file_skill",
                        conversation_id=conversation_id,
                        record_memory=record_memory,
                        duration=duration,
                    )
                except Exception as exc:
                    self._logger.warning("FileSkill async dispatch failed; falling back to LLM: %s", exc)

        # 4. MEMORY: If MemoryManager already supports the command, execute memory operation, otherwise fallback
        elif intent == IntentType.MEMORY:
            if self._memory_manager is not None:
                try:
                    handled = False
                    res = None
                    if hasattr(self._memory_manager, "execute") or hasattr(self._memory_manager, "execute_async"):
                        res = await self._invoke_skill_async(self._memory_manager, query, ("execute", "execute_async"))
                        handled = True
                    elif hasattr(self._memory_manager, "handle") or hasattr(self._memory_manager, "handle_async"):
                        res = await self._invoke_skill_async(self._memory_manager, query, ("handle", "handle_async"))
                        handled = True
                    else:
                        clean_q = query.strip().lower()
                        if clean_q in ("forget that", "forget this", "forget everything", "clear memory"):
                            cleared = self._memory_manager.clear()
                            res = f"Memory cleared ({cleared} entries removed)."
                            handled = True
                    if handled and res is not None:
                        duration = time.perf_counter() - start_time
                        return await self._wrap_skill_response_async(
                            res,
                            query=query,
                            model="memory_manager",
                            conversation_id=conversation_id,
                            record_memory=False,
                            duration=duration,
                        )
                except Exception as exc:
                    self._logger.warning("MemoryManager async dispatch failed; falling back to LLM: %s", exc)

        # 5. TOOL: If ToolManager exists: execute tool else fallback
        elif intent == IntentType.TOOL:
            mgr = self._resolve_tool_manager()
            if mgr is not None:
                try:
                    res = await self._invoke_skill_async(
                        mgr, query, ("execute", "execute_async", "run", "run_async")
                    )
                    duration = time.perf_counter() - start_time
                    return await self._wrap_skill_response_async(
                        res,
                        query=query,
                        model="tool_manager",
                        conversation_id=conversation_id,
                        record_memory=record_memory,
                        duration=duration,
                    )
                except Exception as exc:
                    self._logger.warning("ToolManager async dispatch failed; falling back to LLM: %s", exc)

        return None

    # --------------------------------------------------------------------------
    # Core Operations
    # --------------------------------------------------------------------------

    def generate(
        self,
        query: str,
        *,
        conversation_id: str = "default",
        use_history: bool = True,
        system_prompt: Optional[str] = None,
        extra_context: Optional[dict[str, Any] | str] = None,
        config: Optional[GenerationConfig] = None,
        record_memory: bool = True,
        history_limit: Optional[int] = None,
    ) -> AIResponse:
        """Execute synchronous conversational generation with full system integration.

        Args:
            query: User's command or question.
            conversation_id: Conversation tracking ID.
            use_history: If True, retrieves recent conversation history from MemoryManager.
            system_prompt: Optional custom system prompt override.
            extra_context: Optional contextual dictionary or string.
            config: Optional sampling parameters (temperature, tokens, etc.).
            record_memory: If True, records user query and assistant response in MemoryManager.
            history_limit: Maximum historical turns to provide to model. If None, uses settings.ai.history_limit.

        Returns:
            The structured AIResponse.

        Raises:
            AIError: If generation fails after all provider retries.
        """
        # 1. Plan creation via Planner
        try:
            plan = self._planner.create_plan(query)
        except Exception as exc:
            self._logger.warning("Planner create_plan failed: %s", exc)
            plan = Plan(query=query, tasks=[])

        with self._lock:
            self._last_plan = plan

        if len(plan.tasks) > 1:
            start_time = time.perf_counter()
            self._logger.info(
                "Executing multi-step execution plan with %d tasks for query: '%s'",
                len(plan.tasks),
                query,
            )
            execution_result: Optional[ExecutionResult] = None
            try:
                execution_result = self._executor.execute_plan(plan)
            except Exception as exc:
                self._logger.warning(
                    "Executor failed executing plan for query '%s', falling back to standard pipeline: %s",
                    query,
                    exc,
                )

            if execution_result is not None:
                attempt = 0
                replanned = False
                recovery_plan: Optional[Plan] = None
                all_results: list[ExecutionResult] = [execution_result]
                seen_signatures: set[tuple] = set()

                # Initialize canonical ExecutionMemory from initial execution wave
                memory = ExecutionMemory.from_execution_result(execution_result, wave=1)
                memory.metrics.planning_count = 1

                while self._should_replan(execution_result, attempt):
                    attempt += 1
                    failed_ids = list(execution_result.failed_task_ids)
                    skipped_ids = list(execution_result.skipped_task_ids)
                    self._logger.info(
                        "Recovery triggered for query '%s' (attempt %d/%d). Reason: %d failed task(s) %s, %d skipped task(s) %s",
                        query,
                        attempt,
                        self._max_replans,
                        len(failed_ids),
                        failed_ids,
                        len(skipped_ids),
                        skipped_ids,
                    )
                    recovery_start = time.perf_counter()
                    try:
                        recovery_plan = self._planner.replan(query, memory, plan)
                    except Exception as exc:
                        self._logger.warning("Planner replan failed: %s", exc)
                        recovery_plan = None

                    if recovery_plan is None or recovery_plan.is_empty():
                        self._logger.info("Recovery plan empty or invalid for query '%s', halting replanning", query)
                        break

                    # Edge case: prevent repeated identical recovery plans
                    plan_sig = tuple((t.action, t.target, tuple(sorted(t.dependencies))) for t in recovery_plan.tasks)
                    if plan_sig in seen_signatures:
                        self._logger.info(
                            "Recovery plan is identical to a previous attempt for query '%s', halting replanning to prevent loop",
                            query,
                        )
                        break
                    seen_signatures.add(plan_sig)

                    replanned = True
                    self._logger.info(
                        "Recovery execution started for plan '%s' with %d task(s)",
                        recovery_plan.id,
                        len(recovery_plan.tasks),
                    )
                    try:
                        recovery_result = self._executor.execute_plan(recovery_plan)
                        recovery_duration = time.perf_counter() - recovery_start
                        all_results.append(recovery_result)
                        self._logger.info(
                            "Recovery execution completed: %d completed, %d failed, %d skipped (success=%s)",
                            len(recovery_result.completed_tasks),
                            len(recovery_result.failed_tasks),
                            len(recovery_result.skipped_tasks),
                            recovery_result.success,
                        )

                        # Record recovery wave into execution memory and update metrics
                        memory = memory.record_execution(recovery_result, wave=attempt + 1)
                        memory.metrics.replan_count = attempt
                        memory.metrics.total_recovery_duration += recovery_duration
                        if recovery_result.success:
                            memory.metrics.successful_recoveries += 1
                        else:
                            memory.metrics.failed_recoveries += 1

                        execution_result = execution_result.merge(recovery_result)
                        self._logger.info(
                            "Merged execution result: %d total completed, %d failed, %d skipped (overall success=%s)",
                            len(execution_result.completed_tasks),
                            len(execution_result.failed_tasks),
                            len(execution_result.skipped_tasks),
                            execution_result.success,
                        )

                        # Edge case: halt if recovery produced zero progress
                        if not recovery_result.completed_tasks and not recovery_result.success:
                            self._logger.info(
                                "Recovery execution produced no forward progress (0 completed tasks) for query '%s', halting further replanning",
                                query,
                            )
                            break
                    except Exception as exc:
                        self._logger.warning("Executor failed executing recovery plan: %s", exc)
                        break

                duration = time.perf_counter() - start_time
                execution_result.execution_duration = duration
                response_content = self._format_execution_result_response(execution_result)
                metadata = self._build_replan_response_metadata(
                    original_plan=plan,
                    recovery_plan=recovery_plan,
                    execution_result=execution_result,
                    execution_results=all_results,
                    replanned=replanned,
                    replan_attempts=attempt,
                    execution_duration=duration,
                    memory=memory,
                )
                resp = AIResponse(
                    content=response_content,
                    model="executor",
                    duration=duration,
                    metadata=metadata,
                )
                return self._wrap_skill_response(
                    resp,
                    query=query,
                    model="executor",
                    conversation_id=conversation_id,
                    record_memory=record_memory,
                    duration=duration,
                )

        # 2. Intent classification via IntentRouter
        intent = self._intent_router.classify(query)
        detected_intent = intent.intent
        confidence = intent.confidence
        reason = intent.reason

        with self._lock:
            self._last_intent = intent

        self._logger.info(
            "Detected intent:\n%s\nconfidence=%.2f\nreason=%s",
            detected_intent.value if hasattr(detected_intent, "value") else detected_intent,
            confidence,
            reason,
        )

        # 2. Intent-based dispatching
        dispatched_response = self._dispatch_sync(
            detected_intent,
            query=query,
            conversation_id=conversation_id,
            record_memory=record_memory,
        )
        if dispatched_response is not None:
            if hasattr(self._last_plan, "strategy") and isinstance(dispatched_response.metadata, dict):
                if "planning_strategy" not in dispatched_response.metadata:
                    dispatched_response.metadata["planning_strategy"] = (
                        self._last_plan.strategy.value
                        if hasattr(self._last_plan.strategy, "value")
                        else str(self._last_plan.strategy)
                    )
            return dispatched_response

        resolved_limit = (
            history_limit
            if history_limit is not None
            else getattr(self._config.ai, "history_limit", 20)
        )

        with self._lock:
            history = None
            if use_history and self._memory_manager is not None:
                history = self._memory_manager.get_recent(
                    limit=resolved_limit,
                    conversation_id=conversation_id,
                )

            # Record user query in memory
            if record_memory and self._memory_manager is not None:
                self._memory_manager.add(
                    content=query,
                    role="user",
                    conversation_id=conversation_id,
                    source="user",
                )

            # Build provider payload
            payload = self._prompt_builder.build_payload(
                query=query,
                history=history,
                system_prompt=system_prompt,
                extra_context=extra_context,
            )

        # Emit ai.started event
        self._event_bus.publish(
            "ai.started",
            payload={
                "query": query,
                "model": self._provider.model,
                "conversation_id": conversation_id,
            },
            source="ai_manager",
        )

        try:
            response = self._provider.generate(payload, config=config)

            # Record assistant response in memory
            if record_memory and self._memory_manager is not None:
                self._memory_manager.add(
                    content=response.content,
                    role="assistant",
                    conversation_id=conversation_id,
                    source="ai",
                    metadata={
                        "model": response.model,
                        "tokens": response.total_tokens,
                        "duration": response.duration,
                    },
                )

            # Emit ai.completed event
            self._event_bus.publish(
                "ai.completed",
                payload={
                    "query": query,
                    "response": response.content,
                    "model": response.model,
                    "duration": response.duration,
                    "tokens": response.total_tokens,
                    "conversation_id": conversation_id,
                },
                source="ai_manager",
            )

            # Emit ai.generated event
            self._event_bus.publish(
                "ai.generated",
                payload={
                    "query": query,
                    "response": response.content,
                    "model": response.model,
                    "duration": response.duration,
                    "tokens": response.total_tokens,
                    "conversation_id": conversation_id,
                },
                source="ai_manager",
            )
            if hasattr(self._last_plan, "strategy") and isinstance(response.metadata, dict):
                if "planning_strategy" not in response.metadata:
                    response.metadata["planning_strategy"] = (
                        self._last_plan.strategy.value
                        if hasattr(self._last_plan.strategy, "value")
                        else str(self._last_plan.strategy)
                    )
            return response

        except Exception as exc:
            self._logger.exception(f"AIManager generation failed for query '{query[:50]}...': {exc}")
            self._event_bus.publish(
                "ai.failed",
                payload={
                    "query": query,
                    "error": str(exc),
                    "exception_type": type(exc).__name__,
                    "conversation_id": conversation_id,
                },
                source="ai_manager",
            )
            if isinstance(exc, AIError):
                raise
            raise AIError(f"Generation failed: {exc}") from exc

    async def generate_async(
        self,
        query: str,
        *,
        conversation_id: str = "default",
        use_history: bool = True,
        system_prompt: Optional[str] = None,
        extra_context: Optional[dict[str, Any] | str] = None,
        config: Optional[GenerationConfig] = None,
        record_memory: bool = True,
        history_limit: Optional[int] = None,
    ) -> AIResponse:
        """Execute asynchronous conversational generation with full system integration.

        Args:
            query: User's command or question.
            conversation_id: Conversation tracking ID.
            use_history: If True, retrieves recent conversation history from MemoryManager.
            system_prompt: Optional custom system prompt override.
            extra_context: Optional contextual dictionary or string.
            config: Optional sampling parameters.
            record_memory: If True, records user query and assistant response in MemoryManager.
            history_limit: Maximum historical turns to provide to model. If None, uses settings.ai.history_limit.

        Returns:
            The structured AIResponse.

        Raises:
            AIError: If generation fails after all provider retries.
        """
        # 1. Plan creation via Planner
        try:
            if hasattr(self._planner, "create_plan_async"):
                plan = await self._planner.create_plan_async(query)
            else:
                plan = self._planner.create_plan(query)
        except Exception as exc:
            self._logger.warning("Planner create_plan failed: %s", exc)
            plan = Plan(query=query, tasks=[])

        with self._lock:
            self._last_plan = plan

        if len(plan.tasks) > 1:
            start_time = time.perf_counter()
            self._logger.info(
                "Executing multi-step execution plan with %d tasks asynchronously for query: '%s'",
                len(plan.tasks),
                query,
            )
            execution_result: Optional[ExecutionResult] = None
            try:
                execution_result = await self._executor.execute_plan_async(plan)
            except Exception as exc:
                self._logger.warning(
                    "Executor failed executing plan asynchronously for query '%s', falling back to standard pipeline: %s",
                    query,
                    exc,
                )

            if execution_result is not None:
                attempt = 0
                replanned = False
                recovery_plan: Optional[Plan] = None
                all_results: list[ExecutionResult] = [execution_result]
                seen_signatures: set[tuple] = set()

                # Initialize canonical ExecutionMemory from initial execution wave
                memory = ExecutionMemory.from_execution_result(execution_result, wave=1)
                memory.metrics.planning_count = 1

                while self._should_replan(execution_result, attempt):
                    attempt += 1
                    failed_ids = list(execution_result.failed_task_ids)
                    skipped_ids = list(execution_result.skipped_task_ids)
                    self._logger.info(
                        "Recovery triggered asynchronously for query '%s' (attempt %d/%d). Reason: %d failed task(s) %s, %d skipped task(s) %s",
                        query,
                        attempt,
                        self._max_replans,
                        len(failed_ids),
                        failed_ids,
                        len(skipped_ids),
                        skipped_ids,
                    )
                    recovery_start = time.perf_counter()
                    try:
                        recovery_plan = await self._planner.replan_async(query, memory, plan)
                    except Exception as exc:
                        self._logger.warning("Planner replan_async failed: %s", exc)
                        recovery_plan = None

                    if recovery_plan is None or recovery_plan.is_empty():
                        self._logger.info("Recovery plan empty or invalid asynchronously for query '%s', halting replanning", query)
                        break

                    # Edge case: prevent repeated identical recovery plans
                    plan_sig = tuple((t.action, t.target, tuple(sorted(t.dependencies))) for t in recovery_plan.tasks)
                    if plan_sig in seen_signatures:
                        self._logger.info(
                            "Recovery plan is identical to a previous attempt asynchronously for query '%s', halting replanning to prevent loop",
                            query,
                        )
                        break
                    seen_signatures.add(plan_sig)

                    replanned = True
                    self._logger.info(
                        "Recovery execution started asynchronously for plan '%s' with %d task(s)",
                        recovery_plan.id,
                        len(recovery_plan.tasks),
                    )
                    try:
                        recovery_result = await self._executor.execute_plan_async(recovery_plan)
                        recovery_duration = time.perf_counter() - recovery_start
                        all_results.append(recovery_result)
                        self._logger.info(
                            "Recovery execution completed asynchronously: %d completed, %d failed, %d skipped (success=%s)",
                            len(recovery_result.completed_tasks),
                            len(recovery_result.failed_tasks),
                            len(recovery_result.skipped_tasks),
                            recovery_result.success,
                        )

                        # Record recovery wave into execution memory and update metrics
                        memory = memory.record_execution(recovery_result, wave=attempt + 1)
                        memory.metrics.replan_count = attempt
                        memory.metrics.total_recovery_duration += recovery_duration
                        if recovery_result.success:
                            memory.metrics.successful_recoveries += 1
                        else:
                            memory.metrics.failed_recoveries += 1

                        execution_result = execution_result.merge(recovery_result)
                        self._logger.info(
                            "Merged execution result asynchronously: %d total completed, %d failed, %d skipped (overall success=%s)",
                            len(execution_result.completed_tasks),
                            len(execution_result.failed_tasks),
                            len(execution_result.skipped_tasks),
                            execution_result.success,
                        )

                        # Edge case: halt if recovery produced zero progress
                        if not recovery_result.completed_tasks and not recovery_result.success:
                            self._logger.info(
                                "Recovery execution produced no forward progress (0 completed tasks) asynchronously for query '%s', halting further replanning",
                                query,
                            )
                            break
                    except Exception as exc:
                        self._logger.warning("Executor failed executing recovery plan asynchronously: %s", exc)
                        break

                duration = time.perf_counter() - start_time
                execution_result.execution_duration = duration
                response_content = self._format_execution_result_response(execution_result)
                metadata = self._build_replan_response_metadata(
                    original_plan=plan,
                    recovery_plan=recovery_plan,
                    execution_result=execution_result,
                    execution_results=all_results,
                    replanned=replanned,
                    replan_attempts=attempt,
                    execution_duration=duration,
                    memory=memory,
                )
                resp = AIResponse(
                    content=response_content,
                    model="executor",
                    duration=duration,
                    metadata=metadata,
                )
                return await self._wrap_skill_response_async(
                    resp,
                    query=query,
                    model="executor",
                    conversation_id=conversation_id,
                    record_memory=record_memory,
                    duration=duration,
                )

        # 2. Intent classification via IntentRouter
        intent = self._intent_router.classify(query)
        detected_intent = intent.intent
        confidence = intent.confidence
        reason = intent.reason

        with self._lock:
            self._last_intent = intent

        self._logger.info(
            "Detected intent:\n%s\nconfidence=%.2f\nreason=%s",
            detected_intent.value if hasattr(detected_intent, "value") else detected_intent,
            confidence,
            reason,
        )

        # 2. Intent-based dispatching
        dispatched_response = await self._dispatch_async(
            detected_intent,
            query=query,
            conversation_id=conversation_id,
            record_memory=record_memory,
        )
        if dispatched_response is not None:
            if hasattr(self._last_plan, "strategy") and isinstance(dispatched_response.metadata, dict):
                if "planning_strategy" not in dispatched_response.metadata:
                    dispatched_response.metadata["planning_strategy"] = (
                        self._last_plan.strategy.value
                        if hasattr(self._last_plan.strategy, "value")
                        else str(self._last_plan.strategy)
                    )
            return dispatched_response

        resolved_limit = (
            history_limit
            if history_limit is not None
            else getattr(self._config.ai, "history_limit", 20)
        )

        with self._lock:
            history = None
            if use_history and self._memory_manager is not None:
                history = self._memory_manager.get_recent(
                    limit=resolved_limit,
                    conversation_id=conversation_id,
                )

            # Record user query in memory
            if record_memory and self._memory_manager is not None:
                self._memory_manager.add(
                    content=query,
                    role="user",
                    conversation_id=conversation_id,
                    source="user",
                )

            payload = self._prompt_builder.build_payload(
                query=query,
                history=history,
                system_prompt=system_prompt,
                extra_context=extra_context,
            )

        # Emit ai.started event asynchronously
        await self._event_bus.publish_async(
            "ai.started",
            payload={
                "query": query,
                "model": self._provider.model,
                "conversation_id": conversation_id,
            },
            source="ai_manager",
        )

        try:
            response = await self._provider.generate_async(payload, config=config)

            # Record assistant response in memory
            if record_memory and self._memory_manager is not None:
                self._memory_manager.add(
                    content=response.content,
                    role="assistant",
                    conversation_id=conversation_id,
                    source="ai",
                    metadata={
                        "model": response.model,
                        "tokens": response.total_tokens,
                        "duration": response.duration,
                    },
                )

            # Emit ai.completed event
            await self._event_bus.publish_async(
                "ai.completed",
                payload={
                    "query": query,
                    "response": response.content,
                    "model": response.model,
                    "duration": response.duration,
                    "tokens": response.total_tokens,
                    "conversation_id": conversation_id,
                },
                source="ai_manager",
            )

            # Emit ai.generated event
            await self._event_bus.publish_async(
                "ai.generated",
                payload={
                    "query": query,
                    "response": response.content,
                    "model": response.model,
                    "duration": response.duration,
                    "tokens": response.total_tokens,
                    "conversation_id": conversation_id,
                },
                source="ai_manager",
            )
            if hasattr(self._last_plan, "strategy") and isinstance(response.metadata, dict):
                if "planning_strategy" not in response.metadata:
                    response.metadata["planning_strategy"] = (
                        self._last_plan.strategy.value
                        if hasattr(self._last_plan.strategy, "value")
                        else str(self._last_plan.strategy)
                    )
            return response

        except Exception as exc:
            self._logger.exception(f"AIManager async generation failed: {exc}")
            await self._event_bus.publish_async(
                "ai.failed",
                payload={
                    "query": query,
                    "error": str(exc),
                    "exception_type": type(exc).__name__,
                    "conversation_id": conversation_id,
                },
                source="ai_manager",
            )
            if isinstance(exc, AIError):
                raise
            raise AIError(f"Async generation failed: {exc}") from exc

    def stream_generate(
        self,
        query: str,
        *,
        conversation_id: str = "default",
        **kwargs: Any,
    ) -> Any:
        """Stream conversational generation synchronously (future capability).

        Raises:
            NotImplementedError: Streaming generation is not yet implemented.
        """
        raise NotImplementedError("Streaming generation is not yet implemented.")

    async def stream_generate_async(
        self,
        query: str,
        *,
        conversation_id: str = "default",
        **kwargs: Any,
    ) -> Any:
        """Stream conversational generation asynchronously (future capability).

        Raises:
            NotImplementedError: Asynchronous streaming generation is not yet implemented.
        """
        raise NotImplementedError("Asynchronous streaming generation is not yet implemented.")

    def chat(self, query: str, **kwargs: Any) -> str:
        """Convenience method returning the textual response directly.

        Args:
            query: User's command or question.
            **kwargs: Extra parameters passed to generate().

        Returns:
            Generated response content string.
        """
        response = self.generate(query, **kwargs)
        return response.content

    async def chat_async(self, query: str, **kwargs: Any) -> str:
        """Convenience async method returning the textual response directly.

        Args:
            query: User's command or question.
            **kwargs: Extra parameters passed to generate_async().

        Returns:
            Generated response content string.
        """
        response = await self.generate_async(query, **kwargs)
        return response.content


# Global default AIManager singleton
ai_manager: Final[AIManager] = AIManager()
