"""Planner module for J.A.R.V.I.S.

Provides deterministic, rule-based decomposition of user queries into
ordered sequences of executable Tasks, with an injected LLM fallback
for hybrid reasoning and complex queries.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
import uuid
from typing import Any, Dict, Final, List, Optional, Set, Tuple, Union

from app.ai.planner.events import (
    PlanCompleted,
    PlanFailed,
    PlanStarted,
    PlannerEventBus,
    RecoveryCompleted,
    RecoveryFailed,
    RecoveryStarted,
)
from app.ai.planner.heuristics import RecoveryDecision, evaluate_recovery_viability
from app.ai.planner.memory import ExecutionMemory, FailureCategory, TaskExecutionRecord
from app.ai.planner.memory_summary import MemorySummaryBuilder
from app.ai.planner.models import ExecutionResult, Plan, PlanningStrategy, Task
from app.core.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Supported Actions & Default System Prompt
# ---------------------------------------------------------------------------

KNOWN_ACTIONS: Final[Set[str]] = {
    "open_app",
    "web_search",
    "summarize_file",
    "calculate",
    "save_memory",
    "clear_memory",
}

PLANNER_SYSTEM_PROMPT: Final[str] = (
    "You are the task planning engine for J.A.R.V.I.S.\n"
    "Decompose the user query into an ordered sequence of executable tasks.\n"
    "You MUST respond ONLY with a valid JSON object strictly adhering to this schema:\n"
    "{\n"
    '  "tasks": [\n'
    '    {\n'
    '      "id": "task_1",\n'
    '      "action": "open_app|web_search|summarize_file|calculate|save_memory|clear_memory",\n'
    '      "target": "target string or null",\n'
    '      "parameters": {},\n'
    '      "dependencies": []\n'
    '    }\n'
    '  ]\n'
    "}\n"
    "Rules:\n"
    "1. Only use allowed actions: open_app, web_search, summarize_file, calculate, save_memory, clear_memory.\n"
    "2. Task IDs must be unique strings.\n"
    "3. Dependencies must only reference IDs of tasks defined earlier in the list.\n"
    "4. If the query cannot be decomposed into supported actions, return {\"tasks\": []}.\n"
    "5. Do not include any explanations, markdown code blocks, or preamble."
)

RECOVERY_SYSTEM_PROMPT: Final[str] = (
    "You are the task recovery planning engine for J.A.R.V.I.S.\n"
    "An execution plan partially failed. Generate a recovery plan for ONLY the remaining uncompleted work.\n"
    "Never regenerate or repeat tasks that already completed successfully.\n"
    "You MUST respond ONLY with a valid JSON object strictly adhering to this schema:\n"
    "{\n"
    '  "tasks": [\n'
    '    {\n'
    '      "id": "task_1",\n'
    '      "action": "open_app|web_search|summarize_file|calculate|save_memory|clear_memory",\n'
    '      "target": "target string or null",\n'
    '      "parameters": {},\n'
    '      "dependencies": []\n'
    '    }\n'
    '  ]\n'
    "}\n"
    "Rules:\n"
    "1. Only use allowed actions: open_app, web_search, summarize_file, calculate, save_memory, clear_memory.\n"
    "2. Task IDs must be unique strings.\n"
    "3. Dependencies must only reference IDs of tasks defined earlier in the list or previously completed tasks.\n"
    "4. Do NOT regenerate tasks that completed successfully.\n"
    "5. If no recovery is possible or no remaining tasks are needed, return {\"tasks\": []}.\n"
    "6. Do not include any explanations, markdown code blocks, or preamble."
)

# ---------------------------------------------------------------------------
# Regex Patterns for Atomic Actions
# ---------------------------------------------------------------------------

_RE_OPEN_APP = re.compile(
    r"^(?:open|launch|start|run)\s+(.+)$",
    re.IGNORECASE,
)

_RE_WEB_SEARCH = re.compile(
    r"^(?:search\s+for|search\s+web\s+for|search\s+the\s+web\s+for|search|google|lookup|find)\s+(.+)$",
    re.IGNORECASE,
)

_RE_SUMMARIZE_FILE = re.compile(
    r"^(?:summarize|summarise|read|parse)\s+(?:file\s+|document\s+|the\s+)?(.+)$",
    re.IGNORECASE,
)

_RE_CALCULATE = re.compile(
    r"^(?:calculate|compute|solve|eval|evaluate)\s*(.*)$",
    re.IGNORECASE,
)

_RE_SAVE_MEMORY = re.compile(
    r"^(?:remember\s+this|remember\s+that|remember)\s*(.*)$",
    re.IGNORECASE,
)

_RE_CLEAR_MEMORY = re.compile(
    r"^(?:forget\s+that|forget\s+this|forget\s+everything|forget|clear\s+memory)\b.*$",
    re.IGNORECASE,
)

# Conjunction splitters for composite queries
_RE_CONJUNCTIONS = re.compile(
    r"\b(?:and\s+then|then|after\s+that|and)\b|[,;]",
    re.IGNORECASE,
)


class Planner:
    """Hybrid task planning engine for J.A.R.V.I.S.

    Combines deterministic regex rules for standard commands with an
    injected LLM fallback for natural language decomposition.
    """

    def __init__(
        self,
        provider_instance: Optional[Any] = None,
        strategy: PlanningStrategy = PlanningStrategy.HYBRID,
        container_instance: Optional[Any] = None,
        allowed_actions: Optional[Set[str]] = None,
        event_bus: Optional[PlannerEventBus] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Planner instance.

        Args:
            provider_instance: Optional injected LLM provider for planning.
            strategy: Default PlanningStrategy (HYBRID, RULE_BASED, or LLM).
            container_instance: Optional ServiceContainer instance.
            allowed_actions: Optional set of allowed action names.
            event_bus: Optional PlannerEventBus instance for observability.
        """
        self._lock = threading.RLock()
        self._strategy = strategy
        self._allowed_actions: Set[str] = set(allowed_actions) if allowed_actions else set(KNOWN_ACTIONS)
        self._event_bus: Optional[PlannerEventBus] = event_bus or kwargs.get("planner_event_bus")

        # ServiceContainer resolution
        self._container = container_instance
        if self._container is None:
            try:
                from app.core.container import container
                self._container = container
            except Exception:
                self._container = None

        # Dependency resolution: Planning Provider
        self._provider = provider_instance
        if self._provider is None and self._container is not None:
            if hasattr(self._container, "exists") and hasattr(self._container, "resolve"):
                if self._container.exists("planning_provider"):
                    self._provider = self._container.resolve("planning_provider")

        self._register_with_container()

    def _register_with_container(self) -> None:
        """Self-register with ServiceContainer if available."""
        if self._container is not None and hasattr(self._container, "register_singleton"):
            try:
                self._container.register_singleton("planner", self, allow_override=True)
                logger.debug("Planner registered with ServiceContainer.")
            except Exception:
                pass

    @property
    def event_bus(self) -> Optional[PlannerEventBus]:
        """Return the active planner event bus."""
        return self._event_bus

    @event_bus.setter
    def event_bus(self, value: Optional[PlannerEventBus]) -> None:
        """Set or update the active planner event bus."""
        self._event_bus = value

    @property
    def provider(self) -> Optional[Any]:
        """Return the injected planning provider."""
        return self._provider

    @provider.setter
    def provider(self, value: Optional[Any]) -> None:
        """Set or update the injected planning provider."""
        self._provider = value

    @property
    def strategy(self) -> PlanningStrategy:
        """Return the active planning strategy."""
        return self._strategy

    @strategy.setter
    def strategy(self, value: PlanningStrategy) -> None:
        """Set the default planning strategy."""
        self._strategy = value

    @property
    def allowed_actions(self) -> Set[str]:
        """Return the set of recognized task action identifiers."""
        return set(self._allowed_actions)

    def add_allowed_action(self, action: str) -> None:
        """Register an additional permitted action identifier."""
        if action and action.strip():
            self._allowed_actions.add(action.strip())

    # --------------------------------------------------------------------------
    # Planning Pipeline
    # --------------------------------------------------------------------------

    def create_plan(
        self,
        query: str,
        strategy: Optional[PlanningStrategy] = None,
    ) -> Plan:
        """Create an ordered Plan from a user query using the configured strategy.

        Args:
            query: The user command or prompt.
            strategy: Optional override for planning strategy.

        Returns:
            Plan containing ordered tasks, or an empty Plan if unknown or invalid.
        """
        t_start = time.perf_counter()
        if not query or not query.strip():
            empty_plan = Plan(
                query=query if query is not None else "",
                tasks=[],
                strategy=strategy or self._strategy,
            )
            return empty_plan

        cleaned_query = query.strip()
        active_strategy = strategy if strategy is not None else self._strategy
        res_plan: Optional[Plan] = None

        with self._lock:
            # 1. Deterministic Rule-Based Planning
            if active_strategy in (PlanningStrategy.RULE_BASED, PlanningStrategy.HYBRID):
                rule_tasks = self._create_rule_plan(cleaned_query)
                if rule_tasks:
                    logger.info(
                        "Planner created rule-based plan with %d tasks for query: '%s'",
                        len(rule_tasks),
                        cleaned_query,
                    )
                    res_plan = Plan(
                        query=cleaned_query,
                        tasks=rule_tasks,
                        strategy=PlanningStrategy.RULE_BASED,
                    )
                elif active_strategy == PlanningStrategy.RULE_BASED:
                    logger.debug("Rule-based planner found no actions for: '%s'", cleaned_query)
                    res_plan = Plan(
                        query=cleaned_query,
                        tasks=[],
                        strategy=PlanningStrategy.RULE_BASED,
                    )

            # 2. LLM Fallback / Direct LLM Planning
            if res_plan is None and active_strategy in (PlanningStrategy.LLM, PlanningStrategy.HYBRID):
                if self._provider is None:
                    logger.debug("LLM planning invoked but no provider injected for: '%s'", cleaned_query)
                    res_plan = Plan(
                        query=cleaned_query,
                        tasks=[],
                        strategy=active_strategy,
                    )
                else:
                    logger.info("Planner invoking LLM planning provider for query: '%s'", cleaned_query)
                    prompt = self._build_planning_prompt(cleaned_query)
                    llm_tasks = self._generate_plan_sync(prompt)

                    if llm_tasks is None:
                        logger.warning("LLM plan rejected due to validation failure for query: '%s'", cleaned_query)
                        res_plan = Plan(
                            query=cleaned_query,
                            tasks=[],
                            strategy=PlanningStrategy.LLM,
                        )
                    else:
                        logger.info("Planner successfully generated %d tasks via LLM for query: '%s'", len(llm_tasks), cleaned_query)
                        res_plan = Plan(
                            query=cleaned_query,
                            tasks=llm_tasks,
                            strategy=PlanningStrategy.LLM,
                        )

            if res_plan is None:
                res_plan = Plan(query=cleaned_query, tasks=[], strategy=active_strategy)

        # Emit PlanStarted event if event bus is attached
        if self._event_bus is not None:
            p_latency = time.perf_counter() - t_start
            self._event_bus.publish(
                PlanStarted(
                    execution_id=res_plan.id,
                    plan_id=res_plan.id,
                    query=cleaned_query,
                    task_count=len(res_plan.tasks),
                    strategy=res_plan.strategy.value if hasattr(res_plan.strategy, "value") else str(res_plan.strategy),
                    metadata={"planning_latency": p_latency},
                )
            )

        return res_plan

    async def create_plan_async(
        self,
        query: str,
        strategy: Optional[PlanningStrategy] = None,
    ) -> Plan:
        """Asynchronously create an ordered Plan from a user query.

        Args:
            query: The user command or prompt.
            strategy: Optional override for planning strategy.

        Returns:
            Plan containing ordered tasks, or an empty Plan if unknown or invalid.
        """
        t_start = time.perf_counter()
        if not query or not query.strip():
            return Plan(
                query=query if query is not None else "",
                tasks=[],
                strategy=strategy or self._strategy,
            )

        cleaned_query = query.strip()
        active_strategy = strategy if strategy is not None else self._strategy
        res_plan: Optional[Plan] = None

        # Try rule-based first if enabled
        if active_strategy in (PlanningStrategy.RULE_BASED, PlanningStrategy.HYBRID):
            rule_tasks = self._create_rule_plan(cleaned_query)
            if rule_tasks:
                res_plan = Plan(
                    query=cleaned_query,
                    tasks=rule_tasks,
                    strategy=PlanningStrategy.RULE_BASED,
                )
            elif active_strategy == PlanningStrategy.RULE_BASED:
                res_plan = Plan(
                    query=cleaned_query,
                    tasks=[],
                    strategy=PlanningStrategy.RULE_BASED,
                )

        if res_plan is None and active_strategy in (PlanningStrategy.LLM, PlanningStrategy.HYBRID):
            if self._provider is None:
                res_plan = Plan(query=cleaned_query, tasks=[], strategy=active_strategy)
            else:
                logger.info("Planner invoking LLM planning provider asynchronously for query: '%s'", cleaned_query)
                prompt = self._build_planning_prompt(cleaned_query)
                llm_tasks = await self._generate_plan_async(prompt)

                if llm_tasks is None:
                    res_plan = Plan(query=cleaned_query, tasks=[], strategy=PlanningStrategy.LLM)
                else:
                    res_plan = Plan(query=cleaned_query, tasks=llm_tasks, strategy=PlanningStrategy.LLM)

        if res_plan is None:
            res_plan = Plan(query=cleaned_query, tasks=[], strategy=active_strategy)

        if self._event_bus is not None:
            p_latency = time.perf_counter() - t_start
            await self._event_bus.publish_async(
                PlanStarted(
                    execution_id=res_plan.id,
                    plan_id=res_plan.id,
                    query=cleaned_query,
                    task_count=len(res_plan.tasks),
                    strategy=res_plan.strategy.value if hasattr(res_plan.strategy, "value") else str(res_plan.strategy),
                    metadata={"planning_latency": p_latency},
                )
            )

        return res_plan

    # --------------------------------------------------------------------------
    # Deterministic Rule Planner
    # --------------------------------------------------------------------------

    def _create_rule_plan(self, cleaned_query: str) -> Optional[List[Task]]:
        """Decompose query into tasks using deterministic regex rules."""
        sub_clauses = [c.strip() for c in _RE_CONJUNCTIONS.split(cleaned_query) if c and c.strip()]

        if len(sub_clauses) > 1:
            composite_tasks: List[Task] = []
            all_recognized = True
            for clause in sub_clauses:
                task = self._parse_single_action(clause)
                if task is not None:
                    composite_tasks.append(task)
                else:
                    all_recognized = False

            if all_recognized and composite_tasks:
                return composite_tasks

        single_task = self._parse_single_action(cleaned_query)
        if single_task is not None:
            return [single_task]

        return None

    def _parse_single_action(self, text: str) -> Optional[Task]:
        """Parse an atomic clause into a single Task, or None if unrecognized."""
        clean = text.strip()
        if not clean:
            return None

        # 1. Open App: "open chrome"
        m = _RE_OPEN_APP.match(clean)
        if m:
            target = m.group(1).strip()
            return Task(action="open_app", target=target)

        # 2. Web Search: "search weather"
        m = _RE_WEB_SEARCH.match(clean)
        if m:
            target = m.group(1).strip()
            return Task(action="web_search", target=target)

        # 3. Summarize File: "summarize pdf"
        m = _RE_SUMMARIZE_FILE.match(clean)
        if m:
            target = m.group(1).strip()
            return Task(action="summarize_file", target=target)

        # 4. Calculate: "calculate 5+5"
        m = _RE_CALCULATE.match(clean)
        if m:
            target = m.group(1).strip() if m.group(1) else None
            params = {"expression": target} if target else {}
            return Task(action="calculate", target=target, parameters=params)

        # 5. Save Memory: "remember ..."
        m = _RE_SAVE_MEMORY.match(clean)
        if m:
            target = m.group(1).strip() if m.group(1) else None
            params = {"content": target} if target else {}
            return Task(action="save_memory", target=target, parameters=params)

        # 6. Clear Memory: "forget ..."
        if _RE_CLEAR_MEMORY.match(clean):
            return Task(action="clear_memory", target=None)

        return None

    # --------------------------------------------------------------------------
    # LLM Planning Helpers & Strict Validation
    # --------------------------------------------------------------------------

    def _build_planning_prompt(self, query: str) -> str:
        """Construct the prompt sent to the LLM planning provider."""
        return (
            f"{PLANNER_SYSTEM_PROMPT}\n\n"
            f"User request to decompose:\n\"{query}\""
        )

    def _invoke_provider(self, prompt: str) -> Optional[str]:
        """Invoke the injected LLM provider and return raw text output."""
        if self._provider is None:
            return None

        try:
            if hasattr(self._provider, "generate"):
                try:
                    res = self._provider.generate(prompt)
                except (TypeError, ValueError):
                    payload = {"contents": [{"parts": [{"text": prompt}]}]}
                    res = self._provider.generate(payload)
            elif callable(self._provider):
                res = self._provider(prompt)
            elif hasattr(self._provider, "create_plan"):
                res = self._provider.create_plan(prompt)
            else:
                logger.warning("Planning provider lacks supported generate/callable interface.")
                return None

            if hasattr(res, "content"):
                return str(res.content)
            elif isinstance(res, str):
                return res
            elif isinstance(res, dict):
                return json.dumps(res)
            return str(res)

        except Exception as exc:
            logger.warning("Error invoking planning provider: %s", exc)
            return None

    async def _invoke_provider_async(self, prompt: str) -> Optional[str]:
        """Asynchronously invoke the injected LLM provider and return raw text output."""
        if self._provider is None:
            return None

        raw_response = None
        if hasattr(self._provider, "generate_async"):
            try:
                res = await self._provider.generate_async(prompt)
            except (TypeError, ValueError):
                try:
                    payload = {"contents": [{"parts": [{"text": prompt}]}]}
                    res = await self._provider.generate_async(payload)
                except Exception:
                    res = None
            except Exception as exc:
                logger.warning("Provider generate_async failed, falling back to sync: %s", exc)
                res = None

            if res is not None:
                raw_response = res.content if hasattr(res, "content") else str(res)

        if raw_response is None:
            loop = asyncio.get_running_loop()
            raw_response = await loop.run_in_executor(None, self._invoke_provider, prompt)

        return raw_response

    def _generate_plan_sync(
        self,
        prompt: str,
        allowed_dependencies: Optional[Set[str]] = None,
    ) -> Optional[List[Task]]:
        """Synchronously invoke LLM provider and parse/validate response into tasks.

        Responsibilities:
        - Invokes the injected/configured LLM provider synchronously with the prompt.
        - Checks for non-empty text output.
        - Delegates JSON parsing, schema validation, action whitelisting, and dependency
          checks to `_parse_and_validate_llm_plan()`.

        Guarantees:
        - Always returns either a validated `List[Task]` or `None`.
        - Never returns malformed tasks or tasks with disallowed actions.
        - Never raises an unhandled exception to the caller.

        Invariants:
        - When `allowed_dependencies` is provided, dependencies may reference preceding
          tasks or IDs present in the `allowed_dependencies` set (e.g. completed tasks).

        Delegations:
        - Provider execution delegated to `_invoke_provider()`.
        - Syntactic and structural validation delegated to `_parse_and_validate_llm_plan()`.
        - DAG graph cycle detection and concurrent execution delegated to `Executor`.
        """
        if self._provider is None:
            return None

        raw_response = self._invoke_provider(prompt)
        if not raw_response or not raw_response.strip():
            logger.warning("LLM provider returned empty response for plan generation.")
            return None

        return self._parse_and_validate_llm_plan(raw_response, allowed_dependencies=allowed_dependencies)

    async def _generate_plan_async(
        self,
        prompt: str,
        allowed_dependencies: Optional[Set[str]] = None,
    ) -> Optional[List[Task]]:
        """Asynchronously invoke LLM provider and parse/validate response into tasks.

        Responsibilities:
        - Invokes the injected/configured LLM provider asynchronously with the prompt.
        - Checks for non-empty text output.
        - Delegates JSON parsing, schema validation, action whitelisting, and dependency
          checks to `_parse_and_validate_llm_plan()`.

        Guarantees:
        - Always returns either a validated `List[Task]` or `None`.
        - Never returns malformed tasks or tasks with disallowed actions.
        - Never raises an unhandled exception to the caller.

        Invariants:
        - When `allowed_dependencies` is provided, dependencies may reference preceding
          tasks or IDs present in the `allowed_dependencies` set (e.g. completed tasks).

        Delegations:
        - Provider execution delegated to `_invoke_provider_async()`.
        - Syntactic and structural validation delegated to `_parse_and_validate_llm_plan()`.
        - DAG graph cycle detection and concurrent execution delegated to `Executor`.
        """
        if self._provider is None:
            return None

        raw_response = await self._invoke_provider_async(prompt)
        if not raw_response or not raw_response.strip():
            logger.warning("LLM provider returned empty response for plan generation.")
            return None

        return self._parse_and_validate_llm_plan(raw_response, allowed_dependencies=allowed_dependencies)

    def _build_recovery_prompt(
        self,
        user_query: str,
        execution_result: Union[ExecutionResult, ExecutionMemory],
        original_plan: List[Task],
    ) -> str:
        """Construct prompt sent to LLM provider for failure recovery planning using ExecutionMemory."""
        if isinstance(execution_result, ExecutionMemory):
            memory = execution_result
        else:
            memory = ExecutionMemory.from_execution_result(execution_result)

        summary = MemorySummaryBuilder.build_planner_summary(memory, original_plan)

        prompt = (
            f"{RECOVERY_SYSTEM_PROMPT}\n\n"
            f"Original User Query:\n\"{user_query}\"\n\n"
            f"Execution State & Memory:\n"
            f"{summary}\n\n"
            f"Generate a JSON recovery plan for the remaining uncompleted work."
        )

        logger.info(
            "Generated memory-aware recovery prompt for query '%s' (%d completed, %d failed, %d skipped)",
            user_query,
            len(memory.completed_tasks),
            len(memory.failed_tasks),
            len(memory.skipped_tasks),
        )
        logger.debug("Recovery prompt contents:\n%s", prompt)
        return prompt

    def _validate_recovery_tasks(
        self,
        recovered_tasks: List[Task],
        execution_context: Union[ExecutionResult, ExecutionMemory],
    ) -> tuple[bool, list[str]]:
        """Perform recovery-specific validation on proposed recovery tasks.

        Responsibilities:
        - Validates that recovery tasks only target remaining work without regressing.
        - Enforces that no recovery task reuses an ID from completed tasks.
        - Enforces that no recovery task duplicates an (action, target) pair already completed.
        - Enforces that dependencies point only to other recovery tasks or completed tasks.

        Guarantees:
        - Returns a 2-tuple `(is_valid, error_messages)`.
        - `is_valid` is True if and only if `error_messages` is empty.

        Invariants:
        - Does not mutate the provided task list, execution result, or execution memory.

        Delegations:
        - Schema, JSON structure, and allowed actions are delegated to
          `_parse_and_validate_llm_plan()`.
        - Cycle detection and topological ordering are delegated to
          `Executor.validate_dag()`.
        """
        errors: list[str] = []
        completed_ids = execution_context.completed_task_ids
        completed_action_targets = {(t.action, t.target) for t in execution_context.completed_tasks}
        recovery_ids = {t.id for t in recovered_tasks}

        for task in recovered_tasks:
            # 1. Reject task IDs already completed
            if task.id in completed_ids:
                errors.append(f"Task ID '{task.id}' was already completed in prior execution.")

            # 2. Reject duplicate action+target pairs already completed
            if (task.action, task.target) in completed_action_targets:
                errors.append(
                    f"Task '{task.id}' repeats already completed action '{task.action}' with target '{task.target}'."
                )

            # 3. Verify every dependency is satisfied by either another recovery task or a previously completed task
            for dep_id in task.dependencies:
                if dep_id not in recovery_ids and dep_id not in completed_ids:
                    errors.append(
                        f"Task '{task.id}' has unsatisfied dependency '{dep_id}' (not in recovery tasks or completed tasks)."
                    )

        return (len(errors) == 0, errors)

    def replan(
        self,
        user_query: str,
        execution_result: Union[ExecutionResult, ExecutionMemory],
        original_plan: Optional[Union[Plan, List[Task]]] = None,
    ) -> Plan:
        """Synchronously generate an adaptive recovery plan for remaining work following execution failure.

        Responsibilities:
        - Converts execution outcome into canonical ExecutionMemory.
        - Evaluates deterministic RecoveryHeuristics before calling LLM.
        - Builds concise, memory-aware recovery prompt via MemorySummaryBuilder.
        - Generates tasks via injected LLM provider synchronously.
        - Validates recovery tasks via `_validate_recovery_tasks()`.
        - Validates composite DAG (completed tasks + recovery tasks) using Executor.

        Guarantees:
        - Returns a `Plan` instance with `metadata["is_recovery"] = True`.
        - If heuristics determine replanning is not viable, returns an empty Plan
          with `metadata["recovery_decision"]` and halts immediately.
        - If the LLM generates an invalid or empty recovery plan, returns an empty Plan
          (`tasks=[]`) with validation errors recorded in `metadata["validation_errors"]`.
        - Never raises an unhandled exception.
        """
        rec_start_time = time.perf_counter()
        if isinstance(execution_result, ExecutionMemory):
            memory = execution_result
        else:
            memory = ExecutionMemory.from_execution_result(execution_result)

        if original_plan is None:
            original_tasks: List[Task] = []
        elif isinstance(original_plan, Plan):
            original_tasks = list(original_plan.tasks)
        else:
            original_tasks = list(original_plan)

        exec_id = original_plan.id if isinstance(original_plan, Plan) else ""
        if self._event_bus is not None:
            self._event_bus.publish(
                RecoveryStarted(
                    execution_id=exec_id,
                    query=user_query,
                    attempt=len(memory.waves),
                    failed_task_ids=list(memory.failed_task_ids),
                    skipped_task_ids=list(memory.skipped_task_ids),
                )
            )

        # 1. Evaluate deterministic recovery heuristics
        decision = evaluate_recovery_viability(user_query, original_tasks, memory)
        if not decision.viable:
            logger.info(
                "Recovery heuristic determined replanning non-viable for query '%s': %s",
                user_query,
                decision.reason,
            )
            if self._event_bus is not None:
                self._event_bus.publish(
                    RecoveryFailed(
                        execution_id=exec_id,
                        query=user_query,
                        attempt=len(memory.waves),
                        reason=decision.reason,
                        duration=time.perf_counter() - rec_start_time,
                    )
                )
            return Plan(
                query=user_query,
                tasks=[],
                strategy=PlanningStrategy.LLM,
                metadata={
                    "is_recovery": True,
                    "recovery_decision": decision.to_dict(),
                    "validation_errors": [decision.reason],
                },
            )

        logger.info(
            "Starting recovery planning for query: '%s' (%d completed, %d failed, %d skipped)",
            user_query,
            len(memory.completed_tasks),
            len(memory.failed_tasks),
            len(memory.skipped_tasks),
        )

        prompt = self._build_recovery_prompt(user_query, memory, original_tasks)
        completed_ids = memory.completed_task_ids
        recovered_tasks = self._generate_plan_sync(prompt, allowed_dependencies=completed_ids)

        if not recovered_tasks:
            logger.info("Recovery plan rejected: LLM provider produced no valid tasks for query '%s'", user_query)
            if self._event_bus is not None:
                self._event_bus.publish(
                    RecoveryFailed(
                        execution_id=exec_id,
                        query=user_query,
                        attempt=len(memory.waves),
                        reason="LLM provider produced no valid tasks",
                        duration=time.perf_counter() - rec_start_time,
                    )
                )
            return Plan(
                query=user_query,
                tasks=[],
                strategy=PlanningStrategy.LLM,
                metadata={"is_recovery": True},
            )

        is_valid, errors = self._validate_recovery_tasks(recovered_tasks, memory)
        if not is_valid:
            logger.warning(
                "Recovery plan rejected due to validation errors for query '%s': %s",
                user_query,
                "; ".join(errors),
            )
            if self._event_bus is not None:
                self._event_bus.publish(
                    RecoveryFailed(
                        execution_id=exec_id,
                        query=user_query,
                        attempt=len(memory.waves),
                        reason="; ".join(errors),
                        duration=time.perf_counter() - rec_start_time,
                    )
                )
            return Plan(
                query=user_query,
                tasks=[],
                strategy=PlanningStrategy.LLM,
                metadata={"is_recovery": True, "validation_errors": errors},
            )

        from app.ai.planner.executor import executor as default_executor

        combined_plan = Plan(
            query=user_query,
            tasks=list(memory.completed_tasks) + list(recovered_tasks),
        )
        is_dag_valid, dag_err = default_executor.validate_dag(combined_plan)
        if not is_dag_valid:
            logger.warning(
                "Recovery plan rejected due to DAG validation failure for query '%s': %s",
                user_query,
                dag_err,
            )
            if self._event_bus is not None:
                self._event_bus.publish(
                    RecoveryFailed(
                        execution_id=exec_id,
                        query=user_query,
                        attempt=len(memory.waves),
                        reason=dag_err or "DAG validation failure",
                        duration=time.perf_counter() - rec_start_time,
                    )
                )
            return Plan(
                query=user_query,
                tasks=[],
                strategy=PlanningStrategy.LLM,
                metadata={"is_recovery": True, "validation_errors": [dag_err] if dag_err else []},
            )

        logger.info("Recovery plan accepted with %d task(s) for query '%s'", len(recovered_tasks), user_query)
        rec_plan = Plan(
            query=user_query,
            tasks=recovered_tasks,
            strategy=PlanningStrategy.LLM,
            metadata={"is_recovery": True},
        )
        if self._event_bus is not None:
            self._event_bus.publish(
                RecoveryCompleted(
                    execution_id=exec_id,
                    query=user_query,
                    attempt=len(memory.waves),
                    success=True,
                    new_plan_id=rec_plan.id,
                    task_count=len(recovered_tasks),
                    duration=time.perf_counter() - rec_start_time,
                )
            )
        return rec_plan

    async def replan_async(
        self,
        user_query: str,
        execution_result: Union[ExecutionResult, ExecutionMemory],
        original_plan: Optional[Union[Plan, List[Task]]] = None,
    ) -> Plan:
        """Asynchronously generate an adaptive recovery plan for remaining work following execution failure.

        Responsibilities:
        - Converts execution outcome into canonical ExecutionMemory.
        - Evaluates deterministic RecoveryHeuristics before calling LLM.
        - Builds concise, memory-aware recovery prompt via MemorySummaryBuilder.
        - Generates tasks via injected LLM provider asynchronously.
        - Validates recovery tasks via `_validate_recovery_tasks()`.
        - Validates composite DAG (completed tasks + recovery tasks) using Executor.

        Guarantees:
        - Returns a `Plan` instance with `metadata["is_recovery"] = True`.
        - If heuristics determine replanning is not viable, returns an empty Plan
          with `metadata["recovery_decision"]` and halts immediately.
        - If the LLM generates an invalid or empty recovery plan, returns an empty Plan
          (`tasks=[]`) with validation errors recorded in `metadata["validation_errors"]`.
        - Never raises an unhandled exception.
        """
        rec_start_time = time.perf_counter()
        if isinstance(execution_result, ExecutionMemory):
            memory = execution_result
        else:
            memory = ExecutionMemory.from_execution_result(execution_result)

        if original_plan is None:
            original_tasks: List[Task] = []
        elif isinstance(original_plan, Plan):
            original_tasks = list(original_plan.tasks)
        else:
            original_tasks = list(original_plan)

        exec_id = original_plan.id if isinstance(original_plan, Plan) else ""
        if self._event_bus is not None:
            await self._event_bus.publish_async(
                RecoveryStarted(
                    execution_id=exec_id,
                    query=user_query,
                    attempt=len(memory.waves),
                    failed_task_ids=list(memory.failed_task_ids),
                    skipped_task_ids=list(memory.skipped_task_ids),
                )
            )

        # 1. Evaluate deterministic recovery heuristics
        decision = evaluate_recovery_viability(user_query, original_tasks, memory)
        if not decision.viable:
            logger.info(
                "Recovery heuristic determined replanning non-viable asynchronously for query '%s': %s",
                user_query,
                decision.reason,
            )
            if self._event_bus is not None:
                await self._event_bus.publish_async(
                    RecoveryFailed(
                        execution_id=exec_id,
                        query=user_query,
                        attempt=len(memory.waves),
                        reason=decision.reason,
                        duration=time.perf_counter() - rec_start_time,
                    )
                )
            return Plan(
                query=user_query,
                tasks=[],
                strategy=PlanningStrategy.LLM,
                metadata={
                    "is_recovery": True,
                    "recovery_decision": decision.to_dict(),
                    "validation_errors": [decision.reason],
                },
            )

        logger.info(
            "Starting asynchronous recovery planning for query: '%s' (%d completed, %d failed, %d skipped)",
            user_query,
            len(memory.completed_tasks),
            len(memory.failed_tasks),
            len(memory.skipped_tasks),
        )

        prompt = self._build_recovery_prompt(user_query, memory, original_tasks)
        completed_ids = memory.completed_task_ids
        recovered_tasks = await self._generate_plan_async(prompt, allowed_dependencies=completed_ids)

        if not recovered_tasks:
            logger.info("Recovery plan rejected: LLM provider produced no valid tasks for query '%s'", user_query)
            if self._event_bus is not None:
                await self._event_bus.publish_async(
                    RecoveryFailed(
                        execution_id=exec_id,
                        query=user_query,
                        attempt=len(memory.waves),
                        reason="LLM provider produced no valid tasks",
                        duration=time.perf_counter() - rec_start_time,
                    )
                )
            return Plan(
                query=user_query,
                tasks=[],
                strategy=PlanningStrategy.LLM,
                metadata={"is_recovery": True},
            )

        is_valid, errors = self._validate_recovery_tasks(recovered_tasks, memory)
        if not is_valid:
            logger.warning(
                "Recovery plan rejected due to validation errors for query '%s': %s",
                user_query,
                "; ".join(errors),
            )
            if self._event_bus is not None:
                await self._event_bus.publish_async(
                    RecoveryFailed(
                        execution_id=exec_id,
                        query=user_query,
                        attempt=len(memory.waves),
                        reason="; ".join(errors),
                        duration=time.perf_counter() - rec_start_time,
                    )
                )
            return Plan(
                query=user_query,
                tasks=[],
                strategy=PlanningStrategy.LLM,
                metadata={"is_recovery": True, "validation_errors": errors},
            )

        from app.ai.planner.executor import executor as default_executor

        combined_plan = Plan(
            query=user_query,
            tasks=list(memory.completed_tasks) + list(recovered_tasks),
        )
        is_dag_valid, dag_err = default_executor.validate_dag(combined_plan)
        if not is_dag_valid:
            logger.warning(
                "Recovery plan rejected due to DAG validation failure for query '%s': %s",
                user_query,
                dag_err,
            )
            if self._event_bus is not None:
                await self._event_bus.publish_async(
                    RecoveryFailed(
                        execution_id=exec_id,
                        query=user_query,
                        attempt=len(memory.waves),
                        reason=dag_err or "DAG validation failure",
                        duration=time.perf_counter() - rec_start_time,
                    )
                )
            return Plan(
                query=user_query,
                tasks=[],
                strategy=PlanningStrategy.LLM,
                metadata={"is_recovery": True, "validation_errors": [dag_err] if dag_err else []},
            )

        logger.info("Recovery plan accepted with %d task(s) for query '%s'", len(recovered_tasks), user_query)
        rec_plan = Plan(
            query=user_query,
            tasks=recovered_tasks,
            strategy=PlanningStrategy.LLM,
            metadata={"is_recovery": True},
        )
        if self._event_bus is not None:
            await self._event_bus.publish_async(
                RecoveryCompleted(
                    execution_id=exec_id,
                    query=user_query,
                    attempt=len(memory.waves),
                    success=True,
                    new_plan_id=rec_plan.id,
                    task_count=len(recovered_tasks),
                    duration=time.perf_counter() - rec_start_time,
                )
            )
        return rec_plan

    def _parse_and_validate_llm_plan(
        self,
        raw_text: str,
        allowed_dependencies: Optional[Set[str]] = None,
    ) -> Optional[List[Task]]:
        """Parse raw LLM output into validated Tasks, or None if malformed/unsafe.

        Enforces:
        - Strict JSON structure.
        - Dict with 'tasks' list.
        - Known and safe action identifiers.
        - Valid, unique task IDs.
        - Valid DAG dependencies (referencing only prior task IDs or allowed completed tasks).
        """
        if not raw_text or not raw_text.strip():
            logger.warning("Rejecting plan: Empty LLM output.")
            return None

        clean_text = raw_text.strip()

        # Extract JSON content from markdown code fences if present
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text, re.IGNORECASE)
        if fence_match:
            clean_text = fence_match.group(1).strip()
        else:
            # Fallback: find outermost curly braces
            start_brace = clean_text.find("{")
            end_brace = clean_text.rfind("}")
            if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
                clean_text = clean_text[start_brace : end_brace + 1]

        # 1. Parse JSON
        try:
            data = json.loads(clean_text)
        except Exception as exc:
            logger.warning("Rejecting plan: Invalid JSON from LLM: %s (Raw: '%s')", exc, raw_text[:120])
            return None

        # 2. Schema check: root must be dictionary containing 'tasks' list
        if not isinstance(data, dict):
            logger.warning("Rejecting plan: Root JSON is not an object: %s", type(data).__name__)
            return None

        if "tasks" not in data or not isinstance(data["tasks"], list):
            logger.warning("Rejecting plan: Missing or non-list 'tasks' field.")
            return None

        validated_tasks: List[Task] = []
        seen_task_ids: Set[str] = set()

        for idx, item in enumerate(data["tasks"]):
            if not isinstance(item, dict):
                logger.warning("Rejecting plan: Task at index %d is not an object.", idx)
                return None

            # 3. Action validation
            action = item.get("action")
            if not isinstance(action, str) or not action.strip():
                logger.warning("Rejecting plan: Task %d has missing or invalid action.", idx)
                return None
            action = action.strip()
            if action not in self._allowed_actions:
                logger.warning("Rejecting plan: Unknown or disallowed action '%s' at task %d.", action, idx)
                return None

            # 4. Target validation
            target = item.get("target")
            if target is not None and not isinstance(target, str):
                target = str(target)

            # 5. Parameters validation
            parameters = item.get("parameters", {})
            if not isinstance(parameters, dict):
                logger.warning("Rejecting plan: Parameters must be a dictionary at task %d.", idx)
                return None

            # 6. Task ID validation & uniqueness
            task_id_raw = item.get("id")
            if task_id_raw is None or not str(task_id_raw).strip():
                task_id = f"task_{idx + 1}"
            else:
                task_id = str(task_id_raw).strip()

            if task_id in seen_task_ids:
                logger.warning("Rejecting plan: Duplicate task ID '%s' detected at task %d.", task_id, idx)
                return None

            # 7. Dependency validation (must reference preceding task IDs or allowed completed tasks)
            deps_raw = item.get("dependencies", [])
            if not isinstance(deps_raw, list):
                logger.warning("Rejecting plan: Dependencies must be a list at task %d.", idx)
                return None

            validated_deps: List[str] = []
            for dep in deps_raw:
                if not isinstance(dep, str) or not dep.strip():
                    logger.warning("Rejecting plan: Invalid dependency token at task %d.", idx)
                    return None
                dep_id = dep.strip()
                valid_dep = dep_id in seen_task_ids or (
                    allowed_dependencies is not None and dep_id in allowed_dependencies
                )
                if not valid_dep:
                    logger.warning(
                        "Rejecting plan: Dependency '%s' at task %d does not reference an earlier task.",
                        dep_id,
                        idx,
                    )
                    return None
                validated_deps.append(dep_id)

            seen_task_ids.add(task_id)
            validated_tasks.append(
                Task(
                    action=action,
                    target=target,
                    parameters=dict(parameters),
                    id=task_id,
                    dependencies=validated_deps,
                )
            )

        return validated_tasks


# Global singleton instance
planner: Final[Planner] = Planner()
