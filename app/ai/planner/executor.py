"""Plan Executor for J.A.R.V.I.S.

Responsible for orchestrating the dependency-aware execution of planned tasks
(DAG execution) through registered action handlers, SkillManager routing,
or ServiceContainer resolution.
"""

from __future__ import annotations

import asyncio
from collections import deque
import concurrent.futures
import inspect
import threading
import time
from typing import Any, Callable, Dict, Final, List, Optional, Set, Tuple, Union

from app.ai.planner.control import (
    ExecutionCancelledError,
    ExecutionController,
)
from app.ai.planner.events import (
    PlanCancelled,
    PlanCompleted,
    PlanFailed,
    PlannerEventBus,
    PlanPaused,
    PlanResumed,
    PlanStarted,
    TaskCompleted,
    TaskFailed,
    TaskStarted,
    TaskTimeout,
)
from app.ai.planner.models import ExecutionResult, Plan, Task, TaskStatus
from app.ai.planner.timeouts import (
    TaskTimeoutError,
    TimeoutConfig,
    TimeoutManager,
    TimeoutPolicy,
)
from app.core.container import ServiceContainer, container as default_container
from app.core.event_bus import EventBus, event_bus as default_event_bus
from app.core.logger import get_logger

logger = get_logger(__name__)


class Executor:
    """Lightweight and modular plan execution engine for J.A.R.V.I.S.

    Orchestrates the dependency-aware DAG execution of tasks in a Plan by
    dispatching them through a deterministic multi-tier resolution pipeline:
        1. Registered action handlers (in-memory custom overrides)
        2. SkillManager (registered system skills and dynamic capability routing)
        3. ServiceContainer (services, action handlers, and registered instances)
        4. Unknown action fallback -> Task marked FAILED
    """

    def __init__(
        self,
        container_instance: Optional[ServiceContainer] = None,
        event_bus_instance: Optional[Union[EventBus, PlannerEventBus, Any]] = None,
        skill_manager_instance: Optional[Any] = None,
        handlers: Optional[Dict[str, Callable[[Task], Any]]] = None,
        *,
        auto_register_in_container: bool = True,
        planner_event_bus: Optional[PlannerEventBus] = None,
        controller: Optional[ExecutionController] = None,
        timeout_config: Optional[TimeoutConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Executor instance.

        Args:
            container_instance: Optional ServiceContainer instance.
            event_bus_instance: Optional EventBus or PlannerEventBus instance.
            skill_manager_instance: Optional SkillManager instance.
            handlers: Optional mapping of action names to execution callables.
            auto_register_in_container: Whether to register self in ServiceContainer.
            planner_event_bus: Optional dedicated PlannerEventBus for planner observability.
            controller: Optional ExecutionController for pause/resume/cancellation.
            timeout_config: Optional TimeoutConfig for deadline enforcement.
        """
        self._lock = threading.RLock()
        self._container = container_instance if container_instance is not None else default_container
        self._logger = get_logger("PLAN_EXECUTOR")
        self._controller: Optional[ExecutionController] = controller
        self._timeout_config: Optional[TimeoutConfig] = timeout_config

        # Handle planner_event_bus resolution
        if isinstance(event_bus_instance, PlannerEventBus):
            self._planner_event_bus: Optional[PlannerEventBus] = event_bus_instance
            self._event_bus = default_event_bus
        elif planner_event_bus is not None:
            self._planner_event_bus = planner_event_bus
            self._event_bus = event_bus_instance if event_bus_instance is not None else default_event_bus
        else:
            kw_eb = kwargs.get("event_bus")
            if isinstance(kw_eb, PlannerEventBus):
                self._planner_event_bus = kw_eb
                self._event_bus = default_event_bus
            else:
                self._planner_event_bus = None
                self._event_bus = event_bus_instance if event_bus_instance is not None else default_event_bus

        # Support backward compatibility if handlers was passed as 3rd positional argument
        if isinstance(skill_manager_instance, dict) and handlers is None:
            handlers = skill_manager_instance
            skill_manager_instance = None

        self._handlers: Dict[str, Callable[[Task], Any]] = dict(handlers) if handlers else {}

        # Resolve SkillManager via DI priority: injected -> container -> global singleton
        if skill_manager_instance is not None:
            self._skill_manager = skill_manager_instance
        elif self._container is not None and self._container.exists("skill_manager"):
            self._skill_manager = self._container.resolve("skill_manager")
        elif self._container is not None and self._container.exists("skills"):
            self._skill_manager = self._container.resolve("skills")
        else:
            try:
                from app.skills.manager import skill_manager as default_sm
                self._skill_manager = default_sm
            except Exception:
                self._skill_manager = None

        if auto_register_in_container and self._container is not None:
            self._register_with_container()

    def _register_with_container(self) -> None:
        """Self-register with ServiceContainer if available."""
        try:
            self._container.register_singleton("executor", self, allow_override=True)
            self._container.register_singleton("plan_executor", self, allow_override=True)
            self._logger.debug("Executor registered with ServiceContainer.")
        except Exception as exc:
            self._logger.warning("Could not register Executor into container: %s", exc)

    # --------------------------------------------------------------------------
    # Properties & Handler Management
    # --------------------------------------------------------------------------

    @property
    def container(self) -> Optional[ServiceContainer]:
        """Return the service container."""
        return self._container

    @property
    def event_bus(self) -> Optional[EventBus]:
        """Return the event bus."""
        return self._event_bus

    @property
    def planner_event_bus(self) -> Optional[PlannerEventBus]:
        """Return the active planner event bus."""
        return self._planner_event_bus

    @planner_event_bus.setter
    def planner_event_bus(self, value: Optional[PlannerEventBus]) -> None:
        """Set or update the active planner event bus."""
        self._planner_event_bus = value

    @property
    def controller(self) -> Optional[ExecutionController]:
        """Return the active execution controller."""
        return self._controller

    @controller.setter
    def controller(self, value: Optional[ExecutionController]) -> None:
        """Set or update the active execution controller."""
        self._controller = value

    @property
    def timeout_config(self) -> Optional[TimeoutConfig]:
        """Return the active timeout configuration."""
        return self._timeout_config

    @timeout_config.setter
    def timeout_config(self, value: Optional[TimeoutConfig]) -> None:
        """Set or update the active timeout configuration."""
        self._timeout_config = value

    @property
    def skill_manager(self) -> Optional[Any]:
        """Return the active skill manager."""
        return self._skill_manager

    @property
    def handlers(self) -> Dict[str, Callable[[Task], Any]]:
        """Return a snapshot copy of registered action handlers."""
        with self._lock:
            return dict(self._handlers)

    def register_handler(self, action: str, handler: Callable[[Task], Any]) -> None:
        """Register an execution handler for an action name.

        Args:
            action: Case-insensitive action identifier (e.g. 'open_app', 'web_search').
            handler: Callable accepting a Task and returning a result or coroutine.
        """
        if not action or not callable(handler):
            raise ValueError("Action name must be non-empty and handler must be callable")
        with self._lock:
            key = action.strip().lower()
            self._handlers[key] = handler
            self._logger.debug("Registered action handler for '%s'", key)

    def unregister_handler(self, action: str) -> bool:
        """Unregister an action handler.

        Args:
            action: Action identifier to unregister.

        Returns:
            True if a handler was removed, False otherwise.
        """
        with self._lock:
            key = action.strip().lower()
            return self._handlers.pop(key, None) is not None

    def get_handler(self, action: str) -> Optional[Callable[[Task], Any]]:
        """Retrieve the registered handler for an action if present."""
        with self._lock:
            return self._handlers.get(action.strip().lower())

    # --------------------------------------------------------------------------
    # Resolution Pipeline (Priority: Registered -> SkillManager -> Container)
    # --------------------------------------------------------------------------

    def resolve_handler(self, task: Union[Task, str]) -> Optional[Callable[[Task], Any]]:
        """Dynamically resolve an execution handler for a task.

        Resolution priority:
            1. Registered handler (in self._handlers)
            2. SkillManager (in self._skill_manager)
            3. ServiceContainer (in self._container)
            4. Unknown action -> None (caller marks task as FAILED)

        Args:
            task: The Task or action name string to resolve.

        Returns:
            A callable handler accepting a Task, or None if unresolved.
        """
        action_name = task.action if isinstance(task, Task) else str(task)
        clean_action = action_name.strip().lower()
        task_obj = task if isinstance(task, Task) else None

        # Priority 1: Registered handler
        with self._lock:
            if clean_action in self._handlers:
                return self._handlers[clean_action]
            if action_name in self._handlers:
                return self._handlers[action_name]

        # Priority 2: SkillManager
        if self._skill_manager is not None:
            handler = self._resolve_from_skill_manager(clean_action, task_obj)
            if handler is not None:
                return handler

        # Priority 3: ServiceContainer
        if self._container is not None:
            handler = self._resolve_from_container(clean_action, task_obj)
            if handler is not None:
                return handler

        # Priority 4: Unknown action -> None
        return None

    def _resolve_handler(self, action: str) -> Optional[Callable[[Task], Any]]:
        """Backward-compatible internal resolution alias."""
        return self.resolve_handler(action)

    def _resolve_from_skill_manager(
        self, action: str, task: Optional[Task] = None
    ) -> Optional[Callable[[Task], Any]]:
        """Attempt to resolve a skill handler from SkillManager."""
        sm = self._skill_manager
        if sm is None:
            return None

        # 1. Exact name match
        skill = None
        for getter in ("get", "get_skill"):
            if hasattr(sm, getter) and callable(getattr(sm, getter)):
                try:
                    skill = getattr(sm, getter)(action)
                    if skill is not None:
                        break
                except Exception:
                    pass

        # 2. Action alias matches
        if skill is None:
            alias_map = {
                "open_app": ["system", "app_launcher", "windows"],
                "launch_app": ["system", "app_launcher", "windows"],
                "web_search": ["web_search", "search", "google"],
                "calculate": ["calc", "calculator", "math"],
            }
            for candidate in alias_map.get(action, []):
                for getter in ("get", "get_skill"):
                    if hasattr(sm, getter) and callable(getattr(sm, getter)):
                        try:
                            skill = getattr(sm, getter)(candidate)
                            if skill is not None:
                                break
                        except Exception:
                            pass
                if skill is not None:
                    break

        # 3. Dynamic can_handle check if available
        if skill is None and hasattr(sm, "list_skills") and callable(sm.list_skills):
            try:
                available_skills = sm.list_skills(enabled_only=True)
                test_cmd = (
                    f"{task.action} {task.target or ''}".strip()
                    if task is not None
                    else action
                )
                for s in available_skills:
                    if hasattr(s, "can_handle") and callable(s.can_handle):
                        try:
                            if s.can_handle(test_cmd):
                                skill = s
                                break
                        except Exception:
                            pass
            except Exception:
                pass

        if skill is not None:
            return self._build_skill_adapter(skill)

        # 4. If skill_manager itself has execute and can_handle
        if hasattr(sm, "can_handle") and callable(sm.can_handle) and hasattr(sm, "execute"):
            test_cmd = (
                f"{task.action} {task.target or ''}".strip()
                if task is not None
                else action
            )
            try:
                if sm.can_handle(test_cmd):
                    return lambda t: sm.execute(self._format_task_for_skill(t))
            except Exception:
                pass

        return None

    def _resolve_from_container(
        self, action: str, task: Optional[Task] = None
    ) -> Optional[Callable[[Task], Any]]:
        """Attempt to resolve a handler or skill from ServiceContainer."""
        if self._container is None:
            return None

        candidate_keys = [
            f"handler:{action}",
            f"action:{action}",
            action,
        ]
        if action in ("open_app", "launch_app"):
            candidate_keys.extend(["window_manager", "system_skill", "system"])
        elif action in ("web_search", "search"):
            candidate_keys.extend(["web_search_skill", "web_search", "search"])

        for key in candidate_keys:
            if self._container.exists(key):
                try:
                    item = self._container.resolve(key)
                    if callable(item) and not hasattr(item, "execute"):
                        return item
                    if hasattr(item, "execute") and callable(item.execute):
                        return self._build_skill_adapter(item)
                    if hasattr(item, "handle") and callable(item.handle):
                        return lambda t: item.handle(self._format_task_for_skill(t))
                    if hasattr(item, "run") and callable(item.run):
                        return lambda t: item.run(self._format_task_for_skill(t))
                    if action in ("open_app", "launch_app") and hasattr(item, "launch_application"):
                        return lambda t: item.launch_application(t.target or "")
                except Exception as exc:
                    self._logger.debug("Container resolution for key '%s' failed: %s", key, exc)

        return None

    @staticmethod
    def _format_task_for_skill(task: Task) -> str:
        """Format a Task into natural command text expected by standard skills."""
        action = task.action.lower()
        target = task.target or ""
        if action == "open_app":
            return f"open {target}".strip()
        elif action == "web_search":
            return f"search {target}".strip()
        elif action == "calculate":
            return f"calculate {target}".strip()
        elif action == "summarize_file":
            return f"summarize {target}".strip()
        elif action == "save_memory":
            return f"remember {target}".strip()
        elif action == "clear_memory":
            return "clear memory"
        else:
            return f"{task.action} {target}".strip() if target else task.action

    def _build_skill_adapter(self, skill: Any) -> Callable[[Task], Any]:
        """Wrap a resolved skill into a unified callable handler."""
        def _adapter(task: Task) -> Any:
            cmd = self._format_task_for_skill(task)

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running() and hasattr(skill, "execute_async") and callable(skill.execute_async):
                return skill.execute_async(cmd)

            if hasattr(skill, "execute") and callable(skill.execute):
                try:
                    return skill.execute(cmd)
                except TypeError:
                    return skill.execute(task)
            elif hasattr(skill, "execute_async") and callable(skill.execute_async):
                return skill.execute_async(cmd)
            elif callable(skill):
                try:
                    return skill(cmd)
                except TypeError:
                    return skill(task)
            raise AttributeError(f"Skill {skill} has no executable execute method")

        return _adapter

    # --------------------------------------------------------------------------
    # Synchronous and Asynchronous Invocation Helpers
    # --------------------------------------------------------------------------

    def _invoke_handler_sync(self, handler: Callable[[Task], Any], task: Task) -> Any:
        """Invoke a handler synchronously, handling both sync functions and coroutines."""
        sig = inspect.signature(handler)
        params = list(sig.parameters.values())
        res = handler() if len(params) == 0 else handler(task)

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

    async def _invoke_handler_async(self, handler: Callable[[Task], Any], task: Task) -> Any:
        """Invoke a handler asynchronously."""
        sig = inspect.signature(handler)
        params = list(sig.parameters.values())
        res = handler() if len(params) == 0 else handler(task)

        if inspect.isawaitable(res):
            return await res
        return res

    # --------------------------------------------------------------------------
    # DAG Dependency Validation & Graph Helpers
    # --------------------------------------------------------------------------

    def validate_dag(self, plan: Plan) -> Tuple[bool, Optional[str]]:
        """Validate plan task dependency graph using Kahn's algorithm.

        Checks for:
            1. Duplicate IDs
            2. Missing dependency IDs
            3. Self-dependencies
            4. Circular dependencies (cycles)

        Args:
            plan: The Plan containing tasks to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        if not plan.tasks:
            return True, None

        seen_ids: Set[str] = set()
        task_map: Dict[str, Task] = {}

        # 1. Check duplicate IDs
        for task in plan.tasks:
            if task.id in seen_ids:
                msg = f"Duplicate task ID detected: '{task.id}'"
                self._logger.error("Executor DAG validation failure: %s", msg)
                return False, msg
            seen_ids.add(task.id)
            task_map[task.id] = task

        # 2. Check self-dependencies and missing dependencies
        in_degree: Dict[str, int] = {task.id: 0 for task in plan.tasks}
        adj: Dict[str, List[str]] = {task.id: [] for task in plan.tasks}

        for task in plan.tasks:
            deps = task.dependencies or []
            for dep_id in deps:
                if dep_id == task.id:
                    msg = f"Self-dependency detected in task '{task.id}'"
                    self._logger.error("Executor DAG validation failure: %s", msg)
                    return False, msg
                if dep_id not in task_map:
                    msg = f"Task '{task.id}' depends on missing task ID '{dep_id}'"
                    self._logger.error("Executor DAG validation failure: %s", msg)
                    return False, msg
                adj[dep_id].append(task.id)
                in_degree[task.id] += 1

        # 3. Kahn's Algorithm for cycle detection
        queue: deque[str] = deque([task_id for task_id, deg in in_degree.items() if deg == 0])
        visited_count = 0

        while queue:
            curr = queue.popleft()
            visited_count += 1
            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if visited_count < len(plan.tasks):
            msg = "Circular dependency detected in task plan"
            self._logger.error("Executor DAG validation failure: %s", msg)
            return False, msg

        return True, None

    @staticmethod
    def _find_descendants(root_id: str, reverse_deps: Dict[str, Set[str]]) -> Set[str]:
        """Find all reachable descendant task IDs using BFS."""
        descendants: Set[str] = set()
        queue: deque[str] = deque(reverse_deps.get(root_id, set()))
        while queue:
            curr = queue.popleft()
            if curr not in descendants:
                descendants.add(curr)
                queue.extend(reverse_deps.get(curr, set()))
        return descendants

    def _execute_single_task_sync(
        self,
        task: Task,
        *,
        controller: Optional[ExecutionController] = None,
        timeout_mgr: Optional[TimeoutManager] = None,
    ) -> Tuple[Task, bool, Optional[str], Optional[str]]:
        """Synchronously execute a single task through its resolved handler.

        Returns:
            Tuple of (task, success, result_string, error_message).
        """
        if controller is not None:
            controller.check_checkpoint(f"before_task_{task.id}")

        task.status = TaskStatus.RUNNING
        self._logger.info("Task started: '%s' (%s)", task.id, task.action)
        t_start = time.perf_counter()

        if self._planner_event_bus is not None:
            self._planner_event_bus.publish(
                TaskStarted(
                    execution_id=str(task.parameters.get("plan_id", "")),
                    plan_id=str(task.parameters.get("plan_id", "")),
                    task_id=task.id,
                    action=task.action,
                    target=task.target,
                    dependencies=list(task.dependencies or []),
                )
            )

        if self._event_bus is not None:
            self._event_bus.publish(
                "task.started",
                {"task_id": task.id, "action": task.action, "target": task.target},
                source="executor",
            )

        handler = self.resolve_handler(task)
        if handler is None:
            err_msg = f"No handler registered for action '{task.action}'."
            task.status = TaskStatus.FAILED
            task_duration = time.perf_counter() - t_start
            self._logger.error("Task failed: '%s' (%s): %s", task.id, task.action, err_msg)
            if self._planner_event_bus is not None:
                self._planner_event_bus.publish(
                    TaskFailed(
                        execution_id=str(task.parameters.get("plan_id", "")),
                        plan_id=str(task.parameters.get("plan_id", "")),
                        task_id=task.id,
                        action=task.action,
                        error=err_msg,
                        duration=task_duration,
                    )
                )
            if self._event_bus is not None:
                self._event_bus.publish(
                    "task.failed",
                    {"task_id": task.id, "action": task.action, "error": err_msg},
                    source="executor",
                )
            return task, False, None, err_msg

        try:
            if timeout_mgr is not None:
                plan_id = str(task.parameters.get("plan_id", ""))
                res = timeout_mgr.run_sync(
                    task.id,
                    task.action,
                    lambda: self._invoke_handler_sync(handler, task),
                    plan_id=plan_id,
                    execution_id=plan_id,
                )
            else:
                res = self._invoke_handler_sync(handler, task)

            result_str = str(res) if res is not None else "OK"
            task.status = TaskStatus.COMPLETED
            task_duration = time.perf_counter() - t_start
            self._logger.info("Task completed: '%s' (%s): %s", task.id, task.action, result_str)
            if self._planner_event_bus is not None:
                self._planner_event_bus.publish(
                    TaskCompleted(
                        execution_id=str(task.parameters.get("plan_id", "")),
                        plan_id=str(task.parameters.get("plan_id", "")),
                        task_id=task.id,
                        action=task.action,
                        result=result_str,
                        duration=task_duration,
                    )
                )
            if self._event_bus is not None:
                self._event_bus.publish(
                    "task.completed",
                    {"task_id": task.id, "action": task.action, "result": result_str},
                    source="executor",
                )

            if controller is not None:
                try:
                    controller.check_checkpoint(f"after_task_{task.id}")
                except ExecutionCancelledError:
                    pass

            return task, True, result_str, None
        except TaskTimeoutError as exc:
            task_duration = time.perf_counter() - t_start
            if exc.policy == TimeoutPolicy.ABORT:
                task.status = TaskStatus.CANCELLED
                raise ExecutionCancelledError(str(exc))
            task.status = TaskStatus.SKIPPED if exc.policy == TimeoutPolicy.SKIP else TaskStatus.FAILED
            err_msg = str(exc)
            self._logger.error("Task timed out: '%s' (%s): %s", task.id, task.action, err_msg)
            if self._planner_event_bus is not None:
                self._planner_event_bus.publish(
                    TaskFailed(
                        execution_id=str(task.parameters.get("plan_id", "")),
                        plan_id=str(task.parameters.get("plan_id", "")),
                        task_id=task.id,
                        action=task.action,
                        error=err_msg,
                        duration=task_duration,
                    )
                )
            if self._event_bus is not None:
                self._event_bus.publish(
                    "task.failed",
                    {"task_id": task.id, "action": task.action, "error": err_msg},
                    source="executor",
                )
            return task, False, None, err_msg
        except ExecutionCancelledError:
            task.status = TaskStatus.CANCELLED
            raise
        except Exception as exc:
            err_msg = str(exc)
            task.status = TaskStatus.FAILED
            task_duration = time.perf_counter() - t_start
            self._logger.error("Task failed: '%s' (%s): %s", task.id, task.action, err_msg)
            if self._planner_event_bus is not None:
                self._planner_event_bus.publish(
                    TaskFailed(
                        execution_id=str(task.parameters.get("plan_id", "")),
                        plan_id=str(task.parameters.get("plan_id", "")),
                        task_id=task.id,
                        action=task.action,
                        error=err_msg,
                        duration=task_duration,
                    )
                )
            if self._event_bus is not None:
                self._event_bus.publish(
                    "task.failed",
                    {"task_id": task.id, "action": task.action, "error": err_msg},
                    source="executor",
                )
            return task, False, None, err_msg

    async def _execute_single_task_async(
        self,
        task: Task,
        *,
        controller: Optional[ExecutionController] = None,
        timeout_mgr: Optional[TimeoutManager] = None,
    ) -> Tuple[Task, bool, Optional[str], Optional[str]]:
        """Asynchronously execute a single task through its resolved handler.

        Returns:
            Tuple of (task, success, result_string, error_message).
        """
        if controller is not None:
            await controller.check_checkpoint_async(f"before_task_{task.id}")

        task.status = TaskStatus.RUNNING
        self._logger.info("Task started: '%s' (%s)", task.id, task.action)
        t_start = time.perf_counter()

        if self._planner_event_bus is not None:
            await self._planner_event_bus.publish_async(
                TaskStarted(
                    execution_id=str(task.parameters.get("plan_id", "")),
                    plan_id=str(task.parameters.get("plan_id", "")),
                    task_id=task.id,
                    action=task.action,
                    target=task.target,
                    dependencies=list(task.dependencies or []),
                )
            )

        if self._event_bus is not None:
            await self._event_bus.publish_async(
                "task.started",
                {"task_id": task.id, "action": task.action, "target": task.target},
                source="executor",
            )

        handler = self.resolve_handler(task)
        if handler is None:
            err_msg = f"No handler registered for action '{task.action}'."
            task.status = TaskStatus.FAILED
            task_duration = time.perf_counter() - t_start
            self._logger.error("Task failed: '%s' (%s): %s", task.id, task.action, err_msg)
            if self._planner_event_bus is not None:
                await self._planner_event_bus.publish_async(
                    TaskFailed(
                        execution_id=str(task.parameters.get("plan_id", "")),
                        plan_id=str(task.parameters.get("plan_id", "")),
                        task_id=task.id,
                        action=task.action,
                        error=err_msg,
                        duration=task_duration,
                    )
                )
            if self._event_bus is not None:
                await self._event_bus.publish_async(
                    "task.failed",
                    {"task_id": task.id, "action": task.action, "error": err_msg},
                    source="executor",
                )
            return task, False, None, err_msg

        try:
            if timeout_mgr is not None:
                plan_id = str(task.parameters.get("plan_id", ""))
                res = await timeout_mgr.run_async(
                    task.id,
                    task.action,
                    lambda: self._invoke_handler_async(handler, task),
                    plan_id=plan_id,
                    execution_id=plan_id,
                )
            else:
                res = await self._invoke_handler_async(handler, task)

            result_str = str(res) if res is not None else "OK"
            task.status = TaskStatus.COMPLETED
            task_duration = time.perf_counter() - t_start
            self._logger.info("Task completed: '%s' (%s): %s", task.id, task.action, result_str)
            if self._planner_event_bus is not None:
                await self._planner_event_bus.publish_async(
                    TaskCompleted(
                        execution_id=str(task.parameters.get("plan_id", "")),
                        plan_id=str(task.parameters.get("plan_id", "")),
                        task_id=task.id,
                        action=task.action,
                        result=result_str,
                        duration=task_duration,
                    )
                )
            if self._event_bus is not None:
                await self._event_bus.publish_async(
                    "task.completed",
                    {"task_id": task.id, "action": task.action, "result": result_str},
                    source="executor",
                )

            if controller is not None:
                try:
                    await controller.check_checkpoint_async(f"after_task_{task.id}")
                except ExecutionCancelledError:
                    pass

            return task, True, result_str, None
        except TaskTimeoutError as exc:
            task_duration = time.perf_counter() - t_start
            if exc.policy == TimeoutPolicy.ABORT:
                task.status = TaskStatus.CANCELLED
                raise ExecutionCancelledError(str(exc))
            task.status = TaskStatus.SKIPPED if exc.policy == TimeoutPolicy.SKIP else TaskStatus.FAILED
            err_msg = str(exc)
            self._logger.error("Task timed out: '%s' (%s): %s", task.id, task.action, err_msg)
            if self._planner_event_bus is not None:
                await self._planner_event_bus.publish_async(
                    TaskFailed(
                        execution_id=str(task.parameters.get("plan_id", "")),
                        plan_id=str(task.parameters.get("plan_id", "")),
                        task_id=task.id,
                        action=task.action,
                        error=err_msg,
                        duration=task_duration,
                    )
                )
            if self._event_bus is not None:
                await self._event_bus.publish_async(
                    "task.failed",
                    {"task_id": task.id, "action": task.action, "error": err_msg},
                    source="executor",
                )
            return task, False, None, err_msg
        except ExecutionCancelledError:
            task.status = TaskStatus.CANCELLED
            raise
        except Exception as exc:
            err_msg = str(exc)
            task.status = TaskStatus.FAILED
            task_duration = time.perf_counter() - t_start
            self._logger.error("Task failed: '%s' (%s): %s", task.id, task.action, err_msg)
            if self._planner_event_bus is not None:
                await self._planner_event_bus.publish_async(
                    TaskFailed(
                        execution_id=str(task.parameters.get("plan_id", "")),
                        plan_id=str(task.parameters.get("plan_id", "")),
                        task_id=task.id,
                        action=task.action,
                        error=err_msg,
                        duration=task_duration,
                    )
                )
            if self._event_bus is not None:
                await self._event_bus.publish_async(
                    "task.failed",
                    {"task_id": task.id, "action": task.action, "error": err_msg},
                    source="executor",
                )
            return task, False, None, err_msg

    def _handle_cancellation_sync(
        self,
        plan: Plan,
        ctrl: Optional[ExecutionController],
        completed_tasks: List[Task],
        failed_tasks: List[Task],
        skipped_tasks: List[Task],
        execution_order: List[str],
        dependency_failures: Dict[str, List[str]],
        task_output_map: Dict[str, str],
        reason: str,
    ) -> ExecutionResult:
        """Handle synchronous cancellation by marking unexecuted tasks CANCELLED and emitting events."""
        finished_ids = {t.id for t in completed_tasks} | {t.id for t in failed_tasks} | {t.id for t in skipped_tasks}
        cancelled_tasks: List[Task] = []
        for t in plan.tasks:
            if t.id not in finished_ids:
                t.status = TaskStatus.CANCELLED
                cancelled_tasks.append(t)
                task_output_map[t.id] = f"[{t.action}] CANCELLED: {reason}"

        if self._planner_event_bus is not None:
            self._planner_event_bus.publish(
                PlanCancelled(
                    execution_id=plan.id,
                    plan_id=plan.id,
                    reason=reason,
                )
            )
        if self._event_bus is not None:
            self._event_bus.publish(
                "plan.cancelled",
                {"plan_id": plan.id, "reason": reason},
                source="executor",
            )

        outputs = [task_output_map[t.id] for t in plan.tasks if t.id in task_output_map]
        all_skipped = skipped_tasks + cancelled_tasks
        return ExecutionResult(
            success=False,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            skipped_tasks=all_skipped,
            output="\n".join(outputs) if outputs else f"Plan cancelled: {reason}",
            execution_order=execution_order,
            dependency_failures=dependency_failures,
        )

    # --------------------------------------------------------------------------
    # Plan Execution (DAG Scheduling)
    # --------------------------------------------------------------------------

    def execute_plan(
        self,
        plan: Plan,
        *,
        controller: Optional[ExecutionController] = None,
        timeout_config: Optional[TimeoutConfig] = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute all tasks in the provided plan using DAG dependency scheduling.

        Args:
            plan: The Plan to execute.
            controller: Optional ExecutionController for pause/resume/cancellation.
            timeout_config: Optional TimeoutConfig for deadline enforcement.

        Returns:
            ExecutionResult summary of task executions.
        """
        ctrl = controller if controller is not None else self._controller
        t_cfg = timeout_config if timeout_config is not None else self._timeout_config
        timeout_mgr = TimeoutManager(config=t_cfg, event_bus=self._planner_event_bus, controller=ctrl) if t_cfg is not None else None

        # 1. Validate DAG before execution
        plan_start_time = time.perf_counter()
        is_valid, error_msg = self.validate_dag(plan)
        if not is_valid:
            self._logger.error("Executor DAG validation failure: %s", error_msg)
            if self._planner_event_bus is not None:
                self._planner_event_bus.publish(
                    PlanFailed(
                        execution_id=plan.id,
                        plan_id=plan.id,
                        error=error_msg or "Validation error",
                    )
                )
            if self._event_bus is not None:
                self._event_bus.publish(
                    "plan.failed",
                    {"plan_id": plan.id, "error": error_msg},
                    source="executor",
                )
            return ExecutionResult(
                success=False,
                completed_tasks=[],
                failed_tasks=[],
                output=f"Plan validation failed: {error_msg}",
                skipped_tasks=[],
                execution_order=[],
                dependency_failures={},
            )

        self._logger.info("Executor starting DAG execution for plan '%s' with %d task(s)", plan.id, len(plan.tasks))

        if self._planner_event_bus is not None:
            self._planner_event_bus.publish(
                PlanStarted(
                    execution_id=plan.id,
                    plan_id=plan.id,
                    query=plan.query,
                    task_count=len(plan.tasks),
                    strategy=plan.strategy.value if hasattr(plan.strategy, "value") else str(plan.strategy),
                )
            )

        if self._event_bus is not None:
            self._event_bus.publish(
                "plan.started",
                {"plan_id": plan.id, "query": plan.query, "task_count": len(plan.tasks)},
                source="executor",
            )

        if not plan.tasks:
            return ExecutionResult(
                success=True,
                completed_tasks=[],
                failed_tasks=[],
                output="Plan has no tasks to execute.",
                skipped_tasks=[],
                execution_order=[],
                dependency_failures={},
            )

        # 2. Build Dependency Graph
        task_map: Dict[str, Task] = {t.id: t for t in plan.tasks}
        task_order_index: Dict[str, int] = {t.id: i for i, t in enumerate(plan.tasks)}
        dependency_map: Dict[str, Set[str]] = {t.id: set(t.dependencies or []) for t in plan.tasks}
        reverse_dependency_map: Dict[str, Set[str]] = {t.id: set() for t in plan.tasks}
        for t_id, deps in dependency_map.items():
            for dep_id in deps:
                reverse_dependency_map[dep_id].add(t_id)

        in_degree: Dict[str, int] = {t_id: len(deps) for t_id, deps in dependency_map.items()}

        started_ids: Set[str] = set()
        completed_ids: Set[str] = set()
        failed_ids: Set[str] = set()
        skipped_ids: Set[str] = set()

        completed_tasks: List[Task] = []
        failed_tasks: List[Task] = []
        skipped_tasks: List[Task] = []
        execution_order: List[str] = []
        dependency_failures: Dict[str, List[str]] = {}
        task_output_map: Dict[str, str] = {}

        # 3. Wave-based concurrent execution loop
        while True:
            if timeout_mgr is not None:
                try:
                    timeout_mgr.check_total_timeout(plan_start_time, plan.id)
                except TimeoutError as exc:
                    return self._handle_cancellation_sync(
                        plan, ctrl, completed_tasks, failed_tasks, skipped_tasks,
                        execution_order, dependency_failures, task_output_map, str(exc)
                    )

            if ctrl is not None:
                try:
                    ctrl.check_checkpoint("before_wave")
                except ExecutionCancelledError as exc:
                    return self._handle_cancellation_sync(
                        plan, ctrl, completed_tasks, failed_tasks, skipped_tasks,
                        execution_order, dependency_failures, task_output_map, exc.reason
                    )

            ready_batch = [
                task_map[t_id]
                for t_id in [t.id for t in plan.tasks]
                if in_degree[t_id] == 0 and t_id not in started_ids and t_id not in skipped_ids
            ]

            if not ready_batch:
                break

            for t in ready_batch:
                self._logger.info("Task ready: '%s' (%s)", t.id, t.action)
                started_ids.add(t.id)

            # Concurrent execution of independent tasks
            try:
                if len(ready_batch) == 1:
                    batch_results = [self._execute_single_task_sync(ready_batch[0], controller=ctrl, timeout_mgr=timeout_mgr)]
                else:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(ready_batch), 16)) as pool:
                        future_to_task = [
                            pool.submit(self._execute_single_task_sync, t, controller=ctrl, timeout_mgr=timeout_mgr)
                            for t in ready_batch
                        ]
                        batch_results = [fut.result() for fut in future_to_task]
            except ExecutionCancelledError as exc:
                return self._handle_cancellation_sync(
                    plan, ctrl, completed_tasks, failed_tasks, skipped_tasks,
                    execution_order, dependency_failures, task_output_map, exc.reason
                )

            # 4. Process wave results and release/propagate dependencies
            for t, success, res_str, err_msg in batch_results:
                execution_order.append(t.id)
                if success:
                    completed_ids.add(t.id)
                    completed_tasks.append(t)
                    task_output_map[t.id] = f"[{t.action}] COMPLETED: {res_str}"

                    # Dependency Release: decrement dependent in-degrees
                    for dependent_id in sorted(reverse_dependency_map[t.id], key=lambda x: task_order_index[x]):
                        if dependent_id in skipped_ids:
                            continue
                        in_degree[dependent_id] -= 1
                        self._logger.info(
                            "Dependency released: '%s' -> '%s' (remaining in-degree: %d)",
                            t.id,
                            dependent_id,
                            in_degree[dependent_id],
                        )
                else:
                    failed_ids.add(t.id)
                    failed_tasks.append(t)
                    task_output_map[t.id] = f"[{t.action}] FAILED: {err_msg}"

                    # Failure Handling: cascade skip all reachable descendants
                    descendant_ids = self._find_descendants(t.id, reverse_dependency_map)
                    for desc_id in sorted(descendant_ids, key=lambda x: task_order_index[x]):
                        if desc_id not in skipped_ids and desc_id not in failed_ids and desc_id not in completed_ids:
                            skipped_ids.add(desc_id)
                            desc_task = task_map[desc_id]
                            desc_task.status = TaskStatus.SKIPPED
                            skipped_tasks.append(desc_task)
                            dependency_failures.setdefault(desc_id, []).append(t.id)
                            task_output_map[desc_id] = f"[{desc_task.action}] SKIPPED: Dependency failure ({t.id})"
                            self._logger.warning(
                                "Task skipped: '%s' (%s) due to failed dependency '%s'",
                                desc_id,
                                desc_task.action,
                                t.id,
                            )
                            if self._event_bus is not None:
                                self._event_bus.publish(
                                    "task.skipped",
                                    {
                                        "task_id": desc_id,
                                        "action": desc_task.action,
                                        "failed_dependency": t.id,
                                    },
                                    source="executor",
                                )

            if ctrl is not None:
                try:
                    ctrl.check_checkpoint("between_waves")
                except ExecutionCancelledError as exc:
                    return self._handle_cancellation_sync(
                        plan, ctrl, completed_tasks, failed_tasks, skipped_tasks,
                        execution_order, dependency_failures, task_output_map, exc.reason
                    )

        # Sort task collections by plan definition order for stable formatting
        completed_tasks.sort(key=lambda t: task_order_index[t.id])
        failed_tasks.sort(key=lambda t: task_order_index[t.id])
        skipped_tasks.sort(key=lambda t: task_order_index[t.id])

        outputs = [task_output_map[t.id] for t in plan.tasks if t.id in task_output_map]
        combined_output = "\n".join(outputs)
        overall_success = len(failed_tasks) == 0 and len(skipped_tasks) == 0
        plan_duration = time.perf_counter() - plan_start_time

        self._logger.info(
            "Execution finished for plan '%s': %d completed, %d failed, %d skipped",
            plan.id,
            len(completed_tasks),
            len(failed_tasks),
            len(skipped_tasks),
        )

        if self._planner_event_bus is not None:
            self._planner_event_bus.publish(
                PlanCompleted(
                    execution_id=plan.id,
                    plan_id=plan.id,
                    success=overall_success,
                    completed_count=len(completed_tasks),
                    failed_count=len(failed_tasks),
                    skipped_count=len(skipped_tasks),
                    duration=plan_duration,
                )
            )

        if self._event_bus is not None:
            self._event_bus.publish(
                "plan.completed",
                {
                    "plan_id": plan.id,
                    "success": overall_success,
                    "completed_count": len(completed_tasks),
                    "failed_count": len(failed_tasks),
                    "skipped_count": len(skipped_tasks),
                },
                source="executor",
            )

        return ExecutionResult(
            success=overall_success,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            output=combined_output,
            skipped_tasks=skipped_tasks,
            execution_order=execution_order,
            dependency_failures=dependency_failures,
        )

    async def _handle_cancellation_async(
        self,
        plan: Plan,
        ctrl: Optional[ExecutionController],
        completed_tasks: List[Task],
        failed_tasks: List[Task],
        skipped_tasks: List[Task],
        execution_order: List[str],
        dependency_failures: Dict[str, List[str]],
        task_output_map: Dict[str, str],
        reason: str,
    ) -> ExecutionResult:
        """Handle asynchronous cancellation by marking unexecuted tasks CANCELLED and emitting events."""
        finished_ids = {t.id for t in completed_tasks} | {t.id for t in failed_tasks} | {t.id for t in skipped_tasks}
        cancelled_tasks: List[Task] = []
        for t in plan.tasks:
            if t.id not in finished_ids:
                t.status = TaskStatus.CANCELLED
                cancelled_tasks.append(t)
                task_output_map[t.id] = f"[{t.action}] CANCELLED: {reason}"

        if self._planner_event_bus is not None:
            await self._planner_event_bus.publish_async(
                PlanCancelled(
                    execution_id=plan.id,
                    plan_id=plan.id,
                    reason=reason,
                )
            )
        if self._event_bus is not None:
            await self._event_bus.publish_async(
                "plan.cancelled",
                {"plan_id": plan.id, "reason": reason},
                source="executor",
            )

        outputs = [task_output_map[t.id] for t in plan.tasks if t.id in task_output_map]
        all_skipped = skipped_tasks + cancelled_tasks
        return ExecutionResult(
            success=False,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            skipped_tasks=all_skipped,
            output="\n".join(outputs) if outputs else f"Plan cancelled: {reason}",
            execution_order=execution_order,
            dependency_failures=dependency_failures,
        )

    async def execute_plan_async(
        self,
        plan: Plan,
        *,
        controller: Optional[ExecutionController] = None,
        timeout_config: Optional[TimeoutConfig] = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute all tasks in the provided plan asynchronously using DAG dependency scheduling.

        Args:
            plan: The Plan to execute.
            controller: Optional ExecutionController for pause/resume/cancellation.
            timeout_config: Optional TimeoutConfig for deadline enforcement.

        Returns:
            ExecutionResult summary of task executions.
        """
        ctrl = controller if controller is not None else self._controller
        t_cfg = timeout_config if timeout_config is not None else self._timeout_config
        timeout_mgr = TimeoutManager(config=t_cfg, event_bus=self._planner_event_bus, controller=ctrl) if t_cfg is not None else None

        # 1. Validate DAG before execution
        plan_start_time = time.perf_counter()
        is_valid, error_msg = self.validate_dag(plan)
        if not is_valid:
            self._logger.error("Executor DAG validation failure: %s", error_msg)
            if self._planner_event_bus is not None:
                await self._planner_event_bus.publish_async(
                    PlanFailed(
                        execution_id=plan.id,
                        plan_id=plan.id,
                        error=error_msg or "Validation error",
                    )
                )
            if self._event_bus is not None:
                await self._event_bus.publish_async(
                    "plan.failed",
                    {"plan_id": plan.id, "error": error_msg},
                    source="executor",
                )
            return ExecutionResult(
                success=False,
                completed_tasks=[],
                failed_tasks=[],
                output=f"Plan validation failed: {error_msg}",
                skipped_tasks=[],
                execution_order=[],
                dependency_failures={},
            )

        self._logger.info("Executor starting DAG execution for plan '%s' asynchronously with %d task(s)", plan.id, len(plan.tasks))

        if self._planner_event_bus is not None:
            await self._planner_event_bus.publish_async(
                PlanStarted(
                    execution_id=plan.id,
                    plan_id=plan.id,
                    query=plan.query,
                    task_count=len(plan.tasks),
                    strategy=plan.strategy.value if hasattr(plan.strategy, "value") else str(plan.strategy),
                )
            )

        if self._event_bus is not None:
            await self._event_bus.publish_async(
                "plan.started",
                {"plan_id": plan.id, "query": plan.query, "task_count": len(plan.tasks)},
                source="executor",
            )

        if not plan.tasks:
            return ExecutionResult(
                success=True,
                completed_tasks=[],
                failed_tasks=[],
                output="Plan has no tasks to execute.",
                skipped_tasks=[],
                execution_order=[],
                dependency_failures={},
            )

        # 2. Build Dependency Graph
        task_map: Dict[str, Task] = {t.id: t for t in plan.tasks}
        task_order_index: Dict[str, int] = {t.id: i for i, t in enumerate(plan.tasks)}
        dependency_map: Dict[str, Set[str]] = {t.id: set(t.dependencies or []) for t in plan.tasks}
        reverse_dependency_map: Dict[str, Set[str]] = {t.id: set() for t in plan.tasks}
        for t_id, deps in dependency_map.items():
            for dep_id in deps:
                reverse_dependency_map[dep_id].add(t_id)

        in_degree: Dict[str, int] = {t_id: len(deps) for t_id, deps in dependency_map.items()}

        started_ids: Set[str] = set()
        completed_ids: Set[str] = set()
        failed_ids: Set[str] = set()
        skipped_ids: Set[str] = set()

        completed_tasks: List[Task] = []
        failed_tasks: List[Task] = []
        skipped_tasks: List[Task] = []
        execution_order: List[str] = []
        dependency_failures: Dict[str, List[str]] = {}
        task_output_map: Dict[str, str] = {}

        # 3. Wave-based concurrent execution loop
        while True:
            if timeout_mgr is not None:
                try:
                    timeout_mgr.check_total_timeout(plan_start_time, plan.id)
                except TimeoutError as exc:
                    return await self._handle_cancellation_async(
                        plan, ctrl, completed_tasks, failed_tasks, skipped_tasks,
                        execution_order, dependency_failures, task_output_map, str(exc)
                    )

            if ctrl is not None:
                try:
                    await ctrl.check_checkpoint_async("before_wave")
                except ExecutionCancelledError as exc:
                    return await self._handle_cancellation_async(
                        plan, ctrl, completed_tasks, failed_tasks, skipped_tasks,
                        execution_order, dependency_failures, task_output_map, exc.reason
                    )

            ready_batch = [
                task_map[t_id]
                for t_id in [t.id for t in plan.tasks]
                if in_degree[t_id] == 0 and t_id not in started_ids and t_id not in skipped_ids
            ]

            if not ready_batch:
                break

            for t in ready_batch:
                self._logger.info("Task ready: '%s' (%s)", t.id, t.action)
                started_ids.add(t.id)

            # Concurrent execution of independent tasks
            try:
                if len(ready_batch) == 1:
                    batch_results = [await self._execute_single_task_async(ready_batch[0], controller=ctrl, timeout_mgr=timeout_mgr)]
                else:
                    raw_results = await asyncio.gather(
                        *(self._execute_single_task_async(t, controller=ctrl, timeout_mgr=timeout_mgr) for t in ready_batch),
                        return_exceptions=True,
                    )
                    batch_results = []
                    for t, res in zip(ready_batch, raw_results):
                        if isinstance(res, ExecutionCancelledError):
                            raise res
                        elif isinstance(res, Exception):
                            batch_results.append((t, False, None, str(res)))
                        else:
                            batch_results.append(res)
            except ExecutionCancelledError as exc:
                return await self._handle_cancellation_async(
                    plan, ctrl, completed_tasks, failed_tasks, skipped_tasks,
                    execution_order, dependency_failures, task_output_map, exc.reason
                )

            # 4. Process wave results and release/propagate dependencies
            for t, success, res_str, err_msg in batch_results:
                execution_order.append(t.id)
                if success:
                    completed_ids.add(t.id)
                    completed_tasks.append(t)
                    task_output_map[t.id] = f"[{t.action}] COMPLETED: {res_str}"

                    # Dependency Release: decrement dependent in-degrees
                    for dependent_id in sorted(reverse_dependency_map[t.id], key=lambda x: task_order_index[x]):
                        if dependent_id in skipped_ids:
                            continue
                        in_degree[dependent_id] -= 1
                        self._logger.info(
                            "Dependency released: '%s' -> '%s' (remaining in-degree: %d)",
                            t.id,
                            dependent_id,
                            in_degree[dependent_id],
                        )
                else:
                    failed_ids.add(t.id)
                    failed_tasks.append(t)
                    task_output_map[t.id] = f"[{t.action}] FAILED: {err_msg}"

                    # Failure Handling: cascade skip all reachable descendants
                    descendant_ids = self._find_descendants(t.id, reverse_dependency_map)
                    for desc_id in sorted(descendant_ids, key=lambda x: task_order_index[x]):
                        if desc_id not in skipped_ids and desc_id not in failed_ids and desc_id not in completed_ids:
                            skipped_ids.add(desc_id)
                            desc_task = task_map[desc_id]
                            desc_task.status = TaskStatus.SKIPPED
                            skipped_tasks.append(desc_task)
                            dependency_failures.setdefault(desc_id, []).append(t.id)
                            task_output_map[desc_id] = f"[{desc_task.action}] SKIPPED: Dependency failure ({t.id})"
                            self._logger.warning(
                                "Task skipped: '%s' (%s) due to failed dependency '%s'",
                                desc_id,
                                desc_task.action,
                                t.id,
                            )
                            if self._event_bus is not None:
                                await self._event_bus.publish_async(
                                    "task.skipped",
                                    {
                                        "task_id": desc_id,
                                        "action": desc_task.action,
                                        "failed_dependency": t.id,
                                    },
                                    source="executor",
                                )

            if ctrl is not None:
                try:
                    await ctrl.check_checkpoint_async("between_waves")
                except ExecutionCancelledError as exc:
                    return await self._handle_cancellation_async(
                        plan, ctrl, completed_tasks, failed_tasks, skipped_tasks,
                        execution_order, dependency_failures, task_output_map, exc.reason
                    )

        completed_tasks.sort(key=lambda t: task_order_index[t.id])
        failed_tasks.sort(key=lambda t: task_order_index[t.id])
        skipped_tasks.sort(key=lambda t: task_order_index[t.id])

        outputs = [task_output_map[t.id] for t in plan.tasks if t.id in task_output_map]
        combined_output = "\n".join(outputs)
        overall_success = len(failed_tasks) == 0 and len(skipped_tasks) == 0
        plan_duration = time.perf_counter() - plan_start_time

        self._logger.info(
            "Execution finished for plan '%s' asynchronously: %d completed, %d failed, %d skipped",
            plan.id,
            len(completed_tasks),
            len(failed_tasks),
            len(skipped_tasks),
        )

        if self._planner_event_bus is not None:
            await self._planner_event_bus.publish_async(
                PlanCompleted(
                    execution_id=plan.id,
                    plan_id=plan.id,
                    success=overall_success,
                    completed_count=len(completed_tasks),
                    failed_count=len(failed_tasks),
                    skipped_count=len(skipped_tasks),
                    duration=plan_duration,
                )
            )

        if self._event_bus is not None:
            await self._event_bus.publish_async(
                "plan.completed",
                {
                    "plan_id": plan.id,
                    "success": overall_success,
                    "completed_count": len(completed_tasks),
                    "failed_count": len(failed_tasks),
                    "skipped_count": len(skipped_tasks),
                },
                source="executor",
            )

        return ExecutionResult(
            success=overall_success,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            output=combined_output,
            skipped_tasks=skipped_tasks,
            execution_order=execution_order,
            dependency_failures=dependency_failures,
        )


# Global singleton instance
executor: Final[Executor] = Executor()

