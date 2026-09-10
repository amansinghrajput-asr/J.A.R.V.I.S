"""Timeout Management Subsystem for J.A.R.V.I.S. Adaptive Planner.

Enforces configurable execution deadlines on individual tasks, recovery waves,
and overall plans across sync and async contexts, with policy-based error handling.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, Optional, Union

from app.ai.planner.events import PlannerEventBus, TaskRetried, TaskTimeout

logger = logging.getLogger("app.ai.planner.timeouts")


class TimeoutPolicy(str, Enum):
    """Policies governing handling of tasks that exceed execution deadlines."""

    RETRY = "RETRY"
    SKIP = "SKIP"
    ABORT = "ABORT"
    ESCALATE = "ESCALATE"


class TaskTimeoutError(Exception):
    """Raised when a task exceeds its configured execution deadline."""

    def __init__(self, task_id: str, timeout_seconds: float, policy: TimeoutPolicy) -> None:
        self.task_id = task_id
        self.timeout_seconds = timeout_seconds
        self.policy = policy
        super().__init__(f"Task '{task_id}' timed out after {timeout_seconds:.2f}s (Policy: {policy.value})")


@dataclass
class TimeoutConfig:
    """Configuration governing timeout thresholds and resolution policies.

    Attributes:
        task_timeout: Default deadline per task in seconds (None for unlimited).
        recovery_timeout: Deadline per recovery wave in seconds (None for unlimited).
        total_timeout: Overall execution deadline for plan in seconds (None for unlimited).
        policy: Action to take upon task timeout (RETRY, SKIP, ABORT, ESCALATE).
        max_retries: Number of retry attempts when policy is RETRY.
        action_timeouts: Granular override mapping of action names to timeout seconds.
    """

    task_timeout: Optional[float] = None
    recovery_timeout: Optional[float] = None
    total_timeout: Optional[float] = None
    policy: TimeoutPolicy = TimeoutPolicy.SKIP
    max_retries: int = 1
    action_timeouts: Dict[str, float] = field(default_factory=dict)

    def get_task_timeout(self, action: Optional[str] = None) -> Optional[float]:
        """Resolve effective timeout for a given task action."""
        if action and action in self.action_timeouts:
            return self.action_timeouts[action]
        return self.task_timeout

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary."""
        return {
            "task_timeout": self.task_timeout,
            "recovery_timeout": self.recovery_timeout,
            "total_timeout": self.total_timeout,
            "policy": self.policy.value if hasattr(self.policy, "value") else str(self.policy),
            "max_retries": self.max_retries,
            "action_timeouts": dict(self.action_timeouts),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TimeoutConfig:
        """Construct TimeoutConfig from dictionary."""
        policy_raw = data.get("policy", TimeoutPolicy.SKIP.value)
        try:
            policy = TimeoutPolicy(policy_raw)
        except Exception:
            policy = TimeoutPolicy.SKIP

        return cls(
            task_timeout=data.get("task_timeout"),
            recovery_timeout=data.get("recovery_timeout"),
            total_timeout=data.get("total_timeout"),
            policy=policy,
            max_retries=int(data.get("max_retries", 1)),
            action_timeouts=dict(data.get("action_timeouts", {})),
        )


class TimeoutManager:
    """Manages execution timeouts, policy resolution, and event emission.

    Coordinates sync thread-level timeouts and async asyncio.wait_for deadlines.
    """

    def __init__(
        self,
        config: Optional[TimeoutConfig] = None,
        event_bus: Optional[PlannerEventBus] = None,
        controller: Optional[Any] = None,
    ) -> None:
        """Initialize TimeoutManager.

        Args:
            config: Optional TimeoutConfig. Defaults to default TimeoutConfig().
            event_bus: Optional PlannerEventBus to publish timeout events.
            controller: Optional ExecutionController for triggering aborts.
        """
        self.config = config or TimeoutConfig()
        self.event_bus = event_bus
        self.controller = controller
        self._thread_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._pool_lock = threading.Lock()

    def _get_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        if self._thread_pool is None:
            with self._pool_lock:
                if self._thread_pool is None:
                    self._thread_pool = concurrent.futures.ThreadPoolExecutor(
                        max_workers=16,
                        thread_name_prefix="timeout_worker",
                    )
        return self._thread_pool

    def close(self) -> None:
        """Shut down the internal thread pool executor."""
        with self._pool_lock:
            if self._thread_pool is not None:
                self._thread_pool.shutdown(wait=False)
                self._thread_pool = None

    def __del__(self) -> None:
        self.close()

    def run_sync(
        self,
        task_id: str,
        action: str,
        func: Callable[[], Any],
        plan_id: str = "",
        execution_id: str = "",
    ) -> Any:
        """Execute a callable synchronously under configured timeout and policy rules.

        Args:
            task_id: Identifier of the task being executed.
            action: Action name for granular timeout resolution.
            func: Zero-argument callable to invoke.
            plan_id: Correlation plan ID.
            execution_id: Correlation execution ID.

        Returns:
            Result of the func invocation.

        Raises:
            TaskTimeoutError: If execution times out and policy is ESCALATE or SKIP.
            ExecutionCancelledError: If policy is ABORT.
        """
        timeout = self.config.get_task_timeout(action)
        if timeout is None or timeout <= 0:
            return func()

        attempts = 0
        max_attempts = (self.config.max_retries + 1) if self.config.policy == TimeoutPolicy.RETRY else 1

        while attempts < max_attempts:
            attempts += 1
            executor = self._get_executor()
            future = executor.submit(func)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                    self._emit_timeout_event(task_id, timeout, plan_id, execution_id)
                    logger.warning(
                        "Task '%s' (%s) timed out after %.2fs (Attempt %d/%d, Policy: %s)",
                        task_id,
                        action,
                        timeout,
                        attempts,
                        max_attempts,
                        self.config.policy.value,
                    )

                    if self.config.policy == TimeoutPolicy.RETRY and attempts < max_attempts:
                        self._emit_retry_event(task_id, action, attempts + 1, "Task timeout", plan_id, execution_id)
                        continue

                    return self._handle_timeout_policy(task_id, timeout)

    async def run_async(
        self,
        task_id: str,
        action: str,
        coro_factory: Callable[[], Coroutine[Any, Any, Any]],
        plan_id: str = "",
        execution_id: str = "",
    ) -> Any:
        """Execute a coroutine asynchronously under configured timeout and policy rules.

        Args:
            task_id: Identifier of the task.
            action: Action name for granular timeout resolution.
            coro_factory: Zero-argument factory producing a new coroutine per attempt.
            plan_id: Correlation plan ID.
            execution_id: Correlation execution ID.

        Returns:
            Result of the awaited coroutine.
        """
        timeout = self.config.get_task_timeout(action)
        if timeout is None or timeout <= 0:
            return await coro_factory()

        attempts = 0
        max_attempts = (self.config.max_retries + 1) if self.config.policy == TimeoutPolicy.RETRY else 1

        while attempts < max_attempts:
            attempts += 1
            try:
                return await asyncio.wait_for(coro_factory(), timeout=timeout)
            except asyncio.TimeoutError:
                self._emit_timeout_event(task_id, timeout, plan_id, execution_id)
                logger.warning(
                    "Async task '%s' (%s) timed out after %.2fs (Attempt %d/%d, Policy: %s)",
                    task_id,
                    action,
                    timeout,
                    attempts,
                    max_attempts,
                    self.config.policy.value,
                )

                if self.config.policy == TimeoutPolicy.RETRY and attempts < max_attempts:
                    self._emit_retry_event(task_id, action, attempts + 1, "Task timeout", plan_id, execution_id)
                    continue

                return self._handle_timeout_policy(task_id, timeout)

    def check_total_timeout(self, start_time: float, plan_id: str = "") -> None:
        """Check if overall plan execution has exceeded total configured deadline."""
        if self.config.total_timeout and self.config.total_timeout > 0:
            elapsed = time.perf_counter() - start_time
            if elapsed > self.config.total_timeout:
                msg = f"Overall plan execution exceeded total timeout of {self.config.total_timeout:.2f}s (elapsed: {elapsed:.2f}s)"
                logger.error(msg)
                if self.controller is not None:
                    self.controller.cancel(msg)
                raise TimeoutError(msg)

    # --------------------------------------------------------------------------
    # Policy Enforcement & Event Emission
    # --------------------------------------------------------------------------

    def _handle_timeout_policy(self, task_id: str, timeout: float) -> Any:
        """Apply the configured TimeoutPolicy when deadlines are breached."""
        policy = self.config.policy

        if policy == TimeoutPolicy.ABORT:
            if self.controller is not None:
                self.controller.cancel(f"Task '{task_id}' timed out under ABORT policy.")
            raise TaskTimeoutError(task_id, timeout, policy)

        if policy == TimeoutPolicy.ESCALATE:
            raise TimeoutError(f"Task '{task_id}' escalated after exceeding {timeout:.2f}s deadline.")

        # Default / SKIP / RETRY exhausted: raise TaskTimeoutError for executor to record as skipped/failed
        raise TaskTimeoutError(task_id, timeout, policy)

    def _emit_timeout_event(self, task_id: str, timeout: float, plan_id: str, execution_id: str) -> None:
        """Publish TaskTimeout event through PlannerEventBus."""
        if self.event_bus is not None:
            self.event_bus.publish(
                TaskTimeout(
                    execution_id=execution_id or plan_id,
                    plan_id=plan_id,
                    task_id=task_id,
                    timeout_seconds=timeout,
                    policy=self.config.policy.value,
                )
            )

    def _emit_retry_event(
        self,
        task_id: str,
        action: str,
        attempt: int,
        reason: str,
        plan_id: str,
        execution_id: str,
    ) -> None:
        """Publish TaskRetried event through PlannerEventBus."""
        if self.event_bus is not None:
            self.event_bus.publish(
                TaskRetried(
                    execution_id=execution_id or plan_id,
                    plan_id=plan_id,
                    task_id=task_id,
                    action=action,
                    attempt=attempt,
                    reason=reason,
                )
            )
