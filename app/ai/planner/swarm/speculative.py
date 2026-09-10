"""Speculative Parallel Execution Engine for Phase 19.3.

Provides thread-safe and async-compatible parallel racing of candidate agents,
immediate winner selection, cancellation of losing branches, and exception isolation.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from app.ai.planner.models import Task
from app.ai.planner.multi_agent.models import DelegationRequest, DelegationResponse

logger = logging.getLogger("app.ai.planner.swarm.speculative")


class SpeculativeExecutor:
    """Races multiple candidate agents or strategies in parallel with early winner selection."""

    def __init__(self, max_workers: int = 16) -> None:
        """Initialize SpeculativeExecutor with configurable thread pool capacity.

        Args:
            max_workers: Maximum parallel worker threads for synchronous task racing.
        """
        self._max_workers = max_workers
        self._lock = threading.RLock()

    def _normalize_request(self, task_or_request: Any) -> DelegationRequest:
        """Convert various task representations into a typed DelegationRequest."""
        if isinstance(task_or_request, DelegationRequest):
            return task_or_request
        if isinstance(task_or_request, Task):
            return DelegationRequest(
                task_id=task_or_request.id,
                action=task_or_request.action,
                target=task_or_request.target,
                parameters=dict(task_or_request.parameters),
            )
        if isinstance(task_or_request, dict):
            return DelegationRequest(
                task_id=str(task_or_request.get("task_id", str(uuid.uuid4()))),
                action=str(task_or_request.get("action", "speculative_task")),
                target=task_or_request.get("target"),
                parameters=dict(task_or_request.get("parameters", {})),
            )
        return DelegationRequest(
            task_id=str(uuid.uuid4()),
            action=str(task_or_request),
        )

    def _execute_candidate(
        self,
        candidate: Any,
        request: DelegationRequest,
        cancel_event: threading.Event,
    ) -> DelegationResponse:
        """Execute a single candidate worker with cancellation checking and exception isolation."""
        start = time.perf_counter()
        agent_id = getattr(candidate, "agent_id", str(candidate))

        if cancel_event.is_set():
            return DelegationResponse(
                task_id=request.task_id,
                agent_id=agent_id,
                success=False,
                error="Cancelled before execution.",
                duration=0.0,
            )

        try:
            # Check if candidate has execute_delegation method (BaseAgent)
            if hasattr(candidate, "execute_delegation"):
                res = candidate.execute_delegation(request)
            elif callable(candidate):
                # Check if callable accepts cancel_event or request
                try:
                    res = candidate(request, cancel_event)
                except TypeError:
                    try:
                        res = candidate(request)
                    except TypeError:
                        res = candidate()
            else:
                res = str(candidate)

            elapsed = time.perf_counter() - start

            if isinstance(res, DelegationResponse):
                return res

            return DelegationResponse(
                task_id=request.task_id,
                agent_id=agent_id,
                success=True,
                output=res,
                duration=elapsed,
            )

        except Exception as exc:
            elapsed = time.perf_counter() - start
            logger.warning(
                "Candidate '%s' failed during speculative race: %s", agent_id, exc
            )
            return DelegationResponse(
                task_id=request.task_id,
                agent_id=agent_id,
                success=False,
                error=str(exc),
                duration=elapsed,
                confidence=0.0,
            )

    def race_tasks(
        self,
        task_or_request: Any,
        candidates: List[Any],
        evaluation_fn: Optional[Callable[[DelegationResponse], bool]] = None,
        timeout: float = 10.0,
    ) -> DelegationResponse:
        """Synchronously race candidate agents in parallel and select the fastest acceptable winner.

        Args:
            task_or_request: Target task or DelegationRequest.
            candidates: List of candidate agents or callable workers.
            evaluation_fn: Optional validation callback to accept a response.
            timeout: Maximum race deadline in seconds.

        Returns:
            Winning DelegationResponse, or failure response if none succeed.
        """
        if not candidates:
            return DelegationResponse(
                task_id=str(uuid.uuid4()),
                agent_id="speculative_executor",
                success=False,
                error="No candidates provided for speculative execution.",
            )

        request = self._normalize_request(task_or_request)
        cancel_event = threading.Event()
        start_time = time.perf_counter()

        winner: Optional[DelegationResponse] = None
        acceptable_results: List[DelegationResponse] = []
        timed_out = False

        pool_size = min(len(candidates), self._max_workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as executor:
            future_to_candidate = {
                executor.submit(
                    self._execute_candidate, cand, request, cancel_event
                ): cand
                for cand in candidates
            }

            try:
                for future in concurrent.futures.as_completed(
                    future_to_candidate, timeout=timeout
                ):
                    try:
                        res = future.result()
                    except Exception as exc:
                        logger.warning("Future raised exception during race: %s", exc)
                        continue

                    # Check acceptability
                    is_valid = (
                        evaluation_fn(res)
                        if evaluation_fn is not None
                        else (res.success and res.error is None)
                    )

                    if is_valid:
                        acceptable_results.append(res)
                        # Immediate winner selection
                        winner = res
                        cancel_event.set()
                        break

            except concurrent.futures.TimeoutError:
                timed_out = True
                cancel_event.set()
                logger.warning(
                    "Speculative execution timed out after %.2fs for task '%s'",
                    timeout,
                    request.task_id,
                )

        if winner is not None:
            return winner

        # If race ended without a single acceptable winner
        elapsed = time.perf_counter() - start_time
        err_msg = (
            f"Speculative execution timed out after {timeout:.2f}s (elapsed: {elapsed:.3f}s)."
            if timed_out
            else f"Speculative execution finished without acceptable winner (elapsed: {elapsed:.3f}s)."
        )
        return DelegationResponse(
            task_id=request.task_id,
            agent_id="speculative_executor",
            success=False,
            error=err_msg,
            duration=elapsed,
            confidence=0.0,
        )

    async def race_tasks_async(
        self,
        task_or_request: Any,
        candidates: List[Any],
        evaluation_fn: Optional[Callable[[DelegationResponse], bool]] = None,
        timeout: float = 10.0,
    ) -> DelegationResponse:
        """Asynchronously race candidate agents with immediate cancellation of losers.

        Args:
            task_or_request: Target task or DelegationRequest.
            candidates: List of candidate agents or async callables.
            evaluation_fn: Optional validation callback to accept a response.
            timeout: Maximum race deadline in seconds.

        Returns:
            Winning DelegationResponse, or failure response if none succeed.
        """
        if not candidates:
            return DelegationResponse(
                task_id=str(uuid.uuid4()),
                agent_id="speculative_executor",
                success=False,
                error="No candidates provided for speculative execution.",
            )

        request = self._normalize_request(task_or_request)
        start_time = time.perf_counter()

        async def _run_async_candidate(cand: Any) -> DelegationResponse:
            agent_id = getattr(cand, "agent_id", str(cand))
            s = time.perf_counter()
            try:
                if hasattr(cand, "execute_delegation_async"):
                    res = await cand.execute_delegation_async(request)
                elif hasattr(cand, "execute_delegation"):
                    res = cand.execute_delegation(request)
                elif callable(cand):
                    r = cand(request)
                    res = await r if asyncio.iscoroutine(r) else r
                else:
                    res = str(cand)

                dur = time.perf_counter() - s
                if isinstance(res, DelegationResponse):
                    return res
                return DelegationResponse(
                    task_id=request.task_id,
                    agent_id=agent_id,
                    success=True,
                    output=res,
                    duration=dur,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                dur = time.perf_counter() - s
                return DelegationResponse(
                    task_id=request.task_id,
                    agent_id=agent_id,
                    success=False,
                    error=str(exc),
                    duration=dur,
                    confidence=0.0,
                )

        tasks: Dict[asyncio.Task, Any] = {}
        for cand in candidates:
            t = asyncio.create_task(_run_async_candidate(cand))
            tasks[t] = cand

        winner: Optional[DelegationResponse] = None
        pending = set(tasks.keys())

        try:
            deadline = time.perf_counter() + timeout
            while pending:
                remaining_time = deadline - time.perf_counter()
                if remaining_time <= 0:
                    break

                done, pending = await asyncio.wait(
                    pending,
                    timeout=remaining_time,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Evaluate completed tasks in deterministic order (by agent_id on ties)
                completed_list = list(done)
                completed_list.sort(
                    key=lambda t: str(getattr(tasks.get(t), "agent_id", str(tasks.get(t))))
                )

                for t in completed_list:
                    if t.cancelled():
                        continue
                    exc = t.exception()
                    if exc is not None:
                        continue
                    res = t.result()
                    is_valid = (
                        evaluation_fn(res)
                        if evaluation_fn is not None
                        else (res.success and res.error is None)
                    )
                    if is_valid:
                        winner = res
                        break

                if winner is not None:
                    break

        finally:
            # Cancel all remaining losing branches immediately
            for t in pending:
                if not t.done():
                    t.cancel()

        if winner is not None:
            return winner

        elapsed = time.perf_counter() - start_time
        return DelegationResponse(
            task_id=request.task_id,
            agent_id="speculative_executor",
            success=False,
            error=f"Async speculative execution timed out or produced no valid result after {elapsed:.3f}s.",
            duration=elapsed,
            confidence=0.0,
        )
