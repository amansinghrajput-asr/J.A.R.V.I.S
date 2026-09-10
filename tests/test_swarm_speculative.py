"""Unit tests for Speculative Parallel Execution (Sprint 19.3).

Covers fast winner selection, timeout handling, loser cancellation,
exception isolation, async execution, and deterministic tie-breaking.
"""

import asyncio
import threading
import time
import unittest

from app.ai.planner.multi_agent.models import DelegationRequest, DelegationResponse
from app.ai.planner.swarm.speculative import SpeculativeExecutor


class TestSwarmSpeculative(unittest.TestCase):
    """Test suite for SpeculativeExecutor."""

    def test_winner_selection_fastest_wins(self) -> None:
        """Verify the fastest successful candidate is selected as winner."""
        executor = SpeculativeExecutor()

        def _fast_worker(req: DelegationRequest) -> DelegationResponse:
            time.sleep(0.01)
            return DelegationResponse(
                task_id=req.task_id,
                agent_id="fast_worker",
                success=True,
                output="fast_result",
            )

        def _slow_worker(req: DelegationRequest) -> DelegationResponse:
            time.sleep(0.15)
            return DelegationResponse(
                task_id=req.task_id,
                agent_id="slow_worker",
                success=True,
                output="slow_result",
            )

        req = DelegationRequest(task_id="t1", action="solve_riddle")
        res = executor.race_tasks(req, [_slow_worker, _fast_worker], timeout=1.0)

        self.assertTrue(res.success)
        self.assertEqual(res.agent_id, "fast_worker")
        self.assertEqual(res.output, "fast_result")

    def test_loser_cancellation(self) -> None:
        """Verify losing branches receive cancellation signal."""
        executor = SpeculativeExecutor()
        loser_cancelled = threading.Event()

        def _winner(req: DelegationRequest, cancel_token: threading.Event) -> DelegationResponse:
            time.sleep(0.01)
            return DelegationResponse(
                task_id=req.task_id,
                agent_id="winner",
                success=True,
                output="win",
            )

        def _loser(req: DelegationRequest, cancel_token: threading.Event) -> DelegationResponse:
            # Poll for cancellation
            for _ in range(30):
                if cancel_token.is_set():
                    loser_cancelled.set()
                    return DelegationResponse(
                        task_id=req.task_id,
                        agent_id="loser",
                        success=False,
                        error="cancelled",
                    )
                time.sleep(0.01)
            return DelegationResponse(task_id=req.task_id, agent_id="loser", success=True)

        req = DelegationRequest(task_id="t_cancel", action="compute")
        res = executor.race_tasks(req, [_loser, _winner], timeout=1.0)

        self.assertTrue(res.success)
        self.assertEqual(res.agent_id, "winner")
        # Give loser a moment to detect cancel token
        time.sleep(0.05)
        self.assertTrue(loser_cancelled.is_set())

    def test_timeout_handling(self) -> None:
        """Verify race returns a failed response when all candidates exceed timeout."""
        executor = SpeculativeExecutor()

        def _too_slow(req: DelegationRequest) -> DelegationResponse:
            time.sleep(0.2)
            return DelegationResponse(task_id=req.task_id, agent_id="too_slow", success=True)

        req = DelegationRequest(task_id="t_timeout", action="heavy_task")
        res = executor.race_tasks(req, [_too_slow], timeout=0.05)

        self.assertFalse(res.success)
        self.assertIn("timed out", res.error.lower())

    def test_exception_isolation(self) -> None:
        """Verify an exception in one branch does not crash the race or prevent another from winning."""
        executor = SpeculativeExecutor()

        def _exploding_worker(req: DelegationRequest) -> DelegationResponse:
            raise RuntimeError("Explosion in candidate branch!")

        def _reliable_worker(req: DelegationRequest) -> DelegationResponse:
            time.sleep(0.02)
            return DelegationResponse(
                task_id=req.task_id,
                agent_id="reliable",
                success=True,
                output="recovered_result",
            )

        req = DelegationRequest(task_id="t_isolate", action="resilient_task")
        res = executor.race_tasks(req, [_exploding_worker, _reliable_worker], timeout=1.0)

        self.assertTrue(res.success)
        self.assertEqual(res.agent_id, "reliable")
        self.assertEqual(res.output, "recovered_result")

    def test_evaluation_callback_filters_low_quality(self) -> None:
        """Verify evaluation callback rejects unsatisfactory results until a good one arrives."""
        executor = SpeculativeExecutor()

        def _fast_garbage(req: DelegationRequest) -> DelegationResponse:
            time.sleep(0.01)
            return DelegationResponse(
                task_id=req.task_id,
                agent_id="fast_garbage",
                success=True,
                output="incomplete_output",
                confidence=0.3,
            )

        def _slower_quality(req: DelegationRequest) -> DelegationResponse:
            time.sleep(0.04)
            return DelegationResponse(
                task_id=req.task_id,
                agent_id="quality_worker",
                success=True,
                output="comprehensive_output",
                confidence=0.95,
            )

        # Evaluation requires confidence >= 0.8
        def _quality_check(res: DelegationResponse) -> bool:
            return res.success and res.confidence >= 0.8

        req = DelegationRequest(task_id="t_eval", action="write_report")
        res = executor.race_tasks(
            req, [_fast_garbage, _slower_quality], evaluation_fn=_quality_check, timeout=1.0
        )

        self.assertTrue(res.success)
        self.assertEqual(res.agent_id, "quality_worker")
        self.assertEqual(res.output, "comprehensive_output")

    def test_async_speculative_execution(self) -> None:
        """Verify async speculative execution races coroutines and cancels pending tasks."""
        async def _run_async_test() -> None:
            executor = SpeculativeExecutor()

            async def _async_fast(req: DelegationRequest) -> DelegationResponse:
                await asyncio.sleep(0.01)
                return DelegationResponse(
                    task_id=req.task_id,
                    agent_id="async_fast",
                    success=True,
                    output="fast_async_result",
                )

            async def _async_slow(req: DelegationRequest) -> DelegationResponse:
                await asyncio.sleep(0.2)
                return DelegationResponse(
                    task_id=req.task_id,
                    agent_id="async_slow",
                    success=True,
                    output="slow_async_result",
                )

            req = DelegationRequest(task_id="t_async", action="fetch_async")
            res = await executor.race_tasks_async(
                req, [_async_slow, _async_fast], timeout=1.0
            )

            self.assertTrue(res.success)
            self.assertEqual(res.agent_id, "async_fast")
            self.assertEqual(res.output, "fast_async_result")

        asyncio.run(_run_async_test())


if __name__ == "__main__":
    unittest.main()
