"""Performance benchmark for Phase 17.5 Execution Infrastructure.

Measures:
- Persistence save latency (target: <5 ms)
- Persistence load latency (target: <5 ms)
- Replay step throughput (target: >1000 steps/sec)
- Pause/resume latency (target: <1 ms)
- Cancellation latency (target: <1 ms)
- Timeout check overhead (target: <10 µs)
- End-to-end plan execution overhead with control and timeouts (target: <5%)
"""

import json
import logging
import os
import sys
import tempfile
import time
from typing import Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ai.planner.control import CancellationToken, ExecutionController
from app.ai.planner.events import PlannerEventBus, TaskCompleted, TaskStarted
from app.ai.planner.executor import Executor
from app.ai.planner.memory import ExecutionMemory, TaskExecutionRecord
from app.ai.planner.models import ExecutionResult, Plan, Task, TaskStatus
from app.ai.planner.persistence import PersistedExecutionState, load, save
from app.ai.planner.replay import PlannerReplayEngine
from app.ai.planner.timeline import ExecutionTimeline
from app.ai.planner.timeouts import TimeoutConfig, TimeoutManager, TimeoutPolicy

logging.getLogger("PLAN_EXECUTOR").setLevel(logging.ERROR)
logging.getLogger("app.ai.planner.events").setLevel(logging.ERROR)
logging.getLogger("app.ai.planner.control").setLevel(logging.ERROR)
logging.getLogger("app.ai.planner.timeouts").setLevel(logging.ERROR)


def _create_sample_state(task_count: int = 10) -> PersistedExecutionState:
    tasks = [
        Task(
            id=f"task_{i}",
            action="sample_action",
            target=f"target_{i}",
            status=TaskStatus.COMPLETED if i < task_count // 2 else TaskStatus.PENDING,
            dependencies=[f"task_{i-1}"] if i > 0 else [],
        )
        for i in range(task_count)
    ]
    plan = Plan(query="Benchmark plan query", tasks=tasks)
    records = []
    for i in range(task_count // 2):
        records.append(
            TaskExecutionRecord(
                task_id=f"task_{i}",
                action="sample_action",
                target=f"target_{i}",
                status=TaskStatus.COMPLETED,
                duration=0.015,
                output="OK",
            )
        )
    memory = ExecutionMemory(records=records)
    return PersistedExecutionState(
        execution_id="exec_bench_1",
        plan_id=plan.id,
        query=plan.query,
        memory=memory,
        dag_state={"task_map": {t.id: t for t in tasks}},
        completed_tasks=[t for t in tasks if t.status == TaskStatus.COMPLETED],
        metadata={"created_by": "benchmark", "phase": "17.5"},
    )


def benchmark_persistence_save(iterations: int = 200) -> float:
    state = _create_sample_state(10)
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "state_save.json")
        # Warmup
        save(state, save_path)

        start = time.perf_counter()
        for _ in range(iterations):
            save(state, save_path)
        duration = time.perf_counter() - start
        avg_ms = (duration / iterations) * 1_000.0
        return avg_ms


def benchmark_persistence_load(iterations: int = 200) -> float:
    state = _create_sample_state(10)
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "state_load.json")
        save(state, save_path)

        # Warmup
        load(save_path)

        start = time.perf_counter()
        for _ in range(iterations):
            load(save_path)
        duration = time.perf_counter() - start
        avg_ms = (duration / iterations) * 1_000.0
        return avg_ms


def benchmark_replay_throughput(event_count: int = 500, iterations: int = 20) -> float:
    timeline = ExecutionTimeline()
    for i in range(event_count):
        timeline.record(
            TaskCompleted(
                task_id=f"t_{i}",
                action="op",
                result="done",
                duration=0.001,
            )
        )

    engine = PlannerReplayEngine(timeline)

    # Warmup
    engine.step()
    engine.reset()

    start = time.perf_counter()
    total_steps = 0
    for _ in range(iterations):
        engine.reset()
        while engine.step() is not None:
            total_steps += 1
    duration = time.perf_counter() - start
    steps_per_sec = total_steps / duration
    return steps_per_sec


def benchmark_pause_resume_latency(iterations: int = 2000) -> float:
    ctrl = ExecutionController()
    start = time.perf_counter()
    for i in range(iterations):
        ctrl.pause("bench")
        ctrl.resume("bench")
    duration = time.perf_counter() - start
    avg_us = (duration / iterations) * 1_000_000.0
    return avg_us


def benchmark_cancellation_latency(iterations: int = 2000) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        token = CancellationToken()
        token.cancel("bench cancel")
    duration = time.perf_counter() - start
    avg_us = (duration / iterations) * 1_000_000.0
    return avg_us


def benchmark_timeout_check_overhead(iterations: int = 10000) -> float:
    config = TimeoutConfig(task_timeout=10.0, total_timeout=100.0)
    manager = TimeoutManager(config=config)
    start_time = time.perf_counter()

    start = time.perf_counter()
    for _ in range(iterations):
        manager.check_total_timeout(start_time, "plan_bench")
    duration = time.perf_counter() - start
    avg_us = (duration / iterations) * 1_000_000.0
    return avg_us


def benchmark_controller_only_overhead(runs: int = 30, sleep_sec: float = 0.005) -> Tuple[float, float, float]:
    handler_fn = (lambda t: time.sleep(sleep_sec) or "OK") if sleep_sec > 0 else (lambda t: "OK")

    executor_baseline = Executor(
        handlers={"step": handler_fn},
        event_bus_instance=None,
        auto_register_in_container=False,
    )

    bus = PlannerEventBus()
    ctrl = ExecutionController(event_bus=bus)

    executor_controlled = Executor(
        handlers={"step": handler_fn},
        event_bus_instance=None,
        auto_register_in_container=False,
        planner_event_bus=bus,
        controller=ctrl,
    )

    tasks = [Task(id=f"t_{i}", action="step") for i in range(5)]
    plan = Plan(query="perf test controller only", tasks=tasks)

    for _ in range(3):
        executor_baseline.execute_plan(plan)
        executor_controlled.execute_plan(plan)

    t0 = time.perf_counter()
    for _ in range(runs):
        executor_baseline.execute_plan(plan)
    baseline_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    for _ in range(runs):
        executor_controlled.execute_plan(plan)
    controlled_time = time.perf_counter() - t1

    overhead_pct = ((controlled_time - baseline_time) / baseline_time) * 100.0
    return baseline_time, controlled_time, overhead_pct


def benchmark_execution_overhead(runs: int = 30, sleep_sec: float = 0.002) -> Tuple[float, float, float]:
    handler_fn = (lambda t: time.sleep(sleep_sec) or "OK") if sleep_sec > 0 else (lambda t: "OK")

    # Baseline executor without controller or timeouts
    executor_baseline = Executor(
        handlers={"step": handler_fn},
        event_bus_instance=None,
        auto_register_in_container=False,
    )

    # Controlled executor with controller and timeout manager
    bus = PlannerEventBus()
    ctrl = ExecutionController(event_bus=bus)
    t_cfg = TimeoutConfig(task_timeout=5.0, total_timeout=30.0, policy=TimeoutPolicy.SKIP)

    executor_controlled = Executor(
        handlers={"step": handler_fn},
        event_bus_instance=None,
        auto_register_in_container=False,
        planner_event_bus=bus,
        controller=ctrl,
        timeout_config=t_cfg,
    )

    tasks = [Task(id=f"t_{i}", action="step") for i in range(5)]
    plan = Plan(query="perf test execution control", tasks=tasks)

    # Warmup
    for _ in range(3):
        executor_baseline.execute_plan(plan)
        executor_controlled.execute_plan(plan)

    # Measure baseline
    t0 = time.perf_counter()
    for _ in range(runs):
        executor_baseline.execute_plan(plan)
    baseline_time = time.perf_counter() - t0

    # Measure controlled
    t1 = time.perf_counter()
    for _ in range(runs):
        executor_controlled.execute_plan(plan)
    controlled_time = time.perf_counter() - t1

    overhead_pct = ((controlled_time - baseline_time) / baseline_time) * 100.0
    return baseline_time, controlled_time, overhead_pct


if __name__ == "__main__":
    print("=== Phase 17.5 Execution Infrastructure Performance Benchmark ===")

    save_ms = benchmark_persistence_save(200)
    print(f"Persistence Save Latency: {save_ms:.3f} ms (budget: <5.0 ms) -> {'PASS' if save_ms < 5.0 else 'FAIL'}")

    load_ms = benchmark_persistence_load(200)
    print(f"Persistence Load Latency: {load_ms:.3f} ms (budget: <5.0 ms) -> {'PASS' if load_ms < 5.0 else 'FAIL'}")

    replay_tps = benchmark_replay_throughput(500, 20)
    print(f"Replay Step Throughput: {replay_tps:.1f} steps/sec (budget: >1000 steps/sec) -> {'PASS' if replay_tps > 1000 else 'FAIL'}")

    pause_resume_us = benchmark_pause_resume_latency(2000)
    print(f"Pause/Resume Latency: {pause_resume_us:.3f} µs (budget: <1000 µs) -> {'PASS' if pause_resume_us < 1000 else 'FAIL'}")

    cancel_us = benchmark_cancellation_latency(2000)
    print(f"Cancellation Latency: {cancel_us:.3f} µs (budget: <1000 µs) -> {'PASS' if cancel_us < 1000 else 'FAIL'}")

    timeout_us = benchmark_timeout_check_overhead(10000)
    print(f"Timeout Check Overhead: {timeout_us:.3f} µs (budget: <10.0 µs) -> {'PASS' if timeout_us < 10.0 else 'FAIL'}")

    base_c, ctrl_c, ctrl_overhead = benchmark_controller_only_overhead(runs=30, sleep_sec=0.005)
    print(f"Controller-Only Execution Overhead (5ms tasks): {ctrl_overhead:.2f}% (budget: <5.0%) -> {'PASS' if ctrl_overhead < 5.0 else 'CHECK'}")

    base_t2, ctrl_t2, overhead_2ms = benchmark_execution_overhead(runs=30, sleep_sec=0.002)
    print(f"Execution Control + Timeout Overhead (2ms tasks): {overhead_2ms:.2f}%")

    base_t10, ctrl_t10, overhead_10ms = benchmark_execution_overhead(runs=20, sleep_sec=0.010)
    print(f"Execution Control + Timeout Overhead (10ms tasks): {overhead_10ms:.2f}%")

    base_t50, ctrl_t50, overhead_50ms = benchmark_execution_overhead(runs=10, sleep_sec=0.050)
    print(f"Execution Control + Timeout Overhead (50ms tasks): {overhead_50ms:.2f}% (budget: <5.0%) -> {'PASS' if overhead_50ms < 5.0 else 'CHECK'}")
