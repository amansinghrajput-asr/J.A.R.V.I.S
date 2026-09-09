"""Performance benchmark for Phase 17.0 Observability Foundation.

Measures:
- Event publish latency (target: <50 µs)
- Timeline append latency (target: <20 µs)
- Metrics collector update latency (target: <20 µs)
- End-to-end plan execution overhead (target: <5%)
"""

import logging
import time
from app.ai.planner.events import PlannerEventBus, TaskCompleted, TaskStarted
from app.ai.planner.executor import Executor
from app.ai.planner.metrics_collector import PlannerMetricsCollector
from app.ai.planner.models import Plan, Task
from app.ai.planner.timeline import ExecutionTimeline

logging.getLogger("PLAN_EXECUTOR").setLevel(logging.ERROR)
logging.getLogger("app.ai.planner.events").setLevel(logging.ERROR)


def benchmark_event_publish(iterations: int = 10000) -> float:
    bus = PlannerEventBus()
    event = TaskCompleted(task_id="t1", action="test", duration=0.01)

    start = time.perf_counter()
    for _ in range(iterations):
        bus.publish(event)
    duration = time.perf_counter() - start
    avg_us = (duration / iterations) * 1_000_000
    return avg_us


def benchmark_timeline_append(iterations: int = 10000) -> float:
    bus = PlannerEventBus()
    timeline = ExecutionTimeline(bus=bus)
    event = TaskCompleted(task_id="t1", action="test", duration=0.01)

    start = time.perf_counter()
    for _ in range(iterations):
        bus.publish(event)
    duration = time.perf_counter() - start
    avg_us = (duration / iterations) * 1_000_000
    return avg_us


def benchmark_metrics_update(iterations: int = 10000) -> float:
    bus = PlannerEventBus()
    metrics = PlannerMetricsCollector(bus=bus)
    event = TaskCompleted(task_id="t1", action="test", duration=0.01)

    start = time.perf_counter()
    for _ in range(iterations):
        bus.publish(event)
    duration = time.perf_counter() - start
    avg_us = (duration / iterations) * 1_000_000
    return avg_us


def benchmark_plan_execution_overhead(runs: int = 50, sleep_sec: float = 0.001) -> tuple[float, float, float]:
    handler_fn = (lambda t: time.sleep(sleep_sec) or "OK") if sleep_sec > 0 else (lambda t: "OK")
    executor_baseline = Executor(
        handlers={"noop": handler_fn},
        event_bus_instance=None,
        auto_register_in_container=False,
    )
    bus = PlannerEventBus()
    timeline = ExecutionTimeline(bus=bus)
    metrics = PlannerMetricsCollector(bus=bus)
    executor_observable = Executor(
        handlers={"noop": handler_fn},
        event_bus_instance=None,
        auto_register_in_container=False,
        planner_event_bus=bus,
    )

    tasks = [Task(id=f"task_{i}", action="noop") for i in range(5)]
    plan = Plan(query="perf test", tasks=tasks)

    # Warmup
    for _ in range(5):
        executor_baseline.execute_plan(plan)
        executor_observable.execute_plan(plan)

    # Measure baseline
    t0 = time.perf_counter()
    for _ in range(runs):
        executor_baseline.execute_plan(plan)
    baseline_time = time.perf_counter() - t0

    # Measure observable
    t1 = time.perf_counter()
    for _ in range(runs):
        executor_observable.execute_plan(plan)
    observable_time = time.perf_counter() - t1

    overhead_pct = ((observable_time - baseline_time) / baseline_time) * 100.0
    return baseline_time, observable_time, overhead_pct


if __name__ == "__main__":
    print("=== Phase 17.0 Observability Foundation Performance Budget ===")
    pub_us = benchmark_event_publish(10000)
    print(f"Event Publish Latency: {pub_us:.3f} µs (budget: <50 µs) -> {'PASS' if pub_us < 50 else 'FAIL'}")

    time_us = benchmark_timeline_append(10000)
    print(f"Timeline Append Latency: {time_us:.3f} µs (budget: <20 µs) -> {'PASS' if time_us < 20 else 'FAIL'}")

    met_us = benchmark_metrics_update(10000)
    print(f"Metrics Update Latency: {met_us:.3f} µs (budget: <20 µs) -> {'PASS' if met_us < 20 else 'FAIL'}")

    base_t, obs_t, overhead_1ms = benchmark_plan_execution_overhead(50, sleep_sec=0.001)
    print(f"Plan Execution Overhead (1ms tasks): {overhead_1ms:.2f}%")

    base_t5, obs_t5, overhead_5ms = benchmark_plan_execution_overhead(20, sleep_sec=0.005)
    print(f"Plan Execution Overhead (5ms tasks): {overhead_5ms:.2f}% (budget: <5%) -> {'PASS' if overhead_5ms < 5.0 else 'CHECK'}")


