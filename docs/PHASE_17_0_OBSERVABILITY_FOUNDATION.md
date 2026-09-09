# Phase 17.0 – Sprint 1: Observability Foundation

## Executive Summary

Phase 17.0 establishes a fully decoupled, thread-safe, event-driven observability foundation for the J.A.R.V.I.S. Adaptive Planner without modifying planner intelligence, altering recovery heuristics, or breaking existing public APIs.

Every execution lifecycle milestone—from plan decomposition and concurrent wave scheduling to failure classification and recovery replanning—now emits strongly-typed, immutable events through an isolated `PlannerEventBus`. Observability consumers (timelines, metrics collectors, and structured loggers) subscribe to the event bus independently with zero coupling to planning logic.

---

## 1. Architectural Invariants & Separation of Concerns

```
                               ┌─────────────────────────┐
                               │  AIManager (Lifecycle)  │
                               └────────────┬────────────┘
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
          ┌─────────────────────┐                       ┌─────────────────────┐
          │  Planner (What)     │                       │  Executor (How)     │
          └──────────┬──────────┘                       └──────────┬──────────┘
                     │ emits Plan & Recovery                       │ emits Task & DAG
                     │ lifecycle events                            │ lifecycle events
                     └──────────────────────┬──────────────────────┘
                                            │
                                            ▼
                              ┌───────────────────────────┐
                              │     PlannerEventBus       │
                              │  (Sequence & Dispatch)    │
                              └─────────────┬─────────────┘
                                            │
         ┌──────────────────────────────────┼──────────────────────────────────┐
         ▼                                  ▼                                  ▼
┌──────────────────┐               ┌──────────────────┐               ┌──────────────────┐
│ ExecutionTimeline│               │ MetricsCollector │               │ StructuredLogger │
│ (Ordering/Audit) │               │ (Prometheus/SLA) │               │ (Redacted Logs)  │
└──────────────────┘               └──────────────────┘               └──────────────────┘
```

### Core Contracts Preserved:
1. **Zero Upstream Inversion**: Neither `Planner` nor `Executor` imports or knows about `ExecutionTimeline`, `PlannerMetricsCollector`, or `StructuredLoggingSubscriber`. They interact exclusively with `PlannerEventBus`.
2. **Subscriber Isolation**: An unhandled exception in an event listener is caught, logged, and isolated. It can never disrupt planner execution, abort active tasks, or corrupt execution state.
3. **Immutability**: All events are defined as `@dataclass(frozen=True)` and inherit from `PlannerEvent`. Once emitted, events cannot be mutated.
4. **Deterministic Ordering**: Events receive monotonically increasing atomic sequence numbers (`sequence_id: int`) upon dispatch.

---

## 2. Event Model & Lifecycle Specifications

All events inherit from `PlannerEvent` defined in `app/ai/planner/events.py`:

```python
@dataclass(frozen=True)
class PlannerEvent:
    schema_version: int = 1
    sequence_id: int = 0
    timestamp: float = field(default_factory=time.time)
    execution_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Event Hierarchy:
* **Plan Lifecycle**:
  * `PlanStarted(plan_id, query, task_count, strategy)`
  * `PlanCompleted(plan_id, success, completed_count, failed_count, skipped_count, duration)`
  * `PlanFailed(plan_id, error)`
* **Task Lifecycle**:
  * `TaskStarted(plan_id, task_id, action, target, dependencies)`
  * `TaskCompleted(plan_id, task_id, action, result, duration)`
  * `TaskFailed(plan_id, task_id, action, error, duration)`
  * `TaskRetried(plan_id, task_id, action, attempt, reason)`
* **Recovery Lifecycle**:
  * `RecoveryStarted(query, attempt, failed_task_ids, skipped_task_ids)`
  * `RecoveryCompleted(query, attempt, success, new_plan_id, task_count, duration)`
  * `RecoveryFailed(query, attempt, reason, duration)`
* **Control & Future Boundaries** (Reserved for Phase 17.5):
  * `PlanPaused(plan_id, reason)`
  * `PlanResumed(plan_id, reason)`
  * `PlanCancelled(plan_id, reason)`
  * `TaskTimeout(plan_id, task_id, timeout_seconds, policy)`
  * `RecoveryAborted(query, attempt, reason)`

---

## 3. Event Bus (`PlannerEventBus`)

The `PlannerEventBus` provides thread-safe, multi-subscriber event routing:
- **Type-Filtered Subscriptions**: `bus.subscribe(TaskCompleted, callback)` invokes the listener only when `TaskCompleted` events are published.
- **Wildcard Subscriptions**: `bus.subscribe(PlannerEvent, callback)` invokes the listener for all events in the system.
- **Async Publishing**: `publish_async(event)` seamlessly awaits async subscribers while executing sync handlers concurrently.
- **Scoped Instances**: Eliminates global mutable state; individual executions can carry isolated bus instances or share an orchestrated bus.

---

## 4. Execution Timeline Recorder (`ExecutionTimeline`)

The `ExecutionTimeline` in `app/ai/planner/timeline.py` records events in exact chronological order (`sequence_id`, `timestamp`):

### Querying & Analysis APIs:
- `filter_by_task(task_id: str)`: Isolates events specifically associated with an individual task.
- `filter_by_type(event_type: Type[T])`: Filters events by class.
- `filter_by_time(start: float, end: float)`: Slices event history within a timestamp window.
- `get_plan_duration()`: Computes total wall-clock duration.
- `get_task_durations()`: Returns duration mappings for all completed and failed tasks.
- `get_task_retries(task_id: Optional[str])`: Returns retry history.
- `get_recovery_waves()`: Returns structured summaries of multi-wave recovery attempts.

### Markdown Output Sample:
```markdown
# Planner Execution Timeline

- **Total Events Recorded**: 4
- **Total Plan Duration**: 0.055s

| Seq | Relative Time | Event | Task ID / Detail | Duration | Status |
| :---: | :---: | :--- | :--- | :---: | :---: |
| 1 | +0.000s | `PlanStarted` | Query: `open chrome and search news` (2 tasks) | - | 🚀 |
| 2 | +0.005s | `TaskStarted` | `t1` (open_app) | - | ⏳ |
| 3 | +0.025s | `TaskCompleted` | `t1` (open_app) -> opened chrome | 0.020s | ✅ |
| 4 | +0.055s | `PlanCompleted` | 1 completed, 0 failed, 0 skipped | 0.055s | ✅ |
```

---

## 5. Operational Metrics Collector (`PlannerMetricsCollector`)

The `PlannerMetricsCollector` in `app/ai/planner/metrics_collector.py` aggregates real-time performance indicators:
- **Latencies**: Planning latency, execution wave latency, recovery replanning latency.
- **Task Analytics**: Duration averages overall and grouped by action identifier.
- **Reliability Indicators**: Total retries, retries per task ID, failure categories breakdown, and recovery success rate.
- **DAG Characteristics**: Average task count, average dependency depth.
- **Prometheus Export**: Formatted exposition text output via `to_prometheus()`.

### Sample Prometheus Output:
```prometheus
# HELP jarvis_planner_plans_total Total number of plans by terminal status
# TYPE jarvis_planner_plans_total counter
jarvis_planner_plans_total{status="completed"} 42
jarvis_planner_plans_total{status="failed"} 0

# HELP jarvis_planner_planning_latency_seconds_average Average latency for planning
# TYPE jarvis_planner_planning_latency_seconds_average gauge
jarvis_planner_planning_latency_seconds_average 0.0820

# HELP jarvis_planner_recovery_success_rate Success rate of recovery attempts
# TYPE jarvis_planner_recovery_success_rate gauge
jarvis_planner_recovery_success_rate 1.0000
```

---

## 6. Structured Logging Subscriber & Security Redaction

The `StructuredLoggingSubscriber` in `app/ai/planner/logging_subscriber.py` translates planner events into standardized logs:
- `INFO`: Milestones (plan started, task completed, recovery wave completed).
- `WARNING`: Recoverable states (task failed, task retried, recovery triggered, task timeout).
- `ERROR`: Terminal states (plan failed, recovery failed).
- `DEBUG`: Tracing details (task started with dependencies, in-degree updates).

### Security & Sanitization Rules:
- Keys matching `api_key`, `token`, `secret`, `password`, `auth`, `credential`, `bearer`, or `prompt` are automatically replaced with `[REDACTED]`.
- Regex matching known secret token formats (e.g. `sk-...`, `ghp_...`, `Bearer ...`) is scrubbed automatically.
- High-volume output and query payloads are truncated to safe lengths.
- Raw LLM prompt templates and user context are never exposed in log output.

---

## 7. Performance Budget & Impact Analysis

Measured via `benchmarks/observability_perf.py` across 10,000 operations on Windows:

| Metric | Measured Overhead | Performance Budget | Status |
| :--- | :---: | :---: | :---: |
| **Event Publish Latency** | **6.636 µs** | < 50.0 µs | **PASS (7.5x faster than budget)** |
| **Timeline Append Latency** | **8.779 µs** | < 20.0 µs | **PASS (2.3x faster than budget)** |
| **Metrics Update Latency** | **9.398 µs** | < 20.0 µs | **PASS (2.1x faster than budget)** |
| **Execution Overhead (5ms tasks)**| **< 1.2%** | < 5.0% | **PASS** |

The event layer operates well within performance constraints, adding negligible overhead to task scheduling and wave execution.

---

## 8. Verification & Test Coverage

All 204 unit, integration, and architectural tests pass cleanly:

```powershell
.venv\Scripts\python.exe -m unittest `
  tests/test_ai.py `
  tests/test_application.py `
  tests/test_ai_skill.py `
  tests/test_planner.py `
  tests/test_architecture.py `
  tests/test_planner_events.py `
  tests/test_planner_timeline.py `
  tests/test_planner_metrics.py `
  tests/test_planner_logging.py
```
```text
Ran 204 tests in 0.690s

OK
```

---

## 9. Tradeoffs & Future Extension Points

1. **In-Memory History**: `ExecutionTimeline` currently records events in memory. For extremely long-running plans (>10,000 tasks), an optional rolling buffer or persistence sink (Phase 17.5) will allow streaming events to disk.
2. **Subsystem Independence**: The subscriber model ensures that persistence engines, step-by-step replays, or real-time web dashboards can be added as passive subscribers without modifying a single line of planner or executor code.
3. **Ready for Phase 17.5**: The event foundation is fully prepared for Execution Persistence (`save`/`load`/`resume`), Replay Engines, and Cancellation/Timeout Controllers in the upcoming sprint.
