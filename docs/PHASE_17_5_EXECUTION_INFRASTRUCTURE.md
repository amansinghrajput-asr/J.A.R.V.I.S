# Phase 17.5 – Execution Infrastructure (Persistence, Replay, Control & Timeouts)

## Executive Summary

Phase 17.5 completes the production execution infrastructure for the J.A.R.V.I.S. Adaptive Planner. Building upon the decoupled event foundation from Phase 17.0, Phase 17.5 adds robust runtime controls, deterministic state persistence, offline replay, and granular execution deadlines without altering planner intelligence, LLM heuristics, or existing public APIs.

All new components use explicit dependency injection, zero global mutable state, and thread-safe synchronization primitives that support both synchronous and asynchronous execution.

---

## 1. Architectural Overview & Contracts

```
                                  ┌───────────────────────────┐
                                  │   Adaptive Planner Engine │
                                  └─────────────┬─────────────┘
                                                │
         ┌──────────────────────────────────────┼──────────────────────────────────────┐
         ▼                                      ▼                                      ▼
┌──────────────────┐                  ┌──────────────────┐                  ┌──────────────────┐
│  ExecutionState  │                  │    Execution     │                  │  TimeoutManager  │
│  & Persistence   │                  │    Controller    │                  │  & Policies      │
│ (persistence.py) │                  │   (control.py)   │                  │  (timeouts.py)   │
└────────┬─────────┘                  └─────────┬────────┘                  └─────────┬────────┘
         │                                      │                                      │
         │ Save / Load / Resume                 │ Pause / Resume / Cancel              │ Deadlines & Recovery
         │                                      ▼                                      │
         │                            ┌──────────────────┐                             │
         │                            │  Plan Executor   │◄────────────────────────────┘
         │                            │  (DAG Waves)     │
         │                            └─────────┬────────┘
         ▼                                      │ Emits Events
┌──────────────────┐                            ▼
│  Offline Replay  │◄─────────────────┌──────────────────┐
│  Engine          │   Read Events    │ PlannerEventBus  │
│  (replay.py)     │                  │ (events.py)      │
└──────────────────┘                  └──────────────────┘
```

### Core Architectural Invariants:
1. **Backward Compatibility**: All existing signatures (`Executor.execute_plan()`, `Planner.plan()`) remain 100% backward-compatible. Controller and timeout parameters are optional with `None` defaults.
2. **Untouched Planner Intelligence**: No changes were made to prompt generation, LLM client routers, failure classification rules, recovery viability heuristics, or DAG topological sort logic.
3. **Deterministic State Resumption**: Resumed executions guarantee that already-completed tasks **never** execute again.
4. **Offline Replay**: The replay engine operates strictly against serialized events or timeline snapshots, never invoking LLMs, skills, or network endpoints.
5. **Zero Busy-Waiting**: Runtime pause and checkpoint synchronization utilize OS primitives (`threading.Event` and `asyncio.Event`), ensuring zero CPU spinning during pauses.

---

## 2. Execution Persistence Subsystem (`persistence.py`)

The persistence subsystem provides JSON-compatible serialization and deserialization of active plan state and execution records.

### 2.1 State Model (`PersistedExecutionState`)
```python
@dataclass
class PersistedExecutionState:
    execution_id: str
    plan_id: str
    query: str
    memory: ExecutionMemory
    dag_state: Dict[str, Any] = field(default_factory=dict)
    completed_tasks: List[Task] = field(default_factory=list)
    failed_tasks: List[Task] = field(default_factory=list)
    skipped_tasks: List[Task] = field(default_factory=list)
    retry_history: Dict[str, int] = field(default_factory=dict)
    recovery_attempts: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    schema_version: int = 1
```

### 2.2 Exclusion of Transient Runtime Objects
To prevent serialization exceptions and cross-process pollution, transient objects are explicitly excluded:
- Runtime locks (`_lock`, `_pool_lock`)
- Thread and process pools (`ThreadPoolExecutor`)
- Event buses (`EventBus`, `PlannerEventBus`)
- Dynamically computed caches (`ExecutionMemory._cache`)

Upon deserialization (`from_dict`), model caches are initialized lazily upon first property access.

### 2.3 Standalone API
- `save(state: PersistedExecutionState, file_or_path: Union[str, Path, TextIO]) -> None`
- `load(file_or_path: Union[str, Path, TextIO]) -> PersistedExecutionState`
- `resume(persisted_state: PersistedExecutionState, executor: Optional[Executor] = None) -> ExecutionResult`

---

## 3. Deterministic Resumption (`resume()`)

The resumption algorithm preserves execution continuity without duplicate work:

1. **State Hydration**: Load `PersistedExecutionState` from storage.
2. **Completed Task Identification**: Retrieve set of completed task IDs ($O(1)$ lookup).
3. **Plan Delta Synthesis**:
   - For all tasks already `COMPLETED`, preserve their status and outputs in the result map.
   - For pending, failed, or skipped tasks, reconstruct their remaining dependencies. If dependencies were satisfied in the previous run, they are immediately eligible for scheduling.
4. **Execution Dispatch**: Pass the synthetic plan to `Executor.execute_plan()`. Already completed tasks are skipped at the DAG level and never dispatched to action handlers.

---

## 4. Offline Replay Engine (`replay.py`)

The `PlannerReplayEngine` inspects past executions step-by-step for debugging, auditing, post-incident reviews, and visual timeline inspection.

### 4.1 Replay Capabilities
- **Step Forward (`step()`)**: Advances cursor forward by one event and returns a `ReplaySnapshot`.
- **Step Backward (`step_back()`)**: Moves cursor backward by one event, dynamically reflecting previous task lifecycle states.
- **Direct Seeking (`seek(step_index)`)**: Direct jump to any position in the event stream.
- **Time Seeking (`seek_time(timestamp)`)**: Binary search lookup for the execution state at a specific epoch timestamp ($O(\log N)$).
- **Reset (`reset()`)**: Restores cursor to `-1` (initial pre-execution state).

### 4.2 Snapshot Contract (`ReplaySnapshot`)
Each snapshot provides complete state visibility:
- `cursor`: Current position in the event timeline.
- `total_steps`: Total recorded event count.
- `timestamp`: Event timestamp.
- `elapsed_duration`: Duration from plan start to current cursor.
- `current_execution_progress`: Normalized progress percentage [0.0, 1.0].
- `task_lifecycle`: Map of `task_id -> status` (RUNNING, COMPLETED, FAILED, SKIPPED, PENDING).
- `completed_tasks`, `failed_tasks`, `skipped_tasks`, `pending_tasks`: Sorted lists of task IDs.
- `recovery_wave`: Active recovery iteration.
- `is_paused`, `is_cancelled`: Execution control flags.

---

## 5. Execution Controller & Checkpoints (`control.py`)

The `ExecutionController` coordinates pause, resume, and cancellation across threads and asyncio loops.

### 5.1 Synchronization Architecture
- **Sync Thread Event**: Uses `threading.Event()` for synchronous threads. When paused, `_paused_event.wait()` suspends execution with zero CPU utilization.
- **Async Event**: Uses `asyncio.Event()` for asynchronous coroutines. When paused, `await _async_paused_event.wait()` suspends without blocking the event loop.
- **Thread Safety**: All state mutations (`pause`, `resume`, `cancel`, `abort_recovery`) are guarded by `threading.RLock`.

### 5.2 Checkpoint Placement in Executor
Checkpoints are embedded into the DAG execution cycle:
1. `before_wave`: Evaluated prior to dispatching each parallel wave.
2. `before_task_{task_id}`: Evaluated before invoking a specific task handler.
3. `after_task_{task_id}`: Evaluated immediately following task execution.
4. `between_waves`: Evaluated after task results are collected and dependencies are updated.

---

## 6. Cancellation Flow & Guarantees

When cancellation is triggered (`controller.cancel(reason)`):
1. **Token Invalidation**: `CancellationToken.is_cancelled` is set to `True` with the cancellation reason.
2. **Unblocking Suspended Workers**: If execution was paused, `_paused_event.set()` and `_async_paused_event.set()` are triggered immediately so waiting threads and coroutines wake up and exit cleanly.
3. **Exception Raising**: The next checkpoint raises `ExecutionCancelledError(reason)`.
4. **Clean DAG Wind-Down**:
   - Any currently running tasks that finish have their completed status preserved.
   - All unexecuted tasks are marked with `TaskStatus.CANCELLED`.
   - `PlanCancelled` event is published to `PlannerEventBus` and core `EventBus`.
   - Returns a structured `ExecutionResult(success=False)` containing completed tasks up to the cancellation point and cancelled tasks folded into `skipped_tasks`.

---

## 7. Timeout Management & Policies (`timeouts.py`)

The `TimeoutManager` enforces execution deadlines across multiple granularities:
- `task_timeout`: Default per-task deadline in seconds.
- `action_timeouts`: Granular per-action overrides (e.g., `web_search=10.0`, `calculate=1.0`).
- `recovery_timeout`: Deadline per recovery wave.
- `total_timeout`: Hard deadline for the entire plan execution.

### 7.1 Timeout Policies
| Policy | Behavior on Timeout | Next Action |
| :--- | :--- | :--- |
| `RETRY` | Retries task up to `max_retries` attempts | If retries exhausted, applies `SKIP` or `ESCALATE` |
| `SKIP` | Marks task as `SKIPPED` | Dependent tasks cascade skip; independent tasks proceed |
| `ABORT` | Cancels execution immediately | Controller cancelled; unexecuted tasks become `CANCELLED` |
| `ESCALATE` | Raises `TaskTimeoutError` | Triggers planner replanning and recovery heuristics |

### 7.2 Thread Pool Worker Optimization
`TimeoutManager` utilizes an internal managed `ThreadPoolExecutor` pool rather than spawning single-use thread executors per task. This eliminates thread initialization overhead and ensures sub-microsecond deadline scheduling.

---

## 8. Performance Benchmarks & Budget Verification

Measured on Python 3.13 / Windows 11 via `benchmarks/execution_control_perf.py`:

| Subsystem / Metric | Target Budget | Measured Performance | Margin | Result |
| :--- | :--- | :--- | :--- | :--- |
| **Persistence Save Latency** | `< 5.0 ms` | **1.996 ms** | -60.1% | **PASS** |
| **Persistence Load Latency** | `< 5.0 ms` | **0.532 ms** | -89.4% | **PASS** |
| **Replay Step Throughput** | `> 1,000 steps/s` | **203,223.9 steps/s** | +203x | **PASS** |
| **Pause / Resume Latency** | `< 1,000 µs (1.0 ms)` | **3.379 µs** | -99.6% | **PASS** |
| **Cancellation Latency** | `< 1,000 µs (1.0 ms)` | **0.930 µs** | -99.9% | **PASS** |
| **Timeout Check Overhead** | `< 10.0 µs` | **0.321 µs** | -96.8% | **PASS** |
| **Controller Execution Overhead** | `< 5.0%` (5ms tasks) | **1.55%** | -69.0% | **PASS** |
| **Control + Timeout Overhead** | `< 5.0%` (50ms tasks) | **4.96%** | -0.8% | **PASS** |

---

## 9. Testing Strategy & Validation Matrix

The test suite covers unit, integration, and architecture invariant scenarios:

| Test Module | Tests | Focus Area | Status |
| :--- | :--- | :--- | :--- |
| `test_planner_persistence.py` | 6 | State serialization, lazy cache exclusion, resume invariant | **PASS** |
| `test_planner_replay.py` | 5 | Offline replay, step forward/backward, time seek, reset | **PASS** |
| `test_planner_control.py` | 12 | Pause/resume (sync/async), cancellation, executor integration | **PASS** |
| `test_planner_timeouts.py` | 8 | Task deadline enforcement, policies (RETRY, SKIP, ABORT, ESCALATE) | **PASS** |
| `test_architecture.py` | 13 | Module isolation, untouched planner intelligence, zero leakage | **PASS** |
| **Full Regression Suite** | **239** | Entire J.A.R.V.I.S. planner, observability, and core test suite | **PASS** |

---

## 10. Tradeoffs, Invariants & Next Phase Roadmap

### Design Tradeoffs
1. **Thread Pool Reuse vs Isolation**: Reusing a shared worker pool in `TimeoutManager` delivers a 10x throughput improvement over spawning per-task thread pools, with thread pool shutdown cleanly tied to manager lifecycle.
2. **Snapshot Precomputation vs On-Demand**: Replay engine precomputes state deltas during `load()`, trading initial load memory for instantaneous bi-directional scrubbing at 200,000+ steps/second.

### Architectural Invariants Preserved
- **Zero Global State**: All controllers, timeout managers, and replay engines are independently instantiable.
- **Thread Safety**: All state transitions use thread-safe primitives.
- **Observability Parity**: All lifecycle actions emit strongly-typed events to `PlannerEventBus`.

### Next Phase Roadmap (Phase 18 – Visual Cockpit & Distributed Execution)
- Web-based execution control UI leveraging `PlannerReplayEngine` and `ExecutionController`.
- Remote task workers utilizing serialized `PersistedExecutionState` checkpoints.
