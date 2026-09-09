# Phase 16 – Adaptive Planning and Execution Memory

## Overview

Phase 16 introduces a structured, reusable **execution-memory subsystem** that empowers J.A.R.V.I.S to perform adaptive replanning across multi-wave execution lifecycles while strictly preserving the separation of concerns established in earlier phases (Phase 13 Hybrid Planner, Phase 14 DAG Executor, and Phase 15 Automated Replanning).

---

## Architecture Principles & Subsystem Responsibilities

The execution and planning architecture adheres strictly to single-responsibility boundaries:

| Subsystem | Primary Responsibility | File Location |
| :--- | :--- | :--- |
| **Planner** | Decides **what** tasks should be executed. Validates schemas and DAG invariants. | `app/ai/planner/planner.py` |
| **Executor** | Decides **how** tasks are executed. Evaluates DAG dependencies and runs task handlers. | `app/ai/planner/executor.py` |
| **ExecutionResult** | Represents **one execution wave** snapshot. | `app/ai/planner/models.py` |
| **ExecutionMemory** | Represents **accumulated execution history** across multiple execution waves. | `app/ai/planner/memory.py` |
| **FailureClassifier** | Classifies **why** tasks failed into structured failure categories. | `app/ai/planner/failure_classifier.py` |
| **RecoveryHeuristics** | Deterministically decides **whether** replanning is viable before invoking LLMs. | `app/ai/planner/heuristics.py` |
| **MemorySummaryBuilder** | Transforms execution memory into prompt-ready, metadata, and human summaries. | `app/ai/planner/memory_summary.py` |
| **AIManager** | Orchestrates the overall lifecycle (initial execution, recovery waves, metrics accumulation). | `app/ai/manager.py` |

---

## 1. ExecutionMemory Subsystem

`app/ai/planner/memory.py` provides the canonical execution history.

### Models

1. **`FailureCategory(Enum)`**:
   - `DEPENDENCY_FAILURE`: Task failed because an upstream dependency failed or was skipped.
   - `TOOL_FAILURE`: Execution tool/handler threw an error.
   - `VALIDATION_FAILURE`: Invalid schema, parameters, or whitelist violation.
   - `TIMEOUT`: Execution or network deadline exceeded.
   - `EXECUTION_ERROR`: Unhandled runtime exceptions during task execution.
   - `PROVIDER_ERROR`: LLM/API provider errors (rate limit, HTTP 429/503).
   - `UNKNOWN`: Fallback category when cause cannot be determined.

2. **`TaskExecutionRecord` (dataclass, frozen)**:
   Immutable record of a single task execution attempt.
   - `task_id`: Unique identifier of the task.
   - `action`: Action name (e.g. `open_app`, `web_search`).
   - `target`: Target parameter.
   - `status`: Final `TaskStatus` (`COMPLETED`, `FAILED`, `SKIPPED`).
   - `wave`: Execution wave index (1-indexed).
   - `attempt`: Attempt count for this specific task ID across waves.
   - `output`: Result output summary (if successful).
   - `error`: Error message (if failed/skipped).
   - `failure_category`: Categorized reason for failure.
   - `duration`: Execution duration in seconds.
   - `timestamp`: Epoch timestamp of record creation.

3. **`ExecutionMetrics` (dataclass)**:
   Observational metrics across recovery waves.
   - `planning_count`: Total initial plan requests.
   - `replan_count`: Total replan attempts.
   - `successful_recoveries`: Waves that resolved previous failures.
   - `failed_recoveries`: Waves that failed to resolve failures.
   - `total_recovery_duration`: Cumulative time spent in recovery execution.
   - `execution_waves`: Total execution waves executed.
   - `average_recovery_duration` (property): Computed average time per replan attempt.

4. **`ExecutionMemory`**:
   The single canonical source of truth for execution history.
   - **Properties (Computed dynamically from `records` to prevent state duplication)**:
     - `completed_tasks`: Tasks that succeeded in any wave.
     - `failed_tasks`: Tasks whose latest attempt failed and have not completed.
     - `skipped_tasks`: Tasks skipped due to dependencies.
     - `completed_task_ids`: Set of completed task IDs.
     - `failed_task_ids`: Set of failed task IDs.
     - `skipped_task_ids`: Set of skipped task IDs.
     - `execution_order`: Deterministic list of unique task IDs executed.
     - `retry_history`: Map of task ID to number of execution attempts.
     - `task_outputs`: Map of task ID to latest non-empty output.
   - **Methods**:
     - `from_execution_result(result, wave=1)`: Initializer from wave 1 result.
     - `record_execution(result, wave=None)`: Returns an updated `ExecutionMemory` instance recording a new wave.
     - `merge(other)`: Immutably merges two `ExecutionMemory` instances.
     - `summarize_for_prompt(original_plan=None)`: Generates concise prompt injection text via `MemorySummaryBuilder`.
     - `get_summary()`: Generates structured metadata dictionary.
     - `to_dict()`: Full serialization.

---

## 2. Failure Classification

`app/ai/planner/failure_classifier.py` provides deterministic categorization without mutating `ExecutionMemory` or coupling models to heuristics:

```python
FailureClassifier.classify(
    task: Task,
    error: Optional[str] = None,
    dependency_failures: Optional[Dict[str, List[str]]] = None,
) -> FailureCategory
```

Priority resolution:
1. If `task.id` is present in `dependency_failures` mapping -> `DEPENDENCY_FAILURE`.
2. If error string matches timeout patterns (`timeout`, `timed out`, `deadline`) -> `TIMEOUT`.
3. If error string matches schema/validation patterns (`schema`, `validation`, `illegal action`) -> `VALIDATION_FAILURE`.
4. If error string matches provider/rate limit patterns (`429`, `503`, `rate limit`, `quota`) -> `PROVIDER_ERROR`.
5. If error string matches tool execution patterns (`tool`, `handler`, `exit code`) -> `TOOL_FAILURE`.
6. If error string matches generic exception patterns (`exception`, `runtimeerror`) -> `EXECUTION_ERROR`.
7. Default fallback -> `UNKNOWN`.

---

## 3. Recovery Heuristics

`app/ai/planner/heuristics.py` provides deterministic evaluation of whether recovery planning should proceed before incurring LLM overhead.

### `RecoveryDecision`
- `viable: bool`
- `reason: str`
- `blocking_tasks: List[str]`

### `evaluate_recovery_viability(query, original_plan, memory)`
Checks evaluated in order:
1. **Empty / No History**: Viable by default.
2. **All Work Completed**: If `original_plan` is satisfied and no failed/skipped tasks remain, replanning is aborted (`viable=False`).
3. **No Remaining Work**: If all tasks recorded have succeeded, replanning is aborted.
4. **Retry Budget Exhausted**: If any failed task has reached `MAX_RETRIES_PER_TASK` (default: 3) without forward progress, replanning is aborted.
5. **Permanent Validation Failures**: If remaining failures are permanent schema/validation failures that cannot self-heal, replanning is aborted.
6. **Impossible Dependency Chains**: If remaining tasks depend on unrecoverable tasks, replanning is aborted.

---

## 4. Memory Summary Builder

`app/ai/planner/memory_summary.py` constructs targeted summaries for distinct consumers:

1. **`build_planner_summary(memory, original_plan)`**:
   - Concise format injected into LLM recovery prompt.
   - Clearly lists completed tasks (with instructions DO NOT REPEAT).
   - Highlights failed tasks with their failure categories, attempt counts, and error messages.
   - Lists skipped tasks and remaining tasks from the original plan.
2. **`build_metadata_summary(memory)`**:
   - Structured summary dictionary placed into API response metadata (`execution_summary`).
   - Sized compactly (avoiding massive context bloat).
3. **`build_human_readable_summary(memory)`**:
   - Human-friendly multi-line description of execution waves, completed tasks, and failures.

---

## 5. Recovery Lifecycle & Orchestration

```
User Query
   │
   ▼
AIManager.generate() ──► Planner.create_plan() ──► Executor.execute_plan() [Wave 1]
                                                             │
                                                             ▼
                                                    ExecutionResult (Wave 1)
                                                             │
                                    ExecutionMemory.from_execution_result()
                                                             │
                 ┌───────────────────────────────────────────┴──────────────────────┐
                 │ Failed / Skipped tasks exist and Auto-Replan enabled?            │
                 └───────────────────────────┬──────────────────────────────────────┘
                                             │ YES
                                             ▼
                             RecoveryHeuristics.evaluate_recovery_viability()
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       │ Viable?                                   │
                       └─────────────┬─────────────────────────────┘
                                     │ YES
                                     ▼
                             MemorySummaryBuilder.build_planner_summary()
                                     │
                                     ▼
                             Planner._build_recovery_prompt()
                                     │
                                     ▼
                             Planner._generate_plan_sync() [LLM Provider]
                                     │
                                     ▼
                             _parse_and_validate_llm_plan() [Phase 13 validator]
                                     │
                                     ▼
                             _validate_recovery_tasks() [Phase 15 checks]
                                     │
                                     ▼
                             Executor.validate_dag() [Phase 14 validator]
                                     │
                                     ▼
                             Executor.execute_plan() [Wave 2]
                                     │
                                     ▼
                             memory.record_execution(Wave 2 result)
                                     │
                             Accumulate ExecutionMetrics
                                     │
                                     ▼
                             response.metadata["execution_summary"]
                             response.metadata["execution_metrics"]
```

---

## 6. Backwards Compatibility Contract

- **`ExecutionResult`**: Retains single-wave execution semantics without modification to its public signature.
- **`Planner.replan()` and `replan_async()`**:
  - Accept `Union[ExecutionResult, ExecutionMemory]`.
  - Accept keyword arguments `execution_result` or `memory`.
- **Validation Pipeline**: 100% reuse of Phase 13 JSON/schema validation and Phase 14 DAG cycle checks.
- **Response Metadata**: Response metadata preserves all Phase 15 keys while adding `execution_summary` and `execution_metrics`. Full raw execution memory is omitted to keep response payloads lean.
