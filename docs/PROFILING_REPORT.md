# J.A.R.V.I.S Subsystem Profiling Report

**Generated:** 2026-09-09 22:25:36  
**Workload:** 100 iterations on 250-task synthetic plans  

---

## Top 35 Functions by Cumulative Time

```text
197059 function calls in 0.120 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      100    0.032    0.000    0.051    0.001 C:\Users\ajays\Downloads\J.A.R.V.I.S\app\ai\planner\memory_summary.py:18(build_planner_summary)
      100    0.020    0.000    0.046    0.000 C:\Users\ajays\Downloads\J.A.R.V.I.S\app\ai\planner\memory_summary.py:86(build_metadata_summary)
    34200    0.014    0.000    0.019    0.000 C:\Users\ajays\AppData\Local\Programs\Python\Python313\Lib\enum.py:200(__get__)
    14600    0.008    0.000    0.016    0.000 {built-in method builtins.hasattr}
     5000    0.006    0.000    0.012    0.000 C:\Users\ajays\Downloads\J.A.R.V.I.S\app\ai\planner\failure_classifier.py:37(classify)
    49850    0.012    0.000    0.012    0.000 {method 'get' of 'dict' objects}
      100    0.006    0.000    0.010    0.000 C:\Users\ajays\Downloads\J.A.R.V.I.S\app\ai\planner\heuristics.py:39(evaluate_recovery_viability)
    33200    0.007    0.000    0.007    0.000 {method 'append' of 'list' objects}
    34200    0.005    0.000    0.005    0.000 C:\Users\ajays\AppData\Local\Programs\Python\Python313\Lib\enum.py:1339(value)
     5000    0.004    0.000    0.004    0.000 {method 'search' of 're.Pattern' objects}
    10000    0.002    0.000    0.002    0.000 {method 'strip' of 'str' objects}
     5400    0.001    0.000    0.001    0.000 {built-in method builtins.len}
      500    0.001    0.000    0.001    0.000 {method 'join' of 'str' objects}
      100    0.000    0.000    0.001    0.000 {built-in method builtins.all}
      300    0.000    0.000    0.001    0.000 C:\Users\ajays\Downloads\J.A.R.V.I.S\app\ai\planner\memory.py:171(completed_tasks)
      600    0.000    0.000    0.001    0.000 C:\Users\ajays\Downloads\J.A.R.V.I.S\app\ai\planner\heuristics.py:72(<genexpr>)
      600    0.000    0.000    0.000    0.000 C:\Users\ajays\Downloads\J.A.R.V.I.S\app\ai\planner\memory.py:191(failed_tasks)
      100    0.000    0.000    0.000    0.000 <string>:2(__init__)
      400    0.000    0.000    0.000    0.000 C:\Users\ajays\Downloads\J.A.R.V.I.S\app\ai\planner\memory.py:246(retry_history)
      100    0.000    0.000    0.000    0.000 C:\Users\ajays\Downloads\J.A.R.V.I.S\app\ai\planner\memory.py:233(execution_order)
      400    0.000    0.000    0.000    0.000 C:\Users\ajays\Downloads\J.A.R.V.I.S\app\ai\planner\memory.py:212(skipped_tasks)
      704    0.000    0.000    0.000    0.000 C:\Users\ajays\Downloads\J.A.R.V.I.S\app\ai\planner\memory.py:142(completed_task_ids)
      402    0.000    0.000    0.000    0.000 C:\Users\ajays\Downloads\J.A.R.V.I.S\app\ai\planner\memory.py:132(latest_task_records)
      500    0.000    0.000    0.000    0.000 {method 'add' of 'set' objects}
      100    0.000    0.000    0.000    0.000 C:\Users\ajays\Downloads\J.A.R.V.I.S\app\ai\planner\memory.py:149(failed_task_ids)
      200    0.000    0.000    0.000    0.000 C:\Users\ajays\Downloads\J.A.R.V.I.S\app\ai\planner\memory.py:256(task_outputs)
      100    0.000    0.000    0.000    0.000 C:\Users\ajays\Downloads\J.A.R.V.I.S\app\ai\planner\memory.py:160(skipped_task_ids)
      102    0.000    0.000    0.000    0.000 {method 'items' of 'dict' objects}
      100    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
```

---

## Bottleneck Analysis & Optimization Targets

1. **ExecutionMemory Computed Properties Overhead**:
   Each call to `completed_task_ids`, `failed_task_ids`, `retry_history`, and `latest_task_records` iterates linearly through all records ($O(N)$ for each property access).
   When multiple properties are read consecutively (e.g. during summary generation and heuristic checks), `self.records` is traversed 8-10 times redundantly.
   *Target Optimization*: Implement an internal cached cache structure in `ExecutionMemory` with explicit invalidation on `record_execution()`.

2. **FailureClassifier Regex Compilation Overhead**:
   Currently regex patterns or repeated linear keyword scans can be pre-compiled into compiled pattern tuples for faster priority matching.
   *Target Optimization*: Pre-compile regular expressions and prioritize fast string inclusion checks before regex matching.

3. **MemorySummaryBuilder List Allocations**:
   String concatenation via intermediate list builds can be streamlined to minimize intermediate objects.
