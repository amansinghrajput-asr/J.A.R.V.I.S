# Phase 20 Metacognition & Self-Evolution Engineering Report

## 1. System Architecture & Metacognitive Loop

Phase 20 introduces an autonomous closed-loop metacognition system for J.A.R.V.I.S.:

```
                +-------------------------------------------+
                |          Hierarchical Coordinator         |
                +---------------------+---------------------+
                                      |
                       +--------------+--------------+
                       |                             |
                       v                             v
           [Macro-Skill Fast Path]         [3-Tier Swarm Fallback]
                       |                             |
                       +--------------+--------------+
                                      |
                                      v
                        SwarmExecutionResult Generated
                                      |
                       (Async ThreadPool Background Task)
                                      |
                                      v
                +-------------------------------------------+
                |          MetacognitiveController          |
                +---------------------+---------------------+
                                      |
               +----------------------+----------------------+
               |                      |                      |
               v                      v                      v
    +--------------------+  +--------------------+  +--------------------+
    |  Causal Reflection |  |  Knowledge Graph   |  | Skill Distillation |
    | - Root causes      |  | - Goal & tasks     |  | - Compile subplan  |
    | - Invariant rules  |  | - Artifacts & TTL  |  | - Sandbox verify   |
    | - Prescriptions    |  | - Multi-hop paths  |  | - Hot promotion    |
    +--------------------+  +--------------------+  +--------------------+
```

---

## 2. Execution Flow & Lifecycle Events

### Lifecycle Event Pipeline:
1. `ToolSynthesisStarted`: Dispatched when an unregistered action begins code synthesis.
2. `ToolSynthesisVerified`: Emitted when code passes AST checks and sandbox execution.
3. `ToolSynthesisRejected`: Broadcast when illegal imports or assertion errors fail synthesis.
4. `KnowledgeGraphUpdated`: Broadcast when entities, tasks, or invariants are added/modified.
5. `CausalDiagnosisGenerated`: Emitted when reflection diagnoses an execution failure.
6. `SkillDistillationCompleted`: Broadcast when a subplan compiles into a promoted macro skill.

### Trajectory Lifecycle:
- **On Success**: Trajectory tasks and outputs are indexed in the `SemanticKnowledgeGraph`. If repeatable, `MacroSkillCompiler` creates a compiled skill callable and validates it in an ephemeral sandbox.
- **On Failure**: `CausalReflectionEngine` classifies the root cause deterministically (`MISSING_TOOL`, `PRECONDITION_VIOLATION`, `EXECUTION_TIMEOUT`, `CONSENSUS_FAILURE`, `POLICY_REJECTION`, `UNKNOWN`), generates invariant rules, and records remediation prescriptions in the graph.

---

## 3. Benchmarks & Performance Verification

Microbenchmarks executed via `benchmarks/metacognition_perf.py` verified that all metacognitive operations remain far below their respective latency budgets:

```text
====================================================================
      J.A.R.V.I.S. Phase 20.5 Master Metacognition Benchmark        
====================================================================

[1/5] Benchmarking Controller Overhead (2,000 runs)...
  Result: 0.0008 ms (Budget: < 1.0 ms) -> PASS

[2/5] Benchmarking Causal Reflection Latency (500 runs)...
  Result: 0.0124 ms (Budget: < 2.0 ms) -> PASS

[3/5] Benchmarking Knowledge Graph Update Latency (5,000 updates)...
  Result: 6.0997 µs (Budget: < 100.0 µs) -> PASS

[4/5] Benchmarking Macro Skill Dispatch Latency (2,000 runs)...
  Result: 0.0047 ms (Budget: < 5.0 ms) -> PASS

[5/5] Benchmarking End-to-End Execution & Macro Skill Speedup...
  Macro Skill Fast Path:  2.0770 ms
  Standard Fallback Tier: 91.3662 ms
  Observed Speedup:       44.0x

====================================================================
                     BENCHMARK SUMMARY                           
====================================================================
Controller Overhead:     0.0008 ms       (Budget: < 1.0 ms)       [PASS]
Causal Reflection:       0.0124 ms       (Budget: < 2.0 ms)       [PASS]
Knowledge Graph Update:  6.0997 µs       (Budget: < 100.0 µs)     [PASS]
Macro Skill Dispatch:    0.0047 ms       (Budget: < 5.0 ms)       [PASS]
Macro Skill Speedup:     44.0x speedup
====================================================================
OVERALL STATUS: ALL PERFORMANCE BUDGETS MET [PASS]
====================================================================
```

---

## 4. Security, Isolation & Safety Guarantees

1. **Zero Global State**: Every component (`MetacognitiveController`, `ToolSynthesizer`, `CausalReflectionEngine`, `MacroSkillCompiler`, `SemanticKnowledgeGraph`) is instantiated with complete dependency injection and thread-level reentrant locks (`threading.RLock`).
2. **Subprocess Sandbox & AST Guardrails**: Generated code executes in an ephemeral subprocess isolated from application memory, preventing state pollution. AST scanning prohibits unsafe imports (`subprocess`, `os`, `sys`, `socket`) and dynamic execution primitives (`eval`, `exec`).
3. **Quarantine Mechanisms**: Any macro-skill can be quarantined (`quarantine_skill`) to immediately cut off execution without process restarts.
4. **Non-Blocking Background Threads**: Causal reflection and knowledge graph updates are dispatched onto dedicated worker threads, ensuring user-facing plan latency is completely unaffected.

---

## 5. Backward Compatibility & Migration Guide

The integration guarantees 100% backward compatibility with Phases 16 through 19:
- `HierarchicalCoordinator` preserves its complete constructor signature with `metacognitive_controller=None` as default.
- If no `MetacognitiveController` is supplied, `HierarchicalCoordinator` executes identical 3-tier fallback logic without overhead.
- All event classes inherit from `PlannerEvent` and integrate seamlessly with `PlannerEventBus`.
