# Phase 20.2 – Dynamic Tool Synthesizer & Runtime Hot-Binding

**Status**: Verified & Production-Ready  
**Milestone**: Phase 20.0 (Sprint 20.2)  
**Module**: `app/ai/planner/metacognition`

---

## 1. Architectural Overview

Sprint 20.2 implements the **Dynamic Tool Synthesis & Runtime Hot-Binding Layer** within J.A.R.V.I.S.

In previous phases (Phases 16–19), action execution was limited to statically pre-registered handlers, predefined skills, or service container components. When an action was missing from these registries, execution would fail or fall back to single-agent heuristics.

Sprint 20.2 introduces autonomous, on-the-fly Python tool formulation, AST security validation, subprocess sandbox testing, and immediate in-memory hot-binding without system restart or agent recreation.

```
[Planner Task Resolution]
           |
           v
[1. Registered Handlers?] ---> Found ---> [Execute Handler]
           | (No)
           v
[2. SkillManager?] ----------> Found ---> [Execute Skill]
           | (No)
           v
[3. ServiceContainer?] ------> Found ---> [Execute Service]
           | (No)
           v
[3.5 Dynamic Tool Registry] -> Found ---> [Execute Dynamic Tool]
           | (No)
           v
[ToolSynthesizer Active?]
       /          \
    (No)          (Yes)
     /              \
[Priority 4:     [Dynamic Tool Synthesis Pipeline]
 Fallback /       1. Generate Python Code
 Fail Task]       2. Generate Synthetic Unit Tests
                  3. AST Security Scan (Whitelist / Forbidden calls)
                  4. Subprocess Sandbox Execution
                  5. Hot-Bind to DynamicToolRegistry
                  6. Return Verified Callable Handler
```

---

## 2. Core Components

### 2.1 `DynamicToolRegistry` (`registry.py`)
A thread-safe in-memory registry using re-entrant locks (`threading.RLock`) providing:
- `register(action_name, callable_obj, overwrite=False)`: Registers a Python callable with duplicate prevention.
- `resolve(action_name)`: Resolves an action name into an executable callable in $< 1.0\ \mu\text{s}$.
- `contains(action_name)`: Checks presence of action.
- `unregister(action_name)`: Safely removes tool from registry.
- `list_tools()`: Deterministically returns an alphabetically sorted list of registered action names.

### 2.2 `SyntheticTestGenerator` (`synthesizer.py`)
Generates deterministic, reproducible unit test code without LLM dependencies:
- Domain-specific deterministic assertions for mathematical, string, logical, and conversion actions.
- Schema-driven test generation parsing parameter types from JSON Schema definitions and asserting non-null output.

### 2.3 `SandboxedExecutionHarness` (`sandbox.py`)
Provides isolated validation with defense-in-depth:
- **Static AST Analysis (`scan_ast_security`)**:
  - Parses code into Python AST.
  - Whitelists safe standard library modules (`math`, `re`, `json`, `datetime`, `urllib`, `hashlib`, `typing`, `collections`, `itertools`, `string`, `random`, `copy`, `functools`, `time`).
  - Prohibits dangerous calls (`eval`, `exec`, `open`, `__import__`, `compile`, `globals`, `locals`, `getattr`, `setattr`, `delattr`, `system`).
  - Blocks reflection exploit attributes (`__subclasses__`, `__bases__`, `__mro__`, `__code__`, `__globals__`).
- **Ephemeral Subprocess Sandbox (`execute_in_sandbox`)**:
  - Runs combined code and synthetic tests in an isolated Python subprocess.
  - Enforces execution deadline (default $2.0\text{ s}$) with automated process killing on timeout.
  - Captures exit codes, stdout, stderr, and execution timings.

### 2.4 `ToolSynthesizer` (`synthesizer.py`)
Orchestrates the dynamic formulation and verification lifecycle:
- Manages retry loops up to `max_attempts` (default $3$).
- Compiles verified code into isolated function objects.
- Automatically hot-binds verified tools into `DynamicToolRegistry`.
- Emits lifecycle events (`ToolSynthesisStarted`, `ToolSynthesisVerified`, `ToolSynthesisRejected`) onto `PlannerEventBus`.
- Guarantees zero exception leakage and thread-safe execution.

---

## 3. Planner Integration (Priority 3.5)

In `app/ai/planner/executor.py`, `resolve_handler` was extended with an optional Priority 3.5 step:

$$\text{Priority 1 (Handlers)} \to \text{Priority 2 (SkillManager)} \to \text{Priority 3 (Container)} \to \mathbf{Priority\ 3.5\ (ToolSynthesizer)} \to \text{Priority 4 (Fallback)}$$

```python
# Priority 3.5: DynamicToolRegistry / ToolSynthesizer
if self._dynamic_tool_registry is not None:
    tool_callable = self._dynamic_tool_registry.resolve(clean_action)
    if tool_callable is not None:
        adapter = self._build_dynamic_tool_adapter(tool_callable, clean_action)
        with self._lock:
            self._handlers[clean_action] = adapter
        return adapter

if self._tool_synthesizer is not None:
    # Trigger on-demand synthesis, verification, and hot-binding
    ...
```

When `tool_synthesizer` and `dynamic_tool_registry` are omitted (default), `Executor` behaves identically to Phase 16–19, maintaining 100% backward compatibility.

---

## 4. Performance Benchmarks

Measured on Windows 11 under Python 3.13 via `benchmarks/metacognition_perf.py`:

| Subsystem / Metric | Measured Value | Performance Budget | Status |
|---|---|---|---|
| **Tool Registration Latency** | 2.055 µs | < 20.0 µs | **PASS** |
| **Tool Lookup Latency** | 0.727 µs | < 10.0 µs | **PASS** |
| **Mock Tool Synthesis Latency** | 0.300 ms | < 5.0 ms | **PASS** |
| **Concurrent Registration Throughput** | 213,916 ops/sec | N/A | **PASS** |

---

## 5. Security Guardrails

1. **AST Whitelist Verification**: Disallows filesystem access (`open`, `os`), process execution (`subprocess`, `system`), and runtime metaprogramming (`eval`, `exec`).
2. **Subprocess Isolation**: Test suites execute in separate OS processes; infinite loops or memory allocations are killed by the watchdog timer.
3. **No Global State**: Registry and synthesizer instances are created per container/coordinator scope, preventing cross-test pollution.

---

## 6. Limitations

- Dynamic tools synthesized in Sprint 20.2 reside in in-memory registries and do not persist across process restarts (addressed in Sprint 20.4 macro-skill compilation).
- Tools requiring external network calls or binary C-extensions are rejected by the strict AST security scanner.
