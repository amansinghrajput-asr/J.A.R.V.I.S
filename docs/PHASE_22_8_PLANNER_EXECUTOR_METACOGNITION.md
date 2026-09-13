# Phase 22.8: Planner, Executor & Metacognition Integration Architecture

## 1. Executive Summary & Objective

Phase 22.8 connects the six specialized Phase 22 system skills (`AppSkills`, `FileSkills`, `SystemInfoSkills`, `SystemControlSkills`, `WindowSkills`, `BrowserSkills`) directly into the J.A.R.V.I.S autonomous planning and execution pipeline.

It accomplishes:
1. **Direct Action Resolution**: Routing Planner tasks to specialized Phase 22 skill modules via `Executor._resolve_from_skill_manager`.
2. **Structured Parameter Propagation**: Supplying rich parameter dictionaries (`action`, `target`, `parameters`) to `BaseSystemSkill` instances while preserving string-command compatibility for legacy skills.
3. **Execution Correlation & Lifecycle Telemetry**: Propagating unique execution and task identifiers into `PlannerEventBus` events.
4. **Metacognitive Telemetry & Evolution**: Integrating event-driven telemetry into `MetacognitiveController`, updating `SkillEvolutionEngine` metrics and `SemanticKnowledgeGraph` nodes.
5. **Robust Telemetry Deduplication**: Guaranteeing that duplicate events for the same execution are filtered without suppressing consecutive executions of the same action.
6. **Legacy Facade Backward Compatibility**: Delegating legacy `SystemSkill` calls to specialized modules while preserving exact conversational return formats.

---

## 2. End-to-End Execution Flow

```
                      ┌───────────────────────────┐
                      │          Planner          │
                      └─────────────┬─────────────┘
                                    │ Generates Plan & Tasks
                                    ▼
                      ┌───────────────────────────┐
                      │         Executor          │
                      └─────────────┬─────────────┘
                                    │ _resolve_from_skill_manager()
                                    ▼
                      ┌───────────────────────────┐
                      │       SkillManager        │
                      └─────────────┬─────────────┘
                                    │ Resolves specialized skill
                                    ▼
                      ┌───────────────────────────┐
                      │    BaseSystemSkill        │
                      │    (App/File/Info/Control/│
                      │     Window/Browser)       │
                      └─────────────┬─────────────┘
                                    │ Emits lifecycle events
                                    ▼
                      ┌───────────────────────────┐
                      │      PlannerEventBus      │
                      └─────────────┬─────────────┘
                                    │ Subscribes
                                    ▼
                      ┌───────────────────────────┐
                      │  MetacognitiveController  │
                      └──────┬─────────────┬──────┘
                             │             │
                             ▼             ▼
              ┌─────────────────────┐   ┌───────────────────────┐
              │ SkillEvolutionEngine│   │ SemanticKnowledgeGraph│
              │ (Performance Scoring│   │ (Execution Graph &    │
              │  & Version Registry)│   │  Causal Reflection)   │
              └─────────────────────┘   └───────────────────────┘
```

---

## 3. Phase 22 Action Resolution & Parameter Propagation

In `Executor._resolve_from_skill_manager(task)`:
1. **Target Action Normalization**: Extracts `task.action` and standardizes synonyms/aliases (e.g. `bring_to_front` → `focus_window`, `web_search` → `search_web`).
2. **Specialized Skill Matching**: Queries registered skills for capability matching via `skill.can_handle(action)` and known Phase 22 alias mappings:
   - **AppSkills**: `open_app`, `close_app`, `restart_app`, `is_app_running`, `list_running_apps`
   - **FileSkills**: `create_file`, `create_folder`, `read_file`, `write_file`, `delete_file`, `delete_folder`, `move_file`, `move_folder`, `copy_file`, `copy_folder`, `list_directory`, `get_file_info`, `search_files`
   - **SystemInfoSkills**: `get_system_info`, `get_cpu_info`, `get_memory_info`, `get_disk_info`, `get_battery_info`, `get_gpu_info`, `get_network_info`, `get_system_summary`, `get_platform_info`, `get_os_info`
   - **SystemControlSkills**: `set_volume`, `get_volume`, `volume_up`, `volume_down`, `mute_audio`, `unmute_audio`, `toggle_mute`, `set_brightness`, `get_brightness`, `brightness_up`, `brightness_down`, `lock_workstation`, `sleep_system`, `restart_system`, `shutdown_system`
   - **WindowSkills**: `list_windows`, `list_open_windows`, `get_active_window`, `focus_window`, `bring_to_front`, `minimize_window`, `maximize_window`, `restore_window`, `close_window`
   - **BrowserSkills**: `open_url`, `browse_url`, `open_link`, `search_web`, `web_search`, `open_browser`
3. **Structured Parameter Delivery**:
   - For `BaseSystemSkill` instances:
     ```python
     command_payload = {
         "action": action,
         "target": task.target or task.parameters.get("target"),
         "parameters": task.parameters,
     }
     result = skill.execute(command_payload)
     ```
   - For legacy `BaseSkill` instances: Formats backward-compatible strings (`f"{action} {target}"`).

---

## 4. Metacognition Integration & Deduplication Architecture

### 4.1 Telemetry Routing
`MetacognitiveController` registers handlers for:
- `SkillExecutionCompleted` / `SkillExecutionFailed`
- `TaskCompleted` / `TaskFailed`
- Whole plan execution trajectories

When an execution completes, the controller:
1. Records execution metadata and duration into the `SkillEvolutionEngine`.
2. Adds nodes and execution edges into the `SemanticKnowledgeGraph`.
3. Triggers causal reflection when failures occur.

### 4.2 Deduplication Invariant
Because both fine-grained event streams (`SkillExecutionCompleted`) and coarse trajectory evaluations (`record_execution`) can fire for a single task execution, deduplication is required.

**Defect Solved in Phase 22.8**:
- Previous implementation used coarse deduplication keys: `corr:{skill_name}:{operation}` or `corr:{operation}`. This incorrectly suppressed legitimate consecutive executions of the same operation (e.g. `open_app notepad` followed by `open_app calc` within 60 seconds).
- **Repaired Architecture**:
  - Deduplication strictly uses unique identifiers: `corr:{execution_id}` or `corr:{task_id}` or `corr:{plan_id}`.
  - If no unique execution identifier exists, deduplication fails open to ensure events are never lost.
  - Seen execution records are bound to a 60-second TTL cache (`_seen_executions` and `_recorded_executions`), preventing memory leaks in long-running processes.

---

## 5. Legacy SystemSkill Compatibility

The legacy `SystemSkill` facade (`app/skills/system_skill.py`) maintains 100% backward compatibility:
- Detects whether specialized Phase 22 modules are present in the service container.
- When available, delegates calls internally to `AppSkills`, `FileSkills`, `SystemInfoSkills`, `SystemControlSkills`, `WindowSkills`, and `BrowserSkills`.
- Retains exact legacy conversational text response formats (e.g. `"Volume set to 50%."`, `"Battery at 85%."`) so upstream components expecting conversational strings continue functioning without breakage.

---

## 6. Test Coverage & Verification

Integration across the planner, executor, and metacognition systems is validated by `tests/test_phase22_integration.py` (24 test cases):
1. **Planner Resolution Tests**: Verifies each Phase 22 action resolves to the intended specialized skill module.
2. **Structured Parameter Propagation**: Asserts that `action`, `target`, and `parameters` dictionaries reach the skills unaltered.
3. **Legacy Facade Delegation**: Validates that calling `SystemSkill.execute("set volume 40")` delegates to `SystemControlSkills` and returns legacy string messages.
4. **Metacognitive Telemetry Flow**: Confirms that successful executions update `SkillEvolutionEngine` stability and performance metrics.
5. **Deduplication Validation**: Proves that distinct consecutive executions of identical actions within 60 seconds are both recorded, while duplicate events for the same task ID are deduplicated.
