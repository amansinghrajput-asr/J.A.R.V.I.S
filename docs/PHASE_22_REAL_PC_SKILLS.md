# Phase 22: Real PC & Desktop Skills Architecture Milestone Document

## 1. Phase 22 Objective

The primary objective of **Phase 22 ("Real PC & Desktop Skills")** is to transform J.A.R.V.I.S from an abstract conversational assistant into a secure, deterministic, and highly capable desktop automation agent.

Phase 22 equips J.A.R.V.I.S with native desktop control across six core categories:
1. Application Lifecycle Management (`AppSkills`)
2. Filesystem & Directory Operations (`FileSkills`)
3. Hardware & System Diagnostics (`SystemInfoSkills`)
4. Host System Control (`SystemControlSkills`)
5. Window Management & Desktop Geometry (`WindowSkills`)
6. Browser Navigation & Web Retrieval (`BrowserSkills`)

All capabilities are backed by rigorous security tiers, path containment verification, process protection allowlists, cryptographic confirmation tokens, dependency injection isolation, and integration into the autonomous Planner and Metacognition engine.

---

## 2. Architecture Overview

```
                               ┌─────────────────────────────┐
                               │    Voice / GUI / Planner    │
                               └──────────────┬──────────────┘
                                              │ Task Dispatch
                                              ▼
                               ┌─────────────────────────────┐
                               │      Executor Subsystem     │
                               └──────────────┬──────────────┘
                                              │ _resolve_from_skill_manager()
                                              ▼
                               ┌─────────────────────────────┐
                               │        SkillManager         │
                               └──────────────┬──────────────┘
                                              │
              ┌───────────────────────────────┴───────────────────────────────┐
              ▼                                                               ▼
  ┌───────────────────────┐                                       ┌───────────────────────┐
  │   BaseSystemSkill     │                                       │   SystemSecurity      │
  │   Foundation Contract │                                       │   Policy (3 Tiers)    │
  └───────────┬───────────┘                                       └───────────┬───────────┘
              │                                                               │
  ┌───────────┴───────────────────────────────────────────┐                   │ Enforces
  ▼                ▼              ▼          ▼           ▼                    ▼
┌─────────┐  ┌───────────┐  ┌──────────┐ ┌───────┐ ┌───────────┐   ┌───────────────────────┐
│AppSkills│  │FileSkills │  │SystemInfo│ │Control│ │Window/Web │   │ SystemConfirmation    │
│(psutil) │  │(Safe I/O) │  │(Sensors) │ │(APIs) │ │(user32/   │   │ Manager (Tokens)      │
│         │  │           │  │          │ │       │ │ webbrowser│   └───────────────────────┘
└─────────┘  └───────────┘  └──────────┘ └───────┘ └───────────┘
```

The system is decoupled into:
- **`BaseSystemSkill`**: Extends `BaseSkill` with unified parameter parsing, security verification, structured execution hooks, and event publishing.
- **`SystemSecurityPolicy`**: Thread-safe policy engine classifying actions into `SAFE`, `CONFIRMATION_REQUIRED`, and `RESTRICTED`.
- **`SystemConfirmationManager`**: Generates and validates time-bound, parameter-pinned cryptographic confirmation tokens for high-risk operations.
- **`PlannerEventBus`**: Dispatches execution lifecycle events to the metacognitive feedback loop.

---

## 3. The Six Skill Categories

1. **`AppSkills` (`app.skills.system.app_skills`)**:
   Provides desktop application launching, allowlist enforcement, process snapshot inspection, running process enumeration, and safe application termination.
2. **`FileSkills` (`app.skills.system.file_skills`)**:
   Provides sandboxed file creation, read, write, move, copy, deletion, directory listing, and file searching within strictly validated root boundaries.
3. **`SystemInfoSkills` (`app.skills.system.system_info_skills`)**:
   Provides real-time, non-blocking telemetry for CPU usage, physical memory, storage volumes, battery level, GPU status, and network interfaces.
4. **`SystemControlSkills` (`app.skills.system.system_control_skills`)**:
   Controls host audio volume, display brightness, screen locking, and power management (sleep, restart, shutdown) via native Windows APIs.
5. **`WindowSkills` (`app.skills.system.window_skills`)**:
   Enumerates desktop application windows, finds active foreground windows, and alters window geometry (focus, minimize, maximize, restore, graceful close) via Win32 `user32.dll`.
6. **`BrowserSkills` (`app.skills.system.browser_skills`)**:
   Safely handles web URLs, domain routing, search engine dispatch, and browser launches using Python's standard `webbrowser` library without shell execution.

---

## 4. Complete Action Inventory

| Category | Supported Actions | Safety Tier | Key Dependencies |
| :--- | :--- | :--- | :--- |
| **AppSkills** | `open_app`, `is_app_running`, `list_running_apps` | `SAFE` | `subprocess`, `psutil` |
| | `close_app`, `restart_app` | `CONFIRMATION_REQUIRED` | `psutil` |
| **FileSkills** | `read_file`, `list_directory`, `get_file_info`, `search_files` | `SAFE` | `pathlib`, `os` |
| | `create_file`, `create_folder`, `write_file`, `move_file`, `move_folder`, `copy_file`, `copy_folder` | `SAFE` / Confined | `pathlib`, `shutil` |
| | `delete_file`, `delete_folder` | `CONFIRMATION_REQUIRED` | `send2trash` / `os.unlink` |
| **SystemInfoSkills** | `get_system_info`, `get_cpu_info`, `get_memory_info`, `get_disk_info`, `get_battery_info`, `get_gpu_info`, `get_network_info`, `get_system_summary`, `get_platform_info` | `SAFE` | `psutil`, `platform` |
| **SystemControlSkills**| `get_volume`, `set_volume` (0–99), `get_brightness`, `set_brightness`, `lock_workstation` | `SAFE` | `pycaw`, `sbc`, `user32` |
| | `set_volume(100)`, `mute_audio`, `sleep_system`, `restart_system`, `shutdown_system` | `CONFIRMATION_REQUIRED` | `powrprof`, `shutdown.exe` |
| **WindowSkills** | `list_windows`, `list_open_windows`, `get_active_window`, `focus_window`, `minimize_window`, `maximize_window`, `restore_window` | `SAFE` | Win32 `user32`, `dwmapi` |
| | `close_window` | `CONFIRMATION_REQUIRED` | Win32 `WM_CLOSE` |
| **BrowserSkills** | `open_url`, `browse_url`, `open_link`, `search_web`, `web_search`, `open_browser` | `SAFE` | `webbrowser`, `urllib` |

---

## 5. Planner & Executor Integration

- **Resolution Mechanism**: `Executor._resolve_from_skill_manager(task)` routes actions to registered skill modules based on action name and capability.
- **Structured Parameter Propagation**: Passes `{"action": action, "target": target, "parameters": parameters}` directly to `BaseSystemSkill.execute()`, bypassing brittle string concatenations.
- **Legacy Compatibility**: If an upstream caller supplies plain text commands, `BaseSystemSkill.execute()` parses the string via `parse_command()` without failing.
- **Task & Execution Correlation**: Propagates `task.id` and `plan.id` across all lifecycle events for full distributed tracing.

---

## 6. Metacognition Integration & Telemetry

- **Event Bus Broadcasting**: Operations emit `SkillExecutionStarted`, `SkillExecutionCompleted`, and `SkillExecutionFailed` to `PlannerEventBus`.
- **Scoring & Performance Metrics**: `MetacognitiveController` intercepts completion events and updates `SkillEvolutionEngine` execution times, success rates, and stability scores.
- **Causal Reflection & Graph**: Execution results create observation nodes in `SemanticKnowledgeGraph`. Failures trigger automatic causal diagnosis.
- **Deduplication Engine**: Uses unique execution IDs (`corr:{id}`) with a 60-second TTL cache to eliminate duplicate recordings between lifecycle events and trajectory submissions without suppressing consecutive runs of identical actions.

---

## 7. Security & Confirmation Model

The Phase 22 security architecture enforces a 3-tier authorization model:
1. **`SAFE`**: Read-only diagnostics, safe volume/brightness adjustments, window focus/minimize, allowlisted app launch, and safe file reads. Executed immediately.
2. **`CONFIRMATION_REQUIRED`**: Process termination, file deletion, system sleep, restart, shutdown, and max volume. Execution is halted until a valid confirmation token is supplied.
3. **`RESTRICTED`**: Modifying Windows system directories (`C:\Windows`), executing unauthorized scripts/binaries, accessing raw memory, or invoking arbitrary shell interpreters (`cmd.exe`, `powershell.exe`). Strictly rejected with `SecurityPolicyViolationError`.

### Confirmation Tokens
- Generated via cryptographic UUID4 and pinned to specific parameter hashes:
  $$\text{Signature} = \text{SHA256}(\text{action} + \text{target} + \text{parameters})$$
- Enforces single-use consumption and automatic expiration after configured timeout (default 30 seconds).

---

## 8. Safety Restrictions

To protect the host operating system:
- **No Arbitrary Shell**: `shell=True` is strictly forbidden across all system skills.
- **Direct Process Execution**: Subprocess launches use explicit argument arrays (`["notepad.exe"]`) with zero string interpolation.
- **Critical Process Shield**: Blocking termination of essential Windows processes (`lsass.exe`, `csrss.exe`, `explorer.exe`, `services.exe`, etc.).

---

## 9. Legacy SystemSkill Compatibility

The legacy `SystemSkill` facade (`app/skills/system_skill.py`) maintains 100% backward compatibility:
- Detects the presence of Phase 22 modules in `ServiceContainer`.
- Transparently routes legacy commands (`set volume 50`, `what is cpu usage`) to `SystemControlSkills` or `SystemInfoSkills`.
- Retains exact conversational return strings (e.g. `"Volume set to 50%."`).

---

## 10. Browser Security

`BrowserSkills` enforces web safety:
- **Strict Scheme Allowlist**: Permits only `http://` and `https://`.
- **Exploit Prevention**: Rejects `file://`, `javascript:`, `data:`, `vbscript:`, and `powershell:` URLs.
- **Domain Normalization**: Converts naked domains (`google.com`) into safe HTTPS URLs (`https://google.com`).
- **Standard Library Launch**: Uses `webbrowser.open(url, new=2, autoraise=True)` without spawning shell processes.

---

## 11. Filesystem Safety

`FileSkills` guarantees sandbox integrity:
- **Path Containment**: Every target path is resolved and verified against configured `allowed_roots` using strict `is_relative_to()` canonicalization.
- **Path Traversal Prevention**: Neutralizes `..` traversal and symlink escape attacks.
- **Protected OS Directories**: Operations inside `C:\Windows`, `C:\Program Files`, and system partitions are categorically blocked.
- **Atomic Operations**: Safe writes and directory size bounds prevent storage exhaustion.

---

## 12. Process Safety

`AppSkills` guarantees application safety:
- **Allowlist Resolution**: Validates application names against safe aliases and approved installation directories.
- **Structured Launch**: Applications launch via `subprocess.Popen(args, shell=False)`.
- **Termination Safeguards**: Closing applications requires user confirmation and uses graceful termination before resorting to kill.

---

## 13. Window Management

`WindowSkills` controls desktop windows safely:
- **Win32 CFFI APIs**: Directly invokes `user32.dll` (`EnumWindows`, `GetForegroundWindow`, `SetForegroundWindow`, `ShowWindow`).
- **Cloaking & System Filtering**: Uses `DwmGetWindowAttribute` to filter out invisible frames and background UWP tasks.
- **Graceful Termination**: Closes windows by posting `WM_CLOSE`, allowing applications to prompt users to save unsaved documents.

---

## 14. System Information

`SystemInfoSkills` provides non-invasive telemetry:
- **Read-Only Operation**: Telemetry calls perform zero state mutation.
- **Non-Blocking Architecture**: Uses `psutil.cpu_percent(interval=None)` to avoid freezing the event loop.
- **Hardware Fallback**: Safely handles missing hardware sensors (e.g., virtual machines lacking battery or discrete GPU).

---

## 15. Testing Suite

The Phase 22 test suite contains extensive unit, regression, and integration tests:
- `tests/test_phase22_integration.py`: 24 tests validating Planner → Executor → Metacognition flow.
- `tests/test_system_skill.py`: 12 tests validating legacy facade backward compatibility.
- `tests/test_system_skills_browser.py`: 16 tests validating URL validation, scheme filtering, and browser routing.
- `tests/test_app_skills.py`: Process launching, allowlist enforcement, and lifecycle tests.
- `tests/test_file_skills.py`: Path containment and temporary directory I/O tests.
- `tests/test_system_control_skills.py`: Native volume, display brightness, and lock tests.
- `tests/test_system_info_skills.py`: CPU, RAM, disk, and sensor diagnostics tests.
- `tests/test_window_skills.py`: Win32 window geometry and target resolution tests.
- `tests/test_architecture.py`: Dependency injection, lock isolation, and service container contracts.

All tests run safely using mocks or temporary directories with zero destructive operations.

---

## 16. Benchmark Methodology

Benchmark suites measure realistic throughput and latency while guaranteeing host safety:
- **Safe Execution**: All native OS mutations (process kill, power actions, window changes, browser launches) are strictly mocked.
- **Temporary Filesystem Isolation**: Filesystem tests run within `tempfile.TemporaryDirectory()`.
- **SLA Performance Budgets**: Every operation is tested against latency thresholds (100 µs to 5000 µs).
- **Statistical Rigor**: 500 to 2000 iterations per benchmark, reporting min, mean, max, and standard deviation.

---

## 17. Performance Benchmark References

The Phase 22 unified benchmark suite (`benchmarks/system_skills_perf.py`) reports:
- **Total Operations Benchmarked**: 34 operations across all 6 skill categories.
- **Success Rate**: 34 / 34 passed within SLA budgets.
- **Overall Mean Operation Latency**: **408.65 µs** (~0.41 ms).
- **Category Mean Averages**:
  - `AppSkills`: ~22.87 µs
  - `BrowserSkills`: ~25.47 µs
  - `SystemControlSkills`: ~31.19 µs
  - `SystemInfoSkills`: ~209.73 µs
  - `FileSkills`: ~689.60 µs (NTFS disk I/O)
  - `WindowSkills`: ~1,458.58 µs (desktop multi-window enumeration)

Reference results: `benchmarks/results/system_skills_benchmark_results.json` and `docs/SYSTEM_SKILLS_BENCHMARK_REPORT.md`.

---

## 18. Phase 22.8 Integration Highlights

- Unified action aliasing connects 35+ system operations directly to Planner task schemas.
- High-severity deduplication defect in metacognition resolved: replaced coarse action keys with unique execution IDs (`corr:{id}`).
- Full bidirectional telemetry synchronization established with `SkillEvolutionEngine` and `SemanticKnowledgeGraph`.

---

## 19. Known Limitations

1. **Operating System Specialization**: `WindowSkills` and `SystemControlSkills` are optimized for Windows 10 and 11. Linux and macOS fallback implementations provide limited functionality.
2. **Display Brightness Hardware**: Display brightness control requires monitors supporting DDC/CI or internal laptop panels; unsupported monitors gracefully return diagnostic status.
3. **Elevated Permissions**: Operations requiring administrator privileges (UAC) are not elevated automatically and will gracefully report permission errors.

---

## 20. Completion Status

| Milestone | Status | Verification |
| :--- | :--- | :--- |
| **22.1 Foundation** | **COMPLETE** | `BaseSystemSkill`, `SystemSecurityPolicy`, `SystemConfirmationManager` |
| **22.2 App Skills** | **COMPLETE** | `AppSkills`, `AppResolver`, `ProcessManager`, tests passed |
| **22.3 File Skills** | **COMPLETE** | `FileSkills`, path containment, temporary directory tests passed |
| **22.4 System Info Skills** | **COMPLETE** | `SystemInfoSkills`, hardware telemetry tests passed |
| **22.5 System Control Skills** | **COMPLETE** | `SystemControlSkills`, volume, brightness, power tests passed |
| **22.6 Window Skills** | **COMPLETE** | `WindowSkills`, Win32 API geometry, tests passed |
| **22.7 Browser Skills** | **COMPLETE** | `BrowserSkills`, URL validation, domain routing tests passed |
| **22.8 Planner & Metacognition** | **COMPLETE** | `Executor`, `MetacognitiveController`, evolution scoring, dedup fix |
| **22.9 Benchmarks & Validation** | **COMPLETE** | Micro-benchmarks, architecture tests, comprehensive documentation |

**Phase 22 is 100% COMPLETE.**
