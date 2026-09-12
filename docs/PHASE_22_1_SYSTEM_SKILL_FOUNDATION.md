# Phase 22.1: System Skill Foundation, Security Policy & Confirmation Manager

> **Status**: COMPLETE & VERIFIED  
> **Release**: `v0.22.1`  
> **Prerequisites**: Phase 20 (Metacognition) & Phase 21 (Autonomous Learning & Skill Evolution) Verified  
> **Scope**: Sprint 22.1 System Foundation ONLY (No individual skill implementations)

---

## 1. Executive Summary

Phase 22.1 establishes the foundational layer connecting J.A.R.V.I.S.'s cognitive planning, multi-agent orchestration, and metacognitive skill evolution directly to real Windows PC operations in a secure, thread-safe, and deterministic manner.

Rather than exposing an unrestricted command shell or creating disjointed parallel execution paths, Sprint 22.1 integrates seamlessly into the existing J.A.R.V.I.S. `BaseSkill`, `SkillManager`, `ServiceContainer`, and `PlannerEventBus` architecture.

```
Incoming Action / Task
         ↓
BaseSystemSkill.execute()
         ↓
SystemSecurityPolicy.validate_operation()
  ├─ RESTRICTED ───────────────> Emits SystemSkillPolicyRejected & Raises SecurityPolicyViolationError
  ├─ CONFIRMATION_REQUIRED ────> Requires SystemConfirmationManager Token Verification
  │                                └─ Missing / Rejected ──> Emits SystemSkillConfirmationRequired & Halts
  │                                └─ Approved & Verified ─> Consumes Token & Continues
  └─ SAFE ─────────────────────> Auto-Authorized
         ↓
Emits SystemSkillStarted
         ↓
Concrete PC Operation Execution (_execute_operation)
         ↓
Emits SystemSkillCompleted / SystemSkillFailed
         ↓
Returns Standardized SystemSkillResult (Duration, Success, Telemetry)
         ↓
MetacognitiveController / SkillEvolutionEngine Feedback
```

---

## 2. Architecture & Public Components

### 2.1 Component Overview

| File Location | Component | Role |
|---|---|---|
| `app/skills/system/security.py` | `SystemSecurityPolicy` | 3-tier deterministic action classification, canonical path containment check, critical process protection, shell injection filtering. |
| `app/skills/system/security.py` | `SystemConfirmationManager` | Cryptographic parameter-bound confirmation tokens, single-use consumption, TTL expiration, `InterventionGateway` bridging. |
| `app/skills/system/base_system_skill.py` | `BaseSystemSkill` | Abstract base class extending `BaseSkill` with unified validation, confirmation checks, timing telemetry, and event broadcasting. |
| `app/skills/system/base_system_skill.py` | `SystemSkillResult` | Standardized execution outcome dataclass with timing, error, and metadata. |
| `app/skills/system/__init__.py` | Package Exports & DI | Re-exports all components and provides `register_system_foundation()` helper. |
| `app/ai/planner/events.py` | Planner Events | Added `SystemSkillStarted`, `SystemSkillCompleted`, `SystemSkillFailed`, `SystemSkillConfirmationRequired`, `SystemSkillPolicyRejected`. |

---

## 3. 3-Tier Security Model

Every system skill operation is evaluated deterministically without LLM dependency:

### Tier 1: SAFE (Automatic Execution)
- Read-only telemetry: CPU, RAM, Disk, Battery, GPU, Network, system summary.
- Read-only file inspection: `list_directory`, `read_file`.
- Non-destructive process/app inspection: `list_running_apps`, `is_app_running`.
- Non-destructive window management: `list_windows`, `get_active_window`, `focus_window`, `minimize_window`, `maximize_window`, `restore_window`.
- Safe workstation controls: `get_volume`, `volume_up`, `volume_down`, `mute_volume`, `unmute_volume`, `get_brightness`, `set_brightness`, `lock_workstation`.
- Safe application launch (from allowlist / safe binary paths).
- Validated URL opening (`http://`, `https://`).

### Tier 2: CONFIRMATION REQUIRED (Explicit Interactive Approval)
- File deletion: `delete_path`, `delete_file`, `delete_folder`, `remove_file`, `remove_directory`.
- Destructive file overwrite / cross-root move.
- User process termination: `terminate_process`, `kill_process`, `close_app` (with `force=True`).
- Power operations: `sleep_system`, `restart_system`, `shutdown_system`.
- Extreme system volume settings (`set_volume` with `level=100`).

### Tier 3: RESTRICTED (Hard Rejection)
- Arbitrary shell / script execution (`cmd.exe`, `powershell.exe`, `pwsh.exe`, `bash`, `sh`, `wscript`, `cscript`, `mshta`).
- Commands containing shell injection tokens (`;`, `&&`, `||`, `|`, `` ` ``, `$()`, `-enc`, `-executionpolicy bypass`, `iex`, `invoke-expression`).
- Critical Windows OS process termination (protects `smss.exe`, `csrss.exe`, `wininit.exe`, `services.exe`, `lsass.exe`, `winlogon.exe`, `dwm.exe`, `explorer.exe`, `svchost.exe`, PID 0, PID 4, and the current Python process PID).
- Operations targeting protected Windows system directories (`C:\Windows`, `C:\Windows\System32`, `C:\Windows\SysWOW64`, `C:\ProgramData\Microsoft`, `WindowsApps`, etc.).
- Path traversal outside configured `allowed_roots`.
- Windows reserved device names (`CON`, `PRN`, `AUX`, `NUL`, `COM1`–`COM9`, `LPT1`–`LPT9`).
- Dangerous destructive system commands (`format`, `rm -rf`, `del /f /s /q`, fork bombs, `vssadmin`, `bcdedit`).

---

## 4. Parameter-Bound Confirmation Tokens

The `SystemConfirmationManager` guarantees that an operator confirmation cannot be hijacked or misapplied:

1. **SHA-256 Parameter Signature**:
   $$\text{sig} = \text{SHA256}(\text{normalized\_op} + "|" + \text{normalized\_target} + "|" + \text{sorted\_json\_parameters})$$
2. **Single-Use Verification & Consumption**:
   When `verify_and_consume()` validates an approval, the token is atomically consumed and purged. It cannot be reused for a second operation.
3. **Mismatched Argument Rejection**:
   Approval granted for `delete_file` on `test.txt` with `force=False` immediately raises `ConfirmationRejectedError` if invoked with `critical.sys` or `force=True`.
4. **Time-to-Live (TTL)**:
   Tokens expire after a configurable duration (default 30 seconds). Expired tokens raise `ConfirmationTimeoutError`.
5. **Human-in-the-Loop Gateway Integration**:
   Can bridge directly into `InterventionGateway`, allowing automated resolution via registered operator handlers or UI callbacks.

---

## 5. Dependency Injection & ServiceContainer

All components are completely decoupled and support dependency injection without global state:

```python
from app.core.container import ServiceContainer
from app.skills.system import register_system_foundation

container = ServiceContainer()
register_system_foundation(container)

policy = container.resolve("system_security_policy")
confirmation_mgr = container.resolve("system_confirmation_manager")
```

Isolated container instances remain completely independent in concurrent and multi-tenant testing environments.

---

## 6. Observability & Planner Events

The following strongly-typed `PlannerEvent` subclasses broadcast system skill telemetry across the `PlannerEventBus`:
- `SystemSkillStarted`: Emitted when an authorized operation begins.
- `SystemSkillCompleted`: Emitted with duration and structured outcome data upon success.
- `SystemSkillFailed`: Emitted with error details upon unexpected failure.
- `SystemSkillConfirmationRequired`: Emitted when an operation awaits operator authorization.
- `SystemSkillPolicyRejected`: Emitted when a restricted operation is blocked.

---

## 7. Test Strategy & Results

Focused test suite in `tests/test_system_security.py` covers 20 unit tests:
1. SAFE operation allowed without confirmation.
2. CONFIRMATION_REQUIRED operation detected.
3. RESTRICTED operation rejected.
4. Arbitrary shell execution rejected (`cmd`, `bash`, `sh`, `cscript`).
5. Arbitrary PowerShell execution rejected (encoded commands, bypass flags, piping).
6. Critical process termination rejected (`explorer`, `csrss`, `dwm`, PID 0, 4, current PID).
7. Protected Windows directory operations rejected (`C:\Windows`, `System32`, `WinSxS`).
8. Path traversal rejected when `allowed_roots` is specified.
9. Valid safe paths accepted.
10. Confirmation approval allows the exact requested operation and prevents replay.
11. Confirmation rejection blocks execution.
12. Confirmation expiry blocks execution.
13. Approval for one operation cannot authorize a different operation or altered arguments.
14. Concurrent policy checks and confirmations are safe across multi-threading.
15. Architecture isolation and multi-container separation.
16. Integration with `InterventionGateway`.
17. Concrete skill execution of safe operation with `SystemSkillStarted` and `SystemSkillCompleted` events.
18. Multi-step confirmation request, approval, and execution flow.
19. Execution of restricted operation raising `SecurityPolicyViolationError` and emitting `SystemSkillPolicyRejected`.
20. Execution failure emitting `SystemSkillFailed` and raising `SkillExecutionError`.

---

## 8. Known Limitations & Next Steps

- **No Domain Skills Implemented Yet**: Sprint 22.1 provides solely the foundation, security policies, confirmation manager, and base skill contract.
- **Sprint 22.2 Roadmap**: Application Skills (`AppSkills`: `open_app`, `close_app`, `restart_app`, `is_app_running`, `list_running_apps`).
