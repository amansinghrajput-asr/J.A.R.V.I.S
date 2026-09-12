# Phase 22.2: Application Skills

> **Status**: COMPLETE & VERIFIED  
> **Release**: `v0.22.2`  
> **Prerequisites**: Phase 22.1 (System Skill Foundation & Security) Verified  
> **Scope**: Sprint 22.2 Application Skills ONLY

---

## 1. Executive Summary

Phase 22.2 delivers the **Application Skills** (`AppSkills`) subsystem, connecting J.A.R.V.I.S.'s cognitive planner directly to Windows application management, process inspection, safe process termination, and application lifecycle control.

All operations execute strictly under the Phase 22.1 `BaseSystemSkill` and `SystemSecurityPolicy` framework, preventing arbitrary command injection, protecting critical OS processes from termination, and guarding against Time-Of-Check to Time-Of-Use (TOCTOU) process ID recycling attacks.

---

## 2. Supported Operations

| Operation | Safety Classification | Confirmation Required | Description |
|---|---|---|---|
| `open_app` | `SAFE` | No | Launches an allowlisted desktop application via structured argument list (`subprocess.Popen` with `shell=False`). |
| `close_app` | `CONFIRMATION_REQUIRED` | **Yes** | Terminates an active application or process with mandatory interactive confirmation and critical process protection. |
| `restart_app` | `CONFIRMATION_REQUIRED` | **Yes** | Safely closes running instances and launches a fresh instance. Halts if closing or validation fails. |
| `is_app_running` | `SAFE` | No | Inspects active system processes to verify if an application is running, returning active PIDs. |
| `list_running_apps`| `SAFE` | No | Returns a clean, structured snapshot of active user-facing applications (filtering out critical system daemons). |

---

## 3. Application Resolution Strategy (`AppResolver`)

`AppResolver` provides deterministic, allowlist-based resolution without invoking an unrestricted shell:

1. **Direct Shell & Script Interpreter Rejection**:
   Attempts to launch shell binaries (`cmd`, `cmd.exe`, `powershell`, `powershell.exe`, `pwsh`, `bash`, `sh`, `cscript`, `wscript`, `mshta`) are blocked before execution with `SecurityPolicyViolationError`.
2. **Command Chaining & Token Injection Prevention**:
   Input strings containing shell operators (`;`, `&&`, `||`, `|`, `` ` ``, `$()`, `-enc`) are rejected immediately.
3. **Curated Safe Aliases**:
   Standard desktop applications are mapped to validated executables:
   - `notepad` -> `notepad.exe`
   - `calc` / `calculator` -> `calc.exe`
   - `paint` / `mspaint` -> `mspaint.exe`
   - `wordpad` -> `write.exe`
   - `chrome` -> `chrome.exe`
   - `edge` / `msedge` -> `msedge.exe`
   - `firefox` -> `firefox.exe`
   - `vscode` / `code` -> `code.cmd`
   - `spotify`, `slack`, `discord` -> standard executable names
4. **Dynamic Extensibility**:
   Additional application aliases can be safely registered via `AppResolver.register_alias(alias, path)` without global mutation.
5. **Path & PATH Validation**:
   Explicit executable paths and executables discovered via `shutil.which` are validated using `validate_path()` to ensure they exist, have safe extensions (`.exe`, `.cmd`), and do not reside within protected system roots (`C:\Windows\System32\cmd.exe`).

---

## 4. Process Inspection & TOCTOU Race Safety (`ProcessManager`)

Process termination can be vulnerable to race conditions where a target PID is terminated or exits, and the operating system recycles that same PID for an unrelated process before the kill command executes (TOCTOU).

`ProcessManager` mitigates this with a two-phase verification:
1. **Pre-Snapshot**:
   When `close_app` or `restart_app` is requested, `get_process_snapshots(target)` captures the matching PID, process name, and process creation timestamp (`create_time`).
2. **Re-Validation Before Kill**:
   Immediately before termination, the process table is re-inspected. If the PID now points to a different process name or creation time, termination of that PID is skipped to prevent collateral damage.
3. **Portability & Graceful Fallbacks**:
   - Uses `psutil` if installed for rich process inspection and polite `terminate()` / `wait()` / `kill()`.
   - Automatically falls back to standard library `tasklist` (for queries) and `taskkill` (for termination) if `psutil` is not installed, guaranteeing 100% operation without external dependency failures.

---

## 5. Security & Confirmation Workflow

### Critical Process Protection
Protected processes defined in `CRITICAL_SYSTEM_PROCESSES` (`explorer.exe`, `csrss.exe`, `smss.exe`, `wininit.exe`, `services.exe`, `lsass.exe`, `dwm.exe`, `svchost.exe`, PID 0, PID 4, and `os.getpid()`) cannot be targeted for termination. Any attempt raises `SecurityPolicyViolationError` and emits `SystemSkillPolicyRejected`.

### Interactive Confirmation Tokens
Destructive actions (`close_app`, `restart_app`) require a valid `confirmation_id` issued by `SystemConfirmationManager`. The token is cryptographically bound to the target application and parameters; approval granted for `notepad` cannot be reused to terminate any other process.

---

## 6. Performance Benchmark Results

Measured on Windows 11 with `benchmarks/app_skills_perf.py` (2,000 iterations each):

| Benchmark Item | Measured Latency | Architectural Budget | Status |
|---|---|---|---|
| **Skill Lookup & `can_handle`** | **1.35 µs** | < 100.0 µs | **PASS** |
| **Application Resolution** | **20.88 µs** | < 50.0 µs | **PASS** |
| **Security Policy Evaluation** | **19.35 µs** | < 50.0 µs | **PASS** |
| **`is_app_running` Dispatch** | **66.61 µs** | < 100.0 µs | **PASS** |
| **`list_running_apps` Dispatch** | **38.62 µs** | < 200.0 µs | **PASS** |
| **Confirmed `close_app` Dispatch** | **137.54 µs** | < 250.0 µs | **PASS** |

---

## 7. Verification & Regression Testing

### 7.1 Focused Application Skill Tests (`tests/test_app_skills.py`)
- **18/18 PASS** in **0.166s**
- Validated: allowlisted launch, invalid target rejection, shell injection blocking, cmd/powershell blocking, process running checks, process listing, confirmation enforcement, critical process protection, missing process handling, permission error handling, restart composition, PID reuse protection, event publishing, concurrent queries, and dependency injection isolation.

### 7.2 System Test Discovery (`test_system*.py`)
- **42/42 PASS** in **0.201s**

### 7.3 Metacognition Regression Suite (`test_metacognition_*.py`)
- **88/88 PASS** in **4.196s**

### 7.4 Repository-Wide Full Regression Suite
- **859/859 PASS** (all baseline + Phase 22.1 + Phase 22.2 tests).

---

## 8. Limitations & Scope Boundary

- **Application Skills Only**: File operations (`FileSkills`), system metrics (`SystemInfoSkills`), audio/power controls (`SystemControlSkills`), window state controls (`WindowSkills`), and web launch (`BrowserSkills`) belong strictly to subsequent Phase 22 sprints.
- **Mock Safety**: All automated tests execute against mocked subprocess and process manager interfaces to guarantee that test runs never manipulate real developer workstation applications.
