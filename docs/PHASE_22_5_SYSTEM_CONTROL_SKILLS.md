# Phase 22.5: System Control Skills Architecture & Specification

## 1. Overview & Architectural Integration

Phase 22.5 introduces safe Windows **System Control Skills** (`SystemControlSkills`) to J.A.R.V.I.S., allowing control over host volume, display brightness, workstation locking, and power transitions (sleep, restart, shutdown).

```
                  ┌───────────────────────────────┐
                  │    Planner / Command Router   │
                  └───────────────┬───────────────┘
                                  │ can_handle(query) / execute()
                                  ▼
                  ┌───────────────────────────────┐
                  │   SystemControlSkills (p=60)  │
                  │   (subclasses BaseSystemSkill)│
                  └───────────────┬───────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
┌──────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│SystemSecurity    │    │ Windows Native   │    │ SystemConfirmation  │
│Policy (3 Tiers)  │    │ APIs & Hardware  │    │ Manager (Tokens for │
│Safe vs Confirmed │    │ pycaw / sbc /    │    │ sleep, restart,     │
│                  │    │ user32 / powrprof│    │ shutdown, vol=100)  │
└──────────────────┘    └──────────────────┘    └─────────────────────┘
```

The subsystem:
- Subclasses `BaseSystemSkill` (Phase 22 foundational contract).
- Reuses `SystemSecurityPolicy` to strictly partition operations into `SAFE` vs `CONFIRMATION_REQUIRED`.
- Integrates with `SystemConfirmationManager` for single-use, parameter-pinned cryptographic confirmation tokens before power state changes occur.
- Enforces direct Windows APIs (`user32.LockWorkStation`, `PowrProf.SetSuspendState`, `pycaw.EndpointVolume`, `screen_brightness_control`) and direct executable array dispatch (`shutdown.exe` without shell).
- Completely forbids `shell=True`, `cmd.exe`, `powershell.exe`, and command string interpolation.
- Preserves zero global mutable state and follows Dependency Injection via `ServiceContainer`.

---

## 2. Supported Operations & Result Schemas

### 2.1 `get_volume`
Retrieves master audio volume and mute state.
- **Classification**: `SAFE`
- **Output Schema**:
```json
{
  "level": 45,
  "muted": false
}
```

### 2.2 `set_volume(level)`
Sets master audio volume between 0 and 100. Clamps out-of-bounds inputs. Automatically unmutes if volume is set to a positive value.
- **Classification**: `SAFE` for levels 0–99; `CONFIRMATION_REQUIRED` for level 100 (maximum volume protection).
- **Output Schema**:
```json
{
  "previous_level": 45,
  "current_level": 50,
  "muted": false
}
```

### 2.3 `get_brightness`
Queries display brightness across active monitors.
- **Classification**: `SAFE`
- **Output Schema**:
```json
{
  "supported": true,
  "level": 80,
  "displays": [80]
}
```
If display hardware lacks DDC/CI or driver brightness support (e.g. desktop monitor, VM):
```json
{
  "supported": false,
  "level": null,
  "displays": [],
  "error": "Brightness control unsupported on current display: ..."
}
```

### 2.4 `set_brightness(level)`
Adjusts display brightness between 0 and 100.
- **Classification**: `SAFE`
- **Output Schema**:
```json
{
  "supported": true,
  "level": 75
}
```

### 2.5 `lock_workstation`
Instantly locks the Windows workstation using the direct Windows `user32.LockWorkStation()` API.
- **Classification**: `SAFE` (non-destructive)
- **Output Schema**:
```json
{
  "locked": true,
  "method": "LockWorkStation"
}
```

### 2.6 `sleep`
Puts the workstation into suspend/standby mode via `PowrProf.SetSuspendState(0, 1, 0)`.
- **Classification**: `CONFIRMATION_REQUIRED` (mandatory single-use cryptographic token).
- **Output Schema**:
```json
{
  "action": "sleep",
  "executed": true
}
```

### 2.7 `restart`
Reboots the host machine via Windows shutdown API/executable without shell string interpolation.
- **Classification**: `CONFIRMATION_REQUIRED` (mandatory single-use cryptographic token).
- **Output Schema**:
```json
{
  "action": "restart",
  "executed": true
}
```

### 2.8 `shutdown`
Powers down the host machine.
- **Classification**: `CONFIRMATION_REQUIRED` (mandatory single-use cryptographic token).
- **Output Schema**:
```json
{
  "action": "shutdown",
  "executed": true
}
```

---

## 3. Security Policy & Confirmation Binding Architecture

### 3.1 Confirmation Enforcement
All destructive power operations (`sleep`, `restart`, `shutdown`) and extreme actions (`set_volume(level=100)`) strictly enforce interactive human approval through `SystemConfirmationManager`:
1. **Token Generation**: Upon receiving a command without a valid confirmation ID, `BaseSystemSkill.execute()` requests a confirmation token bound to `(operation, target, parameters)`.
2. **Cryptographic Parameter Hash**: A SHA-256 hash of the parameters is computed and saved with the token.
3. **Single-Use Consumption**: Once verified, the token is consumed. Attempts to reuse the token raise `ConfirmationRejectedError`.
4. **Cross-Action Isolation**: Approval for `restart` cannot execute `shutdown`. Approval for `shutdown` cannot execute `restart`.
5. **Zero Early Execution**: Underlying OS APIs (`_call_sleep_api`, `_call_restart_api`, `_call_shutdown_api`) are never invoked until confirmation validation succeeds.

---

## 4. Performance Benchmark Results

Measured on host hardware (`benchmarks/system_control_perf.py`):

| Operation | Mean (ms) | P50 (ms) | P95 (ms) | Architectural Budget | Status |
|---|---|---|---|---|---|
| `can_handle & routing` | **0.002 ms** | 0.002 ms | 0.002 ms | < 0.1 ms | **PASS** |
| `policy evaluation` | **0.002 ms** | 0.002 ms | 0.002 ms | < 0.5 ms | **PASS** |
| `get_volume dispatch` | **0.255 ms** | 0.235 ms | 0.311 ms | < 5.0 ms | **PASS** |
| `set_volume dispatch` | **0.230 ms** | 0.131 ms | 0.764 ms | < 5.0 ms | **PASS** |
| `get_brightness dispatch` | **31.722 ms** | 30.339 ms | 46.583 ms | < 100.0 ms | **PASS** |
| `lock_workstation dispatch` | **0.022 ms** | 0.022 ms | 0.030 ms | < 1.0 ms | **PASS** |
| `confirmation workflow` | **0.033 ms** | 0.028 ms | 0.084 ms | < 1.0 ms | **PASS** |

---

## 5. Verification & Testing Strategy

1. **Safety Isolation**: All tests in `tests/test_system_control_skills.py` use mocked OS call primitives (`_call_lock_api`, `_call_sleep_api`, `_call_restart_api`, `_call_shutdown_api`) to ensure the host machine never accidentally locks, sleeps, or reboots during automated test runs.
2. **Call Boundary Assertions**: Tests assert that OS API hooks are not invoked when tokens are missing, mismatched, expired, or modified.
3. **Hardware Fallbacks**: Tests verify that systems without audio endpoints or monitor brightness drivers return structured fallback data without raising unhandled exceptions.
4. **Test Counts**:
   - Focused System Control Tests: 40/40 PASS
   - System Tests Suite: 112/112 PASS
   - Metacognition Suite: 88/88 PASS
   - Full Repository Regression: 959/959 PASS with 0 regressions.

---

## 6. Known Limitations & Boundaries

1. **Desktop Display Brightness**: Desktop monitors connected via HDMI/DisplayPort may require DDC/CI enabled in monitor OSD firmware for brightness control. If disabled or unsupported, the skill returns `supported: false` with descriptive error.
2. **Phase 22.6+ Boundaries**: Window management, virtual desktops, browser automation, and voice integration belong to later phases and are strictly out of scope for Phase 22.5.
