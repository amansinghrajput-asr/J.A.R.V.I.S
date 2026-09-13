# Phase 22.6: Window Management Skills Architecture & Specification

## 1. Overview & Architectural Integration

Phase 22.6 introduces the Windows **Window Management Skills** (`WindowSkills`) module into J.A.R.V.I.S., empowering the assistant to enumerate, inspect, focus, minimize, maximize, restore, and close application windows across the desktop environment without shell invocation or third-party binary dependencies.

```
                  ┌───────────────────────────────┐
                  │    Planner / Command Router   │
                  └───────────────┬───────────────┘
                                  │ can_handle(query) / execute()
                                  ▼
                  ┌───────────────────────────────┐
                  │      WindowSkills (p=60)      │
                  │   (subclasses BaseSystemSkill)│
                  └───────────────┬───────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
┌──────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│SystemSecurity    │    │ Windows user32 / │    │ Target Resolution   │
│Policy & Tiers    │    │ dwmapi Native    │    │ Title / App Name /  │
│Safe vs Confirmed │    │ Win32 CFFI APIs  │    │ Exact HWND Matching │
└──────────────────┘    └──────────────────┘    └─────────────────────┘
```

### Key Architectural Invariants
- **Subclasses `BaseSystemSkill`**: Conforms to the standard Phase 22 interface (`can_handle`, `parse_command`, `execute`, `_execute_operation`).
- **Direct Win32 API Interop**: Communicates directly with Windows `user32.dll` and `dwmapi.dll` via `ctypes.windll` with zero `shell=True` execution and zero external scripts.
- **Dependency Injection**: Fully isolated, accepts `SystemSecurityPolicy`, `SystemConfirmationManager`, and `ServiceContainer` without relying on global mutable state.
- **Window Cloaking Awareness**: Filters out invisible, zero-sized, and Windows 10/11 UWP cloaked background frames using `DwmGetWindowAttribute(DWMWA_CLOAKED)`.

---

## 2. Window Identification & Target Resolution

`WindowSkills` implements deterministic multi-strategy target window resolution in `_resolve_target_window(target)`:
1. **HWND Matching**: If the target query is numeric (e.g. `"65824"`), directly queries and validates the window handle.
2. **Exact Title Matching**: Case-insensitive exact match against window titles.
3. **Prefix and Substring Matching**: Substring search inside window titles (e.g., `"Notepad"` matches `"Untitled - Notepad"`).
4. **Process Name Resolution**: Matches against the executable image backing the window (e.g., `"chrome"` matches window titled `"GitHub - Google Chrome"` with process `chrome.exe`).
5. **Foreground / Active Window Targeting**: Direct targeting via `get_active_window`.

---

## 3. Supported Operations & Result Schemas

### 3.1 `list_windows` / `list_open_windows`
Enumerates visible, interactive desktop windows while filtering shell system tray and background worker handles.
- **Classification**: `SAFE`
- **Output Schema**:
```json
{
  "count": 2,
  "windows": [
    {
      "hwnd": 131244,
      "title": "Document - Notepad",
      "process_name": "notepad.exe",
      "pid": 5432,
      "is_active": true,
      "is_minimized": false,
      "is_maximized": false
    },
    {
      "hwnd": 65820,
      "title": "Calculator",
      "process_name": "CalculatorApp.exe",
      "pid": 7890,
      "is_active": false,
      "is_minimized": false,
      "is_maximized": false
    }
  ]
}
```

### 3.2 `get_active_window`
Inspects the current foreground window active on the user's desktop.
- **Classification**: `SAFE`
- **Output Schema**:
```json
{
  "hwnd": 131244,
  "title": "Document - Notepad",
  "process_name": "notepad.exe",
  "pid": 5432,
  "is_active": true
}
```

### 3.3 `focus_window(target)`
Brings the target window to the foreground, unminimizing/restoring it if necessary using `SetForegroundWindow` and `ShowWindow(SW_RESTORE)`.
- **Classification**: `SAFE`
- **Output Schema**:
```json
{
  "action": "focus_window",
  "target": "Document - Notepad",
  "hwnd": 131244,
  "success": true
}
```

### 3.4 `minimize_window(target)`
Minimizes the designated window to the taskbar using `ShowWindow(SW_MINIMIZE)`.
- **Classification**: `SAFE`
- **Output Schema**:
```json
{
  "action": "minimize_window",
  "target": "Document - Notepad",
  "hwnd": 131244,
  "success": true
}
```

### 3.5 `maximize_window(target)`
Maximizes the target window to fill the monitor viewport using `ShowWindow(SW_MAXIMIZE)`.
- **Classification**: `SAFE`
- **Output Schema**:
```json
{
  "action": "maximize_window",
  "target": "Document - Notepad",
  "hwnd": 131244,
  "success": true
}
```

### 3.6 `restore_window(target)`
Restores a minimized or maximized window to its normal geometry via `ShowWindow(SW_RESTORE)`.
- **Classification**: `SAFE`
- **Output Schema**:
```json
{
  "action": "restore_window",
  "target": "Document - Notepad",
  "hwnd": 131244,
  "success": true
}
```

### 3.7 `close_window(target)`
Gracefully asks the window to close by posting `WM_CLOSE` to its message queue.
- **Classification**: `CONFIRMATION_REQUIRED` / `SAFE` depending on security policy configuration.
- **Output Schema**:
```json
{
  "action": "close_window",
  "target": "Document - Notepad",
  "hwnd": 131244,
  "success": true,
  "method": "WM_CLOSE"
}
```

---

## 4. Safety & Confirmation Model

1. **Non-Destructive Defaults**: Focusing, listing, minimizing, and maximizing windows do not alter file data or terminate processes abruptly.
2. **Graceful Termination**: `close_window` posts `WM_CLOSE`, giving applications (like Notepad or Word) an opportunity to prompt the user to save unsaved documents rather than violently killing the process (`SIGKILL` / `TerminateProcess`).
3. **Critical System Protection**: Rejection of commands attempting to manipulate protected system windows (Taskbar, Desktop Shell, Program Manager).
4. **Exception Containment**: Win32 API errors are caught, transformed into structured error payloads, and logged without crashing the background service loop.

---

## 5. Planner Compatibility

`WindowSkills` maps directly into Planner tasks:
- **Action Aliases**: `list_windows`, `list_open_windows`, `get_active_window`, `focus_window`, `bring_to_front`, `minimize_window`, `maximize_window`, `restore_window`, `close_window`.
- **Structured Parameters**: Accepts `{"action": "focus_window", "target": "chrome"}` from `Executor._resolve_from_skill_manager`.
- **Conversational Parsing**: Employs regex extractors in `parse_command()` for queries like `"switch to notepad"`, `"minimize chrome"`, `"show me open windows"`.

---

## 6. Testing & Validation

`WindowSkills` is thoroughly verified by tests in `tests/test_window_skills.py` and `tests/test_system_skills_perf.py`:
- Mocking Win32 `user32` and `dwmapi` ctypes calls to ensure safety during automated testing.
- Target resolution unit tests for exact title, substring title, process name, and HWND matching.
- Concurrency isolation verification in `tests/test_architecture.py`.
- Benchmark execution validating average execution latency well within the sub-millisecond SLA.
