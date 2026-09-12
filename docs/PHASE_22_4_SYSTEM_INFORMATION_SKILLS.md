# Phase 22.4: System Information Skills Architecture & Specification

## 1. Overview & Architecture Integration

Phase 22.4 introduces read-only **System Information Skills** (`SystemInfoSkills`) to J.A.R.V.I.S., enabling safe, instant introspection of host machine telemetry and hardware capabilities.

```
                  ┌───────────────────────────────┐
                  │    Planner / Command Router   │
                  └───────────────┬───────────────┘
                                  │ can_handle(query) / execute()
                                  ▼
                  ┌───────────────────────────────┐
                  │    SystemInfoSkills (p=60)    │
                  │   (subclasses BaseSystemSkill)│
                  └───────────────┬───────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
┌──────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│SystemSecurity    │    │ Hardware Probes  │    │  PlannerEventBus    │
│Policy (SAFE Tier)│    │ psutil / winreg /│    │ (Started/Completed/ │
│Path Validation   │    │ nvidia-smi       │    │  Failed telemetry)  │
└──────────────────┘    └──────────────────┘    └─────────────────────┘
```

The subsystem:
- Subclasses `BaseSystemSkill` (Phase 22 foundational contract).
- Reuses `SystemSecurityPolicy` to categorize all 7 operations as `SAFE` (0 confirmation required).
- Validates disk mount points using `validate_path()` to block traversal (`..`), null bytes, and Windows reserved names (`CON`, `PRN`, `AUX`, `NUL`).
- Provides fallback strategies ensuring hardware agnosticism (desktops without batteries, machines without discrete GPUs, or systems with non-psutil fallback).
- Maintains zero global mutable state and adheres strictly to Dependency Injection via `ServiceContainer`.

---

## 2. Supported Operations & Structured Result Schemas

All operations return deterministic, typed Python dictionaries (wrapped within standard `SystemSkillResult.data`):

### 2.1 `get_cpu_info`
Retrieves non-blocking CPU utilization, core architecture, and current clock frequency.
```json
{
  "usage_percent": 12.4,
  "logical_cores": 16,
  "physical_cores": 8,
  "frequency_mhz": 3194.0
}
```

### 2.2 `get_memory_info`
Queries host virtual memory allocations in bytes and percentage.
```json
{
  "total_bytes": 34098675712,
  "available_bytes": 18253611008,
  "used_bytes": 15845064704,
  "usage_percent": 46.5
}
```

### 2.3 `get_disk_info(path=None)`
Queries filesystem storage capacity and remaining free space for a validated directory or mount.
```json
{
  "path": "C:\\",
  "total_bytes": 1022839959552,
  "used_bytes": 512419979264,
  "free_bytes": 510419980288,
  "usage_percent": 50.1
}
```

### 2.4 `get_battery_info`
Inspects mobile power sensors. Returns `available: false` gracefully on desktop workstations.
```json
{
  "available": true,
  "percent": 88.0,
  "plugged": true,
  "seconds_left": null
}
```

### 2.5 `get_gpu_info`
Queries discrete or integrated GPUs via `nvidia-smi` (safe direct subprocess list) or Windows Registry (`SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}`).
```json
{
  "available": true,
  "name": "NVIDIA GeForce RTX 3050 A Laptop GPU",
  "memory_total_bytes": 4292870144,
  "memory_used_bytes": 0,
  "memory_free_bytes": 4081188864
}
```

### 2.6 `get_network_info`
Collects network adapter states and cumulative socket throughput.
```json
{
  "interfaces": [
    {"name": "Wi-Fi", "is_up": true, "speed_mbps": 866},
    {"name": "Ethernet", "is_up": false, "speed_mbps": 0}
  ],
  "active_interfaces": 1,
  "bytes_sent": 142049281,
  "bytes_received": 918239012
}
```

### 2.7 `get_system_summary`
Aggregates high-level telemetry across all subsystems into a single snapshot without redundant work.
```json
{
  "platform": "Windows 11 (AMD64)",
  "cpu": {"usage_percent": 12.4, "logical_cores": 16, "physical_cores": 8},
  "memory": {"usage_percent": 46.5, "total_gb": 31.76, "available_gb": 17.0},
  "disk": {"path": "C:\\", "usage_percent": 50.1, "total_gb": 952.6, "free_gb": 475.4},
  "battery": {"available": true, "percent": 88.0, "plugged": true},
  "gpu": {"available": true, "name": "NVIDIA GeForce RTX 3050 A Laptop GPU"},
  "network": {"active_interfaces": 1, "bytes_sent": 142049281, "bytes_received": 918239012}
}
```

---

## 3. Security & Safety Model

1. **Safety Tier**: All 7 operations are classified as `SAFE` in `SystemSecurityPolicy`.
2. **Zero Confirmation Required**: Read-only queries execute instantly without issuing interactive tokens.
3. **No Shell Execution**: GPU queries invoke `subprocess.run(["nvidia-smi", ...])` directly as an executable array (`shell=False`). No shell strings, command concatenation, or command interpreters (`cmd.exe`, `powershell.exe`) are used.
4. **Strict Path Security**: `get_disk_info` passes all user-supplied paths through `SystemSecurityPolicy.validate_path()`. Rejects `..` traversal, null bytes, Windows reserved devices (`CON`, `PRN`, `AUX`, `NUL`), and paths escaping `allowed_roots`.
5. **No Mutation**: Operations never modify filesystem files, edit registry values, change power plans, terminate tasks, or alter workstation settings.

---

## 4. Hardware Agnosticism & Fallbacks

The implementation never assumes specific hardware is attached:
- **Battery**: If `psutil.sensors_battery()` returns `None`, returns `available: false` with null fields.
- **GPU**: If `nvidia-smi` is absent, falls back to Windows Registry display adapter enumerator. If on non-Windows/no GPU, returns `available: false`.
- **CPU / RAM**: If `psutil` is missing, falls back to `os.cpu_count()` and Windows `ctypes.windll.kernel32.GlobalMemoryStatusEx`.

---

## 5. Performance Benchmark Results

Measured on Windows 11 host (physical hardware, zero mock speedups):

| Operation | Mean (ms) | P50 (ms) | P95 (ms) | Architectural Budget | Status |
|---|---|---|---|---|---|
| `can_handle & routing` | **0.002 ms** | 0.002 ms | 0.002 ms | < 0.1 ms | **PASS** |
| `get_cpu_info` | **0.403 ms** | 0.329 ms | 0.801 ms | < 5.0 ms | **PASS** |
| `get_memory_info` | **0.137 ms** | 0.124 ms | 0.240 ms | < 5.0 ms | **PASS** |
| `get_disk_info` | **0.292 ms** | 0.244 ms | 0.536 ms | < 5.0 ms | **PASS** |
| `get_battery_info` | **0.009 ms** | 0.009 ms | 0.014 ms | < 5.0 ms | **PASS** |
| `get_gpu_info` | **77.682 ms** | 77.894 ms | 80.234 ms | < 100.0 ms | **PASS** |
| `get_network_info` | **25.980 ms** | 23.317 ms | 47.164 ms | < 50.0 ms | **PASS** |
| `get_system_summary` | **103.150 ms** | 98.460 ms | 123.980 ms | < 150.0 ms | **PASS** |

---

## 6. Testing & Regression Results

- Focused unit test suite: `tests/test_system_info_skills.py` (30/30 PASS).
- Full system test suite: `tests/test_system*.py` (42/42 PASS).
- Metacognition test suite: `tests/test_metacognition_*.py` (88/88 PASS).
- Full repository regression suite: 919/919 tests PASS with 0 regressions.

---

## 7. Known Limitations & Boundaries

1. **Non-NVIDIA VRAM Usage**: On machines with non-NVIDIA GPUs (Intel/AMD), Windows Registry provides adapter name and total installed VRAM, but real-time allocated/free VRAM requires proprietary vendor APIs. In these cases, `memory_total_bytes` is provided while `memory_used_bytes` is `None`.
2. **Phase 22.5+ Scope Boundary**: Modifying volume, brightness, power state, rebooting, or controlling hardware belongs to Phase 22.5 System Control Skills and is strictly out of scope for Phase 22.4.
