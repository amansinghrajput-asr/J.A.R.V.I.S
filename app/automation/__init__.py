"""OS Automation Subsystem for J.A.R.V.I.S.

Provides system telemetry monitoring, application launching, safe process
management, security guardrails, and desktop automation primitives.
"""

from __future__ import annotations

from app.automation.guardrails import (
    CRITICAL_SYSTEM_PROCESSES,
    SecurityGuard,
    security_guard,
)
from app.automation.models import (
    AutomationError,
    BatteryMetrics,
    CpuMetrics,
    DiskMetrics,
    MemoryMetrics,
    ProcessInfo,
    ProcessOperationError,
    SecurityBlockedError,
    SystemTelemetry,
    WindowInfo,
)
from app.automation.system import (
    EVENT_SYSTEM_TELEMETRY,
    SystemMonitor,
    system_monitor,
)
from app.automation.windows import (
    DEFAULT_APP_ALIASES,
    EVENT_APP_LAUNCHED,
    EVENT_PROCESS_TERMINATED,
    WindowManager,
    window_manager,
)
from app.automation.input import (
    InputEvent,
    MockInputBackend,
    VirtualInputBackend,
    Win32InputBackend,
)
from app.automation.visual_action_adapter import (
    ActionPreflightValidator,
    PreflightCheckResult,
    VisualActionAdapter,
    VisualActionResult,
    VisualActionResultStatus,
)

__all__ = [
    "ActionPreflightValidator",
    "AutomationError",
    "BatteryMetrics",
    "CRITICAL_SYSTEM_PROCESSES",
    "CpuMetrics",
    "DEFAULT_APP_ALIASES",
    "DiskMetrics",
    "EVENT_APP_LAUNCHED",
    "EVENT_PROCESS_TERMINATED",
    "EVENT_SYSTEM_TELEMETRY",
    "InputEvent",
    "MemoryMetrics",
    "MockInputBackend",
    "PreflightCheckResult",
    "ProcessInfo",
    "ProcessOperationError",
    "SecurityBlockedError",
    "SecurityGuard",
    "SystemMonitor",
    "SystemTelemetry",
    "VirtualInputBackend",
    "VisualActionAdapter",
    "VisualActionResult",
    "VisualActionResultStatus",
    "Win32InputBackend",
    "WindowInfo",
    "WindowManager",
    "security_guard",
    "system_monitor",
    "window_manager",
]
