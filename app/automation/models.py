"""OS Automation Subsystem Data Models and Exceptions.

Defines telemetry dataclasses, process descriptions, window states, and
the exception hierarchy for J.A.R.V.I.S OS Automation and System Management.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from app.core.container import JarvisException


# ------------------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------------------


class AutomationError(JarvisException):
    """Base exception for all OS automation errors."""


class ProcessOperationError(AutomationError):
    """Raised when an error occurs launching, querying, or terminating a process."""


class SecurityBlockedError(AutomationError):
    """Raised when an operation is blocked by security guardrails."""


# ------------------------------------------------------------------------------
# Metrics Dataclasses
# ------------------------------------------------------------------------------


@dataclass(frozen=True)
class CpuMetrics:
    """CPU telemetry metrics."""

    percent: float
    logical_cores: int
    physical_cores: int
    frequency_mhz: Optional[float] = None


@dataclass(frozen=True)
class MemoryMetrics:
    """Virtual memory telemetry metrics."""

    total_gb: float
    used_gb: float
    free_gb: float
    percent: float


@dataclass(frozen=True)
class DiskMetrics:
    """Disk space telemetry metrics."""

    total_gb: float
    used_gb: float
    free_gb: float
    percent: float
    mount_point: str = "/"


@dataclass(frozen=True)
class BatteryMetrics:
    """Battery telemetry metrics."""

    percent: float
    power_plugged: Optional[bool] = None
    remaining_seconds: Optional[int] = None


@dataclass(frozen=True)
class ProcessInfo:
    """Summary information for a running OS process."""

    pid: int
    name: str
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    status: str = "running"


@dataclass(frozen=True)
class WindowInfo:
    """Information regarding a desktop application window."""

    hwnd: int
    title: str
    pid: Optional[int] = None
    process_name: Optional[str] = None
    is_foreground: bool = False
    is_visible: bool = True


@dataclass
class SystemTelemetry:
    """Aggregated host system telemetry snapshot."""

    cpu: CpuMetrics
    memory: MemoryMetrics
    disk: DiskMetrics
    battery: Optional[BatteryMetrics] = None
    platform: str = ""
    timestamp: float = field(default_factory=time.time)

    def summary(self) -> str:
        """Return a clean, human-readable summary of system health."""
        msg = (
            f"CPU: {self.cpu.percent:.1f}% ({self.cpu.logical_cores} cores) | "
            f"RAM: {self.memory.percent:.1f}% ({self.memory.used_gb:.1f}/{self.memory.total_gb:.1f} GB) | "
            f"Disk: {self.disk.percent:.1f}% ({self.disk.used_gb:.1f}/{self.disk.total_gb:.1f} GB)"
        )
        if self.battery is not None:
            state = "Plugged in" if self.battery.power_plugged else "Discharging"
            msg += f" | Battery: {self.battery.percent:.0f}% ({state})"
        return msg


__all__ = [
    "AutomationError",
    "BatteryMetrics",
    "CpuMetrics",
    "DiskMetrics",
    "MemoryMetrics",
    "ProcessInfo",
    "ProcessOperationError",
    "SecurityBlockedError",
    "SystemTelemetry",
    "WindowInfo",
]
