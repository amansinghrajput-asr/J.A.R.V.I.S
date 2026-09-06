"""System Telemetry and Resource Monitoring Subsystem for J.A.R.V.I.S.

Collects real-time host operating system metrics including CPU utilization,
memory consumption, disk space, battery status, and active processes,
publishing lifecycle events across the Event Bus.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import threading
from typing import Final, List, Optional

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore
    _HAS_PSUTIL = False

from app.automation.models import (
    BatteryMetrics,
    CpuMetrics,
    DiskMetrics,
    MemoryMetrics,
    ProcessInfo,
    SystemTelemetry,
)
from app.core.config import Settings, settings as default_settings
from app.core.container import ServiceContainer, container as default_container
from app.core.event_bus import EventBus, event_bus as default_event_bus
from app.core.logger import get_logger

EVENT_SYSTEM_TELEMETRY: Final[str] = "system.telemetry.collected"


class SystemMonitor:
    """Thread-safe host system telemetry and diagnostics collector.

    Leverages `psutil` when available with standard library fallbacks to inspect
    CPU, RAM, storage, power, and process health.
    """

    def __init__(
        self,
        config: Optional[Settings] = None,
        logger: Optional[logging.Logger] = None,
        container_instance: Optional[ServiceContainer] = None,
        event_bus_instance: Optional[EventBus] = None,
        *,
        auto_register_in_container: bool = True,
    ) -> None:
        """Initialize the SystemMonitor.

        Args:
            config: Optional Settings instance.
            logger: Optional Logger instance.
            container_instance: Optional ServiceContainer instance.
            event_bus_instance: Optional EventBus instance.
            auto_register_in_container: Whether to register self in ServiceContainer.
        """
        self._lock = threading.RLock()
        self._container = container_instance if container_instance is not None else default_container
        self._logger = logger if logger is not None else get_logger("AUTOMATION.SYSTEM")
        self._config = config if config is not None else default_settings
        self._event_bus = event_bus_instance if event_bus_instance is not None else default_event_bus

        if auto_register_in_container:
            self._container.register_singleton("system_monitor", self, allow_override=True)

    @property
    def has_psutil(self) -> bool:
        """Check if psutil is available in the current runtime environment."""
        return _HAS_PSUTIL and psutil is not None

    def get_cpu_metrics(self) -> CpuMetrics:
        """Retrieve CPU usage percentage and core counts.

        Returns:
            CpuMetrics instance.
        """
        with self._lock:
            if self.has_psutil:
                try:
                    percent = float(psutil.cpu_percent(interval=None))
                    logical = psutil.cpu_count(logical=True) or os.cpu_count() or 1
                    physical = psutil.cpu_count(logical=False) or logical
                    freq = None
                    try:
                        f = psutil.cpu_freq()
                        if f and f.current:
                            freq = float(f.current)
                    except Exception:
                        pass
                    return CpuMetrics(
                        percent=percent,
                        logical_cores=logical,
                        physical_cores=physical,
                        frequency_mhz=freq,
                    )
                except Exception as exc:
                    self._logger.debug("psutil cpu query failed: %s", exc)

            # Standard library fallback
            logical = os.cpu_count() or 1
            return CpuMetrics(percent=0.0, logical_cores=logical, physical_cores=logical)

    def get_memory_metrics(self) -> MemoryMetrics:
        """Retrieve virtual memory usage and limits.

        Returns:
            MemoryMetrics instance.
        """
        with self._lock:
            if self.has_psutil:
                try:
                    vmem = psutil.virtual_memory()
                    total_gb = vmem.total / (1024**3)
                    used_gb = vmem.used / (1024**3)
                    free_gb = vmem.available / (1024**3)
                    return MemoryMetrics(
                        total_gb=round(total_gb, 2),
                        used_gb=round(used_gb, 2),
                        free_gb=round(free_gb, 2),
                        percent=float(vmem.percent),
                    )
                except Exception as exc:
                    self._logger.debug("psutil memory query failed: %s", exc)

            # Fallback
            return MemoryMetrics(total_gb=16.0, used_gb=8.0, free_gb=8.0, percent=50.0)

    def get_disk_metrics(self, path: Optional[str] = None) -> DiskMetrics:
        """Retrieve disk space metrics for the specified path or primary drive.

        Args:
            path: Mount path or directory (defaults to root drive).

        Returns:
            DiskMetrics instance.
        """
        with self._lock:
            target_path = path or ("C:\\" if platform.system() == "Windows" else "/")
            try:
                usage = shutil.disk_usage(target_path)
                total_gb = usage.total / (1024**3)
                used_gb = usage.used / (1024**3)
                free_gb = usage.free / (1024**3)
                percent = (usage.used / usage.total * 100.0) if usage.total > 0 else 0.0
                return DiskMetrics(
                    total_gb=round(total_gb, 2),
                    used_gb=round(used_gb, 2),
                    free_gb=round(free_gb, 2),
                    percent=round(percent, 1),
                    mount_point=target_path,
                )
            except Exception as exc:
                self._logger.debug("Disk metrics query failed for '%s': %s", target_path, exc)
                return DiskMetrics(total_gb=0.0, used_gb=0.0, free_gb=0.0, percent=0.0, mount_point=target_path)

    def get_battery_metrics(self) -> Optional[BatteryMetrics]:
        """Retrieve battery status, percentage, and power source.

        Returns:
            BatteryMetrics instance if battery hardware is present, else None.
        """
        with self._lock:
            if self.has_psutil:
                try:
                    sensors = psutil.sensors_battery()
                    if sensors is not None:
                        secs = sensors.secsleft if sensors.secsleft >= 0 else None
                        return BatteryMetrics(
                            percent=float(sensors.percent),
                            power_plugged=sensors.power_plugged,
                            remaining_seconds=secs,
                        )
                except Exception as exc:
                    self._logger.debug("psutil battery query failed: %s", exc)
            return None

    def get_top_processes(self, limit: int = 5, sort_by: str = "cpu") -> List[ProcessInfo]:
        """Retrieve top running processes sorted by resource consumption.

        Args:
            limit: Maximum number of processes to return.
            sort_by: Metric to sort by ('cpu' or 'memory').

        Returns:
            List of ProcessInfo instances.
        """
        with self._lock:
            results: List[ProcessInfo] = []
            if not self.has_psutil:
                return results

            try:
                for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent", "status"]):
                    try:
                        info = proc.info
                        name = info.get("name") or f"PID-{info.get('pid')}"
                        results.append(
                            ProcessInfo(
                                pid=int(info.get("pid", 0)),
                                name=str(name),
                                cpu_percent=float(info.get("cpu_percent") or 0.0),
                                memory_percent=float(info.get("memory_percent") or 0.0),
                                status=str(info.get("status") or "running"),
                            )
                        )
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

                key_fn = (lambda p: p.cpu_percent) if sort_by == "cpu" else (lambda p: p.memory_percent)
                results.sort(key=key_fn, reverse=True)
                return results[:limit]
            except Exception as exc:
                self._logger.debug("Failed retrieving top processes: %s", exc)
                return results

    def collect_telemetry(self, publish: bool = True) -> SystemTelemetry:
        """Collect aggregated host telemetry and optionally publish event.

        Args:
            publish: If True, emits 'system.telemetry.collected' on the EventBus.

        Returns:
            SystemTelemetry dataclass instance.
        """
        with self._lock:
            telemetry = SystemTelemetry(
                cpu=self.get_cpu_metrics(),
                memory=self.get_memory_metrics(),
                disk=self.get_disk_metrics(),
                battery=self.get_battery_metrics(),
                platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
            )

            if publish and self._event_bus is not None:
                self._event_bus.publish(
                    EVENT_SYSTEM_TELEMETRY,
                    payload={
                        "cpu_percent": telemetry.cpu.percent,
                        "memory_percent": telemetry.memory.percent,
                        "disk_percent": telemetry.disk.percent,
                        "battery_percent": telemetry.battery.percent if telemetry.battery else None,
                        "summary": telemetry.summary(),
                    },
                    source="system_monitor",
                )

            return telemetry


# Module default instance
system_monitor = SystemMonitor(auto_register_in_container=False)

__all__ = [
    "EVENT_SYSTEM_TELEMETRY",
    "SystemMonitor",
    "system_monitor",
]
