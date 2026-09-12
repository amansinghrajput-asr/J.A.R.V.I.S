"""System Information and Hardware Telemetry Skills for J.A.R.V.I.S. Phase 22.4.

Provides read-only queries for:
- get_cpu_info: CPU utilization, core counts, and frequency
- get_memory_info: Virtual RAM utilization (total, available, used, percent)
- get_disk_info: Storage space capacity and availability on mount/path
- get_battery_info: Power source, battery percentage, remaining seconds
- get_gpu_info: GPU name, total memory, and real-time VRAM allocation
- get_network_info: Network interface metrics and throughput counters
- get_system_summary: High-level aggregated telemetry snapshot
"""

from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import threading
from typing import Any, Dict, Final, List, Optional, Tuple, Union

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore
    _HAS_PSUTIL = False

try:
    import winreg
    _HAS_WINREG = True
except ImportError:
    winreg = None  # type: ignore
    _HAS_WINREG = False

from app.automation.system import SystemMonitor
from app.core.config import Settings
from app.core.container import ServiceContainer
from app.core.event_bus import EventBus
from app.core.logger import get_logger
from app.skills.base import SkillExecutionError
from app.skills.system.base_system_skill import BaseSystemSkill
from app.skills.system.security import (
    SecurityPolicyViolationError,
    SystemConfirmationManager,
    SystemSecurityPolicy,
    validate_path,
)

# Regex pattern matchers for natural language intent routing
_RE_CPU = re.compile(
    r"\b(?:what(?:\s+is|\s*'s)?\s+(?:my\s+)?cpu(?:\s+usage|\s+load|\s+info)?|cpu(?:\s+usage|\s+load|\s+info)?)\b",
    re.IGNORECASE,
)
_RE_MEMORY = re.compile(
    r"\b(?:how\s+much\s+(?:ram|memory)\s+is\s+being\s+used|what(?:\s+is|\s*'s)?\s+(?:my\s+)?(?:ram|memory)(?:\s+usage|\s+info)?|(?:ram|memory)(?:\s+usage|\s+info)?|check\s+ram)\b",
    re.IGNORECASE,
)
_RE_DISK = re.compile(
    r"\b(?:how\s+much\s+disk\s+space\s+is\s+available|what(?:\s+is|\s*'s)?\s+(?:my\s+)?disk(?:\s+usage|\s+info|\s+space)?|disk(?:\s+usage|\s+info|\s+space)?|storage\s+space)\b",
    re.IGNORECASE,
)
_RE_BATTERY = re.compile(
    r"\b(?:what(?:\s+is|\s*'s)?\s+(?:my\s+)?battery(?:\s+level|\s+status|\s+info)?|battery(?:\s+level|\s+status|\s+info)?|power\s+status)\b",
    re.IGNORECASE,
)
_RE_GPU = re.compile(
    r"\b(?:what\s+gpu(?:\s+do\s+i\s+have)?|what(?:\s+is|\s*'s)?\s+(?:my\s+)?gpu(?:\s+info|\s+status)?|gpu(?:\s+info|\s+status)?)\b",
    re.IGNORECASE,
)
_RE_NETWORK = re.compile(
    r"\b(?:what(?:\s+is|\s*'s)?\s+(?:my\s+)?network\s+status|network(?:\s+status|\s+info)?)\b",
    re.IGNORECASE,
)
_RE_SYSTEM_SUMMARY = re.compile(
    r"\b(?:give\s+me\s+a\s+system\s+summary|system\s+summary|system\s+overview|system\s+telemetry)\b",
    re.IGNORECASE,
)


class SystemInfoSkills(BaseSystemSkill):
    """Production System Information and Telemetry Skills for J.A.R.V.I.S. Phase 22.4.

    All operations are strictly read-only and categorized as SAFE.
    Uses psutil when available with native Windows and standard library fallbacks.
    """

    name: str = "system_info"
    description: str = (
        "Read-only system telemetry, hardware specifications, and resource diagnostics."
    )
    priority: int = 60
    tags: list[str] = [
        "system",
        "info",
        "telemetry",
        "cpu",
        "memory",
        "ram",
        "disk",
        "battery",
        "gpu",
        "network",
        "summary",
    ]
    permissions: set[str] = {"system:read"}

    def __init__(
        self,
        system_monitor_instance: Optional[SystemMonitor] = None,
        *,
        security_policy: Optional[SystemSecurityPolicy] = None,
        confirmation_manager: Optional[SystemConfirmationManager] = None,
        config: Optional[Settings] = None,
        logger: Optional[logging.Logger] = None,
        container: Optional[ServiceContainer] = None,
        event_bus: Optional[Union[EventBus, Any]] = None,
    ) -> None:
        """Initialize SystemInfoSkills instance.

        Args:
            system_monitor_instance: Optional injected SystemMonitor.
            security_policy: Optional injected SystemSecurityPolicy.
            confirmation_manager: Optional injected SystemConfirmationManager.
            config: Optional Settings configuration.
            logger: Custom logger instance.
            container: Dependency injection ServiceContainer.
            event_bus: EventBus instance for lifecycle broadcasting.
        """
        super().__init__(
            name=self.name,
            description=self.description,
            priority=self.priority,
            tags=self.tags,
            permissions=self.permissions,
            security_policy=security_policy,
            confirmation_manager=confirmation_manager,
            config=config,
            logger=logger or get_logger("SKILL.SYSTEM_INFO"),
            container=container,
            event_bus=event_bus,
        )
        self._lock = threading.RLock()
        self._system_monitor = system_monitor_instance

    @property
    def has_psutil(self) -> bool:
        """Check whether psutil is available in runtime."""
        return _HAS_PSUTIL and psutil is not None

    def can_handle(self, command: Any) -> bool:
        """Evaluate whether this skill can handle the given command."""
        op, _, _, _ = self.parse_command(command)
        if op in (
            "get_cpu_info",
            "get_memory_info",
            "get_disk_info",
            "get_battery_info",
            "get_gpu_info",
            "get_network_info",
            "get_system_summary",
            "cpu_info",
            "memory_info",
            "disk_info",
            "battery_info",
            "gpu_info",
            "network_info",
            "system_summary",
        ):
            return True

        if isinstance(command, str):
            clean = command.strip().lower()
            if _RE_CPU.search(clean):
                return True
            if _RE_MEMORY.search(clean):
                return True
            if _RE_DISK.search(clean):
                return True
            if _RE_BATTERY.search(clean):
                return True
            if _RE_GPU.search(clean):
                return True
            if _RE_NETWORK.search(clean):
                return True
            if _RE_SYSTEM_SUMMARY.search(clean):
                return True

        return False

    def parse_command(
        self, command: Any
    ) -> Tuple[str, Optional[str], Dict[str, Any], Optional[str]]:
        """Normalize arbitrary command payload into structured components."""
        if isinstance(command, dict):
            return super().parse_command(command)

        text = str(command or "").strip()
        clean = text.lower()

        if _RE_CPU.search(clean):
            return "get_cpu_info", None, {}, None
        if _RE_MEMORY.search(clean):
            return "get_memory_info", None, {}, None
        if _RE_DISK.search(clean):
            return "get_disk_info", None, {}, None
        if _RE_BATTERY.search(clean):
            return "get_battery_info", None, {}, None
        if _RE_GPU.search(clean):
            return "get_gpu_info", None, {}, None
        if _RE_NETWORK.search(clean):
            return "get_network_info", None, {}, None
        if _RE_SYSTEM_SUMMARY.search(clean):
            return "get_system_summary", None, {}, None

        return super().parse_command(command)

    def _execute_operation(
        self, operation: str, target: Optional[str], parameters: Dict[str, Any]
    ) -> Any:
        """Internal operation dispatcher for SystemInfoSkills."""
        op = operation.strip().lower()

        if op in ("get_cpu_info", "cpu_info", "cpu"):
            return self.get_cpu_info()

        if op in ("get_memory_info", "memory_info", "memory", "ram"):
            return self.get_memory_info()

        if op in ("get_disk_info", "disk_info", "disk", "storage"):
            path = target or parameters.get("path")
            return self.get_disk_info(path=path)

        if op in ("get_battery_info", "battery_info", "battery", "power"):
            return self.get_battery_info()

        if op in ("get_gpu_info", "gpu_info", "gpu"):
            return self.get_gpu_info()

        if op in ("get_network_info", "network_info", "network"):
            return self.get_network_info()

        if op in ("get_system_summary", "system_summary", "summary", "system_info"):
            return self.get_system_summary()

        raise NotImplementedError(f"Operation '{op}' is not supported by {self.name}.")

    # ---------------------------------------------------------------------------
    # Concrete Read-Only System Operations
    # ---------------------------------------------------------------------------

    def get_cpu_info(self) -> Dict[str, Any]:
        """Retrieve CPU utilization percentage, core counts, and frequency.

        Returns:
            Structured dictionary with keys: usage_percent, logical_cores, physical_cores, frequency_mhz.
        """
        with self._lock:
            if self.has_psutil:
                try:
                    usage = float(psutil.cpu_percent(interval=None))
                    logical = int(psutil.cpu_count(logical=True) or os.cpu_count() or 1)
                    physical = int(psutil.cpu_count(logical=False) or logical)
                    freq_mhz = None
                    try:
                        freq_stat = psutil.cpu_freq()
                        if freq_stat and freq_stat.current:
                            freq_mhz = round(float(freq_stat.current), 2)
                    except Exception:
                        pass
                    return {
                        "usage_percent": round(usage, 2),
                        "logical_cores": logical,
                        "physical_cores": physical,
                        "frequency_mhz": freq_mhz,
                    }
                except Exception as exc:
                    self.logger.debug("psutil cpu query failed: %s", exc)

            # Fallback to standard library
            logical = int(os.cpu_count() or 1)
            return {
                "usage_percent": 0.0,
                "logical_cores": logical,
                "physical_cores": logical,
                "frequency_mhz": None,
            }

    def get_memory_info(self) -> Dict[str, Any]:
        """Retrieve virtual memory metrics (total, available, used bytes, and usage percentage).

        Returns:
            Structured dictionary with keys: total_bytes, available_bytes, used_bytes, usage_percent.
        """
        with self._lock:
            if self.has_psutil:
                try:
                    vmem = psutil.virtual_memory()
                    return {
                        "total_bytes": int(vmem.total),
                        "available_bytes": int(vmem.available),
                        "used_bytes": int(vmem.used),
                        "usage_percent": round(float(vmem.percent), 2),
                    }
                except Exception as exc:
                    self.logger.debug("psutil memory query failed: %s", exc)

            # Windows native fallback using ctypes GlobalMemoryStatusEx
            if platform.system() == "Windows":
                try:
                    class MEMORYSTATUSEX(ctypes.Structure):
                        _fields_ = [
                            ("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                        ]

                    stat = MEMORYSTATUSEX()
                    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                        total = int(stat.ullTotalPhys)
                        avail = int(stat.ullAvailPhys)
                        used = total - avail
                        percent = round((used / total * 100.0), 2) if total > 0 else 0.0
                        return {
                            "total_bytes": total,
                            "available_bytes": avail,
                            "used_bytes": used,
                            "usage_percent": percent,
                        }
                except Exception as exc:
                    self.logger.debug("ctypes memory query failed: %s", exc)

            # Generic fallback
            return {
                "total_bytes": 0,
                "available_bytes": 0,
                "used_bytes": 0,
                "usage_percent": 0.0,
            }

    def get_disk_info(self, path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Retrieve filesystem capacity and utilization on the specified path or mount.

        Args:
            path: Target directory or drive mount point. Defaults to primary drive root.

        Returns:
            Structured dictionary with keys: path, total_bytes, used_bytes, free_bytes, usage_percent.

        Raises:
            SecurityPolicyViolationError: If path contains traversals, null bytes, or reserved names.
            FileNotFoundError: If the specified target path does not exist.
        """
        with self._lock:
            allowed_roots = getattr(self.security_policy, "_allowed_roots", None)
            if path is not None and str(path).strip():
                target_str = str(path).strip()
            elif allowed_roots:
                target_str = str(allowed_roots[0])
            else:
                target_str = "C:\\" if platform.system() == "Windows" else "/"

            # Path Security Validation
            canonical_path = validate_path(
                target_str,
                allowed_roots=allowed_roots,
                allow_system_dirs=True,
            )

            if not canonical_path.exists():
                raise FileNotFoundError(f"Target disk path does not exist: '{target_str}'")

            try:
                usage = shutil.disk_usage(str(canonical_path))
                total = int(usage.total)
                used = int(usage.used)
                free = int(usage.free)
                percent = round((used / total * 100.0), 2) if total > 0 else 0.0

                return {
                    "path": str(canonical_path),
                    "total_bytes": total,
                    "used_bytes": used,
                    "free_bytes": free,
                    "usage_percent": percent,
                }
            except Exception as exc:
                self.logger.error("Failed to query disk usage on '%s': %s", canonical_path, exc)
                raise SkillExecutionError(f"Disk query failed for '{canonical_path}': {exc}") from exc

    def get_battery_info(self) -> Dict[str, Any]:
        """Retrieve battery power status, percentage, and estimated time remaining.

        Returns:
            Structured dictionary with keys: available, percent, plugged, seconds_left.
        """
        with self._lock:
            if self.has_psutil:
                try:
                    sensors = psutil.sensors_battery()
                    if sensors is not None:
                        secs = int(sensors.secsleft) if sensors.secsleft >= 0 else None
                        return {
                            "available": True,
                            "percent": round(float(sensors.percent), 2),
                            "plugged": bool(sensors.power_plugged),
                            "seconds_left": secs,
                        }
                except Exception as exc:
                    self.logger.debug("psutil battery query failed: %s", exc)

            # No battery hardware or psutil unavailable
            return {
                "available": False,
                "percent": None,
                "plugged": None,
                "seconds_left": None,
            }

    def get_gpu_info(self) -> Dict[str, Any]:
        """Retrieve GPU hardware specifications and VRAM memory allocation.

        Queries nvidia-smi if available via direct subprocess without shell execution,
        falling back to Windows Registry display adapter enumeration.

        Returns:
            Structured dictionary with keys: available, name, memory_total_bytes, memory_used_bytes, memory_free_bytes.
        """
        with self._lock:
            # 1. Attempt nvidia-smi query for detailed VRAM
            smi_path = shutil.which("nvidia-smi")
            if not smi_path and platform.system() == "Windows":
                common_smi = Path(r"C:\Windows\System32\nvidia-smi.exe")
                if common_smi.exists():
                    smi_path = str(common_smi)

            if smi_path:
                try:
                    # Direct subprocess call without shell execution
                    res = subprocess.run(
                        [
                            smi_path,
                            "--query-gpu=name,memory.total,memory.used,memory.free",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=2.0,
                        check=False,
                    )
                    if res.returncode == 0 and res.stdout.strip():
                        line = res.stdout.strip().splitlines()[0]
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 4:
                            name = parts[0]
                            total_mb = float(parts[1])
                            used_mb = float(parts[2])
                            free_mb = float(parts[3])
                            return {
                                "available": True,
                                "name": name,
                                "memory_total_bytes": int(total_mb * 1024 * 1024),
                                "memory_used_bytes": int(used_mb * 1024 * 1024),
                                "memory_free_bytes": int(free_mb * 1024 * 1024),
                            }
                except Exception as exc:
                    self.logger.debug("nvidia-smi execution failed: %s", exc)

            # 2. Windows Registry fallback for display adapters
            if platform.system() == "Windows" and _HAS_WINREG and winreg is not None:
                try:
                    key_path = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as k:
                        i = 0
                        while True:
                            try:
                                sub = winreg.EnumKey(k, i)
                                i += 1
                                if not sub.isdigit():
                                    continue
                                with winreg.OpenKey(k, sub) as subkey:
                                    try:
                                        desc, _ = winreg.QueryValueEx(subkey, "DriverDesc")
                                        mem_total = 0
                                        try:
                                            mem, _ = winreg.QueryValueEx(subkey, "HardwareInformation.qwMemorySize")
                                            mem_total = int(mem)
                                        except FileNotFoundError:
                                            pass

                                        if desc:
                                            return {
                                                "available": True,
                                                "name": str(desc),
                                                "memory_total_bytes": mem_total if mem_total > 0 else None,
                                                "memory_used_bytes": None,
                                                "memory_free_bytes": None,
                                            }
                                    except FileNotFoundError:
                                        pass
                            except OSError:
                                break
                except Exception as exc:
                    self.logger.debug("winreg GPU query failed: %s", exc)

            # No GPU detected or query unavailable
            return {
                "available": False,
                "name": None,
                "memory_total_bytes": None,
                "memory_used_bytes": None,
                "memory_free_bytes": None,
            }

    def get_network_info(self) -> Dict[str, Any]:
        """Retrieve network interface statuses and I/O throughput counters.

        Returns:
            Structured dictionary with keys: interfaces, active_interfaces, bytes_sent, bytes_received.
        """
        with self._lock:
            interfaces: List[Dict[str, Any]] = []
            active_count = 0
            sent_bytes = 0
            recv_bytes = 0

            if self.has_psutil:
                try:
                    io = psutil.net_io_counters()
                    if io:
                        sent_bytes = int(io.bytes_sent)
                        recv_bytes = int(io.bytes_recv)

                    stats = psutil.net_if_stats()
                    for iface_name, stat in stats.items():
                        is_up = bool(stat.isup)
                        if is_up:
                            active_count += 1
                        interfaces.append({
                            "name": str(iface_name),
                            "is_up": is_up,
                            "speed_mbps": int(stat.speed) if stat.speed >= 0 else 0,
                        })
                except Exception as exc:
                    self.logger.debug("psutil network query failed: %s", exc)

            return {
                "interfaces": interfaces,
                "active_interfaces": active_count,
                "bytes_sent": sent_bytes,
                "bytes_received": recv_bytes,
            }

    def get_system_summary(self) -> Dict[str, Any]:
        """Generate an aggregated high-level telemetry and hardware summary snapshot.

        Returns:
            Structured dictionary combining CPU, RAM, Disk, Battery, GPU, and Network telemetry.
        """
        with self._lock:
            cpu = self.get_cpu_info()
            mem = self.get_memory_info()
            disk = self.get_disk_info()
            battery = self.get_battery_info()
            gpu = self.get_gpu_info()
            net = self.get_network_info()

            return {
                "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
                "cpu": {
                    "usage_percent": cpu["usage_percent"],
                    "logical_cores": cpu["logical_cores"],
                    "physical_cores": cpu["physical_cores"],
                },
                "memory": {
                    "usage_percent": mem["usage_percent"],
                    "total_gb": round(mem["total_bytes"] / (1024**3), 2) if mem["total_bytes"] else 0.0,
                    "available_gb": round(mem["available_bytes"] / (1024**3), 2) if mem["available_bytes"] else 0.0,
                },
                "disk": {
                    "path": disk["path"],
                    "usage_percent": disk["usage_percent"],
                    "total_gb": round(disk["total_bytes"] / (1024**3), 2) if disk["total_bytes"] else 0.0,
                    "free_gb": round(disk["free_bytes"] / (1024**3), 2) if disk["free_bytes"] else 0.0,
                },
                "battery": {
                    "available": battery["available"],
                    "percent": battery["percent"],
                    "plugged": battery["plugged"],
                },
                "gpu": {
                    "available": gpu["available"],
                    "name": gpu["name"],
                },
                "network": {
                    "active_interfaces": net["active_interfaces"],
                    "bytes_sent": net["bytes_sent"],
                    "bytes_received": net["bytes_received"],
                },
            }


__all__ = [
    "SystemInfoSkills",
]
