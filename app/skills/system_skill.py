"""Production OS Automation and System Management Skill for J.A.R.V.I.S.

Provides native system telemetry queries (CPU, RAM, Disk, Battery), process management,
and desktop application launching with safety guardrails and high priority routing.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from typing import Any, Final, Optional, Union

from app.automation.guardrails import SecurityGuard, security_guard as default_security_guard
from app.automation.models import (
    AutomationError,
    ProcessOperationError,
    SecurityBlockedError,
)
from app.automation.system import SystemMonitor, system_monitor as default_system_monitor
from app.automation.windows import WindowManager, window_manager as default_window_manager
from app.core.config import Settings
from app.core.container import ServiceContainer
from app.core.event_bus import EventBus
from app.skills.base import BaseSkill, SkillExecutionError

DEFAULT_SYSTEM_SKILL_NAME: Final[str] = "system"
DEFAULT_SYSTEM_SKILL_DESCRIPTION: Final[str] = (
    "Operating system automation, telemetry monitoring, and application management."
)
DEFAULT_SYSTEM_SKILL_PRIORITY: Final[int] = 50

# Regex pattern matchers for deterministic intent routing
_RE_CPU = re.compile(r"\b(cpu|processor|cores?|cpu\s*usage|cpu\s*load)\b", re.IGNORECASE)
_RE_MEMORY = re.compile(r"\b(ram|memory|ram\s*usage|memory\s*usage|vmem)\b", re.IGNORECASE)
_RE_DISK = re.compile(r"\b(disk|storage|hard\s*drive|drive\s*space|disk\s*usage)\b", re.IGNORECASE)
_RE_BATTERY = re.compile(r"\b(battery|battery\s*level|power\s*status|charge)\b", re.IGNORECASE)
_RE_SYSTEM_STATUS = re.compile(
    r"\b(system\s*status|system\s*info|system\s*diagnostics|health\s*check|specs)\b",
    re.IGNORECASE,
)
_RE_TOP_PROCESSES = re.compile(
    r"\b(top\s*processes|list\s*processes|running\s*tasks|process\s*list|active\s*processes)\b",
    re.IGNORECASE,
)
_RE_KILL_PROCESS = re.compile(
    r"^(?:kill|terminate|stop|end|close)\s+(?:process\s+|task\s+)?([a-zA-Z0-9_\-\.]+)$",
    re.IGNORECASE,
)
_RE_OPEN_APP = re.compile(
    r"^(?:open|launch|start|run)\s+(.+)$",
    re.IGNORECASE,
)


class SystemSkill(BaseSkill):
    """Production OS Automation and System Management Skill.

    Configured with priority = 50, executing deterministically before the
    universal conversational fallback (AISkill at priority = -100).

    Handles:
    - CPU, RAM, Disk, and Battery diagnostics.
    - System health checks and telemetry collection.
    - Process listing and safe termination with security guardrails.
    - Application launching.
    """

    name: str = DEFAULT_SYSTEM_SKILL_NAME
    description: str = DEFAULT_SYSTEM_SKILL_DESCRIPTION
    version: str = "1.0.0"
    enabled: bool = True
    priority: int = DEFAULT_SYSTEM_SKILL_PRIORITY
    tags: list[str] = ["system", "os", "telemetry", "automation", "process", "windows"]
    permissions: set[str] = {"system:read", "system:execute"}

    def __init__(
        self,
        system_monitor_instance: Optional[SystemMonitor] = None,
        window_manager_instance: Optional[WindowManager] = None,
        security_guard_instance: Optional[SecurityGuard] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        version: Optional[str] = None,
        enabled: Optional[bool] = None,
        priority: Optional[int] = None,
        tags: Optional[list[str]] = None,
        permissions: Optional[Union[set[str], list[str]]] = None,
        config: Optional[Settings] = None,
        logger: Optional[logging.Logger] = None,
        container: Optional[ServiceContainer] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        """Initialize the SystemSkill instance.

        All parameters have sensible defaults to allow automatic discovery and
        instantiation by SkillManager.
        """
        super().__init__(
            name=name if name is not None else self.name,
            description=description if description is not None else self.description,
            version=version if version is not None else self.version,
            enabled=enabled if enabled is not None else self.enabled,
            priority=priority if priority is not None else self.priority,
            tags=tags if tags is not None else list(self.tags),
            permissions=permissions if permissions is not None else set(self.permissions),
            config=config,
            logger=logger,
            container=container,
            event_bus=event_bus,
        )
        self._system_monitor = system_monitor_instance
        self._window_manager = window_manager_instance
        self._security_guard = security_guard_instance
        self._lock = threading.RLock()
        self._execution_count: int = 0
        self._error_count: int = 0

    @property
    def system_monitor(self) -> SystemMonitor:
        """Resolve SystemMonitor from injection, container, or default."""
        if self._system_monitor is not None:
            return self._system_monitor

        container_inst = self.container
        if container_inst is not None and container_inst.exists("system_monitor"):
            return container_inst.resolve("system_monitor")

        return default_system_monitor

    @property
    def window_manager(self) -> WindowManager:
        """Resolve WindowManager from injection, container, or default."""
        if self._window_manager is not None:
            return self._window_manager

        container_inst = self.container
        if container_inst is not None and container_inst.exists("window_manager"):
            return container_inst.resolve("window_manager")

        return default_window_manager

    @property
    def security_guard(self) -> SecurityGuard:
        """Resolve SecurityGuard from injection, container, or default."""
        if self._security_guard is not None:
            return self._security_guard

        container_inst = self.container
        if container_inst is not None and container_inst.exists("security_guard"):
            return container_inst.resolve("security_guard")

        return default_security_guard

    @property
    def execution_count(self) -> int:
        """Return total successful executions."""
        with self._lock:
            return self._execution_count

    @property
    def error_count(self) -> int:
        """Return total failed executions."""
        with self._lock:
            return self._error_count

    def _extract_text(self, command: Any) -> str:
        """Extract clean query text from string, Intent, or dict."""
        if command is None:
            return ""
        if isinstance(command, str):
            return command.strip()
        if hasattr(command, "raw_command") or hasattr(command, "normalized_command"):
            raw = getattr(command, "raw_command", "") or ""
            norm = getattr(command, "normalized_command", "") or ""
            return str(raw if raw.strip() else norm).strip()
        if isinstance(command, dict):
            return str(
                command.get("query")
                or command.get("command")
                or command.get("text")
                or command.get("raw_command", "")
            ).strip()
        return str(command).strip()

    def can_handle(self, command: Any) -> bool:
        """Evaluate if the command matches system telemetry or automation intents.

        Args:
            command: Command payload.

        Returns:
            True if matching a system intent, False otherwise.
        """
        text = self._extract_text(command)
        if not text:
            return False

        clean = text.lower().strip()

        # 1. System status or full diagnostics
        if _RE_SYSTEM_STATUS.search(clean):
            return True

        # 2. Resource metrics: CPU, Memory, Disk, Battery
        if _RE_CPU.search(clean) or _RE_MEMORY.search(clean) or _RE_DISK.search(clean) or _RE_BATTERY.search(clean):
            return True

        # 3. Process management
        if _RE_TOP_PROCESSES.search(clean) or _RE_KILL_PROCESS.match(clean):
            return True

        # 4. App launching
        if _RE_OPEN_APP.match(clean):
            return True

        return False

    def execute(self, command: Any) -> str:
        """Execute the system automation command synchronously.

        Args:
            command: Command payload.

        Returns:
            User-facing conversational response string.

        Raises:
            SkillExecutionError: If execution encounters a critical error.
        """
        text = self._extract_text(command)
        if not text:
            raise SkillExecutionError("Command text cannot be empty for SystemSkill.")

        self.logger.info("Executing SystemSkill for query: '%s'", text)

        self.event_bus.publish(
            "system.skill.started",
            payload={"query": text, "timestamp": time.time()},
            source=f"skill.{self.name}",
        )

        try:
            response = self._handle_command(text)

            with self._lock:
                self._execution_count += 1

            self.event_bus.publish(
                "system.skill.completed",
                payload={"query": text, "response": response},
                source=f"skill.{self.name}",
            )
            return response
        except (SecurityBlockedError, ProcessOperationError, AutomationError) as exc:
            with self._lock:
                self._error_count += 1
            self.event_bus.publish(
                "system.skill.failed",
                payload={"query": text, "error": str(exc)},
                source=f"skill.{self.name}",
            )
            return f"System operation failed: {exc}"
        except Exception as exc:
            with self._lock:
                self._error_count += 1
            self.event_bus.publish(
                "system.skill.failed",
                payload={"query": text, "error": str(exc)},
                source=f"skill.{self.name}",
            )
            self.logger.error("Unhandled error in SystemSkill: %s", exc, exc_info=True)
            raise SkillExecutionError(f"System skill execution error: {exc}") from exc

    async def execute_async(self, command: Any) -> str:
        """Asynchronously execute system skill in a thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.execute, command)

    def _handle_command(self, text: str) -> str:
        """Dispatch text command to specific telemetry or automation handler."""
        clean = text.lower().strip()

        # 1. Kill / Terminate process
        kill_match = _RE_KILL_PROCESS.match(clean)
        if kill_match:
            target = kill_match.group(1).strip()
            # Check if target is a PID integer
            arg: Union[str, int] = int(target) if target.isdigit() else target
            count = self.window_manager.terminate_process(arg)
            return f"Successfully terminated {count} process(es) matching '{target}'."

        # 2. Open / Launch application
        open_match = _RE_OPEN_APP.match(clean)
        if open_match:
            app_target = open_match.group(1).strip()
            self.window_manager.launch_application(app_target)
            return f"I have launched {app_target} for you."

        # 3. System Status / Full Diagnostics
        if _RE_SYSTEM_STATUS.search(clean):
            telemetry = self.system_monitor.collect_telemetry(publish=True)
            return f"System Status: {telemetry.summary()}"

        # 4. Top processes
        if _RE_TOP_PROCESSES.search(clean):
            procs = self.system_monitor.get_top_processes(limit=5)
            if not procs:
                return "No process information currently available."
            lines = ["Top active processes:"]
            for p in procs:
                lines.append(f"- {p.name} (PID {p.pid}): CPU {p.cpu_percent:.1f}%, RAM {p.memory_percent:.1f}%")
            return "\n".join(lines)

        # 5. CPU Metrics
        if _RE_CPU.search(clean):
            cpu = self.system_monitor.get_cpu_metrics()
            freq_str = f" @ {cpu.frequency_mhz:.0f} MHz" if cpu.frequency_mhz else ""
            return (
                f"CPU utilization is currently {cpu.percent:.1f}% across "
                f"{cpu.logical_cores} cores ({cpu.physical_cores} physical){freq_str}."
            )

        # 6. Memory / RAM Metrics
        if _RE_MEMORY.search(clean):
            mem = self.system_monitor.get_memory_metrics()
            return (
                f"RAM usage is {mem.percent:.1f}%. "
                f"Used: {mem.used_gb:.1f} GB of {mem.total_gb:.1f} GB total "
                f"({mem.free_gb:.1f} GB available)."
            )

        # 7. Disk / Storage Metrics
        if _RE_DISK.search(clean):
            disk = self.system_monitor.get_disk_metrics()
            return (
                f"Disk usage on {disk.mount_point} is {disk.percent:.1f}%. "
                f"Used: {disk.used_gb:.1f} GB of {disk.total_gb:.1f} GB total "
                f"({disk.free_gb:.1f} GB free)."
            )

        # 8. Battery Metrics
        if _RE_BATTERY.search(clean):
            battery = self.system_monitor.get_battery_metrics()
            if battery is None:
                return "No battery detected; the system is running on direct AC power."
            state = "Plugged in (Charging)" if battery.power_plugged else "Discharging"
            time_left = ""
            if battery.remaining_seconds:
                hours = battery.remaining_seconds // 3600
                mins = (battery.remaining_seconds % 3600) // 60
                time_left = f", approximately {hours}h {mins}m remaining"
            return f"Battery is at {battery.percent:.0f}% ({state}{time_left})."

        # Fallback to general telemetry summary
        telemetry = self.system_monitor.collect_telemetry(publish=True)
        return f"System telemetry: {telemetry.summary()}"


# Module default instance
system_skill = SystemSkill()

__all__ = [
    "DEFAULT_SYSTEM_SKILL_DESCRIPTION",
    "DEFAULT_SYSTEM_SKILL_NAME",
    "DEFAULT_SYSTEM_SKILL_PRIORITY",
    "SystemSkill",
    "system_skill",
]
