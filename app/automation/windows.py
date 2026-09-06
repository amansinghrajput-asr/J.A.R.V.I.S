"""Window Management and Process Automation Subsystem for J.A.R.V.I.S.

Provides application launching, process inspection, process termination with
safety guardrails, and foreground window query capabilities on Windows and desktop OS.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import threading
from typing import Final, List, Optional, Union

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore
    _HAS_PSUTIL = False

from app.automation.guardrails import SecurityGuard, security_guard as default_security_guard
from app.automation.models import (
    ProcessInfo,
    ProcessOperationError,
    SecurityBlockedError,
    WindowInfo,
)
from app.core.config import Settings, settings as default_settings
from app.core.container import ServiceContainer, container as default_container
from app.core.event_bus import EventBus, event_bus as default_event_bus
from app.core.logger import get_logger

EVENT_APP_LAUNCHED: Final[str] = "automation.app.launched"
EVENT_PROCESS_TERMINATED: Final[str] = "automation.process.terminated"

# Common application name mappings for desktop environments
DEFAULT_APP_ALIASES: Final[dict[str, str]] = {
    "notepad": "notepad.exe",
    "calc": "calc.exe",
    "calculator": "calc.exe",
    "cmd": "cmd.exe",
    "terminal": "wt.exe",
    "powershell": "powershell.exe",
    "explorer": "explorer.exe",
    "paint": "mspaint.exe",
    "chrome": "chrome.exe",
    "google chrome": "chrome.exe",
    "edge": "msedge.exe",
    "firefox": "firefox.exe",
    "code": "code.cmd",
    "vscode": "code.cmd",
}


class WindowManager:
    """Thread-safe desktop window and process automation manager."""

    def __init__(
        self,
        security_guard_instance: Optional[SecurityGuard] = None,
        config: Optional[Settings] = None,
        logger: Optional[logging.Logger] = None,
        container_instance: Optional[ServiceContainer] = None,
        event_bus_instance: Optional[EventBus] = None,
        *,
        auto_register_in_container: bool = True,
    ) -> None:
        """Initialize the WindowManager.

        Args:
            security_guard_instance: Optional SecurityGuard instance.
            config: Optional Settings instance.
            logger: Optional Logger instance.
            container_instance: Optional ServiceContainer instance.
            event_bus_instance: Optional EventBus instance.
            auto_register_in_container: Whether to register self in ServiceContainer.
        """
        self._lock = threading.RLock()
        self._security = security_guard_instance if security_guard_instance is not None else default_security_guard
        self._logger = logger if logger is not None else get_logger("AUTOMATION.WINDOWS")
        self._config = config if config is not None else default_settings
        self._container = container_instance if container_instance is not None else default_container
        self._event_bus = event_bus_instance if event_bus_instance is not None else default_event_bus

        if auto_register_in_container and self._container is not None:
            self._container.register_singleton("window_manager", self, allow_override=True)

    @property
    def has_psutil(self) -> bool:
        """Check if psutil is available."""
        return _HAS_PSUTIL and psutil is not None

    def launch_application(self, app_name_or_path: str) -> bool:
        """Launch a desktop application by name, alias, or executable path.

        Args:
            app_name_or_path: Target application name, alias, or path.

        Returns:
            True if launched successfully.

        Raises:
            SecurityBlockedError: If application command is blocked.
            ProcessOperationError: If launching fails.
        """
        raw_target = str(app_name_or_path).strip()
        if not raw_target:
            raise ProcessOperationError("Cannot launch application: empty target specified.")

        self._security.validate_app_launch(raw_target)

        target_cmd = DEFAULT_APP_ALIASES.get(raw_target.lower(), raw_target)

        with self._lock:
            try:
                if platform.system() == "Windows":
                    # Use startfile if direct executable / path exists or run via Popen
                    subprocess.Popen(
                        target_cmd,
                        shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                else:
                    subprocess.Popen(
                        target_cmd,
                        shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )

                self._logger.info("Launched application: '%s' (resolved: '%s')", raw_target, target_cmd)

                if self._event_bus is not None:
                    self._event_bus.publish(
                        EVENT_APP_LAUNCHED,
                        payload={"app": raw_target, "command": target_cmd},
                        source="window_manager",
                    )
                return True
            except Exception as exc:
                msg = f"Failed to launch application '{raw_target}': {exc}"
                self._logger.error(msg)
                raise ProcessOperationError(msg) from exc

    def find_processes(self, query: Union[str, int]) -> List[ProcessInfo]:
        """Find running processes matching a name substring or PID.

        Args:
            query: Process name substring or PID integer.

        Returns:
            List of matching ProcessInfo instances.
        """
        with self._lock:
            matches: List[ProcessInfo] = []
            if not self.has_psutil:
                return matches

            try:
                if isinstance(query, int):
                    if psutil.pid_exists(query):
                        p = psutil.Process(query)
                        matches.append(
                            ProcessInfo(
                                pid=p.pid,
                                name=p.name(),
                                cpu_percent=p.cpu_percent(),
                                memory_percent=p.memory_percent(),
                                status=p.status(),
                            )
                        )
                    return matches

                query_lower = str(query).strip().lower()
                for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent", "status"]):
                    try:
                        p_name = str(p.info.get("name") or "")
                        if query_lower in p_name.lower():
                            matches.append(
                                ProcessInfo(
                                    pid=int(p.info.get("pid", 0)),
                                    name=p_name,
                                    cpu_percent=float(p.info.get("cpu_percent") or 0.0),
                                    memory_percent=float(p.info.get("memory_percent") or 0.0),
                                    status=str(p.info.get("status") or "running"),
                                )
                            )
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            except Exception as exc:
                self._logger.debug("Error finding processes for query '%s': %s", query, exc)

            return matches

    def terminate_process(self, target: Union[str, int], force: bool = False) -> int:
        """Terminate processes by PID or name, enforcing safety guardrails.

        Args:
            target: PID integer or process executable name string.
            force: Whether to forcefully terminate (kill).

        Returns:
            Number of terminated processes.

        Raises:
            SecurityBlockedError: If target is protected.
            ProcessOperationError: If termination fails.
        """
        self._security.validate_kill_target(target)

        with self._lock:
            if not self.has_psutil:
                raise ProcessOperationError("Cannot terminate process: psutil is not available.")

            procs_to_term = self.find_processes(target)
            if not procs_to_term:
                raise ProcessOperationError(f"No running process found matching '{target}'.")

            terminated_count = 0
            for info in procs_to_term:
                # Double-check guardrails per process PID and name
                self._security.validate_kill_target(info.pid)
                self._security.validate_kill_target(info.name)

                try:
                    proc = psutil.Process(info.pid)
                    if force:
                        proc.kill()
                    else:
                        proc.terminate()
                    proc.wait(timeout=2.0)
                    terminated_count += 1
                    self._logger.info("Terminated process %s (PID %d)", info.name, info.pid)
                except (psutil.NoSuchProcess, psutil.TimeoutExpired, psutil.AccessDenied) as exc:
                    self._logger.warning("Could not terminate PID %d (%s): %s", info.pid, info.name, exc)

            if self._event_bus is not None and terminated_count > 0:
                self._event_bus.publish(
                    EVENT_PROCESS_TERMINATED,
                    payload={"target": str(target), "count": terminated_count},
                    source="window_manager",
                )

            return terminated_count

    def get_foreground_window(self) -> Optional[WindowInfo]:
        """Query the currently focused desktop foreground window.

        Returns:
            WindowInfo if available on Windows, else None.
        """
        with self._lock:
            if platform.system() != "Windows":
                return None

            try:
                import ctypes
                user32 = ctypes.windll.user32
                hwnd = user32.GetForegroundWindow()
                if not hwnd:
                    return None

                length = user32.GetWindowTextLengthW(hwnd)
                buff = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buff, length + 1)
                title = buff.value

                pid = ctypes.c_ulong()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                p_name = None

                if self.has_psutil and pid.value:
                    try:
                        p_name = psutil.Process(pid.value).name()
                    except Exception:
                        pass

                return WindowInfo(
                    hwnd=int(hwnd),
                    title=title,
                    pid=int(pid.value) if pid.value else None,
                    process_name=p_name,
                    is_foreground=True,
                    is_visible=True,
                )
            except Exception as exc:
                self._logger.debug("Failed querying foreground window: %s", exc)
                return None


# Module default instance
window_manager = WindowManager(auto_register_in_container=False)

__all__ = [
    "DEFAULT_APP_ALIASES",
    "EVENT_APP_LAUNCHED",
    "EVENT_PROCESS_TERMINATED",
    "WindowManager",
    "window_manager",
]
