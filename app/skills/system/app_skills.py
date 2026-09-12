"""Application Skills for J.A.R.V.I.S. Phase 22.2.

Provides secure desktop application launching, process inspection, process termination,
and application restarting, integrated with BaseSystemSkill and SystemSecurityPolicy.
"""

from __future__ import annotations

import csv
import io
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from typing import Any, Dict, Final, Iterable, List, Optional, Set, Tuple, Union

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore
    _HAS_PSUTIL = False

from app.core.config import Settings
from app.core.container import ServiceContainer
from app.core.event_bus import EventBus
from app.core.logger import get_logger
from app.skills.base import SkillExecutionError
from app.skills.system.base_system_skill import BaseSystemSkill, SystemSkillResult
from app.skills.system.security import (
    BLOCKED_SHELL_COMMANDS,
    BLOCKED_SHELL_PATTERNS,
    ConfirmationRequiredError,
    SecurityPolicyViolationError,
    SystemConfirmationManager,
    SystemSecurityPolicy,
    is_critical_process,
    is_shell_command,
    validate_path,
)

logger = get_logger("SYSTEM.APP_SKILLS")

DEFAULT_SAFE_APP_ALIASES: Final[Dict[str, str]] = {
    "notepad": "notepad.exe",
    "calc": "calc.exe",
    "calculator": "calc.exe",
    "paint": "mspaint.exe",
    "mspaint": "mspaint.exe",
    "wordpad": "write.exe",
    "write": "write.exe",
    "explorer": "explorer.exe",
    "taskmgr": "taskmgr.exe",
    "chrome": "chrome.exe",
    "google chrome": "chrome.exe",
    "edge": "msedge.exe",
    "msedge": "msedge.exe",
    "firefox": "firefox.exe",
    "code": "code.cmd",
    "vscode": "code.cmd",
    "spotify": "spotify.exe",
    "slack": "slack.exe",
    "discord": "discord.exe",
}

# Regex intent matchers
_RE_OPEN = re.compile(r"^(?:open|launch|start|run)\s+(?:app\s+|application\s+)?(.+)$", re.IGNORECASE)
_RE_CLOSE = re.compile(r"^(?:close|terminate|kill|stop|end)\s+(?:app\s+|process\s+|task\s+)?(.+)$", re.IGNORECASE)
_RE_RESTART = re.compile(r"^(?:restart|reboot)\s+(?:app\s+|application\s+)?(.+)$", re.IGNORECASE)
_RE_IS_RUNNING = re.compile(r"^(?:is|check\s+if)\s+(.+?)\s+(?:is\s+)?running\??$", re.IGNORECASE)
_RE_LIST_RUNNING = re.compile(
    r"^(?:list|show|get)\s+(?:all\s+)?(?:running\s+)?(?:apps|applications|processes|tasks)$",
    re.IGNORECASE,
)


class AppResolver:
    """Thread-safe validator and resolver for desktop applications."""

    def __init__(
        self,
        custom_allowlist: Optional[Dict[str, str]] = None,
        allowed_dirs: Optional[Iterable[Union[str, Path]]] = None,
        logger_instance: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize the AppResolver.

        Args:
            custom_allowlist: Optional custom alias-to-executable mappings.
            allowed_dirs: Optional directories where custom executables are permitted.
            logger_instance: Custom logger.
        """
        self._lock = threading.RLock()
        self._logger = logger_instance or get_logger("SYSTEM.APP_RESOLVER")
        self._aliases: Dict[str, str] = {k.lower(): v for k, v in DEFAULT_SAFE_APP_ALIASES.items()}
        if custom_allowlist:
            for k, v in custom_allowlist.items():
                self._aliases[k.lower()] = v

        self._allowed_dirs = [Path(d).expanduser().resolve() for d in allowed_dirs] if allowed_dirs else []

    def register_alias(self, alias: str, executable_path: str) -> None:
        """Register a new application alias in the allowlist."""
        with self._lock:
            self._aliases[alias.strip().lower()] = executable_path.strip()

    def resolve(self, name_or_path: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Resolve and validate an application identifier or path.

        Args:
            name_or_path: Name alias, executable filename, or full path.

        Returns:
            Tuple of (is_valid, resolved_executable_or_cmd, rejection_reason).
        """
        raw = str(name_or_path or "").strip()
        if not raw:
            return False, None, "Application identifier cannot be empty."

        # 1. Shell command checks
        if is_shell_command(raw):
            return False, None, f"Application identifier '{raw}' contains prohibited shell tokens or interpreters."

        first_token = raw.split()[0].lower()
        if Path(first_token).name.lower() in BLOCKED_SHELL_COMMANDS:
            return False, None, f"Launching shell interpreter '{first_token}' is prohibited."

        with self._lock:
            # 2. Check alias allowlist
            clean_name = raw.lower()
            if clean_name in self._aliases:
                resolved_target = self._aliases[clean_name]
                return True, resolved_target, None

            # 3. Check if input is a direct path
            if "\\" in raw or "/" in raw:
                try:
                    p = validate_path(raw, allow_system_dirs=False)
                    if not p.is_file():
                        return False, None, f"Target file '{raw}' does not exist or is not a regular file."

                    if p.suffix.lower() not in (".exe", ".cmd", ".bat"):
                        return False, None, f"Target '{raw}' is not an executable file extension."

                    return True, str(p), None
                except SecurityPolicyViolationError as exc:
                    return False, None, str(exc)

            # 4. Check safe executable in PATH via shutil.which
            which_result = shutil.which(raw)
            if which_result:
                try:
                    p = validate_path(which_result, allow_system_dirs=False)
                    p_name = p.name.lower()
                    if p_name in BLOCKED_SHELL_COMMANDS:
                        return False, None, f"Executable '{p_name}' in PATH is a prohibited shell."
                    return True, str(p), None
                except SecurityPolicyViolationError as exc:
                    return False, None, str(exc)

            # If not in allowlist and cannot resolve
            return False, None, f"Application '{raw}' is not recognized or not allowlisted."


class ProcessManager:
    """Thread-safe inspector and manager for Windows processes.

    Provides process queries, listing, and race-safe termination with TOCTOU guards.
    """

    def __init__(
        self,
        security_policy: Optional[SystemSecurityPolicy] = None,
        logger_instance: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize ProcessManager."""
        self._lock = threading.RLock()
        self._security = security_policy or SystemSecurityPolicy()
        self._logger = logger_instance or get_logger("SYSTEM.PROCESS_MGR")

    @property
    def has_psutil(self) -> bool:
        """Check if psutil is available."""
        return _HAS_PSUTIL and psutil is not None

    def get_process_snapshots(self, target: Union[str, int]) -> List[Dict[str, Any]]:
        """Capture active process snapshots matching target name or PID.

        Used for TOCTOU identity verification before termination.
        """
        snapshots: List[Dict[str, Any]] = []

        with self._lock:
            # 1. Using psutil if available
            if self.has_psutil:
                try:
                    if isinstance(target, int) or (isinstance(target, str) and target.strip().isdigit()):
                        pid = int(target)
                        if psutil.pid_exists(pid):
                            p = psutil.Process(pid)
                            snapshots.append({
                                "pid": p.pid,
                                "name": p.name(),
                                "create_time": getattr(p, "create_time", lambda: 0.0)(),
                                "status": p.status(),
                            })
                        return snapshots

                    target_lower = str(target).strip().lower()
                    for p in psutil.process_iter(["pid", "name", "create_time", "status"]):
                        try:
                            p_name = str(p.info.get("name") or "")
                            if (
                                target_lower == p_name.lower()
                                or target_lower == p_name.lower().replace(".exe", "")
                                or target_lower in p_name.lower()
                            ):
                                snapshots.append({
                                    "pid": int(p.info.get("pid", 0)),
                                    "name": p_name,
                                    "create_time": float(p.info.get("create_time") or 0.0),
                                    "status": str(p.info.get("status") or "running"),
                                })
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
                except Exception as exc:
                    self._logger.debug("psutil get_process_snapshots error: %s", exc)
                return snapshots

            # 2. Standard library fallback using tasklist on Windows
            try:
                out = subprocess.check_output(
                    ["tasklist", "/FO", "CSV", "/NH"],
                    shell=False,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                )
                reader = csv.reader(io.StringIO(out))
                target_str = str(target).strip().lower()
                is_pid_query = target_str.isdigit()

                for row in reader:
                    if len(row) >= 2:
                        p_name = row[0].strip()
                        p_pid_str = row[1].strip()
                        if not p_pid_str.isdigit():
                            continue
                        pid_val = int(p_pid_str)

                        matched = False
                        if is_pid_query and pid_val == int(target_str):
                            matched = True
                        elif not is_pid_query and (
                            target_str == p_name.lower()
                            or target_str == p_name.lower().replace(".exe", "")
                            or target_str in p_name.lower()
                        ):
                            matched = True

                        if matched:
                            snapshots.append({
                                "pid": pid_val,
                                "name": p_name,
                                "create_time": 0.0,
                                "status": "running",
                            })
            except Exception as exc:
                self._logger.debug("tasklist fallback error: %s", exc)

            return snapshots

    def is_running(self, target: Union[str, int]) -> Tuple[bool, List[int]]:
        """Check whether target process is currently active."""
        snapshots = self.get_process_snapshots(target)
        pids = [s["pid"] for s in snapshots]
        return len(pids) > 0, pids

    def list_running(self, user_only: bool = True, limit: int = 50) -> List[Dict[str, Any]]:
        """List active user-relevant applications."""
        results: List[Dict[str, Any]] = []

        with self._lock:
            if self.has_psutil:
                try:
                    for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent", "status"]):
                        try:
                            name = str(p.info.get("name") or "")
                            pid = int(p.info.get("pid", 0))

                            if user_only and is_critical_process(name):
                                continue

                            results.append({
                                "pid": pid,
                                "name": name,
                                "cpu_percent": float(p.info.get("cpu_percent") or 0.0),
                                "memory_percent": float(p.info.get("memory_percent") or 0.0),
                                "status": str(p.info.get("status") or "running"),
                            })
                            if len(results) >= limit:
                                break
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
                except Exception as exc:
                    self._logger.debug("psutil list_running error: %s", exc)
                return results

            # Tasklist fallback
            try:
                out = subprocess.check_output(
                    ["tasklist", "/FO", "CSV", "/NH"],
                    shell=False,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                )
                reader = csv.reader(io.StringIO(out))
                for row in reader:
                    if len(row) >= 2:
                        p_name = row[0].strip()
                        p_pid = int(row[1].strip()) if row[1].strip().isdigit() else 0

                        if user_only and is_critical_process(p_name):
                            continue

                        results.append({
                            "pid": p_pid,
                            "name": p_name,
                            "cpu_percent": 0.0,
                            "memory_percent": 0.0,
                            "status": "running",
                        })
                        if len(results) >= limit:
                            break
            except Exception as exc:
                self._logger.debug("tasklist list_running fallback error: %s", exc)

            return results

    def terminate_processes(
        self,
        target: Union[str, int],
        force: bool = False,
        timeout: float = 3.0,
        expected_snapshots: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Terminate processes matching target, validating process identity against TOCTOU race.

        Args:
            target: Process name or PID.
            force: Force termination flag.
            timeout: Wait timeout for process exit.
            expected_snapshots: Optional pre-captured snapshots to verify PID identity.

        Returns:
            Dict containing termination summary.
        """
        # Guardrail check against target identifier
        if is_critical_process(target):
            raise SecurityPolicyViolationError(
                f"Termination blocked: '{target}' is a protected system process."
            )

        with self._lock:
            # Re-fetch snapshots immediately before killing to prevent TOCTOU race
            current_snapshots = self.get_process_snapshots(target)
            if not current_snapshots:
                return {
                    "target": target,
                    "closed": False,
                    "terminated_count": 0,
                    "pids": [],
                    "reason": "Process not found",
                }

            # If expected_snapshots provided, filter out recycled PIDs whose identity changed
            procs_to_terminate: List[Dict[str, Any]] = []
            if expected_snapshots:
                expected_map = {s["pid"]: s for s in expected_snapshots}
                for curr in current_snapshots:
                    pid = curr["pid"]
                    if pid in expected_map:
                        exp = expected_map[pid]
                        # Verify process name still matches
                        if curr["name"].lower() != exp["name"].lower():
                            self._logger.warning(
                                "PID %d identity changed from '%s' to '%s'; skipping to prevent race termination.",
                                pid, exp["name"], curr["name"]
                            )
                            continue
                    procs_to_terminate.append(curr)
            else:
                procs_to_terminate = current_snapshots

            terminated_pids: List[int] = []

            # 1. Termination via psutil
            if self.has_psutil:
                for proc_info in procs_to_terminate:
                    pid = proc_info["pid"]
                    p_name = proc_info["name"]

                    if is_critical_process(pid) or is_critical_process(p_name):
                        self._logger.warning("Skipping protected process PID %d (%s)", pid, p_name)
                        continue

                    try:
                        p = psutil.Process(pid)
                        # Re-verify process name before kill
                        if p.name().lower() != p_name.lower():
                            continue

                        if force:
                            p.kill()
                        else:
                            p.terminate()

                        try:
                            p.wait(timeout=timeout)
                        except psutil.TimeoutExpired:
                            p.kill()
                            p.wait(timeout=1.0)

                        terminated_pids.append(pid)
                    except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
                        self._logger.debug("Termination exception on PID %d: %s", pid, exc)

                return {
                    "target": target,
                    "closed": len(terminated_pids) > 0,
                    "terminated_count": len(terminated_pids),
                    "pids": terminated_pids,
                }

            # 2. Fallback via taskkill
            for proc_info in procs_to_terminate:
                pid = proc_info["pid"]
                p_name = proc_info["name"]

                if is_critical_process(pid) or is_critical_process(p_name):
                    continue

                cmd = ["taskkill", "/PID", str(pid)]
                if force:
                    cmd.append("/F")

                try:
                    subprocess.run(
                        cmd,
                        shell=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=timeout,
                        check=False,
                    )
                    terminated_pids.append(pid)
                except Exception as exc:
                    self._logger.debug("taskkill exception on PID %d: %s", pid, exc)

            return {
                "target": target,
                "closed": len(terminated_pids) > 0,
                "terminated_count": len(terminated_pids),
                "pids": terminated_pids,
            }


class AppSkills(BaseSystemSkill):
    """Production desktop application skills for J.A.R.V.I.S.

    Handles:
    - open_app: Launch allowlisted desktop applications via structured subprocess
    - close_app: Safely terminate application processes with confirmation
    - restart_app: Safely close and restart applications
    - is_app_running: Check process execution state
    - list_running_apps: Enumerate active user-facing applications
    """

    name: str = "app"
    description: str = "Application lifecycle management, launching, process inspection, and safe termination."
    priority: int = 60
    tags: list[str] = ["app", "process", "system", "windows", "lifecycle"]
    permissions: set[str] = {"system:read", "system:execute"}

    def __init__(
        self,
        *,
        app_resolver: Optional[AppResolver] = None,
        process_manager: Optional[ProcessManager] = None,
        security_policy: Optional[SystemSecurityPolicy] = None,
        confirmation_manager: Optional[SystemConfirmationManager] = None,
        config: Optional[Settings] = None,
        logger: Optional[logging.Logger] = None,
        container: Optional[ServiceContainer] = None,
        event_bus: Optional[Union[EventBus, Any]] = None,
    ) -> None:
        """Initialize AppSkills instance."""
        super().__init__(
            name=self.name,
            description=self.description,
            priority=self.priority,
            tags=self.tags,
            permissions=self.permissions,
            security_policy=security_policy,
            confirmation_manager=confirmation_manager,
            config=config,
            logger=logger or get_logger("SKILL.APP"),
            container=container,
            event_bus=event_bus,
        )
        self._resolver = app_resolver or AppResolver(logger_instance=self.logger)
        self._process_manager = process_manager or ProcessManager(
            security_policy=self.security_policy,
            logger_instance=self.logger,
        )

    @property
    def resolver(self) -> AppResolver:
        """Return the active application resolver."""
        return self._resolver

    @property
    def process_manager(self) -> ProcessManager:
        """Return the active process manager."""
        return self._process_manager

    def can_handle(self, command: Any) -> bool:
        """Evaluate whether this skill can handle the given command."""
        op, target, _, _ = self.parse_command(command)

        if op in (
            "open_app",
            "launch_app",
            "close_app",
            "terminate_process",
            "kill_process",
            "restart_app",
            "is_app_running",
            "list_running_apps",
        ):
            return True

        # Check text string matchers
        if isinstance(command, str):
            cmd_clean = command.strip().lower()
            if _RE_OPEN.match(cmd_clean):
                return True
            if _RE_CLOSE.match(cmd_clean):
                return True
            if _RE_RESTART.match(cmd_clean):
                return True
            if _RE_IS_RUNNING.match(cmd_clean):
                return True
            if _RE_LIST_RUNNING.match(cmd_clean):
                return True

        return False

    def parse_command(
        self, command: Any
    ) -> Tuple[str, Optional[str], Dict[str, Any], Optional[str]]:
        """Parse arbitrary command into structured components."""
        if isinstance(command, dict):
            return super().parse_command(command)

        text = str(command or "").strip()
        m_open = _RE_OPEN.match(text)
        if m_open:
            return "open_app", m_open.group(1).strip(), {}, None

        m_close = _RE_CLOSE.match(text)
        if m_close:
            return "close_app", m_close.group(1).strip(), {}, None

        m_restart = _RE_RESTART.match(text)
        if m_restart:
            return "restart_app", m_restart.group(1).strip(), {}, None

        m_is = _RE_IS_RUNNING.match(text)
        if m_is:
            return "is_app_running", m_is.group(1).strip(), {}, None

        m_list = _RE_LIST_RUNNING.match(text)
        if m_list:
            return "list_running_apps", None, {}, None

        return super().parse_command(command)

    def _execute_operation(
        self, operation: str, target: Optional[str], parameters: Dict[str, Any]
    ) -> Any:
        """Internal operation dispatcher."""
        op = operation.strip().lower()

        if op in ("open_app", "launch_app"):
            return self.open_app(target, parameters)

        if op in ("close_app", "terminate_process", "kill_process"):
            return self.close_app(target, parameters)

        if op == "restart_app":
            return self.restart_app(target, parameters)

        if op == "is_app_running":
            return self.is_app_running(target, parameters)

        if op == "list_running_apps":
            return self.list_running_apps(target, parameters)

        raise NotImplementedError(f"Operation '{op}' is not supported by {self.name}.")

    def open_app(self, app_name: Optional[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Safely launch an allowlisted application."""
        if not app_name:
            raise SkillExecutionError("No application specified to open.")

        is_valid, resolved_target, reason = self.resolver.resolve(app_name)
        if not is_valid or not resolved_target:
            raise SkillExecutionError(f"Cannot launch application: {reason}")

        args: List[str] = [resolved_target]
        extra_args = parameters.get("arguments") or parameters.get("args")
        if extra_args:
            if isinstance(extra_args, list):
                for a in extra_args:
                    if is_shell_command(str(a)):
                        raise SecurityPolicyViolationError(f"Argument '{a}' contains prohibited shell tokens.")
                    args.append(str(a))
            elif isinstance(extra_args, str):
                if is_shell_command(extra_args):
                    raise SecurityPolicyViolationError("Arguments contain prohibited shell syntax.")
                args.extend(extra_args.split())

        try:
            # Launch via structured subprocess (NO shell=True)
            proc = subprocess.Popen(
                args,
                shell=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.logger.info("Launched application '%s' (PID %d)", app_name, proc.pid)
            return {
                "app_name": app_name,
                "executable": resolved_target,
                "pid": proc.pid,
                "launched": True,
            }
        except Exception as exc:
            msg = f"Failed to launch application '{app_name}': {exc}"
            self.logger.error(msg, exc_info=True)
            raise SkillExecutionError(msg) from exc

    def close_app(
        self, target: Optional[Union[str, int]], parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Safely close or terminate an application process."""
        if target is None or str(target).strip() == "":
            raise SkillExecutionError("No application or PID specified to close.")

        force = bool(parameters.get("force", False))
        timeout = float(parameters.get("timeout", 3.0))

        # Capture snapshots before confirmation verification to prevent TOCTOU PID reuse
        pre_snapshots = self.process_manager.get_process_snapshots(target)
        if not pre_snapshots:
            return {
                "target": target,
                "closed": False,
                "terminated_count": 0,
                "pids": [],
                "reason": "Process not found",
            }

        result = self.process_manager.terminate_processes(
            target=target,
            force=force,
            timeout=timeout,
            expected_snapshots=pre_snapshots,
        )
        return result

    def restart_app(self, app_name: Optional[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Safely restart an application by composing validated close and open steps."""
        if not app_name:
            raise SkillExecutionError("No application specified to restart.")

        # 1. Validate open target first before terminating anything
        is_valid, resolved, reason = self.resolver.resolve(app_name)
        if not is_valid:
            raise SkillExecutionError(f"Cannot restart application: {reason}")

        # 2. Close current running instances
        close_result = self.close_app(app_name, parameters)

        # 3. Allow brief pause for OS process table update
        time.sleep(0.1)

        # 4. Launch new instance
        open_result = self.open_app(app_name, parameters)

        return {
            "app_name": app_name,
            "restarted": True,
            "closed": close_result,
            "launched": open_result,
        }

    def is_app_running(
        self, target: Optional[Union[str, int]], parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine whether an application is running."""
        if not target:
            raise SkillExecutionError("No application or PID specified to check.")

        running, pids = self.process_manager.is_running(target)
        return {
            "target": target,
            "running": running,
            "pids": pids,
            "count": len(pids),
        }

    def list_running_apps(
        self, target: Optional[str], parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enumerate user-facing running applications."""
        user_only = bool(parameters.get("user_only", True))
        limit = int(parameters.get("limit", 50))

        apps = self.process_manager.list_running(user_only=user_only, limit=limit)
        return {
            "apps": apps,
            "total_listed": len(apps),
            "user_only": user_only,
        }


__all__ = [
    "AppResolver",
    "AppSkills",
    "DEFAULT_SAFE_APP_ALIASES",
    "ProcessManager",
]
