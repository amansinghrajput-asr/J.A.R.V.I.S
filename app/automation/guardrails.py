"""Security Guardrails and Safety Validation for OS Automation.

Prevents critical OS processes from being terminated, validates execution targets,
and enforces safety policies for desktop automation actions.
"""

from __future__ import annotations

import logging
import os
from typing import Final, Iterable, Optional, Set, Union

from app.automation.models import SecurityBlockedError
from app.core.logger import get_logger

# Critical Windows and Unix processes that must never be terminated
CRITICAL_SYSTEM_PROCESSES: Final[Set[str]] = {
    # Windows core
    "system",
    "system idle process",
    "registry",
    "smss.exe",
    "csrss.exe",
    "wininit.exe",
    "services.exe",
    "lsass.exe",
    "winlogon.exe",
    "fontdrvhost.exe",
    "dwm.exe",
    "svchost.exe",
    "explorer.exe",
    "taskhostw.exe",
    "sihost.exe",
    # Unix core
    "init",
    "systemd",
    "kthreadd",
    "launchd",
    "sshd",
    "kernel_task",
}

# Dangerous executable commands
BLOCKED_EXEC_PATTERNS: Final[Set[str]] = {
    "format",
    "del /f /s /q c:",
    "rmdir /s /q c:",
    "rm -rf /",
    ":(){ :|:& };:",
    "mkfs",
    "dd if=",
}


class SecurityGuard:
    """Thread-safe security policy enforcer for OS automation and system actions."""

    def __init__(
        self,
        *,
        blocked_processes: Optional[Iterable[str]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize the SecurityGuard.

        Args:
            blocked_processes: Optional custom set of process names to protect.
            logger: Custom logger instance.
        """
        self._logger = logger or get_logger("AUTOMATION.SECURITY")
        self._protected_processes: Set[str] = {p.lower() for p in CRITICAL_SYSTEM_PROCESSES}
        if blocked_processes:
            self._protected_processes.update(p.lower() for p in blocked_processes)

        # Always protect current process and its parent
        try:
            self._current_pid = os.getpid()
        except Exception:
            self._current_pid = None

    def is_protected_process(self, process_identifier: Union[str, int]) -> bool:
        """Check if a process is protected by safety guardrails.

        Args:
            process_identifier: Process executable name or PID.

        Returns:
            True if the target is protected, False otherwise.
        """
        if isinstance(process_identifier, int):
            if process_identifier in (0, 4) or process_identifier == self._current_pid:
                return True
            return False

        proc_clean = str(process_identifier).strip().lower()
        if not proc_clean:
            return True

        if proc_clean in self._protected_processes:
            return True

        # Check with or without .exe suffix
        if proc_clean.endswith(".exe"):
            stem = proc_clean[:-4]
            if stem in self._protected_processes:
                return True
        else:
            if f"{proc_clean}.exe" in self._protected_processes:
                return True

        return False

    def validate_kill_target(self, process_identifier: Union[str, int]) -> None:
        """Ensure a process termination target is safe to terminate.

        Args:
            process_identifier: Process executable name or PID.

        Raises:
            SecurityBlockedError: If process is critical or protected.
        """
        if self.is_protected_process(process_identifier):
            msg = (
                f"Termination blocked: '{process_identifier}' is a critical system process "
                "protected by safety guardrails."
            )
            self._logger.warning(msg)
            raise SecurityBlockedError(msg)

    def validate_app_launch(self, app_command: str) -> None:
        """Ensure an application launch command is safe.

        Args:
            app_command: Executable name or command line string.

        Raises:
            SecurityBlockedError: If command contains dangerous keywords.
        """
        cmd_lower = app_command.strip().lower()
        for pattern in BLOCKED_EXEC_PATTERNS:
            if pattern in cmd_lower:
                msg = f"Execution blocked: command '{app_command}' contains unsafe pattern '{pattern}'."
                self._logger.warning(msg)
                raise SecurityBlockedError(msg)


# Module default instance
security_guard = SecurityGuard()

__all__ = [
    "BLOCKED_EXEC_PATTERNS",
    "CRITICAL_SYSTEM_PROCESSES",
    "SecurityGuard",
    "security_guard",
]
