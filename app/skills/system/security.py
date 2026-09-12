"""System Security Policy & Confirmation Management for J.A.R.V.I.S. Phase 22.

Provides deterministic, explainable safety classification (SAFE, CONFIRMATION_REQUIRED,
RESTRICTED), canonical path containment validation, critical Windows process protection,
shell command injection prevention, and parameter-bound confirmation tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Dict, Final, Iterable, List, Optional, Set, Tuple, Union
import uuid

from app.ai.planner.events import (
    PlannerEventBus,
    SystemSkillConfirmationRequired,
    SystemSkillPolicyRejected,
)
from app.ai.planner.swarm.hitl import ApprovalDecision, InterventionGateway, RiskLevel
from app.automation.guardrails import (
    BLOCKED_EXEC_PATTERNS as DEFAULT_BLOCKED_EXEC_PATTERNS,
    CRITICAL_SYSTEM_PROCESSES as DEFAULT_CRITICAL_SYSTEM_PROCESSES,
)
from app.core.container import JarvisException
from app.core.logger import get_logger

logger = get_logger("SYSTEM.SECURITY")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SystemSecurityError(JarvisException):
    """Base exception for all system skill security violations."""


class SecurityPolicyViolationError(SystemSecurityError):
    """Raised when an action violates safety policies or is RESTRICTED."""


class ConfirmationRequiredError(SystemSecurityError):
    """Raised when an action requires user confirmation before execution."""

    def __init__(
        self,
        message: str,
        confirmation_id: str = "",
        operation: str = "",
        target: Optional[str] = None,
        risk_level: str = "HIGH",
    ) -> None:
        super().__init__(message)
        self.confirmation_id = confirmation_id
        self.operation = operation
        self.target = target
        self.risk_level = risk_level


class ConfirmationRejectedError(SystemSecurityError):
    """Raised when an action's confirmation is rejected or mismatched."""


class ConfirmationTimeoutError(SystemSecurityError):
    """Raised when an action's confirmation request has expired."""


# ---------------------------------------------------------------------------
# Safety Tiers & Constants
# ---------------------------------------------------------------------------

class SystemSafetyTier(str, Enum):
    """Safety classification tiers for system operations."""

    SAFE = "SAFE"
    CONFIRMATION_REQUIRED = "CONFIRMATION_REQUIRED"
    RESTRICTED = "RESTRICTED"


# Blocked direct shell / interpreter executables
BLOCKED_SHELL_COMMANDS: Final[Set[str]] = {
    "cmd",
    "cmd.exe",
    "powershell",
    "powershell.exe",
    "pwsh",
    "pwsh.exe",
    "bash",
    "bash.exe",
    "sh",
    "sh.exe",
    "zsh",
    "cscript",
    "cscript.exe",
    "wscript",
    "wscript.exe",
    "mshta",
    "mshta.exe",
}

# Regex patterns indicating shell chaining, piping, or command injection
BLOCKED_SHELL_PATTERNS: Final[List[str]] = [
    r"[;&|`]",
    r"\$\([^\)]*\)",
    r"-enc(?:odedcommand)?\b",
    r"-exec(?:utionpolicy)?\s+bypass\b",
    r"\biex\b",
    r"\binvoke-expression\b",
    r"\bvssadmin\b",
    r"\bbcdedit\b",
    r"\bformat\s+[a-zA-Z]:",
    r"\bdel\s+/[sS]\s+/[qQ]\b",
    r"\brmdir\s+/[sS]\s+/[qQ]\b",
    r"\brm\s+-rf\b",
    r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",  # fork bomb
]

# Protected Windows Directories (canonical forms)
PROTECTED_WINDOWS_DIRS: Final[Set[str]] = {
    r"c:\windows",
    r"c:\windows\system32",
    r"c:\windows\syswow64",
    r"c:\windows\winsxs",
    r"c:\programdata\microsoft",
    r"c:\program files\windowsapps",
    r"c:\system volume information",
    r"c:\$recycle.bin",
}

# Reserved Windows Device Names
WINDOWS_RESERVED_DEVICE_NAMES: Final[Set[str]] = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}


# ---------------------------------------------------------------------------
# Path & Process Validation Utilities
# ---------------------------------------------------------------------------

def validate_path(
    path: Union[str, Path],
    *,
    allow_nonexistent: bool = True,
    allowed_roots: Optional[Iterable[Union[str, Path]]] = None,
    allow_system_dirs: bool = False,
) -> Path:
    """Validate and canonicalize a filesystem path against security constraints.

    Enforces:
    - No null bytes
    - No Windows reserved device names
    - Resolution to canonical absolute path
    - Protection against protected Windows system directories
    - Confinement within allowed roots if specified

    Args:
        path: Path string or Path object.
        allow_nonexistent: Whether target path can be nonexistent.
        allowed_roots: Optional collection of authorized root directories.
        allow_system_dirs: Whether access to system directories is explicitly authorized.

    Returns:
        Canonical, resolved Path instance.

    Raises:
        SecurityPolicyViolationError: If path violates security boundaries.
    """
    if path is None:
        raise SecurityPolicyViolationError("Path cannot be None.")

    path_str = str(path).strip()
    if not path_str:
        raise SecurityPolicyViolationError("Path cannot be empty.")

    if "\0" in path_str:
        raise SecurityPolicyViolationError("Path contains prohibited null byte.")

    try:
        raw_path = Path(path_str).expanduser()
        resolved = raw_path.resolve()
    except Exception as exc:
        raise SecurityPolicyViolationError(f"Invalid path format '{path_str}': {exc}") from exc

    # Check Windows reserved device names (e.g., CON, PRN, AUX, NUL)
    for part in resolved.parts:
        part_upper = part.upper()
        stem_upper = Path(part).stem.upper()
        if part_upper in WINDOWS_RESERVED_DEVICE_NAMES or stem_upper in WINDOWS_RESERVED_DEVICE_NAMES:
            raise SecurityPolicyViolationError(
                f"Path '{path_str}' references reserved Windows device name '{part}'."
            )

    # Check protected Windows directories
    if not allow_system_dirs:
        resolved_str = str(resolved).lower()
        for protected in PROTECTED_WINDOWS_DIRS:
            if resolved_str == protected or resolved_str.startswith(f"{protected}\\") or resolved_str.startswith(f"{protected}/"):
                raise SecurityPolicyViolationError(
                    f"Path '{path_str}' targets protected system directory '{protected}'."
                )

    # Check root confinement if allowed_roots is specified
    if allowed_roots:
        resolved_roots = [Path(r).expanduser().resolve() for r in allowed_roots]
        is_confined = any(
            resolved == r or r in resolved.parents
            for r in resolved_roots
        )
        if not is_confined:
            raise SecurityPolicyViolationError(
                f"Path traversal violation: '{path_str}' escapes allowed roots {allowed_roots}."
            )

    return resolved


def is_critical_process(process_identifier: Union[str, int]) -> bool:
    """Check if process identifier targets a critical OS process or self.

    Args:
        process_identifier: Process executable name or PID.

    Returns:
        True if process is critical and must not be terminated.
    """
    if isinstance(process_identifier, int):
        try:
            current_pid = os.getpid()
        except Exception:
            current_pid = None
        if process_identifier in (0, 4) or process_identifier == current_pid:
            return True
        return False

    proc_clean = str(process_identifier).strip().lower()
    if not proc_clean:
        return True

    # Check if identifier is numeric PID (as int or str)
    if proc_clean.isdigit():
        try:
            pid_val = int(proc_clean)
            current_pid = os.getpid()
            if pid_val in (0, 4) or pid_val == current_pid:
                return True
        except Exception:
            pass

    # Normalize against critical process names
    critical_set = {p.lower() for p in DEFAULT_CRITICAL_SYSTEM_PROCESSES}
    if proc_clean in critical_set:
        return True

    if proc_clean.endswith(".exe"):
        stem = proc_clean[:-4]
        if stem in critical_set:
            return True
    else:
        if f"{proc_clean}.exe" in critical_set:
            return True

    return False


def is_shell_command(command_str: str) -> bool:
    """Check if command invokes an arbitrary shell or script interpreter.

    Args:
        command_str: Executable or full command line.

    Returns:
        True if command is a prohibited shell or interpreter.
    """
    cmd = command_str.strip().lower()
    first_token = cmd.split()[0] if cmd.split() else ""
    first_token_name = Path(first_token).name.lower()

    if first_token_name in BLOCKED_SHELL_COMMANDS:
        return True

    for pattern in BLOCKED_SHELL_PATTERNS:
        if re.search(pattern, cmd, re.IGNORECASE):
            return True

    return False


# ---------------------------------------------------------------------------
# System Security Policy
# ---------------------------------------------------------------------------

class SystemSecurityPolicy:
    """Thread-safe security policy enforcer for Phase 22 system skills.

    Classifies actions into SAFE, CONFIRMATION_REQUIRED, and RESTRICTED tiers.
    Validates filesystem paths, critical OS processes, and command inputs.
    """

    def __init__(
        self,
        *,
        blocked_processes: Optional[Iterable[str]] = None,
        protected_dirs: Optional[Iterable[Union[str, Path]]] = None,
        allowed_roots: Optional[Iterable[Union[str, Path]]] = None,
        event_bus: Optional[PlannerEventBus] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize the SystemSecurityPolicy.

        Args:
            blocked_processes: Additional process names to block.
            protected_dirs: Additional protected directory paths.
            allowed_roots: Root paths confining filesystem actions.
            event_bus: PlannerEventBus for security alert emission.
            logger: Custom logger instance.
        """
        self._lock = threading.RLock()
        self._logger = logger or get_logger("SYSTEM.SECURITY.POLICY")
        self._event_bus = event_bus

        self._blocked_processes = {p.lower() for p in DEFAULT_CRITICAL_SYSTEM_PROCESSES}
        if blocked_processes:
            self._blocked_processes.update(p.lower() for p in blocked_processes)

        self._protected_dirs = set(PROTECTED_WINDOWS_DIRS)
        if protected_dirs:
            for p in protected_dirs:
                try:
                    self._protected_dirs.add(str(Path(p).expanduser().resolve()).lower())
                except Exception:
                    pass

        self._allowed_roots = (
            [Path(r).expanduser().resolve() for r in allowed_roots]
            if allowed_roots is not None
            else None
        )

        # Standard operation classifications
        self._safe_operations: Set[str] = {
            "get_cpu_info",
            "get_memory_info",
            "get_disk_info",
            "get_battery_info",
            "get_gpu_info",
            "get_network_info",
            "get_system_summary",
            "list_directory",
            "read_file",
            "list_running_apps",
            "is_app_running",
            "list_windows",
            "get_active_window",
            "focus_window",
            "minimize_window",
            "maximize_window",
            "restore_window",
            "get_volume",
            "set_volume",
            "volume_up",
            "volume_down",
            "mute_volume",
            "unmute_volume",
            "get_brightness",
            "set_brightness",
            "lock_workstation",
            "open_url",
            "search_web",
            "open_browser",
            "open_app",
            "launch_app",
            "create_folder",
            "create_file",
            "rename_path",
            "copy_path",
            "open_in_explorer",
        }

        self._confirmation_operations: Set[str] = {
            "delete_path",
            "delete_file",
            "delete_folder",
            "remove_file",
            "remove_directory",
            "move_path",
            "overwrite_file",
            "close_app",
            "terminate_process",
            "kill_process",
            "restart_app",
        }

    def classify_operation(
        self,
        operation: str,
        target: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> SystemSafetyTier:
        """Evaluate and return the safety tier for an operation.

        Args:
            operation: Name of the system operation.
            target: Primary target resource (app, process, path, etc.).
            parameters: Additional operation arguments.

        Returns:
            SystemSafetyTier enum member (SAFE, CONFIRMATION_REQUIRED, RESTRICTED).
        """
        tier, _ = self._evaluate(operation, target, parameters)
        return tier

    def validate_operation(
        self,
        operation: str,
        target: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[SystemSafetyTier, Optional[str]]:
        """Validate an operation against security constraints.

        Args:
            operation: Name of the system operation.
            target: Primary target resource.
            parameters: Additional operation arguments.

        Returns:
            Tuple of (SystemSafetyTier, reason_or_none).

        Raises:
            SecurityPolicyViolationError: If action is classified as RESTRICTED.
        """
        tier, reason = self._evaluate(operation, target, parameters)

        if tier == SystemSafetyTier.RESTRICTED:
            msg = f"Security Policy Violation: operation '{operation}' is RESTRICTED. Reason: {reason}"
            self._logger.warning(msg)
            if self._event_bus is not None:
                self._event_bus.publish(
                    SystemSkillPolicyRejected(
                        operation=operation,
                        target=target,
                        reason=reason or "Restricted operation",
                        tier=tier.value,
                    )
                )
            raise SecurityPolicyViolationError(msg)

        return tier, reason

    def _evaluate(
        self,
        operation: str,
        target: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[SystemSafetyTier, Optional[str]]:
        """Internal evaluation helper."""
        with self._lock:
            op = (operation or "").strip().lower()
            target_str = str(target or "").strip()
            params = dict(parameters or {})

            # 1. Reject arbitrary shell execution
            if op in ("shell", "cmd", "powershell", "exec_shell", "run_command"):
                return SystemSafetyTier.RESTRICTED, "Arbitrary shell execution is prohibited."

            if target_str and is_shell_command(target_str):
                return SystemSafetyTier.RESTRICTED, f"Command '{target_str}' contains prohibited shell tokens or interpreters."

            # Check if any parameter contains prohibited shell patterns
            for key, val in params.items():
                if isinstance(val, str) and is_shell_command(val):
                    return SystemSafetyTier.RESTRICTED, f"Parameter '{key}' contains prohibited shell syntax."

            # Check blocked patterns
            for pat in DEFAULT_BLOCKED_EXEC_PATTERNS:
                if pat in target_str.lower():
                    return SystemSafetyTier.RESTRICTED, f"Target matches dangerous pattern '{pat}'."

            # 2. Check process termination against critical OS processes
            if op in ("close_app", "terminate_process", "kill_process", "restart_app"):
                proc_target = target_str or str(params.get("process") or params.get("pid") or "")
                if is_critical_process(proc_target):
                    return (
                        SystemSafetyTier.RESTRICTED,
                        f"Process '{proc_target}' is a protected system process and cannot be terminated.",
                    )

            # 3. Path validation for filesystem operations
            if op in (
                "create_folder",
                "create_file",
                "read_file",
                "list_directory",
                "rename_path",
                "move_path",
                "copy_path",
                "delete_path",
                "delete_file",
                "delete_folder",
                "remove_file",
                "remove_directory",
                "open_in_explorer",
                "get_disk_info",
            ):
                path_target = target_str or str(params.get("path") or "")
                if path_target:
                    try:
                        validate_path(
                            path_target,
                            allowed_roots=self._allowed_roots,
                            allow_system_dirs=(op == "get_disk_info"),
                        )
                    except SecurityPolicyViolationError as exc:
                        return SystemSafetyTier.RESTRICTED, str(exc)

                # Check for filesystem root deletion attempts
                if op in ("delete_path", "delete_file", "delete_folder", "remove_file", "remove_directory"):
                    if path_target:
                        try:
                            resolved_p = Path(path_target).expanduser().resolve()
                            if resolved_p.anchor == str(resolved_p) or len(resolved_p.parts) <= 1:
                                return (
                                    SystemSafetyTier.RESTRICTED,
                                    f"Deletion of filesystem root '{path_target}' is strictly prohibited.",
                                )
                        except Exception:
                            pass

                # Validate secondary path for copy/move/rename
                sec_path = str(params.get("target") or params.get("destination") or "")
                if sec_path:
                    try:
                        validate_path(
                            sec_path,
                            allowed_roots=self._allowed_roots,
                            allow_system_dirs=False,
                        )
                    except SecurityPolicyViolationError as exc:
                        return SystemSafetyTier.RESTRICTED, str(exc)

            # 4. Check confirmation operations
            if op in self._confirmation_operations:
                return SystemSafetyTier.CONFIRMATION_REQUIRED, f"Operation '{op}' requires user confirmation."

            # Destructive overwrites require confirmation
            if bool(params.get("overwrite", False)):
                return (
                    SystemSafetyTier.CONFIRMATION_REQUIRED,
                    f"Overwriting existing target in operation '{op}' requires user confirmation.",
                )

            # Force kill on close_app is confirmation required
            if op == "close_app" and bool(params.get("force", False)):
                return SystemSafetyTier.CONFIRMATION_REQUIRED, "Forced application termination requires confirmation."

            # Extreme volume setting
            if op == "set_volume" and params.get("level") == 100:
                return SystemSafetyTier.CONFIRMATION_REQUIRED, "Setting volume to maximum requires confirmation."

            # 5. Check safe operations
            if op in self._safe_operations or op.startswith("get_"):
                return SystemSafetyTier.SAFE, None

            # 6. Default fallback for unknown actions: require confirmation
            return SystemSafetyTier.CONFIRMATION_REQUIRED, f"Unrecognized operation '{op}' requires confirmation by default."


# ---------------------------------------------------------------------------
# Confirmation Manager
# ---------------------------------------------------------------------------

@dataclass
class ConfirmationRequest:
    """Internal record for a pending or resolved confirmation request."""

    confirmation_id: str
    operation: str
    target: Optional[str]
    parameters: Dict[str, Any]
    param_hash: str
    risk_level: RiskLevel
    created_at: float
    expires_at: float
    description: str = ""
    approved: Optional[bool] = None
    decided_by: Optional[str] = None
    reason: str = ""
    consumed: bool = False


class SystemConfirmationManager:
    """Thread-safe manager for system skill confirmation workflows.

    Coordinates approval tokens, TTL expiration, parameter signature verification,
    and bridges directly into InterventionGateway.
    """

    def __init__(
        self,
        *,
        hitl_gateway: Optional[InterventionGateway] = None,
        event_bus: Optional[PlannerEventBus] = None,
        default_timeout: float = 30.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize the SystemConfirmationManager.

        Args:
            hitl_gateway: Optional InterventionGateway for human proxy integration.
            event_bus: Optional PlannerEventBus for event broadcasting.
            default_timeout: Default confirmation validity duration in seconds.
            logger: Custom logger instance.
        """
        self._lock = threading.RLock()
        self._logger = logger or get_logger("SYSTEM.CONFIRMATION")
        self._hitl_gateway = hitl_gateway
        self._event_bus = event_bus
        self._default_timeout = max(1.0, float(default_timeout))
        self._requests: Dict[str, ConfirmationRequest] = {}

    @staticmethod
    def compute_signature(
        operation: str,
        target: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Compute deterministic SHA-256 signature for operation parameters."""
        norm_op = (operation or "").strip().lower()
        norm_target = str(target or "").strip().lower()
        params = parameters or {}
        try:
            param_json = json.dumps(params, sort_keys=True, default=str)
        except Exception:
            param_json = str(sorted(params.items()))

        content = f"{norm_op}|{norm_target}|{param_json}".encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def request_confirmation(
        self,
        operation: str,
        target: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        *,
        timeout: Optional[float] = None,
        risk_level: RiskLevel = RiskLevel.HIGH,
        description: str = "",
    ) -> str:
        """Create a new confirmation request and return its token.

        Args:
            operation: Name of the operation requiring confirmation.
            target: Primary target resource.
            parameters: Action arguments.
            timeout: Optional custom timeout in seconds (defaults to default_timeout).
            risk_level: Risk severity classification.
            description: Human-readable explanation of the requested action.

        Returns:
            Unique confirmation token (confirmation_id).
        """
        with self._lock:
            self.cleanup_expired()

            conf_id = f"sys_conf_{uuid.uuid4().hex[:10]}"
            ttl = float(timeout) if timeout is not None else self._default_timeout
            now = time.time()
            sig = self.compute_signature(operation, target, parameters)

            request = ConfirmationRequest(
                confirmation_id=conf_id,
                operation=operation.strip().lower(),
                target=target,
                parameters=dict(parameters or {}),
                param_hash=sig,
                risk_level=risk_level,
                created_at=now,
                expires_at=now + ttl,
                description=description,
            )
            self._requests[conf_id] = request

            if self._event_bus is not None:
                self._event_bus.publish(
                    SystemSkillConfirmationRequired(
                        operation=operation,
                        target=target,
                        confirmation_id=conf_id,
                        risk_level=risk_level.value,
                    )
                )

            # Check if HITL gateway can resolve it immediately
            if self._hitl_gateway is not None:
                try:
                    decision: ApprovalDecision = self._hitl_gateway.request_approval(
                        action=operation,
                        target=target,
                        parameters=parameters,
                        risk_level=risk_level,
                        timeout=ttl,
                    )
                    request.approved = decision.approved
                    request.decided_by = decision.decided_by
                    request.reason = decision.reason
                except Exception as exc:
                    self._logger.debug("HITL gateway dispatch returned: %s", exc)

            return conf_id

    def resolve_confirmation(
        self,
        confirmation_id: str,
        approved: bool,
        *,
        decided_by: str = "operator",
        reason: str = "",
    ) -> bool:
        """Explicitly approve or reject a confirmation request.

        Args:
            confirmation_id: The confirmation token to resolve.
            approved: True to authorize execution, False to reject.
            decided_by: Identity of the decision maker.
            reason: Optional justification or comment.

        Returns:
            True if resolved successfully, False if not found or expired.
        """
        with self._lock:
            request = self._requests.get(confirmation_id)
            if request is None:
                return False

            if time.time() > request.expires_at:
                del self._requests[confirmation_id]
                return False

            if request.consumed:
                return False

            request.approved = bool(approved)
            request.decided_by = str(decided_by)
            request.reason = str(reason)
            return True

    def verify_and_consume(
        self,
        confirmation_id: str,
        operation: str,
        target: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Verify that a confirmation token is valid, approved, matches the exact action, and consume it.

        Enforces:
        - Token existence
        - Expiration window (TTL)
        - Single-use consumption
        - Strict signature match (cannot use approval for action A to execute action B)

        Args:
            confirmation_id: Token provided by caller.
            operation: Name of operation being executed.
            target: Primary target resource.
            parameters: Action arguments.

        Returns:
            True if authorization is verified and consumed.

        Raises:
            ConfirmationTimeoutError: If token has expired.
            ConfirmationRejectedError: If token is rejected, consumed, or mismatched.
            ConfirmationRequiredError: If token is still pending.
        """
        with self._lock:
            request = self._requests.get(confirmation_id)
            if request is None:
                raise ConfirmationRejectedError(
                    f"Confirmation token '{confirmation_id}' is invalid or unknown."
                )

            # Check expiration
            if time.time() > request.expires_at:
                del self._requests[confirmation_id]
                raise ConfirmationTimeoutError(
                    f"Confirmation token '{confirmation_id}' has expired."
                )

            # Check consumption
            if request.consumed:
                raise ConfirmationRejectedError(
                    f"Confirmation token '{confirmation_id}' has already been consumed."
                )

            # Verify action signature
            expected_sig = self.compute_signature(operation, target, parameters)
            if request.param_hash != expected_sig:
                raise ConfirmationRejectedError(
                    f"Confirmation token '{confirmation_id}' was issued for a different operation/arguments."
                )

            # Check decision status
            if request.approved is None:
                raise ConfirmationRequiredError(
                    f"Confirmation token '{confirmation_id}' is still pending approval.",
                    confirmation_id=confirmation_id,
                    operation=operation,
                    target=target,
                )

            if not request.approved:
                del self._requests[confirmation_id]
                raise ConfirmationRejectedError(
                    f"Confirmation was rejected: {request.reason or 'Authorization denied by operator.'}"
                )

            # Mark consumed and remove to prevent reuse
            request.consumed = True
            del self._requests[confirmation_id]
            return True

    def is_approved(self, confirmation_id: str) -> bool:
        """Check if request is currently approved without consuming it."""
        with self._lock:
            request = self._requests.get(confirmation_id)
            if request is None or time.time() > request.expires_at or request.consumed:
                return False
            return bool(request.approved)

    def cleanup_expired(self) -> int:
        """Purge expired confirmation requests.

        Returns:
            Count of expired tokens purged.
        """
        with self._lock:
            now = time.time()
            expired = [cid for cid, req in self._requests.items() if now > req.expires_at]
            for cid in expired:
                del self._requests[cid]
            return len(expired)

    def get_pending_requests(self) -> List[Dict[str, Any]]:
        """Return list of all currently active pending requests."""
        with self._lock:
            self.cleanup_expired()
            return [
                {
                    "confirmation_id": req.confirmation_id,
                    "operation": req.operation,
                    "target": req.target,
                    "parameters": dict(req.parameters),
                    "risk_level": req.risk_level.value,
                    "created_at": req.created_at,
                    "expires_at": req.expires_at,
                    "description": req.description,
                    "approved": req.approved,
                }
                for req in self._requests.values()
                if not req.consumed and req.approved is None
            ]


# Module exports
__all__ = [
    "BLOCKED_SHELL_COMMANDS",
    "BLOCKED_SHELL_PATTERNS",
    "ConfirmationRejectedError",
    "ConfirmationRequiredError",
    "ConfirmationTimeoutError",
    "PROTECTED_WINDOWS_DIRS",
    "SecurityPolicyViolationError",
    "SystemConfirmationManager",
    "SystemSafetyTier",
    "SystemSecurityError",
    "SystemSecurityPolicy",
    "WINDOWS_RESERVED_DEVICE_NAMES",
    "is_critical_process",
    "is_shell_command",
    "validate_path",
]
