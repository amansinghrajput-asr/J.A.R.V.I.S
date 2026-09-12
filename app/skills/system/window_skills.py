"""Window Skills for J.A.R.V.I.S. Phase 22.6.

Provides safe, native Windows desktop window automation including:
- list_windows / list_open_windows: Bounded enumeration of top-level user windows
- get_active_window: Query current foreground window metadata
- focus_window: Bring target window safely to the foreground
- minimize_window: Minimize target window safely
- maximize_window: Maximize target window safely
- restore_window: Restore target window safely
- close_window: Request window closure via WM_CLOSE (confirmation-protected)

Safety Invariants Enforced:
1. No process termination, process kill utilities, or OS termination APIs are ever used.
2. Under no circumstances will arbitrary shell execution, cmd interpreters, or CLI terminals be invoked.
3. Every window handle (HWND) is validated via user32.IsWindow prior to operation.
4. Window enumeration is strictly bounded (max 200 windows) to prevent infinite loops.
5. Critical system processes (PID 0, PID 4, J.A.R.V.I.S self-PID, dwm, explorer, csrss, etc.)
   are strictly protected against window closure.
6. Power transitions (power-off, restart, sleep, hibernate) remain permanently absent.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import logging
import os
import platform
import re
import threading
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore
    _HAS_PSUTIL = False

from app.automation.models import WindowInfo
from app.core.config import Settings
from app.core.container import ServiceContainer
from app.core.event_bus import EventBus
from app.core.logger import get_logger
from app.skills.base import SkillExecutionError
from app.skills.system.base_system_skill import BaseSystemSkill, SystemSkillResult
from app.skills.system.security import (
    ConfirmationRequiredError,
    SecurityPolicyViolationError,
    SystemConfirmationManager,
    SystemSecurityPolicy,
    is_critical_process,
)

logger = get_logger("SYSTEM.WINDOW_SKILLS")

# Maximum window enumeration cap to avoid resource exhaustion and infinite loops
MAX_WINDOW_ENUM_LIMIT: Final[int] = 200

# Win32 Constants
WM_CLOSE: Final[int] = 0x0010
SW_HIDE: Final[int] = 0
SW_NORMAL: Final[int] = 1
SW_SHOWMINIMIZED: Final[int] = 2
SW_MAXIMIZE: Final[int] = 3
SW_SHOWNOACTIVATE: Final[int] = 4
SW_SHOW: Final[int] = 5
SW_MINIMIZE: Final[int] = 6
SW_RESTORE: Final[int] = 9

# DWM Window Attributes
DWMWA_CLOAKED: Final[int] = 14

# Regex matchers for natural language window commands
_RE_LIST_WINDOWS = re.compile(
    r"^(?:list|show|get|display)\s+(?:all\s+)?(?:open\s+)?windows?$",
    re.IGNORECASE,
)
_RE_GET_ACTIVE = re.compile(
    r"^(?:get|show|what\s+is)\s+(?:the\s+)?(?:active|current|focused|foreground)\s+window\??$",
    re.IGNORECASE,
)
_RE_FOCUS_WINDOW = re.compile(
    r"^(?:focus|switch\s+to|bring\s+up|activate)\s+(?:window\s+)?(.+)$",
    re.IGNORECASE,
)
_RE_MINIMIZE_WINDOW = re.compile(
    r"^minimize\s+(?:window\s+)?(.+)$",
    re.IGNORECASE,
)
_RE_MAXIMIZE_WINDOW = re.compile(
    r"^maximize\s+(?:window\s+)?(.+)$",
    re.IGNORECASE,
)
_RE_RESTORE_WINDOW = re.compile(
    r"^restore\s+(?:window\s+)?(.+)$",
    re.IGNORECASE,
)
_RE_CLOSE_WINDOW = re.compile(
    r"^close\s+window\s+(.+)$",
    re.IGNORECASE,
)


class WindowSkills(BaseSystemSkill):
    """Production Windows desktop window management skills for J.A.R.V.I.S.

    Inherits from BaseSystemSkill, implementing verified validation, structured
    parameters, confirmation workflows, telemetry, and event bus emissions.
    """

    name: str = "window"
    description: str = "Desktop window management: enumeration, focus, minimize, maximize, restore, and safe close."
    priority: int = 55
    tags: list[str] = ["window", "ui", "automation", "system", "windows"]
    permissions: set[str] = {"system:read", "system:execute"}

    def __init__(
        self,
        *,
        security_policy: Optional[SystemSecurityPolicy] = None,
        confirmation_manager: Optional[SystemConfirmationManager] = None,
        config: Optional[Settings] = None,
        logger: Optional[logging.Logger] = None,
        container: Optional[ServiceContainer] = None,
        event_bus: Optional[Union[EventBus, Any]] = None,
    ) -> None:
        """Initialize WindowSkills instance."""
        super().__init__(
            name=self.name,
            description=self.description,
            priority=self.priority,
            tags=self.tags,
            permissions=self.permissions,
            security_policy=security_policy,
            confirmation_manager=confirmation_manager,
            config=config,
            logger=logger or get_logger("SKILL.WINDOW"),
            container=container,
            event_bus=event_bus,
        )
        self._lock = threading.RLock()

    # -----------------------------------------------------------------------
    # Low-level Native Win32 API Call Hooks (Overridable / Mockable in Tests)
    # -----------------------------------------------------------------------

    @staticmethod
    def _is_windows() -> bool:
        """Check if operating on native Windows platform."""
        return platform.system() == "Windows"

    def _api_is_window(self, hwnd: int) -> bool:
        """Native check if hwnd is a valid window handle."""
        if not self._is_windows():
            return False
        try:
            return bool(ctypes.windll.user32.IsWindow(wintypes.HWND(hwnd)))
        except Exception as exc:
            self.logger.debug("IsWindow API failed for HWND %s: %s", hwnd, exc)
            return False

    def _api_is_window_visible(self, hwnd: int) -> bool:
        """Native check if window is visible."""
        if not self._is_windows():
            return False
        try:
            return bool(ctypes.windll.user32.IsWindowVisible(wintypes.HWND(hwnd)))
        except Exception as exc:
            self.logger.debug("IsWindowVisible API failed for HWND %s: %s", hwnd, exc)
            return False

    def _api_get_foreground_window(self) -> int:
        """Native retrieval of current foreground window handle."""
        if not self._is_windows():
            return 0
        try:
            return int(ctypes.windll.user32.GetForegroundWindow() or 0)
        except Exception as exc:
            self.logger.debug("GetForegroundWindow API failed: %s", exc)
            return 0

    def _api_set_foreground_window(self, hwnd: int) -> bool:
        """Native request to bring window to foreground."""
        if not self._is_windows():
            return False
        try:
            user32 = ctypes.windll.user32
            # Call BringWindowToTop and SetForegroundWindow
            user32.BringWindowToTop(wintypes.HWND(hwnd))
            return bool(user32.SetForegroundWindow(wintypes.HWND(hwnd)))
        except Exception as exc:
            self.logger.debug("SetForegroundWindow API failed for HWND %s: %s", hwnd, exc)
            return False

    def _api_show_window(self, hwnd: int, cmd_show: int) -> bool:
        """Native ShowWindow state modifier."""
        if not self._is_windows():
            return False
        try:
            return bool(ctypes.windll.user32.ShowWindow(wintypes.HWND(hwnd), ctypes.c_int(cmd_show)))
        except Exception as exc:
            self.logger.debug("ShowWindow API failed for HWND %s with cmd %s: %s", hwnd, cmd_show, exc)
            return False

    def _api_post_wm_close(self, hwnd: int) -> bool:
        """Asynchronously post WM_CLOSE message to target window.

        Never blocks and never terminates process.
        """
        if not self._is_windows():
            return False
        try:
            return bool(ctypes.windll.user32.PostMessageW(
                wintypes.HWND(hwnd),
                wintypes.UINT(WM_CLOSE),
                wintypes.WPARAM(0),
                wintypes.LPARAM(0),
            ))
        except Exception as exc:
            self.logger.debug("PostMessageW(WM_CLOSE) failed for HWND %s: %s", hwnd, exc)
            return False

    def _api_get_window_text(self, hwnd: int) -> str:
        """Retrieve window title text."""
        if not self._is_windows():
            return ""
        try:
            user32 = ctypes.windll.user32
            length = user32.GetWindowTextLengthW(wintypes.HWND(hwnd))
            if length <= 0:
                return ""
            buff = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(wintypes.HWND(hwnd), buff, length + 1)
            return str(buff.value).strip()
        except Exception as exc:
            self.logger.debug("GetWindowTextW failed for HWND %s: %s", hwnd, exc)
            return ""

    def _api_get_window_pid(self, hwnd: int) -> Optional[int]:
        """Retrieve process PID owning the window handle."""
        if not self._is_windows():
            return None
        try:
            pid = wintypes.DWORD()
            ctypes.windll.user32.GetWindowThreadProcessId(wintypes.HWND(hwnd), ctypes.byref(pid))
            val = int(pid.value)
            return val if val > 0 else None
        except Exception as exc:
            self.logger.debug("GetWindowThreadProcessId failed for HWND %s: %s", hwnd, exc)
            return None

    def _api_is_cloaked(self, hwnd: int) -> bool:
        """Check if window is cloaked (e.g., hidden Windows 10/11 UWP background app)."""
        if not self._is_windows():
            return False
        try:
            cloaked = wintypes.DWORD()
            res = ctypes.windll.dwmapi.DwmGetWindowAttribute(
                wintypes.HWND(hwnd),
                wintypes.DWORD(DWMWA_CLOAKED),
                ctypes.byref(cloaked),
                ctypes.sizeof(cloaked),
            )
            return (res == 0) and (cloaked.value != 0)
        except Exception:
            return False

    def _api_enum_windows(self) -> List[int]:
        """Enumerate top-level windows safely with hard upper bound."""
        if not self._is_windows():
            return []

        handles: List[int] = []
        user32 = ctypes.windll.user32

        # Define callback signature: BOOL CALLBACK EnumWindowsProc(HWND hwnd, LPARAM lParam)
        enum_proc_type = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

        def _enum_cb(hwnd: Any, _lparam: Any) -> bool:
            try:
                h_val = int(hwnd)
                if h_val > 0:
                    handles.append(h_val)
                # Hard limit to prevent infinite enumeration loops
                if len(handles) >= MAX_WINDOW_ENUM_LIMIT:
                    return False
            except Exception:
                pass
            return True

        cb_func = enum_proc_type(_enum_cb)
        try:
            user32.EnumWindows(cb_func, 0)
        except Exception as exc:
            self.logger.debug("EnumWindows failed: %s", exc)

        return handles

    # -----------------------------------------------------------------------
    # Helper Inspection & Resolution
    # -----------------------------------------------------------------------

    def _get_process_name(self, pid: Optional[int]) -> Optional[str]:
        """Safely resolve process executable name for a given PID."""
        if pid is None or pid <= 0:
            return None
        if _HAS_PSUTIL and psutil is not None:
            try:
                return psutil.Process(pid).name()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return None
            except Exception:
                return None
        return None

    def _inspect_window(self, hwnd: int, foreground_hwnd: Optional[int] = None) -> Optional[WindowInfo]:
        """Inspect and construct WindowInfo for a given HWND.

        Returns None if window handle is invalid or lacks title.
        """
        if not self._api_is_window(hwnd):
            return None

        title = self._api_get_window_text(hwnd)
        if not title:
            return None

        is_visible = self._api_is_window_visible(hwnd)
        if not is_visible or self._api_is_cloaked(hwnd):
            return None

        pid = self._api_get_window_pid(hwnd)
        p_name = self._get_process_name(pid)
        is_fg = (hwnd == foreground_hwnd) if foreground_hwnd is not None else False

        return WindowInfo(
            hwnd=hwnd,
            title=title,
            pid=pid,
            process_name=p_name,
            is_foreground=is_fg,
            is_visible=True,
        )

    def _resolve_target_window(
        self,
        target: Optional[Union[str, int]],
        parameters: Dict[str, Any],
    ) -> WindowInfo:
        """Resolve target identifier (HWND int, hex string, title, PID) into verified WindowInfo.

        Validates HWND existence immediately before execution to minimize TOCTOU window race conditions.

        Raises:
            SkillExecutionError: If window target cannot be found or is invalid.
        """
        # 1. Direct HWND parameter check
        raw_hwnd = parameters.get("hwnd") or parameters.get("handle")
        if raw_hwnd is not None:
            try:
                hwnd_val = int(raw_hwnd, 16) if isinstance(raw_hwnd, str) and raw_hwnd.startswith(("0x", "0X")) else int(raw_hwnd)
                if self._api_is_window(hwnd_val):
                    fg_hwnd = self._api_get_foreground_window()
                    win_info = self._inspect_window(hwnd_val, fg_hwnd)
                    if win_info is not None:
                        return win_info
                    # Valid HWND even if untitled
                    pid = self._api_get_window_pid(hwnd_val)
                    return WindowInfo(
                        hwnd=hwnd_val,
                        title=self._api_get_window_text(hwnd_val) or f"<HWND:{hwnd_val}>",
                        pid=pid,
                        process_name=self._get_process_name(pid),
                        is_foreground=(hwnd_val == fg_hwnd),
                        is_visible=self._api_is_window_visible(hwnd_val),
                    )
            except (ValueError, TypeError):
                pass

        target_str = str(target or "").strip()
        if not target_str:
            raise SkillExecutionError("No window target specified (must provide title, HWND, or PID).")

        # 2. Check if target_str itself is integer or hex HWND
        try:
            target_hwnd = int(target_str, 16) if target_str.startswith(("0x", "0X")) else int(target_str)
            if self._api_is_window(target_hwnd):
                fg_hwnd = self._api_get_foreground_window()
                win_info = self._inspect_window(target_hwnd, fg_hwnd)
                if win_info is not None:
                    return win_info
                pid = self._api_get_window_pid(target_hwnd)
                return WindowInfo(
                    hwnd=target_hwnd,
                    title=self._api_get_window_text(target_hwnd) or f"<HWND:{target_hwnd}>",
                    pid=pid,
                    process_name=self._get_process_name(pid),
                    is_foreground=(target_hwnd == fg_hwnd),
                    is_visible=self._api_is_window_visible(target_hwnd),
                )
        except (ValueError, TypeError):
            pass

        # 3. Enumerate open windows and match title or PID
        all_windows = self._enumerate_open_windows()
        target_lower = target_str.lower()

        # Check exact title match
        for w in all_windows:
            if w.title.lower() == target_lower:
                return w

        # Check substring title match
        for w in all_windows:
            if target_lower in w.title.lower():
                return w

        # Check process name match
        for w in all_windows:
            if w.process_name and target_lower in w.process_name.lower():
                return w

        # Check PID match
        if target_str.isdigit():
            target_pid = int(target_str)
            for w in all_windows:
                if w.pid == target_pid:
                    return w

        raise SkillExecutionError(f"No active window found matching target '{target_str}'.")

    def _enumerate_open_windows(self) -> List[WindowInfo]:
        """Collect list of all active visible user windows, bounded by MAX_WINDOW_ENUM_LIMIT."""
        with self._lock:
            fg_hwnd = self._api_get_foreground_window()
            raw_handles = self._api_enum_windows()
            results: List[WindowInfo] = []

            for hwnd in raw_handles:
                info = self._inspect_window(hwnd, fg_hwnd)
                if info is not None:
                    results.append(info)
                if len(results) >= MAX_WINDOW_ENUM_LIMIT:
                    break

            return results

    # -----------------------------------------------------------------------
    # BaseSystemSkill Protocol Methods
    # -----------------------------------------------------------------------

    def can_handle(self, command: Any) -> bool:
        """Evaluate whether this skill handles the given command."""
        op, _, _, _ = self.parse_command(command)
        if op in (
            "list_windows",
            "list_open_windows",
            "get_active_window",
            "focus_window",
            "minimize_window",
            "maximize_window",
            "restore_window",
            "close_window",
        ):
            return True

        if isinstance(command, str):
            cmd_clean = command.strip().lower()
            if _RE_LIST_WINDOWS.match(cmd_clean):
                return True
            if _RE_GET_ACTIVE.match(cmd_clean):
                return True
            if _RE_FOCUS_WINDOW.match(cmd_clean):
                return True
            if _RE_MINIMIZE_WINDOW.match(cmd_clean):
                return True
            if _RE_MAXIMIZE_WINDOW.match(cmd_clean):
                return True
            if _RE_RESTORE_WINDOW.match(cmd_clean):
                return True
            if _RE_CLOSE_WINDOW.match(cmd_clean):
                return True

        return False

    def parse_command(
        self, command: Any
    ) -> Tuple[str, Optional[str], Dict[str, Any], Optional[str]]:
        """Parse arbitrary command into structured components."""
        if isinstance(command, dict):
            return super().parse_command(command)

        text = str(command or "").strip()

        m_list = _RE_LIST_WINDOWS.match(text)
        if m_list:
            return "list_windows", None, {}, None

        m_act = _RE_GET_ACTIVE.match(text)
        if m_act:
            return "get_active_window", None, {}, None

        m_focus = _RE_FOCUS_WINDOW.match(text)
        if m_focus:
            return "focus_window", m_focus.group(1).strip(), {}, None

        m_min = _RE_MINIMIZE_WINDOW.match(text)
        if m_min:
            return "minimize_window", m_min.group(1).strip(), {}, None

        m_max = _RE_MAXIMIZE_WINDOW.match(text)
        if m_max:
            return "maximize_window", m_max.group(1).strip(), {}, None

        m_res = _RE_RESTORE_WINDOW.match(text)
        if m_res:
            return "restore_window", m_res.group(1).strip(), {}, None

        m_close = _RE_CLOSE_WINDOW.match(text)
        if m_close:
            return "close_window", m_close.group(1).strip(), {}, None

        return super().parse_command(command)

    def _execute_operation(
        self, operation: str, target: Optional[str], parameters: Dict[str, Any]
    ) -> Any:
        """Internal operation dispatcher."""
        op = operation.strip().lower()

        if op in ("list_windows", "list_open_windows"):
            return self._handle_list_windows(parameters)

        if op == "get_active_window":
            return self._handle_get_active_window(parameters)

        if op == "focus_window":
            return self._handle_focus_window(target, parameters)

        if op == "minimize_window":
            return self._handle_minimize_window(target, parameters)

        if op == "maximize_window":
            return self._handle_maximize_window(target, parameters)

        if op == "restore_window":
            return self._handle_restore_window(target, parameters)

        if op == "close_window":
            return self._handle_close_window(target, parameters)

        raise SkillExecutionError(f"Unsupported window operation '{operation}'.")

    # -----------------------------------------------------------------------
    # Operational Handlers
    # -----------------------------------------------------------------------

    def _handle_list_windows(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """List open desktop windows."""
        windows = self._enumerate_open_windows()
        window_dicts = [
            {
                "hwnd": w.hwnd,
                "title": w.title,
                "pid": w.pid,
                "process_name": w.process_name,
                "is_foreground": w.is_foreground,
                "is_visible": w.is_visible,
            }
            for w in windows
        ]
        return {
            "windows": window_dicts,
            "count": len(window_dicts),
        }

    def _handle_get_active_window(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve the currently focused foreground window."""
        with self._lock:
            fg_hwnd = self._api_get_foreground_window()
            if not fg_hwnd or not self._api_is_window(fg_hwnd):
                return {
                    "active_window": None,
                    "status": "no_active_window",
                }

            info = self._inspect_window(fg_hwnd, fg_hwnd)
            if info is None:
                pid = self._api_get_window_pid(fg_hwnd)
                info = WindowInfo(
                    hwnd=fg_hwnd,
                    title=self._api_get_window_text(fg_hwnd) or f"<HWND:{fg_hwnd}>",
                    pid=pid,
                    process_name=self._get_process_name(pid),
                    is_foreground=True,
                    is_visible=self._api_is_window_visible(fg_hwnd),
                )

            return {
                "active_window": {
                    "hwnd": info.hwnd,
                    "title": info.title,
                    "pid": info.pid,
                    "process_name": info.process_name,
                    "is_foreground": True,
                    "is_visible": info.is_visible,
                }
            }

    def _handle_focus_window(
        self, target: Optional[str], parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Bring target window to the foreground."""
        with self._lock:
            win = self._resolve_target_window(target, parameters)
            hwnd = win.hwnd

            # If minimized, restore it first so SetForegroundWindow succeeds
            self._api_show_window(hwnd, SW_RESTORE)

            success = self._api_set_foreground_window(hwnd)
            return {
                "hwnd": hwnd,
                "title": win.title,
                "pid": win.pid,
                "focused": bool(success),
            }

    def _handle_minimize_window(
        self, target: Optional[str], parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Minimize target window."""
        with self._lock:
            win = self._resolve_target_window(target, parameters)
            hwnd = win.hwnd
            success = self._api_show_window(hwnd, SW_MINIMIZE)
            return {
                "hwnd": hwnd,
                "title": win.title,
                "pid": win.pid,
                "minimized": bool(success),
            }

    def _handle_maximize_window(
        self, target: Optional[str], parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Maximize target window."""
        with self._lock:
            win = self._resolve_target_window(target, parameters)
            hwnd = win.hwnd
            success = self._api_show_window(hwnd, SW_MAXIMIZE)
            return {
                "hwnd": hwnd,
                "title": win.title,
                "pid": win.pid,
                "maximized": bool(success),
            }

    def _handle_restore_window(
        self, target: Optional[str], parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Restore target window to normal bounds."""
        with self._lock:
            win = self._resolve_target_window(target, parameters)
            hwnd = win.hwnd
            success = self._api_show_window(hwnd, SW_RESTORE)
            return {
                "hwnd": hwnd,
                "title": win.title,
                "pid": win.pid,
                "restored": bool(success),
            }

    def _handle_close_window(
        self, target: Optional[str], parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Close target window via WM_CLOSE.

        Enforces:
        - Critical OS processes and J.A.R.V.I.S PID are rejected as RESTRICTED.
        - Must be validated immediately before posting message.
        - Uses asynchronous PostMessageW(WM_CLOSE) only.
        - Never terminates process or executes shell commands.
        """
        with self._lock:
            win = self._resolve_target_window(target, parameters)
            hwnd = win.hwnd
            pid = win.pid
            p_name = win.process_name or ""

            # TOCTOU safety guard: Double-check critical process status immediately before posting message
            if pid is not None and is_critical_process(pid):
                raise SecurityPolicyViolationError(
                    f"Target window HWND {hwnd} is owned by critical system PID {pid} and cannot be closed."
                )
            if p_name and is_critical_process(p_name):
                raise SecurityPolicyViolationError(
                    f"Target window HWND {hwnd} is owned by critical system process '{p_name}' and cannot be closed."
                )

            # Post WM_CLOSE message asynchronously
            posted = self._api_post_wm_close(hwnd)
            if not posted:
                raise SkillExecutionError(f"Failed to post WM_CLOSE message to window handle {hwnd}.")

            self.logger.info("Posted WM_CLOSE to window HWND %d ('%s', PID %s)", hwnd, win.title, pid)

            return {
                "hwnd": hwnd,
                "title": win.title,
                "pid": pid,
                "closed": True,
                "method": "WM_CLOSE",
            }


__all__ = [
    "MAX_WINDOW_ENUM_LIMIT",
    "SW_MAXIMIZE",
    "SW_MINIMIZE",
    "SW_RESTORE",
    "WM_CLOSE",
    "WindowSkills",
]
