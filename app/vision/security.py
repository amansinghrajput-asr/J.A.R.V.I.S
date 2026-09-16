"""Vision Privacy and Security Guardrails for J.A.R.V.I.S. Phase 27.2.

Enforces deterministic, explainable safety policies governing all desktop screen
capture operations prior to backend capture execution:
1. Sensitive-window detection (password managers, banking, credential dialogs).
2. Private / Incognito browsing protection where reliably detectable.
3. Explicit capture authorization (ALLOW / BLOCK).
4. Ephemeral buffer lifecycle (TTL expiration, single-frame buffer, zero disk persistence).
5. SecureVisionManager: unified secure capture boundary preventing accidental policy bypass.
6. Safe event bus emission and security logging with zero credential/secret leakage.

Safety Invariants:
- All checks happen ON DEMAND before capture execution.
- No continuous monitoring or background recording loops.
- No screenshots or raw pixel bytes in logs or error messages.
- Zero filesystem persistence.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import logging
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import uuid

from app.core.container import ServiceContainer, container as default_container
from app.core.event_bus import EventBus, event_bus as default_event_bus
from app.core.logger import get_logger
from app.vision.capture import DesktopCaptureEngine
from app.vision.models import (
    CaptureAuthorization,
    CaptureBlockedError,
    CaptureCategory,
    CaptureDecision,
    CaptureError,
    ScreenCapture,
    ScreenObservation,
    WindowBounds,
)

logger = get_logger("VISION.SECURITY")


# --------------------------------------------------------------------------
# Sensitive Window Rules
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SensitiveWindowRule:
    """Configurable rule matching window title or process names against sensitive contexts.

    Attributes:
        name: Unique rule identifier (e.g. 'password_managers').
        category: CaptureCategory classification.
        description: Explainable rationale for why this rule triggers.
        process_patterns: Process name substrings or exact matches (case-insensitive).
        title_patterns: Window title substrings or regex patterns (case-insensitive).
        exact_process_match: If True, requires exact process name match (e.g. 'bitwarden.exe').
        require_title_match: If True, title MUST match title_patterns (prevents blocking all normal windows of an app).
        priority: Evaluation priority (higher numbers evaluated first).
    """

    name: str
    category: CaptureCategory
    description: str
    process_patterns: Tuple[str, ...] = field(default_factory=tuple)
    title_patterns: Tuple[Union[str, re.Pattern[str]], ...] = field(default_factory=tuple)
    exact_process_match: bool = False
    require_title_match: bool = False
    priority: int = 100

    def matches(
        self,
        title: Optional[str] = None,
        process_name: Optional[str] = None,
    ) -> bool:
        """Evaluate whether target window metadata satisfies this rule.

        Matching is strictly case-insensitive.
        """
        proc_clean = (process_name or "").strip().lower()
        title_clean = (title or "").strip().lower()

        # 1. Evaluate process patterns
        proc_matched = False
        if self.process_patterns and proc_clean:
            # Strip .exe extension for comparison flexibility
            proc_no_ext = proc_clean[:-4] if proc_clean.endswith(".exe") else proc_clean

            for p in self.process_patterns:
                p_clean = p.strip().lower()
                p_no_ext = p_clean[:-4] if p_clean.endswith(".exe") else p_clean

                if self.exact_process_match:
                    if proc_clean == p_clean or proc_no_ext == p_no_ext:
                        proc_matched = True
                        break
                else:
                    if p_no_ext in proc_no_ext or p_clean in proc_clean:
                        proc_matched = True
                        break

        # 2. Evaluate title patterns
        title_matched = False
        if self.title_patterns and title_clean:
            for t in self.title_patterns:
                if isinstance(t, re.Pattern):
                    if t.search(title_clean):
                        title_matched = True
                        break
                elif isinstance(t, str):
                    if t.lower() in title_clean:
                        title_matched = True
                        break

        if self.require_title_match:
            # If require_title_match is True, title must match
            if not title_matched:
                return False
            # If process patterns are specified, process must also match if given
            if self.process_patterns and proc_clean and not proc_matched:
                return False
            return True

        return proc_matched or title_matched


# --------------------------------------------------------------------------
# Default Rule Definitions
# --------------------------------------------------------------------------

_DEFAULT_RULES: Tuple[SensitiveWindowRule, ...] = (
    # 1. Password Managers & Secrets Vaults
    SensitiveWindowRule(
        name="password_managers",
        category=CaptureCategory.SENSITIVE_APPLICATION,
        description="Password manager or secrets vault application detected.",
        process_patterns=(
            "bitwarden",
            "1password",
            "keepass",
            "keepassxc",
            "lastpass",
            "dashlane",
            "enpass",
            "roboform",
            "nordpass",
        ),
        title_patterns=(
            "bitwarden",
            "1password",
            "keepass",
            "lastpass",
            "dashlane",
            "enpass",
            "roboform",
            "nordpass",
        ),
        priority=200,
    ),
    # 2. Windows Credential & Security Interfaces
    SensitiveWindowRule(
        name="credential_interfaces",
        category=CaptureCategory.CREDENTIAL_INTERFACE,
        description="Operating system credential prompt or security dialogue detected.",
        process_patterns=(
            "credentialuibroker",
            "consent",
            "sechealthui",
        ),
        title_patterns=(
            "windows security",
            "credential manager",
            "enter credentials",
            "user account control",
            "smart card",
        ),
        priority=190,
    ),
    # 3. Financial & Banking Contexts
    SensitiveWindowRule(
        name="financial_banking",
        category=CaptureCategory.SENSITIVE_APPLICATION,
        description="Banking, financial portal, or payment transaction interface detected.",
        title_patterns=(
            "online banking",
            "netbanking",
            "credit card checkout",
            "paypal checkout",
            "bank of america",
            "chase bank",
            "wells fargo",
            "citi online",
            "wire transfer",
        ),
        priority=180,
    ),
    # 4. Private / Incognito Browsing Contexts
    SensitiveWindowRule(
        name="private_browsing",
        category=CaptureCategory.PRIVATE_BROWSING,
        description="Private or Incognito browsing window detected.",
        process_patterns=(
            "chrome",
            "msedge",
            "brave",
            "firefox",
            "opera",
            "vivaldi",
            "tor",
        ),
        title_patterns=(
            re.compile(r"\bincognito\b", re.IGNORECASE),
            re.compile(r"\binprivate\b", re.IGNORECASE),
            re.compile(r"\bprivate\s+browsing\b", re.IGNORECASE),
            re.compile(r"\btor\s+browser\b", re.IGNORECASE),
        ),
        require_title_match=True,
        priority=170,
    ),
)


# --------------------------------------------------------------------------
# Vision Security Policy
# --------------------------------------------------------------------------


class VisionSecurityPolicy:
    """Thread-safe policy evaluating capture requests against sensitive window rules.

    Guarantees:
    - Case-insensitive, explainable matching.
    - Deterministic ALLOW or BLOCK verdict before screen capture.
    - Zero secrets or private titles in the authorization reason.
    """

    def __init__(
        self,
        rules: Optional[Sequence[SensitiveWindowRule]] = None,
        logger_instance: Optional[logging.Logger] = None,
    ) -> None:
        self._lock = threading.RLock()
        self._logger = logger_instance or logger
        self._rules: List[SensitiveWindowRule] = list(rules) if rules is not None else list(_DEFAULT_RULES)
        self._sort_rules()

    def _sort_rules(self) -> None:
        """Sort rules descending by priority."""
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def add_rule(self, rule: SensitiveWindowRule) -> None:
        """Register a custom sensitive window rule."""
        with self._lock:
            # Replace if already exists with same name
            self._rules = [r for r in self._rules if r.name != rule.name]
            self._rules.append(rule)
            self._sort_rules()
            self._logger.debug("Added vision security rule: %s (Priority: %d)", rule.name, rule.priority)

    def remove_rule(self, rule_name: str) -> bool:
        """Remove a rule by identifier."""
        with self._lock:
            init_len = len(self._rules)
            self._rules = [r for r in self._rules if r.name != rule_name]
            return len(self._rules) < init_len

    def get_rule(self, rule_name: str) -> Optional[SensitiveWindowRule]:
        """Look up a rule by name."""
        with self._lock:
            for r in self._rules:
                if r.name == rule_name:
                    return r
            return None

    def get_rules(self) -> List[SensitiveWindowRule]:
        """Return a copy of all active rules."""
        with self._lock:
            return list(self._rules)

    def authorize(
        self,
        target_hwnd: Optional[int] = None,
        window_title: Optional[str] = None,
        process_name: Optional[str] = None,
        source: str = "screen",
    ) -> CaptureAuthorization:
        """Evaluate whether a screen capture request is permitted.

        Returns an explicit CaptureAuthorization object.
        """
        with self._lock:
            # Sanitize target label for logging (never log full private URL or raw password title)
            target_label = process_name or (f"HWND:{target_hwnd}" if target_hwnd else source)

            for rule in self._rules:
                if rule.matches(title=window_title, process_name=process_name):
                    safe_reason = (
                        f"Screen capture blocked: Protected {rule.category.value} "
                        f"context detected ({rule.name})."
                    )
                    self._logger.warning(
                        "Vision capture blocked by policy rule '%s' [Category: %s, Target: %s]",
                        rule.name,
                        rule.category.value,
                        target_label,
                    )
                    return CaptureAuthorization(
                        decision=CaptureDecision.BLOCK,
                        category=rule.category,
                        reason=safe_reason,
                        rule_name=rule.name,
                        target=target_label,
                        timestamp=time.time(),
                    )

            # Permitted
            self._logger.debug("Vision capture permitted for target: %s [Source: %s]", target_label, source)
            return CaptureAuthorization(
                decision=CaptureDecision.ALLOW,
                category=CaptureCategory.SAFE,
                reason="Screen capture permitted.",
                target=target_label,
                timestamp=time.time(),
            )


# --------------------------------------------------------------------------
# Ephemeral Buffer Manager
# --------------------------------------------------------------------------


class EphemeralBufferManager:
    """Manages the in-memory lifecycle of captured screen observations.

    Guarantees:
    - Transient in-memory storage only (zero filesystem writes).
    - Single-frame replacement by default to prevent memory hoarding.
    - Strict TTL expiration (default 30 seconds).
    - Thread-safe store, get, clear, and close operations.
    """

    def __init__(
        self,
        default_ttl_seconds: float = 30.0,
        max_buffers: int = 1,
        event_bus_instance: Optional[EventBus] = None,
        logger_instance: Optional[logging.Logger] = None,
    ) -> None:
        self._lock = threading.RLock()
        self._default_ttl = max(1.0, float(default_ttl_seconds))
        self._max_buffers = max(1, min(2, int(max_buffers)))
        self._event_bus = event_bus_instance
        self._logger = logger_instance or logger
        self._buffers: Dict[str, ScreenObservation] = {}
        self._latest_id: Optional[str] = None

    @property
    def count(self) -> int:
        """Current number of active non-expired observations."""
        with self._lock:
            self._purge_expired()
            return len(self._buffers)

    @property
    def is_empty(self) -> bool:
        """Return True if no non-expired observations are retained."""
        return self.count == 0

    def _purge_expired(self) -> None:
        """Remove observations whose TTL has lapsed."""
        now = time.time()
        expired_ids = [oid for oid, obs in self._buffers.items() if obs.expires_at and now >= obs.expires_at]
        for oid in expired_ids:
            del self._buffers[oid]
            if self._latest_id == oid:
                self._latest_id = None
            self._emit_event("vision.buffer_cleared", {"id": oid, "reason": "ttl_expired"})

    def store(
        self,
        capture: ScreenCapture,
        authorization: Optional[CaptureAuthorization] = None,
        ttl_seconds: Optional[float] = None,
        observation_id: Optional[str] = None,
        source: str = "screen",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ScreenObservation:
        """Store a new screen capture buffer, evicting previous frame if exceeding capacity."""
        with self._lock:
            self._purge_expired()

            # Enforce max buffer count by evicting oldest
            while len(self._buffers) >= self._max_buffers:
                oldest_id = next(iter(self._buffers))
                del self._buffers[oldest_id]
                self._emit_event("vision.buffer_cleared", {"id": oldest_id, "reason": "capacity_eviction"})

            ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
            now = time.time()
            expires_at = now + ttl

            obs_id = observation_id or str(uuid.uuid4())
            auth = authorization or CaptureAuthorization(
                decision=CaptureDecision.ALLOW,
                category=CaptureCategory.SAFE,
                reason="Direct store authorization.",
                timestamp=now,
            )

            observation = ScreenObservation(
                id=obs_id,
                capture=capture,
                authorization=auth,
                source=source,
                timestamp=now,
                expires_at=expires_at,
                metadata=dict(metadata or {}),
            )

            self._buffers[obs_id] = observation
            self._latest_id = obs_id

            self._emit_event(
                "vision.buffer_stored",
                {
                    "id": obs_id,
                    "width": capture.width,
                    "height": capture.height,
                    "size_bytes": capture.size_bytes,
                    "expires_at": expires_at,
                },
            )

            return observation

    def get(self, observation_id: Optional[str] = None) -> Optional[ScreenObservation]:
        """Retrieve observation by ID, or the latest observation if ID is omitted.

        Returns None if observation is expired or does not exist.
        """
        with self._lock:
            self._purge_expired()

            target_id = observation_id or self._latest_id
            if not target_id:
                return None

            obs = self._buffers.get(target_id)
            if obs is None or obs.is_expired:
                return None

            return obs

    def get_previous(self) -> Optional[ScreenObservation]:
        """Retrieve the observation stored immediately prior to the latest observation.

        Returns None if fewer than 2 unexpired observations exist.
        """
        with self._lock:
            self._purge_expired()
            obs_list = list(self._buffers.values())
            if len(obs_list) >= 2:
                prev_obs = obs_list[-2]
                if not prev_obs.is_expired:
                    return prev_obs
            return None

    def get_pair(self) -> Tuple[Optional[ScreenObservation], Optional[ScreenObservation]]:
        """Retrieve the bounded (previous, latest) observation pair in chronological order.

        Returns (before, after). Either or both may be None if absent or expired.
        """
        with self._lock:
            self._purge_expired()
            latest = self.get()
            previous = self.get_previous()
            return (previous, latest)

    def clear(self) -> None:
        """Explicitly evict and drop all stored screen capture buffers."""
        with self._lock:
            count = len(self._buffers)
            self._buffers.clear()
            self._latest_id = None
            if count > 0:
                self._emit_event("vision.buffer_cleared", {"count": count, "reason": "explicit_clear"})
                self._logger.debug("Cleared %d ephemeral screen buffers.", count)

    def close(self) -> None:
        """Clean up and clear buffer manager."""
        self.clear()

    def _emit_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        """Publish safe lifecycle event to EventBus."""
        if self._event_bus is not None and hasattr(self._event_bus, "publish"):
            try:
                self._event_bus.publish(event_name, payload)
            except Exception:
                pass


# --------------------------------------------------------------------------
# Secure Vision Manager (Facade)
# --------------------------------------------------------------------------


class SecureVisionManager:
    """Security facade wrapping DesktopCaptureEngine with VisionSecurityPolicy.

    Guarantees:
    - Every capture request MUST pass VisionSecurityPolicy authorization BEFORE
      any Win32 GDI BitBlt execution occurs.
    - Protected contexts (passwords, credentials, banking, incognito) raise
      CaptureBlockedError (or return blocked ScreenObservation if configured).
    - Captures are immediately managed by EphemeralBufferManager (in RAM only).
    - Safe logging and EventBus telemetry with zero secret or image exposure.
    - Async non-blocking API wrappers.
    """

    def __init__(
        self,
        engine: Optional[DesktopCaptureEngine] = None,
        policy: Optional[VisionSecurityPolicy] = None,
        buffer_manager: Optional[EphemeralBufferManager] = None,
        container_instance: Optional[ServiceContainer] = None,
        event_bus_instance: Optional[EventBus] = None,
        logger_instance: Optional[logging.Logger] = None,
        *,
        auto_register_in_container: bool = True,
    ) -> None:
        self._lock = threading.RLock()
        self._logger = logger_instance or logger
        self._container = container_instance if container_instance is not None else default_container
        self._event_bus = event_bus_instance if event_bus_instance is not None else default_event_bus

        # 1. Dependency: Engine
        if engine is not None:
            self._engine = engine
        elif self._container is not None and self._container.exists("desktop_capture_engine"):
            self._engine = self._container.resolve("desktop_capture_engine")
        else:
            self._engine = DesktopCaptureEngine(
                container_instance=self._container,
                event_bus_instance=self._event_bus,
                logger_instance=self._logger,
                auto_register_in_container=False,
            )

        # 2. Dependency: Policy
        self._policy = policy or VisionSecurityPolicy(logger_instance=self._logger)

        # 3. Dependency: Ephemeral Buffer Manager
        self._buffer_manager = buffer_manager or EphemeralBufferManager(
            event_bus_instance=self._event_bus,
            logger_instance=self._logger,
        )

        # 4. Register singletons in container
        if auto_register_in_container and self._container is not None:
            try:
                self._container.register_singleton("secure_vision_manager", self, allow_override=True)
                self._container.register_singleton("vision_security_policy", self._policy, allow_override=True)
                self._container.register_singleton("ephemeral_buffer_manager", self._buffer_manager, allow_override=True)
                self._logger.debug("Registered SecureVisionManager singletons in ServiceContainer.")
            except Exception as exc:
                self._logger.warning("Could not register SecureVisionManager in container: %s", exc)

    @property
    def engine(self) -> DesktopCaptureEngine:
        """Return underlying capture engine."""
        return self._engine

    @property
    def policy(self) -> VisionSecurityPolicy:
        """Return active security policy."""
        return self._policy

    @property
    def buffer_manager(self) -> EphemeralBufferManager:
        """Return active ephemeral buffer manager."""
        return self._buffer_manager

    # ----------------------------------------------------------------------
    # Active Window Metadata Resolution Helper
    # ----------------------------------------------------------------------

    def _resolve_window_metadata(self, hwnd: int) -> Tuple[str, Optional[str]]:
        """Safely resolve window title and process name for an HWND via WindowSkills or OS."""
        if not hwnd:
            return ("", None)

        # 1. Try WindowSkills if registered
        if self._container is not None and self._container.exists("window_skills"):
            try:
                ws = self._container.resolve("window_skills")
                if hasattr(ws, "_inspect_window"):
                    win_info = ws._inspect_window(hwnd)
                    if win_info:
                        return (win_info.title, win_info.process_name)
                if hasattr(ws, "_api_get_window_text"):
                    title = ws._api_get_window_text(hwnd)
                    pid = ws._api_get_window_pid(hwnd) if hasattr(ws, "_api_get_window_pid") else None
                    proc = ws._get_process_name(pid) if hasattr(ws, "_get_process_name") else None
                    return (title, proc)
            except Exception:
                pass

        # 2. Try direct Win32 calls if available
        try:
            import ctypes
            from ctypes import wintypes
            user32 = ctypes.windll.user32
            length = user32.GetWindowTextLengthW(wintypes.HWND(hwnd))
            title = ""
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(wintypes.HWND(hwnd), buf, length + 1)
                title = str(buf.value).strip()

            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(wintypes.HWND(hwnd), ctypes.byref(pid))
            proc_name = None
            if pid.value > 0:
                try:
                    import psutil
                    proc_name = psutil.Process(pid.value).name()
                except Exception:
                    pass
            return (title, proc_name)
        except Exception:
            pass

        return ("", None)

    # ----------------------------------------------------------------------
    # Secure Capture Execution Core
    # ----------------------------------------------------------------------

    def _evaluate_and_capture(
        self,
        capture_fn: Callable[[], ScreenCapture],
        target_hwnd: Optional[int] = None,
        source: str = "screen",
        raise_on_blocked: bool = True,
        ttl_seconds: Optional[float] = None,
    ) -> ScreenObservation:
        """Centralized security enforcement gate preceding all capture operations."""
        with self._lock:
            # 1. Determine active/target window metadata
            hwnd_to_check = target_hwnd or self._engine.get_active_window_handle()
            title, proc_name = self._resolve_window_metadata(hwnd_to_check)

            # 2. Authorize via policy BEFORE any GDI BitBlt
            auth = self._policy.authorize(
                target_hwnd=hwnd_to_check,
                window_title=title,
                process_name=proc_name,
                source=source,
            )

            # 3. Handle BLOCK decision
            if not auth.is_allowed:
                self._emit_event(
                    "vision.capture_blocked",
                    {
                        "source": source,
                        "rule_name": auth.rule_name,
                        "category": auth.category.value,
                        "target": auth.target,
                    },
                )
                if raise_on_blocked:
                    raise CaptureBlockedError(auth.reason, authorization=auth)

                # Return empty observation with blocked authorization
                return ScreenObservation(
                    capture=None,
                    authorization=auth,
                    source=source,
                    timestamp=time.time(),
                    metadata={"blocked": True},
                )

            # 4. Handle ALLOW decision -> Perform actual capture
            self._emit_event(
                "vision.capture_allowed",
                {
                    "source": source,
                    "target": auth.target,
                },
            )

            capture = capture_fn()

            # 5. Store in ephemeral buffer with compound window identity
            bounds_tuple = capture.bounds.to_tuple() if (capture and capture.bounds) else None
            metadata = {
                "hwnd": hwnd_to_check,
                "window_title": title,
                "process_name": proc_name,
                "bounds": bounds_tuple,
            }
            observation = self._buffer_manager.store(
                capture=capture,
                authorization=auth,
                ttl_seconds=ttl_seconds,
                source=source,
                metadata=metadata,
            )

            return observation

    # ----------------------------------------------------------------------
    # Public Secure Capture APIs
    # ----------------------------------------------------------------------

    def capture_screen(
        self,
        bounds: Optional[WindowBounds] = None,
        source: str = "screen",
        raise_on_blocked: bool = True,
        ttl_seconds: Optional[float] = None,
    ) -> ScreenObservation:
        """Securely capture rectangular screen region, checking foreground privacy."""
        return self._evaluate_and_capture(
            capture_fn=lambda: self._engine.capture_screen(bounds=bounds, source=source),
            target_hwnd=None,  # Checks foreground window for desktop capture
            source=source,
            raise_on_blocked=raise_on_blocked,
            ttl_seconds=ttl_seconds,
        )

    def capture_active_window(
        self,
        raise_on_blocked: bool = True,
        ttl_seconds: Optional[float] = None,
    ) -> ScreenObservation:
        """Securely capture the currently active foreground window."""
        hwnd = self._engine.get_active_window_handle()
        if not hwnd:
            raise CaptureError("No active foreground window found.")

        return self._evaluate_and_capture(
            capture_fn=lambda: self._engine.capture_active_window(),
            target_hwnd=hwnd,
            source="window",
            raise_on_blocked=raise_on_blocked,
            ttl_seconds=ttl_seconds,
        )

    def capture_window(
        self,
        hwnd: int,
        raise_on_blocked: bool = True,
        ttl_seconds: Optional[float] = None,
    ) -> ScreenObservation:
        """Securely capture a specific window handle, checking its privacy status."""
        return self._evaluate_and_capture(
            capture_fn=lambda: self._engine.capture_window(hwnd),
            target_hwnd=hwnd,
            source="window",
            raise_on_blocked=raise_on_blocked,
            ttl_seconds=ttl_seconds,
        )

    def capture_monitor(
        self,
        monitor_index: int = 0,
        raise_on_blocked: bool = True,
        ttl_seconds: Optional[float] = None,
    ) -> ScreenObservation:
        """Securely capture a specific display monitor."""
        return self._evaluate_and_capture(
            capture_fn=lambda: self._engine.capture_monitor(monitor_index),
            target_hwnd=None,
            source=f"monitor_{monitor_index}",
            raise_on_blocked=raise_on_blocked,
            ttl_seconds=ttl_seconds,
        )

    def capture_virtual_desktop(
        self,
        raise_on_blocked: bool = True,
        ttl_seconds: Optional[float] = None,
    ) -> ScreenObservation:
        """Securely capture the entire multi-monitor virtual desktop."""
        return self._evaluate_and_capture(
            capture_fn=lambda: self._engine.capture_virtual_desktop(),
            target_hwnd=None,
            source="virtual_desktop",
            raise_on_blocked=raise_on_blocked,
            ttl_seconds=ttl_seconds,
        )

    def get_latest_observation(self) -> Optional[ScreenObservation]:
        """Retrieve the most recent unexpired screen observation from ephemeral buffer."""
        return self._buffer_manager.get()

    def get_previous_observation(self) -> Optional[ScreenObservation]:
        """Retrieve the observation captured immediately prior to the latest observation."""
        return self._buffer_manager.get_previous()

    def get_observation_pair(self) -> Tuple[Optional[ScreenObservation], Optional[ScreenObservation]]:
        """Retrieve bounded observation pair (before, after) in chronological order."""
        return self._buffer_manager.get_pair()

    def get_observation(self, observation_id: Optional[str] = None) -> Optional[ScreenObservation]:
        """Retrieve observation by ID, or latest unexpired observation from ephemeral buffer."""
        return self._buffer_manager.get(observation_id)

    def get_active_window_identity(
        self,
    ) -> Tuple[Optional[int], str, Optional[str], Optional[Tuple[int, int, int, int]]]:
        """Resolve current active foreground window (hwnd, window_title, process_name, bounds) safely."""
        with self._lock:
            hwnd = (
                self._engine.get_active_window_handle()
                if self._engine and hasattr(self._engine, "get_active_window_handle")
                else None
            )
            title, proc_name = self._resolve_window_metadata(hwnd)
            bounds = None
            if hwnd and self._engine and hasattr(self._engine, "get_window_bounds"):
                try:
                    b = self._engine.get_window_bounds(hwnd)
                    bounds = b.to_tuple() if b else None
                except Exception:
                    bounds = None
            return (hwnd, title, proc_name, bounds)

    def check_active_window_authorized(
        self, target_hwnd: Optional[int] = None, source: str = "window"
    ) -> CaptureAuthorization:
        """Evaluate whether active foreground window is permitted by security policy.

        Ensures cached observation reuse never bypasses security when foreground context changes.
        """
        with self._lock:
            hwnd_to_check = (
                target_hwnd
                if target_hwnd is not None
                else (
                    self._engine.get_active_window_handle()
                    if self._engine and hasattr(self._engine, "get_active_window_handle")
                    else None
                )
            )
            title, proc_name = self._resolve_window_metadata(hwnd_to_check)
            return self._policy.authorize(
                target_hwnd=hwnd_to_check,
                window_title=title,
                process_name=proc_name,
                source=source,
            )

    def clear_buffers(self) -> None:
        """Evict all ephemeral screen observations from memory."""
        self.invalidate_cache(reason="explicit_clear")

    def invalidate_cache(self, reason: str = "explicit_invalidation") -> None:
        """Explicitly evict all ephemeral screen observations from memory with reason tracking.

        Observable via safe telemetry and logging only; zero persistent or raw data.
        """
        with self._lock:
            self._buffer_manager.clear()
            self._logger.debug("Vision observation cache invalidated: %s", reason)
            self._emit_event("vision.cache_invalidated", {"reason": str(reason)})

    # ----------------------------------------------------------------------
    # Asynchronous Secure Capture Wrappers
    # ----------------------------------------------------------------------

    async def capture_screen_async(
        self,
        bounds: Optional[WindowBounds] = None,
        source: str = "screen",
        raise_on_blocked: bool = True,
        ttl_seconds: Optional[float] = None,
    ) -> ScreenObservation:
        """Asynchronously execute secure screen capture on worker thread."""
        return await asyncio.to_thread(
            self.capture_screen, bounds, source, raise_on_blocked, ttl_seconds
        )

    async def capture_active_window_async(
        self,
        raise_on_blocked: bool = True,
        ttl_seconds: Optional[float] = None,
    ) -> ScreenObservation:
        """Asynchronously execute secure active-window capture on worker thread."""
        return await asyncio.to_thread(
            self.capture_active_window, raise_on_blocked, ttl_seconds
        )

    # ----------------------------------------------------------------------
    # Event Emission Helper
    # ----------------------------------------------------------------------

    def _emit_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        """Publish sanitized security event to EventBus."""
        if self._event_bus is not None and hasattr(self._event_bus, "publish"):
            try:
                self._event_bus.publish(event_name, payload)
            except Exception:
                pass


__all__ = [
    "EphemeralBufferManager",
    "SecureVisionManager",
    "SensitiveWindowRule",
    "VisionSecurityPolicy",
]
