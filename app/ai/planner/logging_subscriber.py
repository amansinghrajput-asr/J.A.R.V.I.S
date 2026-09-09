"""Structured Logging Subscriber for the J.A.R.V.I.S. Planner Subsystem.

Translates PlannerEvents into structured logs across standard levels (INFO, WARNING,
ERROR, DEBUG) while strictly enforcing automatic redaction of secrets, API keys,
bearer tokens, passwords, prompt content, and excessive outputs.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any, Callable, Dict, List, Optional, Set

from app.ai.planner.events import (
    PlanCancelled,
    PlanCompleted,
    PlanFailed,
    PlannerEvent,
    PlannerEventBus,
    PlanPaused,
    PlanResumed,
    PlanStarted,
    RecoveryAborted,
    RecoveryCompleted,
    RecoveryFailed,
    RecoveryStarted,
    TaskCompleted,
    TaskFailed,
    TaskRetried,
    TaskStarted,
    TaskTimeout,
)

logger = logging.getLogger("PLAN_OBSERVABILITY")

# Patterns identifying sensitive fields requiring redaction
_SENSITIVE_KEY_RE = re.compile(
    r"(?i)(api[-_]?key|secret|password|passwd|token|auth(?:orization)?|credential|bearer|prompt)"
)
_SECRET_VALUE_RE = re.compile(
    r"(?i)(?:bearer\s+[a-zA-Z0-9_\-\.]{15,}|sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{20,})"
)


def sanitize_value(val: Any, max_length: int = 150) -> Any:
    """Recursively redact secrets and truncate large outputs."""
    if isinstance(val, str):
        # Scrub explicit secret regex matches
        scrubbed = _SECRET_VALUE_RE.sub("[REDACTED_SECRET]", val)
        if len(scrubbed) > max_length:
            return f"{scrubbed[:max_length]}... [truncated]"
        return scrubbed
    elif isinstance(val, dict):
        sanitized_dict: Dict[str, Any] = {}
        for k, v in val.items():
            if _SENSITIVE_KEY_RE.search(str(k)):
                sanitized_dict[k] = "[REDACTED]"
            else:
                sanitized_dict[k] = sanitize_value(v, max_length=max_length)
        return sanitized_dict
    elif isinstance(val, (list, tuple, set)):
        return [sanitize_value(item, max_length=max_length) for item in val]
    return val


class StructuredLoggingSubscriber:
    """Subscribes to PlannerEventBus and emits secure, structured log statements."""

    def __init__(
        self,
        bus: Optional[PlannerEventBus] = None,
        custom_logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize subscriber with optional custom logger and event bus.

        Args:
            bus: Optional PlannerEventBus to attach to.
            custom_logger: Optional custom logger (defaults to PLAN_OBSERVABILITY).
        """
        self._lock = threading.RLock()
        self._logger = custom_logger or logger
        self._bus: Optional[PlannerEventBus] = bus
        self._unsubscribe_fn: Optional[Callable[[], bool]] = None

        if bus is not None:
            self.attach(bus)

    def attach(self, bus: PlannerEventBus) -> None:
        """Attach to a PlannerEventBus and start logging events."""
        with self._lock:
            if self._unsubscribe_fn is not None:
                self.detach()
            self._bus = bus
            self._unsubscribe_fn = bus.subscribe(PlannerEvent, self.on_event)

    def detach(self) -> None:
        """Detach from the active PlannerEventBus."""
        with self._lock:
            if self._unsubscribe_fn is not None:
                try:
                    self._unsubscribe_fn()
                except Exception:
                    pass
                self._unsubscribe_fn = None
            self._bus = None

    def on_event(self, event: PlannerEvent) -> None:
        """Format and emit a structured log record for the given event."""
        with self._lock:
            # 1. INFO: Major milestones and clean completions
            if isinstance(event, PlanStarted):
                safe_query = sanitize_value(event.query, max_length=80)
                self._logger.info(
                    "Plan started [seq=%d, id='%s']: %d task(s) for query '%s'",
                    event.sequence_id,
                    event.plan_id,
                    event.task_count,
                    safe_query,
                )

            elif isinstance(event, PlanCompleted):
                self._logger.info(
                    "Plan completed [seq=%d, id='%s']: %d completed, %d failed, %d skipped in %.3fs (success=%s)",
                    event.sequence_id,
                    event.plan_id,
                    event.completed_count,
                    event.failed_count,
                    event.skipped_count,
                    event.duration,
                    event.success,
                )

            elif isinstance(event, TaskCompleted):
                safe_res = sanitize_value(event.result, max_length=100)
                self._logger.info(
                    "Task completed [seq=%d, id='%s', action='%s'] in %.3fs: %s",
                    event.sequence_id,
                    event.task_id,
                    event.action,
                    event.duration,
                    safe_res,
                )

            elif isinstance(event, RecoveryCompleted):
                self._logger.info(
                    "Recovery completed [seq=%d, attempt=%d]: %d task(s) generated in %.3fs (success=%s)",
                    event.sequence_id,
                    event.attempt,
                    event.task_count,
                    event.duration,
                    event.success,
                )

            # 2. WARNING: Recoverable failures and retry events
            elif isinstance(event, TaskFailed):
                safe_err = sanitize_value(event.error, max_length=120)
                self._logger.warning(
                    "Task failed [seq=%d, id='%s', action='%s'] after %.3fs: %s",
                    event.sequence_id,
                    event.task_id,
                    event.action,
                    event.duration,
                    safe_err,
                )

            elif isinstance(event, TaskRetried):
                safe_reason = sanitize_value(event.reason, max_length=100)
                self._logger.warning(
                    "Task retried [seq=%d, id='%s', action='%s', attempt=%d]: %s",
                    event.sequence_id,
                    event.task_id,
                    event.action,
                    event.attempt,
                    safe_reason,
                )

            elif isinstance(event, RecoveryStarted):
                self._logger.warning(
                    "Recovery started [seq=%d, attempt=%d]: recovering failed tasks %s",
                    event.sequence_id,
                    event.attempt,
                    event.failed_task_ids,
                )

            elif isinstance(event, TaskTimeout):
                self._logger.warning(
                    "Task timeout [seq=%d, id='%s']: exceeded %.2fs (policy: %s)",
                    event.sequence_id,
                    event.task_id,
                    event.timeout_seconds,
                    event.policy,
                )

            # 3. ERROR: Unrecoverable errors and terminal plan failures
            elif isinstance(event, PlanFailed):
                safe_err = sanitize_value(event.error, max_length=150)
                self._logger.error(
                    "Plan failed [seq=%d, id='%s']: %s",
                    event.sequence_id,
                    event.plan_id,
                    safe_err,
                )

            elif isinstance(event, RecoveryFailed):
                safe_reason = sanitize_value(event.reason, max_length=150)
                self._logger.error(
                    "Recovery failed [seq=%d, attempt=%d]: %s",
                    event.sequence_id,
                    event.attempt,
                    safe_reason,
                )

            # 4. DEBUG: Granular trace information and control states
            elif isinstance(event, TaskStarted):
                safe_target = sanitize_value(event.target, max_length=60)
                self._logger.debug(
                    "Task started [seq=%d, id='%s', action='%s', target='%s', deps=%s]",
                    event.sequence_id,
                    event.task_id,
                    event.action,
                    safe_target,
                    event.dependencies,
                )

            elif isinstance(event, PlanPaused):
                self._logger.debug("Plan paused [seq=%d, id='%s']: %s", event.sequence_id, event.plan_id, event.reason)

            elif isinstance(event, PlanResumed):
                self._logger.debug("Plan resumed [seq=%d, id='%s']: %s", event.sequence_id, event.plan_id, event.reason)

            elif isinstance(event, PlanCancelled):
                self._logger.debug("Plan cancelled [seq=%d, id='%s']: %s", event.sequence_id, event.plan_id, event.reason)

            elif isinstance(event, RecoveryAborted):
                self._logger.debug("Recovery aborted [seq=%d, attempt=%d]: %s", event.sequence_id, event.attempt, event.reason)

            else:
                self._logger.debug("Planner event [seq=%d, type=%s]", event.sequence_id, type(event).__name__)
