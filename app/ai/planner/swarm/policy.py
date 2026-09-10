"""Swarm Policy Engine for Phase 19.5.

Enforces capability boundaries, recursion depth limits, dangerous command blocking,
and rate-limiting across multi-agent hierarchical swarms.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from app.ai.planner.events import PlannerEventBus, PolicyViolationDetected

logger = logging.getLogger("app.ai.planner.swarm.policy")

# Default dangerous operation patterns
DEFAULT_DENIED_PATTERNS = [
    r"\brm\s+-rf\b",
    r"\bformat\s+[a-zA-Z]:",
    r"\bdel\s+/[sS]\s+/[qQ]\b",
    r"\bshutdown\b",
    r"\bdrop\s+database\b",
    r"\bchmod\s+777\b",
    r"\bkill\s+-9\s+1\b",
]


class SwarmPolicyEngine:
    """Thread-safe policy enforcer validating safety boundaries, depth, and rate limits."""

    def __init__(
        self,
        max_depth: int = 3,
        max_rate_per_minute: int = 120,
        allowed_actions: Optional[Set[str]] = None,
        denied_patterns: Optional[List[str]] = None,
        event_bus: Optional[PlannerEventBus] = None,
    ) -> None:
        """Initialize SwarmPolicyEngine.

        Args:
            max_depth: Maximum recursion depth permitted for sub-swarms.
            max_rate_per_minute: Maximum allowed action invocations per minute.
            allowed_actions: Optional strict allowlist of allowed action names.
            denied_patterns: Regex patterns of dangerous substrings to block.
            event_bus: Optional PlannerEventBus for security alert emission.
        """
        self._lock = threading.RLock()
        self.max_depth = max_depth
        self.max_rate_per_minute = max_rate_per_minute
        self.allowed_actions = set(allowed_actions) if allowed_actions is not None else None
        self._denied_regexes = [
            re.compile(p, re.IGNORECASE) for p in (denied_patterns or DEFAULT_DENIED_PATTERNS)
        ]
        self._event_bus = event_bus

        # Rate-limiting sliding window timestamps
        self._call_timestamps: List[float] = []

    def add_denied_pattern(self, pattern: str) -> None:
        """Add a dangerous pattern to the blocklist."""
        with self._lock:
            self._denied_regexes.append(re.compile(pattern, re.IGNORECASE))

    def add_allowed_action(self, action: str) -> None:
        """Add an action to the allowed set."""
        with self._lock:
            if self.allowed_actions is None:
                self.allowed_actions = set()
            self.allowed_actions.add(action)

    def validate_depth(self, current_depth: int) -> Tuple[bool, Optional[str]]:
        """Validate whether a sub-swarm can spawn at the specified depth."""
        if current_depth > self.max_depth:
            reason = f"Exceeded maximum swarm depth {self.max_depth} (requested: {current_depth})."
            logger.warning("Policy violation: %s", reason)
            return False, reason
        return True, None

    def validate_action(
        self,
        action: str,
        target: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        agent_id: str = "unknown",
    ) -> Tuple[bool, Optional[str]]:
        """Validate whether an action satisfies security boundaries and rate limits.

        Args:
            action: Proposed action name.
            target: Proposed target resource.
            parameters: Action parameters.
            agent_id: Identifier of the invoking agent.

        Returns:
            Tuple of (is_valid, violation_reason).
        """
        with self._lock:
            now = time.time()

            # 1. Check Rate Limit
            self._call_timestamps = [t for t in self._call_timestamps if (now - t) < 60.0]
            if len(self._call_timestamps) >= self.max_rate_per_minute:
                reason = f"Rate limit exceeded ({self.max_rate_per_minute} actions/min)."
                self._emit_violation(action, reason, agent_id)
                return False, reason
            self._call_timestamps.append(now)

            # 2. Strict Allowlist Check
            if self.allowed_actions is not None and action not in self.allowed_actions:
                reason = f"Action '{action}' is not in allowed actions policy."
                self._emit_violation(action, reason, agent_id)
                return False, reason

            # 3. Dangerous Pattern Scan
            text_to_scan = f"{action} {target or ''} {str(parameters or '')}"
            for reg in self._denied_regexes:
                if reg.search(text_to_scan):
                    reason = f"Action violates safety policy pattern '{reg.pattern}'."
                    self._emit_violation(action, reason, agent_id)
                    return False, reason

            return True, None

    def _emit_violation(self, action: str, reason: str, agent_id: str) -> None:
        """Internal helper to log and publish a policy violation event."""
        logger.warning("Policy violation by agent '%s' on '%s': %s", agent_id, action, reason)
        if self._event_bus is not None:
            self._event_bus.publish(
                PolicyViolationDetected(
                    action=action,
                    reason=reason,
                    agent_id=agent_id,
                )
            )
