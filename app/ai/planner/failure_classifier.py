"""Failure Classification for J.A.R.V.I.S Planner.

Classifies task failure causes into structured FailureCategory enumerations
to aid adaptive replanning heuristics and observability.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from app.ai.planner.memory import FailureCategory
from app.ai.planner.models import Task


class FailureClassifier:
    """Classifies task execution failures into structured categories."""

    _TIMEOUT_PATTERNS = re.compile(r"\b(time(?:d)?\s*out|timeout|deadline\s*exceeded)\b", re.IGNORECASE)
    _VALIDATION_PATTERNS = re.compile(
        r"\b(schema|validation|invalid\s*action|unknown\s*action|disallowed|missing\s*dependency|self\s*dependency|cycle|circular)\b",
        re.IGNORECASE,
    )
    _PROVIDER_PATTERNS = re.compile(
        r"\b(rate\s*limit|quota|provider|model\s*error|connection\s*refused|network\s*down|50[0-4]|http\s*error|api\s*key)\b",
        re.IGNORECASE,
    )
    _VISUAL_VERIFICATION_PATTERNS = re.compile(
        r"(?:visual\s+verification\s+failed|visual\s+verification|visual\s+goal\s+(?:not_verified|blocked|uncertain)|"
        r"verification_failed|verification\s+failed|verification_uncertain|verification\s+uncertain|"
        r"not_verified|expected\s+visual\s+goal)",
        re.IGNORECASE,
    )
    _VISUAL_TOCTOU_PATTERNS = re.compile(
        r"(?:preflight_stale_coordinates|preflight_window_mismatch|preflight_modal_changed|"
        r"stale_coordinates|window_mismatch|modal_changed|geometry\s+shifted|window\s+switched|"
        r"foreground\s+window\s+switched|coordinates\s+are\s+stale|target\s+window\s+changed|modal\s+appeared)",
        re.IGNORECASE,
    )
    _VISUAL_PRECONDITION_PATTERNS = re.compile(
        r"(?:precondition_failed|precondition\s+failed|grounding_failed|grounding\s+failed|visual\s+grounding|"
        r"security_blocked|security\s+blocked|disabled\s+control|control.*(?:disabled|inactive)|"
        r"blocked_by_modal|sensitive_protected|target\s+not\s+found|element\s+not\s+found|outside\s+control\s+bounds)",
        re.IGNORECASE,
    )
    _TOOL_PATTERNS = re.compile(
        r"\b(tool|skill|handler|app\s*not\s*found|file\s*not\s*found|command\s*failed|permission\s*denied|no\s*such\s*file)\b",
        re.IGNORECASE,
    )
    _EXECUTION_PATTERNS = re.compile(
        r"\b(error|exception|runtimeerror|failed|failure|crash)\b",
        re.IGNORECASE,
    )

    @classmethod
    def classify(
        cls,
        task: Task,
        error: Optional[str] = None,
        dependency_failures: Optional[Dict[str, List[str]]] = None,
    ) -> FailureCategory:
        """Classify a task failure into a structured FailureCategory.

        Resolution Priority:
        1. Dependency Failures: Task ID is recorded in dependency_failures mapping.
        2. Timeout Errors: Timeout patterns in error message.
        3. Visual Failures: Visual TOCTOU, verification, or precondition failures.
        4. Validation Errors: Schema, syntax, disallowed actions, or DAG cycle errors.
        5. Provider Errors: Network down, quota, HTTP 5xx, or provider connectivity issues.
        6. Tool Errors: Skill/tool execution errors, missing files, or CLI commands failing.
        7. Execution Errors: Generic runtime exceptions.
        8. Fallback: UNKNOWN.

        Args:
            task: The Task that failed or was skipped.
            error: Optional error message string.
            dependency_failures: Optional mapping of skipped tasks to failed dependency IDs.

        Returns:
            The resolved FailureCategory enum.
        """
        # 1. Dependency failure check
        if dependency_failures and task.id in dependency_failures and dependency_failures[task.id]:
            return FailureCategory.DEPENDENCY_FAILURE

        if not error or not error.strip():
            return FailureCategory.UNKNOWN

        clean_err = error.strip()

        # 2. Timeout
        if cls._TIMEOUT_PATTERNS.search(clean_err):
            return FailureCategory.TIMEOUT

        # 3. Visual failure categories (prioritized before generic execution/tool errors)
        if cls._VISUAL_TOCTOU_PATTERNS.search(clean_err):
            return FailureCategory.VISUAL_TOCTOU_FAILURE

        if cls._VISUAL_VERIFICATION_PATTERNS.search(clean_err):
            return FailureCategory.VISUAL_VERIFICATION_FAILURE

        if cls._VISUAL_PRECONDITION_PATTERNS.search(clean_err):
            return FailureCategory.VISUAL_PRECONDITION_FAILURE

        # 4. Validation
        if cls._VALIDATION_PATTERNS.search(clean_err):
            return FailureCategory.VALIDATION_FAILURE

        # 5. Provider
        if cls._PROVIDER_PATTERNS.search(clean_err):
            return FailureCategory.PROVIDER_ERROR

        # 6. Tool
        if cls._TOOL_PATTERNS.search(clean_err):
            return FailureCategory.TOOL_FAILURE

        # 7. General execution error
        if cls._EXECUTION_PATTERNS.search(clean_err):
            return FailureCategory.EXECUTION_ERROR

        # 8. Fallback
        return FailureCategory.UNKNOWN
