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
        3. Validation Errors: Schema, syntax, disallowed actions, or DAG cycle errors.
        4. Provider Errors: Network down, quota, HTTP 5xx, or provider connectivity issues.
        5. Tool Errors: Skill/tool execution errors, missing files, or CLI commands failing.
        6. Execution Errors: Generic runtime exceptions.
        7. Fallback: UNKNOWN.

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

        # 3. Validation
        if cls._VALIDATION_PATTERNS.search(clean_err):
            return FailureCategory.VALIDATION_FAILURE

        # 4. Provider
        if cls._PROVIDER_PATTERNS.search(clean_err):
            return FailureCategory.PROVIDER_ERROR

        # 5. Tool
        if cls._TOOL_PATTERNS.search(clean_err):
            return FailureCategory.TOOL_FAILURE

        # 6. General execution error
        if cls._EXECUTION_PATTERNS.search(clean_err):
            return FailureCategory.EXECUTION_ERROR

        # 7. Fallback
        return FailureCategory.UNKNOWN
