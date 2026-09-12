"""Causal Reflection Engine for Phase 20 Metacognition.

Analyzes plan execution failures, diagnoses root causes, extracts reusable
invariant constraints, and generates actionable prescriptions.
"""

from __future__ import annotations

from collections import Counter
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Set

from app.ai.planner.events import CausalDiagnosisGenerated, PlannerEventBus
from app.ai.planner.metacognition.models import CausalDiagnosis

logger = logging.getLogger("app.ai.planner.metacognition.reflection")


class CausalReflectionEngine:
    """Metacognitive diagnostic engine for causal post-mortem analysis and constraint discovery."""

    def __init__(self, event_bus: Optional[PlannerEventBus] = None) -> None:
        """Initialize causal reflection engine.

        Args:
            event_bus: Optional PlannerEventBus for emitting diagnostic findings.
        """
        self._event_bus = event_bus
        self._lock = threading.RLock()
        self._diagnoses: List[CausalDiagnosis] = []
        self._invariants: Set[str] = set()

    def diagnose_failure(self, trajectory: Any) -> CausalDiagnosis:
        """Analyze a failed or suboptimal trajectory to infer its primary root cause.

        Args:
            trajectory: Execution trajectory record or dictionary with failure telemetry.

        Returns:
            CausalDiagnosis containing root cause, invariant constraint, and prescription.
        """
        with self._lock:
            # Extract trajectory properties
            if isinstance(trajectory, dict):
                traj_id = str(trajectory.get("trajectory_id") or "traj_unknown")
                failed_task_id = str(trajectory.get("failed_task_id") or "task_unknown")
                recovery_events = trajectory.get("recovery_events", [])
                error_msg = str(trajectory.get("error", "")).lower()
                goal = str(trajectory.get("goal", ""))
            else:
                traj_id = str(getattr(trajectory, "trajectory_id", "traj_unknown"))
                failed_task_id = str(getattr(trajectory, "failed_task_id", "task_unknown"))
                recovery_events = getattr(trajectory, "recovery_events", [])
                error_msg = str(getattr(trajectory, "error", "")).lower()
                goal = str(getattr(trajectory, "goal", ""))

            # Consolidate error signals
            reasons = [error_msg]
            if isinstance(recovery_events, list):
                for rev in recovery_events:
                    if isinstance(rev, dict):
                        reasons.append(str(rev.get("reason", "")).lower())
                        reasons.append(str(rev.get("error", "")).lower())
                    else:
                        reasons.append(str(rev).lower())

            combined_signal = " ".join(reasons)

            # Causal inference heuristics
            if any(k in combined_signal for k in ("missing_tool", "no handler", "unhandled action", "no agent available", "handler not found", "cannot resolve tool")):
                root_cause = "MISSING_TOOL"
                desc = "Execution halted due to an unhandled action or missing tool handler in the agent registry."
                invariant = "Always verify tool availability in registry before executing task"
                prescription = "Synthesize or register missing tool before dispatching plan"
                confidence = 0.95

            elif any(k in combined_signal for k in ("timeout", "timed out", "deadline", "timeoutexpired")):
                root_cause = "EXECUTION_TIMEOUT"
                desc = "Task execution exceeded allocated deadline threshold."
                invariant = "Apply exponential backoff and allocate higher timeout budget for heavy operations"
                prescription = "Increase timeout limit or partition task into smaller asynchronous chunks"
                confidence = 0.95

            elif any(k in combined_signal for k in ("consensus", "quorum", "tie broken", "no agreement", "ballot")):
                root_cause = "CONSENSUS_FAILURE"
                desc = "Multi-agent deliberation failed to reach requisite quorum or agreement threshold."
                invariant = "Ensure odd number of voting candidates or use weighted fallback on consensus ties"
                prescription = "Adopt Borda count arbitration or relax quorum threshold"
                confidence = 0.90

            elif any(k in combined_signal for k in ("policy", "security", "forbidden", "permission", "violation", "guardrail", "rm -rf")):
                root_cause = "POLICY_REJECTION"
                desc = "Operation violated system security boundary or capability guardrail."
                invariant = "Never dispatch commands violating system safety policy"
                prescription = "Route command through InterventionGateway for explicit human approval"
                confidence = 0.98

            elif any(k in combined_signal for k in ("precondition", "prerequisite", "missing parameter", "invalid schema", "validation error", "input invalid")):
                root_cause = "PRECONDITION_VIOLATION"
                desc = "Downstream task triggered without satisfying upstream data schema or state prerequisites."
                invariant = "Validate schema and task prerequisites before triggering dependent actions"
                prescription = "Ensure upstream task outputs satisfy downstream input schema constraints"
                confidence = 0.92

            else:
                root_cause = "UNKNOWN"
                desc = f"Unspecified runtime failure detected during plan execution: {error_msg or 'Non-zero error code'}"
                invariant = "Isolate failing execution branch and invoke adaptive replanning"
                prescription = "Capture execution trace and retry with alternate agent"
                confidence = 0.50

            diagnosis = CausalDiagnosis(
                trajectory_id=traj_id,
                failed_task_id=failed_task_id,
                root_cause_type=root_cause,
                description=desc,
                invariant_constraint=invariant,
                recommended_action=prescription,
                confidence=confidence,
                created_at=time.time(),
            )

            self._diagnoses.append(diagnosis)
            self._invariants.add(invariant)

            if self._event_bus is not None:
                self._event_bus.publish(
                    CausalDiagnosisGenerated(
                        trajectory_id=traj_id,
                        root_cause=root_cause,
                        invariant=invariant,
                    )
                )

            logger.info("Causal diagnosis generated for '%s': %s (confidence=%.2f)", traj_id, root_cause, confidence)
            return diagnosis

    def discover_invariants(self, trajectories: List[Any]) -> List[str]:
        """Examine a corpus of trajectory records and discover deduplicated invariants.

        Args:
            trajectories: List of executed trajectory objects or dictionaries.

        Returns:
            Deterministically sorted list of unique invariant constraints.
        """
        discovered: Set[str] = set()

        with self._lock:
            for traj in trajectories:
                # If trajectory failed or has failure signals, diagnose it
                is_success = traj.get("success", True) if isinstance(traj, dict) else getattr(traj, "success", True)
                if not is_success:
                    diag = self.diagnose_failure(traj)
                    discovered.add(diag.invariant_constraint)

            # Also incorporate any previously recorded invariants
            discovered.update(self._invariants)

        return sorted(list(discovered))

    def generate_prescriptions(self, goal: str) -> List[str]:
        """Generate targeted recommendations and constraints for an upcoming goal.

        Args:
            goal: Mission prompt or goal text.

        Returns:
            Deterministically sorted list of actionable recommendations.
        """
        clean_goal = goal.lower()
        prescriptions: Set[str] = set()

        with self._lock:
            for diag in self._diagnoses:
                if diag.root_cause_type == "MISSING_TOOL" and any(k in clean_goal for k in ("search", "convert", "calculate", "scrape", "extract")):
                    prescriptions.add(diag.recommended_action)
                elif diag.root_cause_type == "EXECUTION_TIMEOUT" and any(k in clean_goal for k in ("heavy", "batch", "large", "download", "train")):
                    prescriptions.add(diag.recommended_action)
                elif diag.root_cause_type == "POLICY_REJECTION" and any(k in clean_goal for k in ("delete", "remove", "system", "root", "format")):
                    prescriptions.add(diag.recommended_action)
                elif diag.root_cause_type == "CONSENSUS_FAILURE" and any(k in clean_goal for k in ("vote", "consensus", "decide", "judge")):
                    prescriptions.add(diag.recommended_action)

            # Default fallback prescriptions if no specific matches
            if not prescriptions:
                prescriptions.add("Verify tool availability in registry before executing task")
                prescriptions.add("Validate schema and task prerequisites before triggering dependent actions")

        return sorted(list(prescriptions))

    def export_report(self) -> Dict[str, Any]:
        """Export comprehensive diagnostic report and statistics.

        Returns:
            Dictionary containing diagnosis history, frequency stats, and invariants.
        """
        with self._lock:
            counts = Counter(d.root_cause_type for d in self._diagnoses)
            return {
                "total_diagnoses": len(self._diagnoses),
                "by_root_cause": dict(counts),
                "invariants": sorted(list(self._invariants)),
                "diagnoses": [
                    {
                        "trajectory_id": d.trajectory_id,
                        "root_cause": d.root_cause_type,
                        "invariant": d.invariant_constraint,
                        "recommendation": d.recommended_action,
                        "confidence": d.confidence,
                        "created_at": d.created_at,
                    }
                    for d in self._diagnoses
                ],
            }

    def clear(self) -> None:
        """Clear all stored diagnoses and discovered invariants."""
        with self._lock:
            self._diagnoses.clear()
            self._invariants.clear()
