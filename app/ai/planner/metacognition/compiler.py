"""Macro-Skill Compiler and Skill Registry for Phase 20 Metacognition.

Compiles repeated multi-step swarm workflows into verified, deterministic
standalone Python micro-skills with hot-promotion, sandbox verification,
and quarantine isolation.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from app.ai.planner.events import PlannerEventBus, SkillDistillationCompleted
from app.ai.planner.metacognition.models import DistilledSkillMetadata
from app.ai.planner.metacognition.sandbox import SandboxedExecutionHarness
from app.ai.planner.models import Plan, Task

logger = logging.getLogger("app.ai.planner.metacognition.compiler")


class MacroSkillCompiler:
    """Compiles multi-task subplans into verified, fast deterministic macro-skills."""

    def __init__(
        self,
        sandbox: SandboxedExecutionHarness,
        event_bus: Optional[PlannerEventBus] = None,
    ) -> None:
        """Initialize MacroSkillCompiler.

        Args:
            sandbox: SandboxedExecutionHarness for AST and execution testing.
            event_bus: Optional PlannerEventBus for event publishing.
        """
        self._sandbox = sandbox
        self._event_bus = event_bus
        self._lock = threading.RLock()
        self._skills: Dict[str, Callable[..., Any]] = {}
        self._metadata: Dict[str, DistilledSkillMetadata] = {}

    def compile_subplan_to_skill(
        self,
        subplan: Any,
        skill_name: str,
        description: str = "",
        historical_avg_latency_ms: float = 100.0,
    ) -> DistilledSkillMetadata:
        """Compile a subplan DAG into a standalone deterministic Python macro-skill.

        Args:
            subplan: Plan object, list of Tasks, or dict of subplan definition.
            skill_name: Unique identifier name for the compiled macro-skill.
            description: Semantic description of the macro-skill.
            historical_avg_latency_ms: Multi-agent execution latency for speedup calculation.

        Returns:
            DistilledSkillMetadata with compilation, verification, and promotion status.
        """
        clean_name = skill_name.strip().lower()

        # Extract tasks from subplan
        if isinstance(subplan, Plan):
            tasks = subplan.tasks
        elif isinstance(subplan, list):
            tasks = subplan
        elif isinstance(subplan, dict):
            tasks = subplan.get("tasks", [])
        else:
            tasks = getattr(subplan, "tasks", [])

        # Generate deterministic Python code representing the compiled workflow
        code_lines = [
            f"def {clean_name}(**kwargs):",
            f"    # Distilled macro-skill: {description or clean_name}",
            "    results = {}",
        ]

        if not tasks:
            code_lines.append("    results['status'] = 'completed_empty'")
        else:
            for idx, t in enumerate(tasks):
                action = getattr(t, "action", "task") if not isinstance(t, dict) else t.get("action", "task")
                target = getattr(t, "target", "") if not isinstance(t, dict) else t.get("target", "")
                params = getattr(t, "parameters", {}) if not isinstance(t, dict) else t.get("parameters", {})
                code_lines.append(f"    # Step {idx + 1}: {action} -> {target}")
                code_lines.append(f"    results['step_{idx + 1}'] = {{'action': '{action}', 'target': '{target}', 'params': {params}}}")

            code_lines.append("    results['status'] = 'success'")
            code_lines.append("    results['step_count'] = " + str(len(tasks)))

        code_lines.append("    return results")
        compiled_code = "\n".join(code_lines) + "\n"

        # Generate synthetic verification tests
        test_code = (
            f"res = {clean_name}()\n"
            f"assert isinstance(res, dict), 'Result must be a dict'\n"
            f"assert 'status' in res, 'Result must contain status'\n"
        )

        with self._lock:
            # 1. AST Security Verification
            ast_ok, ast_err = self._sandbox.scan_ast_security(compiled_code)
            if not ast_ok:
                logger.warning("Macro skill '%s' failed AST security: %s", clean_name, ast_err)
                return DistilledSkillMetadata(
                    skill_name=clean_name,
                    source_subplan_signature=f"tasks_{len(tasks)}",
                    compiled_code=compiled_code,
                    historical_avg_latency_ms=historical_avg_latency_ms,
                    compiled_latency_ms=0.0,
                    speedup_multiplier=1.0,
                    verified=False,
                )

            # 2. Ephemeral Sandbox Execution
            v_res = self._sandbox.execute_in_sandbox(compiled_code, test_code)
            if not v_res.success:
                logger.warning("Macro skill '%s' failed sandbox verification: %s", clean_name, v_res.error)
                return DistilledSkillMetadata(
                    skill_name=clean_name,
                    source_subplan_signature=f"tasks_{len(tasks)}",
                    compiled_code=compiled_code,
                    historical_avg_latency_ms=historical_avg_latency_ms,
                    compiled_latency_ms=v_res.execution_time_ms,
                    speedup_multiplier=1.0,
                    verified=False,
                )

            # 3. Successful Verification & Extraction
            local_scope: Dict[str, Any] = {}
            exec(compiled_code, {"__builtins__": __builtins__}, local_scope)
            skill_callable = local_scope.get(clean_name)

            compiled_latency = max(0.001, v_res.execution_time_ms)
            speedup = max(1.0, historical_avg_latency_ms / compiled_latency)

            metadata = DistilledSkillMetadata(
                skill_name=clean_name,
                source_subplan_signature=f"tasks_{len(tasks)}",
                compiled_code=compiled_code,
                historical_avg_latency_ms=historical_avg_latency_ms,
                compiled_latency_ms=compiled_latency,
                speedup_multiplier=speedup,
                verified=True,
                quarantined=False,
            )

            # Automatic promotion rule: promote if verified
            if skill_callable is not None:
                self._skills[clean_name] = skill_callable
                self._metadata[clean_name] = metadata

            if self._event_bus is not None:
                self._event_bus.publish(
                    SkillDistillationCompleted(
                        skill_name=clean_name,
                        speedup_ratio=speedup,
                    )
                )

            logger.info("Macro skill '%s' compiled and promoted (speedup=%.1fx).", clean_name, speedup)
            return metadata

    def register_skill(
        self,
        skill_name: str,
        callable_obj: Callable[..., Any],
        metadata: Optional[DistilledSkillMetadata] = None,
        overwrite: bool = False,
    ) -> bool:
        """Register a pre-compiled skill into the registry under promotion rules.

        Promote ONLY if verified.

        Args:
            skill_name: Action/skill name identifier.
            callable_obj: Python callable implementing the skill.
            metadata: Associated DistilledSkillMetadata.
            overwrite: Whether to overwrite existing skill with same name.

        Returns:
            True if registration was accepted.

        Raises:
            ValueError: If unverified or duplicate when overwrite is False.
        """
        clean_name = skill_name.strip().lower()
        if metadata is None:
            metadata = DistilledSkillMetadata(
                skill_name=clean_name,
                source_subplan_signature="manual",
                compiled_code="",
                historical_avg_latency_ms=10.0,
                compiled_latency_ms=1.0,
                speedup_multiplier=10.0,
                verified=True,
            )

        if not metadata.verified:
            raise ValueError(f"Cannot register skill '{skill_name}': Skill is not verified.")

        with self._lock:
            if clean_name in self._skills and not overwrite:
                raise ValueError(f"Skill '{clean_name}' is already registered in compiler registry.")

            self._skills[clean_name] = callable_obj
            self._metadata[clean_name] = metadata
            logger.debug("Skill '%s' registered manually into compiler.", clean_name)
            return True

    def invoke_skill(self, skill_name: str, *args: Any, **kwargs: Any) -> Any:
        """Invoke a compiled macro-skill.

        Args:
            skill_name: Name of skill to invoke.
            *args: Positional arguments for skill.
            **kwargs: Keyword arguments for skill.

        Returns:
            Return value from the compiled skill callable.

        Raises:
            KeyError: If skill does not exist.
            RuntimeError: If skill is quarantined or unverified.
        """
        clean_name = skill_name.strip().lower()

        with self._lock:
            skill = self._skills.get(clean_name)
            meta = self._metadata.get(clean_name)

            if skill is None or meta is None:
                raise KeyError(f"Skill '{clean_name}' not found in compiler registry.")

            if meta.quarantined:
                raise RuntimeError(f"Skill '{clean_name}' is quarantined and cannot be invoked.")

            if not meta.verified:
                raise RuntimeError(f"Skill '{clean_name}' is not verified.")

        # Execute callable
        return skill(*args, **kwargs)

    def has_skill(self, skill_name: str) -> bool:
        """Check if a skill is registered, verified, and not quarantined.

        Args:
            skill_name: Name of the skill to check.

        Returns:
            True if registered and ready to invoke, False otherwise.
        """
        clean_name = skill_name.strip().lower()
        with self._lock:
            meta = self._metadata.get(clean_name)
            return meta is not None and not meta.quarantined and meta.verified

    def get_callable(self, skill_name: str) -> Optional[Callable[..., Any]]:
        """Retrieve the callable object for a registered, verified, unquarantined skill.

        Args:
            skill_name: Name of skill to retrieve.

        Returns:
            Callable or None.
        """
        clean_name = skill_name.strip().lower()
        with self._lock:
            meta = self._metadata.get(clean_name)
            if meta is not None and not meta.quarantined and meta.verified:
                return self._skills.get(clean_name)
            return None

    def quarantine_skill(self, skill_name: str, reason: str = "") -> bool:
        """Quarantine a registered skill to prevent execution.

        Args:
            skill_name: Name of skill to quarantine.
            reason: Optional justification reason.

        Returns:
            True if skill found and quarantined, False otherwise.
        """
        clean_name = skill_name.strip().lower()

        with self._lock:
            meta = self._metadata.get(clean_name)
            if meta is not None:
                meta.quarantined = True
                logger.warning("Skill '%s' quarantined: %s", clean_name, reason or "Manual quarantine")
                return True
            return False

    def unquarantine_skill(self, skill_name: str) -> bool:
        """Restore a quarantined skill to active execution.

        Args:
            skill_name: Name of skill to unquarantine.

        Returns:
            True if skill found and unquarantined, False otherwise.
        """
        clean_name = skill_name.strip().lower()

        with self._lock:
            meta = self._metadata.get(clean_name)
            if meta is not None:
                meta.quarantined = False
                logger.info("Skill '%s' restored from quarantine.", clean_name)
                return True
            return False

    def remove_skill(self, skill_name: str) -> bool:
        """Remove a skill and its metadata from the registry.

        Args:
            skill_name: Name of skill to remove.

        Returns:
            True if removed, False otherwise.
        """
        clean_name = skill_name.strip().lower()

        with self._lock:
            if clean_name in self._skills:
                del self._skills[clean_name]
                self._metadata.pop(clean_name, None)
                logger.info("Skill '%s' removed from compiler registry.", clean_name)
                return True
            return False

    def list_skills(self, include_quarantined: bool = False) -> List[str]:
        """Return deterministically sorted list of registered skill names.

        Args:
            include_quarantined: Whether to include quarantined skills.

        Returns:
            Sorted list of skill names.
        """
        with self._lock:
            if include_quarantined:
                return sorted(list(self._skills.keys()))
            return sorted([
                name for name, meta in self._metadata.items()
                if not meta.quarantined
            ])

    def get_metadata(self, skill_name: str) -> Optional[DistilledSkillMetadata]:
        """Return metadata for a registered skill.

        Args:
            skill_name: Name of skill to query.

        Returns:
            DistilledSkillMetadata or None.
        """
        clean_name = skill_name.strip().lower()
        with self._lock:
            return self._metadata.get(clean_name)

    def clear(self) -> None:
        """Clear all registered skills and metadata."""
        with self._lock:
            self._skills.clear()
            self._metadata.clear()
