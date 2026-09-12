"""Master Metacognitive Controller for Phase 20.5.

Coordinates Dynamic Tool Synthesis, Causal Reflection, Macro-Skill Compilation,
and Semantic Knowledge Graph into the J.A.R.V.I.S. planning and execution pipeline
with zero global state, complete dependency injection, and thread safety.
"""

from __future__ import annotations

import concurrent.futures
import logging
import re
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from app.ai.planner.events import (
    KnowledgeGraphUpdated,
    PlannerEventBus,
)
from app.ai.planner.metacognition.compiler import MacroSkillCompiler
from app.ai.planner.metacognition.evolution import (
    SkillEvolutionEngine,
    SkillStatus,
)
from app.ai.planner.metacognition.knowledge_graph import SemanticKnowledgeGraph
from app.ai.planner.metacognition.models import (
    DistilledSkillMetadata,
    ToolSynthesisStatus,
)
from app.ai.planner.metacognition.reflection import CausalReflectionEngine
from app.ai.planner.metacognition.synthesizer import ToolSynthesizer
from app.ai.planner.swarm.episodic import EpisodicMemoryStore, TrajectoryRecord

logger = logging.getLogger("app.ai.planner.metacognition.controller")


class MetacognitiveController:
    """Master controller orchestrating tool synthesis, causal reflection, skill compilation, and knowledge graph."""

    def __init__(
        self,
        tool_synthesizer: ToolSynthesizer,
        reflection_engine: CausalReflectionEngine,
        skill_compiler: MacroSkillCompiler,
        knowledge_graph: SemanticKnowledgeGraph,
        event_bus: Optional[PlannerEventBus] = None,
        episodic_memory: Optional[EpisodicMemoryStore] = None,
        evolution_engine: Optional[SkillEvolutionEngine] = None,
        executor_handlers: Optional[Dict[str, Callable[..., Any]]] = None,
        max_workers: int = 2,
    ) -> None:
        """Initialize MetacognitiveController with full dependency injection.

        Args:
            tool_synthesizer: ToolSynthesizer instance for dynamic runtime code synthesis.
            reflection_engine: CausalReflectionEngine instance for failure diagnosis and invariant discovery.
            skill_compiler: MacroSkillCompiler instance for compiling repeated subplans into skills.
            knowledge_graph: SemanticKnowledgeGraph instance for multi-entity knowledge storage.
            event_bus: Optional PlannerEventBus for event broadcasting.
            episodic_memory: Optional EpisodicMemoryStore for linking completed trajectories.
            evolution_engine: Optional SkillEvolutionEngine for autonomous skill evolution and tracking.
            executor_handlers: Optional pre-registered executor action handlers.
            max_workers: Number of background worker threads for non-blocking reflection.
        """
        self.tool_synthesizer = tool_synthesizer
        self.reflection_engine = reflection_engine
        self.skill_compiler = skill_compiler
        self.knowledge_graph = knowledge_graph
        self.event_bus = event_bus
        self.episodic_memory = episodic_memory
        self.evolution_engine = evolution_engine
        self._executor_handlers = dict(executor_handlers or {})
        self._lock = threading.RLock()
        self._bg_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, max_workers),
            thread_name_prefix="MetacognitiveControllerWorker",
        )

    def on_trajectory_completed(
        self,
        trajectory: Union[TrajectoryRecord, Dict[str, Any], Any],
        auto_distill: bool = True,
    ) -> Optional[DistilledSkillMetadata]:
        """Analyze a completed trajectory, update knowledge graph, run reflection, and distill skills.

        Args:
            trajectory: Completed TrajectoryRecord or dictionary containing goal, outcomes, tasks, etc.
            auto_distill: Whether to compile successful subplans into reusable macro skills.

        Returns:
            DistilledSkillMetadata if a macro skill was successfully compiled, else None.
        """
        with self._lock:
            # 1. Normalize trajectory data
            goal, success, error, tasks, outputs, subplan, traj_id = self._extract_trajectory_fields(trajectory)

            # 2. Run Causal Reflection on Failure or Record Success
            if not success or error:
                traj_dict = {
                    "trajectory_id": traj_id,
                    "goal": goal,
                    "error": error or "Unknown execution failure",
                    "events": [f"failed_task_{t}" for t in tasks[:3]] if tasks else ["failure_signal"],
                }
                diag = self.reflection_engine.diagnose_failure(traj_dict)

                # Update Knowledge Graph with failure causes and discovered invariant
                self.knowledge_graph.add_relation(goal or traj_id, "failed_due_to", diag.root_cause_type, confidence=diag.confidence)
                self.knowledge_graph.add_relation(goal or traj_id, "violates_invariant", diag.invariant_constraint, confidence=0.9)
                self.knowledge_graph.add_relation(diag.root_cause_type, "recommended_fix", diag.recommended_action, confidence=0.9)

                if self.event_bus is not None:
                    self.event_bus.publish(
                        KnowledgeGraphUpdated(
                            subject=goal or traj_id,
                            predicate="failed_due_to",
                            target=diag.root_cause_type,
                            confidence=diag.confidence,
                        )
                    )
            else:
                # Update Knowledge Graph with success, tasks, and artifacts
                self.knowledge_graph.add_relation("system", "completed_goal", goal, confidence=1.0)
                if self.event_bus is not None:
                    self.event_bus.publish(
                        KnowledgeGraphUpdated(
                            subject="system",
                            predicate="completed_goal",
                            target=goal,
                            confidence=1.0,
                        )
                    )

                for t in tasks:
                    act = getattr(t, "action", t.get("action", "") if isinstance(t, dict) else str(t))
                    if act:
                        self.knowledge_graph.add_relation(goal, "contains_task", str(act), confidence=1.0)

                for k, v in outputs.items():
                    val_summary = str(v)[:60]
                    self.knowledge_graph.add_relation(goal, "produced_artifact", f"{k}:{val_summary}", confidence=1.0)

            # 3. Optional Episodic Memory Link
            if self.episodic_memory is not None:
                if isinstance(trajectory, TrajectoryRecord):
                    self.episodic_memory.record_trajectory(trajectory)
                elif isinstance(trajectory, dict):
                    rec = TrajectoryRecord(
                        goal=goal,
                        plan_signature=trajectory.get("plan_signature", "swarm_trajectory"),
                        success=success,
                        total_duration_ms=trajectory.get("duration_ms", 0.0),
                    )
                    self.episodic_memory.record_trajectory(rec)

            # 4. Trigger Macro-Skill Distillation if Successful and Requested
            distilled_meta: Optional[DistilledSkillMetadata] = None
            if auto_distill and success:
                compile_target = subplan if subplan is not None else tasks
                if compile_target:
                    clean_skill_name = self._normalize_skill_name(goal)
                    try:
                        distilled_meta = self.skill_compiler.compile_subplan_to_skill(
                            subplan=compile_target,
                            skill_name=clean_skill_name,
                            description=f"Distilled macro skill for '{goal}'",
                        )
                        if distilled_meta and distilled_meta.verified and not distilled_meta.quarantined:
                            self.knowledge_graph.add_relation(goal, "distilled_as_macro_skill", clean_skill_name, confidence=1.0)
                            if self.event_bus is not None:
                                self.event_bus.publish(
                                    KnowledgeGraphUpdated(
                                        subject=goal,
                                        predicate="distilled_as_macro_skill",
                                        target=clean_skill_name,
                                        confidence=1.0,
                                    )
                                )
                            if self.evolution_engine is not None:
                                skill_callable = self.skill_compiler.get_callable(clean_skill_name)
                                self.evolution_engine.register_skill(
                                    skill_name=clean_skill_name,
                                    callable_tool=skill_callable,
                                    metadata=distilled_meta,
                                    version="v1.0.0",
                                    is_stable=True,
                                    overwrite=True,
                                )
                    except Exception as exc:
                        logger.warning("Macro-skill compilation bypassed for goal '%s': %s", goal, exc)

            # 5. Performance Feedback & Skill Evolution Updates
            if self.evolution_engine is not None:
                norm_goal = self._normalize_skill_name(goal)
                direct_goal = goal.strip().lower()
                dur = 10.0
                if isinstance(trajectory, dict):
                    dur = float(trajectory.get("duration_ms", 10.0))
                elif hasattr(trajectory, "total_duration_ms"):
                    dur = float(getattr(trajectory, "total_duration_ms", 10.0))

                for s_name in [norm_goal, direct_goal]:
                    if self.evolution_engine.get_metrics(s_name) is not None:
                        self.evolution_engine.record_execution(
                            skill_name=s_name,
                            success=success,
                            latency_ms=dur,
                            error=error or "",
                        )
                        status = self.evolution_engine.get_skill_status(s_name)
                        if status in (SkillStatus.DEGRADED, SkillStatus.QUARANTINED):
                            self._bg_executor.submit(self.evolution_engine.improve_skill, s_name)

                for t in tasks:
                    act = getattr(t, "action", t.get("action", "") if isinstance(t, dict) else str(t))
                    clean_act = str(act).strip().lower()
                    if clean_act and self.evolution_engine.get_metrics(clean_act) is not None:
                        self.evolution_engine.record_execution(
                            skill_name=clean_act,
                            success=success,
                            latency_ms=max(1.0, dur / max(1, len(tasks))),
                            error=error or "",
                        )

            return distilled_meta

    def on_trajectory_completed_async(
        self,
        trajectory: Union[TrajectoryRecord, Dict[str, Any], Any],
        auto_distill: bool = True,
    ) -> concurrent.futures.Future[Optional[DistilledSkillMetadata]]:
        """Asynchronously process completed trajectory in the background without blocking user threads."""
        return self._bg_executor.submit(self.on_trajectory_completed, trajectory, auto_distill)

    def resolve_or_synthesize_action(
        self,
        action_name: str,
        context: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
    ) -> Optional[Callable[..., Any]]:
        """Resolve an action from executor handlers, dynamic registry, or synthesize on demand.

        Workflow:
        1. Check executor handlers
        2. Check dynamic tool registry
        3. Check macro skill compiler
        4. Invoke ToolSynthesizer if missing
        5. Return verified handler or None on failure (Never throws).

        Args:
            action_name: Target action identifier.
            context: Contextual parameters including description, schemas, handlers.
            agent_id: Optional requesting agent identifier.

        Returns:
            Callable action handler if resolved or synthesized, else None.
        """
        if not action_name:
            return None

        clean_name = action_name.strip().lower()
        ctx = dict(context or {})

        # Check Evolution Engine exclusion
        if self.evolution_engine is not None:
            status = self.evolution_engine.get_skill_status(clean_name)
            if status in (SkillStatus.QUARANTINED, SkillStatus.RETIRED):
                logger.warning(
                    "Action '%s' is %s in evolution engine; skipping resolution/synthesis.",
                    clean_name,
                    status.value,
                )
                return None
            elif status == SkillStatus.DEGRADED:
                logger.warning("Action '%s' is in DEGRADED state in evolution engine.", clean_name)

        # Step 1: Check executor handlers (context-provided or pre-registered)
        ctx_handlers = ctx.get("handlers", {})
        if clean_name in ctx_handlers:
            return ctx_handlers[clean_name]
        with self._lock:
            if clean_name in self._executor_handlers:
                return self._executor_handlers[clean_name]

        # Step 2: Check dynamic tool registry
        if self.tool_synthesizer.registry is not None:
            tool = self.tool_synthesizer.registry.resolve(clean_name)
            if tool is not None:
                return tool

        # Step 3: Check macro skill compiler
        if self.skill_compiler.has_skill(clean_name):
            if self.evolution_engine is not None and not self.evolution_engine.is_skill_selectable(clean_name):
                logger.warning(
                    "Macro skill '%s' is not selectable in evolution engine (status=%s); skipping.",
                    clean_name,
                    self.evolution_engine.get_skill_status(clean_name),
                )
            else:
                macro_tool = self.skill_compiler.get_callable(clean_name)
                if macro_tool is not None:
                    return macro_tool

        # Step 4: Invoke ToolSynthesizer if missing
        description = ctx.get("description", f"Autonomous tool for {clean_name}")
        input_schema = ctx.get("input_schema", {"type": "object"})
        output_schema = ctx.get("output_schema", {"type": "object"})
        max_attempts = ctx.get("max_attempts", 3)

        try:
            synth_res = self.tool_synthesizer.synthesize_tool(
                action=clean_name,
                description=description,
                input_schema=input_schema,
                output_schema=output_schema,
                max_attempts=max_attempts,
            )
            if synth_res.status == ToolSynthesisStatus.VERIFIED_AND_PROMOTED and synth_res.callable_tool is not None:
                with self._lock:
                    self.knowledge_graph.add_relation("system", "has_synthesized_tool", clean_name, confidence=1.0)
                if self.event_bus is not None:
                    self.event_bus.publish(
                        KnowledgeGraphUpdated(
                            subject="system",
                            predicate="has_synthesized_tool",
                            target=clean_name,
                            confidence=1.0,
                        )
                    )
                return synth_res.callable_tool
            else:
                logger.warning("Tool synthesis for '%s' failed verification: %s", clean_name, synth_res.status)
                return None
        except Exception as exc:
            logger.warning("Dynamic tool synthesis encountered error for '%s': %s", clean_name, exc)
            return None

    def get_prescriptive_guidance(self, goal: str) -> List[str]:
        """Combine causal reflection recommendations with knowledge graph historical rules.

        Args:
            goal: Human-readable target goal or query.

        Returns:
            Deduplicated, deterministically sorted list of prescriptive advice strings.
        """
        prescriptions: Set[str] = set()

        # 1. Ask CausalReflectionEngine for pattern-based prescriptions
        engine_recs = self.reflection_engine.generate_prescriptions(goal)
        prescriptions.update(engine_recs)

        # 2. Query SemanticKnowledgeGraph for goal-specific historical rules & constraints
        with self._lock:
            triplets = self.knowledge_graph.query(subject=goal)
            for t in triplets:
                if t.predicate == "failed_due_to":
                    fixes = self.knowledge_graph.query(subject=t.object_, predicate="recommended_fix")
                    for fix in fixes:
                        prescriptions.add(f"Past failure '{t.object_}' fix: {fix.object_}")
                elif t.predicate == "violates_invariant":
                    prescriptions.add(f"Invariant: {t.object_}")

            # Also pull universal invariants recorded in graph
            invariants = self.knowledge_graph.query(predicate="violates_invariant")
            for inv in invariants:
                prescriptions.add(f"Rule: {inv.object_}")

        return sorted(list(prescriptions))

    def find_matching_macro_skill(self, goal: str) -> Optional[str]:
        """Look up whether an existing verified macro skill matches the goal.

        Args:
            goal: Target goal string.

        Returns:
            Registered macro skill identifier or None.
        """
        if not goal:
            return None

        def _is_selectable(name: str) -> bool:
            if self.evolution_engine is None:
                return True
            status = self.evolution_engine.get_skill_status(name)
            if status is None:
                return True
            if status in (SkillStatus.QUARANTINED, SkillStatus.RETIRED):
                return False
            if status == SkillStatus.DEGRADED:
                logger.warning("Selected macro skill '%s' is in DEGRADED state.", name)
            return True

        # Check knowledge graph link
        with self._lock:
            links = self.knowledge_graph.query(subject=goal, predicate="distilled_as_macro_skill")
            if links:
                skill_name = links[0].object_
                if self.skill_compiler.has_skill(skill_name) and _is_selectable(skill_name):
                    return skill_name

        # Check normalized name
        candidate = self._normalize_skill_name(goal)
        if self.skill_compiler.has_skill(candidate) and _is_selectable(candidate):
            return candidate

        clean_direct = goal.strip().lower()
        if self.skill_compiler.has_skill(clean_direct) and _is_selectable(clean_direct):
            return clean_direct

        return None

    def invoke_macro_skill(self, skill_name: str, **kwargs: Any) -> Any:
        """Invoke a compiled macro skill directly.

        Args:
            skill_name: Name of the macro skill.
            **kwargs: Arguments to forward to compiled callable.

        Returns:
            Callable output.
        """
        if self.evolution_engine is not None and not self.evolution_engine.is_skill_selectable(skill_name):
            status = self.evolution_engine.get_skill_status(skill_name)
            raise RuntimeError(f"Skill '{skill_name}' is not selectable in evolution engine (status={status}).")
        return self.skill_compiler.invoke_skill(skill_name, **kwargs)

    def shutdown(self) -> None:
        """Shutdown background worker pool cleanly."""
        self._bg_executor.shutdown(wait=False)

    def _normalize_skill_name(self, goal: str) -> str:
        """Produce a safe, deterministic snake_case skill identifier from goal string."""
        cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", goal.strip().lower())
        trimmed = re.sub(r"_+", "_", cleaned).strip("_")
        if not trimmed:
            trimmed = "skill_anonymous"
        return f"macro_{trimmed[:30]}"

    def _extract_trajectory_fields(
        self, trajectory: Any
    ) -> Tuple[str, bool, Optional[str], List[Any], Dict[str, Any], Any, str]:
        """Normalize disparate trajectory input formats into standard tuple."""
        if isinstance(trajectory, TrajectoryRecord):
            goal = trajectory.goal
            success = trajectory.success
            error = None
            tasks = []
            outputs = {}
            subplan = None
            traj_id = getattr(trajectory, "trajectory_id", f"traj_{uuid.uuid4().hex[:8]}")
            return goal, success, error, tasks, outputs, subplan, traj_id

        if isinstance(trajectory, dict):
            goal = trajectory.get("goal", "unnamed_goal")
            success = bool(trajectory.get("success", True))
            error = trajectory.get("error")
            tasks = trajectory.get("tasks", [])
            outputs = trajectory.get("outputs", {})
            subplan = trajectory.get("subplan")
            traj_id = trajectory.get("trajectory_id", f"traj_{uuid.uuid4().hex[:8]}")
            return goal, success, error, tasks, outputs, subplan, traj_id

        # Generic object fallback
        goal = getattr(trajectory, "goal", "unnamed_goal")
        success = bool(getattr(trajectory, "success", True))
        error = getattr(trajectory, "error", None)
        tasks = getattr(trajectory, "tasks", [])
        outputs = getattr(trajectory, "outputs", {})
        subplan = getattr(trajectory, "subplan", None)
        traj_id = getattr(trajectory, "trajectory_id", f"traj_{uuid.uuid4().hex[:8]}")
        return goal, success, error, tasks, outputs, subplan, traj_id
