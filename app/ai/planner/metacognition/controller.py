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
    SystemSkillCompleted,
    SystemSkillFailed,
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
        self._unsub_callbacks: List[Callable[[], bool]] = []
        self._recorded_executions: Dict[str, float] = {}
        self.max_dedup_cache_size: int = 1000
        self.dedup_ttl_seconds: float = 60.0
        self._setup_event_subscriptions()

    def _setup_event_subscriptions(self) -> None:
        """Subscribe to planner observability events on the injected event bus."""
        if self.event_bus is None:
            return
        if hasattr(self.event_bus, "subscribe") and callable(self.event_bus.subscribe):
            try:
                unsub_c = self.event_bus.subscribe(
                    SystemSkillCompleted, self._on_system_skill_completed
                )
                self._unsub_callbacks.append(unsub_c)
            except Exception as exc:
                logger.debug("Could not subscribe to SystemSkillCompleted: %s", exc)

            try:
                unsub_f = self.event_bus.subscribe(
                    SystemSkillFailed, self._on_system_skill_failed
                )
                self._unsub_callbacks.append(unsub_f)
            except Exception as exc:
                logger.debug("Could not subscribe to SystemSkillFailed: %s", exc)

    def _get_event_dedup_keys(
        self, event: Union[SystemSkillCompleted, SystemSkillFailed]
    ) -> List[str]:
        """Generate deduplication keys for a system skill event using unique correlation IDs."""
        keys: List[str] = []
        exec_id = getattr(event, "execution_id", None)
        if exec_id:
            keys.append(f"exec:{exec_id}")
            if event.skill_name:
                keys.append(f"exec:{exec_id}:{event.skill_name}")
            if event.operation:
                keys.append(f"exec:{exec_id}:{event.operation}")

        meta = getattr(event, "metadata", None)
        if isinstance(meta, dict):
            task_id = meta.get("task_id")
            plan_id = meta.get("plan_id")
            traj_id = meta.get("trajectory_id")

            if task_id:
                keys.append(f"task:{task_id}")
                if event.skill_name:
                    keys.append(f"task:{task_id}:{event.skill_name}")
                if event.operation:
                    keys.append(f"task:{task_id}:{event.operation}")
                if plan_id:
                    keys.append(f"plan:{plan_id}:task:{task_id}")

            if traj_id:
                if task_id:
                    keys.append(f"traj:{traj_id}:task:{task_id}")
                else:
                    keys.append(f"exec:{traj_id}")
                    if event.skill_name:
                        keys.append(f"exec:{traj_id}:{event.skill_name}")

        return keys

    def _prune_and_check_dedup(
        self, keys: List[str], now: Optional[float] = None
    ) -> bool:
        """Prune expired telemetry entries, enforce cache bounds, and check if any key was processed.

        Returns True if ANY key is already recognized as recorded (duplicate).
        Returns False if not duplicate (and records all keys).
        If keys is empty, returns False (fail-open telemetry: do not suppress).
        """
        if now is None:
            now = time.time()

        # 1. Prune expired entries
        expired = [
            k for k, ts in self._recorded_executions.items()
            if now - ts > self.dedup_ttl_seconds
        ]
        for k in expired:
            del self._recorded_executions[k]

        # 2. Fail open if no correlation keys exist (never suppress without IDs)
        if not keys:
            return False

        # 3. Check for duplicates
        if any(k in self._recorded_executions for k in keys):
            return True

        # 4. Enforce cache size bound before inserting new keys
        excess = len(self._recorded_executions) + len(keys) - self.max_dedup_cache_size
        if excess > 0:
            sorted_keys = sorted(
                self._recorded_executions.keys(),
                key=lambda k: self._recorded_executions[k],
            )
            for k in sorted_keys[:excess]:
                del self._recorded_executions[k]

        # 5. Record new keys
        for k in keys:
            self._recorded_executions[k] = now

        return False

    def _on_system_skill_completed(self, event: SystemSkillCompleted) -> None:
        """Handle SystemSkillCompleted event from PlannerEventBus."""
        now = time.time()
        dedup_keys = self._get_event_dedup_keys(event)

        with self._lock:
            if self._prune_and_check_dedup(dedup_keys, now):
                logger.debug(
                    "Skipping duplicate SystemSkillCompleted telemetry for '%s.%s'",
                    event.skill_name,
                    event.operation,
                )
                return

        latency_ms = max(0.0, float(getattr(event, "duration", 0.0) or 0.0) * 1000.0)

        # 1. Update SkillEvolutionEngine
        if self.evolution_engine is not None:
            self.evolution_engine.record_execution(
                skill_name=event.skill_name,
                success=True,
                latency_ms=latency_ms,
            )

        # 2. Update SemanticKnowledgeGraph
        if self.knowledge_graph is not None:
            self.knowledge_graph.add_relation(
                event.skill_name, "executed_operation", event.operation, confidence=1.0
            )

        # 3. Publish KnowledgeGraphUpdated if event bus exists
        if self.event_bus is not None:
            try:
                self.event_bus.publish(
                    KnowledgeGraphUpdated(
                        subject=event.skill_name,
                        predicate="executed_operation",
                        target=event.operation,
                        confidence=1.0,
                    )
                )
            except Exception:
                pass

    def _on_system_skill_failed(self, event: SystemSkillFailed) -> None:
        """Handle SystemSkillFailed event from PlannerEventBus."""
        now = time.time()
        dedup_keys = self._get_event_dedup_keys(event)

        with self._lock:
            if self._prune_and_check_dedup(dedup_keys, now):
                logger.debug(
                    "Skipping duplicate SystemSkillFailed telemetry for '%s.%s'",
                    event.skill_name,
                    event.operation,
                )
                return

        latency_ms = max(0.0, float(getattr(event, "duration", 0.0) or 0.0) * 1000.0)

        # 1. Update SkillEvolutionEngine
        if self.evolution_engine is not None:
            self.evolution_engine.record_execution(
                skill_name=event.skill_name,
                success=False,
                latency_ms=latency_ms,
                error=event.error or "",
            )

        # 2. Update SemanticKnowledgeGraph
        if self.knowledge_graph is not None:
            self.knowledge_graph.add_relation(
                event.skill_name, "failed_operation", event.operation, confidence=1.0
            )

        # 3. Publish KnowledgeGraphUpdated if event bus exists
        if self.event_bus is not None:
            try:
                self.event_bus.publish(
                    KnowledgeGraphUpdated(
                        subject=event.skill_name,
                        predicate="failed_operation",
                        target=event.operation,
                        confidence=1.0,
                    )
                )
            except Exception:
                pass

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

                now = time.time()
                for s_name in [norm_goal, direct_goal]:
                    if s_name and self.evolution_engine.get_metrics(s_name) is not None:
                        goal_keys: List[str] = []
                        if traj_id:
                            goal_keys.append(f"traj:{traj_id}:{s_name}")
                            goal_keys.append(f"exec:{traj_id}:{s_name}")

                        with self._lock:
                            is_goal_dup = self._prune_and_check_dedup(goal_keys, now)

                        if is_goal_dup:
                            continue

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
                    t_id = getattr(t, "id", t.get("id") if isinstance(t, dict) else None)
                    t_skill = getattr(t, "skill_name", t.get("skill_name") if isinstance(t, dict) else None)
                    t_exec = getattr(t, "execution_id", t.get("execution_id") if isinstance(t, dict) else None)
                    plan_id = getattr(t, "plan_id", t.get("plan_id") if isinstance(t, dict) else None)

                    task_keys: List[str] = []
                    if t_id:
                        task_keys.append(f"task:{t_id}")
                        if t_skill:
                            task_keys.append(f"task:{t_id}:{t_skill}")
                        task_keys.append(f"task:{t_id}:{clean_act}")
                        if plan_id:
                            task_keys.append(f"plan:{plan_id}:task:{t_id}")
                    if t_exec:
                        task_keys.append(f"exec:{t_exec}")
                        if t_skill:
                            task_keys.append(f"exec:{t_exec}:{t_skill}")
                        task_keys.append(f"exec:{t_exec}:{clean_act}")
                    if traj_id and t_id:
                        task_keys.append(f"traj:{traj_id}:task:{t_id}")

                    with self._lock:
                        is_task_dup = self._prune_and_check_dedup(task_keys, now)

                    if is_task_dup:
                        logger.debug("Skipping duplicate trajectory telemetry for task '%s'", act)
                        continue

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
        """Shutdown background worker pool and unsubscribe event listeners cleanly."""
        with self._lock:
            for unsub in self._unsub_callbacks:
                try:
                    unsub()
                except Exception:
                    pass
            self._unsub_callbacks.clear()
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
