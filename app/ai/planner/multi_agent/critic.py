"""Reflection and Self-Critique Pipeline for Phase 18.0.

Provides automated pre-execution plan validation, post-execution artifact evaluation,
and bounded iterative refinement loops.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from app.ai.planner.events import AgentCritiqueSubmitted, PlannerEventBus
from app.ai.planner.models import Plan, Task
from app.ai.planner.multi_agent.base import BaseAgent, CriticAgent
from app.ai.planner.multi_agent.models import CriticFeedback, DelegationRequest

logger = logging.getLogger("app.ai.planner.multi_agent.critic")


class ReflectionPipeline:
    """Manages pre-execution and post-execution reflection, critique, and refinement loops."""

    def __init__(
        self,
        critic_agent: Optional[CriticAgent] = None,
        event_bus: Optional[PlannerEventBus] = None,
        max_reflection_rounds: int = 2,
    ) -> None:
        """Initialize reflection pipeline.

        Args:
            critic_agent: Optional specialized CriticAgent instance.
            event_bus: Optional PlannerEventBus for critique events.
            max_reflection_rounds: Maximum retry iterations for refinement.
        """
        self.critic = critic_agent or CriticAgent(event_bus=event_bus)
        self._event_bus = event_bus
        self.max_reflection_rounds = max_reflection_rounds

    def critique_plan(self, plan: Plan, context: Optional[Dict[str, Any]] = None) -> CriticFeedback:
        """Evaluate plan quality, dependency consistency, and safety before execution."""
        ctx = context or {}
        task_id = "plan_pre_critique"
        suggestions: List[str] = []
        passed = True
        score = 1.0

        if not plan.tasks:
            passed = False
            score = 0.0
            critique = "Plan contains zero tasks."
            suggestions.append("Decompose query into at least one executable task.")
            fb = CriticFeedback(task_id=task_id, critic_id=self.critic.agent_id, passed=passed, score=score, critique=critique, suggestions=suggestions)
            self._emit_event(plan.id, fb)
            return fb

        # Validate task IDs and dependencies
        task_ids = {t.id for t in plan.tasks}
        if len(task_ids) < len(plan.tasks):
            passed = False
            score -= 0.4
            suggestions.append("Duplicate task IDs detected in plan definition.")

        for t in plan.tasks:
            if not t.action or not t.action.strip():
                passed = False
                score -= 0.3
                suggestions.append(f"Task '{t.id}' missing action identifier.")

            for dep in t.dependencies or []:
                if dep not in task_ids:
                    passed = False
                    score -= 0.3
                    suggestions.append(f"Task '{t.id}' references non-existent dependency '{dep}'.")

        score = max(0.0, score)
        critique = "Plan passed structural verification." if passed else "Plan failed structural verification."

        fb = CriticFeedback(
            task_id=task_id,
            critic_id=self.critic.agent_id,
            passed=passed,
            score=score,
            critique=critique,
            suggestions=suggestions,
        )
        self._emit_event(plan.id, fb)
        return fb

    def critique_task_result(
        self,
        task: Task,
        output: Any,
        plan_id: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> CriticFeedback:
        """Critique an individual task output against quality criteria."""
        ctx = dict(context or {})
        ctx["task_id"] = task.id
        ctx["action"] = task.action

        fb = self.critic.critique(output, context=ctx)
        self._emit_event(plan_id, fb)
        return fb

    def refine_task(
        self,
        task: Task,
        agent: BaseAgent,
        plan_id: str = "",
        context: Optional[Dict[str, Any]] = None,
        max_rounds: Optional[int] = None,
    ) -> Tuple[bool, Any, List[CriticFeedback]]:
        """Execute a task and iteratively refine output if critique detects defects.

        Returns:
            Tuple of (success, final_output, list_of_critiques).
        """
        rounds_limit = max_rounds if max_rounds is not None else self.max_reflection_rounds
        critique_history: List[CriticFeedback] = []
        current_context = dict(context or {})
        current_params = dict(task.parameters or {})

        for attempt in range(rounds_limit + 1):
            req = DelegationRequest(
                task_id=task.id,
                action=task.action,
                target=task.target,
                parameters=current_params,
                context=current_context,
            )

            resp = agent.execute_delegation(req)
            if not resp.success:
                fb_failed = CriticFeedback(
                    task_id=task.id,
                    critic_id=self.critic.agent_id,
                    passed=False,
                    score=0.0,
                    critique=f"Task execution failed: {resp.error}",
                    suggestions=["Check input parameters or handler availability."],
                )
                critique_history.append(fb_failed)
                self._emit_event(plan_id, fb_failed)
                return False, resp.output, critique_history

            fb = self.critique_task_result(task, resp.output, plan_id=plan_id, context=current_context)
            critique_history.append(fb)

            if fb.passed or attempt >= rounds_limit:
                return fb.passed, resp.output, critique_history

            # Inject critique suggestions into parameters for next attempt
            logger.info("Refining task '%s' (Attempt %d/%d): %s", task.id, attempt + 1, rounds_limit, fb.critique)
            current_params["previous_critique"] = fb.critique
            current_params["critique_suggestions"] = fb.suggestions

        return False, None, critique_history

    def _emit_event(self, plan_id: str, fb: CriticFeedback) -> None:
        """Publish critique event to PlannerEventBus."""
        if self._event_bus is not None:
            self._event_bus.publish(
                AgentCritiqueSubmitted(
                    plan_id=plan_id,
                    task_id=fb.task_id,
                    critic_id=fb.critic_id,
                    passed=fb.passed,
                    feedback=fb.critique,
                    score=fb.score,
                )
            )
