"""Metacognition and Dynamic Tool Synthesis module for J.A.R.V.I.S. (Phase 20.0).

Exposes SandboxedExecutionHarness, ToolSynthesizer, DynamicToolRegistry,
SyntheticTestGenerator, and related models.
"""

from app.ai.planner.metacognition.compiler import MacroSkillCompiler
from app.ai.planner.metacognition.controller import MetacognitiveController
from app.ai.planner.metacognition.evolution import (
    SkillEvolutionConfig,
    SkillEvolutionEngine,
    SkillEvaluation,
    SkillMetrics,
    SkillStatus,
    SkillVersionRecord,
)
from app.ai.planner.metacognition.knowledge_graph import SemanticKnowledgeGraph
from app.ai.planner.metacognition.models import (
    CausalDiagnosis,
    DistilledSkillMetadata,
    KnowledgeTriplet,
    SandboxVerificationResult,
    SynthesizedToolResult,
    ToolSynthesisStatus,
)
from app.ai.planner.metacognition.reflection import CausalReflectionEngine
from app.ai.planner.metacognition.registry import DynamicToolRegistry
from app.ai.planner.metacognition.sandbox import SandboxedExecutionHarness
from app.ai.planner.metacognition.synthesizer import (
    SyntheticTestGenerator,
    ToolSynthesizer,
)

__all__ = [
    "ToolSynthesisStatus",
    "SandboxVerificationResult",
    "SynthesizedToolResult",
    "KnowledgeTriplet",
    "CausalDiagnosis",
    "DistilledSkillMetadata",
    "SandboxedExecutionHarness",
    "DynamicToolRegistry",
    "SyntheticTestGenerator",
    "ToolSynthesizer",
    "SemanticKnowledgeGraph",
    "CausalReflectionEngine",
    "MacroSkillCompiler",
    "MetacognitiveController",
    "SkillEvolutionConfig",
    "SkillEvolutionEngine",
    "SkillEvaluation",
    "SkillMetrics",
    "SkillStatus",
    "SkillVersionRecord",
]
