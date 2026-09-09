"""IntentRouter for J.A.R.V.I.S AI Subsystem.

Provides a deterministic, rule-based decision engine that classifies incoming
user queries into distinct operational intents prior to reaching AIManager or
skills. Does not use AI/LLMs, ensuring low latency, zero API costs, and predictable
deterministic routing.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Final, Optional, Pattern

from app.core.container import ServiceContainer, container
from app.core.logger import get_logger


class IntentType(str, Enum):
    """Supported operational query intent classifications.

    Categories:
        CHAT: Conversational small talk, greetings, bot identity questions.
        REASONING: Deep cognitive questions, conceptual explanations, comparisons, causal inquiries.
        CODING: Software engineering, programming, code generation, debugging, syntax fixes.
        SEARCH: Real-time queries, current events, live scores, weather, stock prices.
        SYSTEM: Local OS operations, window management, audio volume, power management.
        FILE: File system operations, reading documents, summarizing PDFs/spreadsheets.
        MEMORY: User preferences, biographical facts, explicit recall or forget requests.
        TOOL: Deterministic utilities like calculator, unit conversions, translation, shell execution.
        UNKNOWN: Fallback category for queries that do not match deterministic patterns.
    """

    CHAT = "chat"
    REASONING = "reasoning"
    CODING = "coding"
    SEARCH = "search"
    SYSTEM = "system"
    FILE = "file"
    MEMORY = "memory"
    TOOL = "tool"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class IntentResult:
    """Classification outcome representing a deterministic intent decision.

    Attributes:
        intent: The identified IntentType enum member.
        confidence: Deterministic confidence score between 0.0 and 1.0.
        reason: Human-readable rationale or matched pattern explanation.
        metadata: Optional dictionary with matched tokens, entities, or rules.
    """

    intent: IntentType
    confidence: float = 1.0
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Rule:
    """Internal rule specification for intent matching."""

    intent: IntentType
    pattern: Pattern[str]
    confidence: float
    reason: str
    priority: int  # Lower number = evaluated first


class IntentRouter:
    """Thread-safe, deterministic, rule-based query intent classifier.

    Evaluates user input against categorized regex rules and keywords in priority
    order, returning an IntentResult with the matched intent, confidence, and reason.
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        container_instance: Optional[ServiceContainer] = None,
        *,
        auto_register_in_container: bool = True,
    ) -> None:
        """Initialize the IntentRouter.

        Args:
            logger: Optional Logger instance.
            container_instance: Optional ServiceContainer instance.
            auto_register_in_container: If True, self-registers into ServiceContainer.
        """
        self._lock = threading.RLock()
        self._container = container_instance if container_instance is not None else container
        self._logger = logger if logger is not None else get_logger("AI.INTENT_ROUTER")

        self._custom_rules: list[_Rule] = []
        self._rules = self._build_default_rules()

        if auto_register_in_container:
            try:
                self._container.register_singleton("intent_router", self, allow_override=True)
                self._logger.debug("Registered 'intent_router' into Service Container.")
            except Exception as exc:
                self._logger.warning(f"Could not register IntentRouter into container: {exc}")

    def classify(self, query: str) -> IntentResult:
        """Deterministically classify an input query into an IntentType.

        Args:
            query: The raw user message string.

        Returns:
            An IntentResult containing intent, confidence, and matching rationale.
        """
        if not query or not query.strip():
            return IntentResult(
                intent=IntentType.UNKNOWN,
                confidence=0.0,
                reason="Empty or whitespace query",
            )

        cleaned = query.strip()
        normalized = " ".join(cleaned.lower().split())

        with self._lock:
            # 1. Check custom user-registered rules first (ordered by priority)
            for rule in sorted(self._custom_rules, key=lambda r: r.priority):
                match = rule.pattern.search(normalized)
                if match:
                    return IntentResult(
                        intent=rule.intent,
                        confidence=rule.confidence,
                        reason=f"Custom rule matched: {rule.reason}",
                        metadata={"matched": match.group(0)},
                    )

            # 2. Check built-in deterministic rules (ordered by priority)
            for rule in self._rules:
                match = rule.pattern.search(normalized)
                if match:
                    return IntentResult(
                        intent=rule.intent,
                        confidence=rule.confidence,
                        reason=rule.reason,
                        metadata={"matched": match.group(0)},
                    )

        # 3. Fallback for unclassified queries
        return IntentResult(
            intent=IntentType.UNKNOWN,
            confidence=0.0,
            reason="No deterministic classification rule matched",
        )

    def register_rule(
        self,
        intent: IntentType,
        pattern: str | Pattern[str],
        *,
        confidence: float = 1.0,
        reason: str = "",
        priority: int = 50,
    ) -> None:
        """Register a custom deterministic matching rule.

        Args:
            intent: Target IntentType.
            pattern: String regex or compiled pattern.
            confidence: Confidence score for matches (0.0 to 1.0).
            reason: Explanation of the rule.
            priority: Evaluation priority (lower numbers evaluate first, default 50).
        """
        compiled = re.compile(pattern, re.IGNORECASE) if isinstance(pattern, str) else pattern
        rule = _Rule(
            intent=intent,
            pattern=compiled,
            confidence=max(0.0, min(1.0, confidence)),
            reason=reason or f"Pattern: {compiled.pattern}",
            priority=priority,
        )
        with self._lock:
            self._custom_rules.append(rule)
            self._logger.debug(f"Registered custom intent rule for {intent.value}: {rule.reason}")

    def clear_custom_rules(self) -> None:
        """Clear all dynamically registered custom rules."""
        with self._lock:
            self._custom_rules.clear()

    # --------------------------------------------------------------------------
    # Rule Construction
    # --------------------------------------------------------------------------

    @staticmethod
    def _build_default_rules() -> list[_Rule]:
        """Construct deterministic rule definitions ordered by priority."""
        rules: list[_Rule] = [
            # ------------------------------------------------------------------
            # 1. MEMORY INTENTS (Priority 10: Explicit knowledge/preference recall/storage)
            # ------------------------------------------------------------------
            _Rule(
                intent=IntentType.MEMORY,
                pattern=re.compile(
                    r"\b(remember\s+this|remember\s+that|remember\s+to|don't\s+forget|do\s+not\s+forget|"
                    r"forget\s+that|forget\s+this|forget\s+everything|"
                    r"what\s+did\s+i\s+tell\s+you|do\s+you\s+remember|recall\s+my|my\s+name\s+is|"
                    r"save\s+in\s+memory|store\s+in\s+memory)\b",
                    re.IGNORECASE,
                ),
                confidence=1.0,
                reason="Explicit conversational memory or recall instruction",
                priority=10,
            ),
            # ------------------------------------------------------------------
            # 2. FILE INTENTS (Priority 20: Documents, PDFs, local file processing)
            # ------------------------------------------------------------------
            _Rule(
                intent=IntentType.FILE,
                pattern=re.compile(
                    r"\b(summarize\s+(pdf|document|file|doc|sheet|spreadsheet)|"
                    r"open\s+(file|document|pdf|doc|folder|directory)|"
                    r"read\s+(file|document|pdf|doc|contents)|"
                    r"parse\s+(pdf|document|file|csv|excel|spreadsheet)|"
                    r"(export|save)\s+to\s+(pdf|csv|excel|file)|"
                    r"\b\w+\.(pdf|docx?|xlsx?|csv|txt|log|json|md)\b)\b",
                    re.IGNORECASE,
                ),
                confidence=0.95,
                reason="Document or file manipulation request",
                priority=20,
            ),
            # ------------------------------------------------------------------
            # 3. SYSTEM INTENTS (Priority 30: OS operations, window control, volume, power)
            # ------------------------------------------------------------------
            _Rule(
                intent=IntentType.SYSTEM,
                pattern=re.compile(
                    r"\b(open\s+(chrome|notepad|calculator|spotify|edge|firefox|browser|terminal|app|application)|"
                    r"close\s+(chrome|notepad|calculator|spotify|edge|firefox|browser|terminal|window|app)|"
                    r"shutdown(\s+pc|\s+computer)?|restart(\s+pc|\s+computer)?|reboot|"
                    r"sleep(\s+pc|\s+computer)?|hibernate|"
                    r"lock(\s+screen|\s+computer|\s+pc)?|"
                    r"volume\s+(up|down|mute|unmute|max|zero)|\b(mute|unmute)\s+volume|"
                    r"brightness\s+(up|down|increase|decrease)|"
                    r"maximize\s+window|minimize\s+window|take\s+screenshot)\b",
                    re.IGNORECASE,
                ),
                confidence=0.95,
                reason="Operating system or local application control command",
                priority=30,
            ),
            # ------------------------------------------------------------------
            # 4. CODING INTENTS (Priority 40: Software, programming, debugging, APIs)
            # ------------------------------------------------------------------
            _Rule(
                intent=IntentType.CODING,
                pattern=re.compile(
                    r"\b(write\s+(python|javascript|typescript|c\+\+|rust|go|java|bash|sql|code|script|program)|"
                    r"debug\s+(this|code|script|error|function)|"
                    r"fix\s+(error|bug|issue|syntax|exception|crash)|"
                    r"create\s+(api|endpoint|function|class|decorator|module|dockerfile|database\s+schema)|"
                    r"refactor\s+(this|code|function)|"
                    r"implement\s+(an?\s+algorithm|binary\s+search|quick\s*sort|rest\s+api)|"
                    r"unit\s+test|stack\s*trace|traceback|syntax\s*error|type\s*error|"
                    r"github|git\s+commit|pull\s+request|regex\s+pattern)\b",
                    re.IGNORECASE,
                ),
                confidence=0.95,
                reason="Software development or programming task",
                priority=40,
            ),
            # ------------------------------------------------------------------
            # 5. TOOL INTENTS (Priority 50: Calculations, conversions, translations, CLI)
            # ------------------------------------------------------------------
            _Rule(
                intent=IntentType.TOOL,
                pattern=re.compile(
                    r"\b(calculate|compute|solve|convert|translate|"
                    r"run\s+command|execute\s+command|"
                    r"timer|stopwatch|set\s+alarm|set\s+timer|"
                    r"currency\s+converter|unit\s+converter)\b|"
                    r"^[\d\s\+\-\*\/\^\(\)\.\%]+$",
                    re.IGNORECASE,
                ),
                confidence=0.90,
                reason="Deterministic computational, conversion, or utility tool instruction",
                priority=50,
            ),
            # ------------------------------------------------------------------
            # 6. SEARCH INTENTS (Priority 60: Real-time queries, weather, news, stocks)
            # ------------------------------------------------------------------
            _Rule(
                intent=IntentType.SEARCH,
                pattern=re.compile(
                    r"\b(latest\s+news|breaking\s+news|weather(\s+today|\s+forecast|\s+report)?|"
                    r"stock\s+price|market\s+price|crypto\s+price|"
                    r"today('?s)?\s+match|live\s+score|score\s+of|"
                    r"current\s+president|who\s+is\s+the\s+(current\s+)?(president|prime\s+minister|ceo)|"
                    r"search\s+(the\s+web\s+for|for|google)|who\s+won\s+the\s+match|"
                    r"what\s+is\s+the\s+weather|how\s+is\s+the\s+weather)\b",
                    re.IGNORECASE,
                ),
                confidence=0.90,
                reason="Real-time web or external status lookup query",
                priority=60,
            ),
            # ------------------------------------------------------------------
            # 7. REASONING INTENTS (Priority 70: Explanations, comparisons, analysis)
            # ------------------------------------------------------------------
            _Rule(
                intent=IntentType.REASONING,
                pattern=re.compile(
                    r"\b(explain\s+(recursion|quantum|relativity|machine\s+learning|gravity|concept|how|why|difference)|"
                    r"compare\s+(\w+)\s+(and|with|to)\s+(\w+)|"
                    r"why\s+is\s+(the\s+)?sky\s+blue|why\s+does|why\s+do|why\s+is|"
                    r"what\s+is\s+the\s+difference\s+between|"
                    r"pros\s+and\s+cons\s+of|advantages\s+and\s+disadvantages|"
                    r"how\s+does\s+(\w+)\s+work|deep\s+dive\s+into|philosophical|prove\s+that)\b",
                    re.IGNORECASE,
                ),
                confidence=0.90,
                reason="Complex conceptual reasoning, causal inquiry, or comparative analysis",
                priority=70,
            ),
            # ------------------------------------------------------------------
            # 8. CHAT INTENTS (Priority 80: Greetings, social small talk, persona)
            # ------------------------------------------------------------------
            _Rule(
                intent=IntentType.CHAT,
                pattern=re.compile(
                    r"\b(hello|hi|hey|greetings|howdy|sup|what'?s\s+up|"
                    r"how\s+are\s+you|how's\s+it\s+going|how\s+do\s+you\s+do|"
                    r"who\s+are\s+you|what\s+is\s+your\s+name|tell\s+me\s+about\s+yourself|"
                    r"good\s+(morning|afternoon|evening|night)|"
                    r"thank\s+you|thanks|bye|goodbye|see\s+you)\b",
                    re.IGNORECASE,
                ),
                confidence=0.85,
                reason="Conversational greeting or agent identity interaction",
                priority=80,
            ),
        ]
        return sorted(rules, key=lambda r: r.priority)


# ------------------------------------------------------------------------------
# Default Singleton Instance Export
# ------------------------------------------------------------------------------
intent_router: Final[IntentRouter] = IntentRouter()
