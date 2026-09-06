"""Prompt builder and system persona configuration for J.A.R.V.I.S."""

from __future__ import annotations

import threading
from typing import Any, Final, Optional, Sequence

from app.ai.models import PromptError, Role
from app.memory.models import ConversationMemory

DEFAULT_SYSTEM_PROMPT: Final[str] = (
    "You are J.A.R.V.I.S (Just A Rather Very Intelligent System), an advanced AI desktop operating system assistant. "
    "You are articulate, courteous, witty, highly analytical, and concise. Address the user respectfully as 'Sir' or 'Boss'. "
    "You are fully bilingual and effortlessly understand and converse in English, Hindi (हिंदी), and Hinglish (code-switched Hindi-English). "
    "Provide direct, high-utility answers. Maintain clarity and precision at all times."
)


class PromptBuilder:
    """Thread-safe builder for constructing structured, multi-turn AI prompts and context."""

    def __init__(
        self,
        default_system_prompt: Optional[str] = None,
        *,
        history_limit: Optional[int] = None,
    ) -> None:
        """Initialize the PromptBuilder.

        Args:
            default_system_prompt: Optional override for the foundational system prompt.
            history_limit: Optional limit on conversational history entries included in prompts.
        """
        self._lock = threading.RLock()
        self._system_prompt = (
            default_system_prompt.strip()
            if default_system_prompt and default_system_prompt.strip()
            else DEFAULT_SYSTEM_PROMPT
        )
        from app.core.config import settings

        if history_limit is not None:
            self._history_limit = int(history_limit)
        elif hasattr(settings, "ai") and hasattr(settings.ai, "history_limit"):
            self._history_limit = int(settings.ai.history_limit)
        else:
            self._history_limit = 20

    @property
    def history_limit(self) -> int:
        """Return the active history limit."""
        with self._lock:
            return self._history_limit

    def set_history_limit(self, limit: int) -> None:
        """Update the history limit.

        Args:
            limit: Maximum history turns to include.
        """
        with self._lock:
            self._history_limit = max(0, int(limit))

    @property
    def system_prompt(self) -> str:
        """Return the active default system prompt."""
        with self._lock:
            return self._system_prompt

    def set_system_prompt(self, prompt: str) -> None:
        """Update the base system prompt.

        Args:
            prompt: Non-empty system instruction string.

        Raises:
            PromptError: If prompt is empty or not a string.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise PromptError("System prompt must be a non-empty string.")
        with self._lock:
            self._system_prompt = prompt.strip()

    def format_history(
        self,
        memories: Sequence[ConversationMemory],
    ) -> list[dict[str, Any]]:
        """Convert a sequence of ConversationMemory entries into Gemini chat turns.

        Enforces role mapping ('assistant' -> 'model') and merges adjacent turns
        with identical roles to adhere to strict provider alternation contracts.

        Args:
            memories: Sequence of historical ConversationMemory instances.

        Returns:
            List of Gemini-formatted message contents: `[{"role": str, "parts": [{"text": str}]}]`.
        """
        contents: list[dict[str, Any]] = []

        for mem in memories:
            # Skip system memories here since system instructions are passed via systemInstruction
            if mem.role == Role.SYSTEM.value:
                continue

            role = "model" if mem.role in ("assistant", "model") else "user"
            text = mem.content.strip() if isinstance(mem.content, str) else str(mem.content)
            if not text:
                continue

            if contents and contents[-1]["role"] == role:
                # Merge consecutive turns with the same role
                existing_text = contents[-1]["parts"][0]["text"]
                contents[-1]["parts"][0]["text"] = f"{existing_text}\n{text}"
            else:
                contents.append({"role": role, "parts": [{"text": text}]})

        return contents

    def build_payload(
        self,
        query: str,
        *,
        history: Optional[Sequence[ConversationMemory]] = None,
        system_prompt: Optional[str] = None,
        extra_context: Optional[dict[str, Any] | str] = None,
    ) -> dict[str, Any]:
        """Construct a complete, provider-agnostic Gemini API generation payload.

        Args:
            query: The user's input query or command.
            history: Optional conversation memories from MemoryManager.
            system_prompt: Optional custom system prompt override for this request.
            extra_context: Optional supplementary contextual metadata or text string.

        Returns:
            Dictionary payload conforming to Gemini API structure with 'contents'
            and 'systemInstruction'.

        Raises:
            PromptError: If query is empty or invalid.
        """
        if not isinstance(query, str) or not query.strip():
            raise PromptError("User query must be a non-empty string.")

        with self._lock:
            active_sys_prompt = (
                system_prompt.strip()
                if system_prompt and isinstance(system_prompt, str) and system_prompt.strip()
                else self._system_prompt
            )

        # Append extra context to system prompt if provided
        if extra_context:
            if isinstance(extra_context, dict):
                ctx_lines = [f"- {k}: {v}" for k, v in extra_context.items()]
                ctx_str = "\n".join(ctx_lines)
                active_sys_prompt += f"\n\n[Active System Context]\n{ctx_str}"
            elif isinstance(extra_context, str) and extra_context.strip():
                active_sys_prompt += f"\n\n[Active System Context]\n{extra_context.strip()}"

        payload: dict[str, Any] = {
            "systemInstruction": {
                "parts": [{"text": active_sys_prompt}]
            }
        }

        # Multi-turn conversational history
        contents: list[dict[str, Any]] = []
        if history:
            limit = self.history_limit
            trimmed_history = list(history)[-limit:] if limit > 0 else []
            contents.extend(self.format_history(trimmed_history))

        # Append current user query
        clean_query = query.strip()
        if contents and contents[-1]["role"] == "user":
            existing_text = contents[-1]["parts"][0]["text"]
            contents[-1]["parts"][0]["text"] = f"{existing_text}\n{clean_query}"
        else:
            contents.append({"role": "user", "parts": [{"text": clean_query}]})

        payload["contents"] = contents
        return payload
