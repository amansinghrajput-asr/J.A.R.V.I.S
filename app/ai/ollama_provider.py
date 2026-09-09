from __future__ import annotations

import time
import requests

from app.ai.models import AIResponse


class OllamaProvider:
    def __init__(
        self,
        model: str = "qwen2.5:3b",
        host: str = "http://127.0.0.1:11434",
    ):
        self._model = model
        self._host = host

    @property
    def model(self):
        return self._model

    def generate(self, payload, config=None):
        prompt = ""

        if isinstance(payload, dict):
            contents = payload.get("contents", [])
            for item in contents:
                for part in item.get("parts", []):
                    prompt += part.get("text", "") + "\n"

        start = time.perf_counter()

        response = requests.post(
            f"{self._host}/api/generate",
            json={
                "model": self._model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )

        response.raise_for_status()

        data = response.json()

        duration = time.perf_counter() - start

        return AIResponse(
            content=data["response"],
            model=self._model,
            duration=duration,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            finish_reason="stop",
            metadata={},
            raw_response=data,
        )