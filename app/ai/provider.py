"""Google Gemini model provider implementation for J.A.R.V.I.S."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Any, Final, Optional

import httpx

from app.ai.models import (
    AIAuthenticationError,
    AIConfigError,
    AIProviderError,
    AIRateLimitError,
    AIResponse,
    AITimeoutError,
    GenerationConfig,
)
from app.core.config import settings
from app.core.logger import get_logger

GEMINI_API_BASE_URL: Final[str] = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_TIMEOUT_SECONDS: Final[float] = 30.0
DEFAULT_MAX_RETRIES: Final[int] = 3
DEFAULT_RETRY_DELAY: Final[float] = 1.0
DEFAULT_BACKOFF_FACTOR: Final[float] = 2.0


class GeminiProvider:
    """Thread-safe Gemini API provider supporting both synchronous and asynchronous generation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        *,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        logger: Optional[logging.Logger] = None,
        client: Optional[httpx.Client] = None,
        async_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        """Initialize the GeminiProvider.

        Args:
            api_key: Optional Gemini API key. If omitted, resolved from config or environment.
            model: Optional target Gemini model. If omitted, resolved dynamically from settings.ai.model.
            timeout: Optional HTTP request timeout in seconds. If omitted, resolved from settings.ai.timeout.
            max_retries: Optional maximum retry attempts. If omitted, resolved from settings.ai.max_retries.
            retry_delay: Initial retry backoff delay in seconds.
            backoff_factor: Exponential multiplier for backoff.
            logger: Custom logger instance.
            client: Optional pre-configured synchronous httpx.Client (useful for unit tests).
            async_client: Optional pre-configured asynchronous httpx.AsyncClient.
        """
        self._lock = threading.RLock()

        # Model is never hardcoded: always read from passed model override or settings.ai.model
        if model is not None and isinstance(model, str) and model.strip():
            self._model = model.strip()
        else:
            self._model = getattr(settings.ai, "model", "")

        # Configurable timeout from settings
        if timeout is not None:
            self._timeout = float(timeout)
        else:
            self._timeout = float(getattr(settings.ai, "timeout", DEFAULT_TIMEOUT_SECONDS))

        # Configurable max retries from settings
        if max_retries is not None:
            self._max_retries = int(max_retries)
        else:
            self._max_retries = int(getattr(settings.ai, "max_retries", DEFAULT_MAX_RETRIES))

        self._retry_delay = retry_delay
        self._backoff_factor = backoff_factor
        self._logger = logger if logger is not None else get_logger("AI.GEMINI")

        # Resolve API Key
        if api_key is not None:
            resolved_key = api_key
        else:
            resolved_key = getattr(settings.ai, "gemini_api_key", None) or os.getenv("GEMINI_API_KEY")
        self._api_key = resolved_key.strip() if resolved_key and isinstance(resolved_key, str) else ""

        # Internal HTTP clients
        self._sync_client = client
        self._async_client = async_client

    @property
    def model(self) -> str:
        """Return the target model identifier."""
        return self._model

    @property
    def api_key(self) -> str:
        """Return the active API key."""
        return self._api_key

    def set_api_key(self, api_key: str) -> None:
        """Update the active API key.

        Args:
            api_key: New Gemini API key.
        """
        with self._lock:
            self._api_key = api_key.strip()

    def set_model(self, model: str) -> None:
        """Update the target Gemini model.

        Args:
            model: New model identifier.
        """
        with self._lock:
            self._model = model.strip()

    def _validate_api_key(self) -> str:
        """Ensure a valid API key is present before dispatching requests.

        Raises:
            AIConfigError: If API key is unset, empty, or placeholder.
        """
        key = self._api_key
        if not key or key in ("your_gemini_api_key_here", "dummy", "placeholder"):
            raise AIConfigError(
                "Gemini API key is not configured. Set GEMINI_API_KEY in .env or pass to GeminiProvider."
            )
        return key

    def _build_url(self) -> str:
        """Build the endpoint URL for Gemini generateContent."""
        return f"{GEMINI_API_BASE_URL}/{self._model}:generateContent"

    def _get_sync_client(self) -> httpx.Client:
        """Get or initialize thread-safe synchronous httpx client."""
        with self._lock:
            if self._sync_client is None or self._sync_client.is_closed:
                self._sync_client = httpx.Client(timeout=self._timeout)
            return self._sync_client

    def _get_async_client(self) -> httpx.AsyncClient:
        """Get or initialize asynchronous httpx client."""
        with self._lock:
            if self._async_client is None or self._async_client.is_closed:
                self._async_client = httpx.AsyncClient(timeout=self._timeout)
            return self._async_client

    def _parse_response(
        self,
        response: httpx.Response,
        duration: float,
    ) -> AIResponse:
        """Parse Gemini API response into typed AIResponse or raise domain exceptions.

        Args:
            response: Completed HTTP response.
            duration: Request round-trip duration in seconds.

        Returns:
            Structured AIResponse.

        Raises:
            AIAuthenticationError: On HTTP 401 or 403.
            AIRateLimitError: On HTTP 429.
            AIProviderError: On other non-200 HTTP statuses or malformed bodies.
        """
        status = response.status_code

        # Error handling
        if status in (401, 403):
            self._logger.error(f"Gemini authentication failed (HTTP {status}): {response.text}")
            raise AIAuthenticationError(
                f"Authentication failed with status {status}: {response.text}",
                status_code=status,
            )

        if status == 429:
            self._logger.warning(f"Gemini rate limit exceeded (HTTP 429): {response.text}")
            raise AIRateLimitError(
                f"Rate limit exceeded (HTTP 429): {response.text}",
                status_code=status,
            )

        if status >= 400:
            self._logger.error(f"Gemini API error (HTTP {status}): {response.text}")
            raise AIProviderError(
                f"Gemini API error (HTTP {status}): {response.text}",
                status_code=status,
            )

        try:
            data = response.json()
        except Exception as exc:
            raise AIProviderError(f"Failed to parse Gemini JSON response: {exc}") from exc

        candidates = data.get("candidates", [])
        if not candidates:
            # Check for content filters or prompt blocking
            prompt_feedback = data.get("promptFeedback", {})
            block_reason = prompt_feedback.get("blockReason", "No candidates returned")
            raise AIProviderError(f"Gemini returned no response candidates: {block_reason}")

        first_cand = candidates[0]
        content_obj = first_cand.get("content", {})
        parts = content_obj.get("parts", [])
        text = "".join(part.get("text", "") for part in parts if "text" in part)
        finish_reason = first_cand.get("finishReason")

        usage = data.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount")
        completion_tokens = usage.get("candidatesTokenCount")
        total_tokens = usage.get("totalTokenCount")

        return AIResponse(
            content=text,
            model=self._model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            duration=duration,
            finish_reason=finish_reason,
            metadata={"status_code": status},
            raw_response=data,
        )

    def generate(
        self,
        payload: dict[str, Any],
        config: Optional[GenerationConfig] = None,
    ) -> AIResponse:
        """Execute synchronous content generation with automatic retries.

        Args:
            payload: Pre-built Gemini dictionary containing 'contents' and 'systemInstruction'.
            config: Optional sampling and generation configuration parameters.

        Returns:
            The structured AIResponse.

        Raises:
            AIConfigError: If API key is missing.
            AITimeoutError: If request times out.
            AIRateLimitError: If rate limit quota is exceeded.
            AIAuthenticationError: If authentication fails.
            AIProviderError: If all retry attempts fail.
        """
        api_key = self._validate_api_key()
        url = self._build_url()
        client = self._get_sync_client()

        body = dict(payload)
        if config is not None:
            body["generationConfig"] = config.to_gemini_dict()

        headers = {"Content-Type": "application/json"}
        params = {"key": api_key}

        start_time = time.perf_counter()
        delay = self._retry_delay

        for attempt in range(1, self._max_retries + 1):
            try:
                self._logger.debug(
                    f"Calling Gemini API '{self._model}' (attempt {attempt}/{self._max_retries})"
                )
                resp = client.post(url, json=body, headers=headers, params=params)
                duration = time.perf_counter() - start_time
                return self._parse_response(resp, duration)

            except (AIAuthenticationError, AIConfigError):
                # Never retry authentication or configuration errors
                raise

            except AIRateLimitError:
                if attempt == self._max_retries:
                    raise
                self._logger.warning(
                    f"Rate limit hit. Retrying in {delay:.2f}s (attempt {attempt}/{self._max_retries})..."
                )
                time.sleep(delay)
                delay *= self._backoff_factor

            except httpx.TimeoutException as exc:
                if attempt == self._max_retries:
                    raise AITimeoutError(
                        f"Request to Gemini API timed out after {self._timeout}s."
                    ) from exc
                self._logger.warning(
                    f"Request timed out. Retrying in {delay:.2f}s (attempt {attempt}/{self._max_retries})..."
                )
                time.sleep(delay)
                delay *= self._backoff_factor

            except (httpx.RequestError, AIProviderError) as exc:
                if attempt == self._max_retries:
                    if isinstance(exc, AIProviderError):
                        raise
                    raise AIProviderError(f"Network error connecting to Gemini API: {exc}") from exc
                self._logger.warning(
                    f"Transient provider error ({exc}). Retrying in {delay:.2f}s..."
                )
                time.sleep(delay)
                delay *= self._backoff_factor

        raise AIProviderError("Unexpected failure: all retry attempts exhausted.")

    async def generate_async(
        self,
        payload: dict[str, Any],
        config: Optional[GenerationConfig] = None,
    ) -> AIResponse:
        """Execute asynchronous content generation with non-blocking retries.

        Args:
            payload: Pre-built Gemini dictionary containing 'contents' and 'systemInstruction'.
            config: Optional sampling and generation configuration parameters.

        Returns:
            The structured AIResponse.

        Raises:
            AIConfigError: If API key is missing.
            AITimeoutError: If request times out.
            AIRateLimitError: If rate limit quota is exceeded.
            AIAuthenticationError: If authentication fails.
            AIProviderError: If all retry attempts fail.
        """
        api_key = self._validate_api_key()
        url = self._build_url()
        client = self._get_async_client()

        body = dict(payload)
        if config is not None:
            body["generationConfig"] = config.to_gemini_dict()

        headers = {"Content-Type": "application/json"}
        params = {"key": api_key}

        start_time = time.perf_counter()
        delay = self._retry_delay

        for attempt in range(1, self._max_retries + 1):
            try:
                self._logger.debug(
                    f"Calling async Gemini API '{self._model}' (attempt {attempt}/{self._max_retries})"
                )
                resp = await client.post(url, json=body, headers=headers, params=params)
                duration = time.perf_counter() - start_time
                return self._parse_response(resp, duration)

            except (AIAuthenticationError, AIConfigError):
                raise

            except AIRateLimitError:
                if attempt == self._max_retries:
                    raise
                self._logger.warning(
                    f"Async rate limit hit. Retrying in {delay:.2f}s (attempt {attempt}/{self._max_retries})..."
                )
                await asyncio.sleep(delay)
                delay *= self._backoff_factor

            except httpx.TimeoutException as exc:
                if attempt == self._max_retries:
                    raise AITimeoutError(
                        f"Async request to Gemini API timed out after {self._timeout}s."
                    ) from exc
                self._logger.warning(
                    f"Async request timed out. Retrying in {delay:.2f}s (attempt {attempt}/{self._max_retries})..."
                )
                await asyncio.sleep(delay)
                delay *= self._backoff_factor

            except (httpx.RequestError, AIProviderError) as exc:
                if attempt == self._max_retries:
                    if isinstance(exc, AIProviderError):
                        raise
                    raise AIProviderError(f"Async network error connecting to Gemini API: {exc}") from exc
                self._logger.warning(
                    f"Async transient error ({exc}). Retrying in {delay:.2f}s..."
                )
                await asyncio.sleep(delay)
                delay *= self._backoff_factor

        raise AIProviderError("Unexpected failure: all retry attempts exhausted.")

    def stream_generate(
        self,
        payload: dict[str, Any],
        config: Optional[GenerationConfig] = None,
    ) -> Any:
        """Stream content generation synchronously (future capability).

        Raises:
            NotImplementedError: Streaming generation is not yet implemented.
        """
        raise NotImplementedError("Streaming generation is not yet implemented.")

    async def stream_generate_async(
        self,
        payload: dict[str, Any],
        config: Optional[GenerationConfig] = None,
    ) -> Any:
        """Stream content generation asynchronously (future capability).

        Raises:
            NotImplementedError: Asynchronous streaming generation is not yet implemented.
        """
        raise NotImplementedError("Asynchronous streaming generation is not yet implemented.")

    def close(self) -> None:
        """Close active HTTP client sessions."""
        with self._lock:
            if self._sync_client is not None and not self._sync_client.is_closed:
                self._sync_client.close()

    async def aclose(self) -> None:
        """Close active asynchronous HTTP client sessions."""
        with self._lock:
            if self._async_client is not None and not self._async_client.is_closed:
                await self._async_client.aclose()
