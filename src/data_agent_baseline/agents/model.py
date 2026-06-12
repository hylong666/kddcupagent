from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError


DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0


@dataclass(frozen=True, slots=True)
class ModelMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ModelStep:
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str


class ModelAdapter(Protocol):
    def complete(self, messages: list[ModelMessage]) -> str:
        raise NotImplementedError


class OpenAIModelAdapter:
    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float,
        request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.request_timeout_seconds = max(1.0, float(request_timeout_seconds))
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))

    def complete(self, messages: list[ModelMessage]) -> str:
        if not self.api_key:
            raise RuntimeError("Missing model API key in config.agent.api_key.")

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )

        last_error: Exception | None = None
        for attempt_index in range(1, self.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": message.role, "content": message.content} for message in messages],
                    temperature=self.temperature,
                    timeout=self.request_timeout_seconds,
                )
                break
            except (APITimeoutError, APIConnectionError, RateLimitError) as exc:
                last_error = exc
            except APIError as exc:
                status_code = getattr(exc, "status_code", None)
                if status_code is not None and 400 <= status_code < 500 and status_code != 429:
                    raise RuntimeError(f"Model request failed: {exc}") from exc
                last_error = exc

            if attempt_index >= self.max_retries:
                assert last_error is not None
                raise RuntimeError(
                    "Model request failed after "
                    f"{self.max_retries} attempts: {last_error}"
                ) from last_error

            sleep_seconds = self.retry_backoff_seconds * attempt_index
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        choices = response.choices or []
        if not choices:
            raise RuntimeError("Model response missing choices.")
        content = choices[0].message.content
        if not isinstance(content, str):
            raise RuntimeError("Model response missing text content.")
        return content


class ScriptedModelAdapter:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def complete(self, messages: list[ModelMessage]) -> str:
        del messages
        if not self._responses:
            raise RuntimeError("No scripted model responses remaining.")
        return self._responses.pop(0)
