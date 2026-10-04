"""Gemma 4 adapter for Google's hosted Gemini API (google-genai SDK)."""

from __future__ import annotations

from typing import Any, Sequence

import httpx
from google import genai
from google.genai import errors, types

from .base import (
    GenerationConfig,
    LLMAuthError,
    LLMConfigurationError,
    LLMError,
    LLMProvider,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponse,
    LLMResponseError,
    LLMTimeoutError,
    Message,
    ToolCall,
    ToolDeclaration,
)
from .config import DEFAULT_MODEL


class GoogleGemmaProvider(LLMProvider):
    def __init__(
        self,
        api_key: str | None,
        model: str = DEFAULT_MODEL,
        default_config: GenerationConfig | None = None,
        client: Any | None = None,
    ) -> None:
        if client is None:
            if not api_key:
                raise LLMConfigurationError(
                    "GOOGLE_API_KEY is not set. Create a key at https://aistudio.google.com/apikey "
                    "and set it in the server environment."
                )
            # Passed explicitly so a stray GEMINI_API_KEY in the environment is never picked up.
            client = genai.Client(api_key=api_key)
        self._client = client
        self.model = model
        self._default_config = default_config or GenerationConfig()

    def generate(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        config: GenerationConfig | None = None,
        tools: Sequence[ToolDeclaration] | None = None,
    ) -> LLMResponse:
        if not messages:
            raise ValueError("messages must not be empty")
        cfg = config or self._default_config
        contents = [
            types.Content(
                role="model" if m.role == "assistant" else "user",
                parts=[types.Part.from_text(text=m.content)],
            )
            for m in messages
        ]
        request_config = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_output_tokens,
            # HttpOptions.timeout is in milliseconds.
            http_options=types.HttpOptions(timeout=int(cfg.timeout_seconds * 1000)),
            tools=[types.Tool(function_declarations=[_declaration(t) for t in tools])] if tools else None,
            # Tools are executed by the application (app/llm/tools.py), never by the SDK.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        try:
            response = self._client.models.generate_content(
                model=self.model, contents=contents, config=request_config
            )
        except Exception as exc:
            raise self._translate(exc, cfg.timeout_seconds) from exc
        return self._parse(response)

    def check(self) -> None:
        """Fetch the model's metadata: validates the key and the model ID without spending tokens."""
        try:
            self._client.models.get(model=self.model)
        except Exception as exc:
            raise self._translate(exc, self._default_config.timeout_seconds) from exc

    def _parse(self, response: Any) -> LLMResponse:
        feedback = getattr(response, "prompt_feedback", None)
        if feedback is not None and getattr(feedback, "block_reason", None):
            raise LLMResponseError(f"Prompt was blocked by the provider: {_name(feedback.block_reason)}.")
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            raise LLMResponseError("The model returned no candidates.")
        candidate = candidates[0]
        finish_reason = _name(getattr(candidate, "finish_reason", None))
        parts = getattr(getattr(candidate, "content", None), "parts", None) or []

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for part in parts:
            # Gemma 4 can return its reasoning as "thought" parts; never surface those as the answer.
            if getattr(part, "thought", False):
                continue
            call = getattr(part, "function_call", None)
            if call is not None and call.name:
                tool_calls.append(ToolCall(name=call.name, arguments=dict(call.args or {})))
            elif getattr(part, "text", None):
                text_parts.append(part.text)

        text = "".join(text_parts).strip()
        if not text and not tool_calls:
            hint = (
                " The output-token limit was reached before any answer text was produced; "
                "raise GEMMA_MAX_OUTPUT_TOKENS."
                if finish_reason == "MAX_TOKENS"
                else ""
            )
            raise LLMResponseError(f"The model returned no text (finish_reason={finish_reason}).{hint}")

        usage_meta = getattr(response, "usage_metadata", None)
        usage = {
            key: value
            for key, value in (
                ("prompt_tokens", getattr(usage_meta, "prompt_token_count", None)),
                ("output_tokens", getattr(usage_meta, "candidates_token_count", None)),
                ("thought_tokens", getattr(usage_meta, "thoughts_token_count", None)),
                ("total_tokens", getattr(usage_meta, "total_token_count", None)),
            )
            if value is not None
        }
        return LLMResponse(
            text=text,
            model=self.model,
            finish_reason=finish_reason,
            tool_calls=tuple(tool_calls),
            usage=usage,
        )

    def _translate(self, exc: Exception, timeout_seconds: float) -> LLMError:
        if isinstance(exc, LLMError):
            return exc
        if isinstance(exc, httpx.TimeoutException):
            return LLMTimeoutError(f"Gemma request timed out after {timeout_seconds:g}s.")
        if isinstance(exc, errors.APIError):
            code, message = exc.code, exc.message or str(exc)
            if code in (401, 403) or (code == 400 and "api key" in message.lower()):
                return LLMAuthError(f"Google rejected GOOGLE_API_KEY ({code}): {message}")
            if code == 404:
                return LLMConfigurationError(
                    f"Model {self.model!r} was not found ({code}). Check GEMMA_MODEL. {message}"
                )
            if code == 429:
                return LLMRateLimitError(f"Gemini API quota or rate limit exceeded: {message}")
            if code in (408, 504):
                return LLMTimeoutError(f"Gemini API timed out ({code}): {message}")
            return LLMProviderError(f"Gemini API error {code} {exc.status or ''}: {message}")
        if isinstance(exc, httpx.HTTPError):
            return LLMProviderError(f"Network error calling the Gemini API: {exc}")
        return LLMProviderError(f"Unexpected error calling the Gemini API: {exc}")


def _declaration(tool: ToolDeclaration) -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name=tool.name,
        description=tool.description,
        parameters_json_schema=tool.parameters,
    )


def _name(value: Any) -> str | None:
    if value is None:
        return None
    return getattr(value, "name", None) or str(value)
