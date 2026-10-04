"""Application-side tool validation and execution.

The hosted Gemma 4 API returns function calls as structured parts but never
runs them, so every call is validated and executed here. Results are fed back
as plain text turns, which keeps the loop independent of any provider's
tool-response wire format.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .base import GenerationConfig, LLMProvider, LLMResponse, Message, ToolCall, ToolDeclaration

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


class ToolValidationError(Exception):
    """The model asked for an unknown tool or passed arguments that fail the schema."""


@dataclass(frozen=True)
class Tool:
    declaration: ToolDeclaration
    handler: Callable[..., Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, description: str, parameters: dict[str, Any], handler: Callable[..., Any]) -> None:
        self._tools[name] = Tool(ToolDeclaration(name, description, parameters), handler)

    def declarations(self) -> list[ToolDeclaration]:
        return [tool.declaration for tool in self._tools.values()]

    def validate(self, call: ToolCall) -> Tool:
        tool = self._tools.get(call.name)
        if tool is None:
            raise ToolValidationError(f"Unknown tool: {call.name!r}")
        schema = tool.declaration.parameters
        properties = schema.get("properties", {})
        missing = [key for key in schema.get("required", []) if key not in call.arguments]
        if missing:
            raise ToolValidationError(f"{call.name}: missing required argument(s): {', '.join(missing)}")
        for key, value in call.arguments.items():
            if key not in properties:
                raise ToolValidationError(f"{call.name}: unexpected argument {key!r}")
            expected = _JSON_TYPES.get(properties[key].get("type", ""))
            # bool is a subclass of int in Python, so reject it explicitly for numeric fields.
            wrong_bool = isinstance(value, bool) and properties[key].get("type") != "boolean"
            if expected and (not isinstance(value, expected) or wrong_bool):
                raise ToolValidationError(
                    f"{call.name}: argument {key!r} must be of type {properties[key]['type']}"
                )
        return tool

    def execute(self, call: ToolCall) -> str:
        """Validate and run a call. Failures are returned as text so the model can recover."""
        try:
            tool = self.validate(call)
            result = tool.handler(**call.arguments)
        except ToolValidationError as exc:
            return f"ERROR: {exc}"
        except Exception as exc:  # a failing tool must not take the request down
            return f"ERROR: {call.name} failed: {exc}"
        return result if isinstance(result, str) else json.dumps(result, default=str)


def run_with_tools(
    provider: LLMProvider,
    messages: Sequence[Message],
    registry: ToolRegistry,
    *,
    system: str | None = None,
    config: GenerationConfig | None = None,
    max_rounds: int = 3,
) -> LLMResponse:
    """Let the model call tools for up to ``max_rounds`` rounds, then return its text answer."""
    conversation = list(messages)
    for _ in range(max_rounds):
        response = provider.generate(conversation, system=system, config=config, tools=registry.declarations())
        if not response.tool_calls:
            return response
        requested = "\n".join(
            f"[tool call] {call.name} {json.dumps(call.arguments, default=str)}" for call in response.tool_calls
        )
        results = "\n\n".join(
            f"[tool result: {call.name}]\n{registry.execute(call)}" for call in response.tool_calls
        )
        conversation.append(Message("assistant", requested))
        conversation.append(Message("user", results))
    # Out of rounds: force a final answer with tools switched off.
    return provider.generate(conversation, system=system, config=config)
