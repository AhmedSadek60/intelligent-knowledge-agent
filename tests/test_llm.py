"""Unit tests for the LLM integration. The Google client is always mocked: no network, no key."""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors, types

from app.llm import (
    INSUFFICIENT_EVIDENCE,
    RAG_SYSTEM_PROMPT,
    GenerationConfig,
    KnowledgeAssistant,
    LLMAuthError,
    LLMConfigurationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponse,
    LLMResponseError,
    LLMSettings,
    LLMTimeoutError,
    Message,
    Passage,
    SQLiteMessageStore,
    ToolCall,
    ToolRegistry,
    build_rag_user_message,
    run_with_tools,
)
from app.llm.gemma import GoogleGemmaProvider
from app.llm.health import check_health


class FakeModels:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.result

    def get(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(name=kwargs["model"])


def provider_with(result=None, error=None, **kwargs):
    models = FakeModels(result, error)
    provider = GoogleGemmaProvider(api_key=None, client=SimpleNamespace(models=models), **kwargs)
    return provider, models


def response(parts, finish=types.FinishReason.STOP):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts), finish_reason=finish)],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10, candidates_token_count=5, total_token_count=15
        ),
    )


def api_error(code, message="boom", status="ERR"):
    cls = errors.ClientError if code < 500 else errors.ServerError
    return cls(code, {"error": {"code": code, "message": message, "status": status}})


# --- Gemma adapter -----------------------------------------------------------


def test_generate_sends_model_config_and_mapped_roles():
    provider, models = provider_with(response([types.Part(text="Hello")]), model="gemma-4-31b-it")
    result = provider.generate(
        [Message("user", "hi"), Message("assistant", "hey"), Message("user", "again")],
        system="be brief",
        config=GenerationConfig(temperature=0.7, max_output_tokens=99, timeout_seconds=12),
    )

    call = models.calls[0]
    assert call["model"] == "gemma-4-31b-it"
    assert [c.role for c in call["contents"]] == ["user", "model", "user"]
    assert call["contents"][2].parts[0].text == "again"
    assert call["config"].system_instruction == "be brief"
    assert call["config"].temperature == 0.7
    assert call["config"].max_output_tokens == 99
    assert call["config"].http_options.timeout == 12000  # milliseconds
    assert call["config"].tools is None
    assert result.text == "Hello"
    assert result.finish_reason == "STOP"
    assert result.usage == {"prompt_tokens": 10, "output_tokens": 5, "total_tokens": 15}


def test_generate_excludes_thought_parts():
    provider, _ = provider_with(response([types.Part(text="reasoning...", thought=True), types.Part(text="Answer")]))
    assert provider.generate([Message("user", "q")]).text == "Answer"


def test_generate_surfaces_tool_calls_without_executing_them():
    part = types.Part(function_call=types.FunctionCall(name="search", args={"query": "x"}))
    provider, models = provider_with(response([part]))
    registry = ToolRegistry()
    registry.register("search", "Search", {"type": "object", "properties": {"query": {"type": "string"}}}, lambda query: "r")

    result = provider.generate([Message("user", "q")], tools=registry.declarations())

    assert result.tool_calls == (ToolCall("search", {"query": "x"}),)
    declared = models.calls[0]["config"].tools[0].function_declarations[0]
    assert declared.name == "search"
    assert declared.parameters_json_schema["properties"]["query"]["type"] == "string"


def test_empty_output_at_token_limit_explains_how_to_fix():
    provider, _ = provider_with(response([types.Part(text="thinking", thought=True)], types.FinishReason.MAX_TOKENS))
    with pytest.raises(LLMResponseError, match="GEMMA_MAX_OUTPUT_TOKENS"):
        provider.generate([Message("user", "q")])


def test_blocked_prompt_raises_response_error():
    blocked = types.GenerateContentResponse(
        prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason=types.BlockedReason.SAFETY)
    )
    provider, _ = provider_with(blocked)
    with pytest.raises(LLMResponseError, match="SAFETY"):
        provider.generate([Message("user", "q")])


@pytest.mark.parametrize(
    "error, expected",
    [
        (api_error(403, "permission denied"), LLMAuthError),
        (api_error(400, "API key not valid. Please pass a valid API key."), LLMAuthError),
        (api_error(404, "models/nope is not found"), LLMConfigurationError),
        (api_error(429, "quota"), LLMRateLimitError),
        (api_error(504, "deadline"), LLMTimeoutError),
        (api_error(500, "internal"), LLMProviderError),
        (api_error(400, "bad request"), LLMProviderError),
        (httpx.ReadTimeout("slow"), LLMTimeoutError),
        (httpx.ConnectError("no route"), LLMProviderError),
    ],
)
def test_errors_are_translated(error, expected):
    provider, _ = provider_with(error=error)
    with pytest.raises(expected):
        provider.generate([Message("user", "q")])


def test_missing_api_key_is_a_configuration_error():
    with pytest.raises(LLMConfigurationError, match="GOOGLE_API_KEY"):
        GoogleGemmaProvider(api_key=None)


def test_check_fetches_the_configured_model():
    provider, models = provider_with()
    provider.check()
    assert models.calls == [{"model": "gemma-4-26b-a4b-it"}]


# --- Settings ----------------------------------------------------------------


def test_settings_defaults_and_overrides():
    assert LLMSettings.from_env({}).model == "gemma-4-26b-a4b-it"
    assert LLMSettings.from_env({}).api_key is None
    settings = LLMSettings.from_env(
        {"GOOGLE_API_KEY": " k ", "GEMMA_MODEL": "gemma-4-31b-it", "GEMMA_TEMPERATURE": "0.5", "GEMMA_TIMEOUT_SECONDS": "5"}
    )
    assert (settings.api_key, settings.model) == ("k", "gemma-4-31b-it")
    assert (settings.generation.temperature, settings.generation.timeout_seconds) == (0.5, 5.0)


def test_settings_never_print_the_key():
    assert "sekret" not in repr(LLMSettings.from_env({"GOOGLE_API_KEY": "sekret"}))


@pytest.mark.parametrize(
    "env", [{"GEMMA_TEMPERATURE": "hot"}, {"GEMMA_TEMPERATURE": "3"}, {"GEMMA_MAX_OUTPUT_TOKENS": "0"}]
)
def test_settings_reject_invalid_values(env):
    with pytest.raises(LLMConfigurationError):
        LLMSettings.from_env(env)


# --- RAG prompt --------------------------------------------------------------


def test_rag_message_contains_passages_ids_and_metadata():
    message = build_rag_user_message(
        " What is the refund window? ",
        [Passage("doc-1", "Refunds within 30 days.", {"title": 'The "Policy"', "page": 4, "url": None})],
    )
    assert '<passage id="doc-1" title="The \'Policy\'" page="4">' in message
    assert "Refunds within 30 days." in message
    assert message.endswith("Question: What is the refund window?")


def test_rag_message_without_passages_says_so():
    assert "(no passages were retrieved)" in build_rag_user_message("q", [])


def test_system_prompt_requires_evidence_citations_and_insufficiency_statement():
    assert "Answer only from the passages" in RAG_SYSTEM_PROMPT
    assert "square brackets" in RAG_SYSTEM_PROMPT
    assert INSUFFICIENT_EVIDENCE in RAG_SYSTEM_PROMPT


# --- Conversation history + service ------------------------------------------


class FakeProvider:
    model = "fake"

    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def generate(self, messages, *, system=None, config=None, tools=None):
        self.calls.append({"messages": list(messages), "system": system, "tools": tools})
        return self.responses.pop(0)

    def check(self):
        pass


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "app.db")
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY, conversation_id TEXT, role TEXT, content TEXT, created_at TEXT)"
    )
    rows = [
        ("c1", "user", "first", "2026-01-01T00:00:01"),
        ("c1", "assistant", "second", "2026-01-01T00:00:02"),
        ("c2", "user", "other conversation", "2026-01-01T00:00:03"),
        ("c1", "system", "internal note", "2026-01-01T00:00:04"),
        ("c1", "user", "third", "2026-01-01T00:00:05"),
        ("c1", "assistant", "fourth", "2026-01-01T00:00:06"),
    ]
    connection.executemany(
        "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)", rows
    )
    connection.commit()
    connection.close()
    return path


def test_sqlite_store_loads_one_conversation_oldest_first(db_path):
    messages = SQLiteMessageStore(db_path).load_messages("c1", limit=10)
    assert [(m.role, m.content) for m in messages] == [
        ("user", "first"),
        ("assistant", "second"),
        ("user", "third"),
        ("assistant", "fourth"),
    ]


def test_sqlite_store_rejects_unsafe_table_name():
    with pytest.raises(ValueError):
        SQLiteMessageStore("x.db", table="messages; DROP TABLE x")


def test_assistant_sends_history_then_evidence_grounded_question(db_path):
    provider = FakeProvider([LLMResponse(text="30 days [doc-1]", model="fake")])
    assistant = KnowledgeAssistant(provider, SQLiteMessageStore(db_path), history_limit=3)

    result = assistant.answer("c1", "How long?", [Passage("doc-1", "Refunds within 30 days.")])

    sent = provider.calls[0]
    # Limit 3 loads [assistant, user, assistant]; the leading assistant turn is dropped.
    assert [(m.role, m.content) for m in sent["messages"][:-1]] == [("user", "third"), ("assistant", "fourth")]
    assert sent["messages"][-1].role == "user"
    assert '<passage id="doc-1">' in sent["messages"][-1].content
    assert sent["system"] == RAG_SYSTEM_PROMPT
    assert sent["tools"] is None
    assert result.text == "30 days [doc-1]"


# --- Tools -------------------------------------------------------------------


@pytest.fixture
def registry():
    registry = ToolRegistry()
    registry.register(
        "lookup",
        "Look up a term",
        {
            "type": "object",
            "properties": {"term": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["term"],
        },
        lambda term, limit=1: {"term": term, "limit": limit},
    )
    return registry


def test_tool_execution_returns_handler_result(registry):
    assert registry.execute(ToolCall("lookup", {"term": "rag", "limit": 2})) == '{"term": "rag", "limit": 2}'


@pytest.mark.parametrize(
    "call, fragment",
    [
        (ToolCall("delete_everything", {}), "Unknown tool"),
        (ToolCall("lookup", {}), "missing required"),
        (ToolCall("lookup", {"term": "x", "extra": 1}), "unexpected argument"),
        (ToolCall("lookup", {"term": 5}), "must be of type string"),
        (ToolCall("lookup", {"term": "x", "limit": True}), "must be of type integer"),
    ],
)
def test_invalid_tool_calls_are_rejected_before_execution(registry, call, fragment):
    result = registry.execute(call)
    assert result.startswith("ERROR:") and fragment in result


def test_tool_loop_executes_call_and_feeds_result_back(registry):
    provider = FakeProvider(
        [
            LLMResponse(text="", model="fake", tool_calls=(ToolCall("lookup", {"term": "rag"}),)),
            LLMResponse(text="done", model="fake"),
        ]
    )
    result = run_with_tools(provider, [Message("user", "define rag")], registry)

    assert result.text == "done"
    followup = provider.calls[1]["messages"]
    assert [m.role for m in followup] == ["user", "assistant", "user"]
    assert "[tool call] lookup" in followup[1].content
    assert '[tool result: lookup]\n{"term": "rag", "limit": 1}' in followup[2].content


def test_tool_loop_stops_after_max_rounds(registry):
    looping = LLMResponse(text="", model="fake", tool_calls=(ToolCall("lookup", {"term": "x"}),))
    provider = FakeProvider([looping, looping, LLMResponse(text="final", model="fake")])
    result = run_with_tools(provider, [Message("user", "q")], registry, max_rounds=2)

    assert result.text == "final"
    assert provider.calls[-1]["tools"] is None


# --- Health ------------------------------------------------------------------


def test_health_reports_missing_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    result = check_health()
    assert result["status"] == "error"
    assert result["api_key_configured"] is False


def test_health_ok_without_network_and_never_returns_the_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "sekret-value")
    monkeypatch.setenv("GEMMA_MODEL", "gemma-4-31b-it")
    result = check_health()
    assert result["status"] == "ok"
    assert result["model"] == "gemma-4-31b-it"
    assert result["live_check"] == "skipped"
    assert "sekret-value" not in str(result)


def test_health_live_check_pass_and_fail(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    ok, _ = provider_with()
    assert check_health(live=True, provider=ok)["live_check"] == "passed"

    bad, _ = provider_with(error=api_error(404, "not found"))
    result = check_health(live=True, provider=bad)
    assert (result["status"], result["live_check"]) == ("error", "failed")
    assert "GEMMA_MODEL" in result["error"]


def test_health_reports_invalid_configuration(monkeypatch):
    monkeypatch.setenv("GEMMA_TEMPERATURE", "hot")
    assert check_health()["status"] == "error"
