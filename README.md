# Intelligent Knowledge Agent

## LLM integration (Gemma 4 on Google's hosted Gemini API)

The assistant generates answers with Gemma 4 through the Gemini API, using the
`google-genai` Python SDK. All model code lives in `app/llm/` and runs on the
server only.

| File | Responsibility |
| --- | --- |
| `base.py` | `LLMProvider` interface, message/response types, error classes. No Google imports. |
| `gemma.py` | `GoogleGemmaProvider`, the Gemini API adapter. |
| `config.py` | `LLMSettings.from_env()`: reads and validates environment variables. |
| `prompts.py` | RAG system prompt and evidence formatting. |
| `conversation.py` | `MessageStore` protocol and a read-only SQLite loader for persisted messages. |
| `service.py` | `KnowledgeAssistant`: history + passages + question → model call. |
| `tools.py` | Application-side tool validation, execution and tool loop. |
| `health.py` | Health/configuration check (function and CLI). |

Model calls are independent of document retrieval, vector search, memory, Tavily
and citation validation. The caller retrieves passages, passes them in, and
validates the citations in the returned text.

### Setup

Requires Python 3.10+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
Copy-Item .env.example .env         # macOS/Linux: cp .env.example .env
```

Create an API key at <https://aistudio.google.com/apikey> and put it in `.env`:

```
GOOGLE_API_KEY=your-key
GEMMA_MODEL=gemma-4-26b-a4b-it
```

`.env` is git-ignored. In deployed environments set these as server environment
variables or secrets instead of using a file. The key is read only on the
server and is never returned by any function in this package, so do not forward
it to the browser.

| Variable | Default | Meaning |
| --- | --- | --- |
| `GOOGLE_API_KEY` | (required) | Gemini API key. |
| `GEMMA_MODEL` | `gemma-4-26b-a4b-it` | Hosted model ID. `gemma-4-31b-it` is the other hosted Gemma 4 model. |
| `GEMMA_TEMPERATURE` | `0.2` | 0 to 2. |
| `GEMMA_MAX_OUTPUT_TOKENS` | `2048` | Output-token limit per response. |
| `GEMMA_TIMEOUT_SECONDS` | `60` | Per-request timeout. |
| `GEMMA_HISTORY_LIMIT` | `20` | Number of persisted messages sent as conversation context. |

### Check the configuration

```powershell
python -m app.llm.health          # configuration only, no network call
python -m app.llm.health --live   # also verifies the key and model ID with Google
```

Both print JSON and exit with code 0 when healthy, 1 otherwise. The CLI loads
`.env`; the library itself only reads `os.environ`, so your server must load
`.env` or have the variables set.

To expose it from the server, return `check_health()` from a route:

```python
from app.llm.health import check_health

@app.get("/health/llm")
def llm_health():
    return check_health(live=False)
```

### Run the tests

```powershell
python -m pytest -q
```

The tests mock the Google client. They need no API key and make no network calls.

### Usage

```python
from app.llm import KnowledgeAssistant, Passage, SQLiteMessageStore, create_provider

provider = create_provider()                      # reads GOOGLE_API_KEY / GEMMA_MODEL
store = SQLiteMessageStore("path/to/app.db")      # or your own MessageStore
assistant = KnowledgeAssistant(provider, store)

passages = [                                      # produced by your retriever
    Passage("doc-12", "Refunds are accepted within 30 days.", {"title": "Refund policy", "page": 3}),
]
response = assistant.answer(conversation_id="c1", question="How long do I have to return an item?", passages=passages)
print(response.text)        # e.g. "You have 30 days [doc-12]."
```

The model is instructed to answer only from the supplied passages, cite source
IDs in square brackets, and reply with the sentence in
`app.llm.INSUFFICIENT_EVIDENCE` when the passages do not contain the answer.
Your application can test for that sentence.

`KnowledgeAssistant` does not write to the database. Persist the user question
and the answer in your own code after the call.

**Conversation history.** `SQLiteMessageStore` reads an existing table with the
columns `id, conversation_id, role, content, created_at`, where `role` is
`user` or `assistant`. It never creates or alters schema. If your database uses
a different engine or schema, implement the one-method `MessageStore` protocol:

```python
class MyStore:
    def load_messages(self, conversation_id: str, limit: int) -> list[Message]:
        ...  # most recent `limit` messages, oldest first
```

**Plain generation** without RAG:

```python
from app.llm import GenerationConfig, Message, create_provider

provider = create_provider()
response = provider.generate(
    [Message("user", "Say hello in one sentence.")],
    config=GenerationConfig(temperature=0.0, max_output_tokens=256, timeout_seconds=30),
)
```

### Tool calling

Checked against Google's documentation (October 2026):

- The hosted Gemma 4 models support system instructions and native function
  calling through the Gemini API. This is the Gemini API's own format, not the
  OpenAI one.
- The API returns function calls as structured parts and never executes them.
  Execution and validation happen in this application, in `app/llm/tools.py`.
- Structured output (`response_schema`) is not documented for Gemma 4 on the
  Gemini API page, so this integration does not use it.

```python
from app.llm import Message, ToolRegistry, create_provider, run_with_tools

registry = ToolRegistry()
registry.register(
    "web_search",
    "Search the web for recent information.",
    {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    handler=my_tavily_search,          # your function; this package does not import Tavily
)
response = run_with_tools(create_provider(), [Message("user", "What changed this week?")], registry)
```

`ToolRegistry` rejects unknown tools, missing or unexpected arguments and wrong
argument types before the handler runs, and returns the failure to the model as
text. `run_with_tools` stops after `max_rounds` (default 3).

### Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `LLMConfigurationError: GOOGLE_API_KEY is not set` | The server process has no key. Set it in `.env` (and load `.env`) or in the host's environment/secrets. |
| `LLMAuthError` (400 "API key not valid", 401, 403) | The key is wrong, revoked, or restricted. Create a new one in AI Studio. |
| `LLMConfigurationError: Model ... was not found` | `GEMMA_MODEL` is not a hosted model ID. Use `gemma-4-26b-a4b-it` or `gemma-4-31b-it`. |
| `LLMRateLimitError` (429) | Quota or rate limit reached. Wait and retry, or check the limits for your key in AI Studio. |
| `LLMTimeoutError` | The request exceeded `GEMMA_TIMEOUT_SECONDS`. Raise it, or send fewer/shorter passages. |
| `LLMResponseError: ... no text (finish_reason=MAX_TOKENS)` | The token limit was used up before any answer text, which can happen when the model spends it on reasoning. Raise `GEMMA_MAX_OUTPUT_TOKENS`. |
| `LLMResponseError: Prompt was blocked` | Google's safety filter blocked the prompt. Inspect the question and the retrieved passages. |
| `LLMProviderError: Network error` | The server cannot reach `generativelanguage.googleapis.com`. Allow that host in the firewall, proxy or sandbox network settings. |
| A different key is being used | The adapter passes `GOOGLE_API_KEY` explicitly and ignores `GEMINI_API_KEY`. |
