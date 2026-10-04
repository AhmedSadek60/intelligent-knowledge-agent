"""Health and configuration check for the LLM integration.

``check_health()`` returns a JSON-serialisable dict that is safe to expose from
a server endpoint: it reports whether the key is configured, never the key.

Run from the command line:
    python -m app.llm.health          # configuration only, no network
    python -m app.llm.health --live   # also verifies the key and model with Google
"""

from __future__ import annotations

import json
import sys
from typing import Any

from .base import LLMError, LLMProvider
from .config import LLMSettings


def check_health(live: bool = False, provider: LLMProvider | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "ok", "provider": "google-gemini-api", "live_check": "skipped"}
    try:
        settings = LLMSettings.from_env()
    except LLMError as exc:
        return {**result, "status": "error", "error": str(exc)}

    result.update(
        model=settings.model,
        api_key_configured=settings.api_key is not None,
        temperature=settings.generation.temperature,
        max_output_tokens=settings.generation.max_output_tokens,
        timeout_seconds=settings.generation.timeout_seconds,
        history_limit=settings.history_limit,
    )
    if settings.api_key is None:
        return {**result, "status": "error", "error": "GOOGLE_API_KEY is not set."}

    if live:
        try:
            if provider is None:
                from . import create_provider

                provider = create_provider(settings)
            provider.check()
            result["live_check"] = "passed"
        except LLMError as exc:
            return {**result, "status": "error", "live_check": "failed", "error": str(exc)}
    return result


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    result = check_health(live="--live" in argv)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
