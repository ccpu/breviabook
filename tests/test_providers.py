"""litellm-backed providers via an injected completer (no network, no litellm)."""

from __future__ import annotations

from typing import Any

import pytest

from breviabook.llm.key_pool import KeyPool
from breviabook.llm.providers.gemini import GeminiProvider
from breviabook.llm.providers.openai import OpenAIProvider
from breviabook.llm.providers.openrouter import OpenRouterProvider


def _response(text: str) -> dict[str, Any]:
    return {"choices": [{"message": {"content": text}}]}


async def test_openai_provider_sends_route_key_and_returns_content() -> None:
    seen: dict[str, Any] = {}

    async def completer(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return _response("hi")

    provider = OpenAIProvider(KeyPool(["k1"]), completer=completer)
    out = await provider.generate([{"role": "user", "content": "x"}], "gpt-4o")
    assert out == "hi"
    assert seen["model"] == "openai/gpt-4o"
    assert seen["api_key"] == "k1"
    assert seen["api_base"] is None


async def test_openai_compatible_base_url() -> None:
    seen: dict[str, Any] = {}

    async def completer(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return _response("ok")

    provider = OpenAIProvider(
        KeyPool(["EMPTY"]), base_url="http://localhost:1234/v1", completer=completer
    )
    await provider.generate([{"role": "user", "content": "x"}], "local-model")
    assert seen["api_base"] == "http://localhost:1234/v1"


async def test_generate_with_image_sends_data_uri() -> None:
    seen: dict[str, Any] = {}

    async def completer(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return _response("a diagram")

    provider = OpenAIProvider(KeyPool(["k"]), completer=completer)
    out = await provider.generate_with_image("describe", [(b"\x89PNG", "image/png")], "gpt-4o")
    assert out == "a diagram"
    content = seen["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "describe"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


async def test_gemini_and_openrouter_routes() -> None:
    routes: list[str] = []

    async def completer(**kwargs: Any) -> dict[str, Any]:
        routes.append(kwargs["model"])
        return _response("ok")

    await GeminiProvider(KeyPool(["g"]), completer=completer).generate(
        [{"role": "user", "content": "x"}], "gemini-2.0"
    )
    await OpenRouterProvider(KeyPool(["o"]), completer=completer).generate(
        [{"role": "user", "content": "x"}], "meta/llama"
    )
    assert routes == ["gemini/gemini-2.0", "openrouter/meta/llama"]


async def test_provider_rotates_key_on_rate_limit() -> None:
    class RateLimitError(Exception):
        pass

    keys_used: list[str | None] = []
    calls = {"n": 0}

    async def completer(**kwargs: Any) -> dict[str, Any]:
        keys_used.append(kwargs["api_key"])
        calls["n"] += 1
        if calls["n"] == 1:
            raise RateLimitError
        return _response("recovered")

    provider = OpenAIProvider(KeyPool(["k1", "k2"]), completer=completer, max_retries=2)
    out = await provider.generate([{"role": "user", "content": "x"}], "gpt-4o")
    assert out == "recovered"
    assert keys_used == ["k1", "k2"]  # rotated to the second key after the rate-limit error


# --------------------------------------------------------------------------- #
# Thinking-budget renegotiation (breviabook.llm.reasoning)
#
# The engine asks reasoning models for no thinking; some models refuse that level
# outright (Gemini 3.8 dropped MINIMAL). A refusal must cost the run a warning, not
# the run itself. Test doubles are named after litellm's exceptions because the
# matcher works off the class-name MRO, not an import.
# --------------------------------------------------------------------------- #


class BadRequestError(Exception):
    pass


def _refuses(*efforts: str, response: str = "ok") -> tuple[Any, list[Any]]:
    """A completer that 400s while reasoning_effort is one of ``efforts``."""
    seen: list[Any] = []

    async def completer(**kwargs: Any) -> dict[str, Any]:
        seen.append(kwargs.get("reasoning_effort", "<unset>"))
        if kwargs.get("reasoning_effort") in efforts:
            raise BadRequestError(
                "GeminiException BadRequestError - Thinking level MINIMAL is not "
                "supported for this model. Please retry with other thinking level."
            )
        return _response(response)

    return completer, seen


async def test_refused_thinking_budget_falls_back_to_the_cheapest_allowed() -> None:
    completer, seen = _refuses("disable")
    provider = GeminiProvider(KeyPool(["k1"]), completer=completer)
    provider.extra_opts["reasoning_effort"] = "disable"

    out = await provider.generate([{"role": "user", "content": "x"}], "gemini-3.8-flash")

    assert out == "ok"
    assert seen == ["disable", "low"]
    # The whole run is retuned by one rejection, not just the call that hit it.
    assert provider.extra_opts["reasoning_effort"] == "low"
    assert len(provider.warnings) == 1
    assert "does not support reasoning effort 'disable'" in provider.warnings[0]


async def test_model_refusing_every_reduced_budget_drops_the_parameter() -> None:
    completer, seen = _refuses("disable", "low")
    provider = GeminiProvider(KeyPool(["k1"]), completer=completer)
    provider.extra_opts["reasoning_effort"] = "disable"

    out = await provider.generate([{"role": "user", "content": "x"}], "gemini-3.8-flash")

    assert out == "ok"
    assert seen == ["disable", "low", "<unset>"]
    assert "reasoning_effort" not in provider.extra_opts
    assert "refuses every reduced thinking budget" in provider.warnings[-1]


async def test_warning_is_recorded_once_across_calls() -> None:
    completer, _ = _refuses("disable")
    provider = GeminiProvider(KeyPool(["k1"]), completer=completer)
    provider.extra_opts["reasoning_effort"] = "disable"

    for _ in range(3):
        await provider.generate([{"role": "user", "content": "x"}], "gemini-3.8-flash")

    assert len(provider.warnings) == 1


async def test_unrelated_bad_request_still_propagates() -> None:
    async def completer(**kwargs: Any) -> dict[str, Any]:
        raise BadRequestError("GeminiException BadRequestError - context length exceeded")

    provider = GeminiProvider(KeyPool(["k1"]), completer=completer)
    provider.extra_opts["reasoning_effort"] = "disable"

    with pytest.raises(BadRequestError):
        await provider.generate([{"role": "user", "content": "x"}], "gemini-3.8-flash")
    assert provider.extra_opts["reasoning_effort"] == "disable"
    assert provider.warnings == []


async def test_per_call_effort_is_the_callers_business_and_is_not_retuned() -> None:
    completer, seen = _refuses("disable")
    provider = GeminiProvider(KeyPool(["k1"]), completer=completer)

    with pytest.raises(BadRequestError):
        await provider.generate(
            [{"role": "user", "content": "x"}], "gemini-3.8-flash", reasoning_effort="disable"
        )
    assert seen == ["disable"]
    assert provider.warnings == []


async def test_provider_without_a_budget_is_untouched() -> None:
    completer, seen = _refuses("disable")
    provider = GeminiProvider(KeyPool(["k1"]), completer=completer)

    out = await provider.generate([{"role": "user", "content": "x"}], "gemini-2.5-flash")

    assert out == "ok"
    assert seen == ["<unset>"]
    assert provider.warnings == []
