"""Base for litellm-backed providers (OpenAI, Gemini, OpenRouter) — ROADMAP §7.4.

Each provider differs only by its litellm route prefix (``openai/``, ``gemini/``,
``openrouter/``) and an optional ``base_url`` (for OpenAI-compatible endpoints). Calls go
through key rotation. litellm is lazy-imported (kept out of import time / test paths); tests
inject a ``completer`` instead.
"""

from __future__ import annotations

import base64
from collections.abc import Awaitable, Callable
from typing import Any

from breviabook.llm.base import Message
from breviabook.llm.key_pool import KeyPool
from breviabook.llm.rate_limit import with_key_rotation
from breviabook.llm.reasoning import is_unsupported_effort_error, next_effort
from breviabook.llm.usage import Usage, extract_usage

Completer = Callable[..., Awaitable[Any]]


def _extract_content(response: Any) -> str:
    content = response["choices"][0]["message"]["content"]
    return str(content) if content is not None else ""


def _data_uri(mime: str, data: bytes) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime or 'application/octet-stream'};base64,{encoded}"


def completion_cost(response: Any) -> float:
    """Best-effort USD cost for a litellm response; 0.0 if the model price is unknown."""
    try:
        import litellm

        return float(litellm.completion_cost(completion_response=response) or 0.0)
    except Exception:
        return 0.0


class LiteLLMProvider:
    """An :class:`~breviabook.llm.base.LLMProvider` backed by litellm with key rotation."""

    def __init__(
        self,
        *,
        name: str,
        route: str,
        pool: KeyPool,
        base_url: str | None = None,
        max_retries: int = 3,
        completer: Completer | None = None,
    ) -> None:
        self.name = name
        self.route = route
        self.pool = pool
        self.base_url = base_url
        self.max_retries = max_retries
        self._completer = completer
        self.usage = Usage()
        # Default options merged into every call (e.g. reasoning_effort to control thinking).
        # Per-call opts passed to generate() override these.
        self.extra_opts: dict[str, object] = {}
        # Non-fatal things the run should tell the user about afterwards — currently only a
        # thinking budget the model refused. Drained by the pipeline into its own warnings.
        self.warnings: list[str] = []

    async def _acompletion(self, **kwargs: Any) -> Any:
        if self._completer is not None:
            return await self._completer(**kwargs)
        import litellm

        return await litellm.acompletion(**kwargs)

    async def _attempt(self, messages: list[Any], model: str, opts: dict[str, object]) -> str:
        async def call(key: str | None) -> str:
            response = await self._acompletion(
                model=f"{self.route}/{model}",
                messages=messages,
                api_key=key,
                api_base=self.base_url,
                **opts,
            )
            prompt, completion, cached = extract_usage(response)
            cost = completion_cost(response) if self._completer is None else 0.0
            self.usage.add(prompt, completion, cached, cost)
            return _extract_content(response)

        return await with_key_rotation(self.pool, call, max_retries=self.max_retries)

    async def _run(self, messages: list[Any], model: str, **opts: object) -> str:
        while True:
            merged: dict[str, object] = {**self.extra_opts, **opts}
            asked = merged.get("reasoning_effort")
            try:
                return await self._attempt(messages, model, merged)
            except Exception as exc:
                # A per-call reasoning_effort is the caller's own choice — don't second-guess
                # it. Only the run-wide default set by the factory is ours to renegotiate.
                if "reasoning_effort" in opts or not self._retune_effort(asked, exc):
                    raise

    def _retune_effort(self, asked: object, exc: BaseException) -> bool:
        """Step the run's thinking budget up to the cheapest one this model accepts.

        Returns True when the caller should retry the call. Mutates ``extra_opts`` so the
        whole run is fixed by one rejected request, not just this call.
        """
        if asked is None or not is_unsupported_effort_error(exc):
            return False
        if self.extra_opts.get("reasoning_effort") != asked:
            # A concurrent call already retuned it. Retry with whatever it settled on
            # rather than stepping the ladder a second time for the same rejection.
            return True

        fallback = next_effort(str(asked))
        if fallback is None:
            del self.extra_opts["reasoning_effort"]
            self._warn_once(
                f"{self.name}: this model refuses every reduced thinking budget "
                f"(asked for {asked!r}) — using the model's own default, which bills its "
                "reasoning as output tokens. Pass --reasoning-effort to set one yourself."
            )
        else:
            self.extra_opts["reasoning_effort"] = fallback
            self._warn_once(
                f"{self.name}: this model does not support reasoning effort {asked!r}; "
                f"the run fell back to {fallback!r}, which costs more than no thinking. "
                "Pass --reasoning-effort to choose another."
            )
        return True

    def _warn_once(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    async def generate(self, messages: list[Message], model: str, **opts: object) -> str:
        return await self._run(list(messages), model, **opts)

    async def generate_with_image(
        self, prompt: str, images: list[tuple[bytes, str]], model: str, **opts: object
    ) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for data, mime in images:
            content.append({"type": "image_url", "image_url": {"url": _data_uri(mime, data)}})
        return await self._run([{"role": "user", "content": content}], model, **opts)
