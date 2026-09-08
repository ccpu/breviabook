"""Thinking-budget negotiation for reasoning models (ROADMAP §7.4).

BreviaBook's work is rewriting, not puzzle-solving: a reasoning model's thinking
tokens are billed as output and buy nothing here, so the engine asks for **no
thinking** by default on providers that think unprompted (Gemini).

Not every model accepts that. Gemini 3.8 dropped the ``MINIMAL`` thinking level
that ``reasoning_effort="disable"`` maps to, and rejects it with a 400 —
``Thinking level MINIMAL is not supported for this model``. Which models refuse is
not knowable before a call: the list changes with every release, and a hardcoded
table of model names would be stale within weeks and would also fail closed for a
model nobody has added to it yet.

So the engine asks for none, and on a refusal steps **up** to the cheapest budget
the model does accept, once, warning as it goes — a run costs a little more rather
than dying at the first call, and the user is told why. ``--reasoning-effort``
overrides the whole negotiation.
"""

from __future__ import annotations

# Accepted values for ``--reasoning-effort``. "auto" is not a budget — it means
# "don't send one", leaving the provider's native default in place.
VALID_EFFORTS = ("auto", "disable", "low", "medium", "high")

# Where to go when a model refuses a budget. "disable" (no thinking at all) is the
# cheapest and the default; "low" is the next cheapest. Past that we stop guessing
# and drop the parameter: a model that rejects "low" is telling us it wants to pick,
# and stepping to "medium"/"high" would spend the user's money on a guess.
_FALLBACK = {"disable": "low"}

_ERROR_CLASSES = {"BadRequestError", "UnsupportedParamsError"}
_SUBJECT_MARKERS = ("thinking", "reasoning")
_REFUSAL_MARKERS = ("not supported", "unsupported", "does not support", "not available")


def is_unsupported_effort_error(exc: BaseException) -> bool:
    """True when ``exc`` is a provider rejecting the thinking budget we asked for.

    Matched on the exception's class-name MRO plus the message, so litellm's
    exceptions and lightweight test doubles are both recognized without importing
    litellm here (same approach as :mod:`breviabook.llm.rate_limit`). Deliberately
    narrow: an unrelated 400 must keep propagating, not silently retune the run.
    """
    if not {cls.__name__ for cls in type(exc).__mro__} & _ERROR_CLASSES:
        return False
    message = str(exc).lower()
    return any(s in message for s in _SUBJECT_MARKERS) and any(
        r in message for r in _REFUSAL_MARKERS
    )


def next_effort(current: str) -> str | None:
    """The next budget to try after ``current`` was refused, or ``None`` to send none."""
    return _FALLBACK.get(current)


def validate_effort(effort: str) -> str:
    """Return ``effort`` if it is a valid ``--reasoning-effort`` value, else raise.

    Raises:
        ValueError: naming the accepted values.
    """
    if effort not in VALID_EFFORTS:
        raise ValueError(
            f"Invalid reasoning effort {effort!r}. Expected one of: {', '.join(VALID_EFFORTS)}."
        )
    return effort
