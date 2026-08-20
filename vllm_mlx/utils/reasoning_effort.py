# SPDX-License-Identifier: Apache-2.0
"""Per-template vocabulary normalization for ``reasoning_effort``.

The OpenAI ``reasoning_effort`` request parameter has no single vocabulary:

* gpt-oss / harmony accept ``low | medium | high``
  (:mod:`vllm_mlx.utils.harmony_render`).
* Qwen3.8's chat template accepts ``xhigh | medium | low`` (default ``xhigh``)
  and calls ``raise_exception()`` on anything else -- so a client sending the
  OpenAI-default ``high`` (Claude Code 2.1.235 does, on every request) turns
  into an HTTP 500 if the value is forwarded verbatim.

This module resolves both problems with one rule: **never let an unsupported
value reach the template.** The accepted-value tuple is introspected out of the
Jinja source once per template (cached), and the request value is resolved
against it in this order — see :func:`normalize_reasoning_effort`:

    exact match -> the route's configured floor -> nearest neighbour -> drop

The floor rung is the important one and is vLLM-aligned: no engine invents a
level for a request it can't honour, so the operator's
``--default-chat-template-kwargs`` decides. Without it an unsupported value
would land on the *template's* default, which on Qwen3.8 is ``xhigh`` — meaning
a garbage effort would buy MORE thinking than sending nothing at all, and
Claude Code's ``high`` would resolve to exactly the runaway-thinking mode this
patch exists to prevent.

:func:`render_with_effort_fallback` is the backstop for templates whose
vocabulary we could not parse: render, and if that raises, retry down the same
ladder (floor, then no effort at all) and warn once per model.

How the field does it
---------------------
* **llama.cpp** (``tools/server``): ``none`` disables thinking, "otherwise the
  value is made available to the jinja template" — verbatim, unvalidated. That
  is precisely the configuration that HTTP-500s on Qwen3.8 when a client sends
  the OpenAI default ``high``.
* **vLLM**: maps ``reasoning_effort`` onto an ``enable_thinking`` **bool**
  (low/medium/high -> true, none -> false), explicit ``chat_template_kwargs``
  taking priority, and then filters every template kwarg through
  ``resolve_chat_template_kwargs`` — the intersection of
  ``jinja2.meta.find_undeclared_variables(template)`` with the
  ``apply_chat_template`` signature. That is **variable**-level filtering.
* **SGLang**: server-wide ``--chat-template-kwargs`` plus per-request
  ``chat_template_kwargs``; no value validation either.

So the presence test here is vLLM's (``jinja2.meta``, the de-facto standard,
with a guard against its known empty-parse bug — vllm-project/vllm#36907), and
the **value**-vocabulary normalization on top of it is ours: no upstream engine
does it, because none of them has a fleet whose default model raises on the
OpenAI default value.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

#: Every effort level the fork knows about, across all template families.
KNOWN_EFFORT_LEVELS: frozenset[str] = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh"}
)

#: Vocabulary of the openai-harmony renderer (``ReasoningEffort`` members).
HARMONY_EFFORT_LEVELS: frozenset[str] = frozenset({"low", "medium", "high"})

#: Ordinal neighbours to try when the requested level is not in the template's
#: vocabulary. Only unambiguous substitutions are listed: ``high`` and ``xhigh``
#: are the same end of the scale, as are ``low`` and ``minimal``. ``medium`` has
#: no unambiguous neighbour, so a template without it gets the kwarg dropped
#: rather than a coin-flip between low and high.
_EFFORT_NEIGHBOURS: dict[str, tuple[str, ...]] = {
    "minimal": ("low",),
    "low": ("minimal",),
    "medium": (),
    "high": ("xhigh",),
    "xhigh": ("high",),
    "none": ("minimal", "low"),
}

#: Quoted string literals in the Jinja source (``'xhigh'`` / ``"xhigh"``).
_LITERAL_RE = re.compile(r"""['"]([A-Za-z_][A-Za-z0-9_-]*)['"]""")

#: How far around each ``reasoning_effort`` mention to look for the vocabulary.
#: Qwen3.8 declares it on the line after the ``|default(...)`` set, and repeats
#: it inside the ``raise_exception`` message on the line after that.
_WINDOW_BEFORE = 200
_WINDOW_AFTER = 300

#: Models already warned about via :func:`render_with_effort_fallback`, so a
#: broken vocabulary logs once rather than once per request.
_warned_models: set[str] = set()

#: Reserved ``chat_template_kwargs`` key carrying the route's
#: ``--default-chat-template-kwargs`` effort so the engine can fall back to it
#: (see the resolution order in :func:`normalize_reasoning_effort`). The server
#: sets it; :func:`normalize_effort_in_template_kwargs` pops it. It must never
#: reach the Jinja context, hence the reserved-looking name.
EFFORT_FALLBACK_KEY = "__vllm_mlx_effort_fallback"


def _clean_level(value: Any) -> str | None:
    """Lowercase/strip a level, or ``None`` if it isn't a usable string."""
    if not isinstance(value, str):
        return None
    level = value.strip().lower()
    return level or None


def _declares_reasoning_effort(template_source: str) -> bool | None:
    """Does the template read a ``reasoning_effort`` variable?

    Uses vLLM's mechanism (``jinja2.meta.find_undeclared_variables``) rather
    than a substring match, so ``resolved_reasoning_effort = ...|default(...)``
    assignments and mentions inside string literals don't count as reads.

    Returns ``None`` when the question can't be answered — jinja2 missing, the
    source not parseable, or the parse yielding *no* variables at all (vLLM's
    #36907: a bare template *name* is valid trivial Jinja and reports an empty
    set, which would silently drop every kwarg).
    """
    try:
        from jinja2 import Environment
        from jinja2.meta import find_undeclared_variables
    except ImportError:  # pragma: no cover - jinja2 ships with transformers
        return None
    try:
        variables = find_undeclared_variables(Environment().parse(template_source))
    except Exception:  # noqa: BLE001 - malformed template is the render's problem
        return None
    if not variables:
        return None
    return "reasoning_effort" in variables


@lru_cache(maxsize=32)
def template_effort_vocabulary(template_source: str) -> frozenset[str] | None:
    """Return the effort levels a chat template accepts, or ``None`` if unknown.

    ``None`` means "could not determine" -- callers should forward the value
    unchanged and rely on :func:`render_with_effort_fallback`. An **empty**
    frozenset means the template does not reference ``reasoning_effort`` at
    all, so the kwarg should be dropped.
    """
    if not template_source:
        return frozenset()

    declares = _declares_reasoning_effort(template_source)
    if declares is False:
        return frozenset()
    if declares is None:
        # Inconclusive (no jinja2, unparseable, or vLLM #36907's empty parse).
        # Say "unknown" rather than "absent": dropping here would silently
        # disable the feature for a template that may well support it, and the
        # render backstop already makes passing it through safe.
        return None

    found: set[str] = set()
    for match in re.finditer("reasoning_effort", template_source):
        start = max(0, match.start() - _WINDOW_BEFORE)
        window = template_source[start : match.end() + _WINDOW_AFTER]
        for literal in _LITERAL_RE.findall(window):
            if literal in KNOWN_EFFORT_LEVELS:
                found.add(literal)

    # A single hit is usually just ``reasoning_effort|default('medium')`` with
    # no validation list -- not enough to call it a vocabulary. Say "unknown"
    # and let the value through; the render backstop covers a bad guess.
    if len(found) < 2:
        return None
    return frozenset(found)


def strip_effort_fallback(template_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """Copy ``template_kwargs`` without the fork-internal fallback key.

    For paths that hand ``chat_template_kwargs`` straight to a third-party
    renderer (mlx-vlm's MLLM ``chat``) instead of through
    :func:`normalize_effort_in_template_kwargs`, which is what normally pops it.
    """
    return {
        k: v for k, v in (template_kwargs or {}).items() if k != EFFORT_FALLBACK_KEY
    }


def normalize_reasoning_effort(
    value: Any,
    vocabulary: Iterable[str] | None,
    fallback: Any = None,
) -> str | None:
    """Map ``value`` onto ``vocabulary``, or return ``None`` to drop the kwarg.

    Resolution order — **exact match, then the operator's floor, then the
    nearest neighbour, then drop**:

    1. ``value`` is in the vocabulary -> use it verbatim.
    2. otherwise ``fallback`` (the route's ``--default-chat-template-kwargs``
       level) if the template accepts it. This is the vLLM-aligned rule: no
       engine invents a level for an unsupported request, and the operator's
       configured default is the authority. It is what stops Claude Code's
       ``high`` from resolving to Qwen3.8's ``xhigh`` — which is that
       template's own default, i.e. the runaway-thinking mode this whole
       patch exists to avoid.
    3. otherwise the nearest unambiguous neighbour, which still serves routes
       with no configured floor (``xhigh`` -> ``high`` on gpt-oss).
    4. otherwise drop, and the template's default applies.

    Args:
        value: the requested effort level (case/whitespace insensitive).
        vocabulary: accepted levels, or ``None`` when they could not be
            determined -- in which case ``value`` passes through unchanged.
        fallback: the route's configured default level, if any.
    """
    level = _clean_level(value)
    if vocabulary is None:
        return level

    accepted = frozenset(vocabulary)
    if not accepted or level is None:
        return None
    if level in accepted:
        return level

    floor = _clean_level(fallback)
    if floor is not None and floor in accepted:
        return floor

    for neighbour in _EFFORT_NEIGHBOURS.get(level, ()):
        if neighbour in accepted:
            return neighbour
    return None


def normalize_effort_in_template_kwargs(
    template_kwargs: dict[str, Any],
    template_source: str | None,
) -> dict[str, Any]:
    """Normalize (or drop) ``template_kwargs["reasoning_effort"]`` in place.

    Also pops :data:`EFFORT_FALLBACK_KEY` — always, even when there is no
    ``reasoning_effort`` to normalize, because that key is fork-internal
    plumbing and must never be handed to the Jinja context.

    Returns the same dict for call-site convenience.
    """
    fallback = template_kwargs.pop(EFFORT_FALLBACK_KEY, None)
    if "reasoning_effort" not in template_kwargs:
        return template_kwargs

    raw = template_kwargs["reasoning_effort"]
    vocabulary = (
        template_effort_vocabulary(template_source)
        if isinstance(template_source, str)
        else None
    )
    resolved = normalize_reasoning_effort(raw, vocabulary, fallback=fallback)
    if resolved is None:
        template_kwargs.pop("reasoning_effort", None)
        logger.debug(
            "Dropping unsupported reasoning_effort=%r (template accepts %s)",
            raw,
            sorted(vocabulary) if vocabulary else "no reasoning_effort",
        )
    else:
        if resolved != raw:
            logger.debug("Normalized reasoning_effort %r -> %r", raw, resolved)
        template_kwargs["reasoning_effort"] = resolved
    return template_kwargs


def render_with_effort_fallback(
    render: Callable[..., Any],
    template_kwargs: dict[str, Any],
    *,
    model_name: str = "",
    fallback: Any = None,
) -> Any:
    """Call ``render(**template_kwargs)``; on failure retry once without effort.

    Backstop for templates whose vocabulary :func:`template_effort_vocabulary`
    could not parse. If the retry also fails the *original* exception is
    re-raised, so unrelated template breakage is never masked.

    ``fallback`` (the route's configured floor) is tried before dropping
    outright, so an unparseable template resolves the same way an introspected
    one does — see :func:`normalize_reasoning_effort`'s resolution order.

    On a successful retry ``reasoning_effort`` is **also updated in the
    caller's dict**: SimpleEngine reuses the same ``template_kwargs`` for the
    system-prefix divergence probe, and a probe rendered with kwargs the real
    prompt didn't use would derive a prefix that never matches (silent KV
    cache miss).
    """
    try:
        return render(**template_kwargs)
    except Exception as exc:
        if "reasoning_effort" not in template_kwargs:
            raise
        dropped = template_kwargs["reasoning_effort"]

        # Floor first, then no effort at all.
        attempts: list[tuple[str, dict[str, Any]]] = []
        floor = _clean_level(fallback)
        if floor is not None and floor != _clean_level(dropped):
            attempts.append((floor, {**template_kwargs, "reasoning_effort": floor}))
        without = dict(template_kwargs)
        without.pop("reasoning_effort", None)
        attempts.append(("<dropped>", without))

        for label, retry_kwargs in attempts:
            try:
                result = render(**retry_kwargs)
            except Exception:  # noqa: BLE001 - try the next rung down
                continue
            # Keep the caller's dict in step with what actually rendered.
            if "reasoning_effort" in retry_kwargs:
                template_kwargs["reasoning_effort"] = retry_kwargs["reasoning_effort"]
            else:
                template_kwargs.pop("reasoning_effort", None)
            if model_name not in _warned_models:
                _warned_models.add(model_name)
                logger.warning(
                    "Chat template for %s rejected reasoning_effort=%r (%s); "
                    "rendering with %s for this and subsequent requests. Add "
                    "the level to the template vocabulary probe if this is "
                    "unexpected.",
                    model_name or "<unknown model>",
                    dropped,
                    exc,
                    label,
                )
            return result
        raise exc from None
