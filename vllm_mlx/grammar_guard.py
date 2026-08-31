"""Stop-vs-grammar arbitration (PATCHES.md #89).

A structured-output request carries two independent terminators that know
nothing about each other:

* the schema processor (patch #73 llguidance, or the lm-format-enforcer
  fallback) which masks logits so the output *must* be a legal JSON value, and
* the stop terminators — user/parser ``stop`` strings (patch #32) and the
  repetition detector (patch #77) — which end a request on a text match.

Nothing stopped the second from firing inside the first. A pretty-printed
schema legitimately emits ``"\\n\\n"`` between members, so a client that also
sends ``stop=["\\n\\n"]`` (every OpenAI-shaped agent harness does) got a
truncated JSON object reported as ``finish_reason="stop"`` — a clean stop on
broken output, the worst of the two failure shapes. The non-streaming path
catches it after the fact with a 422 re-parse (``_apply_response_format_or_raise``);
the streaming path has already put the broken JSON on the wire.

The fix model is vLLM #49227/#50595: while a schema processor is attached and
its matcher is NOT in an accepting state, the stop terminators must not fire.
This module is the seam — pure duck typing, no imports from ``constrained/``,
so any stop path can consult it without dragging llguidance/mlx into scope.

**The protocol.** A logits processor participates by exposing a nullary
``is_accepting() -> bool``. It reports the grammar state as of the last mask
application, i.e. one token stale by construction (the processor masks step
N's logits before token N is sampled, and the stop paths inspect text that
already contains token N). The staleness is safe in the only direction that
matters for the schemas these routes serve: for an object or array schema
the matcher accepts only at the closing brace/bracket, so the token that
completes the value flips accepting False → True and a guard consulted at
that moment reports "un-terminated" — *suppressing* a stop that had nothing
left to cut, never the reverse.

That is a property of container schemas, not of JSON grammars in general: a
top-level scalar schema (``{"type": "integer"}``) accepts after ``1`` while
``12`` is still reachable, so accepting can be true mid-value and a stop
could fire there. Agent traffic sends object schemas; if scalar roots ever
matter, the fix is a "no token can extend the value" predicate, not a
staleness fix.

**Callers must not read this live off a queue.** A consumer draining
``RequestOutputCollector`` can be many tokens behind the model (and the
collector merges chunks when the producer runs ahead), so the producer
stamps its verdict onto each ``RequestOutput`` and the consumer reads the
stamp. See ``_GrammarStopGate`` in ``engine/batched.py``.

Processors deliberately cache a plain ``bool`` rather than calling into the
matcher on demand: the stop paths run on the consumer/asyncio thread while
generation mutates the matcher on the model thread, and a bool read is atomic
under the GIL where an FFI call into a ``&mut self`` Rust matcher is not.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_warned_predicate_failure = False


def _flatten(processors: Any) -> Iterator[Any]:
    """Yield individual processors from a flat OR per-sequence nested list.

    The engine and ``SamplingParams`` carry a flat list; the batched scheduler
    carries one list per sequence in the batch. Accepting both means callers
    never have to know which shape they are holding.
    """
    if processors is None:
        return
    if isinstance(processors, (list, tuple, set)):
        for entry in processors:
            yield from _flatten(entry)
        return
    yield processors


_MAX_DELEGATE_DEPTH = 8


def _resolve(processor: Any) -> Any:
    """Return the processor that can answer for this one, or ``None``.

    A schema processor can be *wrapped*: both ``ThinkingAwareLogitsProcessor``
    and ``server.py``'s ``_ThinkingAwareLogitsProcessor`` hold the real one as
    ``_inner`` and gate it behind ``</think>`` (neither composition is built
    on a live route today — ``_attach_response_format_logits_processor``
    forces ``enable_thinking=False`` whenever a grammar exists, audit bug 1 —
    but the guard must not go silently blind if that changes). A wrapper that
    answers for itself wins; otherwise descend, mirroring the ``schema`` /
    ``_disabled`` forwarding those wrappers already do.

    Descending is conservative during a wrapper's pre-delegation phase: the
    inner processor has not been called, so it reports mid-value and stops
    stay suppressed through the thinking block. That is the right answer
    anyway — under ``response_format`` a stop firing before the value is
    complete yields no JSON at all, which the non-streaming re-parse turns
    into a 422 and streaming turns into broken output on the wire.
    """
    seen = 0
    while processor is not None and seen < _MAX_DELEGATE_DEPTH:
        if callable(getattr(processor, "is_accepting", None)):
            return processor
        processor = getattr(processor, "_inner", None)
        seen += 1
    return None


def iter_grammar_processors(processors: Any) -> Iterator[Any]:
    """Yield the attached processors that answer the accepting protocol."""
    for processor in _flatten(processors):
        resolved = _resolve(processor)
        if resolved is not None:
            yield resolved


def has_grammar(processors: Any) -> bool:
    """True when a schema processor is constraining this request's output.

    The whole generated text of such a request IS the structured value: once
    the grammar reaches its accepting state only EOS is unmasked, so nothing
    can follow. Any stop-string match therefore lands *inside* the value —
    which is why the non-streaming path suppresses truncation wholesale
    instead of trying to reconstruct per-position grammar state after the
    fact.
    """
    for _ in iter_grammar_processors(processors):
        return True
    return False


def grammar_unterminated(processors: Any) -> bool:
    """True when an attached schema processor is mid-value.

    False when no schema processor is attached — every stop path stays
    byte-identical to its pre-#89 behaviour on unconstrained requests.

    A predicate that raises is treated as un-terminated: a broken matcher
    means the grammar state is unknown, and the fail-closed reading (keep
    generating, let EOS or the length cap end it) cannot manufacture a
    truncated-JSON-reported-as-clean-stop, which is the shape this patch
    exists to prevent.
    """
    global _warned_predicate_failure
    for processor in iter_grammar_processors(processors):
        try:
            if not processor.is_accepting():
                return True
        except Exception:
            if not _warned_predicate_failure:
                _warned_predicate_failure = True
                logger.warning(
                    "[grammar-guard] %s.is_accepting() raised; treating the "
                    "grammar as un-terminated (stop terminators suppressed)",
                    type(processor).__name__,
                    exc_info=True,
                )
            return True
    return False
