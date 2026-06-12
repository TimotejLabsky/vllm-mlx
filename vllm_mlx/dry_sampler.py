# SPDX-License-Identifier: Apache-2.0
"""DRY (Don't Repeat Yourself) sequence-level repetition penalty.

Fork patch: targets the block-level repetition collapse that long agentic
sessions hit on quantized models (Qwen3.6-27B-4bit at deep context emits the
same paragraph indefinitely at T=0). Token-level penalties (repetition /
presence / frequency) cannot break these loops — they penalize individual
token reuse, while the loop is a *sequence* phenomenon where every token is
locally optimal. DRY (by p-e-w, community-proven in koboldcpp/llama.cpp/
text-generation-webui; never merged into vLLM — PR #11368 died stale)
penalizes the token that would EXTEND a repeated suffix, with a penalty that
grows exponentially in the repeat length:

    penalty(z) = multiplier * base ** (match_len - allowed_length)

where ``match_len`` is the length of the longest suffix of the context that,
followed by candidate ``z``, already occurred earlier in the window. Short
incidental repeats (< allowed_length) are free; an 8-token loop at default
parameters eats a ~27-logit penalty — enough to break argmax at T=0.

Implementation notes:
- Match lengths for ALL earlier positions are computed in O(window) per step
  with a Z-array over the reversed window (the koboldcpp approach). The
  pathological case — a highly repetitive context — is exactly when naive
  per-occurrence backward scans degrade to O(window²), so the linear
  algorithm is load-bearing, not an optimization nicety.
- Sequence breakers reset matching: a match never extends across a breaker
  token, and breaker tokens themselves are never penalized (they are the
  escape hatch out of a loop). The defaults ("\\n", ":", "\\"", "*") make
  structured tool-call JSON near-immune — key/value content between quotes
  and colons is too short to clear ``allowed_length``.
- mlx-lm logits-processor contract: ``processor(tokens, logits) -> logits``
  with ``tokens`` the running history (mx.array) and ``logits`` ``[B, vocab]``.

Env defaults (per-model via llama-swap ``env:`` or globally):
    VLLM_MLX_DRY_MULTIPLIER        (default 0 = off)
    VLLM_MLX_DRY_BASE              (default 1.75)
    VLLM_MLX_DRY_ALLOWED_LENGTH    (default 2)
    VLLM_MLX_DRY_RANGE             (default 2048 — token window)
    VLLM_MLX_DRY_SEQUENCE_BREAKERS (comma-separated strings)
Request fields ``dry_*`` override the env defaults per call.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Iterable

logger = logging.getLogger(__name__)

DEFAULT_SEQUENCE_BREAKERS = ("\n", ":", '"', "*")
# Cap on the exponent so base**n stays finite in float16 logits (max 65504):
# 0.8 * 1.75**18 ≈ 1.9e4 — already far beyond any real logit scale, while
# safely under fp16 overflow even with multiplier up to ~3.
_MAX_EXPONENT = 18.0


def zarray(seq: list) -> list:
    """Standard Z-algorithm: z[i] = length of the longest common prefix of
    ``seq`` and ``seq[i:]``. O(n)."""
    n = len(seq)
    z = [0] * n
    if n == 0:
        return z
    z[0] = n
    left = right = 0
    for i in range(1, n):
        if i < right:
            z[i] = min(right - i, z[i - left])
        while i + z[i] < n and seq[z[i]] == seq[i + z[i]]:
            z[i] += 1
        if i + z[i] > right:
            left, right = i, i + z[i]
    return z


def resolve_breaker_ids(tokenizer: Any, breakers: Iterable[str]) -> set[int]:
    """Map breaker strings to token ids.

    Tokenizers merge punctuation into many surface forms, so a breaker char
    rarely owns a single id. The community approach (text-generation-webui):
    encode the breaker bare AND glued to a leading letter, collecting the
    final id of each encoding — catching both the standalone token and the
    most common merged form. Approximate by design; missing an exotic merge
    weakens the reset slightly, it never corrupts output.
    """
    ids: set[int] = set()
    encode = getattr(tokenizer, "encode", None)
    if encode is None:
        return ids
    for s in breakers:
        for probe in (s, f"a{s}"):
            try:
                toks = encode(probe, add_special_tokens=False)
            except TypeError:
                toks = encode(probe)
            if toks:
                ids.add(int(toks[-1]))
    return ids


class DRYLogitsProcessor:
    """mlx-lm-compatible logits processor implementing DRY.

    GENERATION-ONLY matching (deliberate divergence from the original DRY,
    which matches over the whole context): the processor records the token
    count of its FIRST invocation — mlx-lm applies processors with
    ``tokens == prompt`` before sampling the first token — and matches only
    within tokens generated after that point. Whole-context matching
    corrupted agentic tool calls in production: re-running a shell command
    from earlier in the conversation is a long verbatim repeat with no
    sequence breakers inside bash text, so the exponential penalty forced
    mid-command divergence ("head -20" became "headdefault-20"). Loops —
    the thing DRY exists to break — repeat within the current generation,
    so the restriction costs nothing.

    Stateful per request: construct a fresh instance per generation (the
    engine does).
    """

    def __init__(
        self,
        multiplier: float,
        base: float = 1.75,
        allowed_length: int = 2,
        window: int = 2048,
        breaker_ids: set[int] | None = None,
    ) -> None:
        self.multiplier = float(multiplier)
        self.base = max(1.01, float(base))
        self.allowed_length = max(1, int(allowed_length))
        self.window = max(8, int(window))
        self.breaker_ids = breaker_ids or set()
        self._gen_start: int | None = None

    def __call__(self, tokens: Any, logits: Any):
        import mlx.core as mx

        if self.multiplier <= 0:
            return logits
        n_total = len(tokens)
        if self._gen_start is None:
            # First invocation: tokens == the prompt. Everything after this
            # boundary is generated output — the only region we match in.
            self._gen_start = n_total
            return logits
        gen_len = n_total - self._gen_start
        if gen_len < 1:
            return logits
        span = min(gen_len, self.window)
        try:
            seq = tokens[-span:].tolist()
        except Exception:
            seq = list(tokens)[-span:]
        n = len(seq)
        if n < self.allowed_length + 1:
            return logits
        if seq[-1] in self.breaker_ids:
            # Any suffix match would end on a breaker — nothing to extend.
            return logits

        # run[p] = length of the breaker-free run ending at position p;
        # caps match lengths so repeats never span a structural boundary.
        run = [0] * n
        acc = 0
        for p, t in enumerate(seq):
            acc = 0 if t in self.breaker_ids else acc + 1
            run[p] = acc

        # z over the reversed window: z_rev[i] = common-suffix length between
        # the window's tail and the subsequence ending at original position
        # n-1-i.
        z_rev = zarray(seq[::-1])
        tail_run = run[n - 1]

        best: dict[int, int] = {}
        for i in range(1, n - 1):
            e = n - 1 - i  # earlier match END position (candidate at e+1)
            match_len = z_rev[i]
            if match_len <= 0:
                continue
            if match_len > tail_run:
                match_len = tail_run
            if match_len > run[e]:
                match_len = run[e]
            if match_len < self.allowed_length:
                continue
            cand = seq[e + 1]
            if cand in self.breaker_ids:
                continue
            prev = best.get(cand, 0)
            if match_len > prev:
                best[cand] = match_len

        if not best:
            return logits

        idx = mx.array(list(best.keys()))
        exps = [
            min(float(L - self.allowed_length), _MAX_EXPONENT)
            for L in best.values()
        ]
        vals = mx.array(
            [self.multiplier * (self.base ** e) for e in exps],
            dtype=logits.dtype,
        )
        return logits.at[:, idx].add(-vals)


def dry_params_from_env() -> dict:
    """Read VLLM_MLX_DRY_* env defaults (cheap; call per request)."""
    out = {}
    try:
        out["multiplier"] = float(os.environ.get("VLLM_MLX_DRY_MULTIPLIER", "0"))
        out["base"] = float(os.environ.get("VLLM_MLX_DRY_BASE", "1.75"))
        out["allowed_length"] = int(
            os.environ.get("VLLM_MLX_DRY_ALLOWED_LENGTH", "2")
        )
        out["window"] = int(os.environ.get("VLLM_MLX_DRY_RANGE", "2048"))
        raw = os.environ.get("VLLM_MLX_DRY_SEQUENCE_BREAKERS")
        out["breakers"] = (
            tuple(raw.split(",")) if raw else DEFAULT_SEQUENCE_BREAKERS
        )
    except (TypeError, ValueError):
        logger.warning("Invalid VLLM_MLX_DRY_* env value; DRY disabled")
        return {"multiplier": 0.0, "base": 1.75, "allowed_length": 2,
                "window": 2048, "breakers": DEFAULT_SEQUENCE_BREAKERS}
    return out


def build_dry_processor(
    tokenizer: Any,
    multiplier: float | None = None,
    base: float | None = None,
    allowed_length: int | None = None,
    window: int | None = None,
    sequence_breakers: Iterable[str] | None = None,
) -> DRYLogitsProcessor | None:
    """Resolve request values over env defaults; None when DRY is off."""
    env = dry_params_from_env()
    eff_multiplier = env["multiplier"] if multiplier is None else float(multiplier)
    if eff_multiplier <= 0:
        return None
    breakers = (
        tuple(sequence_breakers)
        if sequence_breakers is not None
        else env["breakers"]
    )
    return DRYLogitsProcessor(
        multiplier=eff_multiplier,
        base=env["base"] if base is None else float(base),
        allowed_length=(
            env["allowed_length"] if allowed_length is None else int(allowed_length)
        ),
        window=env["window"] if window is None else int(window),
        breaker_ids=resolve_breaker_ids(tokenizer, breakers),
    )
