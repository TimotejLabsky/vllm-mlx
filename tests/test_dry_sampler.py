# SPDX-License-Identifier: Apache-2.0
"""Tests for the DRY sequence-level repetition penalty (fork patch)."""

from __future__ import annotations

import mlx.core as mx

from vllm_mlx.dry_sampler import (
    DEFAULT_SEQUENCE_BREAKERS,
    DRYLogitsProcessor,
    build_dry_processor,
    resolve_breaker_ids,
    zarray,
)

VOCAB = 64


def _logits():
    return mx.zeros((1, VOCAB), dtype=mx.float32)


def _apply(proc, seq, prompt=(999,)):
    """Prime the stateful processor with a prompt call (records gen_start),
    then apply with ``seq`` as the GENERATED region."""
    proc(mx.array(list(prompt)), _logits())
    out = proc(mx.array(list(prompt) + list(seq)), _logits())
    mx.eval(out)
    return out


def _naive_match_lengths(seq, breakers):
    """O(n^2) reference: for each earlier end position e, the longest common
    suffix between seq[..e] and the full sequence, not crossing breakers."""
    n = len(seq)
    best = {}
    for e in range(1, n - 1):
        L = 0
        while (
            L < e + 1
            and seq[e - L] == seq[n - 1 - L]
            and seq[e - L] not in breakers
            and seq[n - 1 - L] not in breakers
        ):
            L += 1
        cand = seq[e + 1]
        if L > 0 and cand not in breakers:
            best[cand] = max(best.get(cand, 0), L)
    return best


def test_zarray_basic():
    assert zarray(list("aabxaab")) == [7, 1, 0, 0, 3, 1, 0]
    assert zarray([]) == []
    assert zarray([5]) == [1]


def test_zarray_matches_naive_on_random():
    import random

    rng = random.Random(42)
    for _ in range(50):
        seq = [rng.randrange(4) for _ in range(rng.randrange(1, 60))]
        z = zarray(seq)
        for i in range(len(seq)):
            L = 0
            while i + L < len(seq) and seq[L] == seq[i + L]:
                L += 1
            assert z[i] == L, (seq, i)


def test_loop_extension_is_penalized_and_grows():
    proc = DRYLogitsProcessor(multiplier=0.8, base=1.75, allowed_length=2,
                              window=512)
    # context: [10 11 12 13 | 10 11 12] -> token 13 would extend a 3-token
    # repeat of the suffix [10 11 12].
    out = _apply(proc, [10, 11, 12, 13, 10, 11, 12])
    pen3 = -out[0, 13].item()
    assert pen3 > 0.79  # 0.8 * 1.75**(3-2)... wait, L=3 -> 0.8*1.75 = 1.4
    # longer repeat -> exponentially larger penalty
    out = _apply(proc, [9, 10, 11, 12, 13, 9, 10, 11, 12])
    pen4 = -out[0, 13].item()
    assert pen4 > pen3
    # everything else untouched
    assert out[0, 5].item() == 0.0


def test_allowed_length_gate():
    proc = DRYLogitsProcessor(multiplier=0.8, base=1.75, allowed_length=4,
                              window=512)
    # only a 3-token suffix repeat: below allowed_length=4 -> no penalty
    out = _apply(proc, [10, 11, 12, 13, 10, 11, 12])
    assert out[0, 13].item() == 0.0


def test_sequence_breakers_reset_and_protect():
    BRK = 50
    proc = DRYLogitsProcessor(multiplier=0.8, base=1.75, allowed_length=2,
                              window=512, breaker_ids={BRK})
    # repeat spans a breaker: [10 BRK 11 12] ... suffix [10 BRK 11] -> the
    # match is capped at the breaker, leaving only [11] (< allowed) -> free
    out = _apply(proc, [10, BRK, 11, 12, 10, BRK, 11])
    assert out[0, 12].item() == 0.0
    # candidate that IS a breaker is never penalized
    out = _apply(proc, [10, 11, BRK, 12, 10, 11])
    assert out[0, BRK].item() == 0.0
    # last token being a breaker -> no-op step
    out = _apply(proc, [10, 11, 12, 10, 11, BRK])
    assert mx.all(out == 0).item()


def test_matches_naive_reference_on_random():
    import random

    rng = random.Random(7)
    breakers = {0}
    proc = DRYLogitsProcessor(multiplier=1.0, base=2.0, allowed_length=1,
                              window=512, breaker_ids=breakers)
    for _ in range(30):
        seq = [rng.randrange(6) for _ in range(rng.randrange(8, 80))]
        if seq[-1] in breakers:
            seq[-1] = 1
        out = _apply(proc, seq)
        ref = _naive_match_lengths(seq, breakers)
        for tok in range(6):
            expected = (
                -1.0 * 2.0 ** min(ref[tok] - 1, 18)
                if tok in ref and ref[tok] >= 1
                else 0.0
            )
            got = out[0, tok].item()
            assert abs(got - expected) < 1e-4, (seq, tok, got, expected)


def test_window_limits_lookback():
    proc = DRYLogitsProcessor(multiplier=0.8, base=1.75, allowed_length=2,
                              window=8)
    # the repeat lies entirely outside the 8-token window -> no penalty
    seq = [10, 11, 12, 13] + [1, 2, 3, 4, 5, 6, 7, 8]
    out = _apply(proc, seq)
    assert mx.all(out == 0).item()


def test_fp16_no_overflow():
    proc = DRYLogitsProcessor(multiplier=2.0, base=1.75, allowed_length=2,
                              window=512)
    seq = list(range(20, 40)) + [99] + list(range(20, 40))
    logits = mx.zeros((1, 128), dtype=mx.float16)
    proc(mx.array([999]), logits)  # prime gen_start
    out = proc(mx.array([999] + seq), logits)
    mx.eval(out)
    assert mx.isfinite(out).all().item()


def test_prompt_repeats_are_ignored():
    """Regression for the 2026-06-12 incident: re-emitting a command that
    appears in the PROMPT (conversation history) is legitimate agentic
    repetition and must not be penalized. Only generation-internal repeats
    count."""
    proc = DRYLogitsProcessor(multiplier=0.8, base=1.75, allowed_length=2,
                              window=512)
    command = [30, 31, 32, 33, 34, 35]  # "git log ... | head -20"
    prompt = [1, 2] + command + [3, 4]
    # generation re-emits the command up to its last token: the only match
    # is in the prompt region -> token 35 must NOT be penalized
    out = _apply(proc, command[:-1], prompt=prompt)
    assert out[0, 35].item() == 0.0
    # but the SAME repeat occurring twice within the generation is caught
    proc2 = DRYLogitsProcessor(multiplier=0.8, base=1.75, allowed_length=2,
                               window=512)
    out2 = _apply(proc2, command + [9] + command[:-1], prompt=prompt)
    assert out2[0, 35].item() < 0.0


def test_first_call_records_gen_start_and_is_noop():
    proc = DRYLogitsProcessor(multiplier=0.8, base=1.75, allowed_length=2,
                              window=512)
    # a prompt full of repeats produces NO penalty on the first call
    seq = [10, 11, 12] * 10
    out = proc(mx.array(seq), _logits())
    mx.eval(out)
    assert mx.all(out == 0).item()
    assert proc._gen_start == len(seq)


def test_build_from_env_and_request(monkeypatch):
    class Tok:
        def encode(self, s, add_special_tokens=False):
            return [hash(s) % 1000]

    # off by default
    monkeypatch.delenv("VLLM_MLX_DRY_MULTIPLIER", raising=False)
    assert build_dry_processor(Tok()) is None
    # request value wins
    p = build_dry_processor(Tok(), multiplier=0.8, allowed_length=3)
    assert p is not None and p.allowed_length == 3
    # env default
    monkeypatch.setenv("VLLM_MLX_DRY_MULTIPLIER", "0.6")
    p = build_dry_processor(Tok())
    assert p is not None and abs(p.multiplier - 0.6) < 1e-9
    # request can force OFF over env
    assert build_dry_processor(Tok(), multiplier=0) is None


def test_resolve_breaker_ids_smoke():
    class Tok:
        def encode(self, s, add_special_tokens=False):
            return [len(s) + 100]

    ids = resolve_breaker_ids(Tok(), DEFAULT_SEQUENCE_BREAKERS)
    assert ids  # non-empty; exact ids are tokenizer-dependent
