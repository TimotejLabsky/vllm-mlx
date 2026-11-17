"""Patch #160 — external MTP drafter on the text path.

Everything here drives mlx-lm's real ``BatchGenerator`` on a tiny
random-weight Qwen3.5 (two recurrent + two attention layers). The drafter is
an oracle that knows the plain-decode stream and gets every third draft wrong
on purpose, so accepts, partial accepts and rejects all happen.

The property under test: with the drafter on, the emitted tokens AND the cache
handed back at finish are exactly what plain decoding produces.
"""

import logging

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from mlx_lm.generate import BatchGenerator
from mlx_lm.models import qwen3_5 as q35

from vllm_mlx import text_mtp
from vllm_mlx.constrained.thinking_processor import ThinkingAwareLogitsProcessor

VOCAB = 96
THINK_START, THINK_END, STOP = 1, 2, 3
PROMPT = [5, 9, 17, 23, 31, 42, 7]
# float32 rounding between a multi-position and a single-position forward on
# the tiny model; a wrong rollback is off by ~1e-1 (the 8-bit 27B measured 0).
TOL = 1e-4


def _build_model():
    mx.random.seed(160)
    args = q35.ModelArgs.from_dict(
        {
            "model_type": "qwen3_5",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": VOCAB,
            "linear_num_value_heads": 2,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 32,
            "full_attention_interval": 2,
        }
    )
    m = q35.Model(args)
    # Default init gives near-uniform logits; spread them so the greedy
    # stream is not one repeated token.
    params = dict(tree_flatten(m.parameters()))
    m.load_weights(
        [
            (k, v * 6.0 if k.endswith("lm_head.weight") else v)
            for k, v in params.items()
        ],
        strict=False,
    )
    m.train(False)
    mx.eval(m.parameters())
    return m


@pytest.fixture(scope="module")
def model():
    return _build_model()


class OracleDrafter:
    """Drafts the true continuation, except at every third generated index."""

    def __init__(self, ref, batch, prompt_len, wrong_every=3, fail=False):
        self.ref, self.batch, self.prompt_len = ref, batch, prompt_len
        self.wrong_every, self.fail = wrong_every, fail
        self.resets = 0

    def reset(self, lm):
        self.resets += 1

    def set_shared_kv(self, shared, **kwargs):
        assert shared == {}

    def draft_block(self, b, hidden, cache, block, sampler, dtype, greedy=False):
        if self.fail is True:
            raise RuntimeError("drafter exploded")
        index = len(self.batch.tokens[0]) - self.prompt_len  # index of b
        out = []
        for j in range(1, block):
            i = index + j
            token = self.ref[i] if i < len(self.ref) else 0
            if self.wrong_every and i % self.wrong_every == 0:
                token = (token + 1) % VOCAB
            out.append(token)
        return mx.array([out], dtype=dtype)

    def accept_verified_tokens(self, *args, **kwargs):
        if self.fail == "after_verify":
            raise RuntimeError("drafter exploded after the verify")

    def draft_eval_state(self):
        return [None]


def _arrays(cache):
    out = []
    for c in cache:
        state = c.state
        for item in state if isinstance(state, (list, tuple)) else [state]:
            if isinstance(item, (list, tuple)):
                out.extend(x for x in item if isinstance(x, mx.array))
            elif isinstance(item, mx.array):
                out.append(item)
    return out


def _max_diff(a, b):
    xs, ys = _arrays(a), _arrays(b)
    assert len(xs) == len(ys) and xs
    worst = 0.0
    for x, y in zip(xs, ys):
        assert x.shape == y.shape
        if x.size:
            worst = max(
                worst,
                float(mx.abs(x.astype(mx.float32) - y.astype(mx.float32)).max().item()),
            )
    return worst


def _run(
    model,
    max_tokens,
    ref=None,
    block=3,
    processors=None,
    wrong_every=3,
    fail=False,
    stop=False,
    prompts=(PROMPT,),
    samplers=None,
    late=None,
):
    """``late=(n, prompt)``: insert a second request after ``n`` calls."""
    gen = BatchGenerator(
        model, max_tokens=max_tokens, stop_tokens=[[STOP]] if stop else None
    )
    mtp = None
    if ref is not None:
        mtp = text_mtp.TextMTP("oracle")
        mtp.block = block
        mtp.lm = model.language_model
        mtp.drafter = OracleDrafter(
            ref, gen._generation_batch, len(prompts[0]), wrong_every, fail
        )
        text_mtp._install_class_patches()
        gen._generation_batch._text_mtp = text_mtp._BatchState(mtp)
    kwargs = {}
    if processors is not None:
        kwargs["logits_processors"] = [processors() for _ in prompts]
    if samplers is not None:
        kwargs["samplers"] = [samplers for _ in prompts]
    uids = gen.insert([list(p) for p in prompts], **kwargs)
    tokens = {u: [] for u in uids}
    caches, reasons = {}, {}
    calls = 0
    while len(reasons) < len(uids):
        if late is not None and calls == late[0]:
            late_kwargs = {}
            if processors is not None:
                late_kwargs["logits_processors"] = [processors()]
            (late_uid,) = gen.insert([list(late[1])], **late_kwargs)
            uids.append(late_uid)
            tokens[late_uid] = []
        calls += 1
        for r in gen.next_generated():
            tokens[r.uid].append(r.token)
            if r.finish_reason is not None:
                reasons[r.uid] = r.finish_reason
                caches[r.uid] = r.prompt_cache
    gen.close()
    return tokens, caches, reasons, mtp


def _thinking(budget):
    def make():
        return [
            ThinkingAwareLogitsProcessor(
                [THINK_START],
                [THINK_END],
                budget,
                vocab_size=VOCAB,
                prompt_has_think_tag=True,
            )
        ]

    return make


@pytest.mark.parametrize("block", [2, 3, 4])
@pytest.mark.parametrize("max_tokens", [1, 2, 3, 7, 24, 61])
def test_160_tokens_and_cache_match_plain_decoding(model, block, max_tokens):
    plain_tokens, plain_caches, _, _ = _run(model, max_tokens)
    ref = plain_tokens[0]
    assert len(set(ref)) > 3 or max_tokens < 7  # the stream is not degenerate

    tokens, caches, reasons, mtp = _run(model, max_tokens, ref=ref, block=block)

    assert tokens[0] == ref
    assert reasons[0] == "length"
    assert _max_diff(caches[0], plain_caches[0]) < TOL
    stats = mtp.stats()
    assert stats["errors"] == 0 and stats["disabled"] is False
    if max_tokens >= 24:
        assert stats["accepted"] > 0 and stats["rejected"] > 0
        assert stats["rollbacks"] > 0


def test_160_finish_inside_a_round_rolls_the_cache_back(model):
    """max_tokens lands while verified tokens are still queued: the cache at
    finish must not contain them."""
    ref = _run(model, 40)[0][0]
    hit = False
    for max_tokens in range(4, 16):
        plain = _run(model, max_tokens)
        tokens, caches, _, mtp = _run(
            model, max_tokens, ref=ref, block=4, wrong_every=0
        )
        assert tokens[0] == plain[0][0]
        assert _max_diff(caches[0], plain[1][0]) < TOL
        hit = hit or mtp.stats()["sync_rollbacks"] > 0
    assert hit


def test_160_stop_token_inside_a_round(model):
    ref = _run(model, 40)[0][0]
    stop_at = ref[9]
    global STOP
    saved, STOP = STOP, stop_at
    try:
        plain_tokens, plain_caches, plain_reasons, _ = _run(model, 40, stop=True)
        tokens, caches, reasons, _ = _run(
            model, 40, ref=ref, block=4, wrong_every=0, stop=True
        )
    finally:
        STOP = saved
    assert plain_reasons[0] == "stop" and reasons[0] == "stop"
    assert tokens[0] == plain_tokens[0]
    assert _max_diff(caches[0], plain_caches[0]) < TOL


@pytest.mark.parametrize("budget", [0, 2, 5, 11, 1000])
@pytest.mark.parametrize("block", [2, 4])
def test_160_thinking_budget_processor_is_exact(model, budget, block):
    plain_tokens, plain_caches, _, _ = _run(model, 40, processors=_thinking(budget))
    ref = plain_tokens[0]

    tokens, caches, _, mtp = _run(
        model, 40, ref=ref, block=block, processors=_thinking(budget)
    )

    assert tokens[0] == ref
    assert _max_diff(caches[0], plain_caches[0]) < TOL
    stats = mtp.stats()
    assert stats["errors"] == 0
    assert stats["bypass_counts"]["logits_processors"] == 0
    assert stats["accepted"] > 0


def test_160_thinking_stream_exercises_the_forced_close(model):
    forced = _run(model, 40, processors=_thinking(5))[0][0]
    free = _run(model, 40)[0][0]
    assert forced[5] == THINK_END
    assert forced != free


def test_160_unknown_processor_bypasses(model):
    def passthrough():
        return [lambda tokens, logits: logits]

    plain_tokens = _run(model, 20, processors=passthrough)[0][0]
    tokens, _, _, mtp = _run(model, 20, ref=plain_tokens, processors=passthrough)

    assert tokens[0] == plain_tokens
    stats = mtp.stats()
    assert stats["attempted"] == 0
    assert stats["bypass_counts"]["logits_processors"] > 0


def test_160_two_requests_decode_plain_and_correct(model):
    other = [11, 3, 77, 8]
    plain_tokens, plain_caches, _, _ = _run(model, 20, prompts=(PROMPT, other))
    tokens, caches, _, mtp = _run(
        model, 20, ref=plain_tokens[0], prompts=(PROMPT, other)
    )

    assert tokens == plain_tokens
    for uid in tokens:
        assert _max_diff(caches[uid], plain_caches[uid]) < TOL
    assert mtp.stats()["bypass_counts"]["concurrent_batch"] > 0


def test_160_failing_drafter_is_disabled_and_output_stays_correct(model, caplog):
    ref = _run(model, 40)[0][0]
    with caplog.at_level(logging.WARNING, logger="vllm_mlx.text_mtp"):
        tokens, caches, _, mtp = _run(model, 40, ref=ref, fail=True)

    assert tokens[0] == ref
    assert _max_diff(caches[0], _run(model, 40)[1][0]) < TOL
    stats = mtp.stats()
    assert stats["disabled"] is True
    assert stats["errors"] == text_mtp.DISABLE_AFTER_ERRORS
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1 and "DISABLED" in errors[0].getMessage()


def test_160_recording_forward_matches_mlx_lm(model):
    """The recording GatedDeltaNet call re-implements mlx-lm's layer body. If
    mlx-lm changes that layer, this is the test that says so."""
    from mlx_lm.models.cache import make_prompt_cache

    text_mtp._install_class_patches()
    lm = model.language_model
    tokens = mx.array([PROMPT])
    extra = mx.array([[4, 44, 13]])

    def forward(recording):
        cache = make_prompt_cache(model)
        lm.model(tokens, cache)
        text_mtp._REC["active"] = recording
        try:
            h = lm.model(extra, cache)
        finally:
            text_mtp._REC["active"] = False
        mx.eval(h)
        return h, cache

    h_plain, cache_plain = forward(False)
    h_rec, cache_rec = forward(True)

    assert float(mx.abs(h_plain - h_rec).max().item()) < 1e-4
    assert _max_diff(cache_rec, cache_plain) < 1e-4
    recurrent = [c for c in cache_rec if not c.is_trimmable()]
    assert recurrent and all(len(c._text_mtp_spec[1]) == 3 for c in recurrent)


def test_160_not_armed_without_the_env(monkeypatch):
    monkeypatch.delenv(text_mtp.ENV_DRAFTER, raising=False)

    class Gen:
        _generation_batch = object()

    gen, scheduler = Gen(), type("S", (), {})()
    text_mtp.maybe_install(scheduler, gen)
    assert not hasattr(gen, "get_mtp_stats")
    assert not hasattr(scheduler, "_text_mtp")


def test_160_masks_apply_at_drafted_positions(model):
    """A sampler that takes the reasoning-open token whenever it is not
    masked: in CONTENT only the mask stops it, at drafted positions too."""

    def greedy_for_think_start(logprobs):
        allowed = logprobs[:, THINK_START] > float("-inf")
        return mx.where(allowed, THINK_START, mx.argmax(logprobs, axis=-1))

    plain = _run(model, 30, processors=_thinking(4), samplers=greedy_for_think_start)
    ref = plain[0][0]
    assert ref[:4] == [THINK_START] * 4 and ref[4] == THINK_END
    assert THINK_START not in ref[5:]

    tokens, caches, _, mtp = _run(
        model,
        30,
        ref=ref,
        block=4,
        wrong_every=0,
        processors=_thinking(4),
        samplers=greedy_for_think_start,
    )

    assert tokens[0] == ref
    assert _max_diff(caches[0], plain[1][0]) < TOL
    assert mtp.stats()["accepted"] > 0


def test_160_failure_after_the_verify_restores_the_cache(model):
    plain_tokens, plain_caches, _, _ = _run(model, 30)
    tokens, caches, _, mtp = _run(model, 30, ref=plain_tokens[0], fail="after_verify")

    assert tokens[0] == plain_tokens[0]
    assert _max_diff(caches[0], plain_caches[0]) < TOL
    assert mtp.stats()["errors"] > 0


@pytest.mark.parametrize("join_at", [3, 4, 5, 6, 7])
def test_160_request_joining_mid_round_keeps_both_streams(model, join_at):
    """The second request arrives while verified tokens are still queued: the
    cache and the processor's token context must be rolled back before the
    batches merge, and the first request must carry on correctly."""
    other = [11, 3, 77, 8]
    late = (join_at, other)
    plain_tokens, plain_caches, _, _ = _run(
        model, 30, processors=_thinking(9), late=late
    )
    solo = _run(model, 30, processors=_thinking(9))[0][0]

    tokens, caches, _, mtp = _run(
        model,
        30,
        ref=solo,
        block=4,
        wrong_every=0,
        processors=_thinking(9),
        late=late,
    )

    assert tokens == plain_tokens
    for uid in tokens:
        assert _max_diff(caches[uid], plain_caches[uid]) < TOL
    stats = mtp.stats()
    assert stats["accepted"] > 0
    assert stats["bypass_counts"]["concurrent_batch"] > 0


def test_160_some_join_lands_inside_a_round(model):
    hits = 0
    for join_at in [3, 4, 5, 6, 7]:
        solo = _run(model, 30, processors=_thinking(9))[0][0]
        mtp = _run(
            model,
            30,
            ref=solo,
            block=4,
            wrong_every=0,
            processors=_thinking(9),
            late=(join_at, [11, 3, 77, 8]),
        )[3]
        hits += mtp.stats()["sync_rollbacks"] > 0
    assert hits > 0


def test_160_plan_leaves_the_processor_untouched():
    end = [THINK_END, 4]
    proc = ThinkingAwareLogitsProcessor(
        [THINK_START], end, 50, vocab_size=VOCAB, prompt_has_think_tag=True
    )
    logits = mx.zeros((1, VOCAB))
    proc(mx.array([], dtype=mx.int32), logits)
    proc(mx.array([7]), logits)

    assert proc.speculation_plan([THINK_END, 4, 9]) == [
        (),
        tuple(dict.fromkeys([THINK_START, THINK_END])),
        tuple(dict.fromkeys([THINK_START, THINK_END])),
    ]
    assert proc.speculation_plan([THINK_END]) == [()]
    # Had the plan left THINK_END in the matcher, this lone 4 would close
    # the reasoning.
    proc(mx.array([7, 4]), logits)
    assert proc.state.value == "thinking"
    assert proc.thinking_tokens == 2


def test_160_feeds_are_ignored_once_disabled():
    proc = ThinkingAwareLogitsProcessor(
        [THINK_START], [THINK_END], 50, vocab_size=VOCAB, prompt_has_think_tag=True
    )
    logits = mx.zeros((1, VOCAB))
    proc(mx.array([], dtype=mx.int32), logits)
    proc.disable_feeds()
    proc(mx.array([7, 8, 9]), logits)
    for token in (7, 8, THINK_END):  # late, already-consumed feeds
        proc.note_generated(token)
    assert proc.state.value == "thinking"
    assert proc.thinking_tokens == 3
