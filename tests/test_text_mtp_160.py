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
    drafter=None,
):
    """``late=(n, prompt)``: insert a second request after ``n`` calls.
    ``drafter(model)``: a drafter to use instead of the oracle (any ``ref``)."""
    gen = BatchGenerator(
        model, max_tokens=max_tokens, stop_tokens=[[STOP]] if stop else None
    )
    mtp = None
    if ref is not None:
        mtp = text_mtp.TextMTP("oracle")
        mtp.block = block
        mtp.lm = model.language_model
        mtp.drafter = (
            drafter(model)
            if drafter is not None
            else OracleDrafter(
                ref, gen._generation_batch, len(prompts[0]), wrong_every, fail
            )
        )
        mtp.spec_sampling = drafter is not None and getattr(
            mtp.drafter, "spec_sampling", False
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


def test_160_length_finish_never_drafts_past_max_tokens(model):
    """(#161) A round is capped to the tokens the request may still return,
    so a length finish needs no rollback - and the cache is still exact."""
    ref = _run(model, 40)[0][0]
    for max_tokens in range(1, 16):
        plain = _run(model, max_tokens)
        tokens, caches, reasons, mtp = _run(
            model, max_tokens, ref=ref, block=4, wrong_every=0
        )
        assert tokens[0] == plain[0][0] and reasons[0] == "length"
        assert _max_diff(caches[0], plain[1][0]) < TOL
        assert mtp.stats()["sync_rollbacks"] == 0


@pytest.mark.parametrize("stop_index", range(4, 14))
def test_160_stop_token_inside_a_round(model, stop_index):
    """The stop token lands on a first token or on an accepted draft, with
    further drafts and an already launched round behind it (#161): all of
    that must be undone before the cache is handed back."""
    ref = _run(model, 40)[0][0]
    global STOP
    saved, STOP = STOP, ref[stop_index]
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


def test_160_stop_sweep_reaches_every_undo_path(model):
    """The sweep above only means something if it produces: a stop on a first
    token, a stop on a draft with drafts still queued, and a launched round
    whose first token was / was not returned yet."""
    seen = set()
    real_sync = text_mtp._sync

    def spy(batch):
        state = getattr(batch, "_text_mtp", None)
        if state is not None and (state.pending is not None or state.queue):
            seen.add(
                (
                    state.pending is not None,
                    bool(state.queue) and not state.queue[-1][2],
                    sum(1 for item in state.queue if item[2]) > 0,
                )
            )
        return real_sync(batch)

    ref = _run(model, 40)[0][0]
    global STOP
    saved = STOP
    text_mtp._sync = spy
    try:
        for stop_index in range(4, 14):
            STOP = ref[stop_index]
            _run(model, 40, ref=ref, block=4, wrong_every=0, stop=True)
    finally:
        STOP = saved
        text_mtp._sync = real_sync
    # (round launched, its first token still queued, drafts queued)
    assert (True, True, True) in seen
    assert (True, True, False) in seen
    assert (True, False, False) in seen


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


def _dry_and_thinking(budget):
    from vllm_mlx.dry_sampler import DRYLogitsProcessor

    def make():
        return [
            DRYLogitsProcessor(5.0, allowed_length=1, window=64),
            *_thinking(budget)(),
        ]

    return make


@pytest.mark.parametrize("block", [2, 4])
def test_160_dry_is_applied_at_drafted_positions(model, block):
    """DRY depends only on the token context, so it is called with each
    drafted position's context instead of switching speculation off."""
    plain = _run(model, 70, processors=_dry_and_thinking(6))
    ref = plain[0][0]
    # DRY really shapes this stream; otherwise the comparison proves nothing.
    assert ref != _run(model, 70, processors=_thinking(6))[0][0]

    tokens, caches, _, mtp = _run(
        model, 70, ref=ref, block=block, wrong_every=0, processors=_dry_and_thinking(6)
    )

    assert tokens[0] == ref
    assert _max_diff(caches[0], plain[1][0]) < TOL
    stats = mtp.stats()
    assert stats["bypass_counts"]["logits_processors"] == 0
    assert stats["accepted"] > 20 and stats["errors"] == 0


def test_160_failure_after_the_forward_inside_a_launch(model, monkeypatch):
    """(#161) A launch that fails after its forward advanced the cache must
    put the cache back before the plain step runs."""
    plain_tokens, plain_caches, _, _ = _run(model, 40)
    real_head = text_mtp._head
    calls = {"n": 0}

    def flaky_head(lm, hidden):
        calls["n"] += 1
        if calls["n"] % 5 == 3:
            raise RuntimeError("head exploded")
        return real_head(lm, hidden)

    monkeypatch.setattr(text_mtp, "_head", flaky_head)
    tokens, caches, _, mtp = _run(model, 40, ref=plain_tokens[0], wrong_every=0)

    assert tokens[0] == plain_tokens[0]
    assert _max_diff(caches[0], plain_caches[0]) < TOL
    stats = mtp.stats()
    assert stats["errors"] > 2 and stats["accepted"] > 0
    assert stats["disabled"] is False


def test_160_overlap_off_gives_the_same_stream(model, monkeypatch):
    plain_tokens, plain_caches, _, _ = _run(model, 40)
    monkeypatch.setenv(text_mtp.ENV_OVERLAP, "0")
    tokens, caches, _, mtp = _run(model, 40, ref=plain_tokens[0])
    assert mtp.overlap is False
    assert tokens[0] == plain_tokens[0]
    assert _max_diff(caches[0], plain_caches[0]) < TOL


def _tiny_scheduler(tmp_path, monkeypatch):
    from types import SimpleNamespace

    (tmp_path / "model.safetensors").write_bytes(b"x" * 1000)
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setenv(text_mtp.ENV_DRAFTER, str(tmp_path))
    monkeypatch.delenv(text_mtp.ENV_BLOCK, raising=False)
    text_mtp._weight_bytes_cache.clear()
    layers = [
        SimpleNamespace(
            linear_attn=SimpleNamespace(num_v_heads=2, head_v_dim=3, head_k_dim=5)
        ),
        SimpleNamespace(self_attn=object()),
        SimpleNamespace(
            linear_attn=SimpleNamespace(num_v_heads=2, head_v_dim=3, head_k_dim=5)
        ),
    ]
    model = SimpleNamespace(
        language_model=SimpleNamespace(model=SimpleNamespace(layers=layers))
    )
    return SimpleNamespace(model=model)


def test_162_admission_prices_unloaded_weights_and_round_states(tmp_path, monkeypatch):
    scheduler = _tiny_scheduler(tmp_path, monkeypatch)
    # two recurrent layers x (2*3*5 float32) x (block 2 positions + 1 rollback copy)
    states = 2 * (2 * 3 * 5 * 4) * 3
    assert text_mtp.admission_extra_bytes(scheduler) == 1000 + states

    mtp = text_mtp.TextMTP(str(tmp_path))
    scheduler._text_mtp = mtp
    assert text_mtp.admission_extra_bytes(scheduler) == 1000 + states
    mtp.drafter = object()  # loaded: its weights are inside active memory now
    assert text_mtp.admission_extra_bytes(scheduler) == states
    mtp.block = 4
    assert text_mtp.admission_extra_bytes(scheduler) == 2 * (2 * 3 * 5 * 4) * 5
    mtp.disabled_reason = "broken"
    assert text_mtp.admission_extra_bytes(scheduler) == 0.0


def test_162_admission_is_zero_without_the_env_and_fails_open(tmp_path, monkeypatch):
    scheduler = _tiny_scheduler(tmp_path, monkeypatch)
    monkeypatch.setenv(text_mtp.ENV_DRAFTER, str(tmp_path / "missing"))
    text_mtp._weight_bytes_cache.clear()
    # unresolvable weights: only the states are priced, nothing raises
    assert text_mtp.admission_extra_bytes(scheduler) == 2 * (2 * 3 * 5 * 4) * 3
    monkeypatch.delenv(text_mtp.ENV_DRAFTER)
    assert text_mtp.admission_extra_bytes(scheduler) == 0.0


# ------------------------------------------- #167 speculative sampling ----
from mlx_lm.sample_utils import apply_top_k, apply_top_p, make_sampler  # noqa: E402

SS = (1.0, 0.9, 8)  # temp, top_p, top_k


def _dense(logits, params=SS):
    """make_sampler's distribution, the slow way (sort of the vocabulary)."""
    temp, top_p, top_k = params
    lp = logits.astype(mx.float32) - mx.logsumexp(logits.astype(mx.float32))
    lp = apply_top_k(apply_top_p(lp[None], top_p), top_k)[0]
    return mx.softmax(lp / temp)


def test_167_sparse_filter_is_make_samplers_distribution():
    for seed in range(20):
        logits = mx.random.normal((VOCAB,), key=mx.random.key(seed)) * 3.0
        for params in (SS, (0.7, 0.95, 20), (1.3, 1.0, 5), (1.0, 0.3, 64)):
            idx, lp = text_mtp._sparse_dist(logits, params)
            sparse = mx.zeros((VOCAB,)).at[idx].add(mx.exp(lp))
            assert mx.abs(sparse - _dense(logits, params)).max().item() < 1e-5


def test_167_emitted_token_is_distributed_as_the_sampler():
    """Draft from q, accept with min(1, p/q), else the residual: whatever q
    is, the token at the verified position must be distributed as p."""
    state = text_mtp._BatchState(text_mtp.TextMTP("x"))
    state.ss = SS
    p_logits = mx.random.normal((2, VOCAB), key=mx.random.key(1)) * 2.0
    q_logits = mx.random.normal((VOCAB,), key=mx.random.key(2)) * 2.0  # a poor q
    draw = text_mtp._q_sampler(state)
    counts, n = [0] * VOCAB, 6000
    mx.random.seed(167)
    accepted = 0
    for _ in range(n):
        d = draw(q_logits)
        accept, rejected, _bonus, _plain = text_mtp._ss_targets(
            p_logits, d, state
        ).tolist()
        token = int(d.item()) if accept else rejected
        accepted += accept
        counts[token] += 1
    p = _dense(p_logits[0]).tolist()
    tv = 0.5 * sum(abs(c / n - x) for c, x in zip(counts, p))
    assert tv < 0.03, tv
    q = _dense(q_logits).tolist()
    expected = sum(min(a, b) for a, b in zip(p, q))  # sum(min(p, q))
    assert abs(accepted / n - expected) < 0.03


class SamplingDrafter:
    """Drafts like the real head: logits from the hidden state (the target's
    own head here, slightly off), through the sampler it is given."""

    spec_sampling = True

    def __init__(self, model):
        self.lm = model.language_model
        self.sampled = 0

    def reset(self, lm):
        pass

    def set_shared_kv(self, shared, **kwargs):
        pass

    def draft_block(self, b, hidden, cache, block, sampler, dtype, greedy=False):
        logits = text_mtp._head(self.lm, hidden) * 0.9
        if greedy:
            return mx.argmax(logits, axis=-1).astype(dtype)
        self.sampled += 1
        return sampler(logits).astype(dtype)

    def accept_verified_tokens(self, *args, **kwargs):
        pass

    def draft_eval_state(self):
        return [None]


def _tagged_sampler(temp=1.0, top_p=0.9, top_k=8):
    s = make_sampler(temp=temp, top_p=top_p, top_k=top_k)
    s._vllm_mlx_sampling = (temp, top_p, 0.0, top_k)
    return s


def _replay(tokens):
    it = iter(list(tokens) + [0])  # mlx-lm samples one step ahead
    return lambda logprobs: mx.array([next(it)])


def test_167_sampled_stream_leaves_the_cache_plain_decoding_would(model):
    mx.random.seed(7)
    tokens, caches, reasons, mtp = _run(
        model, 61, ref=[], block=2, samplers=_tagged_sampler(),
        drafter=SamplingDrafter,
    )  # fmt: skip
    stats = mtp.stats()
    assert reasons[0] == "length" and len(tokens[0]) == 61
    assert stats["errors"] == 0 and stats["disabled"] is False
    assert stats["ss_rounds"] > 10 and 0 < stats["ss_acceptance_rate"] < 1
    assert mtp.drafter.sampled > 10
    # plain decoding forced to emit the same tokens leaves the same cache
    plain_tokens, plain_caches, _, _ = _run(model, 61, samplers=_replay(tokens[0]))
    assert plain_tokens[0] == tokens[0]
    assert _max_diff(caches[0], plain_caches[0]) < TOL


@pytest.mark.parametrize(
    "case",
    ["off", "greedy", "min_p", "no_top_k", "big_top_k", "block3", "untagged"],
)
def test_167_only_tagged_sampling_requests_speculate_by_sampling(case):
    mtp = text_mtp.TextMTP("x")
    mtp.spec_sampling, mtp.block = case != "off", 3 if case == "block3" else 2
    params = {
        "greedy": (0.0, 0.9, 0.0, 8),
        "min_p": (1.0, 0.9, 0.05, 8),
        "no_top_k": (1.0, 0.9, 0.0, 0),
        "big_top_k": (1.0, 0.9, 0.0, 100),
    }.get(case, (1.0, 0.9, 0.0, 8))
    sampler = make_sampler(temp=1.0)
    if case != "untagged":
        sampler._vllm_mlx_sampling = params
    batch = type("B", (), {"samplers": [sampler]})()
    assert text_mtp._ss_params(batch, mtp) is None
    mtp.spec_sampling, mtp.block = True, 2
    sampler._vllm_mlx_sampling = (1.0, 0.9, 0.0, 8)
    assert text_mtp._ss_params(batch, mtp) == (1.0, 0.9, 8)


# ------- #105 finished/bypassed speculative state is not left pinned
def test_105_join_drops_the_last_rounds_recorded_states(model):
    """A verify leaves ``_text_mtp_spec`` (per-position recurrent states) on
    every recurrent layer. After a second request joins, the row decodes
    bypassed and nothing can roll back into it: it must not stay pinned."""
    solo = _run(model, 30)[0][0]
    gen = BatchGenerator(model, max_tokens=30)
    mtp = text_mtp.TextMTP("oracle")
    mtp.block = 4
    mtp.lm = model.language_model
    mtp.drafter = OracleDrafter(solo, gen._generation_batch, len(PROMPT), 0)
    text_mtp._install_class_patches()
    gen._generation_batch._text_mtp = text_mtp._BatchState(mtp)
    gen.insert([list(PROMPT)])
    seen_spec = joined = False
    for call in range(40):
        if call == 5:
            gen.insert([[11, 3, 77, 8]])
        gen.next_generated()
        recurrent = [
            c for c in gen._generation_batch.prompt_cache if text_mtp._is_recurrent(c)
        ]
        specs = [getattr(c, "_text_mtp_spec", None) for c in recurrent]
        if call < 5:
            seen_spec = seen_spec or all(s is not None for s in specs)
        elif len(gen._generation_batch.uids) > 1:
            joined = True
            assert recurrent and all(s is None for s in specs)
    gen.close()
    assert seen_spec and joined


def _join_and_leave(model, ref=None):
    """A short request joins at call 4 and finishes (extend, then filter);
    the first request then decodes alone again."""
    gen = BatchGenerator(model, max_tokens=40)
    mtp = None
    if ref is not None:
        mtp = text_mtp.TextMTP("oracle")
        mtp.block = 4
        mtp.lm = model.language_model
        mtp.drafter = OracleDrafter(ref, gen._generation_batch, len(PROMPT), 3)
        text_mtp._install_class_patches()
        gen._generation_batch._text_mtp = text_mtp._BatchState(mtp)
    (first,) = gen.insert([list(PROMPT)])
    tokens, cache, calls = [], None, 0
    joined = left = recorded_after_leave = False
    while cache is None:
        if calls == 4:
            gen.insert([[11, 3, 77, 8]], max_tokens=[3])
        calls += 1
        for r in gen.next_generated():
            if r.uid == first:
                tokens.append(r.token)
                if r.finish_reason is not None:
                    cache = r.prompt_cache
        batch = gen._generation_batch
        joined = joined or len(batch.uids) > 1
        left = left or (joined and len(batch.uids) == 1)
        recorded_after_leave = recorded_after_leave or (
            left
            and any(
                getattr(c, "_text_mtp_spec", None) is not None
                for c in batch.prompt_cache
            )
        )
    gen.close()
    return tokens, cache, (mtp, recorded_after_leave)


def test_105_dropped_recording_never_feeds_a_later_rollback(model):
    """After the join and the leave dropped the recorded states, the row
    speculates alone again with wrong drafts: every rollback must read a
    recording its own verify wrote, so the stream and the cache are plain
    decode's."""
    plain_tokens, plain_cache, _ = _join_and_leave(model)
    solo = _run(model, 40)[0][0]
    tokens, cache, (mtp, recorded_after_leave) = _join_and_leave(model, ref=solo)
    assert recorded_after_leave  # it speculated again after the drop
    assert tokens == plain_tokens
    assert _max_diff(cache, plain_cache) < TOL
    stats = mtp.stats()
    assert stats["bypass_counts"]["concurrent_batch"] > 0
    assert stats["accepted"] > 0 and stats["rejected"] > 0
