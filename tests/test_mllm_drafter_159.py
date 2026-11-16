"""Patch #159 — the MLLM assistant drafter on Qwen targets.

Three things, each of which made the drafter useless on the 27B before:

1. mlx-vlm's shared-KV helper needs ``layer.layer_type`` (Gemma only); on Qwen
   it raised on every decode step and the drafter never ran.
2. That failure was a WARNING per token and nothing else.
3. The thinking-budget processor switched speculation off for every thinking
   request, although it passes logits through for almost all of them.
"""

import hashlib
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import numpy as np
import pytest

from vllm_mlx.constrained.thinking_processor import (
    Phase,
    ThinkingAwareLogitsProcessor,
)
from vllm_mlx.mllm_batch_generator import (
    _MTP_DISABLE_AFTER_ERRORS,
    MLLMBatchResponse,
    install_mtp_mllm,
)

VOCAB = 16
THINK_START = 0
THINK_END = 1


def _request(**extra):
    return SimpleNamespace(
        uid=7,
        request_id="req-159",
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        mllm_draft=True,
        output_tokens=[],
        **extra,
    )


def _greedy(logprobs):
    return mx.argmax(logprobs, axis=-1)


class _TrimmableCache:
    def __init__(self, tokens=()):
        self.tokens = list(tokens)

    def is_trimmable(self):
        return True

    def trim(self, count):
        del self.tokens[len(self.tokens) - count :]


# --------------------------------------------------------------------------
# 1. shared KV on a target without layer_type
# --------------------------------------------------------------------------


def _external_setup(
    monkeypatch, layers, helper, verify_token=2, temperature=0.0, top_k=0
):
    import mlx_vlm.speculative.common as speculative_common
    import mlx_vlm.speculative.mtp as speculative_mtp

    monkeypatch.setattr(
        speculative_common, "_batch_cache_left_padding", lambda cache: None
    )
    monkeypatch.setattr(speculative_mtp, "_mtp_shared_kv_from_prompt_cache", helper)
    monkeypatch.setattr(
        speculative_mtp,
        "_mtp_cache_positions",
        lambda cache, batch_size: (10, [10] * batch_size),
    )
    monkeypatch.setattr(
        speculative_mtp, "_mtp_draft_position", lambda positions: positions
    )
    monkeypatch.setattr(
        speculative_mtp,
        "_mtp_draft_block_active",
        lambda *args, **kwargs: mx.array([[2]], dtype=mx.uint32),
    )

    request = _request()
    request.temperature, request.top_k = temperature, top_k
    active_batch = SimpleNamespace(
        uids=[7], requests=[request], num_tokens=[0], max_tokens=[64]
    )

    class Generator:
        def __init__(self):
            self.model = object()
            self.active_batch = active_batch
            self._step = MagicMock(return_value=(mx.array([9]), [mx.zeros(VOCAB)]))
            self._next = MagicMock(return_value=[])
            self.sampler = _greedy
            self.stop_tokens = set()
            self._maybe_store_prefix_cache = MagicMock()

    class DraftModel:
        def __init__(self):
            self.accept_lens = []
            self.draft_lens = []
            self.shared_kv_seen = []

        def reset(self, model):
            pass

        def set_shared_kv(self, shared_kv, **kwargs):
            self.shared_kv_seen.append(shared_kv)

    class LanguageModel:
        def __init__(self):
            self.model = SimpleNamespace(layers=layers)

        def __call__(self, input_tokens, cache=None, return_hidden=False):
            seq_len = input_tokens.shape[1]
            logits = mx.full((1, seq_len, VOCAB), -1000.0)
            # primary is 1; the drafter drafts 2 for the position after it
            logits[:, 0, verify_token if seq_len == 2 else 1] = 0.0
            return logits, mx.zeros((1, seq_len, 2))

    generator, draft_model = Generator(), DraftModel()
    install_mtp_mllm(generator, LanguageModel(), draft_model=draft_model)
    return generator, draft_model


def _one_step(generator):
    return generator._step(
        mx.array([[0]], dtype=mx.uint32),
        cache=[_TrimmableCache()],
        logits_processors=None,
        output_tokens=None,
        samplers=None,
    )


def test_159_target_without_layer_type_drafts_instead_of_failing(monkeypatch):
    def helper(model, cache):
        raise AssertionError("the layer_type helper must not run on this target")

    qwen_layers = [SimpleNamespace(is_linear=False), SimpleNamespace(is_linear=True)]
    generator, draft_model = _external_setup(monkeypatch, qwen_layers, helper)

    _one_step(generator)

    stats = generator.get_mtp_stats()
    assert (stats["attempted"], stats["accepted"], stats["errors"]) == (1, 1, 0)
    assert draft_model.shared_kv_seen == [{}]
    assert stats["disabled"] is False


def test_159_target_with_layer_type_still_requires_the_payload(monkeypatch):
    gemma_layers = [SimpleNamespace(layer_type="full_attention")]
    generator, _ = _external_setup(monkeypatch, gemma_layers, lambda model, cache: {})

    _one_step(generator)

    stats = generator.get_mtp_stats()
    assert (stats["accepted"], stats["errors"]) == (0, 1)


# --------------------------------------------------------------------------
# 2. a drafter that keeps failing is switched off, loudly, once
# --------------------------------------------------------------------------


def test_159_repeated_failures_disable_the_drafter_with_one_error(monkeypatch, caplog):
    gemma_layers = [SimpleNamespace(layer_type="full_attention")]
    generator, _ = _external_setup(monkeypatch, gemma_layers, lambda model, cache: {})

    with caplog.at_level(logging.WARNING, logger="vllm_mlx.mllm_batch_generator"):
        for _ in range(_MTP_DISABLE_AFTER_ERRORS + 5):
            _one_step(generator)

    stats = generator.get_mtp_stats()
    assert stats["disabled"] is True
    assert "shared-KV" in stats["disabled_reason"]
    # No draft is attempted once it is off.
    assert stats["attempted"] == _MTP_DISABLE_AFTER_ERRORS
    assert stats["errors"] == _MTP_DISABLE_AFTER_ERRORS
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert "DISABLED" in errors[0].getMessage()


def test_159_a_success_resets_the_failure_run(monkeypatch):
    payload = {"value": {}}
    gemma_layers = [SimpleNamespace(layer_type="full_attention")]
    generator, _ = _external_setup(
        monkeypatch, gemma_layers, lambda model, cache: payload["value"]
    )

    for _ in range(_MTP_DISABLE_AFTER_ERRORS - 1):
        _one_step(generator)
    payload["value"] = {"full_attention": object()}
    _one_step(generator)
    payload["value"] = {}
    for _ in range(_MTP_DISABLE_AFTER_ERRORS - 1):
        _one_step(generator)

    assert generator.get_mtp_stats()["disabled"] is False


# --------------------------------------------------------------------------
# 3. the thinking budget: what the processor does to the next position
# --------------------------------------------------------------------------


def _processor(budget, **kwargs):
    proc = ThinkingAwareLogitsProcessor(
        [THINK_START],
        [THINK_END],
        budget,
        vocab_size=VOCAB,
        prompt_has_think_tag=True,
        **kwargs,
    )
    proc(mx.array([], dtype=mx.int32), mx.zeros((1, VOCAB)))
    return proc


def test_159_speculation_view_matches_the_phase_machine():
    proc = _processor(10)
    assert proc.state == Phase.THINKING
    assert proc.speculation_mask_ids(5) == ()
    # the next token closes the reasoning: the position after it is CONTENT
    assert set(proc.speculation_mask_ids(THINK_END)) == {THINK_START, THINK_END}
    assert proc.state == Phase.THINKING  # read-only

    proc(mx.array([5, THINK_END]), mx.zeros((1, VOCAB)))
    assert proc.state == Phase.CONTENT
    assert set(proc.speculation_mask_ids(5)) == {THINK_START, THINK_END}


def test_159_speculation_view_refuses_near_the_budget_and_with_an_inner():
    proc = _processor(4)
    proc(mx.array([5, 5]), mx.zeros((1, VOCAB)))  # 2 of 4 used
    assert proc.speculation_mask_ids(5) is None

    constrained = _processor(10, inner=lambda tokens, logits: logits)
    assert constrained.speculation_mask_ids(5) is None

    unsynced = ThinkingAwareLogitsProcessor([THINK_START], [THINK_END], 10)
    assert unsynced.speculation_mask_ids(5) is None


class _RecordingProcessor:
    """Speculation-aware pass-through that records the context it is given."""

    def __init__(self):
        self.contexts = []

    def speculation_mask_ids(self, next_token):
        return ()

    def __call__(self, tokens, logits):
        self.contexts.append(tokens.tolist())
        return logits


@pytest.mark.parametrize(
    "verify_token, temperature, top_k, committed",
    [
        (2, 0.0, 0, 2),  # draft accepted: the draft is in the cache
        (3, 0.7, 1, 3),  # sampled reject: the target's own token is replayed
    ],
)
def test_159_processor_context_includes_the_token_committed_last_step(
    monkeypatch, verify_token, temperature, top_k, committed
):
    qwen_layers = [SimpleNamespace(is_linear=False)]
    generator, _ = _external_setup(
        monkeypatch,
        qwen_layers,
        lambda model, cache: {},
        verify_token=verify_token,
        temperature=temperature,
        top_k=top_k,
    )
    recorder = _RecordingProcessor()
    cache = [_TrimmableCache([0, 0, 0])]

    for context in ([0], [0, 1]):
        generator._step(
            mx.array([[context[-1]]], dtype=mx.uint32),
            cache=cache,
            logits_processors=[[recorder]],
            output_tokens=[context],
            samplers=None,
        )

    assert recorder.contexts == [[0], [0, 1, committed]]
    assert generator.get_mtp_stats()["errors"] == 0


def test_159_other_processors_still_bypass_speculation():
    original_step = MagicMock(return_value=(mx.array([7]), [mx.zeros(VOCAB)]))

    class Generator:
        def __init__(self):
            self._step = original_step
            self._next = MagicMock(return_value=[])
            self.active_batch = SimpleNamespace(uids=[7], requests=[_request()])
            self.sampler = _greedy

    generator = Generator()
    language_model = MagicMock()
    install_mtp_mllm(generator, language_model)

    def dry_like(tokens, logits):
        return logits

    generator._step(
        mx.array([[3]]),
        cache=[],
        logits_processors=[[_processor(10), dry_like]],
        output_tokens=[[1, 2]],
        samplers=[None],
    )

    original_step.assert_called_once()
    language_model.assert_not_called()
    assert generator.get_mtp_stats()["bypass_counts"]["logits_processors"] == 1


# --------------------------------------------------------------------------
# 3b. equivalence: speculation with the processor attached emits exactly the
#     tokens plain decoding does
# --------------------------------------------------------------------------


def _raw_logits(history, seed):
    digest = hashlib.sha256(repr((seed, tuple(history))).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    logits = rng.normal(size=VOCAB).astype(np.float32)
    logits[THINK_END] -= 0.5  # the reasoning closes at a different step per seed
    if history[-1] == THINK_END:
        # Right after the close the raw model wants to reopen the reasoning,
        # so the CONTENT mask decides this position - including when the
        # close was a drafted token the processor has not been shown yet.
        logits[THINK_START] += 10.0
    return logits


def _reference_stream(seed, budget, length):
    """One processor call per position, on generated tokens only."""
    proc = ThinkingAwareLogitsProcessor(
        [THINK_START], [THINK_END], budget, vocab_size=VOCAB, prompt_has_think_tag=True
    )
    history, generated = [5], []
    for _ in range(length):
        logits = mx.array(_raw_logits(history, seed))[None]
        logits = proc(mx.array(generated, dtype=mx.int32), logits)
        token = int(mx.argmax(logits, axis=-1).item())
        generated.append(token)
        history.append(token)
    return generated


class _HistoryModel:
    """Logits depend on the whole consumed history; the head drafts the true
    raw next token two times out of three and a wrong one otherwise."""

    def __init__(self, cache, seed):
        self.cache = cache
        self.seed = seed

    def __call__(self, input_tokens, cache=None, return_hidden=False):
        rows = []
        for token in input_tokens.tolist()[0]:
            self.cache.tokens.append(int(token))
            rows.append(_raw_logits(self.cache.tokens, self.seed))
        logits = mx.array(np.stack(rows))[None]
        return logits, mx.zeros((1, len(rows), 2))

    def mtp_forward(self, hidden, primary, mtp_cache=None):
        history = self.cache.tokens + [int(primary.item())]
        true_next = int(np.argmax(_raw_logits(history, self.seed)))
        if len(history) % 3 == 0:
            true_next = (true_next + 1) % VOCAB
        logits = np.full((1, 1, VOCAB), -1000.0, dtype=np.float32)
        logits[0, 0, true_next] = 0.0
        return mx.array(logits)


def _speculative_stream(seed, budget, length):
    proc = ThinkingAwareLogitsProcessor(
        [THINK_START], [THINK_END], budget, vocab_size=VOCAB, prompt_has_think_tag=True
    )
    cache = _TrimmableCache([5])
    request = _request()
    first = proc(mx.array([], dtype=mx.int32), mx.array(_raw_logits([5], seed))[None])

    class Generator:
        def __init__(self):
            self.model = object()
            self.active_batch = SimpleNamespace(
                uids=[7], requests=[request], num_tokens=[0], max_tokens=[10**6]
            )
            self.sampler = _greedy
            self.stop_tokens = set()
            self._maybe_store_prefix_cache = MagicMock()
            self.y = int(mx.argmax(first, axis=-1).item())

        def _next(self):
            # MLLMBatchGenerator._next in miniature: emit the pending token,
            # compute the one after it, context = generated + [pending].
            y = self.y
            tokens, _ = self._step(
                mx.array([[y]], dtype=mx.uint32),
                [cache],
                [[proc]],
                [list(request.output_tokens) + [y]],
                [None],
            )
            self.y = int(tokens[0].item())
            self.active_batch.num_tokens[0] += 1
            request.output_tokens.append(y)
            return [
                MLLMBatchResponse(
                    uid=7, request_id="req-159", token=y, logprobs=mx.zeros(VOCAB)
                )
            ]

        def _step(self, *args, **kwargs):
            raise AssertionError("a speculation-aware processor must not bypass")

    generator = Generator()
    install_mtp_mllm(generator, _HistoryModel(cache, seed))
    emitted = []
    while len(emitted) < length:
        emitted.extend(response.token for response in generator._next())
    return emitted[:length], generator.get_mtp_stats()


@pytest.mark.parametrize("budget", [3, 6, 12, 1000])
@pytest.mark.parametrize("seed", range(12))
def test_159_speculation_with_thinking_budget_emits_the_plain_stream(seed, budget):
    length = 48
    expected = _reference_stream(seed, budget, length)
    emitted, stats = _speculative_stream(seed, budget, length)

    assert emitted == expected
    assert stats["errors"] == 0
    # speculation really ran with the processor attached
    assert stats["accepted"] > 0


def test_159_equivalence_covers_both_phases_and_the_forced_close():
    """The parametrized case above only means something if the streams it
    compares include a natural close, a forced close and content tokens."""
    natural = forced = False
    for seed in range(12):
        long_run = _reference_stream(seed, 1000, 48)
        if THINK_END in long_run[:-4]:
            natural = True
        short_run = _reference_stream(seed, 3, 48)
        if short_run[3] == THINK_END and THINK_END not in short_run[:3]:
            forced = True
    assert natural and forced
