# SPDX-License-Identifier: Apache-2.0
"""Per-row MRoPE rope-delta bookkeeping (glm4v / qwen3_vl families).

Those language models keep ONE mutable ``_rope_deltas`` / ``_position_ids``
instance attribute, set by whichever prefill ran last. Under continuous
batching, request B's prefill overwrote the delta request A's decode needed
— corrupting A's RoPE positions for the rest of its generation — and a
stale ``_position_ids`` could be sliced into a later text row's prefill.

The generator now: captures the delta per request at prefill end
(``_capture_rope_delta``), resets the shared state before every single-row
prefill forward (``_arm_rope_state``), and re-broadcasts the per-row stack
before every batched decode step (``_arm_decode_rope_deltas``). Models
without the attribute (gemma4, qwen3_5 — the deployed fleet) are untouched.
"""

import mlx.core as mx
import pytest

from vllm_mlx.mllm_batch_generator import (
    MLLMBatch,
    MLLMBatchGenerator,
    MLLMBatchRequest,
    MLLMBatchStats,
)

pytestmark = pytest.mark.skipif(mx is None, reason="MLX not available")


class _RopeTrackingLM:
    """Minimal stand-in for a glm4v/qwen3_vl LanguageModel."""

    def __init__(self):
        self._rope_deltas = None
        self._position_ids = None
        self.call_time_deltas = []

    def __call__(self, tokens, cache=None, **kwargs):
        # Record the state visible to the model at forward time.
        self.call_time_deltas.append(self._rope_deltas)
        batch = tokens.shape[0]
        return mx.zeros((batch, tokens.shape[1], 16))


class _PlainLM:
    """A model without the MRoPE delta mechanism (gemma4/qwen3_5 class)."""

    def __call__(self, tokens, cache=None, **kwargs):
        return mx.zeros((tokens.shape[0], tokens.shape[1], 16))


def _bare_generator(lm):
    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen.language_model = lm
    return gen


def _request(uid, rope_delta=None):
    req = MLLMBatchRequest(uid=uid, request_id=f"req-{uid}", prompt="p")
    req.rope_delta = rope_delta
    return req


class TestArmRopeState:
    def test_fresh_prefill_clears_both(self):
        lm = _RopeTrackingLM()
        lm._rope_deltas = mx.array([[37]])
        lm._position_ids = mx.zeros((3, 1, 8))
        gen = _bare_generator(lm)

        gen._arm_rope_state(continuation=False)

        assert lm._rope_deltas is None
        assert lm._position_ids is None

    def test_continuation_arms_zero_delta(self):
        lm = _RopeTrackingLM()
        lm._rope_deltas = mx.array([[37]])
        lm._position_ids = mx.zeros((3, 1, 8))
        gen = _bare_generator(lm)

        gen._arm_rope_state(continuation=True)

        assert lm._rope_deltas is not None
        assert lm._rope_deltas.shape == (1, 1)
        assert lm._rope_deltas.item() == 0
        assert lm._position_ids is None

    def test_plain_model_untouched(self):
        lm = _PlainLM()
        gen = _bare_generator(lm)

        gen._arm_rope_state(continuation=False)
        gen._arm_rope_state(continuation=True)

        assert not hasattr(lm, "_rope_deltas")
        assert not hasattr(lm, "_position_ids")


class TestCaptureRopeDelta:
    def test_captures_current_delta(self):
        lm = _RopeTrackingLM()
        delta = mx.array([[41]])
        lm._rope_deltas = delta
        gen = _bare_generator(lm)
        req = _request(1)

        gen._capture_rope_delta(req)

        assert req.rope_delta is delta

    def test_plain_model_leaves_none(self):
        gen = _bare_generator(_PlainLM())
        req = _request(1)

        gen._capture_rope_delta(req)

        assert req.rope_delta is None


def _batch_for(requests):
    n = len(requests)
    return MLLMBatch(
        uids=[r.uid for r in requests],
        request_ids=[r.request_id for r in requests],
        y=mx.array([100 + i for i in range(n)]),
        logprobs=[mx.array([0.0]) for _ in range(n)],
        max_tokens=[100] * n,
        num_tokens=[0] * n,
        cache=[],
        requests=requests,
    )


class TestArmDecodeRopeDeltas:
    def test_stacks_in_batch_order_with_zero_default(self):
        lm = _RopeTrackingLM()
        gen = _bare_generator(lm)
        batch = _batch_for(
            [_request(1, mx.array([[7]])), _request(2, None), _request(3, mx.array([[-3]]))]
        )

        gen._arm_decode_rope_deltas(batch)

        assert lm._rope_deltas.shape == (3, 1)
        assert lm._rope_deltas.reshape(-1).tolist() == [7, 0, -3]

    def test_restacks_after_filter(self):
        """Row order is re-derived from batch.requests each step, so
        filter() churn cannot desynchronize rows and deltas."""
        lm = _RopeTrackingLM()
        gen = _bare_generator(lm)
        batch = _batch_for(
            [_request(1, mx.array([[7]])), _request(2, mx.array([[9]]))]
        )

        batch.filter([1])  # request 1 finished; only uid=2 remains
        gen._arm_decode_rope_deltas(batch)

        assert lm._rope_deltas.reshape(-1).tolist() == [9]

    def test_plain_model_untouched(self):
        lm = _PlainLM()
        gen = _bare_generator(lm)

        gen._arm_decode_rope_deltas(_batch_for([_request(1)]))

        assert not hasattr(lm, "_rope_deltas")


class TestDecodeStepWiring:
    """_next() must broadcast per-row deltas BEFORE the decode forward."""

    def _generator_with_batch(self, lm, requests):
        gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        gen.language_model = lm
        gen.sampler = lambda logprobs: mx.argmax(logprobs, axis=-1)
        gen.active_batch = _batch_for(requests)
        gen.unprocessed_requests = []
        gen._pending_error_responses = []
        gen._aborted_request_ids = set()
        gen._prefill_progress = {}
        gen._stats = MLLMBatchStats()
        gen.stop_tokens = set()
        gen.prefix_cache = None
        gen.completion_batch_size = 16
        return gen

    def test_next_arms_deltas_before_forward(self):
        lm = _RopeTrackingLM()
        # A media row (delta 37) decoding next to a text row (None -> 0).
        requests = [_request(1, mx.array([[37]])), _request(2, None)]
        gen = self._generator_with_batch(lm, requests)

        responses = gen._next()

        assert len(responses) == 2
        assert len(lm.call_time_deltas) == 1
        seen = lm.call_time_deltas[0]
        assert seen is not None
        assert seen.reshape(-1).tolist() == [37, 0]
