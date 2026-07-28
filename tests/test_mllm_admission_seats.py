# SPDX-License-Identifier: Apache-2.0
"""Admission seats for atomic vision encodes (vision series).

``prefill_batch_size`` was dead config — stored but never used to bound
anything, so up to 16 queue slots could stack back-to-back ATOMIC vision
encodes (full VLM forward each) in a single step. Selection now clips
media rows to the budget, defers all non-head media while the allocator
is over the watermark, always admits the queue head (progress guarantee),
and consumes the queue by uid so deferred rows keep their place.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from vllm_mlx.memory_pressure import PressureManager
from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator, MLLMBatchRequest

pytestmark = pytest.mark.skipif(mx is None, reason="MLX not available")


class _FixedPressure(PressureManager):
    def __init__(self, hot):
        super().__init__(watermark_pct=0)
        self._hot = hot

    def under_pressure(self):
        return self._hot


def _req(uid, media=False):
    return MLLMBatchRequest(
        uid=uid,
        request_id=f"req-{uid}",
        prompt="p",
        images=["x.png"] if media else None,
    )


def _generator(hot=False, prefill_batch_size=2, completion_batch_size=16):
    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen._pressure = _FixedPressure(hot)
    gen.prefill_batch_size = prefill_batch_size
    gen.completion_batch_size = completion_batch_size
    gen.vision_encodes_deferred = 0
    return gen


class TestSelectPrefillBatch:
    def test_text_rows_unconstrained(self):
        gen = _generator()
        candidates = [_req(i) for i in range(10)]

        selected, deferred = gen._select_prefill_batch(candidates)

        assert [r.uid for r in selected] == list(range(10))
        assert deferred == 0

    def test_media_rows_clipped_to_budget(self):
        gen = _generator(prefill_batch_size=2)
        candidates = [_req(i, media=True) for i in range(5)]

        selected, deferred = gen._select_prefill_batch(candidates)

        assert [r.uid for r in selected] == [0, 1]
        assert deferred == 0  # budget clip is not a pressure deferral

    def test_hot_defers_all_but_head_media(self):
        gen = _generator(hot=True, prefill_batch_size=4)
        candidates = [_req(0, media=True), _req(1, media=True), _req(2), _req(3, media=True)]

        selected, deferred = gen._select_prefill_batch(candidates)

        # Head media admitted (progress guarantee), text admitted,
        # other media deferred.
        assert [r.uid for r in selected] == [0, 2]
        assert deferred == 2

    def test_head_always_admitted_even_hot(self):
        gen = _generator(hot=True)
        candidates = [_req(0, media=True)]

        selected, deferred = gen._select_prefill_batch(candidates)

        assert [r.uid for r in selected] == [0]
        assert deferred == 0

    def test_empty(self):
        gen = _generator()
        assert gen._select_prefill_batch([]) == ([], 0)


class TestNextConsumesByUid:
    def _bare(self, hot, requests):
        gen = _generator(hot=hot, prefill_batch_size=1)
        gen.unprocessed_requests = list(requests)
        gen._pending_error_responses = []
        gen.active_batch = None
        gen._stats = SimpleNamespace(prompt_time=0.0, generation_time=0.0)
        processed = []

        def fake_process(reqs):
            processed.extend(r.uid for r in reqs)
            return None

        gen._process_prompts = fake_process
        return gen, processed

    def test_deferred_media_stays_queued_in_place(self):
        reqs = [_req(0, media=True), _req(1, media=True), _req(2)]
        gen, processed = self._bare(hot=True, requests=reqs)

        gen._next()

        # Head media + text processed; second media deferred, still queued.
        assert processed == [0, 2]
        assert [r.uid for r in gen.unprocessed_requests] == [1]
        assert gen.vision_encodes_deferred == 1

    def test_budget_clip_keeps_rows_queued(self):
        reqs = [_req(0, media=True), _req(1, media=True), _req(2, media=True)]
        gen, processed = self._bare(hot=False, requests=reqs)

        gen._next()

        assert processed == [0]  # budget 1
        assert [r.uid for r in gen.unprocessed_requests] == [1, 2]
        assert gen.vision_encodes_deferred == 0
