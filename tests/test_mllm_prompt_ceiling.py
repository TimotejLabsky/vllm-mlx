# SPDX-License-Identifier: Apache-2.0
"""Prompt-token ceiling on the batched MLLM branch (#50-parity).

Two gates: a text-token estimate at add_request (load-bearing tokenize —
previously `except: pass` silently zeroed the count), and a media-aware
gate on the TRUE post-processor token count at preprocess end (vision
placeholder tokens dominate multi-image prompts). The media-aware breach
is carried as error_kind="prompt_too_long" and translated back into
PromptTooLong for non-stream callers.
"""

import asyncio
from collections import deque
from types import SimpleNamespace

import mlx.core as mx
import pytest

from vllm_mlx.engine.base import PromptTooLong
from vllm_mlx.mllm_batch_generator import (
    MLLMBatchGenerator,
    MLLMBatchRequest,
    _error_kind_for,
)
from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig
from vllm_mlx.request import RequestOutput


class _Tokenizer:
    def encode(self, text):
        return list(range(len(text.split())))


def _bare_scheduler(max_prompt_tokens=0):
    sched = MLLMScheduler.__new__(MLLMScheduler)
    sched.config = MLLMSchedulerConfig()
    sched.queue_cap = 0
    sched.queue_rejections = 0
    sched.max_prompt_tokens = max_prompt_tokens
    sched.prompt_rejections = 0
    sched.waiting = deque()
    sched.requests = {}
    sched.processor = SimpleNamespace(tokenizer=_Tokenizer())
    return sched


class TestAddRequestCeiling:
    def test_rejects_over_ceiling(self):
        sched = _bare_scheduler(max_prompt_tokens=3)

        with pytest.raises(PromptTooLong):
            sched.add_request(prompt="one two three four five")

        assert sched.prompt_rejections == 1
        assert len(sched.requests) == 0

    def test_admits_under_ceiling(self):
        sched = _bare_scheduler(max_prompt_tokens=10)

        rid = sched.add_request(prompt="short prompt")

        assert rid in sched.requests
        assert sched.prompt_rejections == 0

    def test_inert_when_unset(self):
        sched = _bare_scheduler(max_prompt_tokens=0)

        rid = sched.add_request(prompt=" ".join(["w"] * 1000))

        assert rid in sched.requests

    def test_tokenizer_failure_does_not_disarm_silently(self, caplog):
        sched = _bare_scheduler(max_prompt_tokens=3)

        class _Broken:
            def encode(self, text):
                raise RuntimeError("boom")

        sched.processor = SimpleNamespace(tokenizer=_Broken())

        with caplog.at_level("WARNING"):
            rid = sched.add_request(prompt="one two three four five")

        # Request admitted (count unknown) but loudly, not silently.
        assert rid in sched.requests
        assert any("prompt ceiling" in r.message for r in caplog.records)


class TestGeneratorMediaAwareCeiling:
    def _generator(self, max_prompt_tokens):
        gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        gen.max_prompt_tokens = max_prompt_tokens
        gen.prompt_rejections = 0
        return gen

    def _request(self, n_tokens):
        req = MLLMBatchRequest(uid=1, request_id="r", prompt="p")
        req.input_ids = mx.zeros((1, n_tokens), dtype=mx.int32)
        return req

    def test_rejects_over_ceiling(self):
        gen = self._generator(8)

        with pytest.raises(PromptTooLong):
            gen._check_prompt_ceiling(self._request(9))

        assert gen.prompt_rejections == 1

    def test_admits_at_ceiling(self):
        gen = self._generator(8)

        gen._check_prompt_ceiling(self._request(8))  # no raise

    def test_inert_when_unset(self):
        gen = self._generator(0)

        gen._check_prompt_ceiling(self._request(10_000))  # no raise


class TestErrorKindPlumbing:
    def test_error_kind_for_prompt_too_long(self):
        assert _error_kind_for(PromptTooLong("x")) == "prompt_too_long"

    def test_error_kind_for_generic(self):
        assert _error_kind_for(ValueError("x")) is None

    def test_generate_translates_prompt_too_long(self):
        sched = _bare_scheduler()

        async def fake_add(**kwargs):
            return "req-1"

        async def fake_stream(request_id):
            yield RequestOutput(
                request_id=request_id,
                finished=True,
                finish_reason="error",
                error_kind="prompt_too_long",
            )

        sched.add_request_async = fake_add
        sched.stream_outputs = fake_stream

        with pytest.raises(PromptTooLong):
            asyncio.run(sched.generate(prompt="p"))

    def test_generate_passes_through_generic_error(self):
        sched = _bare_scheduler()

        async def fake_add(**kwargs):
            return "req-1"

        async def fake_stream(request_id):
            yield RequestOutput(
                request_id=request_id,
                finished=True,
                finish_reason="error",
            )

        sched.add_request_async = fake_add
        sched.stream_outputs = fake_stream

        out = asyncio.run(sched.generate(prompt="p"))
        assert out.finish_reason == "error"
        assert out.error_kind is None
