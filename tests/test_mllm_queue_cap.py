# SPDX-License-Identifier: Apache-2.0
"""Queue cap + pre-stream probe on the batched MLLM branch (#39-parity).

VLLM_MLX_BATCHED_MAX_QUEUE was never read by MLLMScheduler (add_request
appended unconditionally), and raise_if_serialized_busy returned early
when self._engine was None — so on the MLLM branch a queue-capped 503
was impossible, and streaming requests could never be rejected before
SSE headers went out.
"""

import inspect
from collections import deque
from types import SimpleNamespace

import pytest

from vllm_mlx.engine.base import EngineBusy
from vllm_mlx.engine.batched import BatchedEngine
from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig


def _bare_scheduler(queue_cap=0):
    sched = MLLMScheduler.__new__(MLLMScheduler)
    sched.config = MLLMSchedulerConfig()
    sched.queue_cap = queue_cap
    sched.queue_rejections = 0
    sched.waiting = deque()
    sched.requests = {}
    sched.processor = SimpleNamespace(tokenizer=None)
    return sched


class TestAddRequestQueueCap:
    def test_rejects_at_capacity(self):
        sched = _bare_scheduler(queue_cap=2)
        sched.waiting.extend([object(), object()])

        with pytest.raises(EngineBusy):
            sched.add_request(prompt="hi")

        assert sched.queue_rejections == 1
        assert len(sched.requests) == 0  # no state created for the reject

    def test_admits_below_capacity(self):
        sched = _bare_scheduler(queue_cap=2)
        sched.waiting.append(object())

        rid = sched.add_request(prompt="hi")

        assert rid in sched.requests
        assert sched.queue_rejections == 0

    def test_inert_when_cap_unset(self):
        sched = _bare_scheduler(queue_cap=0)
        sched.waiting.extend(object() for _ in range(50))

        rid = sched.add_request(prompt="hi")

        assert rid in sched.requests

    def test_env_parsed_at_init(self, monkeypatch):
        monkeypatch.setenv("VLLM_MLX_BATCHED_MAX_QUEUE", "7")
        monkeypatch.setattr(MLLMScheduler, "_get_stop_tokens", lambda self: set())
        sched = MLLMScheduler(
            model=SimpleNamespace(),
            processor=SimpleNamespace(tokenizer=None),
            config=MLLMSchedulerConfig(),
        )
        assert sched.queue_cap == 7
        assert sched.queue_rejections == 0


class TestPreStreamProbeMllm:
    def _engine(self, mllm_scheduler):
        engine = BatchedEngine.__new__(BatchedEngine)
        engine._engine = None
        engine._mllm_scheduler = mllm_scheduler
        return engine

    def test_probe_503s_on_mllm_queue_cap(self):
        sched = _bare_scheduler(queue_cap=1)
        sched.waiting.append(object())
        engine = self._engine(sched)

        with pytest.raises(EngineBusy):
            engine.raise_if_serialized_busy("req-1")

        assert sched.queue_rejections == 1

    def test_probe_noop_below_cap(self):
        sched = _bare_scheduler(queue_cap=4)
        engine = self._engine(sched)

        engine.raise_if_serialized_busy("req-1")  # no raise

    def test_probe_noop_without_any_scheduler(self):
        engine = self._engine(None)

        engine.raise_if_serialized_busy("req-1")  # no raise

    def test_probe_signature_matches_server_contract(self):
        """The server passes the request id POSITIONALLY (day-one-incident
        regression pin, same as test_batched_flip_enablement.py)."""
        sig = inspect.signature(BatchedEngine.raise_if_serialized_busy)
        params = list(sig.parameters.values())[1:]  # drop self
        assert params, "probe must accept a positional request id"
        assert params[0].kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
