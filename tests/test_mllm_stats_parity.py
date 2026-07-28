# SPDX-License-Identifier: Apache-2.0
"""Stats parity for the batched MLLM branch (vision series).

The MLLM branch's rail counters (#60-#63) must reach /v1/status and the
Prometheus exporter: scheduler-level keys (steps/queue/ceiling), the
pressure counters folded into the memory_aware_cache block (the dict
metrics.py selects as the active cache), and the engine promote-list.
"""

from types import SimpleNamespace

from vllm_mlx.engine.batched import BatchedEngine
from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig


class _FakeGenerator:
    def __init__(self):
        self.pressure_cache_clears = 3
        self.pressure_evictions = 2
        self.vision_encodes_deferred = 5
        self.prompt_rejections = 1

    def stats(self):
        return SimpleNamespace(to_dict=lambda: {"prompt_tokens": 0})

    def get_vision_cache_stats(self):
        return {"enabled": True}

    def get_prefix_cache_stats(self):
        return {"hits": 7, "misses": 1}


def _bare_scheduler():
    sched = MLLMScheduler.__new__(MLLMScheduler)
    sched.config = MLLMSchedulerConfig()
    sched.batch_generator = _FakeGenerator()
    sched.waiting = __import__("collections").deque()
    sched.running = {}
    sched.requests = {}
    sched.finished_req_ids = set()
    sched.num_requests_processed = 4
    sched.total_prompt_tokens = 10
    sched.total_completion_tokens = 20
    sched._step_count = 42
    sched.queue_cap = 8
    sched.queue_rejections = 6
    sched.max_prompt_tokens = 1000
    sched.prompt_rejections = 2
    return sched


def test_scheduler_stats_carry_rail_counters():
    stats = _bare_scheduler().get_stats()

    assert stats["steps_executed"] == 42
    assert stats["queue_cap"] == 8
    assert stats["queue_rejections"] == 6
    assert stats["max_prompt_tokens"] == 1000
    # scheduler-side (2) + generator-side media-aware (1)
    assert stats["prompt_rejections"] == 3
    assert stats["vision_encodes_deferred"] == 5


def test_pressure_counters_folded_into_cache_block():
    stats = _bare_scheduler().get_stats()

    cache = stats["memory_aware_cache"]
    assert cache["hits"] == 7  # original stats preserved
    assert cache["pressure_evictions"] == 2
    assert cache["pressure_cache_clears"] == 3
    assert cache["admission_deferrals"] == 5  # exporter gauge reuse


def test_engine_promotes_rail_keys():
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._mllm_scheduler = _bare_scheduler()
    engine._engine = None
    engine._loaded = True
    engine._model_name = "test"
    engine._is_mllm = True
    engine._created_at = 0.0
    engine._stream_interval = 1

    stats = engine.get_stats()

    for key in (
        "steps_executed",
        "queue_cap",
        "queue_rejections",
        "max_prompt_tokens",
        "prompt_rejections",
        "vision_encodes_deferred",
        "memory_aware_cache",
    ):
        assert key in stats, f"{key} not promoted"
    assert stats["vision_encodes_deferred"] == 5
