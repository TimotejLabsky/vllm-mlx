# SPDX-License-Identifier: Apache-2.0
"""#48/#53-style memory-pressure relief on the batched MLLM branch.

The MLLM stack had none of the LLM branch's relief: no watermark hook, an
unbounded atomic vision encode, and a buffer-cache clear cadence that never
returned memory under a breach. These tests pin the generator-side relief
(via the #59 PressureManager seam) with the same fake-allocator pattern as
tests/test_batched_flip_enablement.py.
"""

from collections import OrderedDict
from types import SimpleNamespace

import mlx.core as mx
import pytest

from vllm_mlx.memory_pressure import PressureManager
from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator
from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig

pytestmark = pytest.mark.skipif(mx is None, reason="MLX not available")


def _mb(n):
    return n * 1024 * 1024


class _FakePrefixCache:
    """LRU surface the relief provider drives (entries + _evict_lru)."""

    def __init__(self, n_entries):
        self._entries = OrderedDict((i, object()) for i in range(n_entries))
        self.evict_calls = 0

    def _evict_lru(self):
        self.evict_calls += 1
        if self._entries:
            self._entries.popitem(last=False)


class _FakeVisionCache:
    def __init__(self, populated=True):
        self._pixel_cache = {"k": 1} if populated else {}
        self._pixel_only_cache = {}
        self._encoding_cache = {}
        self.clear_calls = 0

    def clear(self):
        self.clear_calls += 1
        self._pixel_cache = {}
        self._pixel_only_cache = {}
        self._encoding_cache = {}


def _generator(watermark_pct, prefix_entries=0, vision_populated=False):
    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen._pressure = PressureManager(watermark_pct)
    gen.pressure_cache_clears = 0
    gen.pressure_evictions = 0
    gen.vision_encodes_deferred = 0
    gen.prefix_cache = _FakePrefixCache(prefix_entries) if prefix_entries else None
    gen.vision_cache = _FakeVisionCache(populated=vision_populated)
    return gen


def _fake_allocator(monkeypatch, active_mb, ceiling_mb=100):
    """Fake mx memory API; returns the mutable mem dict."""
    mem = {"active": _mb(active_mb), "peak": _mb(active_mb), "clears": 0}
    monkeypatch.setattr(mx, "get_active_memory", lambda: mem["active"])
    monkeypatch.setattr(mx, "get_peak_memory", lambda: mem["peak"])
    monkeypatch.setattr(
        mx, "reset_peak_memory", lambda: mem.__setitem__("peak", mem["active"])
    )
    monkeypatch.setattr(
        mx, "clear_cache", lambda: mem.__setitem__("clears", mem["clears"] + 1)
    )
    monkeypatch.setattr(
        mx,
        "device_info",
        lambda: {"max_recommended_working_set_size": _mb(ceiling_mb)},
    )
    return mem


def test_relief_inert_without_watermark(monkeypatch):
    mem = _fake_allocator(monkeypatch, active_mb=95)
    gen = _generator(0, prefix_entries=3)

    assert gen.maybe_relieve_pressure() == 0
    assert gen.pressure_cache_clears == 0
    assert mem["clears"] == 0
    assert len(gen.prefix_cache._entries) == 3


def test_relief_noop_under_watermark(monkeypatch):
    _fake_allocator(monkeypatch, active_mb=50)
    gen = _generator(90, prefix_entries=3)

    assert gen.maybe_relieve_pressure() == 0
    assert gen.pressure_cache_clears == 0


def test_peak_breach_evicts_prefix_lru_until_under(monkeypatch):
    mem = _fake_allocator(monkeypatch, active_mb=95)
    gen = _generator(90, prefix_entries=3)

    # First eviction brings active back under the 90 MB threshold.
    orig_evict = gen.prefix_cache._evict_lru

    def evict_and_relax():
        orig_evict()
        mem["active"] = _mb(50)

    gen.prefix_cache._evict_lru = evict_and_relax

    assert gen.maybe_relieve_pressure() == 1
    assert gen.pressure_cache_clears == 1
    assert gen.pressure_evictions == 1
    assert len(gen.prefix_cache._entries) == 2


def test_breach_with_empty_caches_still_clears_buffer_cache(monkeypatch):
    """#53: the clear must fire even when there is nothing to evict."""
    mem = _fake_allocator(monkeypatch, active_mb=95)
    gen = _generator(90, prefix_entries=0, vision_populated=False)

    assert gen.maybe_relieve_pressure() == 0
    assert gen.pressure_cache_clears == 1
    assert mem["clears"] >= 1


def test_vision_cache_dropped_after_prefix_exhausted(monkeypatch):
    _fake_allocator(monkeypatch, active_mb=95)  # active never relaxes
    gen = _generator(90, prefix_entries=2, vision_populated=True)

    evicted = gen.maybe_relieve_pressure()

    # 2 prefix entries + 1 vision-cache clear, then nothing left.
    assert evicted == 3
    assert len(gen.prefix_cache._entries) == 0
    assert gen.vision_cache.clear_calls == 1


def test_scheduler_step_head_calls_relief():
    calls = []

    fake_gen = SimpleNamespace(
        process_pending_removals=lambda: calls.append("removals"),
        maybe_relieve_pressure=lambda: calls.append("relief"),
    )

    sched = MLLMScheduler.__new__(MLLMScheduler)
    sched.config = MLLMSchedulerConfig()
    sched.batch_generator = fake_gen
    sched.waiting = __import__("collections").deque()
    sched.running = {}
    sched.requests = {}
    sched.finished_req_ids = set()
    sched._clear_cache_interval = 32
    sched._step_count = 0

    output = sched.step()

    assert calls == ["removals", "relief"]
    assert output.scheduled_request_ids == []
