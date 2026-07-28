# SPDX-License-Identifier: Apache-2.0
"""clear_runtime_caches()/reset() must clear the vision cache that actually
exists.

The vision cache lives on the MLLMBatchGenerator (``batch_generator.
vision_cache``), but both methods used to dereference ``self.vision_cache`` —
an attribute MLLMScheduler never sets — so the cache-clear route 500'd with
AttributeError on every batched MLLM engine.  Model-free: the generator and
its caches are faked.
"""

from collections import deque

from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig


class _RecordingCache:
    def __init__(self):
        self.clear_calls = 0

    def clear(self):
        self.clear_calls += 1


class _FakeGenerator:
    def __init__(self, vision_cache=None, prefix_cache=None):
        self.vision_cache = vision_cache
        self.prefix_cache = prefix_cache
        self.closed = False

    def close(self):
        self.closed = True


def _bare_scheduler(batch_generator):
    """Construct an MLLMScheduler without running its heavy __init__."""
    sched = MLLMScheduler.__new__(MLLMScheduler)
    sched.config = MLLMSchedulerConfig()
    sched.batch_generator = batch_generator
    sched.requests = {}
    sched.waiting = deque()
    sched.running = {}
    sched.finished_req_ids = set()
    sched.request_id_to_uid = {}
    sched.uid_to_request_id = {}
    sched._detokenizer_pool = {}
    return sched


def test_clear_runtime_caches_clears_generator_vision_cache():
    vision = _RecordingCache()
    prefix = _RecordingCache()
    sched = _bare_scheduler(_FakeGenerator(vision_cache=vision, prefix_cache=prefix))

    cleared = sched.clear_runtime_caches()

    assert cleared == {"vision_cache": True, "prefix_cache": True}
    assert vision.clear_calls == 1
    assert prefix.clear_calls == 1


def test_clear_runtime_caches_without_generator():
    sched = _bare_scheduler(None)

    cleared = sched.clear_runtime_caches()

    assert cleared == {"vision_cache": False, "prefix_cache": False}


def test_clear_runtime_caches_generator_without_prefix_cache():
    vision = _RecordingCache()
    sched = _bare_scheduler(_FakeGenerator(vision_cache=vision, prefix_cache=None))

    cleared = sched.clear_runtime_caches()

    assert cleared == {"vision_cache": True, "prefix_cache": False}
    assert vision.clear_calls == 1


def test_reset_clears_vision_cache_and_closes_generator():
    vision = _RecordingCache()
    gen = _FakeGenerator(vision_cache=vision)
    sched = _bare_scheduler(gen)

    sched.reset()

    assert gen.closed
    assert sched.batch_generator is None
    assert vision.clear_calls == 1


def test_reset_without_generator_does_not_raise():
    sched = _bare_scheduler(None)

    sched.reset()

    assert sched.batch_generator is None
