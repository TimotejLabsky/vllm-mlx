"""Cross-thread lifecycle tests for the batched system-KV cache.

The production thread shape: capture/store run on the engine-core executor
(whose default stream is rebound by ``bind_generation_streams``), fetch runs
on the event loop, and the restored cache is consumed back on the executor.
Both Studio live-only bugs in patch #34 lived exactly in these seams and were
invisible to the single-threaded tests. These tests run the real cache
through the real thread shape via :class:`StreamBoundWorker`.
"""

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

from tests.stream_harness import StreamBoundWorker
from vllm_mlx.batched_system_kv import BatchedSystemKV

TOKENS = list(range(1000, 1800))


class _HybridModel:
    def make_cache(self):
        return [KVCache(), ArraysCache(size=2), KVCache()]


class _RotatingModel:
    def make_cache(self):
        return [KVCache(), RotatingKVCache(max_size=128)]


def _fill_kv(cache: KVCache, n: int, seed: float = 0.0):
    keys = mx.arange(n, dtype=mx.float32).reshape(1, 1, n, 1) + seed
    cache.update_and_fetch(keys, keys + 0.5)


def _hybrid_donor(pos: int):
    kv1, rec, kv2 = KVCache(), ArraysCache(size=2), KVCache()
    _fill_kv(kv1, pos)
    _fill_kv(kv2, pos, 100.0)
    rec[0] = mx.full((1, 4), float(pos))
    rec[1] = mx.full((1, 2, 3), float(pos) * 2)
    return [kv1, rec, kv2]


def _rotating_donor(pos: int):
    kv, rot = KVCache(), RotatingKVCache(max_size=128)
    _fill_kv(kv, pos)
    rot.update_and_fetch(
        mx.arange(pos, dtype=mx.float32).reshape(1, 1, pos, 1),
        mx.arange(pos, dtype=mx.float32).reshape(1, 1, pos, 1) + 0.5,
    )
    return [kv, rot]


def _consume(cache):
    """Evaluate every state array — what the executor's merge does."""
    for c in cache:
        st = c.state
        items = st if isinstance(st, (list, tuple)) else [st]
        mx.eval([a for a in items if a is not None])


def test_harness_detects_the_crash_class():
    """Self-test: a lazy graph recorded under the worker's rebound stream
    must fail to evaluate on the main thread. If this stops raising, the
    harness (and MLX's threading model) changed — revisit every test here."""
    with StreamBoundWorker() as w:
        lazy = w.run(lambda: mx.exp(mx.arange(8, dtype=mx.float32)))
        with pytest.raises(RuntimeError, match="[Ss]tream"):
            mx.eval(lazy)


@pytest.mark.parametrize(
    "model_cls,donor",
    [
        (_HybridModel, _hybrid_donor),
        (_RotatingModel, _rotating_donor),
    ],
    ids=["deltanet-hybrid", "sliding-window"],
)
def test_divergent_restore_lifecycle_across_threads(model_cls, donor):
    """capture+store on the executor, fetch on main, consume on the executor.
    The sliding-window case is the exact shape of the gpt-oss live bug
    (tuple checkpoint states left lazy); the hybrid case covers the original
    fetch-slice bug (lazy KV slices crossing the other way)."""
    kv = BatchedSystemKV(model_cls())
    with StreamBoundWorker() as w:
        w.run(lambda: kv.note_scheduled("r1", 0))
        w.run(lambda: kv.capture_segment("r1", 448, donor(448)))
        w.run(lambda: kv.store("r1", TOKENS, donor(len(TOKENS))))

        result = kv.fetch(TOKENS[:600] + [1, 2, 3])  # main thread, like add_request
        assert result is not None
        cache, remaining, pos = result
        assert pos == 448

        w.run(lambda: _consume(cache))  # executor merge


def test_exact_resend_fast_path_across_threads():
    """The d == donor_len fast path restores checkpoint-class state straight
    from the snapshot — those arrays came from the executor's store()."""
    kv = BatchedSystemKV(_HybridModel())
    with StreamBoundWorker() as w:
        w.run(lambda: kv.store("r1", TOKENS, _hybrid_donor(len(TOKENS))))

        result = kv.fetch(TOKENS + [7, 8, 9])
        assert result is not None
        cache, _, pos = result
        assert pos == len(TOKENS)

        w.run(lambda: _consume(cache))


def test_inherited_ladder_survives_thread_roundtrip():
    """Donor-ladder inheritance hands checkpoint state captured on the
    executor to a NEW request's pending ladder via fetch (main thread); the
    follow-up store and a second fetch must stay consumable."""
    kv = BatchedSystemKV(_HybridModel())
    with StreamBoundWorker() as w:
        w.run(lambda: kv.note_scheduled("r1", 0))
        w.run(lambda: kv.capture_segment("r1", 448, _hybrid_donor(448)))
        w.run(lambda: kv.store("r1", TOKENS, _hybrid_donor(len(TOKENS))))

        assert kv.fetch(TOKENS + [7, 8], request_id="r2") is not None
        w.run(lambda: kv.store("r2", TOKENS + [7, 8, 9], _hybrid_donor(len(TOKENS) + 3)))

        result = kv.fetch(TOKENS[:600] + [1, 2, 3], request_id="r3")
        assert result is not None
        assert result[2] == 448
        w.run(lambda: _consume(result[0]))
