# SPDX-License-Identifier: Apache-2.0
"""Fork #138 — ``MemoryAwarePrefixCache`` eviction on hybrid caches.

Ports of upstream waybarrios/vllm-mlx #770 (hybrid entries keep strict
prefixes on publish) and #766 (under pressure, evict a superseded prefix
before the LRU entry), plus the fork's answer to the upstream review of
#766: a hybrid prefix that was used AFTER its longer extension is serving
branches (a shared system prefix) and is not a preferred victim.

Production reach (see PATCHES.md #138): this cache is only constructed on
the MLLM BatchedEngine path (vision routes + REAP-288); the text fleet runs
``batched_system_kv.py`` instead.  No MLX, no weights — mock layers only.
"""

from unittest.mock import MagicMock

from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig

from tests.test_memory_cache import MockArray, MockKVCache

KB = 1024


class _TrimmableKV(MockKVCache):
    """KV-style layer that reports itself rewindable, like ``KVCache``."""

    def __init__(self, nbytes: int):
        super().__init__(nbytes // 2, nbytes // 2)
        self.offset = 8

    def is_trimmable(self) -> bool:
        return True


class _RecurrentLayer:
    """State container without a token dimension, like ``ArraysCache``."""

    def __init__(self, nbytes: int):
        self.state = [MockArray(nbytes)]


def _cache(max_entries: int = 100, max_memory_mb: float = 1.0):
    return MemoryAwarePrefixCache(
        MagicMock(),
        MemoryCacheConfig(
            max_memory_mb=max_memory_mb,
            max_entries=max_entries,
            min_prefix_tokens=1,
        ),
    )


def _kv(nbytes: int = 4 * KB):
    return [_TrimmableKV(nbytes)]


def _hybrid(nbytes: int = 4 * KB):
    return [_TrimmableKV(nbytes // 2), _RecurrentLayer(nbytes // 2)]


# ---------------------------------------------------------------------------
# upstream #770 — hybrid entries keep their strict prefixes on publish
# ---------------------------------------------------------------------------


def test_770_trimmable_cache_still_evicts_strict_prefix():
    cache = _cache()
    assert cache.store([1, 2, 3], _kv())
    assert cache.store([1, 2, 3, 4, 5], _kv())
    assert [1, 2, 3] not in cache
    assert [1, 2, 3, 4, 5] in cache
    assert cache.get_stats()["evictions"] == 1


def test_770_hybrid_cache_keeps_strict_prefix():
    """The measured upstream scenario: request A's store must not delete
    the prewarmed prefix request B (branching after it) needs."""
    cache = _cache()
    prewarmed, turn_a, turn_b = [1, 2, 3], [1, 2, 3, 4, 5], [1, 2, 3, 6, 7]
    assert cache.store(prewarmed, _hybrid())
    assert cache.store(turn_a, _hybrid())
    assert prewarmed in cache and turn_a in cache
    assert cache.get_stats()["evictions"] == 0
    assert cache.store(turn_b, _hybrid())
    assert prewarmed in cache
    assert len(cache) == 3


def test_770_explicit_evict_prefixes_false_unaffected():
    cache = _cache()
    assert cache.store([1, 2], _hybrid(2 * KB))
    assert cache.store([1, 2, 3], _hybrid(2 * KB), evict_prefixes=False)
    assert [1, 2] in cache and [1, 2, 3] in cache


def test_770_hybrid_entries_still_bounded_by_memory():
    cache = _cache(max_memory_mb=0.5)
    for i in range(6):
        tokens = list(range(1, 4 + i))  # each a strict prefix of the next
        assert cache.store(tokens, _hybrid(200 * KB))
    assert cache.memory_usage_mb <= 0.5
    assert cache.get_stats()["evictions"] > 0
    assert len(cache) < 6


def test_138_guard_lives_in_commit_prepared_not_store():
    """Fork: the MLLM path publishes via prepare_store + commit_prepared.
    The exemption must hold there with the DEFAULT evict_prefixes=True, so
    no current or future call site can bypass it (upstream put it in a
    store() that predates the prepare/commit split)."""
    cache = _cache()
    assert cache.commit_prepared(cache.prepare_store([1, 2, 3], _hybrid()))
    entry = cache.prepare_store([1, 2, 3, 4], _hybrid())
    assert cache.commit_prepared(entry)  # default evict_prefixes=True
    assert [1, 2, 3] in cache and [1, 2, 3, 4] in cache


# ---------------------------------------------------------------------------
# upstream #766 — under pressure, a superseded prefix goes before LRU
# ---------------------------------------------------------------------------


def _turns(tag):
    """Three turns of one conversation, each a strict prefix of the next."""
    return [[tag] * (3 * n) for n in (1, 2, 3)]


def test_766_eviction_prefers_superseded_prefix_over_lru():
    """Upstream's test: ~200 KB entries, 1 MB budget => 5 resident, 8 stores
    force 3 evictions.  Strict LRU takes A1, A2 and then A3 — the live turn
    conversation A is about to reuse."""
    cache = _cache(max_memory_mb=1.0)
    a_turns, b_turns = _turns(1), _turns(2)
    # evict_prefixes=False lets superseded turns accumulate, as on hybrid.
    for seq in a_turns + b_turns:
        cache.store(seq, _kv(200 * KB), evict_prefixes=False)
    for seq in _turns(3)[:2]:
        cache.store(seq, _kv(200 * KB), evict_prefixes=False)
    assert cache.get_stats()["evictions"] >= 3
    assert a_turns[-1] in cache, "live entry of the older conversation evicted"
    assert b_turns[-1] in cache, "live entry of the newer conversation evicted"


def test_766_no_superseded_entry_falls_back_to_plain_lru():
    cache = _cache(max_entries=3)
    for seq in ([1, 1], [2, 2], [3, 3], [4, 4]):
        assert cache.store(seq, _kv())
    assert [1, 1] not in cache  # the oldest, nothing subsumes anything
    assert [[2, 2] in cache, [3, 3] in cache, [4, 4] in cache] == [True] * 3


def test_766_lone_shared_prefix_is_not_preferred():
    """A prefix with no longer resident entry is not superseded."""
    cache = _cache(max_entries=3)
    assert cache.store([9, 9], _kv())  # oldest, unrelated
    assert cache.store([1, 2], _hybrid())  # lone shared prefix
    assert cache.store([5, 5], _kv())
    assert cache.store([6, 6], _kv())  # forces one eviction
    assert [9, 9] not in cache
    assert [1, 2] in cache


def test_766_trimmable_superseded_is_preferred_even_when_fresher():
    """A rewindable extension serves everything its prefix serves (LCP /
    supersequence trim), so recency doesn't protect the shorter entry."""
    cache = _cache(max_entries=3)
    assert cache.store([9, 9], _kv())  # oldest, unrelated
    assert cache.store([1, 2, 3, 4], _kv())
    assert cache.store([1, 2], _kv(), evict_prefixes=False)  # fresher prefix
    assert cache.store([7, 7], _kv())  # forces one eviction
    assert [1, 2] not in cache
    assert [9, 9] in cache and [1, 2, 3, 4] in cache


def test_766_victim_is_spilled_to_ssd_and_logged_as_superseded(caplog):
    cache = _cache(max_entries=3)
    tier = MagicMock()
    cache._ssd_tier = tier
    assert cache.store([9, 9], _kv())
    assert cache.store([1, 2, 3, 4], _kv())
    assert cache.store([1, 2], _kv(), evict_prefixes=False)
    with caplog.at_level("DEBUG", logger="vllm_mlx.memory_cache"):
        assert cache.store([7, 7], _kv())
    spilled = [c.args[0] for c in tier.enqueue_spill.call_args_list]
    assert spilled == [(1, 2)]
    assert "[lru_evict:superseded]" in caplog.text


# ---------------------------------------------------------------------------
# fork semantics — the upstream #766 review blocker
# ---------------------------------------------------------------------------


def test_138_hybrid_refreshed_shared_prefix_survives_pressure():
    """The reviewer's regression: store S, S+A, refresh S, apply pressure,
    then S+B must still reuse S.  fetch() refuses to rewind the hybrid S+A,
    so S is the only entry the sibling branch can hit."""
    cache = _cache(max_entries=4)
    s, s_a, s_b = [1, 2, 3], [1, 2, 3, 4, 5], [1, 2, 3, 6, 7]
    assert cache.store([9, 9], _hybrid())  # oldest, unrelated
    assert cache.store(s, _hybrid())
    assert cache.store(s_a, _hybrid())
    assert cache.store(s, _hybrid())  # refresh: S used after S+A
    assert cache.store([8, 8], _hybrid())
    assert cache.store([7, 7], _hybrid())  # pressure: one eviction
    assert cache.get_stats()["evictions"] == 1
    assert [9, 9] not in cache  # plain LRU victim
    assert s in cache and s_a in cache

    cached, remaining = cache.fetch(s_b)
    assert cached is not None, "sibling branch lost the shared prefix"
    assert list(remaining) == [6, 7]


def test_138_hybrid_superseded_dead_turn_goes_first():
    """A hybrid prefix untouched since its extension was stored is a dead
    conversation turn: it goes before an older unrelated LRU entry (the
    upstream #766 win, kept for hybrid)."""
    cache = _cache(max_entries=3)
    assert cache.store([9, 9], _hybrid())  # oldest, unrelated
    assert cache.store([1, 2, 3], _hybrid())  # turn 1
    assert cache.store([1, 2, 3, 4, 5], _hybrid())  # turn 2
    assert cache.store([7, 7], _hybrid())  # forces one eviction
    assert [1, 2, 3] not in cache
    assert [9, 9] in cache and [1, 2, 3, 4, 5] in cache


def test_138_pressure_relief_drop_uses_the_same_victim_order():
    """The MLLM pressure-relief hook drives ``_evict_lru`` directly: one
    call, one entry, the superseded one."""
    cache = _cache()
    assert cache.store([9, 9], _hybrid())
    assert cache.store([1, 2, 3], _hybrid())
    assert cache.store([1, 2, 3, 4], _hybrid())
    cache._evict_lru()
    assert len(cache) == 2
    assert [1, 2, 3] not in cache
    stats = cache.get_stats()
    assert stats["evictions"] == 1
