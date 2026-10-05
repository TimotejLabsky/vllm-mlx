"""PATCHES.md #137 — the fork's cache stack speaks the legacy (pre-mlx-lm#1778)
state protocol on BOTH mlx-lm builds.

Two layers of tests:

- stand-in classes shaped exactly like post-#1778 mlx-lm (``state`` = full
  buffers + scalars, no ``meta_state``, one-arg ``from_state``) exercise the
  conversion paths on any installed mlx-lm;
- real ``mlx_lm`` caches exercise whichever protocol is installed (the old
  pin on the laptop/CI, the new shape in a throwaway venv) — the pair is the
  dual-shape proof, same method as #81.
"""

from __future__ import annotations

import types

import mlx.core as mx
import pytest

from vllm_mlx import cache_state_compat as csc
from vllm_mlx.system_kv import (
    apply_snapshot_states,
    capture_checkpoint_states,
    capture_snapshot_meta,
    classify_layers,
)


def _kv(n, dim=4, heads=1, fill=None):
    if fill is None:
        k = mx.random.normal((1, heads, n, dim))
    else:
        k = mx.full((1, heads, n, dim), float(fill))
    return k, k + 1


# ---------------------------------------------------------------------------
# Stand-ins with the post-#1778 protocol (mirrors mlx-lm ee19be4 cache.py).
# ---------------------------------------------------------------------------


class _BaseCache:
    @property
    def state(self):
        return []

    @state.setter
    def state(self, v):
        if v:
            raise ValueError("no state")

    @classmethod
    def from_state(cls, state):
        obj = cls.__new__(cls)
        obj.state = state
        return obj


class KVCache(_BaseCache):
    step = 256

    def __init__(self):
        self.keys = self.values = None
        self.offset = 0

    def update_and_fetch(self, k, v):
        # Over-allocate like mlx-lm: capacity rounds up to ``step``.
        prev = self.offset
        need = prev + k.shape[2]
        if self.keys is None or need > self.keys.shape[2]:
            cap = ((need + self.step - 1) // self.step) * self.step
            shape = (k.shape[0], k.shape[1], cap, k.shape[3])
            nk, nv = mx.zeros(shape), mx.zeros(shape)
            if self.keys is not None:
                nk[..., :prev, :] = self.keys[..., :prev, :]
                nv[..., :prev, :] = self.values[..., :prev, :]
            self.keys, self.values = nk, nv
        self.offset = need
        self.keys[..., prev:need, :] = k
        self.values[..., prev:need, :] = v
        return self.keys_and_values()

    def keys_and_values(self):
        if self.offset < self.keys.shape[2]:
            return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
        return self.keys, self.values

    @property
    def state(self):
        return self.keys, self.values, self.offset

    @state.setter
    def state(self, v):
        self.keys, self.values, self.offset = v


class RotatingKVCache(_BaseCache):
    def __init__(self, max_size, keep=0):
        self.keys = self.values = None
        self.offset = 0
        self.max_size = max_size
        self.keep = keep
        self._idx = 0

    def keys_and_values(self):
        if self.offset < self.keys.shape[2]:
            return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
        return self.keys, self.values

    @property
    def state(self):
        return self.keys, self.values, self.offset, self.keep, self.max_size, self._idx

    @state.setter
    def state(self, v):
        self.keys, self.values, self.offset, self.keep, self.max_size, self._idx = v


class QuantizedKVCache(KVCache):
    def __init__(self, group_size=64, bits=8):
        super().__init__()
        self.group_size, self.bits = group_size, bits

    @property
    def state(self):
        return self.keys, self.values, self.offset, self.group_size, self.bits

    @state.setter
    def state(self, v):
        self.keys, self.values, self.offset, self.group_size, self.bits = v


class ArraysCache(_BaseCache):
    def __init__(self, size, left_padding=None):
        self.cache = [None] * size
        self.left_padding = mx.array(left_padding) if left_padding else None
        self.lengths = None

    @property
    def state(self):
        return self.cache, self.left_padding, self.lengths

    @state.setter
    def state(self, v):
        self.cache, self.left_padding, self.lengths = v


class MambaLike(ArraysCache):
    """Subclass inheriting ArraysCache.state (like the fork's
    BatchMambaCache): must convert like its base."""


class CacheList(_BaseCache):
    def __init__(self, *caches):
        self.caches = list(caches)

    @property
    def state(self):
        return [(c.state, type(c).__name__) for c in self.caches]

    @state.setter
    def state(self, v):
        self.caches = [globals()[n].from_state(s) for s, n in v]


class LegacyVlmCache:
    """An mlx-vlm-style cache still on the legacy pair: must pass through."""

    def __init__(self):
        self.keys = self.values = None
        self.meta = ""

    @property
    def state(self):
        return self.keys, self.values

    @state.setter
    def state(self, v):
        self.keys, self.values = v

    @property
    def meta_state(self):
        return self.meta

    @meta_state.setter
    def meta_state(self, v):
        self.meta = v

    @classmethod
    def from_state(cls, state, meta_state):
        obj = cls()
        obj.state = state
        obj.meta_state = meta_state
        return obj


_FAKE_LM = types.SimpleNamespace(
    _BaseCache=_BaseCache,
    KVCache=KVCache,
    RotatingKVCache=RotatingKVCache,
    QuantizedKVCache=QuantizedKVCache,
    ArraysCache=ArraysCache,
    CacheList=CacheList,
)


@pytest.fixture
def new_shape(monkeypatch):
    """Point the compat layer at the post-#1778 stand-ins."""
    monkeypatch.setattr(csc, "_lm_cache", _FAKE_LM)
    monkeypatch.setattr(csc, "NEW_STATE_SHAPE", True)
    return _FAKE_LM


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def test_detects_new_shape_on_the_stand_in_base():
    assert csc._detect_new_state_shape(_BaseCache) is True


def test_detects_legacy_shape_on_a_two_arg_base():
    class Old:
        meta_state = ""

        @classmethod
        def from_state(cls, state, meta_state):
            return cls()

    assert csc._detect_new_state_shape(Old) is False
    assert csc._detect_new_state_shape(None) is False


def test_detection_matches_the_installed_mlx_lm():
    from mlx_lm.models.cache import _BaseCache as Installed

    expected = not hasattr(Installed, "meta_state")
    assert csc.NEW_STATE_SHAPE is expected


# ---------------------------------------------------------------------------
# New-shape conversion (stand-ins)
# ---------------------------------------------------------------------------


def test_kvcache_legacy_view_is_sliced_and_restore_is_exactly_full(new_shape):
    c = KVCache()
    c.update_and_fetch(*_kv(5))
    assert c.state[0].shape[2] == 256  # native: the padded buffer
    k, v = csc.legacy_state(c)
    assert k.shape[2] == 5 and v.shape[2] == 5
    assert csc.legacy_meta_state(c) == ""

    r = KVCache()
    csc.set_legacy_state(r, (k, v))
    assert r.offset == 5 and r.keys.shape[2] == 5
    with pytest.raises(ValueError):
        csc.set_legacy_state(KVCache(), (k, v), ("1",))


def test_kvcache_restores_from_one_snapshot_do_not_alias(new_shape):
    """The load-bearing reason for the legacy view: a NATIVE post-#1778
    round trip hands two caches the same over-allocated buffer, so their
    next writes clobber each other."""
    donor = KVCache()
    donor.update_and_fetch(*_kv(3, fill=1))

    # Native round trip: corrupts (documents the hazard).
    snap = donor.state
    a, b = KVCache(), KVCache()
    a.state, b.state = snap, snap
    a.update_and_fetch(*_kv(1, fill=7))
    b.update_and_fetch(*_kv(1, fill=9))
    assert float(a.keys_and_values()[0][0, 0, 3, 0].item()) == 9.0

    # Legacy round trip: isolated.
    snap = csc.legacy_state(donor)
    a, b = KVCache(), KVCache()
    csc.set_legacy_state(a, snap)
    csc.set_legacy_state(b, snap)
    a.update_and_fetch(*_kv(1, fill=7))
    b.update_and_fetch(*_kv(1, fill=9))
    assert float(a.keys_and_values()[0][0, 0, 3, 0].item()) == 7.0
    assert float(b.keys_and_values()[0][0, 0, 3, 0].item()) == 9.0


def test_rotating_meta_round_trip(new_shape):
    c = RotatingKVCache(max_size=8, keep=2)
    c.keys, c.values = _kv(8)
    c.offset, c._idx = 15, 7
    st = csc.legacy_state(c)
    meta = csc.legacy_meta_state(c)
    assert meta == ("2", "8", "15", "7")
    assert len(st) == 2

    r = csc.from_legacy_state(RotatingKVCache, st, meta)
    assert (r.keep, r.max_size, r.offset, r._idx) == (2, 8, 15, 7)
    assert r.keys is st[0]

    # State WITHOUT meta leaves the ring indices alone (patch #12 control).
    bare = RotatingKVCache(max_size=8)
    csc.set_legacy_state(bare, st)
    assert (bare.offset, bare._idx) == (0, 0)


def test_quantized_meta_round_trip(new_shape):
    c = QuantizedKVCache(group_size=32, bits=4)
    c.keys, c.values = _kv(6)
    c.offset = 6
    meta = csc.legacy_meta_state(c)
    assert meta == ("6", "32", "4")
    r = csc.from_legacy_state(QuantizedKVCache, csc.legacy_state(c), meta)
    assert (r.offset, r.group_size, r.bits) == (6, 32, 4)


def test_arrays_cache_none_metadata_maps_to_the_1632_empty_arrays(new_shape):
    c = MambaLike(2)
    c.cache = [mx.ones((1, 2)), mx.zeros((1, 3))]
    st = csc.legacy_state(c)
    assert isinstance(st, tuple) and st[0] is c.cache
    assert st[1].size == 0 and st[2].size == 0
    assert csc.legacy_meta_state(c) == ""

    r = MambaLike(2)
    csc.set_legacy_state(r, (list(st[0]), st[1], st[2]))
    assert r.left_padding is None and r.lengths is None
    assert r.cache[0] is c.cache[0]

    # Real metadata survives; a pre-1632 bare list is accepted too.
    lp = mx.array([1, 0])
    csc.set_legacy_state(r, ([None, None], lp, mx.array([])))
    assert r.left_padding is lp and r.lengths is None
    csc.set_legacy_state(r, [mx.ones((1,)), mx.ones((1,))])
    assert r.left_padding is None and len(r.cache) == 2


def test_cache_list_round_trip(new_shape):
    kv = KVCache()
    kv.update_and_fetch(*_kv(4))
    rot = RotatingKVCache(max_size=8)
    rot.keys, rot.values = _kv(3)
    rot.offset, rot._idx = 3, 3
    cl = CacheList(kv, rot)

    st = csc.legacy_state(cl)
    meta = csc.legacy_meta_state(cl)
    assert meta[0] == ["KVCache", "RotatingKVCache"]
    assert meta[1][0] == "" and meta[1][1] == ("0", "8", "3", "3")
    assert st[0][0].shape[2] == 4

    r = csc.from_legacy_state(CacheList, st, meta)
    assert [type(c) for c in r.caches] == [KVCache, RotatingKVCache]
    assert r.caches[0].offset == 4 and r.caches[1]._idx == 3


def test_legacy_protocol_classes_pass_through(new_shape):
    c = LegacyVlmCache()
    c.keys, c.values = _kv(2)
    c.meta = ("x",)
    assert csc.legacy_state(c) == (c.keys, c.values)
    assert csc.legacy_meta_state(c) == ("x",)
    assert csc.has_meta_state(c)
    r = csc.from_legacy_state(LegacyVlmCache, csc.legacy_state(c), ("y",))
    assert r.meta == ("y",)


def test_system_kv_classifies_and_restores_through_the_compat_layer(new_shape):
    """classify/capture/apply on post-#1778 caches: attention stays ``trim``
    (the #81 KV tripwire's failure mode — every layer opaque — is gone)."""
    kv = KVCache()
    kv.update_and_fetch(*_kv(6))
    arr = ArraysCache(2)
    arr.cache = [mx.ones((1, 2)), mx.ones((1, 2))]
    rot = RotatingKVCache(max_size=4)
    rot.keys, rot.values = _kv(4)
    rot.offset, rot._idx = 6, 2
    live = [kv, arr, rot]

    assert classify_layers(live) == ["trim", "ckpt", "ckpt"]
    metas = capture_snapshot_meta(live)
    assert metas == [None, None, ("0", "4", "6", "2")]
    states, cmetas = capture_checkpoint_states(live)
    assert sorted(states) == [1, 2] and cmetas[2] == ("0", "4", "6", "2")

    fresh = [KVCache(), ArraysCache(2), RotatingKVCache(max_size=4)]
    apply_snapshot_states(fresh, [csc.legacy_state(c) for c in live], metas)
    assert fresh[0].offset == 6 and fresh[0].keys.shape[2] == 6
    assert fresh[1].left_padding is None
    assert (fresh[2].offset, fresh[2]._idx) == (6, 2)


# ---------------------------------------------------------------------------
# Real mlx-lm caches — whichever protocol is installed.
# ---------------------------------------------------------------------------


def test_real_kvcache_restores_do_not_alias():
    from mlx_lm.models.cache import KVCache as RealKV

    donor = RealKV()
    donor.update_and_fetch(*_kv(3, fill=1))
    snap = csc.legacy_state(donor)
    a, b = RealKV(), RealKV()
    csc.set_legacy_state(a, snap)
    csc.set_legacy_state(b, snap)
    a.update_and_fetch(*_kv(1, fill=7))
    b.update_and_fetch(*_kv(1, fill=9))
    ka = csc.legacy_state(a)[0]
    kb = csc.legacy_state(b)[0]
    assert ka.shape[2] == kb.shape[2] == 4
    assert float(ka[0, 0, 3, 0].item()) == 7.0
    assert float(kb[0, 0, 3, 0].item()) == 9.0
    assert float(ka[0, 0, 0, 0].item()) == 1.0


def test_real_rotating_round_trip_continues_identically():
    from mlx_lm.models.cache import RotatingKVCache as RealRot

    steps = [_kv(1) for _ in range(20)]
    donor = RealRot(max_size=8)
    for k, v in steps[:15]:
        donor.update_and_fetch(k, v)
    st, meta = csc.legacy_state(donor), csc.legacy_meta_state(donor)
    truth = [donor.update_and_fetch(k, v) for k, v in steps[15:]]

    r = csc.from_legacy_state(RealRot, st, meta)
    replay = [r.update_and_fetch(k, v) for k, v in steps[15:]]
    for (tk, tv), (rk, rv) in zip(truth, replay):
        assert mx.array_equal(tk, rk) and mx.array_equal(tv, rv)


def test_real_arrays_cache_legacy_shape_is_the_1632_tuple():
    from mlx_lm.models.cache import ArraysCache as RealArrays

    c = RealArrays(2)
    c.cache = [mx.ones((1, 2)), mx.ones((1, 2))]
    st = csc.legacy_state(c)
    assert isinstance(st, tuple) and len(st) == 3 and st[0] is c.cache
    assert st[1].size == 0 and st[2].size == 0
    r = csc.from_legacy_state(RealArrays, (list(st[0]), st[1], st[2]), "")
    assert r.left_padding is None and r.lengths is None


def test_real_cache_list_round_trip():
    from mlx_lm.models.cache import CacheList as RealList
    from mlx_lm.models.cache import KVCache as RealKV

    a, b = RealKV(), RealKV()
    a.update_and_fetch(*_kv(3))
    b.update_and_fetch(*_kv(5))
    cl = RealList(a, b)
    r = csc.from_legacy_state(RealList, csc.legacy_state(cl), csc.legacy_meta_state(cl))
    assert [c.offset for c in r.caches] == [3, 5]


def test_old_pin_helpers_are_the_bare_expressions():
    """On the legacy protocol the helpers ARE the pre-#137 expressions."""
    if csc.NEW_STATE_SHAPE:
        pytest.skip("installed mlx-lm speaks the #1778 protocol")
    from mlx_lm.models.cache import KVCache as RealKV
    from mlx_lm.models.cache import RotatingKVCache as RealRot

    c = RealKV()
    c.update_and_fetch(*_kv(4))
    s1, s2 = c.state, csc.legacy_state(c)
    assert all(mx.array_equal(x, y) for x, y in zip(s1, s2))
    rot = RealRot(max_size=8)
    rot.update_and_fetch(*_kv(3))
    assert csc.legacy_meta_state(rot) == rot.meta_state
    assert csc.legacy_meta_state(c, "") == c.meta_state
