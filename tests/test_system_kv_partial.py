# SPDX-License-Identifier: Apache-2.0
"""Tests for the checkpointed partial-prefix restore (system-kv-partial-restore).

Covers the pure-python planning/selection helpers, real-KVCache trim restore,
checkpoint thinning, SSD format v2 round-trip (snapshot + checkpoints),
v1 back-compat, and the shared-prefix (divergent entry) index lookup.

Requires mlx (runs on Apple Silicon). Runnable via pytest or directly:
``python tests/test_system_kv_partial.py``.
"""

from __future__ import annotations

import shutil
import tempfile

import mlx.core as mx

from vllm_mlx.system_kv import (
    SystemKVManager,
    append_checkpoint,
    build_partial_restore_states,
    capture_recurrent_states,
    common_prefix_len,
    select_restore_pos,
)
from vllm_mlx.system_kv_ssd import (
    SystemKVSSDConfig,
    SystemKVSSDStore,
    flatten_checkpoints,
    flatten_snapshot,
    unflatten_checkpoints,
    _snapshot_nbytes,
)


def _kv_layer(seq: int, dim: int = 8, heads: int = 2):
    k = mx.random.normal((1, heads, seq, dim)).astype(mx.bfloat16)
    v = mx.random.normal((1, heads, seq, dim)).astype(mx.bfloat16)
    mx.eval(k, v)
    return (k, v)


def _rec_states(tag: float):
    st = [mx.full((1, 4, 8), tag).astype(mx.float32),
          mx.full((1, 2, 16), tag + 0.5).astype(mx.float32)]
    mx.eval(st)
    return st


def _hybrid_snapshot(seq: int):
    """layers: 0=kv, 1=recurrent, 2=kv"""
    snap = [_kv_layer(seq), _rec_states(float(seq)), _kv_layer(seq)]
    return snap


def _arrays_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and bool((a == b).all())


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_common_prefix_len():
    assert common_prefix_len([1, 2, 3], [1, 2, 4]) == 2
    assert common_prefix_len([1, 2], [1, 2, 3]) == 2
    assert common_prefix_len([], [1]) == 0
    assert common_prefix_len([9], [1]) == 0


def test_append_checkpoint_thinning():
    cps = []
    for pos in range(1, 13):
        cps = append_checkpoint(cps, pos * 100, {"states": pos}, capacity=4)
    # bounded
    assert len(cps) <= 5  # capacity + the always-kept newest
    # sorted ascending, newest retained
    positions = [c["pos"] for c in cps]
    assert positions == sorted(positions)
    assert positions[-1] == 1200
    # non-monotonic appends are ignored
    before = list(cps)
    cps = append_checkpoint(cps, 50, {}, capacity=4)
    assert cps == before


def test_select_restore_pos_pure_attention():
    plan = {"d": 700, "snapshot": [_kv_layer(800)], "checkpoints": [],
            "has_recurrent": False, "donor_len": 800}
    # exact divergence point, capped by the new extended length
    assert select_restore_pos(plan, 1000)[0] == 700
    assert select_restore_pos(plan, 300)[0] == 300
    assert select_restore_pos(plan, 0) == (0, None)


def test_select_restore_pos_hybrid():
    snap = _hybrid_snapshot(800)
    cps = [
        {"pos": 256, "states": {1: _rec_states(256.0)}},
        {"pos": 512, "states": {1: _rec_states(512.0)}},
    ]
    plan = {"d": 700, "snapshot": snap, "checkpoints": cps,
            "has_recurrent": True, "donor_len": 800}
    pos, states = select_restore_pos(plan, 10_000)
    assert pos == 512 and states is cps[1]["states"]
    # cap below the best checkpoint falls back to the earlier one
    pos, states = select_restore_pos(plan, 400)
    assert pos == 256 and states is cps[0]["states"]
    # cap below every checkpoint: nothing usable
    assert select_restore_pos(plan, 200) == (0, None)
    # d == donor_len: the snapshot itself is the checkpoint
    plan_full = dict(plan, d=800)
    pos, states = select_restore_pos(plan_full, 10_000)
    assert pos == 800
    assert _arrays_equal(states[1][0], snap[1][0])


def test_build_partial_restore_states_trims_attention():
    snap = _hybrid_snapshot(800)
    ck = {1: _rec_states(512.0)}
    states = build_partial_restore_states(snap, ck, 512)
    assert states is not None
    k512, v512 = states[0]
    assert k512.shape[2] == 512 and v512.shape[2] == 512
    assert _arrays_equal(k512, snap[0][0][..., :512, :])
    assert states[1] is ck[1]
    # hybrid restore without recurrent checkpoint state must refuse
    assert build_partial_restore_states(snap, {}, 512) is None
    assert build_partial_restore_states(snap, None, 512) is None


def test_kvcache_trim_restore_real_cache():
    """Slice-assign into a real mlx_lm KVCache and check offset semantics."""
    from mlx_lm.models.cache import KVCache

    donor = KVCache()
    donor.update_and_fetch(*_kv_layer(700))
    full_k, _ = donor.state

    target = KVCache()
    target.state = (full_k[..., :512, :], donor.state[1][..., :512, :])
    assert target.offset == 512
    # forward growth from the restored position works
    target.update_and_fetch(*_kv_layer(8))
    assert target.offset == 520
    k_now, _ = target.state
    assert _arrays_equal(k_now[..., :512, :], full_k[..., :512, :])


def test_capture_recurrent_states_shallow_copy():
    from mlx_lm.models.cache import ArraysCache, KVCache

    kv = KVCache()
    kv.update_and_fetch(*_kv_layer(32))
    rec = ArraysCache(size=2)
    rec[0] = mx.full((1, 4), 1.0)
    rec[1] = mx.full((1, 4), 2.0)
    cache = [kv, rec]
    captured = capture_recurrent_states(cache)
    assert list(captured.keys()) == [1]
    # rebinding the live cache must not disturb the captured copy
    rec[0] = mx.full((1, 4), 99.0)
    assert float(captured[1][0][0, 0]) == 1.0


# ---------------------------------------------------------------------------
# manager-level planning
# ---------------------------------------------------------------------------


def _manager(min_partial=4):
    m = SystemKVManager()
    m.partial_min = min_partial
    return m


def test_plan_partial_restore_gates():
    m = _manager()
    assert m.plan_partial_restore([1, 2, 3]) is None  # no snapshot
    m.snapshot = _hybrid_snapshot(64)
    m.system_hash = "h"
    m.token_count = 8
    m.token_ids = [1, 2, 3, 4, 5, 6, 7, 8]
    m.checkpoints = [{"pos": 4, "states": {1: _rec_states(4.0)}}]
    # diverges after 6 shared tokens
    plan = m.plan_partial_restore([1, 2, 3, 4, 5, 6, 99, 100, 101])
    assert plan is not None and plan["d"] == 6 and plan["has_recurrent"]
    pos, states = select_restore_pos(plan, 9)
    assert pos == 4 and states is m.checkpoints[0]["states"]
    # below partial_min: no plan
    m.partial_min = 7
    assert m.plan_partial_restore([1, 2, 3, 4, 5, 6, 99]) is None


def test_carry_checkpoints():
    m = _manager()
    cps = [{"pos": 2, "states": {}}, {"pos": 5, "states": {}},
           {"pos": 9, "states": {}}]
    carried = m.carry_checkpoints(cps, 5)
    assert [c["pos"] for c in carried] == [2, 5]
    assert carried is not cps
    assert m.carry_checkpoints(None, 5) == []


def test_store_extended_keeps_checkpoints_per_slot():
    m = _manager()
    snap_a = _hybrid_snapshot(32)
    cps_a = [{"pos": 16, "states": {1: _rec_states(16.0)}}]
    m.store_extended("hash-a", snap_a, list(range(32)), promoted=False,
                     checkpoints=cps_a)
    assert len(m.checkpoints) == 1
    # different system hash demotes A (with its checkpoints) into the bag
    m.store_extended("hash-b", _hybrid_snapshot(16), list(range(100, 116)),
                     promoted=False)
    assert m.checkpoints == []
    assert len(m.lru["hash-a"]["checkpoints"]) == 1
    # promoting A back restores its checkpoints
    assert m.lru_promote("hash-a")
    assert m.checkpoints[0]["pos"] == 16


# ---------------------------------------------------------------------------
# SSD format v2
# ---------------------------------------------------------------------------


def test_flatten_checkpoints_inverse():
    cps = [
        {"pos": 256, "states": {1: _rec_states(1.0)}},
        {"pos": 512, "states": {1: _rec_states(2.0), 3: _rec_states(3.0)}},
    ]
    tensors, meta = flatten_checkpoints(cps)
    rebuilt = unflatten_checkpoints(tensors, meta)
    assert [c["pos"] for c in rebuilt] == [256, 512]
    assert _arrays_equal(rebuilt[0]["states"][1][0], cps[0]["states"][1][0])
    assert _arrays_equal(rebuilt[1]["states"][3][1], cps[1]["states"][3][1])
    assert flatten_checkpoints(None) == ({}, [])


def test_ssd_roundtrip_with_checkpoints():
    d = tempfile.mkdtemp(prefix="skv-partial-")
    try:
        store = SystemKVSSDStore(SystemKVSSDConfig(cache_dir=d))
        snap = _hybrid_snapshot(96)
        cps = [{"pos": 48, "states": {1: _rec_states(48.0)}}]
        tokens = tuple(range(96))
        tensors, meta = flatten_snapshot(snap)
        ck_tensors, ck_meta = flatten_checkpoints(cps)
        tensors.update(ck_tensors)
        store._write_entry(tokens, tensors, meta, ck_meta,
                           _snapshot_nbytes(snap))

        hit = store.lookup_prefix(tuple(range(150)))
        assert hit is not None
        entry = store.read_entry(tokens, hit["file_path"])
        assert entry is not None
        assert len(entry["checkpoints"]) == 1
        assert entry["checkpoints"][0]["pos"] == 48
        assert _arrays_equal(
            entry["checkpoints"][0]["states"][1][0], cps[0]["states"][1][0]
        )
        # meta-only read agrees
        rm = store.read_meta(hit["file_path"])
        assert rm["checkpoints"][0]["pos"] == 48
        store.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_ssd_lookup_shared():
    d = tempfile.mkdtemp(prefix="skv-shared-")
    try:
        store = SystemKVSSDStore(SystemKVSSDConfig(cache_dir=d))
        snap = _hybrid_snapshot(32)
        tensors, meta = flatten_snapshot(snap)
        # entry A: tokens 0..79
        a_tokens = tuple(range(80))
        store._write_entry(a_tokens, dict(tensors), meta, [],
                           _snapshot_nbytes(snap))
        # query shares the first 50 tokens with A, then diverges and is
        # longer — A is NOT a full prefix of it.
        query = tuple(range(50)) + tuple(range(900, 960))
        assert store.lookup_prefix(query) is None
        cands = store.lookup_shared(query)
        assert len(cands) == 1
        assert cands[0]["common_len"] == 50
        # an entry that IS a full prefix of the query is excluded here...
        b_tokens = tuple(range(40))
        store._write_entry(b_tokens, dict(tensors), meta, [],
                           _snapshot_nbytes(snap))
        assert store.lookup_prefix(query)["num_tokens"] == 40
        shared = store.lookup_shared(query)
        assert all(c["num_tokens"] != 40 for c in shared)
        # ...and queries shorter than the prefix filter return nothing
        assert store.lookup_shared(tuple(range(8))) == []
        store.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    test_common_prefix_len()
    print("OK  common_prefix_len")
    test_append_checkpoint_thinning()
    print("OK  checkpoint thinning")
    test_select_restore_pos_pure_attention()
    print("OK  select_restore_pos (pure attention)")
    test_select_restore_pos_hybrid()
    print("OK  select_restore_pos (hybrid + donor_len fast path)")
    test_build_partial_restore_states_trims_attention()
    print("OK  build_partial_restore_states trim")
    test_kvcache_trim_restore_real_cache()
    print("OK  real KVCache trim-restore offset semantics")
    test_capture_recurrent_states_shallow_copy()
    print("OK  capture_recurrent_states aliasing")
    test_plan_partial_restore_gates()
    print("OK  plan_partial_restore gates")
    test_carry_checkpoints()
    print("OK  carry_checkpoints")
    test_store_extended_keeps_checkpoints_per_slot()
    print("OK  per-slot checkpoint LRU")
    test_flatten_checkpoints_inverse()
    print("OK  flatten/unflatten checkpoints inverse")
    test_ssd_roundtrip_with_checkpoints()
    print("OK  SSD v2 roundtrip with checkpoints")
    test_ssd_lookup_shared()
    print("OK  SSD shared-prefix lookup")
    print("\nALL PASS")
