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
    capture_checkpoint_states,
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
        cps = append_checkpoint(cps, pos * 100, {"states": pos}, {}, capacity=4)
    # bounded
    assert len(cps) <= 5  # capacity + the always-kept newest
    # sorted ascending, newest retained
    positions = [c["pos"] for c in cps]
    assert positions == sorted(positions)
    assert positions[-1] == 1200
    # non-monotonic appends are ignored
    before = list(cps)
    cps = append_checkpoint(cps, 50, {}, {}, capacity=4)
    assert cps == before


def test_select_restore_pos_pure_attention():
    plan = {"d": 700, "snapshot": [_kv_layer(800)], "checkpoints": [],
            "kinds": ["trim"], "metas": None, "donor_len": 800}
    # exact divergence point, capped by the new extended length
    assert select_restore_pos(plan, 1000)[0] == 700
    assert select_restore_pos(plan, 300)[0] == 300
    assert select_restore_pos(plan, 0) == (0, None, None)


def test_select_restore_pos_hybrid():
    snap = _hybrid_snapshot(800)
    cps = [
        {"pos": 256, "states": {1: _rec_states(256.0)}, "metas": {1: None}},
        {"pos": 512, "states": {1: _rec_states(512.0)}, "metas": {1: None}},
    ]
    plan = {"d": 700, "snapshot": snap, "checkpoints": cps,
            "kinds": ["trim", "ckpt", "trim"], "metas": None,
            "donor_len": 800}
    pos, states, metas = select_restore_pos(plan, 10_000)
    assert pos == 512 and states is cps[1]["states"]
    # cap below the best checkpoint falls back to the earlier one
    pos, states, metas = select_restore_pos(plan, 400)
    assert pos == 256 and states is cps[0]["states"]
    # cap below every checkpoint: nothing usable
    assert select_restore_pos(plan, 200) == (0, None, None)
    # d == donor_len: the snapshot itself is the checkpoint
    plan_full = dict(plan, d=800)
    pos, states, metas = select_restore_pos(plan_full, 10_000)
    assert pos == 800
    assert _arrays_equal(states[1][0], snap[1][0])


def test_build_partial_restore_states_trims_attention():
    snap = _hybrid_snapshot(800)
    ck = {1: _rec_states(512.0)}
    states, metas = build_partial_restore_states(snap, ck, 512)
    assert states is not None
    k512, v512 = states[0]
    assert k512.shape[2] == 512 and v512.shape[2] == 512
    assert _arrays_equal(k512, snap[0][0][..., :512, :])
    assert states[1] is ck[1]
    # hybrid restore without recurrent checkpoint state must refuse
    assert build_partial_restore_states(snap, {}, 512) == (None, None)
    assert build_partial_restore_states(snap, None, 512) == (None, None)
    # opaque layers refuse partial restore entirely
    assert build_partial_restore_states(
        snap, ck, 512, kinds=["trim", "ckpt", "opaque"]
    ) == (None, None)


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
    captured, captured_metas = capture_checkpoint_states(cache)
    assert list(captured.keys()) == [1]
    assert captured_metas[1] is None
    # rebinding the live cache must not disturb the captured copy.
    # Dual-shape (#81): pre-1632 ArraysCache.state is the bare list; 1632+
    # returns (cache, left_padding, lengths) — the inner list either way.
    inner = captured[1][0] if isinstance(captured[1], tuple) else captured[1]
    rec[0] = mx.full((1, 4), 99.0)
    assert float(inner[0][0, 0]) == 1.0


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
    m.checkpoints = [{"pos": 4, "states": {1: _rec_states(4.0)}, "metas": {1: None}}]
    # diverges after 6 shared tokens
    plan = m.plan_partial_restore([1, 2, 3, 4, 5, 6, 99, 100, 101])
    assert plan is not None and plan["d"] == 6
    pos, states, _pmetas = select_restore_pos(plan, 9)
    assert pos == 4 and states is m.checkpoints[0]["states"]
    # below partial_min: no plan
    m.partial_min = 7
    assert m.plan_partial_restore([1, 2, 3, 4, 5, 6, 99]) is None


def test_carry_checkpoints():
    m = _manager()
    cps = [{"pos": 2, "states": {}, "metas": {}},
           {"pos": 5, "states": {}, "metas": {}},
           {"pos": 9, "states": {}, "metas": {}}]
    carried = m.carry_checkpoints(cps, 5)
    assert [c["pos"] for c in carried] == [2, 5]
    assert carried is not cps
    assert m.carry_checkpoints(None, 5) == []


def test_store_extended_keeps_checkpoints_per_slot():
    m = _manager()
    snap_a = _hybrid_snapshot(32)
    cps_a = [{"pos": 16, "states": {1: _rec_states(16.0)}, "metas": {1: None}}]
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
        {"pos": 256, "states": {1: _rec_states(1.0)},
         "metas": {1: None}},
        {"pos": 512, "states": {1: _rec_states(2.0), 3: _rec_states(3.0)},
         "metas": {1: None, 3: ("0", "4", "10", "10")}},
    ]
    tensors, meta = flatten_checkpoints(cps)
    rebuilt = unflatten_checkpoints(tensors, meta)
    assert [c["pos"] for c in rebuilt] == [256, 512]
    assert _arrays_equal(rebuilt[0]["states"][1][0], cps[0]["states"][1][0])
    assert _arrays_equal(rebuilt[1]["states"][3][1], cps[1]["states"][3][1])
    assert rebuilt[1]["metas"][3] == ("0", "4", "10", "10")
    assert rebuilt[0]["metas"][1] is None
    assert flatten_checkpoints(None) == ({}, [])


def test_ssd_roundtrip_with_checkpoints():
    d = tempfile.mkdtemp(prefix="skv-partial-")
    try:
        store = SystemKVSSDStore(SystemKVSSDConfig(cache_dir=d))
        snap = _hybrid_snapshot(96)
        cps = [{"pos": 48, "states": {1: _rec_states(48.0)}, "metas": {1: None}}]
        tokens = tuple(range(96))
        tensors, meta = flatten_snapshot(snap)
        ck_tensors, ck_meta = flatten_checkpoints(cps)
        tensors.update(ck_tensors)
        store._write_entry(tokens, tensors, meta, ck_meta,
                           [None, ["0", "4", "96", "96"], None],
                           ["trim", "ckpt", "trim"],
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
        assert entry["meta"][1] == ("0", "4", "96", "96")
        assert entry["kinds"] == ["trim", "ckpt", "trim"]
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
        store._write_entry(a_tokens, dict(tensors), meta, [], None, None,
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
        store._write_entry(b_tokens, dict(tensors), meta, [], None, None,
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


# ---------------------------------------------------------------------------
# template-family marker detection (fixtures = REAL rendered templates,
# captured via apply_chat_template on the deployed models 2026-06-10)
# ---------------------------------------------------------------------------


def test_detect_template_markers():
    from vllm_mlx.system_kv import detect_template_markers

    fixtures = {
        "chatml": ("<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\nU1"
                   "<|im_end|>\n<|im_start|>assistant\nA1<|im_end|>\n"
                   "<|im_start|>user\nU2<|im_end|>\n<|im_start|>assistant\n"),
        "gemma4": ("<bos><|turn>system\nSYS<turn|>\n<|turn>user\nU1<turn|>\n"
                   "<|turn>model\nA1<turn|>\n<|turn>user\nU2<turn|>\n"
                   "<|turn>model\n<|channel>thought\n<channel|>"),
        "glm4": ("[gMASK]<sop><|system|>SYS<|user|>U1<|assistant|></think>A1"
                 "<|user|>U2<|assistant|><think>"),
        "mistral": ("<s>[SYSTEM_PROMPT]SYS[/SYSTEM_PROMPT][INST]U1[/INST]A1"
                    "</s>[INST]U2[/INST]"),
        "llama3": ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
                   "\n\nSYS<|eot_id|><|start_header_id|>user<|end_header_id|>"
                   "\n\nU1<|eot_id|><|start_header_id|>assistant"
                   "<|end_header_id|>\n\n"),
        "harmony": ("<|start|>system<|message|>meta<|end|><|start|>developer"
                    "<|message|>SYS<|end|><|start|>user<|message|>U1<|end|>"
                    "<|start|>assistant"),
        "phi4": ("<|im_start|>system<|im_sep|>SYS<|im_end|><|im_start|>user"
                 "<|im_sep|>U1<|im_end|><|im_start|>assistant<|im_sep|>"),
    }
    for family, prompt in fixtures.items():
        det_family, boundary, gen_marker = detect_template_markers(prompt)
        assert det_family == family, f"{family}: detected {det_family}"
        assert 0 < boundary < len(prompt)
        # the system content must be fully inside the detected prefix
        assert "SYS" in prompt[:boundary], f"{family}: boundary cuts system"
        assert "U1" not in prompt[:boundary], f"{family}: boundary leaks user"
        # the gen marker must anchor the FINAL generation prompt
        gi = prompt.rfind(gen_marker)
        assert gi > boundary, f"{family}: gen marker before boundary"
        assert "U2" not in prompt[gi:] or family in ("mistral",), (
            f"{family}: gen boundary excludes last user turn"
        )

    # no family: plain text falls back to uncached (-1)
    assert detect_template_markers("just some text") == (None, -1, None)


# ---------------------------------------------------------------------------
# RotatingKVCache (sliding-window) meta_state round-trip — the mechanism
# that makes Gemma 4-class models cacheable (meta-state patch)
# ---------------------------------------------------------------------------


def test_classify_layers_and_meta_capture():
    from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache
    from vllm_mlx.system_kv import (
        capture_checkpoint_states,
        capture_snapshot_meta,
        classify_layers,
    )

    kv = KVCache()
    kv.update_and_fetch(*_kv_layer(16))
    rot = RotatingKVCache(max_size=8)
    rot.update_and_fetch(*_kv_layer(16))
    rec = ArraysCache(size=1)
    rec[0] = mx.full((1, 4), 1.0)
    cache = [kv, rot, rec]

    assert classify_layers(cache) == ["trim", "ckpt", "ckpt"]
    metas = capture_snapshot_meta(cache)
    assert metas[0] is None  # plain KVCache: trivial meta
    assert metas[1] is not None and len(metas[1]) == 4  # ring indices
    states, st_metas = capture_checkpoint_states(cache)
    assert sorted(states.keys()) == [1, 2]
    assert st_metas[1] is not None and st_metas[2] is None


def test_rotating_meta_roundtrip_exact_continuation():
    """Snapshot+meta restore of a RotatingKVCache must continue generation
    identically to the uninterrupted cache — including PAST the rotation
    point, where state-only restore desynchronizes the ring (the original
    patch #12 failure mode)."""
    from mlx_lm.models.cache import RotatingKVCache
    from vllm_mlx.system_kv import apply_snapshot_states, capture_snapshot_meta

    window = 8
    donor = RotatingKVCache(max_size=window)
    # Prefill well past the window so the ring has rotated.
    steps = [_kv_layer(1, dim=4, heads=1) for _ in range(20)]
    for k, v in steps[:15]:
        donor.update_and_fetch(k, v)

    # Snapshot exactly the way the engine does: state + meta_state.
    snap_state = donor.state
    snap_meta = capture_snapshot_meta([donor])[0]
    assert snap_meta is not None

    # Continue the donor (ground truth).
    truth = [donor.update_and_fetch(k, v) for k, v in steps[15:]]

    # Restore into a fresh cache and replay the same continuation.
    restored = RotatingKVCache(max_size=window)
    apply_snapshot_states([restored], [snap_state], [snap_meta])
    assert restored.offset == 15
    replay = [restored.update_and_fetch(k, v) for k, v in steps[15:]]

    for (tk, tv), (rk, rv) in zip(truth, replay):
        assert _arrays_equal(tk, rk) and _arrays_equal(tv, rv)


def test_rotating_state_only_restore_would_desync():
    """Negative control: without meta, the restored ring loses its indices —
    documents WHY meta_state is load-bearing (state-only restore yields a
    different offset/_idx than the donor)."""
    from mlx_lm.models.cache import RotatingKVCache

    window = 8
    donor = RotatingKVCache(max_size=window)
    for _ in range(15):
        donor.update_and_fetch(*_kv_layer(1, dim=4, heads=1))
    bare = RotatingKVCache(max_size=window)
    bare.state = donor.state
    assert (bare.offset, bare._idx) != (donor.offset, donor._idx)


# ---------------------------------------------------------------------------
# RAM byte budget (patch: system-kv-ram-budget) — cap the resident slot set,
# evict the LRU bag (never the in-use active slot), SSD-spilled slots first.
# ---------------------------------------------------------------------------


def test_ram_budget_zero_is_noop():
    """Default (unset) budget == 0 == unbounded: storing distinct slots up to
    capacity keeps them all — no byte-driven eviction, prior behavior."""
    m = SystemKVManager()
    assert m.ram_budget_bytes == 0
    for i in range(4):  # == default capacity
        m.store_extended(f"h{i}", _hybrid_snapshot(64),
                         list(range(i * 100, i * 100 + 32)), promoted=False)
    assert (1 if m.snapshot is not None else 0) + len(m.lru) == 4
    assert m.evictions == 0


def test_ram_budget_evicts_bag_keeps_active():
    """A budget below the full slot set evicts oldest bag entries while the
    in-use active slot is always retained."""
    from vllm_mlx.system_kv import entry_bytes

    m = SystemKVManager()
    per = entry_bytes(_hybrid_snapshot(64))  # identical-shape slots
    m.ram_budget_bytes = int(per * 2.4)  # fits active + 1 bag, not 3
    for i in range(4):
        m.store_extended(f"h{i}", _hybrid_snapshot(64),
                         list(range(i * 100, i * 100 + 32)), promoted=False)
    assert m.system_hash == "h3"          # most-recent is active
    assert m.snapshot is not None         # active never evicted
    resident = entry_bytes(m.snapshot) + sum(e["bytes"] for e in m.lru.values())
    assert resident <= m.ram_budget_bytes
    assert len(m.lru) <= 1
    assert m.evictions >= 2


def test_ram_budget_evicts_spilled_first():
    """Among bag entries, SSD-spilled slots (cheap promote to recover) are
    evicted before unspilled ones (which need a full cold prefill)."""
    from collections import OrderedDict

    m = SystemKVManager()
    m.snapshot = None          # isolate: cap the bag alone (active_bytes == 0)
    m.ram_budget_bytes = 120   # fits exactly one 100-byte bag entry
    m.lru = OrderedDict()      # insertion order == LRU oldest -> newest
    m.lru["sp1"] = {"snapshot": None, "bytes": 100, "spilled": True}
    m.lru["un"] = {"snapshot": None, "bytes": 100, "spilled": False}
    m.lru["sp2"] = {"snapshot": None, "bytes": 100, "spilled": True}
    m.enforce_ram_budget()
    assert "un" in m.lru                       # unspilled survivor kept
    assert "sp1" not in m.lru and "sp2" not in m.lru
    assert m.evictions == 2


def test_ram_budget_keeps_oversized_active():
    """If the active slot alone exceeds the budget and the bag is empty,
    enforce keeps it (the in-flight request needs it) without crashing."""
    m = SystemKVManager()
    m.store_extended("solo", _hybrid_snapshot(64), list(range(32)),
                     promoted=False)
    m.ram_budget_bytes = 1  # smaller than any real slot
    m.enforce_ram_budget()
    assert m.snapshot is not None
    assert len(m.lru) == 0


# ---------------------------------------------------------------------------
# Idle-but-enabled stats (c9 observability): /v1/status cache must not be null
# before the first request when the system-KV path is live.
# ---------------------------------------------------------------------------


def test_stats_idle_emits_enabled_block():
    """A fresh manager (no snapshot/hits/misses) reports an enabled, zeroed
    block — not None — so the cache is provable as live pre-first-request."""
    m = SystemKVManager()
    s = m.stats()
    assert s is not None
    assert s["enabled"] is True
    assert s["hits"] == 0 and s["misses"] == 0 and s["tokens_saved"] == 0
    assert s["capacity"] == m.capacity
    # Zero activity → must not mask an active cache in the selection logic.
    assert not any(s.get(k) for k in ("hits", "misses", "tokens_saved"))


def test_stats_idle_returns_none_when_kill_switch_set():
    """With the kill switch set (is_safe() False) the path is genuinely off —
    stats() returns None so no system_kv_cache block is emitted."""
    m = SystemKVManager()
    m._safe_cached = False  # simulate VLLM_MLX_DISABLE_SYSTEM_KV=1
    assert m.stats() is None


def test_stats_active_carries_enabled_flag():
    """Once a snapshot is stored, the block still carries enabled=True."""
    m = SystemKVManager()
    m.store_extended("h", _hybrid_snapshot(32), list(range(16)), promoted=False)
    s = m.stats()
    assert s is not None and s["enabled"] is True
    assert s["entry_count"] == 1


# ── mlx-lm CEILING tripwire (PATCHES.md #78, re-aimed by #81) ─────────────
# classify_layers identifies recurrent layers by state *shape*. Since #81 the
# 1632 3-tuple (cache, left_padding, lengths) is a SUPPORTED shape; the guard
# now fires on whatever comes NEXT — any ArraysCache state that is neither the
# bare list nor the 1632 tuple classifies 'opaque' and must announce itself.


class _FakeArraysCache:
    """Stands in for mlx-lm's ArraysCache with the CURRENT (list) state."""

    def __init__(self, state):
        self.state = state


_FakeArraysCache.__name__ = "ArraysCache"


class _FakeMambaCache(_FakeArraysCache):
    """A subclass, as mlx-lm ships per-architecture. Name-only matching on the
    exact class misses this — the guard walks the MRO for that reason."""


_FakeMambaCache.__name__ = "MambaCache"


def _reset_ceiling_latch():
    import vllm_mlx.system_kv as skv

    skv._recurrent_shape_warned = False
    skv._kv_shape_warned = False


def test_ceiling_guard_silent_on_current_list_state():
    from vllm_mlx.system_kv import classify_layers

    _reset_ceiling_latch()
    import vllm_mlx.system_kv as skv

    assert classify_layers([_FakeArraysCache([mx.zeros((1, 2))])]) == ["ckpt"]
    assert skv._recurrent_shape_warned is False


def test_ceiling_guard_fires_on_tuple_state(caplog):
    from vllm_mlx.system_kv import classify_layers

    _reset_ceiling_latch()
    # NOT the 1632 shape: its metadata slots are ints, not arrays — the
    # guard must treat this as an UNKNOWN (post-1632) shape and fire.
    future = _FakeArraysCache((mx.zeros((1, 2)), 0, 4))
    with caplog.at_level("ERROR"):
        kinds = classify_layers([future])
    # The classification really does degrade — that is what makes it dangerous.
    assert kinds == ["opaque"]
    assert "SYSTEM-KV CEILING CROSSED" in caplog.text


def test_ceiling_guard_matches_subclasses_over_the_mro(caplog):
    from vllm_mlx.system_kv import classify_layers

    _reset_ceiling_latch()
    future = _FakeMambaCache((mx.zeros((1, 2)), 0, 4))
    with caplog.at_level("ERROR"):
        classify_layers([future])
    assert "SYSTEM-KV CEILING CROSSED" in caplog.text


def test_ceiling_guard_is_one_shot(caplog):
    from vllm_mlx.system_kv import classify_layers

    _reset_ceiling_latch()
    layers = [_FakeArraysCache((mx.zeros((1, 2)), 0, 4)) for _ in range(4)]
    with caplog.at_level("ERROR"):
        classify_layers(layers)
        classify_layers(layers)
    assert caplog.text.count("SYSTEM-KV CEILING CROSSED") == 1


def test_ceiling_guard_ignores_plain_kv_and_rotating():
    """A 2-tuple KVCache and a RotatingKVCache are normal, not the ceiling."""
    from vllm_mlx.system_kv import classify_layers

    _reset_ceiling_latch()
    import vllm_mlx.system_kv as skv

    class _KV:
        state = (mx.zeros((1, 2, 3, 4)), mx.zeros((1, 2, 3, 4)))

    class _Rot:
        state = (mx.zeros((1, 2, 3, 4)), mx.zeros((1, 2, 3, 4)))

    _Rot.__name__ = "RotatingKVCache"
    assert classify_layers([_KV(), _Rot()]) == ["trim", "ckpt"]
    assert skv._recurrent_shape_warned is False


# ── mlx-lm 1632 recurrent state shape (PATCHES.md #81) ────────────────────
# Post-11a6ce7 ArraysCache.state is (cache_list, left_padding, lengths).
# The fork accepts BOTH shapes so the pin can move without a flag day.


def _rec_states_1632(tag: float, empty_meta: bool = False):
    cache = _rec_states(tag)
    if empty_meta:
        lp = mx.array([], dtype=mx.float32)
        ln = mx.array([], dtype=mx.float32)
    else:
        lp = mx.array([0], dtype=mx.int32)
        ln = mx.array([int(tag) % 97 + 1], dtype=mx.int32)
    mx.eval(lp, ln)
    return (cache, lp, ln)


def test_classify_1632_state_is_ckpt_and_silent(caplog):
    from vllm_mlx.system_kv import classify_layers

    _reset_ceiling_latch()
    import vllm_mlx.system_kv as skv

    with caplog.at_level("ERROR"):
        kinds = classify_layers([_FakeArraysCache(_rec_states_1632(1.0))])
    assert kinds == ["ckpt"]
    assert skv._recurrent_shape_warned is False
    assert "CEILING" not in caplog.text


def test_classify_1632_empty_metadata_is_ckpt():
    """Rows without padding metadata encode it as empty arrays (mlx-lm's
    None-serialization convention) — still the supported shape."""
    from vllm_mlx.system_kv import classify_layers

    _reset_ceiling_latch()
    kinds = classify_layers(
        [_FakeArraysCache(_rec_states_1632(2.0, empty_meta=True))]
    )
    assert kinds == ["ckpt"]


def test_pin_state_1632_does_not_alias_the_live_cache_list():
    """pin_state copies the element-0 CONTAINER: a later in-place slot
    rebind on the live cache (self.cache[i] = v) must not reach the pin."""
    from vllm_mlx.system_kv import pin_state

    live = _rec_states_1632(3.0)
    pinned = pin_state(live)
    original = live[0][0]
    live[0][0] = mx.zeros_like(live[0][0])  # the live cache moves on
    assert pinned[0][0] is original
    assert pinned[1] is live[1] and pinned[2] is live[2]  # arrays shared


def test_state_arrays_walks_every_shape():
    from vllm_mlx.system_kv import state_arrays

    st3 = _rec_states_1632(4.0)
    arrs = state_arrays(st3)
    assert len(arrs) == len(st3[0]) + 2  # cache items + lp + ln
    kv = _kv_layer(8)
    assert state_arrays(kv) == [kv[0], kv[1]]
    assert len(state_arrays(_rec_states(1.0))) == 2


def test_entry_and_ckpt_bytes_count_1632_metadata():
    from vllm_mlx.system_kv import ckpt_bytes, entry_bytes

    st3 = _rec_states_1632(5.0)
    expected = sum(a.nbytes for a in st3[0]) + st3[1].nbytes + st3[2].nbytes
    assert entry_bytes([st3]) == expected
    cps = [{"pos": 8, "states": {1: st3}, "metas": {1: None}}]
    assert ckpt_bytes(cps) == expected


def test_partial_restore_fallback_treats_1632_as_ckpt():
    """Legacy plans without kinds classify by shape — the 1632 tuple must
    land on the ckpt branch, not be unpacked as (k, v)."""
    snap = [_kv_layer(800), _rec_states_1632(6.0), _kv_layer(800)]
    ck = {1: _rec_states_1632(512.0)}
    states, metas = build_partial_restore_states(snap, ck, 512)
    assert states is not None
    assert states[1] is ck[1]
    k512, _ = states[0]
    assert k512.shape[2] == 512


def test_select_restore_pos_full_donor_with_1632_snapshot():
    from vllm_mlx.system_kv import select_restore_pos

    snap = [_kv_layer(64), _rec_states_1632(7.0)]
    plan = {"d": 64, "snapshot": snap, "checkpoints": [],
            "kinds": None, "metas": None, "donor_len": 64}
    pos, states, metas = select_restore_pos(plan, 10_000)
    assert pos == 64
    assert states == {1: snap[1]}


def test_apply_snapshot_states_pins_1632_shape():
    from vllm_mlx.system_kv import apply_snapshot_states

    class _Settable:
        state = None

    target = [_Settable()]
    st3 = _rec_states_1632(8.0)
    apply_snapshot_states(target, [st3])
    assert target[0].state[0] is not st3[0]  # container copied
    assert target[0].state[0][0] is st3[0][0]  # arrays shared


def test_flatten_unflatten_snapshot_roundtrips_1632(tmp_path):
    """arrays3 layers survive flatten -> safetensors -> unflatten, empty
    metadata arrays included (the mlx-lm None encoding)."""
    from vllm_mlx.system_kv_ssd import unflatten_snapshot

    snap = [
        _kv_layer(16),
        _rec_states_1632(9.0),
        _rec_states_1632(10.0, empty_meta=True),
    ]
    tensors, layer_meta = flatten_snapshot(snap)
    assert [lm["kind"] for lm in layer_meta] == ["kv", "arrays3", "arrays3"]
    path = str(tmp_path / "snap.safetensors")
    mx.save_safetensors(path, tensors)
    loaded = mx.load(path)
    rebuilt = unflatten_snapshot(loaded, layer_meta)
    assert isinstance(rebuilt[1], tuple) and isinstance(rebuilt[1][0], list)
    assert _arrays_equal(rebuilt[1][0][0], snap[1][0][0])
    assert _arrays_equal(rebuilt[1][1], snap[1][1])
    assert rebuilt[2][1].size == 0 and rebuilt[2][2].size == 0


def test_flatten_unflatten_checkpoints_roundtrips_1632():
    cps = [{
        "pos": 256,
        "states": {1: _rec_states_1632(11.0), 2: _rec_states(12.0)},
        "metas": {1: None, 2: None},
    }]
    tensors, ckpt_meta = flatten_checkpoints(cps)
    rebuilt = unflatten_checkpoints(tensors, ckpt_meta)
    st = rebuilt[0]["states"][1]
    assert isinstance(st, tuple) and isinstance(st[0], list)
    assert _arrays_equal(st[0][1], cps[0]["states"][1][0][1])
    assert _arrays_equal(st[1], cps[0]["states"][1][1])
    assert isinstance(rebuilt[0]["states"][2], list)  # old shape untouched


def test_batched_derive_kinds_and_nbytes_understand_1632():
    from vllm_mlx.batched_system_kv import _derive_kinds, _entry_nbytes

    st3 = _rec_states_1632(13.0)
    assert _derive_kinds([_kv_layer(8), st3]) == ["trim", "ckpt"]
    seg = [_kv_layer(8)]
    expected = (
        seg[0][0].nbytes + seg[0][1].nbytes
        + sum(a.nbytes for a in st3[0]) + st3[1].nbytes + st3[2].nbytes
    )
    assert _entry_nbytes([seg, st3]) == expected


# ── next-ceiling tripwire: base KVCache shape (mlx-lm PR #1778) ───────────


def test_kv_guard_fires_on_1778_full_state(caplog):
    from vllm_mlx.system_kv import classify_layers

    _reset_ceiling_latch()

    class _KV:
        state = (mx.zeros((1, 2, 4, 8)), mx.zeros((1, 2, 4, 8)),
                 mx.array([4]))

    _KV.__name__ = "KVCache"
    with caplog.at_level("ERROR"):
        kinds = classify_layers([_KV()])
    assert kinds == ["opaque"]  # the degradation is the danger
    assert "KVCache.state" in caplog.text and "CEILING" in caplog.text


def test_kv_guard_silent_on_normal_2tuple_and_subclasses(caplog):
    from vllm_mlx.system_kv import classify_layers

    _reset_ceiling_latch()

    class _KV:
        state = (mx.zeros((1, 2, 4, 8)), mx.zeros((1, 2, 4, 8)))

    _KV.__name__ = "KVCache"

    class _Quant:
        state = (mx.zeros((1, 2)), mx.zeros((1, 2)), mx.zeros((1, 2)))

    _Quant.__name__ = "QuantizedKVCache"
    with caplog.at_level("ERROR"):
        classify_layers([_KV(), _Quant()])
    assert "CEILING" not in caplog.text


def test_kv_guard_is_one_shot(caplog):
    from vllm_mlx.system_kv import classify_layers

    _reset_ceiling_latch()

    class _KV:
        state = (mx.zeros((1, 2)),)

    _KV.__name__ = "KVCache"
    with caplog.at_level("ERROR"):
        classify_layers([_KV(), _KV()])
        classify_layers([_KV()])
    assert caplog.text.count("CEILING CROSSED") == 1
