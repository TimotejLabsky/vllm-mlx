# SPDX-License-Identifier: Apache-2.0
"""SSD checkpoint dedup (PATCHES.md #119): every partial-restore checkpoint
is written once, as a blob keyed by its token prefix, and shared by every
entry whose chain passes through that prefix."""

from __future__ import annotations

import json
import os

import mlx.core as mx

from vllm_mlx.system_kv_ssd import SystemKVSSDConfig, SystemKVSSDStore

_SYSTEM = tuple(range(1000, 1600))  # shared system prompt
_A = _SYSTEM + tuple(range(5000, 5400))
_B = _SYSTEM + tuple(range(6000, 6400))


def _rec(tag: float):
    st = [
        mx.full((1, 4, 8), tag).astype(mx.float32),
        mx.full((1, 2, 16), tag + 0.5).astype(mx.float32),
    ]
    mx.eval(st)
    return st


def _snapshot(seq: int):
    kv = mx.zeros((1, 2, seq, 8)).astype(mx.bfloat16)
    mx.eval(kv)
    return [(kv, kv), _rec(float(seq)), (kv, kv)]


def _ckpt(pos, tag=None):
    return {
        "pos": pos,
        "states": {1: _rec(float(pos if tag is None else tag))},
        "metas": {1: None},
    }


def _store(d, **cfg):
    return SystemKVSSDStore(SystemKVSSDConfig(cache_dir=str(d), **cfg))


def _spill(d, tokens, ckpts, **cfg):
    store = _store(d, **cfg)
    store.start_writer()
    assert store.enqueue_spill(
        tokens,
        _snapshot(len(tokens)),
        checkpoints=ckpts,
        kinds=["trim", "ckpt", "trim"],
    )
    store.close()  # drains the writer


def _blobs(d):
    return sorted(
        n for n in os.listdir(os.path.join(d, "ckpt")) if n.endswith(".safetensors")
    )


def _load(store, tokens):
    hit = store.lookup_prefix(tokens)
    return store.read_entry(tokens, hit["file_path"])


def test_shared_checkpoint_is_written_once(tmp_path):
    _spill(tmp_path, _A, [_ckpt(512), _ckpt(800, tag=1.0)])
    _spill(tmp_path, _B, [_ckpt(512), _ckpt(800, tag=2.0)])

    # 512 is inside the shared system prompt -> one blob; 800 is past the
    # divergence (600) -> one per chain.
    assert len(_blobs(tmp_path)) == 3
    store = _store(tmp_path)
    for tokens, tag in ((_A, 1.0), (_B, 2.0)):
        entry = _load(store, tokens)
        assert [cp["pos"] for cp in entry["checkpoints"]] == [512, 800]
        assert float(entry["checkpoints"][0]["states"][1][0][0, 0, 0]) == 512.0
        assert float(entry["checkpoints"][1]["states"][1][0][0, 0, 0]) == tag
    stats = store.get_stats()
    assert stats["ckpt_blob_count"] == 3
    snapshots = sum(e["memory_bytes"] for e in store._index.all_entries())
    assert stats["total_bytes"] == snapshots + stats["ckpt_blob_bytes"]
    store.close()


def test_dedup_hit_is_counted(tmp_path):
    store = _store(tmp_path)
    store.start_writer()
    store.enqueue_spill(_A, _snapshot(len(_A)), checkpoints=[_ckpt(512)])
    store.enqueue_spill(_B, _snapshot(len(_B)), checkpoints=[_ckpt(512)])
    store.close()
    assert store.get_stats()["ckpt_dedup_hits"] == 1
    assert store.get_stats()["ckpt_dedup_bytes"] > 0


def test_a_blob_on_disk_is_not_held_in_the_spill_queue(tmp_path):
    _spill(tmp_path, _A, [_ckpt(512)])
    store = _store(tmp_path)
    _meta, blobs = store._flatten_checkpoint_blobs(_B, [_ckpt(512), _ckpt(800)])
    held = [k for k, t in blobs.items() if t is not None]
    assert len(blobs) == 2 and len(held) == 1  # only the new 800 blob
    store.close()


def test_blob_lives_until_its_last_entry_is_deleted(tmp_path):
    _spill(tmp_path, _A, [_ckpt(512)])
    _spill(tmp_path, _B, [_ckpt(512)])
    store = _store(tmp_path)
    for tokens in (_A, _B):
        assert len(_blobs(tmp_path)) == 1
        fp = store.lookup_prefix(tokens)["file_path"]
        store._quarantine(tokens, fp)
    assert _blobs(tmp_path) == []
    assert store.get_stats()["ckpt_blob_bytes"] == 0
    store.close()


def test_capacity_counts_a_shared_blob_once(tmp_path):
    _spill(tmp_path, _A, [_ckpt(512)])
    _spill(tmp_path, _B, [_ckpt(512)])
    store = _store(tmp_path)
    blob = store.get_stats()["ckpt_blob_bytes"]
    snaps = sum(e["memory_bytes"] for e in store._index.all_entries())
    # a cap that fits both entries only if the blob is counted once
    store._config.max_size_gb = (snaps + blob + 1) / (1024**3)
    store._enforce_capacity()
    assert store._index.get_entry_count() == 2
    store._config.max_size_gb = (snaps + blob - 1) / (1024**3)
    store._enforce_capacity()
    assert store._index.get_entry_count() == 1
    assert len(_blobs(tmp_path)) == 1  # still referenced by the survivor
    store.close()


def test_respill_of_the_same_chain_keeps_one_reference(tmp_path):
    _spill(tmp_path, _A, [_ckpt(512)])
    _spill(tmp_path, _A, [_ckpt(512)])
    store = _store(tmp_path)
    assert list(store._blob_refs.values()) == [1]
    store._quarantine(_A, store.lookup_prefix(_A)["file_path"])
    assert _blobs(tmp_path) == []
    store.close()


def test_reconcile_removes_unreferenced_blobs_and_broken_entries(tmp_path):
    _spill(tmp_path, _A, [_ckpt(512)])
    _spill(tmp_path, _B, [_ckpt(512), _ckpt(800)])
    ckpt_dir = os.path.join(tmp_path, "ckpt")
    stray = os.path.join(ckpt_dir, "0" * 40 + ".safetensors")
    mx.save_safetensors(stray, {"x": mx.zeros((4,))})
    # B loses its private 800 blob
    store = _store(tmp_path)
    b_meta = store.read_meta(store.lookup_prefix(_B)["file_path"])
    os.remove(store._blob_path(b_meta["checkpoints"][1]["blob"]))
    store.close()

    store = _store(tmp_path)
    store.start_writer()  # reconcile
    assert not os.path.exists(stray)
    assert store.lookup_prefix(_B) is None  # dropped, not left to fail a promote
    entry = _load(store, _A)
    assert [cp["pos"] for cp in entry["checkpoints"]] == [512]  # shared blob kept
    store.close()


def test_pre_dedup_entries_still_load_next_to_blob_entries(tmp_path):
    _spill(tmp_path, _A, [_ckpt(512)], dedup_checkpoints=False)
    _spill(tmp_path, _B, [_ckpt(512)])
    store = _store(tmp_path)
    for tokens in (_A, _B):
        entry = _load(store, tokens)
        assert [cp["pos"] for cp in entry["checkpoints"]] == [512]
    a_dir = os.path.join(tmp_path, "data", store.lookup_prefix(_A)["file_path"])
    with open(os.path.join(a_dir, "meta.json")) as f:
        assert json.load(f)["format"] == 3
    assert len(_blobs(tmp_path)) == 1  # only B's
    store.close()


def test_different_prefixes_at_the_same_position_do_not_collide(tmp_path):
    other = tuple(range(7000, 7600)) + tuple(range(5000, 5400))
    _spill(tmp_path, _A, [_ckpt(512, tag=1.0)])
    _spill(tmp_path, other, [_ckpt(512, tag=9.0)])
    assert len(_blobs(tmp_path)) == 2
    store = _store(tmp_path)
    entry = _load(store, other)
    assert float(entry["checkpoints"][0]["states"][1][0][0, 0, 0]) == 9.0
    store.close()


def test_env_gate_restores_the_old_layout(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_MLX_SSD_SYSTEM_KV_CKPT_DEDUP", "0")
    _spill(tmp_path, _A, [_ckpt(512)])
    assert _blobs(tmp_path) == []
    entry = _load(_store(tmp_path), _A)
    assert [cp["pos"] for cp in entry["checkpoints"]] == [512]
