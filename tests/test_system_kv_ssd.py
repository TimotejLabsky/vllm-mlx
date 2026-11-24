# SPDX-License-Identifier: Apache-2.0
"""Round-trip + prefix + eviction + corruption tests for system_kv_ssd.

Requires mlx (runs on Apple Silicon / the Mac Studio venv). Runnable either
via pytest or directly: ``python tests/test_system_kv_ssd.py``.
"""

from __future__ import annotations

import os
import shutil
import tempfile

import mlx.core as mx

from vllm_mlx.system_kv_ssd import (
    SystemKVSSDConfig,
    SystemKVSSDStore,
    flatten_snapshot,
    unflatten_snapshot,
)


def _make_hybrid_snapshot(seq: int = 96):
    """A snapshot mixing bf16 KV tuples and f32 recurrent ArraysCache lists,
    mirroring a Qwen3.6-style hybrid (Gated DeltaNet interleaved with attn)."""
    snapshot = []
    for layer in range(6):
        if layer % 3 == 2:
            # recurrent ArraysCache layer: list of f32 state arrays
            conv = mx.random.normal((1, 16, 64)).astype(mx.float32)
            ssm = mx.random.normal((1, 8, 128)).astype(mx.float32)
            snapshot.append([conv, ssm])
        else:
            # attention KVCache layer: (keys, values) bf16 tuple
            keys = mx.random.normal((1, 8, seq, 64)).astype(mx.bfloat16)
            values = mx.random.normal((1, 8, seq, 64)).astype(mx.bfloat16)
            snapshot.append((keys, values))
    mx.eval([a for st in snapshot for a in st])
    return snapshot


def _assert_snapshot_identical(a, b):
    assert len(a) == len(b), f"layer count {len(a)} != {len(b)}"
    for i, (sa, sb) in enumerate(zip(a, b)):
        assert type(sa) is type(sb), f"layer {i} kind {type(sa)} != {type(sb)}"
        pair_a = sa if isinstance(sa, (list, tuple)) else (sa,)
        pair_b = sb if isinstance(sb, (list, tuple)) else (sb,)
        assert len(pair_a) == len(pair_b)
        for arr_a, arr_b in zip(pair_a, pair_b):
            assert arr_a.dtype == arr_b.dtype, (
                f"layer {i} dtype {arr_a.dtype} != {arr_b.dtype}"
            )
            assert arr_a.shape == arr_b.shape, f"layer {i} shape mismatch"
            assert bool((arr_a == arr_b).all()), f"layer {i} values differ"


def test_flatten_unflatten_inverse():
    snap = _make_hybrid_snapshot()
    tensors, meta = flatten_snapshot(snap)
    rebuilt = unflatten_snapshot(tensors, meta)
    _assert_snapshot_identical(snap, rebuilt)


def test_roundtrip_bit_identical():
    d = tempfile.mkdtemp(prefix="skv-ssd-")
    try:
        store = SystemKVSSDStore(SystemKVSSDConfig(cache_dir=d))
        snap = _make_hybrid_snapshot()
        tokens = tuple(range(100))
        # synchronous write (bypass the async writer for determinism)
        tensors, meta = flatten_snapshot(snap)
        from vllm_mlx.system_kv_ssd import _snapshot_nbytes

        store._write_entry(tokens, tensors, meta, [], None, None, _snapshot_nbytes(snap))

        # prefix lookup with a superset of the stored tokens
        hit = store.lookup_prefix(tuple(range(150)))
        assert hit is not None, "prefix lookup missed"
        assert hit["num_tokens"] == 100

        restored = store.read_entry(tokens, hit["file_path"])
        assert restored is not None
        _assert_snapshot_identical(snap, restored["snapshot"])
        assert restored["checkpoints"] == []  # v1-shaped entry

        # a query SHORTER than the stored entry must NOT match (we never trim)
        assert store.lookup_prefix(tuple(range(50))) is None
        store.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_async_writer_roundtrip():
    import time

    d = tempfile.mkdtemp(prefix="skv-ssd-")
    try:
        store = SystemKVSSDStore(SystemKVSSDConfig(cache_dir=d))
        store.start_writer()
        snap = _make_hybrid_snapshot()
        tokens = tuple(range(80))
        assert store.enqueue_spill(tokens, snap)
        # wait for the writer thread to land the entry
        deadline = time.time() + 5.0
        hit = None
        while time.time() < deadline:
            hit = store.lookup_prefix(tuple(range(120)))
            if hit:
                break
            time.sleep(0.05)
        assert hit is not None, "async spill never landed"
        restored = store.read_entry(tokens, hit["file_path"])
        _assert_snapshot_identical(snap, restored["snapshot"])
        store.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_capacity_eviction():
    d = tempfile.mkdtemp(prefix="skv-ssd-")
    try:
        store = SystemKVSSDStore(
            SystemKVSSDConfig(cache_dir=d, max_entries=2, max_size_gb=999.0)
        )
        from vllm_mlx.system_kv_ssd import _snapshot_nbytes

        for base in (0, 1000, 2000):
            snap = _make_hybrid_snapshot(seq=32)
            toks = tuple(range(base, base + 40))
            tensors, meta = flatten_snapshot(snap)
            store._write_entry(toks, tensors, meta, [], None, None, _snapshot_nbytes(snap))
        assert store._index.get_entry_count() <= 2, "capacity not enforced"
        assert store._stats.evictions >= 1
        store.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_corrupt_quarantine():
    d = tempfile.mkdtemp(prefix="skv-ssd-")
    try:
        store = SystemKVSSDStore(SystemKVSSDConfig(cache_dir=d))
        from vllm_mlx.system_kv_ssd import _SNAPSHOT_FILE, _snapshot_nbytes

        snap = _make_hybrid_snapshot(seq=32)
        tokens = tuple(range(40))
        tensors, meta = flatten_snapshot(snap)
        store._write_entry(tokens, tensors, meta, [], None, None, _snapshot_nbytes(snap))
        hit = store.lookup_prefix(tuple(range(60)))
        assert hit is not None
        # corrupt the snapshot file
        snap_path = os.path.join(store._data_dir, hit["file_path"], _SNAPSHOT_FILE)
        with open(snap_path, "wb") as f:
            f.write(b"not a safetensors file")
        assert store.read_entry(tokens, hit["file_path"]) is None
        # entry de-indexed so it isn't retried
        assert store.lookup_prefix(tuple(range(60))) is None
        store.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    test_flatten_unflatten_inverse()
    print("OK  flatten/unflatten inverse")
    test_roundtrip_bit_identical()
    print("OK  roundtrip bit-identical (bf16 KV + f32 recurrent)")
    test_async_writer_roundtrip()
    print("OK  async writer roundtrip")
    test_capacity_eviction()
    print("OK  capacity eviction")
    test_corrupt_quarantine()
    print("OK  corrupt-entry quarantine")
    print("\nALL PASS")


def test_spill_defers_until_idle():
    """Defer-until-idle (2026-06-12): the writer must hold heavy
    serialization while the engine is busy and land it once idle."""
    import time as _time

    d = tempfile.mkdtemp(prefix="skv-idle-")
    try:
        busy = {"flag": True}
        store = SystemKVSSDStore(
            SystemKVSSDConfig(cache_dir=d),
            idle_check=lambda: not busy["flag"],
        )
        store.start_writer()
        snap = _make_hybrid_snapshot(seq=32)
        tokens = tuple(range(48))
        assert store.enqueue_spill(tokens, snap)
        # busy: nothing may land
        _time.sleep(2.5)
        assert store.lookup_prefix(tuple(range(60))) is None, (
            "spill landed while engine busy"
        )
        # idle: spill drains
        busy["flag"] = False
        deadline = _time.time() + 10.0
        hit = None
        while _time.time() < deadline:
            hit = store.lookup_prefix(tuple(range(60)))
            if hit:
                break
            _time.sleep(0.2)
        assert hit is not None, "spill never landed after idle"
        store.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_close_drains_even_when_busy():
    """Shutdown must not deadlock on a never-idle engine: close() drains."""
    d = tempfile.mkdtemp(prefix="skv-drain-")
    try:
        store = SystemKVSSDStore(
            SystemKVSSDConfig(cache_dir=d),
            idle_check=lambda: False,  # never idle
        )
        store.start_writer()
        snap = _make_hybrid_snapshot(seq=32)
        tokens = tuple(range(40))
        assert store.enqueue_spill(tokens, snap)
        store.close()  # must return (poison pill flips writer to drain mode)
        # entry either landed during drain or was dropped — but close returned
        assert True
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# SSD tier hardening (patch: ssd-tier-hardening): real-disk-byte accounting,
# startup reconcile (orphan sweep + byte backfill), queued-spill byte cap.
# ---------------------------------------------------------------------------


def test_index_counts_disk_bytes_including_checkpoints():
    """The index must record the actual on-disk size (snapshot AND the
    flattened checkpoint tensors), not the snapshot-only in-RAM nbytes — else
    the capacity cap under-counts and never fires."""
    from vllm_mlx.system_kv_ssd import (
        _SNAPSHOT_FILE,
        _snapshot_nbytes,
        flatten_checkpoints,
    )

    d = tempfile.mkdtemp(prefix="skv-bytes-")
    try:
        store = SystemKVSSDStore(SystemKVSSDConfig(cache_dir=d))
        snap = _make_hybrid_snapshot(seq=64)
        ck_state = [mx.random.normal((1, 8, 128)).astype(mx.float32)]
        mx.eval(ck_state)
        ckpts = [{"pos": 32, "states": {2: ck_state}, "metas": {2: None}}]
        tokens = tuple(range(64))
        tensors, layer_meta = flatten_snapshot(snap)
        ck_tensors, ck_meta = flatten_checkpoints(ckpts)
        tensors.update(ck_tensors)
        store._write_entry(
            tokens, tensors, layer_meta, ck_meta, None, None,
            _snapshot_nbytes(snap),
        )
        rows = store._index.all_entries()
        assert len(rows) == 1
        snap_file = os.path.join(
            store._data_dir, rows[0]["file_path"], _SNAPSHOT_FILE
        )
        disk = os.path.getsize(snap_file)
        assert rows[0]["memory_bytes"] == disk
        # strictly more than the snapshot-only nbytes (checkpoint counted)
        assert rows[0]["memory_bytes"] > _snapshot_nbytes(snap)
        store.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_queue_byte_cap_drops_backlog():
    """A spill that would push queued bytes over the cap is dropped; an empty
    queue always admits one (even oversized) spill. No writer thread runs, so
    the queue is never drained and the accounting is deterministic."""
    from vllm_mlx.system_kv_ssd import _BYTES_PER_GB, _snapshot_nbytes

    d = tempfile.mkdtemp(prefix="skv-qcap-")
    try:
        snap = _make_hybrid_snapshot(seq=64)
        one = _snapshot_nbytes(snap)
        cap_gb = (one * 1.5) / _BYTES_PER_GB  # fits one, not two
        store = SystemKVSSDStore(
            SystemKVSSDConfig(cache_dir=d, max_queued_gb=cap_gb)
        )
        assert store.enqueue_spill(tuple(range(64)), snap) is True
        assert store.enqueue_spill(
            tuple(range(100, 164)), _make_hybrid_snapshot(seq=64)
        ) is False
        assert store._stats.spill_drops == 1
        store.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_queue_empty_admits_oversized():
    """A single spill larger than the cap is still admitted when the queue is
    empty (so huge deep-context prefixes can persist)."""
    from vllm_mlx.system_kv_ssd import _BYTES_PER_GB

    d = tempfile.mkdtemp(prefix="skv-qbig-")
    try:
        store = SystemKVSSDStore(
            SystemKVSSDConfig(cache_dir=d, max_queued_gb=1 / _BYTES_PER_GB)
        )  # 1-byte cap
        assert store.enqueue_spill(
            tuple(range(64)), _make_hybrid_snapshot(seq=64)
        ) is True
        store.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_reconcile_sweeps_orphans_and_backfills_bytes():
    """Startup reconcile: drop stale *.tmp dirs, delete index rows whose data
    dir vanished, remove orphan data dirs, and backfill wrong memory_bytes."""
    import time as _time

    from vllm_mlx.ssd_cache import _blob_to_tokens
    from vllm_mlx.system_kv_ssd import _META_FILE, _SNAPSHOT_FILE, _snapshot_nbytes

    d = tempfile.mkdtemp(prefix="skv-recon-")
    try:
        store = SystemKVSSDStore(SystemKVSSDConfig(cache_dir=d))
        data_dir = store._data_dir
        good = tuple(range(40))
        gone = tuple(range(100, 140))
        for toks in (good, gone):
            snap = _make_hybrid_snapshot(seq=32)
            tensors, meta = flatten_snapshot(snap)
            store._write_entry(
                toks, tensors, meta, [], None, None, _snapshot_nbytes(snap)
            )
        rows = {
            tuple(_blob_to_tokens(e["tokens_blob"])): e
            for e in store._index.all_entries()
        }
        good_fp = rows[good]["file_path"]
        gone_fp = rows[gone]["file_path"]
        store.close()

        # (a) stale .tmp dir from an interrupted write
        stale = os.path.join(data_dir, "deadbeef.tmp")
        os.makedirs(stale)
        old = _time.time() - 10_000
        os.utime(stale, (old, old))
        # (b) one entry's data dir vanished (SIGKILL mid-cleanup), row remains
        shutil.rmtree(os.path.join(data_dir, gone_fp))
        # (c) orphan data dir with no index row
        orphan = os.path.join(data_dir, "orphanhash")
        os.makedirs(orphan)
        with open(os.path.join(orphan, _META_FILE), "w") as f:
            f.write("{}")

        store2 = SystemKVSSDStore(SystemKVSSDConfig(cache_dir=d))
        # (d) corrupt the good entry's accounting before reconcile
        store2._index.update_memory_bytes(good, 123)
        store2._reconcile()

        assert not os.path.exists(stale)            # (a)
        toks_now = {
            tuple(_blob_to_tokens(e["tokens_blob"]))
            for e in store2._index.all_entries()
        }
        assert gone not in toks_now and good in toks_now  # (b)
        assert not os.path.exists(orphan)           # (c)
        row = next(
            e for e in store2._index.all_entries()
            if tuple(_blob_to_tokens(e["tokens_blob"])) == good
        )
        snap_file = os.path.join(data_dir, good_fp, _SNAPSHOT_FILE)
        assert row["memory_bytes"] == os.path.getsize(snap_file)  # (d)
        assert row["memory_bytes"] != 123
        store2.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_upstream_index_schema_bump_keeps_existing_spills():
    """Upstream #740 bumped ``SSDIndex._SCHEMA_VERSION`` 1 -> 2 for its
    CacheList serializer, and a version mismatch drops every row. Our entries
    are not written by those serializers, so an index written by the deployed
    build (version 1) must survive the upgrade — without the pin, the rows are
    purged and reconcile then deletes every data dir as an orphan."""
    import sqlite3

    from vllm_mlx.ssd_cache import SSDIndex, _blob_to_tokens
    from vllm_mlx.system_kv_ssd import _snapshot_nbytes

    d = tempfile.mkdtemp(prefix="skv-schema-")
    try:
        store = SystemKVSSDStore(SystemKVSSDConfig(cache_dir=d))
        toks = tuple(range(40))
        snap = _make_hybrid_snapshot(seq=32)
        tensors, meta = flatten_snapshot(snap)
        store._write_entry(toks, tensors, meta, [], None, None, _snapshot_nbytes(snap))
        (row,) = store._index.all_entries()
        entry_dir = os.path.join(store._data_dir, row["file_path"])
        store.close()

        # The index as the pre-#740 deployed build left it on disk.
        conn = sqlite3.connect(os.path.join(d, "index.db"))
        conn.execute("UPDATE schema_version SET version = 1")
        conn.commit()
        conn.close()
        assert SSDIndex._SCHEMA_VERSION != 1  # the bump this test guards

        store2 = SystemKVSSDStore(SystemKVSSDConfig(cache_dir=d))
        store2._reconcile()
        assert store2._index.migrated_from is None
        assert [
            tuple(_blob_to_tokens(e["tokens_blob"]))
            for e in store2._index.all_entries()
        ] == [toks]
        assert os.path.isdir(entry_dir)
        store2.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _write_with_ckpts(store, n_tokens: int, ckpt_positions):
    """Synchronous spill of an ``n_tokens`` chain with checkpoints at
    ``ckpt_positions`` (meta only matters to #170; the tensors are tiny)."""
    from vllm_mlx.system_kv_ssd import _snapshot_nbytes

    snap = _make_hybrid_snapshot(seq=8)
    tensors, meta = flatten_snapshot(snap)
    ckpt_meta = [{"pos": p, "layers": []} for p in ckpt_positions]
    store._write_entry(
        tuple(range(n_tokens)),
        tensors,
        meta,
        ckpt_meta,
        None,
        None,
        _snapshot_nbytes(snap),
    )


def test_supersede_deletes_covered_prefix_copies_170():
    """#170: a longer spill of the same chain deletes the shorter copies its
    ladder covers and leaves an unrelated chain alone."""
    from vllm_mlx.system_kv_ssd import _snapshot_nbytes

    d = tempfile.mkdtemp(prefix="skv-ssd-")
    try:
        store = SystemKVSSDStore(SystemKVSSDConfig(cache_dir=d, supersede_slack=512))
        _write_with_ckpts(store, 4000, [3990])
        _write_with_ckpts(store, 8000, [3990, 7990])
        other = tuple(range(1, 3000))  # a different chain (starts at 1)
        snap = _make_hybrid_snapshot(seq=8)
        t, m = flatten_snapshot(snap)
        store._write_entry(other, t, m, [], None, None, _snapshot_nbytes(snap))
        _write_with_ckpts(store, 12000, [7800, 11990])
        rows = store._index.lookup_prefix(tuple(range(20000)))
        assert [r["num_tokens"] for r in rows] == [12000]
        assert store.lookup_prefix(other + (5,))["num_tokens"] == 2999
        assert store.get_stats()["superseded"] == 2
        # The deleted copies' dirs are gone; the survivors' remain.
        assert len(os.listdir(os.path.join(d, "data"))) == 2
        store.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_supersede_keeps_copy_without_nearby_checkpoint_170():
    d = tempfile.mkdtemp(prefix="skv-ssd-")
    try:
        store = SystemKVSSDStore(SystemKVSSDConfig(cache_dir=d, supersede_slack=512))
        _write_with_ckpts(store, 4000, [3990])
        _write_with_ckpts(store, 8000, [1000, 7990])  # nothing in [3488, 4000]
        rows = store._index.lookup_prefix(tuple(range(9000)))
        assert [r["num_tokens"] for r in rows] == [8000, 4000]
        assert store.get_stats()["superseded"] == 0
        store.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_supersede_off_by_default_170(monkeypatch):
    monkeypatch.delenv("VLLM_MLX_SSD_SUPERSEDE_SLACK", raising=False)
    d = tempfile.mkdtemp(prefix="skv-ssd-")
    try:
        store = SystemKVSSDStore(SystemKVSSDConfig(cache_dir=d))
        assert store._config.supersede_slack == 0
        _write_with_ckpts(store, 4000, [3990])
        _write_with_ckpts(store, 8000, [3990, 7990])
        rows = store._index.lookup_prefix(tuple(range(9000)))
        assert [r["num_tokens"] for r in rows] == [8000, 4000]
        store.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_supersede_named_key_and_gen_start_round_trip_170():
    """A spill naming an older copy of its conversation deletes exactly that
    key once written; gen_start survives the round trip."""
    from vllm_mlx.system_kv_ssd import _snapshot_nbytes

    d = tempfile.mkdtemp(prefix="skv-ssd-")
    try:
        store = SystemKVSSDStore(SystemKVSSDConfig(cache_dir=d, supersede_slack=512))
        snap = _make_hybrid_snapshot(seq=8)
        t, m = flatten_snapshot(snap)
        nb = _snapshot_nbytes(snap)
        old = tuple(range(350)) + (9,) * 50
        other = tuple(range(200)) + (4,) * 300
        store._write_entry(old, t, m, [], None, None, nb, gen_start=350)
        store._write_entry(other, t, m, [], None, None, nb)
        new = tuple(range(345)) + (5,) * 200
        store._write_entry(new, t, m, [], None, None, nb, gen_start=545, supersede=old)
        assert store.lookup_prefix(old + (1,)) is None
        assert store.lookup_prefix(other + (1,))["num_tokens"] == 500
        hit = store.lookup_prefix(new + (1,))
        assert store.read_entry(new, hit["file_path"])["gen_start"] == 545
        assert store.get_stats()["superseded"] == 1
        store.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_low_disk_floor_refuses_spill_and_counts_173(monkeypatch):
    """#173: under the free-space floor a spill is refused, counted, and
    never raises; above it (or with the floor off) it lands."""
    import collections

    from vllm_mlx import system_kv_ssd
    from vllm_mlx.system_kv_ssd import _snapshot_nbytes

    DU = collections.namedtuple("DU", "total used free")
    gb = 10**9
    free = {"v": 50 * gb}
    monkeypatch.setattr(
        system_kv_ssd.shutil, "disk_usage", lambda p: DU(1000 * gb, 0, free["v"])
    )
    d = tempfile.mkdtemp(prefix="skv-ssd-")
    try:
        store = SystemKVSSDStore(SystemKVSSDConfig(cache_dir=d, min_free_pct=10))
        snap = _make_hybrid_snapshot(seq=8)
        t, m = flatten_snapshot(snap)
        nb = _snapshot_nbytes(snap)
        store._write_entry(tuple(range(100)), t, m, [], None, None, nb)
        st = store.get_stats()
        assert st["spill_count"] == 0 and st["spill_refused_low_disk"] == 1
        assert os.listdir(os.path.join(d, "data")) == []
        free["v"] = 400 * gb  # 40% free: above the floor
        store._write_entry(tuple(range(100)), t, m, [], None, None, nb)
        assert store.get_stats()["spill_count"] == 1
        # the floor is on the volume AFTER the write: 100.5 GB free, 100 GB floor
        free["v"] = 100 * gb + nb // 2
        store._write_entry(tuple(range(200)), t, m, [], None, None, nb)
        assert store.get_stats()["spill_refused_low_disk"] == 2
        store._config.min_free_pct = 0  # off
        store._write_entry(tuple(range(200)), t, m, [], None, None, nb)
        assert store.get_stats()["spill_count"] == 2
        # an unreadable volume fails open
        store._config.min_free_pct = 10

        def boom(p):
            raise OSError("gone")

        monkeypatch.setattr(system_kv_ssd.shutil, "disk_usage", boom)
        assert store._disk_floor_ok(nb) is True
        store.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_min_free_pct_env_default_173(monkeypatch):
    monkeypatch.delenv("VLLM_MLX_SSD_MIN_FREE_PCT", raising=False)
    assert SystemKVSSDConfig(cache_dir="x").min_free_pct == 10.0
    monkeypatch.setenv("VLLM_MLX_SSD_MIN_FREE_PCT", "15")
    assert SystemKVSSDConfig(cache_dir="x").min_free_pct == 15.0
    monkeypatch.setenv("VLLM_MLX_SSD_MIN_FREE_PCT", "junk")
    assert SystemKVSSDConfig(cache_dir="x").min_free_pct == 10.0


def test_177_entry_cap_comes_from_the_env(monkeypatch):
    """#177: the tier's entry cap was a fixed 64. A daily batch of more
    distinct prompts than that cycled through LRU and never hit."""
    monkeypatch.delenv("VLLM_MLX_SSD_SYSTEM_KV_MAX_ENTRIES", raising=False)
    assert SystemKVSSDConfig(cache_dir="unused").max_entries == 64
    monkeypatch.setenv("VLLM_MLX_SSD_SYSTEM_KV_MAX_ENTRIES", "nonsense")
    assert SystemKVSSDConfig(cache_dir="unused").max_entries == 64
    monkeypatch.setenv("VLLM_MLX_SSD_SYSTEM_KV_MAX_ENTRIES", "256")
    assert SystemKVSSDConfig(cache_dir="unused").max_entries == 256

    monkeypatch.setenv("VLLM_MLX_SSD_SYSTEM_KV_MAX_ENTRIES", "3")
    d = tempfile.mkdtemp(prefix="skv-ssd-")
    try:
        store = SystemKVSSDStore(SystemKVSSDConfig(cache_dir=d, max_size_gb=999.0))
        from vllm_mlx.system_kv_ssd import _snapshot_nbytes

        for base in (0, 1000, 2000, 3000):
            snap = _make_hybrid_snapshot(seq=32)
            toks = tuple(range(base, base + 40))
            tensors, meta = flatten_snapshot(snap)
            store._write_entry(
                toks, tensors, meta, [], None, None, _snapshot_nbytes(snap)
            )
        assert store._index.get_entry_count() == 3
        assert store._index.lookup_prefix(tuple(range(0, 40))) == []  # LRU went
        store.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)
