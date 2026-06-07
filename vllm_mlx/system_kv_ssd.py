# SPDX-License-Identifier: Apache-2.0
"""SSD persistence for the SimpleEngine system-KV snapshot (fork patch #16).

The SimpleEngine system-prefix KV cache (patches #4/#6/#9/#12/#13) lives
entirely in the serving process. Whenever vllm-mlx exits — TTL eviction,
llama-swap model swap, manual restart, OOM — every snapshot is lost and the
next request pays a full cold prefill (~25 s on a 4 K-token dense prompt,
~70 s on a 13 K-token MoE workload). This module persists snapshots to NVMe
so the next process can promote a stored prefix (~100 ms–1.5 s disk read)
instead of recomputing it. ~100×–300× speedup vs cold prefill.

Why a dedicated store instead of reusing ``ssd_cache.SSDCacheTier``:
the existing tier serializes via numpy (``np.array(layer.keys)``), which
**raises** on MLX ``bfloat16`` arrays (``Item size 2 ... does not match
dtype B item size 1``). Our snapshots are unquantized (bf16 KV) and hybrid
(bf16 KV layers interleaved with possibly-f32 recurrent ``ArraysCache``
state), so a numpy/float32 bridge cannot round-trip them losslessly. MLX
safetensors (``mx.save_safetensors`` / ``mx.load``) preserves every array's
dtype exactly with zero bookkeeping, so this store uses it for the snapshot
data while reusing the tested ``SSDIndex`` (SQLite, prefix-searchable) for
metadata.

Snapshot shape (matches ``SimpleEngine._system_kv_snapshot``):
``list[ tuple(keys, values) | list[mx.array] ]`` — a 2-tuple per ``KVCache``
layer (``c.state``), a list per ``ArraysCache`` layer.

Thread-safety: spills run on a background writer thread (non-blocking for
the engine). Promotes (``read_entry``) run synchronously on the caller's
thread — the engine calls them from inside its serialized generation worker,
which is already off the event loop.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import threading
import time
from dataclasses import dataclass
from typing import Any

from .ssd_cache import SSDIndex, _blob_to_tokens, _tokens_hash

logger = logging.getLogger(__name__)

_BYTES_PER_GB = 1024 * 1024 * 1024
_SNAPSHOT_FILE = "snapshot.safetensors"
_META_FILE = "meta.json"


@dataclass
class SystemKVSSDStats:
    """Counters for the system-KV SSD store. Surfaced via get_stats()."""

    spill_count: int = 0
    spill_bytes: int = 0
    spill_drops: int = 0
    promote_count: int = 0
    promote_bytes: int = 0
    promote_latency_sum: float = 0.0
    promote_misses: int = 0
    read_failures: int = 0
    evictions: int = 0

    def to_dict(self) -> dict:
        avg_ms = (
            (self.promote_latency_sum / self.promote_count * 1000.0)
            if self.promote_count
            else 0.0
        )
        return {
            "spill_count": self.spill_count,
            "spill_bytes": self.spill_bytes,
            "spill_drops": self.spill_drops,
            "promote_count": self.promote_count,
            "promote_bytes": self.promote_bytes,
            "avg_promote_latency_ms": round(avg_ms, 2),
            "promote_misses": self.promote_misses,
            "read_failures": self.read_failures,
            "evictions": self.evictions,
        }


@dataclass
class SystemKVSSDConfig:
    cache_dir: str
    max_size_gb: float = 50.0
    max_entries: int = 64
    spill_queue_size: int = 16
    dir_permissions: int = 0o700
    file_permissions: int = 0o600

    @property
    def max_size_bytes(self) -> int:
        return int(self.max_size_gb * _BYTES_PER_GB)


def _snapshot_nbytes(snapshot: list) -> int:
    total = 0
    for st in snapshot:
        if isinstance(st, tuple):
            for a in st:
                total += int(getattr(a, "nbytes", 0))
        else:
            for a in st:
                total += int(getattr(a, "nbytes", 0))
    return total


def flatten_snapshot(snapshot: list) -> tuple[dict, list[dict]]:
    """Flatten a snapshot into an mx-array dict + per-layer metadata.

    KVCache layers (tuple state) -> ``l{i}_k`` / ``l{i}_v``.
    ArraysCache layers (list state) -> ``l{i}_s{j}`` for j in range(n).
    """
    tensors: dict[str, Any] = {}
    layer_meta: list[dict] = []
    for i, st in enumerate(snapshot):
        if isinstance(st, tuple):
            if len(st) != 2:
                raise ValueError(
                    f"unexpected KV state arity {len(st)} at layer {i}"
                )
            tensors[f"l{i}_k"] = st[0]
            tensors[f"l{i}_v"] = st[1]
            layer_meta.append({"i": i, "kind": "kv"})
        elif isinstance(st, list):
            for j, a in enumerate(st):
                tensors[f"l{i}_s{j}"] = a
            layer_meta.append({"i": i, "kind": "arrays", "n": len(st)})
        else:
            raise ValueError(
                f"unexpected snapshot layer type {type(st).__name__} at layer {i}"
            )
    return tensors, layer_meta


def unflatten_snapshot(tensors: dict, layer_meta: list[dict]) -> list:
    """Rebuild a snapshot from an mx-array dict + per-layer metadata.

    Inverse of ``flatten_snapshot``. Returns the same shape the engine
    restores via ``cache[i].state = saved_state`` (tuple for KV, list for
    ArraysCache). Dtypes are whatever ``mx.load`` returned — i.e. the exact
    dtypes captured at spill time.
    """
    snapshot: list = []
    for lm in layer_meta:
        i = lm["i"]
        if lm["kind"] == "kv":
            snapshot.append((tensors[f"l{i}_k"], tensors[f"l{i}_v"]))
        elif lm["kind"] == "arrays":
            snapshot.append([tensors[f"l{i}_s{j}"] for j in range(lm["n"])])
        else:
            raise ValueError(f"unknown layer kind {lm['kind']!r} at layer {i}")
    return snapshot


class SystemKVSSDStore:
    """NVMe-backed persistence for SimpleEngine system-KV snapshots."""

    def __init__(self, config: SystemKVSSDConfig) -> None:
        self._config = config
        self._cache_dir = config.cache_dir
        self._data_dir = os.path.join(self._cache_dir, "data")
        os.makedirs(self._cache_dir, mode=config.dir_permissions, exist_ok=True)
        os.makedirs(self._data_dir, mode=config.dir_permissions, exist_ok=True)
        self._index = SSDIndex(self._cache_dir)
        self._stats = SystemKVSSDStats()
        self._lock = threading.Lock()
        self._spill_queue: queue.Queue = queue.Queue(maxsize=config.spill_queue_size)
        self._writer_stop = threading.Event()
        self._writer_thread: threading.Thread | None = None

    # ---- writer lifecycle -------------------------------------------------

    def start_writer(self) -> None:
        if self._writer_thread is not None:
            return
        self._writer_stop.clear()
        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="system-kv-ssd-writer"
        )
        self._writer_thread.start()
        logger.info("[system_kv_ssd] writer thread started (%s)", self._cache_dir)

    def _writer_loop(self) -> None:
        while not self._writer_stop.is_set():
            try:
                item = self._spill_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:  # poison pill
                break
            tokens, tensors, layer_meta, nbytes = item
            try:
                self._write_entry(tokens, tensors, layer_meta, nbytes)
            except Exception:
                logger.exception(
                    "[system_kv_ssd] failed to write entry (%d tokens)", len(tokens)
                )

    def close(self) -> None:
        """Drain pending spills, stop the writer, close the index. Idempotent.

        The poison pill is enqueued WITHOUT setting the stop flag first, so the
        writer drains everything already queued (write-through spills from the
        session) before exiting on the pill — they aren't lost on shutdown.
        """
        if self._writer_thread is not None:
            try:
                self._spill_queue.put(None, timeout=2.0)
            except queue.Full:
                self._writer_stop.set()
            self._writer_thread.join(timeout=10.0)
            self._writer_stop.set()
            self._writer_thread = None
        try:
            self._index.close()
        except Exception:
            logger.debug("[system_kv_ssd] index close failed", exc_info=True)

    # ---- spill ------------------------------------------------------------

    def enqueue_spill(self, tokens: tuple[int, ...], snapshot: list) -> bool:
        """Queue a snapshot for async write-through. False if dropped.

        Flattening (cheap, no copy — mx arrays are reference-held) happens on
        the caller's thread so the snapshot reference can't drift before the
        writer runs; the actual disk write is on the writer thread.
        """
        try:
            tensors, layer_meta = flatten_snapshot(snapshot)
        except Exception:
            logger.exception("[system_kv_ssd] flatten failed; skipping spill")
            return False
        nbytes = _snapshot_nbytes(snapshot)
        try:
            self._spill_queue.put_nowait((tokens, tensors, layer_meta, nbytes))
            return True
        except queue.Full:
            with self._lock:
                self._stats.spill_drops += 1
            logger.warning(
                "[system_kv_ssd] spill queue full, dropping (%d tokens)", len(tokens)
            )
            return False

    def _write_entry(
        self,
        tokens: tuple[int, ...],
        tensors: dict,
        layer_meta: list[dict],
        nbytes: int,
    ) -> None:
        import mlx.core as mx

        entry_hash = _tokens_hash(tokens)
        entry_dir = os.path.join(self._data_dir, entry_hash)
        tmp_dir = entry_dir + ".tmp"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, mode=self._config.dir_permissions, exist_ok=True)

        snap_path = os.path.join(tmp_dir, _SNAPSHOT_FILE)
        # mx.save_safetensors requires contiguous, evaluated arrays.
        mx.eval(list(tensors.values()))
        mx.save_safetensors(snap_path, tensors)
        os.chmod(snap_path, self._config.file_permissions)

        meta_path = os.path.join(tmp_dir, _META_FILE)
        with open(meta_path, "w") as f:
            json.dump({"layers": layer_meta, "num_tokens": len(tokens)}, f)
        os.chmod(meta_path, self._config.file_permissions)

        disk_bytes = os.path.getsize(snap_path)

        if os.path.exists(entry_dir):
            shutil.rmtree(entry_dir)
        os.rename(tmp_dir, entry_dir)

        self._index.insert_entry(
            tokens_key=tokens,
            file_path=entry_hash,
            memory_bytes=nbytes,
            num_tokens=len(tokens),
        )
        with self._lock:
            self._stats.spill_count += 1
            self._stats.spill_bytes += disk_bytes
        logger.info(
            "[system_kv_ssd] spilled %d-token snapshot (%.1f MB on disk)",
            len(tokens),
            disk_bytes / 1e6,
        )
        self._enforce_capacity()

    def _enforce_capacity(self) -> None:
        cfg = self._config
        try:
            while (
                self._index.get_total_bytes() > cfg.max_size_bytes
                or self._index.get_entry_count() > cfg.max_entries
            ):
                victims = self._index.get_lru(limit=1)
                if not victims:
                    break
                v = victims[0]
                self._delete_entry(v["file_path"])
                self._index.delete_entry(_blob_to_tokens(v["tokens_blob"]))
                with self._lock:
                    self._stats.evictions += 1
        except Exception:
            logger.debug("[system_kv_ssd] capacity enforcement failed", exc_info=True)

    def _delete_entry(self, file_path: str) -> None:
        entry_dir = os.path.join(self._data_dir, file_path)
        if os.path.isdir(entry_dir):
            shutil.rmtree(entry_dir, ignore_errors=True)

    # ---- promote ----------------------------------------------------------

    def lookup_prefix(self, tokens: tuple[int, ...]) -> dict | None:
        """Longest stored entry whose tokens are a prefix of ``tokens``."""
        results = self._index.lookup_prefix(tokens)
        if results:
            return results[0]  # sorted num_tokens DESC
        return None

    def read_entry(self, tokens: tuple[int, ...], file_path: str) -> list | None:
        """Load a snapshot from disk. Returns the snapshot or None on failure.

        Synchronous (safetensors read + dtype-exact mx.load). Quarantines and
        de-indexes a corrupt entry so it isn't retried.
        """
        import mlx.core as mx

        entry_dir = os.path.join(self._data_dir, file_path)
        snap_path = os.path.join(entry_dir, _SNAPSHOT_FILE)
        meta_path = os.path.join(entry_dir, _META_FILE)
        t0 = time.time()
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            tensors = mx.load(snap_path)
            snapshot = unflatten_snapshot(tensors, meta["layers"])
            mx.eval([a for st in snapshot for a in st])
        except Exception as e:
            logger.warning("[system_kv_ssd] corrupt entry %s: %s", file_path, e)
            self._quarantine(tokens, file_path)
            with self._lock:
                self._stats.read_failures += 1
            return None
        dt = time.time() - t0
        nbytes = _snapshot_nbytes(snapshot)
        self._index.touch(tokens)
        with self._lock:
            self._stats.promote_count += 1
            self._stats.promote_bytes += nbytes
            self._stats.promote_latency_sum += dt
        logger.info(
            "[system_kv_ssd] promoted %d-token snapshot (%.1f MB, %.0f ms)",
            len(tokens),
            nbytes / 1e6,
            dt * 1000.0,
        )
        return snapshot

    def _quarantine(self, tokens: tuple[int, ...], file_path: str) -> None:
        try:
            self._index.delete_entry(tokens)
        except Exception:
            pass
        self._delete_entry(file_path)

    # ---- stats ------------------------------------------------------------

    def get_stats(self) -> dict:
        d = self._stats.to_dict()
        try:
            d["entry_count"] = self._index.get_entry_count()
            d["total_bytes"] = self._index.get_total_bytes()
        except Exception:
            pass
        return d
