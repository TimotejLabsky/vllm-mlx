"""Hybrid-safe checkpoint prefix cache for the batched LLM scheduler.

PATCHES.md #34 — the "item B" port from
docs/fork/continuous-batching-hybrid-caching.md. The default batched
prefix cache (`MemoryAwarePrefixCache`) gets ZERO hits on hybrid
(attention + SSM) models: its supersequence and LCP match paths need to
rewind cached state, recurrent `ArraysCache` state cannot be rewound, and
the `has_non_trimmable` gates correctly skip the match. Every request on
a Qwen3.5/3.6-class model pays full prefill.

This cache reuses the fork's engine-agnostic checkpoint engine
(`system_kv.py`, patches #19/#21): entries hold the full per-layer
snapshot (state + meta_state + trim/ckpt/opaque kind) at
prompt+completion end, plus a position-indexed ladder of recurrent-layer
checkpoints captured at segment boundaries during prefill
(`BatchGenerator.insert_segments` stops exactly there). Fetch = token LCP
over entries → `select_restore_pos` (nearest checkpoint <= divergence;
attention KV slices to any position) → `build_partial_restore_states` →
a fresh per-layer cache list handed to `BatchGenerator.insert(caches=…)`.
The 2026-07-02 gate spike proved mlx-lm 0.31.3 merges such restored
mid-sequence hybrid caches bit-identically into concurrent batches,
including mid-flight insertion.

Enabled per-model via ``VLLM_MLX_BATCHED_SYSTEM_KV=1`` (off by default).
When active it REPLACES the memory-aware cache on the LLM scheduler —
running both would double-store every entry. Knobs shared with the
SimpleEngine stack: ``VLLM_MLX_SYSTEM_KV_SLOTS`` (default 4),
``VLLM_MLX_SYSTEM_KV_RAM_MB`` (0 = unlimited),
``VLLM_MLX_SYSTEM_KV_CHECKPOINTS`` (default 8),
``VLLM_MLX_SYSTEM_KV_PARTIAL_MIN`` (default 256). Batched-only knob:
``VLLM_MLX_BATCHED_KV_CKPT_INTERVAL`` (default 2048 — segment size for
checkpoint capture; keep aligned with ``prefill_step_size``).
"""

import logging
import os
import re
import threading
from collections import OrderedDict
from typing import Any, Optional

from .system_kv import (
    append_checkpoint,
    apply_snapshot_states,
    build_partial_restore_states,
    capture_checkpoint_states,
    capture_snapshot_meta,
    ckpt_bytes,
    classify_layers,
    common_prefix_len,
    entry_bytes,
    select_restore_pos,
)

logger = logging.getLogger(__name__)

ENABLE_ENV = "VLLM_MLX_BATCHED_SYSTEM_KV"


def batched_system_kv_enabled() -> bool:
    return os.environ.get(ENABLE_ENV, "").lower() in ("1", "true", "yes")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _model_slug(tokenizer: Any) -> str:
    """Stable per-model subdir name for the SSD store, derived from the
    tokenizer (the scheduler doesn't know the model name). HF-cache snapshot
    paths reduce to their ``models--org--name`` segment so the slug survives
    revision updates; anything else is sanitized wholesale."""
    name = str(getattr(tokenizer, "name_or_path", "") or "")
    m = re.search(r"models--([A-Za-z0-9._-]+--[A-Za-z0-9._-]+)", name)
    if m:
        name = m.group(1)
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("_")
    return name or "model"


class BatchedSystemKV:
    """LRU of hybrid-safe snapshot entries + per-request checkpoint ladders.

    Thread contract: ``fetch`` runs on the event-loop thread
    (``add_request``); ``capture_segment``/``store`` run on the scheduler's
    executor thread (``step``). All entry/pending mutation is under one
    lock. Stored states follow the patch-#6 aliasing discipline (list
    states shallow-copied, arrays immutable); restored caches are fresh
    ``make_prompt_cache`` objects, so nothing aliases the running batch.
    """

    def __init__(self, model: Any, tokenizer: Any = None, idle_check=None):
        self._model = model
        self._lock = threading.Lock()
        self._entries: OrderedDict[int, dict] = OrderedDict()
        self._entry_seq = 0
        # request_id -> in-flight checkpoint ladder (list of {pos, states, metas})
        self._pending: dict[str, list] = {}
        # request_id -> absolute position already covered by a restored cache
        self._base_pos: dict[str, int] = {}

        self.slots = max(1, _env_int("VLLM_MLX_SYSTEM_KV_SLOTS", 4))
        self.ram_mb = _env_int("VLLM_MLX_SYSTEM_KV_RAM_MB", 0)
        self.ckpt_capacity = max(1, _env_int("VLLM_MLX_SYSTEM_KV_CHECKPOINTS", 8))
        self.partial_min = max(1, _env_int("VLLM_MLX_SYSTEM_KV_PARTIAL_MIN", 256))
        self.ckpt_interval = max(
            256, _env_int("VLLM_MLX_BATCHED_KV_CKPT_INTERVAL", 2048)
        )

        self.hits = 0
        self.misses = 0
        self.partial_hits = 0
        self.tokens_saved = 0
        self.partial_tokens_saved = 0
        self.evictions = 0
        self.boundary_stores = 0
        self.ssd_promotes = 0

        # SSD persistence (fork patch #36) — same store module, format, and
        # envs as the SimpleEngine tier (patch #16/#19/#25), so one llama-swap
        # env block works for either engine. Per-model subdir keeps capacity
        # accounting per model; the slug deliberately differs from
        # SimpleEngine's (tokenizer-derived vs model-name) so the two engines
        # never share a directory — the store is single-writer.
        self._ssd = None
        ssd_base = os.environ.get("VLLM_MLX_SSD_SYSTEM_KV_DIR")
        if ssd_base:
            try:
                from .system_kv_ssd import SystemKVSSDConfig, SystemKVSSDStore

                max_gb = float(
                    os.environ.get("VLLM_MLX_SSD_SYSTEM_KV_GB", "50") or 50
                )
                cache_dir = os.path.join(
                    ssd_base, "batched-" + _model_slug(tokenizer)
                )
                self._ssd = SystemKVSSDStore(
                    SystemKVSSDConfig(cache_dir=cache_dir, max_size_gb=max_gb),
                    idle_check=idle_check,
                )
                self._ssd.start_writer()
                logger.info(
                    "[batched_system_kv] SSD persistence enabled: %s (cap %.0f GB)",
                    cache_dir,
                    max_gb,
                )
            except Exception:
                logger.warning(
                    "[batched_system_kv] SSD init failed; disabled", exc_info=True
                )
                self._ssd = None

    @property
    def has_ssd(self) -> bool:
        return self._ssd is not None

    def close(self) -> None:
        """Drain and close the SSD writer (scheduler reset/shutdown)."""
        if self._ssd is not None:
            try:
                self._ssd.close()
            except Exception:
                logger.debug("[batched_system_kv] SSD close failed", exc_info=True)
            self._ssd = None

    # ------------------------------------------------------------- schedule

    def split_segments(self, tokens: list) -> list:
        """Split inserted tokens at checkpoint boundaries for
        ``insert_segments`` — the generator stops at each boundary, which is
        where ``capture_segment`` snapshots recurrent state."""
        if len(tokens) <= self.ckpt_interval:
            return [list(tokens)]
        return [
            list(tokens[i : i + self.ckpt_interval])
            for i in range(0, len(tokens), self.ckpt_interval)
        ]

    def note_scheduled(self, request_id: str, cached_tokens: int) -> None:
        """Record the restored-prefix offset so segment positions (relative
        to the inserted tokens) map to absolute sequence positions."""
        with self._lock:
            self._base_pos[request_id] = cached_tokens

    # -------------------------------------------------------------- capture

    def capture_segment(self, request_id: str, processed: int, cache_list) -> None:
        """Checkpoint recurrent-layer state at a segment boundary.

        ``cache_list`` is a single-row extraction from the live batch
        (``BatchGenerator.extract_cache``); only ckpt-class layer states are
        kept (attention KV at any position is recoverable from the final
        snapshot by slicing, so the extraction's KV views are dropped).
        """
        try:
            kinds = classify_layers(cache_list)
        except Exception:
            return
        if "ckpt" not in kinds:
            return  # pure-attention model: any position slices, no ladder needed
        states, metas = capture_checkpoint_states(cache_list, kinds=kinds)
        if not states:
            return
        # Materialize now: the extracted states are lazy views into the
        # batch arrays; eval detaches them into independent buffers.
        import mlx.core as mx

        arrs = []
        for st in states.values():
            # ckpt-class states are LISTS for recurrent ArraysCache but
            # TUPLES for RotatingKVCache (sliding-window: gpt-oss, gemma
            # text). Missing the tuple case left Rotating checkpoint states
            # lazy on the executor stream — fetch's cross-thread eval then
            # died with "no Stream(gpu, N)" (found live: gpt-oss batched
            # smoke; the 27B was clean because deltanet states are lists).
            items = st if isinstance(st, (list, tuple)) else [st]
            arrs.extend(a for a in items if a is not None and hasattr(a, "ndim"))
        if arrs:
            mx.eval(arrs)

        with self._lock:
            pos = self._base_pos.get(request_id, 0) + processed
            # append_checkpoint may rebuild the list when thinning — rebind.
            self._pending[request_id] = append_checkpoint(
                self._pending.get(request_id, []),
                pos,
                states,
                metas,
                self.ckpt_capacity,
            )

    def discard_pending(self, request_id: str) -> None:
        with self._lock:
            self._pending.pop(request_id, None)
            self._base_pos.pop(request_id, None)

    # ---------------------------------------------------------------- store

    def _snapshot_cache_list(self, cache_list):
        """(kinds, snapshot, metas) with per-layer eval on the CALLING thread
        (the executor) — patch-#6 aliasing discipline + the realize contract."""
        import mlx.core as mx

        kinds = classify_layers(cache_list)
        snapshot = []
        for c in cache_list:
            st = c.state
            snapshot.append(list(st) if isinstance(st, list) else st)
            # Per-layer incremental eval (mirrors _cleanup_finished's
            # discipline: avoid one deferred lazy-eval spike later).
            items = st if isinstance(st, (list, tuple)) else [st]
            mx.eval([a for a in items if a is not None and hasattr(a, "ndim")])
        return kinds, snapshot, capture_snapshot_meta(cache_list)

    def _insert_entry_locked(self, tokens_list, kinds, snapshot, metas, checkpoints):
        """Insert an entry, absorbing every existing entry whose tokens are a
        (proper or equal) PREFIX of the new chain: their ladders merge in
        (same token chain => their checkpoint states are valid here) and the
        subsumed entries drop. Covers both the identical-re-send shadowing
        case (found by the e2e concurrent scenario) and prompt-boundary
        entries being replaced by their finished chain."""
        absorbed = [
            k
            for k, e in self._entries.items()
            if len(e["tokens"]) <= len(tokens_list)
            and tokens_list[: len(e["tokens"])] == e["tokens"]
        ]
        if absorbed:
            merged = {}
            for k in absorbed:
                for cp in self._entries.pop(k)["checkpoints"]:
                    merged[cp["pos"]] = cp
            for cp in checkpoints:
                merged[cp["pos"]] = cp
            checkpoints = [merged[p] for p in sorted(merged)]
            while len(checkpoints) > max(1, self.ckpt_capacity):
                tail = checkpoints[-1]
                checkpoints = checkpoints[:-1][::2] + [tail]
        entry = {
            "tokens": tokens_list,
            "snapshot": snapshot,
            "metas": metas,
            "kinds": kinds,
            "checkpoints": checkpoints,
            "bytes": entry_bytes(snapshot) + ckpt_bytes(checkpoints),
        }
        self._entry_seq += 1
        self._entries[self._entry_seq] = entry
        self._enforce_budgets_locked()
        return entry

    def store(self, request_id: str, tokens: list, cache_list) -> bool:
        """Store the finished request's full snapshot + its checkpoint ladder."""
        try:
            kinds, snapshot, metas = self._snapshot_cache_list(cache_list)
        except Exception:
            logger.debug("[batched_system_kv] store classify failed", exc_info=True)
            self.discard_pending(request_id)
            return False

        tokens_list = list(tokens)
        with self._lock:
            checkpoints = self._pending.pop(request_id, [])
            self._base_pos.pop(request_id, None)
            entry = self._insert_entry_locked(
                tokens_list, kinds, snapshot, metas, checkpoints
            )
        self._spill(tokens_list, entry)
        return True

    def store_prompt_boundary(self, request_id: str, tokens: list, cache_list) -> bool:
        """Store an entry at the END-OF-PROMPT boundary, mid-request.

        Called from the scheduler's end_of_prompt capture (executor thread)
        so that an aborted/cancelled request — routine for agent clients —
        still leaves its prompt prefill behind as a warm entry. The pending
        ladder is COPIED, not popped: generation continues and the final
        ``store`` still owns it (the finished chain then absorbs this entry
        via prefix subsumption).

        Skipped when the request added less than ``partial_min`` new tokens
        beyond its restored prefix — the donor entry already covers the
        chain, and a near-duplicate would only burn RAM.
        """
        tokens_list = list(tokens)
        with self._lock:
            new_content = len(tokens_list) - self._base_pos.get(request_id, 0)
        if new_content < self.partial_min:
            return False
        try:
            kinds, snapshot, metas = self._snapshot_cache_list(cache_list)
        except Exception:
            logger.debug(
                "[batched_system_kv] boundary snapshot failed", exc_info=True
            )
            return False
        with self._lock:
            checkpoints = list(self._pending.get(request_id, []))
            entry = self._insert_entry_locked(
                tokens_list, kinds, snapshot, metas, checkpoints
            )
            self.boundary_stores += 1
        # Write-through: the boundary entry is exactly what a restart must
        # recover (the agent prompt prefill), so it spills too.
        self._spill(tokens_list, entry)
        logger.info(
            "[batched_system_kv] prompt-boundary store request=%s tokens=%d",
            request_id[:12],
            len(tokens_list),
        )
        return True

    def _spill(self, tokens_list: list, entry: dict) -> None:
        """Async write-through of an entry (post-subsumption: richest ladder)."""
        if self._ssd is None:
            return
        try:
            self._ssd.enqueue_spill(
                tuple(tokens_list),
                entry["snapshot"],
                checkpoints=entry["checkpoints"],
                meta=entry["metas"],
                kinds=entry["kinds"],
            )
        except Exception:
            logger.debug("[batched_system_kv] spill enqueue failed", exc_info=True)

    # ------------------------------------------------------------- ssd tier

    def check_ssd(self, tokens: list) -> Optional[dict]:
        """Index-level probe only (event-loop safe, no blob I/O): a
        full-prefix or shared-prefix SSD candidate, or None. The blob read
        happens on the executor via ``promote_ssd`` — the scheduler's
        ``ssd_pending`` pattern keeps disk reads out of ``add_request``.
        """
        if self._ssd is None:
            return None
        toks = tuple(tokens)
        try:
            row = self._ssd.lookup_prefix(toks)
            if row is not None and row.get("num_tokens", 0) >= self.partial_min:
                return {
                    "tokens": toks[: row["num_tokens"]],
                    "file_path": row["file_path"],
                }
            for row in self._ssd.lookup_shared(toks):
                if row.get("common_len", 0) >= self.partial_min:
                    return {
                        "tokens": tuple(row["tokens"]),
                        "file_path": row["file_path"],
                    }
        except Exception:
            logger.debug("[batched_system_kv] check_ssd failed", exc_info=True)
        return None

    def promote_ssd(self, candidate: dict) -> bool:
        """Load an SSD candidate into the RAM LRU (EXECUTOR thread — the
        store realizes loaded arrays on the calling thread, which keeps the
        cross-thread realize contract). A follow-up ``fetch`` then restores
        through the normal path."""
        if self._ssd is None:
            return False
        entry = self._ssd.read_entry(
            tuple(candidate["tokens"]), candidate["file_path"]
        )
        if entry is None:
            return False
        with self._lock:
            self._insert_entry_locked(
                list(candidate["tokens"]),
                entry["kinds"],
                entry["snapshot"],
                entry["meta"],
                entry["checkpoints"],
            )
            self.ssd_promotes += 1
        return True

    def _enforce_budgets_locked(self) -> None:
        evicted = False
        while len(self._entries) > self.slots:
            self._entries.popitem(last=False)
            self.evictions += 1
            evicted = True
        if self.ram_mb > 0:
            budget = self.ram_mb * 1024 * 1024
            while (
                len(self._entries) > 1
                and sum(e["bytes"] for e in self._entries.values()) > budget
            ):
                self._entries.popitem(last=False)
                self.evictions += 1
                evicted = True
        if evicted:
            import mlx.core as mx

            mx.clear_cache()

    # ---------------------------------------------------------------- fetch

    def fetch(self, tokens: list, request_id: Optional[str] = None) -> Optional[tuple]:
        """Longest-common-prefix match over entries → checkpoint restore.

        Returns ``(cache_list, remaining_tokens, restore_pos)`` or None.
        ``remaining_tokens`` is never empty — at minimum the last token is
        left for the generation kickoff.

        With ``request_id``, a hit seeds the request's pending ladder with
        the donor's checkpoints up to the restore point (plus the restore
        point itself): the restored request CONTINUES the donor's chain, so
        those positions stay valid for the entry it will eventually store.
        Without this, an exact re-send that prefills only the kickoff token
        would store a duplicate chain whose ladder has no usable positions.
        """
        tokens = list(tokens)
        with self._lock:
            best_key = None
            best_lcp = 0
            for key, entry in self._entries.items():
                lcp = common_prefix_len(tokens, entry["tokens"])
                if lcp > best_lcp:
                    best_lcp = lcp
                    best_key = key
            if best_key is None or best_lcp < self.partial_min:
                self.misses += 1
                return None

            entry = self._entries[best_key]
            cap = min(best_lcp, len(tokens) - 1)
            plan = {
                "d": best_lcp,
                "donor_len": len(entry["tokens"]),
                "snapshot": entry["snapshot"],
                "metas": entry["metas"],
                "kinds": entry["kinds"],
                "checkpoints": entry["checkpoints"],
            }
            pos, ck_states, ck_metas = select_restore_pos(plan, cap)
            if pos < self.partial_min:
                self.misses += 1
                return None
            states, metas = build_partial_restore_states(
                entry["snapshot"], ck_states, pos,
                kinds=entry["kinds"], ckpt_metas=ck_metas,
            )
            if states is None:
                self.misses += 1
                return None

            # LRU touch
            self._entries.move_to_end(best_key)
            self.hits += 1
            self.tokens_saved += pos
            divergent = best_lcp < min(len(tokens), len(entry["tokens"]))
            if divergent:
                self.partial_hits += 1
                self.partial_tokens_saved += pos

            if request_id is not None:
                inherited = [
                    cp for cp in entry["checkpoints"] if cp["pos"] <= pos
                ]
                if ck_states and (
                    not inherited or inherited[-1]["pos"] < pos
                ):
                    inherited.append(
                        {"pos": pos, "states": ck_states, "metas": ck_metas}
                    )
                if inherited:
                    self._pending[request_id] = list(inherited)

        from mlx_lm.models.cache import make_prompt_cache

        fresh = make_prompt_cache(self._model)
        apply_snapshot_states(fresh, states, metas)
        # Realize NOW, on the thread that recorded the slice graphs.
        # build_partial_restore_states slices trim-layer KV lazily and fetch
        # runs on the event-loop thread, while the batch steps on the
        # engine-core executor — evaluating the slices over there trips the
        # MLX stream/thread mismatch (patch #28's crash class). Without this,
        # engine_core catches the first step's crash, self-heals to
        # model-thread stepping, and silently re-prefills the request COLD —
        # found live in the Studio A/B (R2 re-send: 32s despite a logged
        # restore), invisible to the single-threaded dev e2e. Realizing here
        # hands concrete buffers across the thread boundary and detaches the
        # restored copy from the donor snapshot.
        import mlx.core as mx

        arrs = []
        for st in states:
            items = st if isinstance(st, (list, tuple)) else [st]
            arrs.extend(a for a in items if a is not None and hasattr(a, "ndim"))
        mx.eval(arrs)
        remaining = tokens[pos:]
        logger.info(
            "[batched_system_kv] restore at %d/%d tokens (lcp=%d%s), "
            "prefilling %d",
            pos, len(tokens), best_lcp,
            ", divergent" if divergent else "", len(remaining),
        )
        return fresh, remaining, pos

    # ---------------------------------------------------------------- stats

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            mem = sum(e["bytes"] for e in self._entries.values())
            return {
                "enabled": True,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": (self.hits / total) if total else 0.0,
                "partial_hits": self.partial_hits,
                "tokens_saved": self.tokens_saved,
                "partial_tokens_saved": self.partial_tokens_saved,
                "evictions": self.evictions,
                "boundary_stores": self.boundary_stores,
                "ssd_promotes": self.ssd_promotes,
                "entry_count": len(self._entries),
                "capacity": self.slots,
                "memory_mb": mem / (1024 * 1024),
                "current_memory_mb": mem / (1024 * 1024),
                "checkpoint_interval": self.ckpt_interval,
                **(
                    {"ssd": self._ssd.get_stats()}
                    if self._ssd is not None
                    else {}
                ),
            }
