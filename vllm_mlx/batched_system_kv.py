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
    capture_checkpoint_states,
    capture_snapshot_meta,
    ckpt_bytes,
    classify_layers,
    common_prefix_len,
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


def _derive_kinds(snapshot: list) -> list:
    """Shape-based kind fallback for legacy (pre-kinds) SSD entries: tuple =>
    trim, list => ckpt — correct for everything that could have produced
    such an entry (same reasoning as build_partial_restore_states)."""
    return ["ckpt" if isinstance(st, list) else "trim" for st in snapshot]


def _is_segments(layer: Any) -> bool:
    return isinstance(layer, list) and bool(layer) and isinstance(layer[0], tuple)


def _entry_nbytes(snapshot: list) -> int:
    """entry_bytes analog that also understands segmented trim layers
    (list of (k, v) tuples). Shared donor segments are counted in full for
    every entry holding them — a conservative overcount that only errs
    toward earlier eviction."""
    n = 0
    for layer in snapshot:
        if _is_segments(layer):
            for k, v in layer:
                n += k.nbytes + v.nbytes
        elif isinstance(layer, tuple) and len(layer) == 2:
            n += layer[0].nbytes + layer[1].nbytes
        elif isinstance(layer, list):
            n += sum(a.nbytes for a in layer if a is not None)
    return n


def _segments_upto(segments: list, pos: int):
    """Store-side prefix reuse: whole segments by REFERENCE up to ``pos``;
    a boundary segment that straddles ``pos`` is sliced and EVALUATED on the
    calling (executor) thread — an O(partial-segment) copy, never O(chain)."""
    import mlx.core as mx

    out = []
    acc = 0
    for k, v in segments:
        n = k.shape[2]
        if acc + n <= pos:
            out.append((k, v))
            acc += n
        else:
            take = pos - acc
            if take > 0:
                pk = k[..., :take, :]
                pv = v[..., :take, :]
                mx.eval(pk, pv)
                out.append((pk, pv))
            acc = pos
        if acc >= pos:
            break
    return out


def _slice_segments(segments: list, pos: int):
    """Restore-side assembly: one (k, v) covering [:pos]. Lazy — the caller
    (fetch) evaluates on its own thread, same single materialization the
    unsegmented slice paid. A single whole segment passes through by
    reference (zero-copy for the pure-extension fast path)."""
    import mlx.core as mx

    parts_k, parts_v = [], []
    acc = 0
    for k, v in segments:
        n = k.shape[2]
        if acc + n <= pos:
            parts_k.append(k)
            parts_v.append(v)
            acc += n
        else:
            take = pos - acc
            if take > 0:
                parts_k.append(k[..., :take, :])
                parts_v.append(v[..., :take, :])
            acc = pos
        if acc >= pos:
            break
    if len(parts_k) == 1:
        return parts_k[0], parts_v[0]
    return mx.concatenate(parts_k, axis=2), mx.concatenate(parts_v, axis=2)


# Consolidate a grown chain's segment list once it exceeds this many pieces
# (one O(chain) concat per ~N turns, amortized — keeps fetch assembly and
# bookkeeping bounded on long agent sessions).
_SEGMENT_CONSOLIDATE_AT = 16


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
        # request_id -> entry key of the chain this request continues
        # (grow-on-HIT donor linkage, fork patch #37)
        self._restore_source: dict[str, int] = {}

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
        self.grown_stores = 0

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
            self._restore_source.pop(request_id, None)

    # ---------------------------------------------------------------- store

    def _build_snapshot(self, request_id: str, tokens_list: list, cache_list):
        """(kinds, snapshot, metas, grown) with per-layer eval on the CALLING
        thread (the executor) — patch-#6 aliasing discipline + the realize
        contract.

        Grow-on-HIT (fork patch #37): when the request continues a chain we
        restored it from (``_restore_source``) and that donor entry is still
        resident with a usable common prefix, trim-layer state is built as
        DONOR SEGMENTS BY REFERENCE plus one O(delta) evaluated slice of the
        finished row — SimpleEngine's grow economics instead of an O(chain)
        copy per turn (multi-GB at deep context). Checkpoint-class layers are
        position-bound and fixed-size, so they copy whole as before.
        """
        import mlx.core as mx

        with self._lock:
            donor_key = self._restore_source.get(request_id)
            donor = self._entries.get(donor_key) if donor_key is not None else None
            donor_tokens = donor["tokens"] if donor is not None else None
            donor_snapshot = donor["snapshot"] if donor is not None else None
            donor_kinds = donor["kinds"] if donor is not None else None

        prefix_len = 0
        if donor_tokens is not None:
            prefix_len = common_prefix_len(tokens_list, donor_tokens)

        kinds = classify_layers(cache_list)
        grown = prefix_len >= self.partial_min and donor_kinds == kinds

        snapshot = []
        for i, c in enumerate(cache_list):
            st = c.state
            if kinds[i] == "trim":
                if grown and _is_segments(donor_snapshot[i]):
                    segs = _segments_upto(donor_snapshot[i], prefix_len)
                    if prefix_len < len(tokens_list):
                        k, v = st
                        dk = k[..., prefix_len:, :]
                        dv = v[..., prefix_len:, :]
                        mx.eval(dk, dv)
                        segs.append((dk, dv))
                    if len(segs) > _SEGMENT_CONSOLIDATE_AT:
                        ck = mx.concatenate([s[0] for s in segs], axis=2)
                        cv = mx.concatenate([s[1] for s in segs], axis=2)
                        mx.eval(ck, cv)
                        segs = [(ck, cv)]
                    snapshot.append(segs)
                else:
                    mx.eval([a for a in st if a is not None])
                    snapshot.append([tuple(st)])  # single segment
            else:
                snapshot.append(list(st) if isinstance(st, list) else st)
                items = st if isinstance(st, (list, tuple)) else [st]
                mx.eval([a for a in items if a is not None and hasattr(a, "ndim")])
        return kinds, snapshot, capture_snapshot_meta(cache_list), grown

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
        if kinds is None:
            kinds = _derive_kinds(snapshot)
        # Normalize: trim layers always hold SEGMENT lists internally (a
        # plain state from an SSD promote or legacy path becomes one segment).
        snapshot = [
            [layer] if kinds[i] == "trim" and not _is_segments(layer) else layer
            for i, layer in enumerate(snapshot)
        ]
        entry = {
            "tokens": tokens_list,
            "snapshot": snapshot,
            "metas": metas,
            "kinds": kinds,
            "checkpoints": checkpoints,
            "bytes": _entry_nbytes(snapshot) + ckpt_bytes(checkpoints),
        }
        self._entry_seq += 1
        self._entries[self._entry_seq] = entry
        self._enforce_budgets_locked()
        return entry

    def store(self, request_id: str, tokens: list, cache_list) -> bool:
        """Store the finished request's snapshot + its checkpoint ladder.

        Grows from the request's donor chain when possible (O(delta));
        grown entries do NOT re-spill — SimpleEngine's policy: a restart
        promotes the stored prefix and re-grows cheaply."""
        tokens_list = list(tokens)
        try:
            kinds, snapshot, metas, grown = self._build_snapshot(
                request_id, tokens_list, cache_list
            )
        except Exception:
            logger.debug("[batched_system_kv] store snapshot failed", exc_info=True)
            self.discard_pending(request_id)
            return False

        with self._lock:
            checkpoints = self._pending.pop(request_id, [])
            self._base_pos.pop(request_id, None)
            self._restore_source.pop(request_id, None)
            entry = self._insert_entry_locked(
                tokens_list, kinds, snapshot, metas, checkpoints
            )
            if grown:
                self.grown_stores += 1
        if not grown:
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
            kinds, snapshot, metas, grown = self._build_snapshot(
                request_id, tokens_list, cache_list
            )
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
            if grown:
                self.grown_stores += 1
            # The final store grows from THIS entry (it absorbed the donor):
            # cascade the linkage to the boundary entry's key.
            self._restore_source[request_id] = self._entry_seq
        # Write-through: the boundary entry is exactly what a restart must
        # recover (the agent prompt prefill). Grown boundary entries skip the
        # re-spill like grown finals — the donor prefix is already on disk.
        if not grown:
            self._spill(tokens_list, entry)
        logger.info(
            "[batched_system_kv] prompt-boundary store request=%s tokens=%d%s",
            request_id[:12],
            len(tokens_list),
            " (grown)" if grown else "",
        )
        return True

    def _spill(self, tokens_list: list, entry: dict) -> None:
        """Async write-through of an entry (post-subsumption: richest ladder).

        Only non-grown entries spill, and their trim layers are single
        segments by construction — unwrap them to the plain states the SSD
        flattener expects."""
        if self._ssd is None:
            return
        try:
            spill_snapshot = [
                layer[0] if _is_segments(layer) and len(layer) == 1 else layer
                for layer in entry["snapshot"]
            ]
            self._ssd.enqueue_spill(
                tuple(tokens_list),
                spill_snapshot,
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

    def _build_restore_states(self, entry: dict, ck_states, pos: int, ck_metas):
        """Per-layer (states, metas) to install at ``pos`` — the segmented
        analog of system_kv.build_partial_restore_states. Trim layers
        assemble from segments (lazy; fetch evaluates on its own thread);
        ckpt layers take the checkpoint captured AT pos; opaque refuses."""
        out, metas = [], []
        for i, layer in enumerate(entry["snapshot"]):
            kind = entry["kinds"][i]
            if kind == "trim":
                out.append(_slice_segments(layer, pos))
                metas.append(None)
            elif kind == "ckpt":
                ck = ck_states.get(i) if ck_states else None
                if ck is None:
                    return None, None
                out.append(ck)
                metas.append(ck_metas.get(i) if ck_metas else None)
            else:  # opaque
                return None, None
        return out, metas

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
            states, metas = self._build_restore_states(
                entry, ck_states, pos, ck_metas
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
                # Grow-on-HIT donor linkage (fork patch #37): the eventual
                # store reuses this entry's trim segments by reference.
                self._restore_source[request_id] = best_key

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
                "grown_stores": self.grown_stores,
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


# ---------------------------------------------------------------------------
# Scheduler seam (fork patches #33–#37).
#
# The bodies of the scheduler's fork hooks live HERE so upstream churn in
# vllm_mlx/scheduler.py stays away from fork logic — the same containment
# pattern patch #18 proved for engine/simple.py. Scheduler methods are
# one-line delegators into these functions; behavior is pinned by the
# scheduler-level wiring tests (pure code motion, zero test edits).
# ---------------------------------------------------------------------------


def maybe_create(model: Any, tokenizer: Any, idle_check=None):
    """Construct the cache when enabled — the scheduler __init__ hook."""
    if not batched_system_kv_enabled():
        return None
    kv = BatchedSystemKV(model, tokenizer=tokenizer, idle_check=idle_check)
    logger.info(
        "[batched_system_kv] enabled: slots=%d ram_mb=%d "
        "ckpt_interval=%d partial_min=%d — replaces memory-aware cache",
        kv.slots,
        kv.ram_mb,
        kv.ckpt_interval,
        kv.partial_min,
    )
    return kv


def fetch_for_request(hybrid_kv: "BatchedSystemKV", request) -> None:
    """add_request hook: hybrid-safe checkpoint restore (#34) with the
    index-only SSD cold-tier probe on miss (#36 — the blob read happens on
    the executor via promote_ssd_pending)."""
    result = hybrid_kv.fetch(
        request.prompt_token_ids, request_id=request.request_id
    )
    if result is not None:
        cache, remaining, pos = result
        request.cache_hit_type = "system_kv"
        request.prompt_cache = cache
        request.cached_tokens = pos
        request.remaining_tokens = remaining
    else:
        request.cache_hit_type = "miss"
        request.remaining_tokens = request.prompt_token_ids
        candidate = hybrid_kv.check_ssd(request.prompt_token_ids)
        if candidate is not None:
            request.cache_hit_type = "ssd_pending"
            request._ssd_candidate = candidate


def promote_ssd_pending(scheduler) -> None:
    """_schedule_waiting hook (#36): promote SSD candidates for waiting
    ssd_pending requests. Runs on the executor thread — the blob read +
    array realize happen here, never on the event loop."""
    hybrid_kv = scheduler.hybrid_kv
    for request in scheduler.waiting:
        if getattr(request, "cache_hit_type", None) != "ssd_pending":
            continue
        candidate = getattr(request, "_ssd_candidate", None)
        request._ssd_candidate = None
        result = None
        if candidate is not None:
            try:
                if hybrid_kv.promote_ssd(candidate):
                    result = hybrid_kv.fetch(
                        request.prompt_token_ids,
                        request_id=request.request_id,
                    )
            except Exception:
                logger.debug(
                    "[batched_system_kv] SSD promote failed "
                    f"request={request.request_id[:12]}",
                    exc_info=True,
                )
        if result is not None:
            cache, remaining, pos = result
            request.cache_hit_type = "system_kv"
            request.prompt_cache = cache
            request.cached_tokens = pos
            request.remaining_tokens = remaining
            logger.info(
                f"[batched_system_kv] SSD promote request="
                f"{request.request_id[:12]} restored={pos} "
                f"remaining={len(remaining)}"
            )
        else:
            request.cache_hit_type = "miss"


def capture_checkpoints(scheduler, prompt_responses) -> None:
    """step() hook (#34/#35): at each segment boundary, extract the row's
    cache and checkpoint its recurrent-layer state at that absolute
    position; at end_of_prompt, persist the prompt-boundary entry for
    abort resilience.

    Runs on the scheduler executor thread (same thread as the generation
    step), so extract_cache is safe. Only the ckpt-class layer states are
    kept from mid-prefill extractions; the attention-KV views are dropped.
    The boundary store is SOLO-REQUEST ONLY: under a concurrent burst the
    ~500MB snapshot materialization per boundary lands inside the busy
    step loop and measurably inflates batch TTFT (Studio bench:
    0.37s -> ~1.7s at conc=4); concurrent chains still store at finish.
    """
    hybrid_kv = scheduler.hybrid_kv
    for pr in prompt_responses:
        if not getattr(pr, "end_of_segment", False):
            continue
        request_id = scheduler.uid_to_request_id.get(pr.uid)
        if request_id is None:
            continue
        try:
            extracted = scheduler.batch_generator.extract_cache([pr.uid])
            entry = extracted.get(pr.uid)
            if not entry:
                continue
            progress = getattr(pr, "progress", None)
            processed = (
                progress[0] if isinstance(progress, tuple) else int(progress or 0)
            )
            hybrid_kv.capture_segment(request_id, processed, entry[0])
            if getattr(pr, "end_of_prompt", False) and len(scheduler.running) <= 1:
                request = scheduler.requests.get(request_id)
                if request is not None and request.prompt_token_ids:
                    hybrid_kv.store_prompt_boundary(
                        request_id,
                        request.prompt_token_ids,
                        entry[0],
                    )
        except Exception:
            logger.debug(
                "[batched_system_kv] checkpoint capture failed "
                f"request={request_id[:12]}",
                exc_info=True,
            )


def store_finished(hybrid_kv: "BatchedSystemKV", request_id: str, request) -> None:
    """_cleanup_finished hook (#34/#37): store the finished chain (grows
    from its donor when possible) and release the extraction reference."""
    if getattr(request, "_extracted_cache", None) is not None:
        try:
            full_token_sequence = list(request.prompt_token_ids) + list(
                request.output_token_ids
            )
            stored = hybrid_kv.store(
                request_id,
                full_token_sequence,
                request._extracted_cache,
            )
            logger.info(
                f"[batched_system_kv] store request={request_id[:12]} "
                f"tokens={len(full_token_sequence)} stored={stored}"
            )
            # Release: the cache holds its own snapshot refs.
            request._extracted_cache = None
        except Exception:
            logger.debug(
                f"[batched_system_kv] store failed {request_id}",
                exc_info=True,
            )
            hybrid_kv.discard_pending(request_id)
    else:
        hybrid_kv.discard_pending(request_id)


def insert_segmented(hybrid_kv: "BatchedSystemKV", batch_generator, request, tokens, insert_kwargs):
    """_schedule_waiting insert hook (#34): split the prompt at checkpoint
    boundaries so the generator stops there (insert_segments) and
    capture_checkpoints can snapshot recurrent state."""
    hybrid_kv.note_scheduled(request.request_id, request.cached_tokens)
    return batch_generator.insert_segments(
        [hybrid_kv.split_segments(tokens)],
        **insert_kwargs,
    )
