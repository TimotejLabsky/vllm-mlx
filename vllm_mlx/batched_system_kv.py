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

from .memory_pressure import PressureManager

from .system_kv import (
    TEMPLATE_MARKERS,
    CacheTimingRecorder,
    append_checkpoint,
    detect_template_markers,
    thin_checkpoints,
    apply_snapshot_states,
    capture_checkpoint_states,
    capture_snapshot_meta,
    ckpt_bytes,
    classify_layers,
    common_prefix_len,
    is_recurrent_state,
    pin_state,
    select_restore_pos,
    state_arrays,
    timing_key,
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
    return ["ckpt" if is_recurrent_state(st) else "trim" for st in snapshot]


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
        else:
            n += sum(a.nbytes for a in state_arrays(layer))
    return n


def _trim_nbytes(snapshot: list, kinds: list) -> int:
    """Bytes of the trim-class (attention KV) layers only — the part of an
    entry that scales with chain length. Checkpoint-class layers (recurrent
    ArraysCache, Rotating windows) are fixed-size per entry (#100)."""
    n = 0
    for i, layer in enumerate(snapshot):
        if kinds[i] != "trim":
            continue
        if _is_segments(layer):
            for k, v in layer:
                n += k.nbytes + v.nbytes
        else:
            n += sum(a.nbytes for a in state_arrays(layer))
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
        # Entry-lifecycle timing (fork observability). Keyed by content
        # (timing_key of the token chain), NOT _entry_seq — a re-store of
        # an evicted chain must match its tombstone.
        self._timing = CacheTimingRecorder()
        # request_id -> in-flight checkpoint ladder (list of {pos, states, metas})
        self._pending: dict[str, list] = {}
        # request_id -> absolute position already covered by a restored cache
        self._base_pos: dict[str, int] = {}
        # request_id -> entry key of the chain this request continues
        # (grow-on-HIT donor linkage, fork patch #37)
        self._restore_source: dict[str, int] = {}

        self.slots = max(1, _env_int("VLLM_MLX_SYSTEM_KV_SLOTS", 4))
        self.ram_mb = _env_int("VLLM_MLX_SYSTEM_KV_RAM_MB", 0)
        # Dynamic RAM budget (#101): RAM_MB becomes the FLOOR; the bag may grow
        # into free headroom under the watermark (minus a prefill reserve), up
        # to RAM_MAX_MB. Needs RAM_MB > 0 and the watermark armed; otherwise
        # the static budget applies unchanged.
        self.ram_dynamic = _env_int("VLLM_MLX_SYSTEM_KV_RAM_DYNAMIC", 0) > 0
        self.ram_max_mb = _env_int("VLLM_MLX_SYSTEM_KV_RAM_MAX_MB", 0)
        self.ram_reserve_mb = _env_int("VLLM_MLX_SYSTEM_KV_RAM_RESERVE_MB", 8192)
        self.ckpt_capacity = max(1, _env_int("VLLM_MLX_SYSTEM_KV_CHECKPOINTS", 8))
        self.partial_min = max(1, _env_int("VLLM_MLX_SYSTEM_KV_PARTIAL_MIN", 256))
        self.ckpt_interval = max(
            256, _env_int("VLLM_MLX_BATCHED_KV_CKPT_INTERVAL", 2048)
        )
        # Message-boundary checkpoints (#88): minimum token distance between
        # accepted boundary splits, so a burst of short turns doesn't shred
        # the ladder (llama.cpp #24176 raised theirs 256 -> 8192; start at
        # the proven interval and tune from ladder data).
        self.boundary_min_step = max(
            0, _env_int("VLLM_MLX_BATCHED_KV_BOUNDARY_MIN_STEP", 2048)
        )
        # request_id -> frozenset of ABSOLUTE message-boundary positions,
        # recorded at insert so capture_segment can flag boundary-aligned
        # checkpoints for the thinning policy.
        self._boundary_pos: dict[str, frozenset] = {}
        # template family -> tuple of marker token-id sequences (encoded
        # once per family per process; markers are special tokens with
        # stable ids, so token-space scanning avoids the byte-offset
        # translation llama.cpp #24176 dropped for the same reason).
        self._marker_ids: dict[str, tuple] = {}
        # Admission-gate budgets (0 = disabled). See should_defer_cobatch.
        self.pad_waste_mb = _env_int("VLLM_MLX_BATCHED_PAD_WASTE_MB", 0)
        # Total padded-KV byte budget: effective concurrency floats on the
        # request mix instead of a fixed --max-num-seqs (deep contexts
        # serialize themselves, short ones batch up to the hard cap).
        self.kv_budget_mb = _env_int("VLLM_MLX_BATCHED_KV_BUDGET_MB", 0)
        # (#102) floor for the admission gates' bytes/token: a freshly spawned
        # (or just-recovered) generator measures only the KV already
        # allocated, which prices not-yet-prefilled rows near zero. Set it to
        # the model's KV bytes/token (Qwen3.8-27B: 64 KiB).
        self.bpt_floor_kb = _env_int("VLLM_MLX_BATCHED_BPT_FLOOR_KB", 0)
        # (#105) Solo-prefill guard. Every admission gate above prices
        # CO-batching; a request that finds nothing running is admitted
        # unexamined ("progress guarantee"), and #48 relief only acts between
        # steps, so a solo prefill whose KV + per-chunk transient does not
        # fit below the OOM wall dies mid-step. TRANSIENT_MB is the measured
        # active->peak jump of one prefill chunk (0 = guard off; 27B-4bit:
        # +6.7 GB measured 2026-09-16). CEILING_PCT is the share of the
        # recommended working set the projection must stay under — the OOM
        # wall, deliberately NOT the relief watermark (crossing that is
        # normal for a deep solo prefill, and relief handles it).
        self.solo_transient_mb = _env_int("VLLM_MLX_BATCHED_SOLO_TRANSIENT_MB", 0)
        self.solo_ceiling_pct = _env_int("VLLM_MLX_BATCHED_SOLO_CEILING_PCT", 95)
        self.solo_relief_passes = 0
        self.solo_rejections = 0
        # (#106) Two holes the 2026-09-17 OOMs (63.0-63.6 GB, three times in
        # 2.5 h) went through with KV_BUDGET_MB=8192, all with one signature:
        # a ~65-70K row decoding, a second deep row co-batched, and a 67-72K
        # token cache RESTORE realised for a request that was still WAITING.
        #
        # LAZY_RESTORE: add_request only MATCHES (and LRU-touches the entry);
        # the multi-GB restored copy is built on the executor at the moment
        # the request is admitted. Before, every queued request pinned its
        # restored cache (4-5 GB at 70K) for its whole wait, in no budget.
        #
        # PROJECTED_ADMISSION: the co-batch gates price padded KV against a
        # budget but never look at what the process already holds. Project
        # active + merged padded copy + the new row's growth + its restore +
        # one chunk's transient against the OOM wall (the #105 knobs), make
        # room first, and DEFER - never reject - what still does not fit.
        self.lazy_restore = _env_int("VLLM_MLX_BATCHED_LAZY_RESTORE", 0) > 0
        self.projected_admission = (
            _env_int("VLLM_MLX_BATCHED_PROJECTED_ADMISSION", 0) > 0
        )
        self.lazy_restores = 0
        self.lazy_restore_misses = 0
        # (#107) request_id -> key of the entry a QUEUED request matched at
        # enqueue. Make-room and budget eviction spare these while anything
        # else can go: on 2026-09-21 a deep follow-up waited ~10 min behind
        # the KV budget, lost its entry to make-room, and re-prefilled 60K
        # tokens cold (18 min). Watermark relief can still take everything.
        self._peeked: dict[str, int] = {}
        self.lazy_ssd_fallbacks = 0
        self.projected_defers = 0
        self.projected_relief_passes = 0
        # Ground-truth backstop: defer co-batching while MLX active memory
        # exceeds this percentage of the device's recommended working set.
        self.mem_watermark_pct = _env_int("VLLM_MLX_BATCHED_MEM_WATERMARK_PCT", 0)
        self.admission_deferrals = 0
        # Memory-pressure relief (#48) — same watermark env as the admission
        # gate. Watermark math + relief loop live in the cache-agnostic
        # PressureManager (memory_pressure.py, extracted for the MLLM branch);
        # this bag contributes its LRU eviction and keeps the counters.
        self._pressure = PressureManager(self.mem_watermark_pct)
        self._bpt_hint = 0.0  # survives an emptied bag (see bytes_per_token)
        # Fixed per-entry bytes of checkpoint-class layer state (#100) —
        # priced once per store copy, not per token.
        self._fixed_hint = 0.0
        self.pressure_evictions = 0
        self.pressure_skipped_stores = 0
        self.pressure_cache_clears = 0  # watermark breaches relieved (#53)
        self.pressure_backlog_drops = 0  # SSD spill backlogs relieved (#103)

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
                # (#103) The store's defaults (12 GB backlog, write only when
                # idle) were sized for SimpleEngine's turn gaps. A batched
                # route under agent load has none: the backlog sat at the cap
                # for hours, ~12 GB the bag never accounted for and relief
                # never touched. Smaller cap, and a bounded wait after which
                # a spill may be written while busy if memory allows.
                queued_gb = float(
                    os.environ.get("VLLM_MLX_SSD_SYSTEM_KV_MAX_QUEUED_GB", "4") or 4
                )
                busy_after = float(
                    os.environ.get("VLLM_MLX_SSD_SYSTEM_KV_BUSY_WRITE_S", "30") or 0
                )
                self._ssd = SystemKVSSDStore(
                    SystemKVSSDConfig(
                        cache_dir=cache_dir,
                        max_size_gb=max_gb,
                        max_queued_gb=queued_gb,
                        busy_write_after_s=busy_after,
                    ),
                    idle_check=idle_check,
                    can_write_busy=lambda: not self.under_pressure(),
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

    def split_segments(self, tokens: list, boundaries=()) -> list:
        """Split inserted tokens at checkpoint boundaries for
        ``insert_segments`` — the generator stops at each boundary, which is
        where ``capture_segment`` snapshots recurrent state.

        #88: ``boundaries`` (positions relative to ``tokens``) add message-
        boundary cuts IN ADDITION to the uniform interval; the interval
        stays a hard upper bound on segment length, so a 60K-token tool
        result can never become one enormous prefill chunk (the #48/#53
        memory rails are tuned against interval-sized transients)."""
        n = len(tokens)
        cuts = sorted({int(b) for b in boundaries if 0 < int(b) < n})
        if not cuts and n <= self.ckpt_interval:
            return [list(tokens)]
        segments = []
        start = 0
        for cut in cuts + [n]:
            while cut - start > self.ckpt_interval:
                segments.append(list(tokens[start : start + self.ckpt_interval]))
                start += self.ckpt_interval
            if cut > start:
                segments.append(list(tokens[start:cut]))
                start = cut
        return segments

    def boundary_marker_ids(self, prompt_text, tokenizer) -> tuple:
        """Token-id sequences of the prompt's template-family turn markers
        (#88). Detected from the rendered prompt via patch #20's marker
        table, encoded once per family per process."""
        if not prompt_text or tokenizer is None:
            return ()
        family, _idx, _gen = detect_template_markers(prompt_text)
        if family is None:
            return ()
        cached = self._marker_ids.get(family)
        if cached is not None:
            return cached
        seqs = []
        for fam, boundary_markers, _g in TEMPLATE_MARKERS:
            if fam != family:
                continue
            for marker in boundary_markers:
                try:
                    ids = tokenizer.encode(marker, add_special_tokens=False)
                except TypeError:
                    ids = tokenizer.encode(marker)
                    bos = getattr(tokenizer, "bos_token_id", None)
                    if bos is not None and ids and ids[0] == bos:
                        ids = ids[1:]
                if ids:
                    seqs.append(tuple(ids))
            break
        self._marker_ids[family] = tuple(seqs)
        return self._marker_ids[family]

    def note_scheduled(
        self, request_id: str, cached_tokens: int, boundaries=()
    ) -> None:
        """Record the restored-prefix offset so segment positions (relative
        to the inserted tokens) map to absolute sequence positions — and
        (#88) the absolute message-boundary set, so captures at those
        positions get the preferred-survivor flag."""
        with self._lock:
            self._base_pos[request_id] = cached_tokens
            if boundaries:
                self._boundary_pos[request_id] = frozenset(
                    cached_tokens + int(b) for b in boundaries
                )

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
            # ckpt-class states are LISTS for recurrent ArraysCache (or the
            # 1632 3-tuple, #81) but (k, v) TUPLES for RotatingKVCache
            # (sliding-window: gpt-oss, gemma text). Missing the tuple case
            # left Rotating checkpoint states lazy on the executor stream —
            # fetch's cross-thread eval then died with "no Stream(gpu, N)"
            # (found live: gpt-oss batched smoke; the 27B was clean because
            # deltanet states are lists). state_arrays walks every shape.
            arrs.extend(state_arrays(st))
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
                boundary=pos in self._boundary_pos.get(request_id, ()),
            )

    def discard_pending(self, request_id: str) -> None:
        with self._lock:
            self._pending.pop(request_id, None)
            self._base_pos.pop(request_id, None)
            self._boundary_pos.pop(request_id, None)
            self._restore_source.pop(request_id, None)
            self._peeked.pop(request_id, None)

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
                snapshot.append(pin_state(st))
                mx.eval(state_arrays(st))
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
                popped = self._entries.pop(k)
                # Subsumed, not lost — the chain lives on in the new entry.
                self._timing.forget(timing_key(popped["tokens"]))
                for cp in popped["checkpoints"]:
                    merged[cp["pos"]] = cp
            for cp in checkpoints:
                merged[cp["pos"]] = cp
            checkpoints = [merged[p] for p in sorted(merged)]
            checkpoints = thin_checkpoints(checkpoints, self.ckpt_capacity)
        if kinds is None:
            kinds = _derive_kinds(snapshot)
        # Normalize: trim layers always hold SEGMENT lists internally (a
        # plain state from an SSD promote or legacy path becomes one segment).
        snapshot = [
            [layer] if kinds[i] == "trim" and not _is_segments(layer) else layer
            for i, layer in enumerate(snapshot)
        ]
        snapshot_bytes = _entry_nbytes(snapshot)
        trim_bytes = _trim_nbytes(snapshot, kinds)
        entry = {
            "tokens": tokens_list,
            "snapshot": snapshot,
            "metas": metas,
            "kinds": kinds,
            "checkpoints": checkpoints,
            "bytes": snapshot_bytes + ckpt_bytes(checkpoints),
            "trim_bytes": trim_bytes,
            "fixed_bytes": snapshot_bytes - trim_bytes,
        }
        if tokens_list:
            # Learn bytes/token AT insert — a lazily-learned hint misses the
            # serial workload where relief empties the bag before anything
            # reads it (live 2026-07-09 round 3). Per-token cost counts the
            # attention KV only (#100): folding a hybrid's fixed recurrent
            # state (~hundreds of MB on a 27B) into bytes/len inflated the
            # estimate by orders of magnitude after a short entry, and the
            # overshoot gate then refused every ordinary store.
            self._bpt_hint = trim_bytes / len(tokens_list)
            self._fixed_hint = float(entry["fixed_bytes"])
        self._entry_seq += 1
        self._entries[self._entry_seq] = entry
        self._timing.note_store(timing_key(tokens_list))
        self._enforce_budgets_locked()
        return entry

    def store(self, request_id: str, tokens: list, cache_list) -> bool:
        """Store the finished request's snapshot + its checkpoint ladder.

        Grows from the request's donor chain when possible (O(delta));
        grown entries do NOT re-spill — SimpleEngine's policy: a restart
        promotes the stored prefix and re-grows cheaply."""
        tokens_list = list(tokens)
        if len(tokens_list) < self.partial_min:
            # A sub-floor chain can never serve a restore (fetch needs an
            # LCP >= partial_min) — storing it only burns a slot of the bag
            # and skews the learned footprint. Before #100 the finish store
            # never ran on hybrids, so this case was unreachable; now every
            # short request (warmups, titles) would otherwise land here.
            self.discard_pending(request_id)
            return False
        refusal = None
        if self.under_pressure():
            refusal = "over_watermark"
        elif self._store_would_overshoot(tokens_list):
            refusal = "copy_would_overshoot"
        if refusal and not self._may_grow(request_id, tokens_list):
            # A non-grown store materializes a full-chain snapshot copy
            # (multi-GB at deep context) — the copy itself is the spike, so
            # the gate prices it in (#48 crash math). Skip it; the SSD tier
            # keeps restart recovery and the next completed turn stores
            # normally.
            self.pressure_skipped_stores += 1
            self.discard_pending(request_id)
            logger.info(
                "[batched_system_kv] skipping store under memory pressure "
                "request=%s tokens=%d reason=%s",
                request_id[:12],
                len(tokens_list),
                refusal,
            )
            return False
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
        if self.under_pressure() or self._store_would_overshoot(tokens_list):
            # Boundary stores land at the WORST moment — the end of a deep
            # prefill IS the observed crash peak (#48) — and they are pure
            # abort insurance. Skip; the final store covers the finished
            # chain, and grown boundary deltas can be re-prefilled.
            self.pressure_skipped_stores += 1
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
            # (#108) While the spill sits in the write queue the queue holds
            # the SAME arrays as this entry, so evicting the entry frees no
            # memory - it only makes the chain unreachable (not in RAM, not
            # yet in the SSD index) until the write lands. 2026-09-21: a
            # follow-up arriving 10 s after its chain finished re-prefilled
            # 60K tokens cold because of exactly that. The flag clears when
            # the spill is written, fails, or is dropped by relief.
            entry["spill_pending"] = True

            def _landed(entry=entry):
                entry["spill_pending"] = False

            if not self._ssd.enqueue_spill(
                tuple(tokens_list),
                spill_snapshot,
                checkpoints=entry["checkpoints"],
                meta=entry["metas"],
                kinds=entry["kinds"],
                on_done=_landed,
            ):
                entry["spill_pending"] = False
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

    def effective_ram_bytes(self, bag_bytes: float) -> float:
        """RAM budget for the bag right now (#101).

        Static mode: ``RAM_MB``. Dynamic mode: the bag may keep what it holds
        plus the free headroom under the memory watermark, minus a reserve for
        the next prefill/batch spike, clamped to ``[RAM_MB, RAM_MAX_MB]``:

            budget = bag + (watermark - active - reserve)

        ``bag`` is inside ``active``, so evicting an entry lowers both sides
        equally — the budget is stable across one enforcement pass. Idle, a
        single session's chains grow into the free working set; under
        concurrent load the budget falls back to the floor and #48 relief keeps
        shrinking the bag between steps. Falls back to static whenever the
        watermark is disabled or Metal is unavailable (e.g. unit schedulers).
        """
        static = float(self.ram_mb) * 1024 * 1024
        if not self.ram_dynamic or self.ram_mb <= 0:
            return static
        threshold = self._pressure.threshold_bytes()
        if threshold is None:
            return static
        try:
            import mlx.core as mx

            active = float(mx.get_active_memory())
        except Exception:
            return static
        budget = bag_bytes + (
            threshold - active - float(self.ram_reserve_mb) * 1024 * 1024
        )
        if self.ram_max_mb > 0:
            budget = min(budget, float(self.ram_max_mb) * 1024 * 1024)
        return max(budget, static)

    def _pop_victim_locked(self, spare_pinned: bool, keep_newest: bool = False):
        """(#107) Pop the LRU entry no queued request has matched. With every
        candidate pinned: None when ``spare_pinned``, else plain LRU (a budget
        still has to hold). ``keep_newest`` never offers the entry that was
        just inserted - a promote must not evict itself."""
        pinned = set(self._peeked.values())
        keys = list(self._entries)
        if keep_newest:
            keys = keys[:-1]

        def queued(key) -> bool:  # (#108) its spill still sits in the queue
            return bool(self._entries[key].get("spill_pending"))

        for key in keys:
            if key not in pinned and not queued(key):
                return self._entries.pop(key)
        if spare_pinned or not keys:
            return None
        # Everything is spoken for and a budget still has to hold. A pinned
        # entry at least FREES memory; a spill-pending one frees nothing until
        # its write lands, so it goes last.
        for key in keys:
            if not queued(key):
                return self._entries.pop(key)
        return self._entries.pop(keys[0])

    def _enforce_budgets_locked(self) -> None:
        evicted = False
        while len(self._entries) > self.slots:
            ev = self._pop_victim_locked(spare_pinned=False, keep_newest=True)
            if ev is None:
                break
            self._timing.note_evict(timing_key(ev["tokens"]))
            self.evictions += 1
            evicted = True
        if self.ram_mb > 0:
            budget = self.effective_ram_bytes(
                sum(e["bytes"] for e in self._entries.values())
            )
            while (
                len(self._entries) > 1
                and sum(e["bytes"] for e in self._entries.values()) > budget
            ):
                ev = self._pop_victim_locked(spare_pinned=False, keep_newest=True)
                if ev is None:
                    break
                self._timing.note_evict(timing_key(ev["tokens"]))
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
            self._timing.note_hit(timing_key(entry["tokens"]))
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

    def peek(self, tokens: list, request_id: Optional[str] = None) -> int:
        """(#106) The restore position ``fetch`` WOULD use, without building
        anything: no state slices, no copy, no counters. 0 = no usable match.

        LRU-touches the matched entry, so the chain a queued request is about
        to extend outlives an idle one while it waits."""
        tokens = list(tokens)
        with self._lock:
            best_key, best_lcp = None, 0
            for key, entry in self._entries.items():
                lcp = common_prefix_len(tokens, entry["tokens"])
                if lcp > best_lcp:
                    best_lcp, best_key = lcp, key
            if best_key is None or best_lcp < self.partial_min:
                return 0
            entry = self._entries[best_key]
            plan = {
                "d": best_lcp,
                "donor_len": len(entry["tokens"]),
                "snapshot": entry["snapshot"],
                "metas": entry["metas"],
                "kinds": entry["kinds"],
                "checkpoints": entry["checkpoints"],
            }
            pos, _states, _metas = select_restore_pos(
                plan, min(best_lcp, len(tokens) - 1)
            )
            if pos < self.partial_min:
                return 0
            self._entries.move_to_end(best_key)
            if request_id is not None:
                self._peeked[request_id] = best_key  # (#107) spared while it waits
            return pos

    def release_peek(self, request_id: str) -> None:
        with self._lock:
            self._peeked.pop(request_id, None)

    def restore_bytes(self, cached_tokens: int) -> float:
        """What materialising a restore of ``cached_tokens`` will add to
        active memory: sliced attention KV + one copy of the fixed
        checkpoint-class state."""
        if cached_tokens <= 0:
            return 0.0
        bpt = max(self.bytes_per_token(), float(self.bpt_floor_kb * 1024))
        return cached_tokens * bpt + self._fixed_hint

    def bytes_per_token(self) -> float:
        """Per-token KV footprint learned from the newest resident entry
        (≈200 KB/token on a 27B-4bit). 0.0 until an entry has EVER been
        inserted — the guards stay inert on a genuinely cold cache (the
        first request runs solo anyway). The estimate is learned at insert
        time and persists after the bag empties: pressure relief evicts
        the whole bag exactly when the next deep store arrives (live
        2026-07-09 rounds 2–3 — the 94K store priced itself against a 0
        estimate and landed a 7 GB copy), so an empty bag must not disarm
        the overshoot gate."""
        with self._lock:
            for entry in reversed(self._entries.values()):
                n = len(entry["tokens"])
                if n > 0:
                    return entry.get("trim_bytes", entry["bytes"]) / n
            return self._bpt_hint

    def fixed_store_bytes(self) -> float:
        """Fixed (length-independent) bytes a full snapshot copy materializes:
        the checkpoint-class layer state of the newest entry, or the hint
        learned at insert once the bag has emptied (#100)."""
        with self._lock:
            for entry in reversed(self._entries.values()):
                if entry["tokens"]:
                    return float(entry.get("fixed_bytes", 0))
            return self._fixed_hint

    # ------------------------------------------------- memory pressure (#48)

    def _threshold_bytes(self) -> Optional[float]:
        """Watermark threshold in bytes, or None when disabled/unavailable
        (delegated — see memory_pressure.PressureManager)."""
        return self._pressure.threshold_bytes()

    def watermark_status(self) -> tuple:
        """``(over, active_bytes, ceiling_bytes)`` against the watermark.
        ``(False, 0, 0)`` when the watermark env is unset or Metal is
        unavailable."""
        return self._pressure.watermark_status()

    def under_pressure(self) -> bool:
        return self._pressure.under_pressure()

    def _store_would_overshoot(self, tokens_list: list) -> bool:
        """Would materializing a full snapshot of this chain push active
        memory over the watermark? The live 2026-07-09 deploy smoke: a
        94K-token final store passed the instantaneous-active gate at
        48.9 GB (the batch KV had just been freed) and then materialized
        a 7 GB copy — the store itself IS the spike, so the gate must
        price it in (entry size ≈ tokens × learned bytes/token). Inert on
        a cold cache (no bytes/token estimate) and when the watermark is
        unset."""
        threshold = self._threshold_bytes()
        if threshold is None:
            return False
        bpt = self.bytes_per_token()
        if bpt <= 0:
            return False
        try:
            import mlx.core as mx

            copy_bytes = len(tokens_list) * bpt + self.fixed_store_bytes()
            return mx.get_active_memory() + copy_bytes > threshold
        except Exception:
            return False

    def _drop_lru_entry(self, spare_pinned: bool = False) -> bool:
        """Drop the least-recently-used snapshot entry; False when empty.
        ``spare_pinned`` (#107): skip entries a queued request has matched -
        for make-room passes. Watermark relief calls it without: against a
        real OOM everything is fair game.

        The bag is pure cache: non-grown entries reached SSD via
        write-through and grown entries re-grow from their spilled prefix,
        so dropping entries costs at most a re-prefill of recent deltas —
        against what relief prevents: Metal "Insufficient Memory" inside a
        command-buffer completion handler is an uncatchable SIGABRT (the
        2026-07-08 canary crashes)."""
        with self._lock:
            if not self._entries:
                return False
            ev = self._pop_victim_locked(spare_pinned=spare_pinned)
            if ev is None:
                return False
            self._timing.note_evict(timing_key(ev["tokens"]))
            self.evictions += 1
            self.pressure_evictions += 1
        return True

    def relieve_pressure(self) -> int:
        """Evict resident snapshot entries when the PEAK memory since the
        last check crossed the watermark.

        The peak-trigger / clear-even-when-empty (#53) / LRU-until-under
        discipline lives in ``memory_pressure.PressureManager.relieve``
        (extracted for the batched MLLM branch); this bag contributes
        ``_drop_lru_entry`` and keeps the counters."""
        def _count_clear() -> None:
            self.pressure_cache_clears += 1

        def _drop_spill_backlog() -> bool:
            # (#103) Queued spills are resident copies no request can use —
            # give them back before any entry a chain is about to extend.
            if self._ssd is None or not self._ssd.drop_backlog():
                return False
            self.pressure_backlog_drops += 1
            return True

        _, evicted = self._pressure.relieve(
            self._drop_lru_entry,
            log_label="batched_system_kv",
            on_cache_clear=_count_clear,
            before_evict=_drop_spill_backlog,
        )
        return evicted

    def solo_prefill_verdict(self, tokens_to_prefill: int) -> Optional[str]:
        """(#105) Examine a request about to run ALONE. None = admit;
        otherwise the ``error_kind`` to reject it with.

        Projects ``active + tokens x bytes/token + one chunk's transient``
        against the OOM wall. Over it, everything that is pure cache is given
        back first (spill backlog, then the bag LRU-first, then the MLX
        buffer cache) and the projection re-checked; only a request that
        still cannot fit is rejected — before it costs a prefill and takes
        the process's state down with it. Inert without TRANSIENT_MB, a
        bytes/token estimate, or Metal."""
        if self.solo_transient_mb <= 0 or tokens_to_prefill <= 0:
            return None
        ceiling = self._pressure.ceiling_bytes()
        bpt = max(self.bytes_per_token(), float(self.bpt_floor_kb * 1024))
        if not ceiling or bpt <= 0:
            return None
        limit = ceiling * self.solo_ceiling_pct / 100
        need = tokens_to_prefill * bpt + self.solo_transient_mb * 1024 * 1024
        try:
            import mlx.core as mx

            if mx.get_active_memory() + need <= limit:
                return None
            relieved = False
            if self._ssd is not None and self._ssd.drop_backlog():
                self.pressure_backlog_drops += 1
                relieved = True
            mx.clear_cache()
            while mx.get_active_memory() + need > limit and (
                self._drop_lru_entry(spare_pinned=True) or self._drop_lru_entry()
            ):
                relieved = True
                mx.clear_cache()
            active = mx.get_active_memory()
        except Exception:
            logger.debug("[batched_system_kv] solo guard failed open", exc_info=True)
            return None
        if relieved:
            self.solo_relief_passes += 1
        if active + need <= limit:
            return None
        self.solo_rejections += 1
        logger.warning(
            "[batched_system_kv] solo prefill rejected: %d tokens need ≈ %.1f GB "
            "(%.1f KB/tok + %d MB chunk transient) on top of %.1f GB active "
            "> %d%% of %.1f GB working set, with the cache already emptied",
            tokens_to_prefill,
            need / 1e9,
            bpt / 1024,
            self.solo_transient_mb,
            active / 1e9,
            self.solo_ceiling_pct,
            ceiling / 1e9,
        )
        return "insufficient_memory"

    def projected_cobatch_verdict(
        self, need_bytes: float, describe: str
    ) -> Optional[str]:
        """(#106) Would admitting this row beside the running ones cross the
        OOM wall? None = admit; otherwise the defer reason.

        ``need_bytes`` is everything the admission adds on top of current
        active memory. Over the limit, pure cache is given back first - the
        SSD spill backlog, then bag entries LRU-first but only down to the
        RAM floor (a second seat is not worth an empty cache; relief will
        take the rest if real pressure arrives) - and the projection is
        re-checked. What still does not fit WAITS: rows finish and free their
        KV, so deferral always makes progress. Inert unless armed; shares
        #105's TRANSIENT_MB / CEILING_PCT."""
        if not self.projected_admission or self.solo_transient_mb <= 0:
            return None
        ceiling = self._pressure.ceiling_bytes()
        if not ceiling:
            return None
        limit = ceiling * self.solo_ceiling_pct / 100
        need = need_bytes + self.solo_transient_mb * 1024 * 1024
        try:
            import mlx.core as mx

            active = mx.get_active_memory()
            if active + need <= limit:
                return None
            relieved = False
            floor = self.ram_mb * 1024 * 1024
            with self._lock:
                bag = sum(e["bytes"] for e in self._entries.values())
                pinned = set(self._peeked.values())
                unpinned = sum(
                    e["bytes"]
                    for k, e in self._entries.items()
                    if k not in pinned and not e.get("spill_pending")
                )
            backlog = 0
            if self._ssd is not None:
                backlog = int(self._ssd.get_stats().get("queued_bytes", 0) or 0)
            # (#107) entries a queued request matched are not on offer: a
            # second seat now is not worth an 18-minute cold prefill later
            sheddable = max(0, min(unpinned, bag - floor)) + backlog
            if active + need - sheddable > limit:
                # Shedding cannot close the gap: keep the cache (it holds the
                # entry this very request will restore from) and just wait.
                self.projected_defers += 1
                return (
                    f"projected peak {(active + need) / 1e9:.1f} GB > "
                    f"{self.solo_ceiling_pct}% of {ceiling / 1e9:.1f} GB working "
                    f"set ({active / 1e9:.1f} GB active + {describe} + "
                    f"{self.solo_transient_mb} MB chunk transient; only "
                    f"{sheddable / 1e9:.1f} GB of cache could be shed)"
                )
            if backlog and self._ssd.drop_backlog():
                self.pressure_backlog_drops += 1
                relieved = True
            mx.clear_cache()
            while mx.get_active_memory() + need > limit:
                with self._lock:
                    bag = sum(e["bytes"] for e in self._entries.values())
                if bag <= floor or not self._drop_lru_entry(spare_pinned=True):
                    break
                relieved = True
                mx.clear_cache()
            active = mx.get_active_memory()
        except Exception:
            logger.debug(
                "[batched_system_kv] projected admission failed open", exc_info=True
            )
            return None
        if relieved:
            self.projected_relief_passes += 1
        if active + need <= limit:
            return None
        self.projected_defers += 1
        return (
            f"projected peak {(active + need) / 1e9:.1f} GB > "
            f"{self.solo_ceiling_pct}% of {ceiling / 1e9:.1f} GB working set "
            f"({active / 1e9:.1f} GB active + {describe} + "
            f"{self.solo_transient_mb} MB chunk transient)"
        )

    def _may_grow(self, request_id: str, tokens_list: list) -> bool:
        """Cheap pre-check of #37's grow path: is the request's donor entry
        still resident with a usable common prefix? (kinds are confirmed
        later by ``_build_snapshot``; a rare mismatch there just falls back
        to the full copy.)"""
        with self._lock:
            donor_key = self._restore_source.get(request_id)
            donor = (
                self._entries.get(donor_key) if donor_key is not None else None
            )
            if donor is None:
                return False
            return (
                common_prefix_len(tokens_list, donor["tokens"])
                >= self.partial_min
            )

    # ---------------------------------------------------------------- stats

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            mem = sum(e["bytes"] for e in self._entries.values())
            budget = self.effective_ram_bytes(mem) if self.ram_mb > 0 else 0.0
            return {
                "type": "batched_system_kv",  # bench-serve cache provenance
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
                "admission_deferrals": self.admission_deferrals,
                "pressure_evictions": self.pressure_evictions,
                "pressure_skipped_stores": self.pressure_skipped_stores,
                "pressure_cache_clears": self.pressure_cache_clears,
                "pressure_backlog_drops": self.pressure_backlog_drops,
                "solo_relief_passes": self.solo_relief_passes,
                "solo_rejections": self.solo_rejections,
                "lazy_restores": self.lazy_restores,
                "lazy_restore_misses": self.lazy_restore_misses,
                "lazy_ssd_fallbacks": self.lazy_ssd_fallbacks,
                "pinned_entries": len(set(self._peeked.values()) & set(self._entries)),
                "spill_pending_entries": sum(
                    1 for e in self._entries.values() if e.get("spill_pending")
                ),
                "projected_defers": self.projected_defers,
                "projected_relief_passes": self.projected_relief_passes,
                # (#103) memory held OUTSIDE the entries: in-flight checkpoint
                # ladders. Must read 0 whenever nothing is running or waiting
                # — a non-zero idle value is a leaked ladder.
                "pending_ladders": len(self._pending),
                "pending_ladder_mb": sum(
                    ckpt_bytes(ladder) for ladder in self._pending.values()
                )
                / (1024 * 1024),
                "timing_verdict": self._timing.verdict_totals(),
                "entry_count": len(self._entries),
                "capacity": self.slots,
                "memory_mb": mem / (1024 * 1024),
                "current_memory_mb": mem / (1024 * 1024),
                # exporter compatibility: mac-studio-exporter reads these
                # from /v1/status → cache; 0 budget = unlimited
                # (#101) the CURRENT effective budget — equals RAM_MB in
                # static mode; floats with free headroom in dynamic mode
                "max_memory_mb": budget / (1024 * 1024),
                "memory_utilization": (mem / budget) if budget > 0 else 0.0,
                "ram_budget_mode": "dynamic" if self.ram_dynamic else "static",
                "ram_floor_mb": float(self.ram_mb),
                "ram_max_mb": float(self.ram_max_mb),
                "ram_reserve_mb": float(self.ram_reserve_mb),
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
    the executor via promote_ssd_pending).

    With LAZY_RESTORE (#106) a hit is only MATCHED here; the copy is built by
    ``materialize_pending_restore`` when the request is admitted."""
    if getattr(hybrid_kv, "lazy_restore", False) is True:
        pos = hybrid_kv.peek(request.prompt_token_ids, request_id=request.request_id)
        if pos > 0:
            request.cache_hit_type = "system_kv_pending"
            request.prompt_cache = None
            request.cached_tokens = pos
            request.remaining_tokens = request.prompt_token_ids[pos:]
            return
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


def solo_prefill_verdict(scheduler, request) -> Optional[str]:
    """_schedule_waiting hook (#105): the one admission check for a request
    that finds nothing running. None = admit, else an ``error_kind``."""
    hybrid_kv = scheduler.hybrid_kv
    if hybrid_kv is None or scheduler.running:
        return None
    remaining = getattr(request, "remaining_tokens", None)
    tokens = (
        len(remaining)
        if remaining is not None
        else int(getattr(request, "num_prompt_tokens", 0) or 0)
    )
    return hybrid_kv.solo_prefill_verdict(tokens)


def materialize_pending_restore(scheduler, request) -> None:
    """_schedule_waiting hook (#106): build the restore a LAZY add_request
    only matched. Runs on the executor thread at admission, so the slices are
    recorded and evaluated on the thread that steps the batch.

    The entry may have been evicted while the request waited (it was
    LRU-touched at enqueue, so this is rare): ``fetch`` then returns a
    shallower hit or nothing, and the request simply prefills more."""
    if getattr(request, "cache_hit_type", None) != "system_kv_pending":
        return
    hybrid_kv = scheduler.hybrid_kv
    result = None
    if hybrid_kv is not None:
        try:
            # (#107) The entry matched at enqueue may be gone from RAM by now
            # (the request can wait minutes behind the KV budget). The eager
            # miss path probes SSD; the lazy one went straight to a cold
            # prefill - 18 minutes for a 60K chain whose entry was 0.5 s away
            # on disk (2026-09-21). Promote first, then fetch once.
            if getattr(hybrid_kv, "has_ssd", False) is True and hybrid_kv.peek(
                request.prompt_token_ids
            ) < (getattr(request, "cached_tokens", 0) or 0):
                candidate = hybrid_kv.check_ssd(request.prompt_token_ids)
                if candidate is not None and hybrid_kv.promote_ssd(candidate):
                    hybrid_kv.lazy_ssd_fallbacks += 1
            result = hybrid_kv.fetch(
                request.prompt_token_ids, request_id=request.request_id
            )
            hybrid_kv.release_peek(request.request_id)
        except Exception:
            logger.debug(
                "[batched_system_kv] lazy restore failed request=%s",
                request.request_id[:12],
                exc_info=True,
            )
    if result is not None:
        cache, remaining, pos = result
        request.cache_hit_type = "system_kv"
        request.prompt_cache = cache
        request.cached_tokens = pos
        request.remaining_tokens = remaining
        hybrid_kv.lazy_restores += 1
    else:
        request.cache_hit_type = "miss"
        request.prompt_cache = None
        request.cached_tokens = 0
        request.remaining_tokens = request.prompt_token_ids
        if hybrid_kv is not None:
            hybrid_kv.lazy_restore_misses += 1


def promote_ssd_pending(scheduler) -> None:
    """_schedule_waiting hook (#36): promote SSD candidates for waiting
    ssd_pending requests. Runs on the executor thread — the blob read +
    array realize happen here, never on the event loop."""
    hybrid_kv = scheduler.hybrid_kv
    # (#103) Snapshot: add_request appends to ``waiting`` on the event loop
    # while this runs on the executor — iterating the live deque raised
    # "deque mutated during iteration" (15x on 2026-09-16), and that raise
    # took the whole batch down through generation_error_recovery.
    for request in list(scheduler.waiting):
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


def find_message_boundaries(tokens, marker_seqs, min_step) -> tuple:
    """Positions in ``tokens`` where a template turn marker STARTS (#88) —
    checkpoint placement at message boundaries, on the token stream.

    ``min_step`` thins boundaries closer than that to the previous accepted
    cut (short-turn bursts must not shred the ladder); the LAST boundary is
    always kept regardless (llama.cpp #24176's rule — the newest turn start
    is the likeliest divergence point of the next request)."""
    if not marker_seqs:
        return ()
    n = len(tokens)
    hits = set()
    for seq in marker_seqs:
        m = len(seq)
        if m == 0 or m > n:
            continue
        first = seq[0]
        i = 0
        while True:
            try:
                i = tokens.index(first, i)
            except ValueError:
                break
            if tuple(tokens[i : i + m]) == seq:
                hits.add(i)
            i += 1
    ordered = sorted(h for h in hits if 0 < h < n)
    if not ordered:
        return ()
    accepted = []
    last_cut = 0
    for h in ordered:
        if h - last_cut >= min_step:
            accepted.append(h)
            last_cut = h
    if not accepted or accepted[-1] != ordered[-1]:
        accepted.append(ordered[-1])  # newest boundary bypasses min_step
    return tuple(accepted)


def insert_segmented(
    hybrid_kv: "BatchedSystemKV",
    batch_generator,
    request,
    tokens,
    insert_kwargs,
    tokenizer=None,
):
    """_schedule_waiting insert hook (#34): split the prompt at checkpoint
    boundaries so the generator stops there (insert_segments) and
    capture_checkpoints can snapshot recurrent state. #88 adds message-
    boundary cuts (detected on the token stream from the request's
    template family) so checkpoints land where agent histories actually
    diverge; failure to detect degrades to the uniform interval."""
    boundaries = ()
    try:
        marker_seqs = hybrid_kv.boundary_marker_ids(
            getattr(request, "prompt", None), tokenizer
        )
        boundaries = find_message_boundaries(
            tokens, marker_seqs, hybrid_kv.boundary_min_step
        )
    except Exception:
        logger.debug("[batched_system_kv] boundary detection failed", exc_info=True)
    hybrid_kv.note_scheduled(
        request.request_id, request.cached_tokens, boundaries=boundaries
    )
    return batch_generator.insert_segments(
        [hybrid_kv.split_segments(tokens, boundaries=boundaries)],
        **insert_kwargs,
    )


def maybe_relieve_pressure(scheduler) -> None:
    """Step-loop hook (fork patch #48): shed snapshot RAM while the device
    is over the memory watermark.

    #40's watermark only gates NEW co-batch admissions — the 2026-07-08
    canary crashes were a SOLO deep-context prefill ramping ~20 GB/min
    from a sub-watermark baseline straight past the wired limit, with the
    snapshot bag holding ~5.5 GB of copies the whole way up. This runs
    once per scheduler step, so it fires MID-prefill (each step processes
    one prompt chunk) and returns that headroom before the ramp peaks.
    Same env gate as the admission watermark; inert when unset, and a
    single sub-watermark memory read per step otherwise."""
    hybrid_kv = scheduler.hybrid_kv
    if hybrid_kv is None:
        return
    try:
        hybrid_kv.relieve_pressure()
    except Exception:
        logger.debug(
            "[batched_system_kv] pressure relief failed", exc_info=True
        )


def _measured_bytes_per_token(scheduler) -> float:
    """Ground-truth per-token KV cost read from the live BatchGenerator.

    ``BatchGenerator.prompt_cache_nbytes`` sums the actually-allocated
    cache arrays (right-justified padding, quantized layers, hybrid SSM
    state — everything the hardware is really paying for) across its
    unprocessed/prompt/generation stages; dividing by the running set's
    tracked token count yields bytes/token as measured, not as estimated
    from the newest cache snapshot (#40's proxy, which is stale after a
    model of different shape warmed the bag and ABSENT on a cold cache —
    leaving the budget gates inert exactly when a fresh process serves
    its first deep request). Two deliberate biases, both conservative
    (over-price a token → defer sooner):

    - the denominator uses per-row logical lengths, so when rows are
      padded to the batch max the ratio charges the padding to the
      logical tokens;
    - scheduler bookkeeping can lag the generator by a step.

    Returns 0.0 whenever the generator or the measurement is unavailable
    (unit-test schedulers, generator mid-recreate) — callers fall back to
    the learned-at-insert estimate.
    """
    gen = getattr(scheduler, "batch_generator", None)
    if gen is None:
        return 0.0
    try:
        measured = float(gen.prompt_cache_nbytes)
    except Exception:
        return 0.0
    if measured <= 0:
        return 0.0
    total_tokens = sum(
        r.num_prompt_tokens + len(r.output_token_ids)
        for r in scheduler.running.values()
    )
    if total_tokens <= 0:
        return 0.0
    return measured / total_tokens


def should_defer_cobatch(scheduler, request) -> bool:
    """Memory-aware admission gate (#39 pad-waste + #40 dynamic concurrency).

    Three independent, env-gated checks — any one defers admission (FCFS
    queueing instead of a memory spike). Solo requests are never deferred,
    so progress is always guaranteed. All inert by default.

    1. **Pad waste** (``VLLM_MLX_BATCHED_PAD_WASTE_MB``): mlx-lm's
       BatchKVCache.merge right-justifies every row to the longest chain,
       so mixing very different context lengths allocates padding ≈
       Σ(L_max − L_i) × bytes/token — a short request joining an 80K-token
       chain transiently costs ~5 GB at 27B scale.
    2. **Total padded-KV budget** (``VLLM_MLX_BATCHED_KV_BUDGET_MB``):
       (B+1) × L_max × bytes/token vs the budget — the dynamic
       ``max-num-seqs``: seats float on the live request mix (deep
       contexts serialize themselves, short ones batch up to the hard
       ``--max-num-seqs`` cap).
    3. **Memory watermark** (``VLLM_MLX_BATCHED_MEM_WATERMARK_PCT``):
       ground-truth backstop — defer while ``mx.get_active_memory()``
       exceeds this percentage of the device's recommended working set
       (catches pressure the estimates can't see: spill queue, cache
       entries, other processes' share of unified memory).

    Checks 1–2 price tokens with ground-truth bytes/token measured from
    the live BatchGenerator (``prompt_cache_nbytes`` over the running
    set's tokens — see ``_measured_bytes_per_token``), falling back to
    the footprint learned from the newest cache entry when no
    measurement is available; with neither (genuinely cold process) they
    are inert — the first request runs solo anyway. Runs on the executor
    thread from ``_schedule_waiting``.
    """
    hybrid_kv = scheduler.hybrid_kv
    if not scheduler.running:
        return False

    reason = None
    # (#102) Price with the LARGEST of the three estimates. "measured" is
    # allocated KV over the running set's LOGICAL tokens, so right after the
    # generator is recreated (generation_error_recovery, cold spawn) rows that
    # are admitted but not yet prefilled count their full prompt with ~no bytes
    # — the ratio collapses and the whole waiting queue passes in one pass.
    # Live 2026-09-16: 45K + 66K + 46K-token rows admitted within 2 ms after a
    # recovery -> Metal OOM again (peaks 63.2-63.9 GB), a recovery loop.
    # "learned" (#100: exact KV-only bytes/token from the newest entry,
    # persisting across an emptied bag) and the env floor cannot collapse.
    candidates = [
        ("measured", _measured_bytes_per_token(scheduler)),
        ("learned", hybrid_kv.bytes_per_token()),
        ("floor", float(hybrid_kv.bpt_floor_kb) * 1024),
    ]
    bpt_source, bpt = max(candidates, key=lambda c: c[1])
    if bpt > 0 and (hybrid_kv.pad_waste_mb > 0 or hybrid_kv.kv_budget_mb > 0):
        lengths = [
            r.num_prompt_tokens + len(r.output_token_ids)
            for r in scheduler.running.values()
        ]
        lengths.append(request.num_prompt_tokens)
        l_max = max(lengths)
        if hybrid_kv.pad_waste_mb > 0:
            waste_mb = sum((l_max - li) for li in lengths) * bpt / (1024 * 1024)
            if waste_mb > hybrid_kv.pad_waste_mb:
                reason = (
                    f"padded-KV waste ≈ {waste_mb:.0f} MB > "
                    f"{hybrid_kv.pad_waste_mb} MB; lengths {sorted(lengths)}"
                )
        if reason is None and hybrid_kv.kv_budget_mb > 0:
            total_mb = len(lengths) * l_max * bpt / (1024 * 1024)
            if total_mb > hybrid_kv.kv_budget_mb:
                reason = (
                    f"padded-KV total ≈ {total_mb:.0f} MB > "
                    f"{hybrid_kv.kv_budget_mb} MB at {len(lengths)} seats; "
                    f"lengths {sorted(lengths)}"
                )
        if reason is not None:
            reason += f" [bpt {bpt / 1024:.1f} KB/tok, {bpt_source}]"

    # 4. (#106) Projected peak vs the OOM wall. Gates 1-2 price padded KV
    # against a budget and gate 3 looks at the present; none asks what THIS
    # admission will add to what the process already holds.
    if (
        reason is None
        and bpt > 0
        and getattr(hybrid_kv, "projected_admission", False) is True
    ):
        lengths = [
            r.num_prompt_tokens + len(r.output_token_ids)
            for r in scheduler.running.values()
        ]
        lengths.append(request.num_prompt_tokens)
        merged = len(lengths) * max(lengths) * bpt
        remaining = getattr(request, "remaining_tokens", None)
        growth = (
            len(remaining) if remaining is not None else request.num_prompt_tokens
        ) * bpt
        restore = (
            hybrid_kv.restore_bytes(getattr(request, "cached_tokens", 0) or 0)
            if getattr(request, "cache_hit_type", None) == "system_kv_pending"
            else 0.0
        )
        reason = hybrid_kv.projected_cobatch_verdict(
            merged + growth + restore,
            f"{merged / 1e9:.1f} GB merged KV at {len(lengths)} seats + "
            f"{growth / 1e9:.1f} GB new-row growth + {restore / 1e9:.1f} GB restore",
        )

    if reason is None and hybrid_kv.mem_watermark_pct > 0:
        over, active, ceiling = hybrid_kv.watermark_status()
        if over:
            reason = (
                f"active memory {active / 1e9:.1f} GB > "
                f"{hybrid_kv.mem_watermark_pct}% of "
                f"{ceiling / 1e9:.1f} GB working set"
            )

    if reason is None:
        return False
    hybrid_kv.admission_deferrals += 1
    if not getattr(request, "_pad_defer_logged", False):
        request._pad_defer_logged = True
        logger.info(
            "[batched_system_kv] deferring co-batch of request=%s (%s)",
            request.request_id[:12],
            reason,
        )
    return True
