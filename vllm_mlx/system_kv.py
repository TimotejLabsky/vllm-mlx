# SPDX-License-Identifier: Apache-2.0
"""System-prompt KV snapshot cache for SimpleEngine.

This module carries the fork's system-KV snapshot stack — patches #4/#6/#9/
#12/#13 (see PATCHES.md), plus the patch #16 SSD-persistence integration —
extracted out of ``vllm_mlx/engine/simple.py`` to minimize that file's diff
vs upstream (upstream churns ``simple.py`` constantly; this state machine is
pure fork code).

``SystemKVManager`` owns:

- The "active" slot (snapshot / hash / token_count / token_ids — the legacy
  single-slot interface) plus the multi-slot LRU side-stash (patch #13).
- Hit/miss/tokens_saved/eviction counters (patch #7 metrics).
- The ``VLLM_MLX_DISABLE_SYSTEM_KV`` kill switch (patch #5).
- The start()-time RotatingKVCache denylist probe (patch #12).
- The optional ``SystemKVSSDStore`` lifecycle (patch #16).

Threading contract: all LRU/active-slot mutations run inside SimpleEngine's
``_generation_lock`` (either on the event loop while the lock is held, or in
the serialized generation worker thread). ``SimpleEngine`` re-exposes the
legacy ``_system_kv_*`` attribute names via delegating properties so tests
and downstream code keep working unchanged.
"""

import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)


class SystemKVManager:
    """State + helpers for the system-prompt KV snapshot cache."""

    def __init__(self):
        # System prompt KV cache (reduces repeated prefill across requests).
        #
        # The "active" slot is held in these four ivars (legacy single-slot
        # interface). On lookup, ``lru_promote(system_hash)`` swaps in the
        # matching slot from ``lru`` if one exists, so downstream
        # code (which still reads the legacy ivars) sees the right snapshot.
        # On store, ``lru_demote_active_to_bag()`` pushes the about-to-be-
        # replaced active slot into the LRU before the new one is written.
        # Default capacity (active + LRU) is 4, tunable via
        # VLLM_MLX_SYSTEM_KV_SLOTS=N. =1 restores the single-slot behavior
        # from before this patch.
        self.snapshot = None  # List of (keys, values) per backbone layer
        self.system_hash = None  # Hash of system prefix text
        self.token_count = 0
        self.token_ids = None  # EXTENDED_PREFIX_MARKER: cached prefix tokens
        # Side-stash LRU for inactive slots, keyed by system_hash.
        # value: {"snapshot": list, "token_count": int, "token_ids": list}
        self.lru: "OrderedDict[str, dict]" = OrderedDict()
        # Capacity is the TOTAL slot count (active + LRU bag).
        # The LRU bag therefore holds capacity-1 entries at most.
        import os as _os
        self.capacity = max(1, int(
            _os.environ.get("VLLM_MLX_SYSTEM_KV_SLOTS", "4")
        ))
        # Prometheus-side counters — see vllm-mlx-system-kv-metrics.py
        self.hits = 0
        self.misses = 0
        self.tokens_saved = 0
        self.evictions = 0
        # True only when the model's prompt cache is composed entirely of
        # plain ``KVCache`` entries. Sliding-window models (gemma3_text,
        # olmo3, recurrent_gemma) return ``RotatingKVCache`` whose ``.state``
        # aliases buffers ``update_and_fetch`` mutates in place — snapshot
        # restore would silently desynchronize. Probed once in ``start()``.
        self.supports_snapshot: bool = False
        # SSD persistence for the system-KV snapshot (patch #16). Off unless
        # VLLM_MLX_SSD_SYSTEM_KV_DIR is set; constructed in start() once the
        # snapshot-safety probe has run (shares its gate). Lets a stored
        # system prefix survive process restart / TTL eviction / model swap:
        # promote (~100 ms-1.5 s disk read) instead of a 25-70 s cold prefill.
        self.ssd_store = None  # SystemKVSSDStore | None
        self.ssd_promotes = 0
        # Cached-once result of the VLLM_MLX_DISABLE_SYSTEM_KV kill switch.
        self._safe_cached = None

    # ------------------------------------------------------------------
    # Kill switch (patch #5)
    # ------------------------------------------------------------------

    def is_safe(self):
        """Return True if the system-KV snapshot is enabled for this engine.

        Reads ``VLLM_MLX_DISABLE_SYSTEM_KV`` env var once and caches the
        result. Set the var to ``1``/``true``/``yes`` (typically per-model
        in the llama-swap launch config) to disable the snapshot path on
        models known to produce drifted/looping output on cache replay
        (hybrid Qwen3.5/3.6/Qwen3-Next family — see mlx-lm#1162).
        """
        cached = self._safe_cached
        if cached is not None:
            return cached

        import os

        disabled = os.environ.get("VLLM_MLX_DISABLE_SYSTEM_KV", "").lower() in (
            "1",
            "true",
            "yes",
        )
        if disabled:
            logger.info(
                "system-KV snapshot disabled by VLLM_MLX_DISABLE_SYSTEM_KV env var"
            )
        self._safe_cached = not disabled
        return self._safe_cached

    # ------------------------------------------------------------------
    # start()/stop() lifecycle
    # ------------------------------------------------------------------

    def probe_snapshot_support(self, model_wrapper) -> None:
        """Probe whether the wrapped model's prompt cache is snapshot-safe.

        ``model_wrapper`` is the engine's loaded model object (e.g.
        ``MLXLanguageModel``); its inner ``.model`` is dereferenced INSIDE
        the try below so wrappers without one (test doubles, exotic
        loaders) fail closed exactly like a probe failure.

        Probe whether this model's prompt cache is snapshot-safe for the
        stream_chat system-prefix cache branch. We use a denylist rather
        than an allowlist:

        - KVCache:        safe (tuple state, immutable on capture).
        - ArraysCache:    safe AFTER patch #6 (system-kv-hybrid-aliasing)
                          shallow-copies the list at capture and restore.
                          Used by hybrid attention layers (Gated DeltaNet
                          in Qwen3.6, etc.) interleaved with KVCache.
        - RotatingKVCache: UNSAFE. Sliding-window models (gemma3_text,
                           olmo3, recurrent_gemma) expose ``.state`` as
                           an alias of in-place-mutated ring buffers,
                           which the snapshot mechanism cannot capture
                           without restore drift. Disable for these.

        Only relevant for the LLM path; MLLM gates via is_safe().
        """
        try:
            from mlx_lm.models.cache import (
                RotatingKVCache,
                make_prompt_cache,
            )

            probe_cache = make_prompt_cache(model_wrapper.model)
            has_rotating = any(
                isinstance(c, RotatingKVCache) for c in probe_cache
            )
            self.supports_snapshot = bool(probe_cache) and not has_rotating
            if not self.supports_snapshot:
                cache_types = sorted({type(c).__name__ for c in probe_cache})
                logger.info(
                    "System KV cache snapshot disabled: model returned "
                    "RotatingKVCache entries (%s); stream_chat will use "
                    "the uncached path. Set "
                    "VLLM_MLX_DISABLE_SYSTEM_KV=1 to also bypass on "
                    "models that pass the probe.",
                    cache_types,
                )
            else:
                cache_types = sorted({type(c).__name__ for c in probe_cache})
                logger.info(
                    "System KV cache snapshot enabled (probe cache "
                    "types: %s)",
                    cache_types,
                )
        except Exception as e:
            logger.debug(
                "System KV cache support probe failed (%s); "
                "disabling snapshot path",
                e,
            )
            self.supports_snapshot = False

    def maybe_start_ssd_store(self, model_name: str) -> None:
        """Patch #16: SSD persistence for the system-KV snapshot.

        Opt-in via VLLM_MLX_SSD_SYSTEM_KV_DIR. Gated on is_safe() — the
        SAME gate the _stream_generate_text spill/promote sites use — so
        the store is built whenever that path can cache (covers both
        non-MLLM models and MLLM models whose text route uses the LLM
        path, e.g. Qwen3.6-27B which loads MLLM=True). Not gated on the
        ``supports_snapshot`` probe: that only runs for non-MLLM and
        only gates the separate stream_chat (MLLM+media) cache branch.
        """
        import os as _os

        ssd_base = _os.environ.get("VLLM_MLX_SSD_SYSTEM_KV_DIR")
        if ssd_base and self.is_safe():
            try:
                import re as _re

                from .system_kv_ssd import (
                    SystemKVSSDConfig,
                    SystemKVSSDStore,
                )

                safe = _re.sub(r"[^A-Za-z0-9._-]", "_", model_name)
                max_gb = float(
                    _os.environ.get("VLLM_MLX_SSD_SYSTEM_KV_GB", "50")
                )
                self.ssd_store = SystemKVSSDStore(
                    SystemKVSSDConfig(
                        cache_dir=_os.path.join(ssd_base, safe),
                        max_size_gb=max_gb,
                    )
                )
                self.ssd_store.start_writer()
                logger.info(
                    "System KV SSD persistence ENABLED: %s (cap %.0f GB)",
                    _os.path.join(ssd_base, safe),
                    max_gb,
                )
            except Exception as e:
                logger.warning(
                    "System KV SSD persistence init failed (%s); disabled",
                    e,
                )
                self.ssd_store = None

    def reset(self) -> None:
        """Release the full snapshot stack (SimpleEngine.stop())."""
        self.snapshot = None
        self.system_hash = None
        self.token_count = 0
        self.token_ids = None  # EXTENDED_PREFIX_MARKER: cached prefix tokens
        self.lru.clear()
        self.evictions = 0
        self.hits = 0
        self.misses = 0
        self.tokens_saved = 0
        self.supports_snapshot = False
        if self.ssd_store is not None:
            try:
                self.ssd_store.close()
            except Exception:
                logger.debug("System KV SSD store close failed", exc_info=True)
            self.ssd_store = None

    # ------------------------------------------------------------------
    # Multi-slot LRU (patch #13)
    # ------------------------------------------------------------------

    def lru_promote(self, system_hash):
        """Try to promote a slot matching ``system_hash`` from the LRU bag
        into the "active" position (the legacy single-slot ivars).

        - If the active slot already matches: no-op, return True.
        - If a matching slot is in the LRU bag: swap it with active.
          The displaced active slot (if any) goes into the bag.
        - If nothing matches: return False.

        Must run inside ``_generation_lock`` so the swap is atomic with
        downstream reads. Multi-slot capacity is bounded by
        ``capacity`` (active + bag).
        """
        if system_hash is None:
            return False
        if self.system_hash == system_hash and self.snapshot is not None:
            return True
        entry = self.lru.pop(system_hash, None)
        if entry is None:
            return False
        # Demote current active to bag (if any) before promoting the match.
        if self.snapshot is not None and self.system_hash is not None:
            self.lru[self.system_hash] = {
                "snapshot": self.snapshot,
                "token_count": self.token_count,
                "token_ids": self.token_ids,
            }
            self.lru.move_to_end(self.system_hash)
        self.snapshot = entry["snapshot"]
        self.system_hash = system_hash
        self.token_count = entry["token_count"]
        self.token_ids = entry["token_ids"]
        return True

    def lru_demote_active_to_bag(self):
        """Move the current active slot (if any) into the LRU bag, then
        clear active. Called before overwriting active with a new MISS.

        Evicts oldest entry from the bag if (bag + 1) would exceed
        ``capacity - 1`` (leaving room for the incoming new
        active slot). Issues ``mx.clear_cache()`` only on the eviction
        path so the Metal allocator's reuse pool isn't flushed on the
        common case (PR #541 measurement).

        Must run inside ``_generation_lock``.
        """
        if self.snapshot is None or self.system_hash is None:
            return
        self.lru[self.system_hash] = {
            "snapshot": self.snapshot,
            "token_count": self.token_count,
            "token_ids": self.token_ids,
        }
        self.lru.move_to_end(self.system_hash)
        # Clear active so the caller's assignment is clean.
        self.snapshot = None
        self.system_hash = None
        self.token_count = 0
        self.token_ids = None
        # Trim bag to capacity-1 (leaving one slot for the incoming active).
        evicted = 0
        max_bag = max(0, self.capacity - 1)
        while len(self.lru) > max_bag:
            ev_hash, _ = self.lru.popitem(last=False)
            self.evictions += 1
            evicted += 1
            logger.info(
                "System KV cache EVICTED: hash=%s (capacity=%d, bag=%d)",
                ev_hash,
                self.capacity,
                len(self.lru),
            )
        if evicted:
            try:
                import mlx.core as mx
                mx.clear_cache()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Lookup / store bookkeeping (patches #4/#9/#13)
    # ------------------------------------------------------------------

    def lookup_active(self, system_hash, token_count):
        """LRU-promote ``system_hash`` and return the active snapshot on an
        exact (hash, token_count) match, else None (stream_chat lookup).

        The returned reference is the gate-time snapshot capture: callers
        pass it explicitly into their serialized worker closure so a later
        concurrent MISS that reassigns the active slot cannot alias what
        was validated here. Must run inside ``_generation_lock``.
        """
        # LRU promote: bring the matching slot (if any) into
        # the active position so the legacy hash equality
        # check below sees the right snapshot.
        self.lru_promote(system_hash)
        # Read the snapshot reference once.
        candidate_snapshot = self.snapshot
        if (
            system_hash == self.system_hash
            and candidate_snapshot is not None
            and token_count == self.token_count
        ):
            return candidate_snapshot
        return None

    def match_extended_prefix(self, full_tokens_list):
        """EXTENDED_PREFIX_MARKER: prefix-match against the cached extended
        token sequence. Hit when the new request's ``full_tokens_list``
        starts with our cached ``token_ids``.

        Returns ``(cached_token_ids, cached_len)`` on match, else
        ``(None, 0)``. Must run inside ``_generation_lock``.
        """
        _cached_ids = self.token_ids
        _cached_len = self.token_count
        _prefix_match = (
            _cached_ids is not None
            and _cached_len > 0
            and len(full_tokens_list) > _cached_len
            and full_tokens_list[:_cached_len] == _cached_ids
        )
        if _prefix_match:
            return _cached_ids, _cached_len
        return None, 0

    def record_hit(self, tokens_saved):
        """Bump the HIT counters (Prometheus gauges read these)."""
        self.hits += 1
        self.tokens_saved += tokens_saved

    def store_snapshot(self, system_hash, snapshot, token_count):
        """Install ``snapshot`` as the active slot (stream_chat MISS store).

        LRU: demote the about-to-be-replaced active slot
        to the bag before overwriting (only if it holds a
        different system_hash from the new MISS).

        Does not touch ``token_ids`` — the stream_chat branch caches the
        literal system prefix, not the extended prefix. Runs inside the
        serialized generation worker.
        """
        if self.system_hash and self.system_hash != system_hash:
            self.lru_demote_active_to_bag()
        self.snapshot = snapshot
        self.system_hash = system_hash
        self.token_count = token_count

    def store_extended(self, system_hash, snapshot, extended_token_ids, promoted):
        """Install an EXTENDED-prefix snapshot as the active slot
        (text-route MISS store), gated on the kill switch: when disabled
        the request still prefilled its own prompt cache, but NOTHING
        persists. Runs inside the serialized generation worker.
        """
        if self.is_safe():
            # LRU: push the current active slot (if any, for a
            # different system_hash) into the bag before we
            # overwrite active. Evicts oldest bag entry if needed.
            if self.system_hash and self.system_hash != system_hash:
                self.lru_demote_active_to_bag()
            self.snapshot = snapshot
            self.system_hash = system_hash
            self.token_count = len(extended_token_ids)
            # EXTENDED_PREFIX_MARKER: store cached token IDs for prefix matching
            self.token_ids = list(extended_token_ids)
            # Patch #16: write-through to SSD when freshly computed (not
            # promoted from disk). One async write per distinct system
            # prefix at creation; the grow path does NOT re-spill — a
            # restart promotes the stored prefix and re-grows cheaply.
            if self.ssd_store is not None and not promoted:
                self.ssd_store.enqueue_spill(
                    tuple(extended_token_ids), snapshot
                )

    # ------------------------------------------------------------------
    # Stats (patches #7/#10/#13/#16 + perf-observability)
    # ------------------------------------------------------------------

    def stats(self):
        """Assemble the ``system_kv_cache`` stats dict, or None when idle.

        System KV cache stats — patched to emit Prometheus-compatible
        fields (hits/misses/tokens_saved/entry_count) so metrics.py
        maps them onto the vllm_mlx_cache_* gauges. Broadened gate so
        misses are reported even before a snapshot is stored.

        Multi-slot LRU note: active slot + bag entries are summed for
        the aggregate fields (memory_mb, current_memory_mb,
        entry_count). Legacy single-slot fields (tokens, hash) still
        describe the ACTIVE slot for backward compat with the
        Prometheus exporter and existing dashboards.
        """

        def _entry_bytes(snap):
            n = 0
            if not snap:
                return 0
            for layer in snap:
                if isinstance(layer, tuple) and len(layer) == 2:
                    n += layer[0].nbytes + layer[1].nbytes
                elif isinstance(layer, list):
                    n += sum(a.nbytes for a in layer if a is not None)
            return n

        if not (
            self.snapshot is not None
            or self.lru
            or self.hits
            or self.misses
        ):
            return None

        active_bytes = _entry_bytes(self.snapshot)
        bag_bytes = sum(_entry_bytes(e["snapshot"]) for e in self.lru.values())
        total_bytes = active_bytes + bag_bytes
        slot_count = (1 if self.snapshot is not None else 0) + len(self.lru)
        slots_view = []
        if self.snapshot is not None:
            slots_view.append({
                "hash": self.system_hash,
                "tokens": self.token_count,
                "memory_mb": round(active_bytes / 1e6, 1),
                "active": True,
            })
        for slot_hash, entry in self.lru.items():
            slots_view.append({
                "hash": slot_hash,
                "tokens": entry["token_count"],
                "memory_mb": round(_entry_bytes(entry["snapshot"]) / 1e6, 1),
                "active": False,
            })
        result = {
            # Legacy single-slot fields (describe ACTIVE slot):
            "tokens": self.token_count,
            "hash": self.system_hash,
            # Aggregate over active + bag (Prometheus exporter reads these):
            "memory_mb": round(total_bytes / 1e6, 1),
            "current_memory_mb": round(total_bytes / 1e6, 1),
            "entry_count": slot_count,
            "hits": self.hits,
            "misses": self.misses,
            # hit_rate is read by metrics.py (vllm_mlx_cache_hit_rate gauge);
            # the legacy hit_rate key only existed on the MLLM
            # memory_aware_cache block, so the production non-MLLM path
            # reported a constant 0% even at the real ~89% hit rate.
            "hit_rate": (
                self.hits
                / (self.hits + self.misses)
                if (self.hits + self.misses)
                else 0.0
            ),
            "tokens_saved": self.tokens_saved,
            # New multi-slot fields:
            "evictions": self.evictions,
            "capacity": self.capacity,
            "slots": slots_view,
        }
        # Patch #16: SSD persistence tier (present only when enabled).
        if self.ssd_store is not None:
            try:
                ssd_stats = self.ssd_store.get_stats()
                ssd_stats["promotes"] = self.ssd_promotes
                result["ssd"] = ssd_stats
            except Exception:
                pass
        return result
