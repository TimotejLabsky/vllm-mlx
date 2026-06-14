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


# Template-family turn markers for the extended-prefix cache. The cache
# needs two anchors in the RENDERED prompt: where the system prefix ends
# (first user/assistant turn marker) and where the final generation prompt
# begins (rfind of the gen marker — everything before it is stable across
# turns and cacheable). These were hard-coded ChatML strings, which silently
# disabled caching for every non-ChatML family (Gemma 4, GLM, Mistral,
# Llama-3, Phi-4, Harmony) — warm TTFT == cold TTFT, measured 2026-06-10.
# Markers below were verified against each model's actual rendered template
# (apply_chat_template with a multi-turn fixture), not guessed.
#
# Order matters only for collision safety: ChatML first (most common),
# Mistral's bare "[INST]" last (likeliest to appear inside user content).
TEMPLATE_MARKERS = (
    # (family, boundary markers (first find() ends the system prefix),
    #  generation-prompt marker (rfind() = extended-cache boundary))
    ("chatml", ("<|im_start|>user\n", "<|im_start|>assistant\n"),
     "<|im_start|>assistant\n"),
    ("phi4", ("<|im_start|>user<|im_sep|>",),
     "<|im_start|>assistant<|im_sep|>"),
    ("gemma4", ("<|turn>user\n",), "<|turn>model\n"),
    ("llama3", ("<|start_header_id|>user<|end_header_id|>",),
     "<|start_header_id|>assistant<|end_header_id|>"),
    ("glm4", ("<|user|>",), "<|assistant|>"),
    ("harmony", ("<|start|>user<|message|>",), "<|start|>assistant"),
    ("mistral", ("[INST]",), "[/INST]"),
)


def detect_template_markers(full_prompt):
    """Detect the chat-template family of a rendered prompt.

    Returns ``(family, system_prefix_end, gen_marker)`` where
    ``system_prefix_end`` is the index of the first user-turn marker
    (> 0 on success), or ``(None, -1, None)`` when no family matches —
    callers fall back to the uncached path, same as before this table
    existed.
    """
    for family, boundary_markers, gen_marker in TEMPLATE_MARKERS:
        for marker in boundary_markers:
            idx = full_prompt.find(marker)
            if idx > 0:
                return family, idx, gen_marker
    return None, -1, None


def common_prefix_len(a, b):
    """Length of the longest common prefix of two token-id sequences."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def entry_bytes(snap):
    """In-RAM bytes of one snapshot (sum of per-layer array ``nbytes``).

    ``KVCache`` layers are 2-tuples ``(keys, values)``; recurrent / rotating
    layers are lists of arrays. Cheap (attribute reads, no ``mx.eval``).
    Hoisted to module scope (was nested in ``stats()``) so the RAM-budget
    enforcement path can account each slot at store time, not only on the
    stats poll.
    """
    n = 0
    if not snap:
        return 0
    for layer in snap:
        if isinstance(layer, tuple) and len(layer) == 2:
            n += layer[0].nbytes + layer[1].nbytes
        elif isinstance(layer, list):
            n += sum(a.nbytes for a in layer if a is not None)
    return n


def ckpt_bytes(ckpts):
    """In-RAM bytes of a slot's partial-restore checkpoints."""
    n = 0
    for cp in ckpts or []:
        for st in cp["states"].values():
            n += sum(a.nbytes for a in st if a is not None)
    return n


def classify_layers(prompt_cache):
    """Per-layer snapshot semantics, from the live cache CLASSES (state
    shape alone cannot distinguish a RotatingKVCache from a KVCache — both
    return a (keys, values) tuple, but slicing a rotated ring buffer is
    wrong).

    - ``trim``:   plain KVCache — state at any position p is recoverable
                  from the final snapshot by slicing ``keys[..., :p, :]``.
    - ``ckpt``:   position-bound state — ArraysCache (recurrent) and
                  RotatingKVCache (sliding-window ring + meta indices).
                  Restorable only at positions where a checkpoint captured
                  it.
    - ``opaque``: everything else (e.g. QuantizedKVCache) — whole-snapshot
                  restore round-trips via state/meta_state, but no partial
                  restore.
    """
    kinds = []
    for c in prompt_cache:
        st = c.state
        if isinstance(st, list) or type(c).__name__ == "RotatingKVCache":
            kinds.append("ckpt")
        elif (
            isinstance(st, tuple)
            and len(st) == 2
            and all(hasattr(a, "ndim") for a in st)
        ):
            kinds.append("trim")
        else:
            kinds.append("opaque")
    return kinds


def capture_snapshot_meta(prompt_cache):
    """Per-layer ``meta_state`` (None when trivial). RotatingKVCache carries
    its ring indices here — restoring its state WITHOUT meta desynchronizes
    the window (the original reason patch #12 denylisted it). mlx-lm's own
    ``save_prompt_cache``/``load_prompt_cache`` round-trip exactly this pair.
    """
    metas = []
    for c in prompt_cache:
        m = getattr(c, "meta_state", "")
        metas.append(m if m else None)
    return metas


def apply_snapshot_states(prompt_cache, states, metas=None):
    """Install per-layer states (+ meta_state where present) into a fresh
    prompt cache. Lists are shallow-copied (patch #6 aliasing discipline);
    meta is applied AFTER state, mirroring mlx-lm's ``from_state`` order.
    """
    for i, st in enumerate(states):
        prompt_cache[i].state = list(st) if isinstance(st, list) else st
        m = metas[i] if metas and i < len(metas) else None
        if m:
            prompt_cache[i].meta_state = m


def capture_checkpoint_states(prompt_cache, kinds=None):
    """Shallow-copy the checkpoint-class layers of a live prompt cache at
    the current position. Returns ``(states, metas)`` keyed by layer index
    — ``({}, {})`` for models with no checkpoint-class layers.

    Same aliasing discipline as patch #6: list states are rebound (never
    mutated in place) by subsequent forwards, so a shallow copy pins them;
    tuple states are immutable mx.arrays. Rotating layers additionally pin
    ``meta_state`` (ring indices at this position).
    """
    if kinds is None:
        kinds = classify_layers(prompt_cache)
    states = {}
    metas = {}
    for i, c in enumerate(prompt_cache):
        if kinds[i] != "ckpt":
            continue
        st = c.state
        states[i] = list(st) if isinstance(st, list) else st
        m = getattr(c, "meta_state", "")
        metas[i] = m if m else None
    return states, metas


def append_checkpoint(checkpoints, pos, states, metas, capacity):
    """Append a {pos, states, metas} checkpoint, keeping the list sorted
    and bounded. When the cap is exceeded, drop every other checkpoint
    (geometric thinning — early positions stay covered at coarser
    granularity, which is what divergence-point restore needs).
    """
    if checkpoints and checkpoints[-1]["pos"] >= pos:
        return checkpoints
    checkpoints.append({"pos": pos, "states": states, "metas": metas})
    if len(checkpoints) > max(1, capacity):
        thinned = checkpoints[::2]
        if thinned[-1] is not checkpoints[-1]:
            thinned.append(checkpoints[-1])
        checkpoints = thinned
    return checkpoints


def select_restore_pos(plan, cap):
    """Pick the restore position for a partial-restore ``plan``, bounded by
    ``cap`` (the new request's extended-prefix length — restoring past it
    would leave nothing to prefill forward from).

    All-trimmable donor: exactly ``min(d, cap)`` — any position is
    recoverable by slicing. Donor with checkpoint-class layers (recurrent
    or sliding-window): the highest checkpoint position <= ``min(d, cap)``
    (such state restores only where captured).

    Returns ``(pos, ckpt_states, ckpt_metas)``; ``(0, None, None)`` when
    nothing usable.
    """
    d = min(plan["d"], cap)
    if d <= 0:
        return 0, None, None
    kinds = plan.get("kinds")
    has_ckpt_layers = (
        any(k == "ckpt" for k in kinds)
        if kinds
        else any(isinstance(st, list) for st in plan["snapshot"])
    )
    if has_ckpt_layers:
        if d == plan["donor_len"]:
            # The donor's entire chain is shared (e.g. identical request
            # resent): the snapshot itself holds the checkpoint-class state
            # AT d — restore it whole, no checkpoint needed.
            states = {}
            metas = {}
            snap_metas = plan.get("metas")
            for i, st in enumerate(plan["snapshot"]):
                is_ckpt = (
                    kinds[i] == "ckpt" if kinds else isinstance(st, list)
                )
                if is_ckpt:
                    states[i] = st
                    metas[i] = (
                        snap_metas[i]
                        if snap_metas and i < len(snap_metas)
                        else None
                    )
            return d, states, metas
        pos = 0
        states = None
        metas = None
        for cp in plan["checkpoints"]:
            if pos < cp["pos"] <= d:
                pos = cp["pos"]
                states = cp["states"]
                metas = cp.get("metas")
        if states is None:
            return 0, None, None
        return pos, states, metas
    return d, {}, {}


def build_partial_restore_states(snapshot, ckpt_states, pos,
                                 kinds=None, ckpt_metas=None):
    """Per-layer ``(states, metas)`` to install for a restore at ``pos``.

    ``trim`` layers (plain KVCache): slice the donor snapshot to ``pos`` —
    the ``state`` setter re-derives ``offset`` from ``keys.shape[2]``, so
    assignment is position-exact. ``ckpt`` layers (ArraysCache recurrent,
    RotatingKVCache sliding-window): take the checkpoint's state (+ meta —
    ring indices for Rotating) captured AT ``pos``. Never trims either.
    ``opaque`` layers refuse partial restore entirely.

    Without ``kinds`` (legacy donors, e.g. SSD format <= v2), falls back to
    shape-based classification: tuple => trim, list => ckpt — correct for
    every model that could have produced such an entry (Rotating models
    could not cache before kinds existed).

    Returns ``(states_list, metas_list)``, or ``(None, None)`` if a ckpt
    layer has no checkpoint state or an opaque layer is present (caller
    falls back to the next restore source / cold prefill).
    """
    out = []
    metas = []
    for i, st in enumerate(snapshot):
        kind = kinds[i] if kinds else (
            "ckpt" if isinstance(st, list) else "trim"
        )
        if kind == "trim":
            k, v = st
            out.append((k[..., :pos, :], v[..., :pos, :]))
            metas.append(None)
        elif kind == "ckpt":
            ck = ckpt_states.get(i) if ckpt_states else None
            if ck is None:
                return None, None
            out.append(ck)
            metas.append(ckpt_metas.get(i) if ckpt_metas else None)
        else:  # opaque
            return None, None
    return out, metas


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
        # Per-layer meta_state + kind for the active snapshot (meta-state
        # patch): RotatingKVCache needs meta (ring indices) to round-trip;
        # kinds drive partial-restore semantics (trim/ckpt/opaque) since
        # state shape can't distinguish Rotating from plain KVCache.
        self.snapshot_meta = None  # list[meta_state | None] | None
        self.snapshot_kinds = None  # list["trim"|"ckpt"|"opaque"] | None
        # Partial-restore checkpoints (patch: system-kv-partial-restore).
        # list[{"pos": int, "states": {layer_idx: state},
        #       "metas": {layer_idx: meta_state | None}}], sorted ascending
        # by pos. Checkpoint-class layers only (recurrent + sliding-window)
        # — attention KV at any position is sliced from ``snapshot``. Empty
        # for all-trimmable models (any position restorable by slicing).
        self.checkpoints: list = []
        # Side-stash LRU for inactive slots, keyed by system_hash.
        # value: {"snapshot": list, "token_count": int, "token_ids": list,
        #         "checkpoints": list}
        self.lru: "OrderedDict[str, dict]" = OrderedDict()
        # Capacity is the TOTAL slot count (active + LRU bag).
        # The LRU bag therefore holds capacity-1 entries at most.
        import os as _os
        self.capacity = max(1, int(
            _os.environ.get("VLLM_MLX_SYSTEM_KV_SLOTS", "4")
        ))
        # Partial-restore knobs: max checkpoints kept per slot (geometric
        # thinning above the cap) and the minimum restorable prefix worth a
        # partial restore (below it, cold prefill is cheap enough).
        self.ckpt_capacity = max(1, int(
            _os.environ.get("VLLM_MLX_SYSTEM_KV_CHECKPOINTS", "8")
        ))
        self.partial_min = max(0, int(
            _os.environ.get("VLLM_MLX_SYSTEM_KV_PARTIAL_MIN", "256")
        ))
        # RAM byte budget for the resident slot set (active + LRU bag +
        # checkpoints). The slot snapshots are the only UNBOUNDED RAM term in
        # the serving process: a grown deep-context slot is multi-GB (a ~80K
        # 27B chain ≈ 5 GB), so 4 slots can hold ~14 GB beside ~15 GB weights
        # on a 64 GB box — observed alongside the live jetsam/Metal-abort
        # cluster on the 27B coding route. When set (MB; 0/unset = unlimited,
        # preserving the prior unbounded behavior), ``enforce_ram_budget``
        # evicts LRU-bag entries (SSD-spilled first — cheap to re-promote —
        # then oldest, never the in-use active slot) until the resident set
        # fits. Deploy per-model via VLLM_MLX_SYSTEM_KV_RAM_MB.
        self.ram_budget_bytes = max(0, int(
            _os.environ.get("VLLM_MLX_SYSTEM_KV_RAM_MB", "0")
        )) * 1024 * 1024
        # Whether the ACTIVE slot's prefix is covered by an SSD entry (came
        # from a promote, or its write-through spill was accepted) — drives
        # the cheap-to-recover-first eviction order. Tracked here because it
        # rides into the bag dict on demote.
        self.active_spilled = False
        # Prometheus-side counters — see vllm-mlx-system-kv-metrics.py
        self.hits = 0
        self.misses = 0
        self.tokens_saved = 0
        self.evictions = 0
        self.partial_hits = 0
        self.partial_tokens_saved = 0
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

    def maybe_start_ssd_store(self, model_name: str, idle_check=None) -> None:
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
                    ),
                    idle_check=idle_check,
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
        self.snapshot_meta = None
        self.snapshot_kinds = None
        self.checkpoints = []
        self.active_spilled = False
        self.lru.clear()
        self.evictions = 0
        self.hits = 0
        self.misses = 0
        self.tokens_saved = 0
        self.partial_hits = 0
        self.partial_tokens_saved = 0
        self.ssd_promotes = 0
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
                "checkpoints": self.checkpoints,
                "meta": self.snapshot_meta,
                "kinds": self.snapshot_kinds,
                "bytes": entry_bytes(self.snapshot) + ckpt_bytes(self.checkpoints),
                "spilled": self.active_spilled,
            }
            self.lru.move_to_end(self.system_hash)
        self.snapshot = entry["snapshot"]
        self.system_hash = system_hash
        self.token_count = entry["token_count"]
        self.token_ids = entry["token_ids"]
        self.checkpoints = entry.get("checkpoints", [])
        self.snapshot_meta = entry.get("meta")
        self.snapshot_kinds = entry.get("kinds")
        self.active_spilled = entry.get("spilled", False)
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
            "checkpoints": self.checkpoints,
            "meta": self.snapshot_meta,
            "kinds": self.snapshot_kinds,
            "bytes": entry_bytes(self.snapshot) + ckpt_bytes(self.checkpoints),
            "spilled": self.active_spilled,
        }
        self.lru.move_to_end(self.system_hash)
        # Clear active so the caller's assignment is clean.
        self.snapshot = None
        self.system_hash = None
        self.token_count = 0
        self.token_ids = None
        self.snapshot_meta = None
        self.snapshot_kinds = None
        self.checkpoints = []
        self.active_spilled = False
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

    def enforce_ram_budget(self):
        """Evict LRU-bag slots until the resident set fits ``ram_budget_bytes``.

        No-op when the budget is 0/unset (default — prior unbounded
        behavior). The in-use ACTIVE slot is never evicted: the budget caps
        the BAG overhang (the slots that exist only to avoid a future cold
        prefill), which is the unbounded term behind the 27B memory-abort
        cluster. If the active slot alone exceeds the budget, we log and keep
        it — the in-flight request needs it.

        Eviction order: SSD-spilled bag entries first (re-acquiring them is a
        ~1.3 s promote, not a ~25-39 s cold prefill), oldest-first within each
        class. ``mx.clear_cache()`` fires once after eviction so the Metal
        allocator returns the freed buffers (same discipline as
        ``lru_demote_active_to_bag``).

        Runs inside ``_generation_lock`` (called from the store sites). The
        TOCTOU contract is unchanged: a worker holding an evicted snapshot ref
        in its closure keeps it alive by refcount after the dict drop.
        """
        budget = self.ram_budget_bytes
        if budget <= 0:
            return
        active_bytes = entry_bytes(self.snapshot) + ckpt_bytes(self.checkpoints)

        def _total():
            return active_bytes + sum(
                e.get("bytes", 0) for e in self.lru.values()
            )

        if _total() <= budget:
            return
        evicted = 0
        # Two passes: spilled (cheap to recover) then unspilled, each
        # oldest-first (OrderedDict insertion order == LRU order).
        for spilled_first in (True, False):
            if _total() <= budget:
                break
            for ev_hash in list(self.lru.keys()):
                if _total() <= budget:
                    break
                entry = self.lru.get(ev_hash)
                if entry is None or bool(entry.get("spilled")) != spilled_first:
                    continue
                self.lru.pop(ev_hash, None)
                self.evictions += 1
                evicted += 1
                logger.info(
                    "System KV cache RAM-budget EVICTED: hash=%s "
                    "(%.0f MB, spilled=%s; budget=%.0f MB, resident=%.0f MB)",
                    ev_hash,
                    entry.get("bytes", 0) / 1e6,
                    bool(entry.get("spilled")),
                    budget / 1e6,
                    _total() / 1e6,
                )
        if active_bytes > budget and not self.lru:
            logger.warning(
                "System KV active slot (%.0f MB) exceeds RAM budget "
                "(%.0f MB); keeping it (in-flight request needs it)",
                active_bytes / 1e6,
                budget / 1e6,
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

    def record_partial_hit(self, tokens_restored):
        """Bump the PARTIAL-HIT counters. The restored tokens also count
        toward the aggregate ``tokens_saved`` gauge (they are prompt tokens
        the prefill did not recompute), while ``partial_*`` keep the
        partial path separately observable.
        """
        self.partial_hits += 1
        self.partial_tokens_saved += tokens_restored
        self.tokens_saved += tokens_restored

    def plan_partial_restore(self, full_tokens_list):
        """Plan a checkpointed partial restore against the ACTIVE slot.

        Called at gate time (after ``lru_promote``) when the strict
        extended-prefix match failed: the new prompt shares only the first
        D tokens with the cached chain (divergence — opencode compaction,
        edited turns, regenerated/retried turns, interleaved sessions on
        one system prompt).

        Returns gate-time references (concurrent slot reassignment cannot
        alias them — same pattern as ``lookup_active``): ``{"d", "snapshot",
        "checkpoints", "has_recurrent", "donor_len"}``, or None when even
        the full divergence point cannot clear ``partial_min``. The actual
        restore position is chosen in the worker via
        ``select_restore_pos`` — it must also be capped to the new
        request's extended-prefix length, which only the worker knows.
        Must run inside ``_generation_lock``.
        """
        if not self.is_safe():
            return None
        snapshot = self.snapshot
        token_ids = self.token_ids
        if snapshot is None or not token_ids:
            return None
        kinds = self.snapshot_kinds
        # Opaque layers (e.g. QuantizedKVCache) can't be trimmed or
        # checkpoint-restored — no partial plan for such donors.
        if kinds and any(k == "opaque" for k in kinds):
            return None
        d = common_prefix_len(full_tokens_list, token_ids)
        if d < self.partial_min:
            return None
        return {
            "d": d,
            "snapshot": snapshot,
            "checkpoints": self.checkpoints,
            "kinds": kinds,
            "metas": self.snapshot_meta,
            "donor_len": self.token_count,
        }

    def carry_checkpoints(self, checkpoints, pos):
        """Checkpoints from a donor chain that remain valid on a new chain
        restored at ``pos`` (common prefix ⇒ every checkpoint at or before
        the restore position describes the same tokens). Returns a fresh
        list so donor and new slot never share the mutable container.
        """
        return [cp for cp in (checkpoints or []) if cp["pos"] <= pos]

    def store_snapshot(self, system_hash, snapshot, token_count):
        """Install ``snapshot`` as the active slot (stream_chat MISS store).

        LRU: demote the about-to-be-replaced active slot
        to the bag before overwriting (only if it holds a
        different system_hash from the new MISS).

        Clears ``token_ids`` — the stream_chat branch caches the literal
        system prefix, not the extended prefix, so a stale extended token
        list from a prior ``store_extended`` for the same hash must not
        linger. (It could never produce a false extended HIT —
        ``match_extended_prefix`` requires ``len(token_ids) == token_count``
        to match — but the slot state should be honest, not
        accidentally-safe.) Runs inside the serialized generation worker.
        """
        if self.system_hash and self.system_hash != system_hash:
            self.lru_demote_active_to_bag()
        self.snapshot = snapshot
        self.system_hash = system_hash
        self.token_count = token_count
        self.token_ids = None
        self.checkpoints = []
        # stream_chat (MLLM+media) branch has no SSD tier; the slot is not
        # SSD-covered, so it is evicted last under RAM pressure.
        self.active_spilled = False
        self.enforce_ram_budget()

    def store_extended(
        self, system_hash, snapshot, extended_token_ids, promoted,
        checkpoints=None, meta=None, kinds=None,
    ):
        """Install an EXTENDED-prefix snapshot as the active slot
        (text-route MISS store), gated on the kill switch: when disabled
        the request still prefilled its own prompt cache, but NOTHING
        persists. Runs inside the serialized generation worker.

        ``checkpoints`` is the partial-restore checkpoint list captured
        during this prefill (plus any carried over a partial restore);
        ``meta``/``kinds`` are the per-layer meta_state and trim/ckpt/opaque
        classification captured alongside the snapshot.
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
            self.checkpoints = list(checkpoints) if checkpoints else []
            self.snapshot_meta = meta
            self.snapshot_kinds = kinds
            # Patch #16: write-through to SSD when freshly computed (not
            # promoted from disk). One async write per distinct system
            # prefix at creation; the grow path does NOT re-spill — a
            # restart promotes the stored prefix and re-grows cheaply.
            spilled_ok = False
            if self.ssd_store is not None and not promoted:
                spilled_ok = self.ssd_store.enqueue_spill(
                    tuple(extended_token_ids), snapshot,
                    checkpoints=self.checkpoints,
                    meta=meta, kinds=kinds,
                )
            # The slot is SSD-covered if it came from a promote, or its
            # write-through spill was accepted (not dropped on a full queue).
            # A queue-full drop leaves it uncovered → evicted last (we can't
            # cheaply re-promote what was never written).
            self.active_spilled = bool(
                self.ssd_store is not None and (promoted or spilled_ok)
            )
            # Cap the resident slot set (active + bag) — see __init__.
            self.enforce_ram_budget()

    # ------------------------------------------------------------------
    # Stats (patches #7/#10/#13/#16 + perf-observability)
    # ------------------------------------------------------------------

    def stats(self):
        """Assemble the ``system_kv_cache`` stats dict, or None when disabled.

        System KV cache stats — patched to emit Prometheus-compatible
        fields (hits/misses/tokens_saved/entry_count) so metrics.py
        maps them onto the vllm_mlx_cache_* gauges. Broadened gate so
        misses are reported even before a snapshot is stored.

        Idle-but-enabled emits a ZEROED block (not None) carrying
        ``enabled: True`` — so ``/v1/status`` and the exporter can prove the
        system-KV path is live BEFORE the first request, instead of the
        ``cache: null`` blind spot that previously hid a disabled/degraded
        route. Returns None only when the kill switch (``is_safe() == False``)
        genuinely disables the path. The zeroed block has no hit/miss/
        tokens_saved activity, so neither the ``/v1/status`` cache selection
        nor metrics.py's activity-preferring scan will let it mask an active
        ``memory_aware_cache`` (verified: both gate on those three fields).

        Multi-slot LRU note: active slot + bag entries are summed for
        the aggregate fields (memory_mb, current_memory_mb,
        entry_count). Legacy single-slot fields (tokens, hash) still
        describe the ACTIVE slot for backward compat with the
        Prometheus exporter and existing dashboards.
        """

        if not (
            self.snapshot is not None
            or self.lru
            or self.hits
            or self.misses
        ):
            if not self.is_safe():
                return None
            return {
                "enabled": True,
                "tokens": 0,
                "hash": None,
                "memory_mb": 0.0,
                "current_memory_mb": 0.0,
                "entry_count": 0,
                "hits": 0,
                "misses": 0,
                "hit_rate": 0.0,
                "tokens_saved": 0,
                "partial_hits": 0,
                "partial_tokens_saved": 0,
                "checkpoints": 0,
                "evictions": 0,
                "capacity": self.capacity,
                "slots": [],
            }

        active_bytes = entry_bytes(self.snapshot) + ckpt_bytes(self.checkpoints)
        bag_bytes = sum(
            entry_bytes(e["snapshot"]) + ckpt_bytes(e.get("checkpoints"))
            for e in self.lru.values()
        )
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
                "memory_mb": round(entry_bytes(entry["snapshot"]) / 1e6, 1),
                "active": False,
            })
        result = {
            "enabled": True,
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
            # Partial-restore (checkpointed) fields:
            "partial_hits": self.partial_hits,
            "partial_tokens_saved": self.partial_tokens_saved,
            "checkpoints": len(self.checkpoints),
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
