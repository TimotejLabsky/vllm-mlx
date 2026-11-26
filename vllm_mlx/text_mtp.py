"""Fork #160 — external MTP drafter on the TEXT path (BatchedEngine).

The drafter head (``mlx-community/Qwen3.8-27B-MTP-*``, loaded through mlx-vlm)
proposes the next tokens from the target's last hidden state; the target
verifies them in one forward pass. This module runs that on mlx-lm's
``GenerationBatch`` - the route the batched system-KV cache lives on - instead
of the cacheless ``--mllm`` route (#159).

Design (measured on the Studio 2026-11-16, ``reports/mtp-drafter-2026-11-16``):

* **Transactional rollback.** The verify forward records the recurrent
  (GatedDeltaNet) state after every position, so a partial accept selects a
  state instead of re-running the target. KV layers are trimmed.
* **The cache never holds an unemitted token when anyone looks.** A round can
  leave the cache up to ``block - 1`` verified tokens ahead of what
  ``GenerationBatch.next()`` has returned. Every way of reading or reshaping
  the batch cache (``extract_cache``, ``filter``, ``extend``) first rolls the
  cache back to the emitted position. A cached entry is therefore the same
  state plain decoding produces (the spike measured zero difference).
* **Single stream only.** With more than one row in the batch the plain step
  runs; so does it for any logits processor that can neither plan its effect
  on drafted positions (the thinking budget can) nor be called with a drafted
  position's context (DRY can). Structured output is the one that bypasses.

* **Rounds overlap with emission (#161).** A round is *launched* (graph
  built, ``mx.async_eval``) as soon as its first token is known and
  *completed* (accept walk, rollback) only when its results are needed. The
  tokens of the previous round are returned by host-only steps while the GPU
  already verifies the next one - the same overlap plain decoding gets from
  mlx-lm's own ``async_eval``. ``VLLM_MLX_TEXT_MTP_OVERLAP=0`` blocks at
  launch instead (the #160 behaviour), for A/B.

Enable with ``VLLM_MLX_TEXT_MTP_DRAFTER=<repo or path>``; block size (verified
positions per round) via ``VLLM_MLX_TEXT_MTP_BLOCK`` (default 2: one draft per round was
the robust optimum at every context length measured).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from . import vision_unified as _vu

logger = logging.getLogger(__name__)

ENV_DRAFTER = "VLLM_MLX_TEXT_MTP_DRAFTER"
ENV_BLOCK = "VLLM_MLX_TEXT_MTP_BLOCK"
ENV_OVERLAP = "VLLM_MLX_TEXT_MTP_OVERLAP"
# (#167) Speculative SAMPLING for requests that sample (T > 0): the draft is
# drawn from the drafter's own filtered distribution q and accepted with
# probability min(1, p/q), a rejection resampled from max(p - q, 0). Output is
# distributed exactly as the request's sampler; acceptance is sum(min(p, q))
# instead of p(argmax q). Greedy requests keep the exact-match rule.
ENV_SPEC_SAMPLING = "VLLM_MLX_TEXT_MTP_SPEC_SAMPLING"
SPEC_SAMPLING_MAX_TOP_K = 64
DISABLE_AFTER_ERRORS = 8

_REC = {"active": False}
_patch_lock = threading.Lock()
_patched = False
_orig: Dict[str, Any] = {}


def drafter_path() -> Optional[str]:
    value = os.environ.get(ENV_DRAFTER, "").strip()
    return value or None


def _block_size() -> int:
    try:
        return max(2, min(int(os.environ.get(ENV_BLOCK, "2")), 6))
    except ValueError:
        return 2


# --------------------------------------------------------------------------
# memory admission (#162)
# --------------------------------------------------------------------------
_weight_bytes_cache: Dict[str, int] = {}


def _drafter_weight_bytes(path: str) -> int:
    """Size of the drafter's weights on disk (what loading adds to active
    memory). 0 when it cannot be resolved offline - the guard fails open."""
    if path in _weight_bytes_cache:
        return _weight_bytes_cache[path]
    total = 0
    try:
        local = path
        if not os.path.isdir(local):
            from huggingface_hub import snapshot_download

            local = snapshot_download(
                path, local_files_only=True, allow_patterns=["*.safetensors"]
            )
        total = sum(
            os.path.getsize(os.path.join(local, name))
            for name in os.listdir(local)
            if name.endswith(".safetensors")
        )
    except Exception:  # noqa: BLE001 - pricing must never block a request
        logger.debug("[text-mtp] drafter weight size unavailable", exc_info=True)
    _weight_bytes_cache[path] = total
    return total


def _round_state_bytes(model: Any, block: int) -> int:
    """Recurrent states one round keeps until the next: one float32 state per
    verified position per recurrent layer, plus the copy a rollback makes."""
    from .mlx_lm_compat import text_tree

    lm, _ = text_tree(model)  # (#180) either mlx-lm qwen3_5 tree
    layers = getattr(getattr(lm, "model", None), "layers", None) or []
    per_position = 0
    for layer in layers:
        m = getattr(layer, "linear_attn", None)
        if m is not None:
            per_position += (
                int(m.num_v_heads) * int(m.head_v_dim) * int(m.head_k_dim) * 4
            )
    return per_position * (block + 1)


def admission_extra_bytes(scheduler: Any) -> float:
    """What a request that will decode alone adds to active memory because
    the drafter runs, beyond its own KV: the drafter's weights if they are
    not loaded yet (they load on the first drafted round, AFTER the admission
    check), and one round's recorded recurrent states. 0 without the env or
    once the drafter is disabled. On the 27B-8bit: 451 MB + 453 MB."""
    path = drafter_path()
    if path is None:
        return 0.0
    mtp = getattr(scheduler, "_text_mtp", None)
    if mtp is not None and mtp.disabled_reason is not None:
        return 0.0
    loaded = mtp is not None and mtp.drafter is not None
    block = mtp.block if mtp is not None else _block_size()
    try:
        states = _round_state_bytes(getattr(scheduler, "model", None), block)
    except Exception:  # noqa: BLE001
        states = 0
    return float((0 if loaded else _drafter_weight_bytes(path)) + states)


class TextMTP:
    """Per-scheduler drafter and counters; survives generator re-creation."""

    def __init__(self, path: str):
        self.path = path
        self.block = _block_size()
        self.overlap = os.environ.get(ENV_OVERLAP, "1") != "0"
        self.spec_sampling = os.environ.get(ENV_SPEC_SAMPLING, "0") == "1"
        self.drafter: Any = None
        self.lm: Any = None
        self.lock = threading.Lock()
        self.counters = {
            "rounds": 0,
            "bootstrap_rounds": 0,
            "drafted_tokens": 0,
            "accepted_tokens": 0,
            "emitted_tokens": 0,
            "rollbacks": 0,
            "sync_rollbacks": 0,
            "errors": 0,
            "ss_rounds": 0,
            "ss_accepted": 0,
        }
        # Where a round's wall time goes (seconds, cumulative): building and
        # submitting it, waiting for the GPU at completion, and everything
        # the caller does between two steps of a speculating request.
        self.seconds = {"launch": 0.0, "wait": 0.0, "between_steps": 0.0, "step": 0.0}
        self.last_step_end: Optional[float] = None
        self.bypass = {
            "concurrent_batch": 0,
            "logits_processors": 0,
            "cache_mismatch": 0,
            "disabled": 0,
        }
        self.consecutive_errors = 0
        self.disabled_reason: Optional[str] = None

    # -- loading -----------------------------------------------------------
    def ensure_loaded(self, model: Any) -> bool:
        """Load on the generation thread at first use (MLX streams are
        thread-local; see PATCHES #28)."""
        if self.drafter is not None:
            return True
        if self.disabled_reason is not None:
            return False
        try:
            from .mlx_lm_compat import text_tree

            # (#180) the legacy TextModel on 0.32.2, a view with the same
            # surface (.model, .args, .lm_head) on the post-#2082 tree.
            lm, how = text_tree(model)
            if not callable(getattr(lm, "model", None)):
                raise TypeError(
                    f"{type(model).__name__} exposes no hidden-state model ({how})"
                )
            # _head needs the text args and, untied, an lm_head: a miss is
            # one load-time error, not a failed round per step.
            lm_args = getattr(lm, "args", None)
            if lm_args is None:
                raise TypeError(f"{type(model).__name__} exposes no text args ({how})")
            if not getattr(lm_args, "tie_word_embeddings", False) and not callable(
                getattr(lm, "lm_head", None)
            ):
                raise TypeError(
                    f"{type(model).__name__} is untied but exposes no lm_head ({how})"
                )
            if not _supports_recording(lm):
                raise TypeError("target recurrent layers are not mlx-lm GatedDeltaNet")
            from mlx_vlm.speculative.drafters import load_drafter

            drafter, kind = load_drafter(self.path, kind="mtp")
            drafter.train(False)
            mx.eval(drafter.parameters())
            _install_class_patches()
            self.lm = lm
            self.drafter = drafter
            logger.info(
                "[text-mtp] drafter loaded: %s kind=%s block=%d",
                self.path,
                kind,
                self.block,
            )
            return True
        except Exception as e:  # noqa: BLE001 - a broken drafter must not kill decode
            self.disabled_reason = f"load failed: {type(e).__name__}: {e}"
            logger.error("[text-mtp] DISABLED: %s", self.disabled_reason, exc_info=True)
            return False

    # -- stats ---------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        with self.lock:
            c = dict(self.counters)
            bypass = dict(self.bypass)
        drafted, accepted = c["drafted_tokens"], c["accepted_tokens"]
        rounds = c["rounds"]
        return {
            "enabled": True,
            "implementation": "text_external_assistant",
            "mode": "transactional_verified",
            "overlap": self.overlap,
            "draft_model": self.path,
            "requested_draft_tokens": self.block - 1,
            "effective_draft_tokens": self.block - 1,
            "attempted": rounds,
            "accepted": accepted,
            "rejected": drafted - accepted,
            "errors": c["errors"],
            "acceptance_rate": accepted / drafted if drafted else 0.0,
            "tokens_per_round": c["emitted_tokens"] / rounds if rounds else 0.0,
            "bootstrap_rounds": c["bootstrap_rounds"],
            "rollbacks": c["rollbacks"],
            "sync_rollbacks": c["sync_rollbacks"],
            "ms_per_round": {
                k: round(1000.0 * v / rounds, 2) if rounds else 0.0
                for k, v in dict(self.seconds).items()
            },
            "spec_sampling": self.spec_sampling,
            "ss_rounds": c["ss_rounds"],
            "ss_acceptance_rate": (
                c["ss_accepted"] / c["ss_rounds"] if c["ss_rounds"] else 0.0
            ),
            "bypass_counts": bypass,
            "bypass_counts_semantics": "per_condition_overlapping_not_total_steps",
            "disabled": self.disabled_reason is not None,
            "disabled_reason": self.disabled_reason,
        }

    def count(self, key: str, n: int = 1) -> None:
        with self.lock:
            self.counters[key] += n

    def count_bypass(self, key: str) -> None:
        with self.lock:
            self.bypass[key] = self.bypass.get(key, 0) + 1


class _BatchState:
    """Speculation state of one persistent GenerationBatch."""

    __slots__ = (
        "mtp",
        "uid",
        "hidden",
        "queue",
        "round_h",
        "kept",
        "drafter_ready",
        "pending",
        "cache_ok",
        "ss",
        "q",
    )

    def __init__(self, mtp: TextMTP):
        self.mtp = mtp
        self.uid: Any = None
        self.hidden: Optional[mx.array] = None
        # Tokens the cache already holds that _step has not returned yet, in
        # emission order: (token, logprobs, is_draft). Accepted drafts of the
        # last completed round, then the first token of the launched round.
        self.queue: Deque[Tuple[int, mx.array, bool]] = deque()
        self.round_h: Optional[mx.array] = None
        self.kept = 0  # tokens of the last COMPLETED round the cache holds
        self.drafter_ready = False
        self.pending: Optional["_Pending"] = None  # launched, not completed
        # The cache-shape check reads one scalar per recurrent layer off the
        # GPU (24 round trips on the 27B). Its answer cannot change while the
        # same request decodes alone - ``left_padding`` only ever decreases -
        # so it is asked once per request, not once per round.
        self.cache_ok = False
        # (#167) This request's (temp, top_p, top_k) when it speculates by
        # sampling, else None - fixed per request, because the draft a round
        # verifies was drawn in the previous round. ``q`` is the sparse
        # distribution (indices, log-probs) the newest draft was drawn from.
        self.ss: Optional[Tuple[float, float, int]] = None
        self.q: Optional[Tuple[mx.array, mx.array]] = None

    def forget(self) -> None:
        """Only from a clean state (after ``_sync``): drops, does not undo."""
        self.hidden = None
        self.round_h = None
        self.queue.clear()
        self.kept = 0
        self.drafter_ready = False
        self.pending = None
        self.cache_ok = False
        self.q = None


class _Pending:
    """A launched round: everything needed to complete it or to undo it."""

    __slots__ = (
        "b",
        "b_logprobs",
        "snap",
        "ctx_size",
        "block",
        "drafts",
        "draft_list",
        "h",
        "logprobs",
        "targets",
        "limit",
        "processors",
        "ss",
    )


# --------------------------------------------------------------------------
# install
# --------------------------------------------------------------------------
def maybe_install(scheduler: Any, batch_gen: Any) -> None:
    """Scheduler hook: attach the drafter to a freshly built BatchGenerator."""
    path = drafter_path()
    if path is None:
        return
    batch = getattr(batch_gen, "_generation_batch", None)
    if batch is None or not hasattr(type(batch), "_step"):
        logger.error(
            "[text-mtp] %s is set but this mlx-lm has no GenerationBatch._step; "
            "decoding continues without the drafter",
            ENV_DRAFTER,
        )
        return
    mtp = getattr(scheduler, "_text_mtp", None)
    if mtp is None or mtp.path != path:
        mtp = TextMTP(path)
        scheduler._text_mtp = mtp
    gone = hook_symbols_missing()
    if gone:
        # (#175) mlx-lm moved what the drafter hooks (#2082); patching would
        # raise out of the batch generator build. Same semantics as a failed
        # load (#159): one ERROR, ``disabled`` in the stats, decoding
        # continues without the drafter.
        if mtp.disabled_reason is None:
            mtp.disabled_reason = "mlx-lm layout: " + "; ".join(gone)
            logger.error("[text-mtp] DISABLED: %s", mtp.disabled_reason)
        batch_gen.get_mtp_stats = mtp.stats
        return
    _install_class_patches()
    batch._text_mtp = _BatchState(mtp)
    batch_gen.get_mtp_stats = mtp.stats
    logger.info("[text-mtp] armed on the text path: %s block=%d", path, mtp.block)


# (#175) The mlx-lm symbols the drafter binds, resolved on both the 0.32.2
# layout and the post-#2082 one (``mlx_lm_compat``).
HOOK_SYMBOLS = ("qwen3_5.GatedDeltaNet", "gated_delta_update", "normalize_qk")


def hook_symbols_missing() -> List[str]:
    """Errors for the hooked mlx-lm symbols that do not resolve."""
    from .mlx_lm_compat import missing

    return missing(HOOK_SYMBOLS)


def _gdn_class() -> Any:
    from .mlx_lm_compat import resolve

    return resolve("qwen3_5.GatedDeltaNet")


def _supports_recording(lm: Any) -> bool:
    gdn_cls = _gdn_class()
    layers = getattr(getattr(lm, "model", None), "layers", None) or []
    recurrent = [
        layer.linear_attn for layer in layers if getattr(layer, "is_linear", False)
    ]
    return bool(recurrent) and all(
        type(m) is gdn_cls and m.sharding_group is None for m in recurrent
    )


def _install_class_patches() -> None:
    """Idempotent. Every wrapper is a no-op for a batch without ``_text_mtp``
    and for a GatedDeltaNet call outside a recording verify."""
    global _patched
    with _patch_lock:
        if _patched:
            return
        from mlx_lm.generate import GenerationBatch

        gdn_cls = _gdn_class()
        _orig["step"] = GenerationBatch._step
        _orig["extract_cache"] = GenerationBatch.extract_cache
        _orig["filter"] = GenerationBatch.filter
        _orig["extend"] = GenerationBatch.extend
        _orig["gdn_call"] = gdn_cls.__call__

        def _step(self):
            state = getattr(self, "_text_mtp", None)
            if state is None:
                return _orig["step"](self)
            return _speculative_step(self, state)

        def extract_cache(self, idx):
            _sync(self)
            return _orig["extract_cache"](self, idx)

        def filter(self, keep):  # noqa: A001 - mlx-lm's name
            _sync(self)
            result = _orig["filter"](self, keep)
            state = getattr(self, "_text_mtp", None)
            if state is not None:
                state.forget()
                _drop_recording(self.prompt_cache)
            return result

        def extend(self, batch):
            _sync(self)
            result = _orig["extend"](self, batch)
            state = getattr(self, "_text_mtp", None)
            if state is not None:
                state.forget()
                _drop_recording(self.prompt_cache)
            return result

        GenerationBatch._step = _step
        GenerationBatch.extract_cache = extract_cache
        GenerationBatch.filter = filter
        GenerationBatch.extend = extend
        gdn_cls.__call__ = _recording_gdn_call
        _patched = True


# --------------------------------------------------------------------------
# recurrent layers: keep the state after every verified position
# --------------------------------------------------------------------------
def _recording_gdn_call(self, inputs, mask=None, cache=None):
    """mlx-lm's ``GatedDeltaNet.__call__`` with the recurrence stepped one
    position at a time, so the state after each position survives. Used only
    inside a recording verify (2-6 positions, batch 1, nothing masked - the
    caller checked); the projections stay batched over the positions. Pinned
    against the original by ``test_160_recording_forward_matches_mlx_lm``."""
    if not _REC["active"] or cache is None:
        return _orig["gdn_call"](self, inputs, mask, cache)
    from .mlx_lm_compat import resolve

    normalize_qk = resolve("normalize_qk")
    gated_delta_update = resolve("gated_delta_update")
    B, S, _ = inputs.shape
    qkv = self.in_proj_qkv(inputs)
    z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
    b = self.in_proj_b(inputs)
    a = self.in_proj_a(inputs)
    conv_state = cache[0]
    if conv_state is None:
        conv_state = mx.zeros(
            (B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype
        )
    if mask is not None:
        qkv = mx.where(mask[..., None], qkv, 0)
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    n_keep = self.conv_kernel_size - 1
    cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
    conv_out = nn.silu(self.conv1d(conv_input))
    q, k, v = [
        t.reshape(B, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
            [self.num_k_heads, self.num_k_heads, self.num_v_heads],
            [self.head_k_dim, self.head_k_dim, self.head_v_dim],
        )
    ]
    q, k = normalize_qk(q, k, inv_scale=self.head_k_dim**-0.5, eps=1e-6)
    state = cache[1]
    outs, states = [], []
    for t in range(S):
        out_t, state = gated_delta_update(
            q[:, t : t + 1],
            k[:, t : t + 1],
            v[:, t : t + 1],
            a[:, t : t + 1],
            b[:, t : t + 1],
            self.A_log,
            self.dt_bias,
            state,
            None if mask is None else mask[:, t : t + 1],
            use_kernel=not self.training,
        )
        outs.append(out_t)
        states.append(state)
    cache[1] = state
    cache.advance(S)
    cache._text_mtp_spec = (conv_input, states, n_keep)
    out = self.norm(mx.concatenate(outs, axis=1), z)
    return self.out_proj(out.reshape(B, S, -1))


def _is_recurrent(c: Any) -> bool:
    return not c.is_trimmable()


def _drop_recording(cache: List[Any]) -> None:
    """(#105) Free the last verify's per-position recurrent states and conv
    window. Only from a clean state (after ``_sync``): nothing can roll back
    into them, and the next verify writes its own. Left in place they stay
    pinned on every recurrent layer for as long as the row lives, including
    every step it runs bypassed beside another request's prefill."""
    for c in cache:
        if getattr(c, "_text_mtp_spec", None) is not None:
            c._text_mtp_spec = None


def _rollback(cache: List[Any], held: int, keep: int) -> None:
    """The cache holds ``held`` tokens of the last verify; keep the first
    ``keep``."""
    for c in cache:
        if _is_recurrent(c):
            conv_input, states, n_keep = c._text_mtp_spec
            c[0] = mx.contiguous(conv_input[:, keep : keep + n_keep, :])
            # ``+ 0``: a bare reference would pin the whole round (#149).
            c[1] = states[keep - 1] + 0
            # ``advance`` counted every verified position.
            if c.left_padding is not None:
                c.left_padding = c.left_padding + (held - keep)
            if c.lengths is not None:
                c.lengths = c.lengths + (held - keep)
        else:
            c.trim(held - keep)


def _snapshot(cache: List[Any]) -> List[Any]:
    """Zero-copy: recurrent arrays are replaced, never written in place."""
    out = []
    for c in cache:
        if _is_recurrent(c):
            out.append(
                (
                    list(c.cache),
                    c.left_padding,
                    c.lengths,
                    getattr(c, "_text_mtp_spec", None),
                )
            )
        else:
            out.append(getattr(c, "_idx", None))
    return out


def _restore(cache: List[Any], snap: List[Any]) -> None:
    for c, s in zip(cache, snap):
        if _is_recurrent(c):
            c.cache, c.left_padding, c.lengths = list(s[0]), s[1], s[2]
            c._text_mtp_spec = s[3]
        elif s is not None:
            c.trim(c._idx - s)


def _cache_supported(cache: List[Any]) -> bool:
    for c in cache:
        if _is_recurrent(c):
            if not hasattr(c, "cache") or len(c.cache) != 2:
                return False
            # Nothing may be masked: a finished prefill leaves ``lengths``
            # unset and ``left_padding`` at or below zero (all-true mask).
            if c.lengths is not None:
                return False
            if c.left_padding is not None and int(c.left_padding.max().item()) > 0:
                return False
        elif not (hasattr(c, "trim") and hasattr(c, "_idx")):
            return False
    return True


# --------------------------------------------------------------------------
# the step
# --------------------------------------------------------------------------
def _greedy(logits: mx.array) -> mx.array:
    return mx.argmax(logits, axis=-1)


# -------------------------------------------------------------- (#167) ----
def _ss_params(batch: Any, mtp: TextMTP) -> Optional[Tuple[float, float, int]]:
    """(temp, top_p, top_k) when this request may speculate by sampling: the
    switch is on, one draft per round, and its sampler is mlx-lm's
    temperature/top-p/top-k chain (the scheduler tags it) with a top-k small
    enough for the sparse filter and no min-p. None keeps exact matching."""
    if not mtp.spec_sampling or mtp.block != 2:
        return None
    sampler = batch.samplers[0] if batch.samplers else None
    params = getattr(sampler, "_vllm_mlx_sampling", None)
    if params is None:
        return None
    temp, top_p, min_p, top_k = params
    if not temp or temp <= 0 or min_p:
        return None
    if not top_k or not 0 < int(top_k) <= SPEC_SAMPLING_MAX_TOP_K:
        return None
    top_p = float(top_p) if top_p and 0 < top_p < 1 else 1.0
    return float(temp), top_p, int(top_k)


def _sparse_dist(
    logits: mx.array, params: Tuple[float, float, int]
) -> Tuple[mx.array, mx.array]:
    """The distribution mlx-lm's ``make_sampler(temp, top_p, top_k)`` samples
    from, over its at most ``top_k`` tokens: (indices, log-probs), dropped
    tokens at -inf. Top-p on the full distribution then top-k keeps the
    longest prefix of the descending order inside both, so the top-k
    candidates with their full-vocabulary probabilities decide it - no
    sort of the vocabulary."""
    temp, top_p, top_k = params
    lp = logits.reshape(-1).astype(mx.float32)
    lp = lp - mx.logsumexp(lp)
    idx = mx.argpartition(-lp, kth=top_k - 1)[:top_k]
    vals = lp[idx]
    order = mx.argsort(-vals)
    idx, vals = idx[order], vals[order]
    probs = mx.exp(vals)
    keep = (mx.cumsum(probs) - probs) < top_p
    keep = mx.logical_or(keep, mx.arange(top_k) == 0)
    tempered = mx.where(keep, vals * (1.0 / temp), -float("inf"))
    return idx, tempered - mx.logsumexp(tempered)


def _q_sampler(state: _BatchState):
    """The drafter's sampler under #167: draw from its filtered distribution
    and remember that distribution for the verify."""

    def sample(logits: mx.array) -> mx.array:
        idx, lp = _sparse_dist(logits, state.ss)
        state.q = (idx, lp)
        return idx[mx.random.categorical(lp)].reshape(1, 1)

    return sample


def _ss_targets(logits: mx.array, draft: mx.array, state: _BatchState) -> mx.array:
    """[accept, rejection token, bonus token, plain token] for one draft
    (block 2): accept with probability min(1, p(d)/q(d)); on reject the
    token comes from norm(max(p - q, 0)); ``plain`` is position 0 sampled
    from p alone, for a round whose draft may not be accepted."""
    pidx0, plp0 = _sparse_dist(logits[0], state.ss)
    pidx1, plp1 = _sparse_dist(logits[1], state.ss)
    qidx, qlp = state.q
    p0, q = mx.exp(plp0), mx.exp(qlp)
    d = draft.reshape(-1)[0]
    pd = mx.sum(mx.where(pidx0 == d, p0, 0.0))
    qd = mx.sum(mx.where(qidx == d, q, 0.0))
    accept = mx.random.uniform() * qd < pd
    q_at_p = mx.sum(mx.where(pidx0[:, None] == qidx[None, :], q[None, :], 0.0), axis=1)
    resid = mx.maximum(p0 - q_at_p, 0.0)
    rejected = pidx0[mx.random.categorical(mx.log(resid + 1e-30))]
    bonus = pidx1[mx.random.categorical(plp1)]
    plain = pidx0[mx.random.categorical(plp0)]
    return mx.stack(
        [accept.astype(mx.int32), rejected.astype(mx.int32),
         bonus.astype(mx.int32), plain.astype(mx.int32)]
    )  # fmt: skip


def _head(lm: Any, hidden: mx.array) -> mx.array:
    if lm.args.tie_word_embeddings:
        return lm.model.embed_tokens.as_linear(hidden)
    return lm.lm_head(hidden)


def _bypass_reason(batch: Any, state: _BatchState) -> Optional[str]:
    mtp = state.mtp
    if mtp.disabled_reason is not None:
        return "disabled"
    if len(batch.uids) != 1:
        return "concurrent_batch"
    if _vu.batch_has_media_row(batch.prompt_cache):
        return "media_row"  # (#181) never drafts for an image request
    processors = (batch.logits_processors or [None])[0] or ()
    for p in processors:
        if not (
            callable(getattr(type(p), "speculation_plan", None))
            or getattr(type(p), "speculation_context_pure", False)
        ):
            return "logits_processors"
    if not mtp.ensure_loaded(batch.model):
        return "disabled"
    if not state.cache_ok:
        if not _cache_supported(batch.prompt_cache):
            return "cache_mismatch"
        state.cache_ok = True
    return None


def _processors(batch: Any) -> Optional[List[Any]]:
    return (batch.logits_processors[0] if batch.logits_processors else None) or None


def _sync(batch: Any) -> None:
    """Make the cache hold exactly the tokens ``_step`` has returned."""
    state: Optional[_BatchState] = getattr(batch, "_text_mtp", None)
    if state is None or (state.pending is None and not state.queue):
        return
    cache = batch.prompt_cache
    pending = state.pending
    if pending is not None:
        if state.queue and not state.queue[-1][2]:
            # Its first token was never returned: the whole round goes.
            state.queue.pop()
            state.pending = None
            _restore(cache, pending.snap)
            if pending.ctx_size is not None:
                batch._token_context[0]._size = pending.ctx_size
            state.drafter_ready = False
        else:
            # Its first token is out, so the cache must keep it: finish the
            # round, then drop whatever it accepted beyond that token.
            _complete_or_recover(batch, state)
    ahead = len(state.queue)
    if not ahead:
        return
    keep = state.kept - ahead
    _rollback(cache, state.kept, keep)
    token, logprobs, _ = state.queue[0]
    batch._next_tokens = mx.array([token], dtype=batch._next_tokens.dtype)
    batch._next_logprobs = [logprobs]
    if _processors(batch):
        batch._token_context[0]._size -= ahead
    state.hidden = state.round_h[:, keep - 1 : keep, :]
    state.queue.clear()
    state.kept = keep
    state.drafter_ready = False  # its own KV is now ahead; restart it cold
    state.mtp.count("sync_rollbacks")


def _note_failure(mtp: TextMTP, where: str, e: Exception) -> None:
    mtp.count("errors")
    mtp.consecutive_errors += 1
    logger.warning("[text-mtp] %s failed, plain decode continues: %s", where, e)
    if mtp.consecutive_errors >= DISABLE_AFTER_ERRORS and mtp.disabled_reason is None:
        mtp.disabled_reason = f"{type(e).__name__}: {e}"
        logger.error(
            "[text-mtp] DISABLED for this process after %d consecutive "
            "failures; decoding continues without speculation. Last error: %s",
            mtp.consecutive_errors,
            mtp.disabled_reason,
            exc_info=True,
        )


def _speculative_step(batch: Any, state: _BatchState):
    mtp = state.mtp
    now = time.perf_counter()
    if mtp.last_step_end is not None and (state.queue or state.pending is not None):
        mtp.seconds["between_steps"] += now - mtp.last_step_end
    try:
        return _speculative_step_inner(batch, state)
    finally:
        mtp.last_step_end = time.perf_counter()
        mtp.seconds["step"] += mtp.last_step_end - now


def _speculative_step_inner(batch: Any, state: _BatchState):
    if not batch.uids:
        return _orig["step"](batch)
    if state.uid != batch.uids[0] or len(batch.uids) != 1:
        _sync(batch)  # membership changed without a hook
        state.forget()
        state.uid = batch.uids[0]
        state.ss = _ss_params(batch, state.mtp) if len(batch.uids) == 1 else None

    if not state.queue:
        if state.pending is not None:
            _complete_or_recover(batch, state)
        reason = _bypass_reason(batch, state)
        # Tokens this request may still return, beyond what is queued.
        room = batch.max_tokens[0] - batch._num_tokens[0] - len(state.queue)
        if reason is None and room >= 1:
            try:
                _launch(batch, state, _processors(batch), max_block=room)
            except Exception as e:  # noqa: BLE001 - decode must continue
                _note_failure(state.mtp, "launch", e)
        elif reason is not None and not state.queue:
            state.mtp.count_bypass(reason)

    if state.queue:
        token, logprobs, _ = state.queue.popleft()
        batch.tokens[0].append(token)
        return [token], [logprobs]
    # Nothing queued and nothing launched: the cache holds exactly what was
    # returned and ``_next_tokens`` is the token to feed.
    state.forget()
    return _orig["step"](batch)


def _launch(
    batch: Any,
    state: _BatchState,
    processors: Optional[List[Any]],
    max_block: int,
    enqueue: bool = True,
) -> None:
    """Build and start one round for the pending token ``b``. On an exception
    the cache and the token context are left as they were."""
    mtp, lm, drafter = state.mtp, state.mtp.lm, state.mtp.drafter
    cache = batch.prompt_cache
    b = int(batch._next_tokens[0].item())
    tic = time.perf_counter()
    snap = _snapshot(cache)
    ctx_size = batch._token_context[0]._size if processors else None
    try:
        # Without the hidden state that predicted ``b`` there is nothing to
        # draft from: verify ``b`` alone (a plain step that yields the hidden).
        block = mtp.block if state.hidden is not None else 1
        block = max(1, min(block, max_block))
        if not state.drafter_ready:
            # Cold start: the drafter's own KV begins at this position. Its
            # attention is relative, so only consistency matters.
            drafter.reset(lm)
            drafter.set_shared_kv(
                {}, kv_offset=len(batch.tokens[0]) + (0 if block > 1 else 1)
            )
            state.drafter_ready = True
        ss = state.ss is not None and block == 2
        if block > 1:
            drafts = drafter.draft_block(
                b,
                state.hidden,
                None,
                block,
                _q_sampler(state) if ss else _greedy,
                mx.int32,
                greedy=not ss,
            ).astype(mx.int32)
            verify_in = mx.concatenate(
                [mx.array([[b]], dtype=mx.int32), drafts], axis=1
            )
        else:
            drafts = mx.zeros((1, 0), dtype=mx.int32)
            verify_in = mx.array([[b]], dtype=mx.int32)

        _REC["active"] = block > 1
        try:
            h = lm.model(verify_in, cache)
        finally:
            _REC["active"] = False
        mx.async_eval(h)  # the forward runs while processors do host work
        logits = _head(lm, h)[0]  # (block, vocab)

        draft_list: Optional[List[int]] = None
        limit = block - 1  # how many drafts may be accepted
        if processors:
            draft_list = drafts[0].tolist() if block > 1 else []
            context = batch._token_context[0].update_and_fetch(mx.array([b]))
            first = logits[0:1]
            for processor in processors:
                feeds_off = getattr(processor, "disable_feeds", None)
                if feeds_off is not None:
                    feeds_off()
                first = processor(context, first)
            rows = [first]
            masks: List[Tuple[int, ...]] = []
            # Two kinds of processor get here (see _bypass_reason): one that
            # can plan its effect on drafted positions (thinking budget), and
            # one whose output depends only on the token context (DRY) and is
            # simply called with each drafted position's context.
            pure = [
                p
                for p in processors
                if getattr(type(p), "speculation_context_pure", False)
            ]
            for processor in processors:
                if processor in pure:
                    continue
                plan = processor.speculation_plan(draft_list)
                limit = min(limit, len(plan))
                masks = (
                    plan[:limit]
                    if not masks
                    else [m + q for m, q in zip(masks[:limit], plan[:limit])]
                )
            for j in range(1, limit + 1):
                row = logits[j : j + 1]
                if pure:
                    context_j = mx.concatenate(
                        [context, mx.array(draft_list[:j], dtype=context.dtype)]
                    )
                    for processor in pure:
                        row = processor(context_j, row)
                ids = masks[j - 1] if masks else ()
                if ids:
                    row = mx.array(row)
                    for token_id in ids:
                        row[0, token_id] = float("-inf")
                rows.append(row)
            logits = mx.concatenate(rows, axis=0)

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        if ss and state.q is not None and logits.shape[0] == 2:
            targets = _ss_targets(logits, drafts, state)
        else:
            ss = False
            sampler = (
                batch.samplers[0] if batch.samplers else None
            ) or batch.fallback_sampler
            targets = mx.concatenate(
                [sampler(logprobs[j : j + 1]) for j in range(logprobs.shape[0])],
                axis=0,
            )
        if mtp.overlap:
            mx.async_eval(targets, drafts)
        else:
            mx.eval(targets, drafts)
    except Exception:
        _REC["active"] = False
        _restore(cache, snap)
        if ctx_size is not None:
            batch._token_context[0]._size = ctx_size
        state.hidden = None
        state.drafter_ready = False
        raise

    pending = _Pending()
    pending.b = b
    pending.b_logprobs = batch._next_logprobs[0] if batch._next_logprobs else None
    pending.snap = snap
    pending.ctx_size = ctx_size
    pending.block = block
    pending.drafts = drafts
    pending.draft_list = draft_list
    pending.h = h
    pending.logprobs = logprobs
    pending.targets = targets
    pending.limit = limit
    pending.processors = processors
    pending.ss = ss
    state.pending = pending
    if enqueue:
        state.queue.append((b, pending.b_logprobs, False))
    mtp.seconds["launch"] += time.perf_counter() - tic


def _complete(batch: Any, state: _BatchState) -> None:
    """Finish the launched round: accept walk, rollback, next pending token.
    Only called once the round's first token has been returned."""
    mtp, lm, drafter = state.mtp, state.mtp.lm, state.mtp.drafter
    p = state.pending
    cache = batch.prompt_cache
    block = p.block
    tic = time.perf_counter()
    target_list = p.targets.tolist()  # waits for the round
    mtp.seconds["wait"] += time.perf_counter() - tic
    draft_list = p.draft_list
    if draft_list is None:
        draft_list = p.drafts[0].tolist() if block > 1 else []

    if p.ss:  # (#167) [accept, rejected, bonus, plain]
        accept, rejected, bonus, plain = (int(t) for t in target_list)
        if p.limit < 1:
            accepted, new_tokens = 0, [plain]
        elif accept:
            accepted, new_tokens = 1, [int(draft_list[0]), bonus]
        else:
            accepted, new_tokens = 0, [rejected]
        if p.limit >= 1:
            with mtp.lock:
                mtp.counters["ss_rounds"] += 1
                mtp.counters["ss_accepted"] += accepted
    else:
        accepted = 0
        while accepted < p.limit and draft_list[accepted] == target_list[accepted]:
            accepted += 1
        new_tokens = draft_list[:accepted] + [int(target_list[accepted])]
    keep = accepted + 1
    if state.ss is not None and block == 2:
        drafter.accept_verified_tokens(
            p.h, p.drafts, accepted, new_tokens, _q_sampler(state), mx.int32,
            greedy=False,
        )  # fmt: skip
    else:
        drafter.accept_verified_tokens(
            p.h, p.drafts, accepted, new_tokens, _greedy, mx.int32, greedy=True
        )
    # Nothing below can fail: the round is committed from here on.
    state.pending = None
    if keep < block:
        _rollback(cache, block, keep)
        mtp.count("rollbacks")
    state.hidden = p.h[:, accepted : accepted + 1, :]
    state.round_h = p.h
    state.kept = keep
    state.queue.extend(
        (int(draft_list[j]), p.logprobs[j], True) for j in range(accepted)
    )
    if p.processors and accepted:
        batch._token_context[0].update_and_fetch(mx.array(draft_list[:accepted]))
    dtype = batch._next_tokens.dtype
    batch._current_tokens = mx.array([p.b], dtype=dtype)
    batch._current_logprobs = [p.b_logprobs]
    batch._next_tokens = mx.array([new_tokens[-1]], dtype=dtype)
    batch._next_logprobs = [p.logprobs[accepted]]

    with mtp.lock:
        c = mtp.counters
        c["emitted_tokens"] += keep
        if block > 1:
            c["rounds"] += 1
            c["drafted_tokens"] += block - 1
            c["accepted_tokens"] += accepted
            # Only a round that drafted proves the drafter works: a failure
            # is always followed by a draft-free bootstrap round.
            mtp.consecutive_errors = 0
        else:
            c["bootstrap_rounds"] += 1


def _complete_or_recover(batch: Any, state: _BatchState) -> None:
    """Complete the launched round. If that fails, its first token is already
    out, so undo the round and verify that token alone, without the drafter;
    an error there is a real engine error and propagates."""
    p = state.pending
    try:
        _complete(batch, state)
        return
    except Exception as e:  # noqa: BLE001 - decode must continue
        _note_failure(state.mtp, "round", e)
    state.pending = None
    _restore(batch.prompt_cache, p.snap)
    if p.ctx_size is not None:
        batch._token_context[0]._size = p.ctx_size
    state.hidden = None
    state.drafter_ready = True  # block 1 never touches the drafter...
    _launch(batch, state, p.processors, max_block=1, enqueue=False)
    q = state.pending
    state.pending = None
    token = int(q.targets.tolist()[0])
    dtype = batch._next_tokens.dtype
    batch._current_tokens = mx.array([q.b], dtype=dtype)
    batch._current_logprobs = [q.b_logprobs]
    batch._next_tokens = mx.array([token], dtype=dtype)
    batch._next_logprobs = [q.logprobs[0]]
    state.forget()  # ...and it restarts cold on the next round
    state.mtp.count("bootstrap_rounds")
