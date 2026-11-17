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

Enable with ``VLLM_MLX_TEXT_MTP_DRAFTER=<repo or path>``; block size (verified
positions per round) via ``VLLM_MLX_TEXT_MTP_BLOCK`` (default 2: one draft per round was
the robust optimum at every context length measured).
"""

from __future__ import annotations

import logging
import os
import threading
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

ENV_DRAFTER = "VLLM_MLX_TEXT_MTP_DRAFTER"
ENV_BLOCK = "VLLM_MLX_TEXT_MTP_BLOCK"
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


class TextMTP:
    """Per-scheduler drafter and counters; survives generator re-creation."""

    def __init__(self, path: str):
        self.path = path
        self.block = _block_size()
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
        }
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
            lm = getattr(model, "language_model", model)
            if not callable(getattr(lm, "model", None)):
                raise TypeError(f"{type(model).__name__} exposes no hidden-state model")
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
            self.bypass[key] += 1


class _BatchState:
    """Speculation state of one persistent GenerationBatch."""

    __slots__ = ("mtp", "uid", "hidden", "queue", "round_h", "kept", "drafter_ready")

    def __init__(self, mtp: TextMTP):
        self.mtp = mtp
        self.uid: Any = None
        self.hidden: Optional[mx.array] = None
        # verified tokens already in the cache, not yet returned by _step
        self.queue: Deque[Tuple[int, mx.array]] = deque()
        self.round_h: Optional[mx.array] = None
        self.kept = 0  # tokens of the last round the cache currently holds
        self.drafter_ready = False

    def forget(self) -> None:
        self.hidden = None
        self.round_h = None
        self.queue.clear()
        self.kept = 0
        self.drafter_ready = False


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
    _install_class_patches()
    batch._text_mtp = _BatchState(mtp)
    batch_gen.get_mtp_stats = mtp.stats
    logger.info("[text-mtp] armed on the text path: %s block=%d", path, mtp.block)


def _supports_recording(lm: Any) -> bool:
    from mlx_lm.models import qwen3_5 as q35

    layers = getattr(getattr(lm, "model", None), "layers", None) or []
    recurrent = [
        layer.linear_attn for layer in layers if getattr(layer, "is_linear", False)
    ]
    return bool(recurrent) and all(
        type(m) is q35.GatedDeltaNet and m.sharding_group is None for m in recurrent
    )


def _install_class_patches() -> None:
    """Idempotent. Every wrapper is a no-op for a batch without ``_text_mtp``
    and for a GatedDeltaNet call outside a recording verify."""
    global _patched
    with _patch_lock:
        if _patched:
            return
        from mlx_lm.generate import GenerationBatch
        from mlx_lm.models import qwen3_5 as q35

        _orig["step"] = GenerationBatch._step
        _orig["extract_cache"] = GenerationBatch.extract_cache
        _orig["filter"] = GenerationBatch.filter
        _orig["extend"] = GenerationBatch.extend
        _orig["gdn_call"] = q35.GatedDeltaNet.__call__

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
            return result

        def extend(self, batch):
            _sync(self)
            result = _orig["extend"](self, batch)
            state = getattr(self, "_text_mtp", None)
            if state is not None:
                state.forget()
            return result

        GenerationBatch._step = _step
        GenerationBatch.extract_cache = extract_cache
        GenerationBatch.filter = filter
        GenerationBatch.extend = extend
        q35.GatedDeltaNet.__call__ = _recording_gdn_call
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
    from mlx_lm.models import gated_delta as gd
    from mlx_lm.models import qwen3_5 as q35

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
    q, k = q35.normalize_qk(q, k, inv_scale=self.head_k_dim**-0.5, eps=1e-6)
    state = cache[1]
    outs, states = [], []
    for t in range(S):
        out_t, state = gd.gated_delta_update(
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
            out.append((list(c.cache), c.left_padding, c.lengths))
        else:
            out.append(getattr(c, "_idx", None))
    return out


def _restore(cache: List[Any], snap: List[Any]) -> None:
    for c, s in zip(cache, snap):
        if _is_recurrent(c):
            c.cache, c.left_padding, c.lengths = list(s[0]), s[1], s[2]
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
def _sync(batch: Any) -> None:
    """Roll the cache back to the last token ``_step`` returned."""
    state: Optional[_BatchState] = getattr(batch, "_text_mtp", None)
    if state is None or not state.queue:
        return
    ahead = len(state.queue)
    keep = state.kept - ahead
    _rollback(batch.prompt_cache, state.kept, keep)
    token, logprobs = state.queue[0]
    batch._next_tokens = mx.array([token], dtype=batch._next_tokens.dtype)
    batch._next_logprobs = [logprobs]
    if batch.logits_processors and batch.logits_processors[0]:
        batch._token_context[0]._size -= ahead
    state.hidden = state.round_h[:, keep - 1 : keep, :]
    state.queue.clear()
    state.kept = keep
    state.drafter_ready = False  # its own KV is now ahead; restart it cold
    state.mtp.count("sync_rollbacks")


def _bypass_reason(batch: Any, state: _BatchState) -> Optional[str]:
    mtp = state.mtp
    if mtp.disabled_reason is not None:
        return "disabled"
    if len(batch.uids) != 1:
        return "concurrent_batch"
    processors = (batch.logits_processors or [None])[0] or ()
    for p in processors:
        if not (
            callable(getattr(type(p), "speculation_plan", None))
            or getattr(type(p), "speculation_context_pure", False)
        ):
            return "logits_processors"
    if not mtp.ensure_loaded(batch.model):
        return "disabled"
    if not _cache_supported(batch.prompt_cache):
        return "cache_mismatch"
    return None


def _speculative_step(batch: Any, state: _BatchState):
    if not batch.uids:
        return _orig["step"](batch)
    if state.uid != batch.uids[0] or len(batch.uids) != 1:
        if state.queue:  # membership changed without a hook: resync first
            _sync(batch)
        state.forget()
        state.uid = batch.uids[0]

    if state.queue:
        token, logprobs = state.queue.popleft()
        batch.tokens[0].append(token)
        return [token], [logprobs]

    reason = _bypass_reason(batch, state)
    if reason is not None:
        state.mtp.count_bypass(reason)
        state.forget()
        return _orig["step"](batch)

    mtp = state.mtp
    snap = _snapshot(batch.prompt_cache)
    processors = batch.logits_processors[0] if batch.logits_processors else None
    ctx_size = batch._token_context[0]._size if processors else None
    try:
        return _round(batch, state, processors)
    except Exception as e:  # noqa: BLE001 - decode must continue
        _REC["active"] = False
        _restore(batch.prompt_cache, snap)
        if ctx_size is not None:
            batch._token_context[0]._size = ctx_size
        state.forget()
        mtp.count("errors")
        mtp.consecutive_errors += 1
        logger.warning("[text-mtp] round failed, plain step taken: %s", e)
        if (
            mtp.consecutive_errors >= DISABLE_AFTER_ERRORS
            and mtp.disabled_reason is None
        ):
            mtp.disabled_reason = f"{type(e).__name__}: {e}"
            logger.error(
                "[text-mtp] DISABLED for this process after %d consecutive "
                "failures; decoding continues without speculation. Last error: %s",
                mtp.consecutive_errors,
                mtp.disabled_reason,
                exc_info=True,
            )
        return _orig["step"](batch)


def _greedy(logits: mx.array) -> mx.array:
    return mx.argmax(logits, axis=-1)


def _head(lm: Any, hidden: mx.array) -> mx.array:
    if lm.args.tie_word_embeddings:
        return lm.model.embed_tokens.as_linear(hidden)
    return lm.lm_head(hidden)


def _round(batch: Any, state: _BatchState, processors: Optional[List[Any]]):
    mtp, lm, drafter = state.mtp, state.mtp.lm, state.mtp.drafter
    cache = batch.prompt_cache
    current_tokens = batch._next_tokens
    current_logprobs = batch._next_logprobs
    b = int(current_tokens[0].item())

    # Without the hidden state that predicted ``b`` there is nothing to draft
    # from: verify ``b`` alone (a plain step that also yields the hidden).
    block = mtp.block if state.hidden is not None else 1
    if not state.drafter_ready:
        # Cold start: the drafter's own KV begins at this position. Its
        # attention is relative, so only consistency matters.
        drafter.reset(lm)
        drafter.set_shared_kv(
            {}, kv_offset=len(batch.tokens[0]) + (0 if block > 1 else 1)
        )
        state.drafter_ready = True
    if block > 1:
        drafts = drafter.draft_block(
            b, state.hidden, None, block, _greedy, mx.int32, greedy=True
        ).astype(mx.int32)
        verify_in = mx.concatenate([mx.array([[b]], dtype=mx.int32), drafts], axis=1)
    else:
        drafts = mx.zeros((1, 0), dtype=mx.int32)
        verify_in = mx.array([[b]], dtype=mx.int32)

    _REC["active"] = block > 1
    try:
        h = lm.model(verify_in, cache)
    finally:
        _REC["active"] = False
    logits = _head(lm, h)[0]  # (block, vocab)

    draft_list: List[int] = drafts[0].tolist() if block > 1 else []
    limit = block - 1  # how many drafts may be accepted
    if processors:
        context = batch._token_context[0].update_and_fetch(mx.array([b]))
        first = logits[0:1]
        for processor in processors:
            processor_feeds_off = getattr(processor, "disable_feeds", None)
            if processor_feeds_off is not None:
                processor_feeds_off()
            first = processor(context, first)
        rows = [first]
        masks: List[Tuple[int, ...]] = []
        # Two kinds of processor get here (see _bypass_reason): one that can
        # plan its effect on drafted positions (thinking budget), and one
        # whose output depends only on the token context (DRY) and is simply
        # called with each drafted position's context.
        pure = [
            p for p in processors if getattr(type(p), "speculation_context_pure", False)
        ]
        for processor in processors:
            if processor in pure:
                continue
            plan = processor.speculation_plan(draft_list)
            limit = min(limit, len(plan))
            masks = (
                plan[:limit]
                if not masks
                else [m + p for m, p in zip(masks[:limit], plan[:limit])]
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
    sampler = (batch.samplers[0] if batch.samplers else None) or batch.fallback_sampler
    targets = mx.concatenate(
        [sampler(logprobs[j : j + 1]) for j in range(logprobs.shape[0])], axis=0
    )
    mx.eval(targets)
    target_list = targets.tolist()

    accepted = 0
    while accepted < limit and draft_list[accepted] == target_list[accepted]:
        accepted += 1
    new_tokens = draft_list[:accepted] + [int(target_list[accepted])]
    keep = accepted + 1
    if keep < block:
        _rollback(cache, block, keep)
        mtp.count("rollbacks")

    drafter.accept_verified_tokens(
        h, drafts, accepted, new_tokens, _greedy, mx.int32, greedy=True
    )
    state.hidden = h[:, accepted : accepted + 1, :]
    state.round_h = h
    state.kept = keep
    state.queue = deque((int(draft_list[j]), logprobs[j]) for j in range(accepted))
    if processors and accepted:
        batch._token_context[0].update_and_fetch(mx.array(draft_list[:accepted]))

    next_tokens = mx.array([new_tokens[-1]], dtype=current_tokens.dtype)
    mx.eval(
        state.hidden,
        next_tokens,
        [a for a in drafter.draft_eval_state() if a is not None],
    )
    # Nothing below can fail: the batch is only touched once the round is done.
    batch._current_tokens = current_tokens
    batch._current_logprobs = current_logprobs
    batch._next_tokens = next_tokens
    batch._next_logprobs = [logprobs[accepted]]
    batch.tokens[0].append(b)

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
    return [b], current_logprobs
