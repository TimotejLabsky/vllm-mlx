# SPDX-License-Identifier: Apache-2.0
"""(fork #168) Qwen3-VL-MoE decode without a host sync per layer.

mlx-vlm's ``qwen3_vl_moe`` attention slices the mask before the cache
update with ``kv_seq_len.max().item()``. Under BatchKVCache ``cache.offset``
is an ``mx.array``, so every attention layer of every decoded token waits
for the GPU: 48 syncs per token on Qwen3-VL-30B-A3B (census 2026-11-18), the
pipelining of ``MLLMBatchGenerator._next`` lost. mlx-vlm's dense ``qwen3_vl``
attention already slices after the update, to the fetched keys' length -
no sync - and so does this copy. The mask covers exactly the fetched keys on
every cache the fork uses, so the slice is a no-op there either way.

Applied only while the installed mlx-vlm still has the syncing line; if
upstream changes that method, this patch steps aside (and logs it).
"""

from __future__ import annotations

import importlib
import inspect
import logging
from typing import Any, Optional

import mlx.core as mx

logger = logging.getLogger(__name__)

_SYNC_LINE = "kv_seq_len = kv_seq_len.max().item()"


def patch_qwen3_vl_moe_attention() -> bool:
    """True when the sync-free attention is in place."""
    try:
        lang = importlib.import_module("mlx_vlm.models.qwen3_vl_moe.language")
    except ImportError:
        return False
    attn = getattr(lang, "Attention", None)
    if attn is None:
        return False
    if getattr(attn, "_fork_168_patched", False):
        return True
    try:
        source = inspect.getsource(attn.__call__)
    except (OSError, TypeError):
        source = ""
    if _SYNC_LINE not in source:
        logger.info(
            "[qwen3_vl_moe patch] upstream attention changed (no per-layer "
            "sync line); leaving it as is"
        )
        return False
    rope = lang.apply_multimodal_rotary_pos_emb
    sdpa = lang.scaled_dot_product_attention

    def _call(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_ids: Optional[mx.array] = None,
        position_embeddings: Optional[tuple] = None,
    ) -> mx.array:
        B, L, _D = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries = self.q_norm(
            queries.reshape(B, L, self.n_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            keys.reshape(B, L, self.n_kv_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        if position_ids is None:
            offset = cache.offset if cache is not None else 0
            position_ids = mx.arange(offset, offset + L)
            position_ids = mx.expand_dims(position_ids, axis=0)
            position_ids = mx.tile(position_ids, (3, 1, 1))
        if position_embeddings is None:
            queries, keys = self.rotary_emb.apply_rotary(
                queries, keys, position_ids, unsqueeze_dim=1
            )
        else:
            cos, sin = position_embeddings
            queries, keys = rope(queries, keys, cos, sin)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
        if isinstance(mask, mx.array):
            mask = mask[..., : keys.shape[-2]]  # host int: no sync
        output = sdpa(queries, keys, values, cache, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)

    attn._fork_168_original_call = attn.__call__
    attn.__call__ = _call
    attn._fork_168_patched = True
    logger.info("[qwen3_vl_moe patch] attention: no host sync per layer (#168)")
    return True
