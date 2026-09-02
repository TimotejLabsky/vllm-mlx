# SPDX-License-Identifier: Apache-2.0
"""MoE gate/up projection fusion (fork patch #96).

At decode shapes (B=1, top-k experts) the expert MLP is dispatch-bound: two
back-to-back ``gather_qmm`` calls (gate_proj, up_proj) cost ~2 dispatches
where one would do. Concatenating the two quantized weight tensors along the
output-row axis turns them into ONE ``gather_qmm`` followed by a split —
measured on the M1 Ultra at Qwen3.6-35B-A3B production shapes
(2048→256×512, top-8, 4-bit g64): 29–50µs saved per MoE layer, 1.2–2.0ms
per decode step over 40 layers (~10–15% of an ~11.5ms step). Upstream asked
for exactly this in mlx-lm #956 (+8.6% measured there); until it lands, the
fork fuses at load time.

Correctness: quantized rows are independent — each output row's packed
weights/scales/biases concatenate unchanged, and the fused matmul computes
the same per-row dot products. T=0 output is expected byte-identical
(asserted in tests; the live A/B gate re-checks).

Scope and safety:
- Env-gated ``VLLM_MLX_MOE_GATEUP_FUSION=1`` (default OFF — arm per route
  after the live gate, the REPDETECT pattern).
- Only fuses ``SwitchGLU`` whose gate/up are quantized with IDENTICAL
  shape/group_size/bits/mode and matching bias-ness; anything unexpected is
  skipped, counted, and the model runs unfused (fail-open to the stock
  path, never to a broken one).
- Text models only (the MLLM path loads through mlx-vlm's own classes).
- The mlx-lm import is guarded and matched by class NAME as well, so a
  future mlx-lm restructuring degrades to a no-op, not an ImportError
  (pin-ceiling discipline, see PATCHES.md #78/#81).
"""

from __future__ import annotations

import copy
import logging
import os

import mlx.core as mx

logger = logging.getLogger(__name__)


def fusion_enabled() -> bool:
    return os.environ.get("VLLM_MLX_MOE_GATEUP_FUSION", "").lower() in (
        "1",
        "true",
        "yes",
    )


def _switch_glu_class():
    try:
        from mlx_lm.models.switch_layers import SwitchGLU

        return SwitchGLU
    except Exception:
        return None


def _is_quantized_switch_linear(m) -> bool:
    return (
        hasattr(m, "weight")
        and hasattr(m, "scales")
        and hasattr(m, "group_size")
        and hasattr(m, "bits")
    )


def _fusable(gate, up) -> bool:
    if not (_is_quantized_switch_linear(gate) and _is_quantized_switch_linear(up)):
        return False
    if gate.weight.shape != up.weight.shape:
        return False
    for attr in ("group_size", "bits"):
        if getattr(gate, attr) != getattr(up, attr):
            return False
    if getattr(gate, "mode", "affine") != getattr(up, "mode", "affine"):
        return False
    # bias-ness must match on both the quantization biases and a linear bias
    if hasattr(gate, "biases") != hasattr(up, "biases"):
        return False
    if ("bias" in gate) != ("bias" in up):
        return False
    return True


class FusedSwitchGLU:
    """Drop-in replacement for ``SwitchGLU`` with gate/up fused.

    Layout is ``[up_rows; gate_rows]`` so the split maps directly onto the
    ``activation(x_up, x_gate)`` argument order. down_proj and activation
    are the ORIGINAL modules (shared by reference — weights are immutable).

    Deliberately not an ``nn.Module``: the model is already loaded and
    quantized; this object only needs to be callable and to keep its arrays
    referenced. Keeping it out of the module tree also keeps
    ``model.parameters()``/sanitize flows exactly as upstream built them.
    """

    def __init__(self, glu):
        gate, up = glu.gate_proj, glu.up_proj
        fused = copy.copy(up)
        fused.weight = mx.concatenate([up.weight, gate.weight], axis=1)
        fused.scales = mx.concatenate([up.scales, gate.scales], axis=1)
        if getattr(up, "biases", None) is not None:
            fused.biases = mx.concatenate([up.biases, gate.biases], axis=1)
        if "bias" in up:
            fused.bias = mx.concatenate([up["bias"], gate["bias"]], axis=1)
        mx.eval(fused.weight, fused.scales)
        self.gate_up_proj = fused
        self.down_proj = glu.down_proj
        self.activation = glu.activation
        self.training = False

    def __call__(self, x, indices) -> mx.array:
        from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        gu = self.gate_up_proj(x, idx, sorted_indices=do_sort)
        x_up, x_gate = mx.split(gu, 2, axis=-1)
        x = self.down_proj(
            self.activation(x_up, x_gate),
            idx,
            sorted_indices=do_sort,
        )
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)


def _is_target(m, glu_cls) -> bool:
    if glu_cls is not None and isinstance(m, glu_cls):
        return True
    # Name match covers subclasses/reloads across mlx-lm restructurings.
    return type(m).__name__ == "SwitchGLU" and hasattr(m, "gate_proj")


def _try_fuse(glu):
    try:
        if not _fusable(glu.gate_proj, glu.up_proj):
            return None
        return FusedSwitchGLU(glu)
    except Exception:
        logger.debug("[moe-fusion] fuse failed for a module", exc_info=True)
        return None


def fuse_moe_gate_up(model) -> int:
    """Replace every fusable ``SwitchGLU`` under ``model``. Returns the
    number fused. Never raises — any failure leaves the model unfused."""
    glu_cls = _switch_glu_class()
    fused = 0
    skipped = 0
    try:
        stack = [model]
        seen = set()
        while stack:
            mod = stack.pop()
            if id(mod) in seen:
                continue
            seen.add(id(mod))
            children = getattr(mod, "children", None)
            if not callable(children):
                continue
            for name, child in list(children().items()):
                if isinstance(child, (list, tuple)):
                    for i, item in enumerate(child):
                        if _is_target(item, glu_cls):
                            repl = _try_fuse(item)
                            if repl is not None:
                                getattr(mod, name)[i] = repl
                                fused += 1
                            else:
                                skipped += 1
                        else:
                            stack.append(item)
                elif _is_target(child, glu_cls):
                    repl = _try_fuse(child)
                    if repl is not None:
                        setattr(mod, name, repl)
                        fused += 1
                    else:
                        skipped += 1
                else:
                    stack.append(child)
    except Exception:
        logger.warning("[moe-fusion] walk failed; model left unfused", exc_info=True)
        return fused
    if fused or skipped:
        logger.info(
            "[moe-fusion] fused %d SwitchGLU module(s), skipped %d "
            "(VLLM_MLX_MOE_GATEUP_FUSION)",
            fused,
            skipped,
        )
    return fused


def maybe_fuse(model, model_name: str = "") -> int:
    """Entry point for the engines: fuse iff the env gate is armed."""
    if not fusion_enabled():
        return 0
    n = fuse_moe_gate_up(model)
    if n:
        logger.info("[moe-fusion] %s: %d expert MLP(s) fused", model_name, n)
    return n
