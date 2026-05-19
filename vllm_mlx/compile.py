# SPDX-License-Identifier: Apache-2.0
"""
Compile wrapper for MLX model forward passes.

Wraps a model's ``__call__`` with ``mx.compile()`` for fused Metal kernels.
Used when ``--compile`` flag is passed to ``vllm-mlx serve``.

Ported from upstream PR #270 (jackneil). Reference:
    https://github.com/waybarrios/vllm-mlx/pull/270
"""

import logging

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

_COMPILED_ATTR = "_vllm_mlx_compiled"


def apply_compile(model: nn.Module, enabled: bool = True) -> nn.Module:
    """Wrap model's forward pass with ``mx.compile`` for fused kernels.

    Args:
        model: The MLX model to compile.
        enabled: If False, return model unchanged (for easy toggling).

    Returns:
        The model with compiled ``__call__``, or original if disabled/already
        compiled. On compile failure, logs a warning and returns the original
        model — never raises.
    """
    if not enabled:
        return model

    if is_compiled(model):
        logger.debug("Model already compiled, skipping")
        return model

    try:
        original_call = model.__call__
        compiled_call = mx.compile(original_call, shapeless=True)
        model.__call__ = compiled_call
        setattr(model, _COMPILED_ATTR, True)
        logger.info("Model forward pass compiled with mx.compile(shapeless=True)")
        return model

    except Exception as e:
        logger.warning("mx.compile failed, using uncompiled model: %s", e)
        return model


def is_compiled(model: nn.Module) -> bool:
    """Return True if a model has been wrapped by ``apply_compile``."""
    return getattr(model, _COMPILED_ATTR, False)
