# SPDX-License-Identifier: Apache-2.0
"""Construct an mlx_lm TextModel from mlx_vlm-loaded model weights.

When mlx_vlm loads a model, it strips MTP weights in sanitize().
This module builds a parallel mlx_lm TextModel that:
1. Shares backbone + lm_head weights with the vlm model (zero-copy)
2. Loads MTP weights from safetensors on disk
3. Provides full mlx_lm API: return_hidden, n_confirmed, mtp_forward, make_mtp_cache
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.utils

logger = logging.getLogger(__name__)


def _import_text_model_classes(model_type: str):
    if model_type == "gemma4_text":
        from mlx_lm.models.gemma4_text import Model, ModelArgs

        return Model, ModelArgs

    # qwen3_5.TextModel and TextModelArgs handle both dense and MoE natively
    # (MTPDecoderLayer auto-selects SparseMoeBlock when args.num_experts > 0).
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

    return TextModel, TextModelArgs


def build_text_model(vlm_model: Any, model_path: str | Path) -> Any | None:
    """Build an mlx_lm TextModel from a vlm-loaded model's weights.

    Args:
        vlm_model: The mlx_vlm-loaded model (has .language_model attribute)
        model_path: Local directory OR HuggingFace repo ID. Repo IDs are
            resolved via the local HF cache.

    Returns:
        mlx_lm TextModel with MTP support, or None on failure.

    Bug history: the old code only handled local directory paths. When
    called with an HF repo ID (what ``SimpleEngine.start()`` passes on
    the MLLM text-routing setup), the directory check failed and this
    returned None. The caller set ``_text_model = None``, which made
    ``stream_chat`` routing fall back to the MLLM media path — which has
    no system-KV snapshot cache. Result: text-only requests to a true
    VLM (Qwen3.6-27B-4bit dense, etc.) paid full cold prefill on every
    turn, with ``cache_hits=0`` and ``cache_misses=0`` because the cache
    code path never executed. Fix resolves the repo ID via the HF cache
    so the function actually finds ``config.json``.
    """
    if vlm_model is None:
        return None

    model_path = _resolve_model_path(model_path)
    if model_path is None:
        return None

    try:
        config = json.loads((model_path / "config.json").read_text())
        text_config = config.get("text_config", config)

        model_type = str(
            text_config.get("model_type") or config.get("model_type") or ""
        )
        if model_type.startswith("qwen3_5") or model_type == "":
            # Qwen3.5/3.6 family — text_config carries model_type
            # "qwen3_5_text" / "qwen3_5_moe_text" (the top-level config says
            # "qwen3_5"/"qwen3_5_moe"), so match by PREFIX: an exact-set
            # check here once silently broke text routing (and with it the
            # whole system-KV cache) for every Qwen MLLM in the lineup.
            # Import from qwen3_5 — TextModel and TextModelArgs handle both
            # dense and MoE natively (MTPDecoderLayer auto-selects
            # SparseMoeBlock when args.num_experts > 0). qwen3_5_moe.py does
            # NOT export these.
            from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

            # Build args with proper __post_init__ (handles
            # partial_rotary_factor, rope_scaling, head_dim derivation)
            args = TextModelArgs.from_dict(text_config)
            text_model = TextModel(args)
        else:
            # Generic path: build the text model from the mlx_lm class for
            # this model_type (e.g. gemma4_text -> mlx_lm.models.gemma4_text).
            # Previously this function fed every config into the qwen3_5
            # TextModelArgs, which crashed on gemma-4 ("float division by
            # zero") — and the caller's None fallback silently routed text
            # requests through the cacheless mlx_vlm path (no system-KV).
            # Fallback chain: exact module; then the name with a trailing
            # "_text" stripped (text-config model_types often add the suffix
            # while the mlx_lm module is named after the family).
            import importlib

            mod = None
            candidates = [model_type]
            if model_type.endswith("_text"):
                candidates.append(model_type[: -len("_text")])
            for cand in candidates:
                try:
                    mod = importlib.import_module(f"mlx_lm.models.{cand}")
                    break
                except ImportError:
                    continue
            if mod is None:
                logger.warning(
                    "No mlx_lm model class for text model_type %r "
                    "(tried %s); MLLM text routing unavailable",
                    model_type,
                    candidates,
                )
                return None
            model_cls = getattr(mod, "TextModel", None) or mod.Model
            args_cls = getattr(mod, "TextModelArgs", None) or mod.ModelArgs
            args = args_cls.from_dict(text_config)
            text_model = model_cls(args)

        # Collect all weights first: backbone from vlm + MTP from safetensors
        vlm_lm = vlm_model.language_model
        vlm_weights = mlx.utils.tree_flatten(vlm_lm.parameters())
        mtp_weights = _load_mtp_weights(model_path)

        all_weight_names = set(name for name, _ in vlm_weights)
        all_weight_names.update(name for name, _ in mtp_weights)

        # Quantize the TextModel skeleton to match source weights.
        # Use a predicate that only quantizes layers that have .scales in source.
        # This prevents quantizing layers like mtp.fc which are BF16.
        # Mixed-precision quants (gemma-4, OptiQ-style) carry PER-LAYER
        # overrides in the quantization config — returning the override dict
        # from the predicate makes nn.quantize use it (mlx-lm loader
        # convention). Override keys use the VLM namespace
        # (``language_model.<path>``), our skeleton paths don't — check both.
        quantization = text_config.get("quantization", config.get("quantization", None))
        if quantization is not None:
            # Per-layer quantization overrides (e.g. 8-bit MoE gates over a 4-bit
            # body) may be keyed with the source checkpoint's "language_model."
            # wrapper prefix, which the extracted TextModel doesn't carry. Index
            # them by suffix so the prefix-stripped module path resolves to the
            # correct bits/group_size; without this the override is ignored and the
            # layer is quantized at the global width, producing a quantized_matmul
            # shape mismatch at decode.
            per_layer_overrides = {
                k: v for k, v in quantization.items() if isinstance(v, dict)
            }

            def _class_predicate(path, module):
                for key in (path, f"language_model.{path}"):
                    override = quantization.get(key)
                    if isinstance(override, dict):
                        return override
                if not hasattr(module, "to_quantized"):
                    return False
                if f"{path}.scales" not in all_weight_names:
                    return False
                if path in quantization:
                    return quantization[path]
                for key, override in per_layer_overrides.items():
                    if key.endswith("." + path):
                        return override
                return True

            nn.quantize(
                text_model,
                group_size=quantization.get("group_size", 64),
                bits=quantization.get("bits", 8),
                class_predicate=_class_predicate,
            )

        # Fail closed before serving a half-loaded model: strict=False below
        # silently ignores name mismatches in BOTH directions, so a namespace
        # divergence between the vlm's language_model and the mlx_lm class
        # would produce a model that loads fine and generates garbage. Require
        # the vlm/MTP weights to cover (nearly) all skeleton parameters —
        # MTP-only gaps are expected and handled by the second load below.
        skeleton_names = {
            name for name, _ in mlx.utils.tree_flatten(text_model.parameters())
        }
        coverage = (
            len(skeleton_names & all_weight_names) / len(skeleton_names)
            if skeleton_names
            else 0.0
        )
        if coverage < 0.9:
            logger.warning(
                "TextModel weight-name coverage only %.0f%% for model_type %r "
                "(vlm namespace mismatch); refusing half-loaded text model",
                coverage * 100,
                model_type,
            )
            return None

        # Transfer backbone + lm_head weights from vlm language_model (zero-copy).
        # strict=False because TextModel has MTP params that vlm doesn't have yet.
        text_model.load_weights(vlm_weights, strict=False)

        logger.info(
            "Transferred %d weight arrays from vlm language_model", len(vlm_weights)
        )

        # Load MTP weights from safetensors
        if mtp_weights:
            text_model.load_weights(mtp_weights, strict=False)
            logger.info("Loaded %d MTP weights from safetensors", len(mtp_weights))
        else:
            logger.warning("No MTP weights found in %s", model_path.name)

        # Inject MTP if TextModel doesn't have native MTP support.
        # mlx_lm's qwen3_5.TextModel strips MTP weights in sanitize(),
        # so we inject MTP module + methods at runtime.
        if not hasattr(text_model, "mtp") or text_model.mtp is None:
            num_mtp = text_config.get("mtp_num_hidden_layers", 0)
            if num_mtp == 0:
                num_mtp = text_config.get("num_nextn_predict_layers", 0)
            if num_mtp > 0:
                from .patches.qwen3_5_mtp import inject_mtp_support

                inject_mtp_support(text_model, model_path, config)

        # Put the derived TextModel in eval mode. mlx_lm.load / mlx_vlm.load
        # both eval() their models; this freshly-built TextModel defaults to
        # training=True. Hybrid gated-delta layers (Qwen3.5/3.6 linear
        # attention) select their compute path with `use_kernel = not
        # self.training`, so in training mode every gated-delta forward falls
        # to the slow Python recurrence instead of the Metal kernel — a large,
        # context-scaling prefill penalty plus a decode hit on this VLM->text
        # path (upstream measured ~6x prefill / +15% decode on Qwen3.6-35B-A3B
        # 4-bit). Placed AFTER inject_mtp_support so it recurses into the mtp
        # submodule, and BEFORE the warmup forward so the warmup materializes
        # lazy buffers on the actual (kernel) compute path. Numerically
        # identical output; only the compute path changes.
        # Cherry-picked from upstream 527f457 (#606).
        text_model.train(False)

        # Materialize every lazy init-time array — parameters AND ad-hoc
        # module buffers (rope frequency tables etc.) that parameters()
        # cannot enumerate — by running a one-token forward HERE, on the
        # build thread. Lazy arrays pin to the stream of the thread that
        # created them (thread-local since mlx_lm's generation_stream
        # rework); without this warmup the first real forward happens on a
        # generation worker thread and dies with "There is no
        # Stream(gpu, N) in current thread" (observed on gemma-4, whose
        # rope buffers are built at init — Qwen3.5 computes them per
        # forward and never hit this).
        try:
            from mlx_lm.models.cache import make_prompt_cache

            _warm_cache = make_prompt_cache(text_model)
            mx.eval(text_model(mx.array([[1]]), cache=_warm_cache))
            del _warm_cache
        except Exception as e:
            logger.warning(
                "TextModel warmup forward failed (%s); refusing text model "
                "(first worker-thread forward would crash)",
                e,
            )
            return None

        if hasattr(text_model, "mtp") and text_model.mtp is not None:
            mx.eval(text_model.mtp.parameters())
            num_mtp = text_config.get(
                "mtp_num_hidden_layers",
                text_config.get("num_nextn_predict_layers", 0),
            )
            logger.info("TextModel built with MTP support (%d layers)", num_mtp)
        else:
            logger.info("TextModel built without MTP")

        # NOTE: eval mode is set above (before the warmup forward) so the warmup
        # runs on the kernel compute path. Upstream #606 (527f457) also sets it
        # here just before return, but that is now redundant with the earlier
        # call and would run after the warmup — removed on the a48c86c rebase.

        # Realize every array the model holds before it leaves the build
        # thread — including underscore-private module attributes such as
        # RoPE._freqs, which parameters() excludes. MLX lazy graphs are tagged
        # to the stream of the thread that recorded them; a lazy array
        # surviving into generation dies with "There is no Stream(gpu, N) in
        # current thread" the moment a worker on another thread evaluates it
        # (Gemma 4: the scaled-RoPE _freqs of the first full_attention layer).
        if hasattr(text_model, "modules"):
            mx.eval(
                [
                    v
                    for module in text_model.modules()
                    for v in module.values()
                    if isinstance(v, mx.array)
                ]
            )

        return text_model

    except ImportError as e:
        logger.error("Cannot import mlx_lm TextModel (need PR #990): %s", e)
        return None
    except Exception as e:
        logger.error("Failed to build TextModel from vlm: %s", e)
        return None


def _resolve_model_path(model_path: str | Path) -> Path | None:
    """Resolve an HF repo ID or local dir to the snapshot directory holding
    config.json, or None."""
    if (
        isinstance(model_path, str)
        and "/" in model_path
        and not Path(model_path).is_dir()
    ):
        try:
            from huggingface_hub import try_to_load_from_cache  # type: ignore

            cached = try_to_load_from_cache(
                repo_id=model_path, filename="config.json"
            )
            if isinstance(cached, str):
                cdir = Path(cached).parent
                if cdir.is_dir():
                    model_path = cdir
        except Exception:
            pass
    path = Path(model_path) if model_path else None
    if path is None or not (path / "config.json").exists():
        return None
    return path


def wrap_tokenizer_with_eos(tokenizer: Any, model_path: str | Path) -> Any:
    """Wrap a bare HF tokenizer in mlx_lm's TokenizerWrapper carrying the
    model's FULL eos set from ``generation_config.json``.

    The MLLM text route hands mlx_lm's ``stream_generate`` the processor's
    raw tokenizer; mlx_lm then wraps it with a single ``eos_token_id``.
    Models whose generation_config declares SEVERAL terminators (gemma-4:
    ``[1, 106, 50]`` — end-of-text, ``<turn|>``, end-of-channel) never stop
    on the extra ones via that path: turn markers leak into content and
    generation runs to max_tokens. Pre-wrapping with the full set fixes the
    stop behavior for every family (the old code special-cased Qwen3.5's
    ``<|im_end|>`` only).
    """
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    if isinstance(tokenizer, TokenizerWrapper):
        return tokenizer
    eos_ids: set[int] = set()
    base_eos = getattr(tokenizer, "eos_token_id", None)
    if base_eos is not None:
        eos_ids.add(int(base_eos))
    path = _resolve_model_path(model_path)
    if path is not None:
        gen_cfg = path / "generation_config.json"
        if gen_cfg.exists():
            try:
                declared = json.loads(gen_cfg.read_text()).get("eos_token_id")
                if isinstance(declared, int):
                    eos_ids.add(declared)
                elif isinstance(declared, list):
                    eos_ids.update(int(x) for x in declared)
            except Exception:
                logger.debug("generation_config eos read failed", exc_info=True)
    return TokenizerWrapper(tokenizer, eos_token_ids=eos_ids or None)


def _load_mtp_weights(model_path: Path) -> list[tuple[str, mx.array]]:
    """Load MTP weights from safetensors, stripping the language_model. prefix.

    mlx_vlm's sanitize() strips mtp.* keys during model loading,
    but the weights are still on disk in the safetensors files.
    """
    index_file = model_path / "model.safetensors.index.json"
    if not index_file.exists():
        return []

    index = json.loads(index_file.read_text())
    weight_map = index.get("weight_map", {})

    # Find MTP keys and their shard files
    mtp_keys: dict[str, tuple[str, str]] = {}
    for key, shard in weight_map.items():
        if ".mtp." in key:
            # Strip "language_model." prefix to match mlx_lm namespace
            clean = (
                key.replace("language_model.", "", 1)
                if key.startswith("language_model.")
                else key
            )
            mtp_keys[key] = (clean, shard)

    if not mtp_keys:
        return []

    # Group by shard to minimize I/O
    shards: dict[str, list[tuple[str, str]]] = {}
    for orig, (clean, shard) in mtp_keys.items():
        shards.setdefault(shard, []).append((orig, clean))

    weights = []
    for shard_file, key_pairs in shards.items():
        shard_path = model_path / shard_file
        if not shard_path.exists():
            logger.warning("MTP shard not found: %s", shard_file)
            continue
        shard_data = mx.load(str(shard_path))
        for orig, clean in key_pairs:
            if orig in shard_data:
                weights.append((clean, shard_data[orig]))

    return weights
