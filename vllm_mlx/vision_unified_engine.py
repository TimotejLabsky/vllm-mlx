# SPDX-License-Identifier: Apache-2.0
"""(#181) ``--vision-unified``: one engine, one scheduler, text AND images.

The checkpoint loads once through mlx-vlm (vision tower + processor). The
model the batched TEXT scheduler runs is
:func:`~vllm_mlx.text_model_from_vlm.build_text_model`'s zero-copy mlx-lm
``TextModel``, wrapped in the same outer ``qwen3_5.Model`` class a
``--text-only`` load produces, with the tokenizer a ``--text-only`` load
produces. A text request therefore runs the exact ``--text-only`` path.

A request with images:

1. renders its chat template with ``{"type": "image"}`` parts (the
   tokenizer's template emits the vision placeholders) and runs mlx-vlm's
   processor on it (``prepare_inputs``): placeholder-expanded token ids,
   pixel values, image grids. This happens before admission, so the prompt
   ceiling and the admission gates see the real token count, image tokens
   included;
2. enters the same ``Scheduler`` as text, as a ``Request`` whose
   ``images`` carries one :class:`MediaInputs`;
3. at schedule time (on the generation worker) is vision-encoded and
   prefilled up to the end of its last image with merged embeddings and
   MRoPE positions (:mod:`vllm_mlx.vision_unified`), then inserted into the
   ``BatchGenerator`` like a prefix-cache hit. Trailing text and decode are
   ordinary batched work.

Nothing here runs unless the route was started with ``--vision-unified``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import numpy as np

from . import vision_unified as vu
from .vision_unified import MediaInputs, MediaRows, _np
from .engine.base import GenerationOutput, MediaNotSupported
from .engine.batched import BatchedEngine

logger = logging.getLogger(__name__)

MEDIA_PART_TYPES = frozenset(
    {"image_url", "image", "video", "video_url", "audio", "audio_url"}
)


class VisionUnifiedError(RuntimeError):
    """The route cannot start as configured. Never degraded silently."""


# ---------------------------------------------------------------------------
# Server seams
# ---------------------------------------------------------------------------


def _part_type(part: Any) -> str:
    if isinstance(part, dict):
        return str(part.get("type", "") or "")
    return str(getattr(part, "type", "") or "")


def messages_have_media(messages: Any) -> bool:
    for msg in messages or ():
        content = (
            msg.get("content")
            if isinstance(msg, dict)
            else getattr(msg, "content", None)
        )
        if isinstance(content, list):
            for part in content:
                if _part_type(part) in MEDIA_PART_TYPES:
                    return True
    return False


def is_vision_unified(engine: Any) -> bool:
    return getattr(engine, "vision_unified", False) is True


def request_is_mllm(engine: Any, request_messages: Any) -> bool:
    """Whether the server prepares THIS request MLLM-style (media parts kept
    in the messages). ``engine.is_mllm`` on every other route, unchanged."""
    if is_vision_unified(engine):
        return messages_have_media(request_messages)
    return bool(getattr(engine, "is_mllm", False))


def engine_serves_media(engine: Any) -> bool:
    """The #65 media guard's question: may this route answer media at all?"""
    return bool(getattr(engine, "is_mllm", False)) or is_vision_unified(engine)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def check_vision_unified_config(
    *,
    use_batching: bool,
    force_text_only: bool = False,
    force_mllm: bool = False,
    auto_unload_idle_seconds: float = 0.0,
    lazy_load_model: bool = False,
    mllm_draft_model: Optional[str] = None,
    enable_mtp: bool = False,
    compile: bool = False,
    models_config: Optional[str] = None,
    max_kv_size: int = 0,
    chunked_prefill_tokens: int = 0,
) -> None:
    """Raise :class:`VisionUnifiedError` naming every reason the flag cannot
    apply. Called by the CLI before any download and again in load_model."""
    problems = []
    if not use_batching:
        problems.append("requires --continuous-batching")
    if force_text_only:
        problems.append("is mutually exclusive with --text-only")
    if force_mllm:
        problems.append("is mutually exclusive with --mllm")
    if auto_unload_idle_seconds and auto_unload_idle_seconds > 0:
        problems.append("does not support --auto-unload-idle-seconds")
    if lazy_load_model:
        problems.append("does not support --lazy-load-model")
    if mllm_draft_model:
        problems.append("does not support an MLLM draft model")
    if enable_mtp:
        problems.append(
            "does not support --enable-mtp (the text drafter env "
            "VLLM_MLX_TEXT_MTP_DRAFTER works for text requests)"
        )
    if compile:
        problems.append("does not support --compile")
    if models_config:
        problems.append("serves one model; it does not support --models-config")
    if max_kv_size and max_kv_size > 0:
        problems.append(
            "does not support --max-kv-size (rotating KV caches do not carry "
            "the per-row RoPE delta)"
        )
    if chunked_prefill_tokens and chunked_prefill_tokens > 0:
        problems.append("does not support the legacy --chunked-prefill-tokens path")
    from .moe_fusion import fusion_enabled

    if fusion_enabled():
        problems.append(
            "cannot run with VLLM_MLX_MOE_GATEUP_FUSION armed (the fused "
            "gate/up tensor would be a second copy of the expert weights next "
            "to the vision model's)"
        )
    if problems:
        raise VisionUnifiedError("--vision-unified " + "; ".join(problems))


def _local_model_dir(model_path: str) -> Path:
    from .text_model_from_vlm import _resolve_model_path

    local = _resolve_model_path(model_path)
    if local is None:
        try:
            from mlx_lm.utils import _download

            local = Path(_download(model_path))
        except Exception as e:  # noqa: BLE001
            raise VisionUnifiedError(
                f"--vision-unified: cannot resolve a local snapshot for {model_path!r}"
            ) from e
    return Path(local)


def read_checkpoint_config(model_path: str) -> dict:
    try:
        return json.loads((_local_model_dir(model_path) / "config.json").read_text())
    except VisionUnifiedError:
        raise
    except Exception as e:  # noqa: BLE001
        raise VisionUnifiedError(
            f"--vision-unified: cannot read config.json for {model_path!r}: {e}"
        ) from e


def checkpoint_support(config: dict) -> tuple[str, list[int]]:
    """``(text model_type, mrope_section)`` for a supported checkpoint
    config; raises :class:`VisionUnifiedError` naming what is missing."""
    if not isinstance(config.get("vision_config"), dict):
        raise VisionUnifiedError(
            "--vision-unified needs a checkpoint with a vision tower "
            "(config.json has no vision_config)"
        )
    text_config = config.get("text_config", config)
    model_type = str(text_config.get("model_type") or config.get("model_type") or "")
    base = model_type[: -len("_text")] if model_type.endswith("_text") else model_type
    if base not in vu.SUPPORTED_MODEL_TYPES:
        raise VisionUnifiedError(
            f"--vision-unified supports text model_type in "
            f"{vu.SUPPORTED_MODEL_TYPES}; this checkpoint is {model_type!r}"
        )
    rope = text_config.get("rope_parameters") or text_config.get("rope_scaling") or {}
    section = rope.get("mrope_section")
    if not (isinstance(section, list) and len(section) == 3):
        raise VisionUnifiedError(
            f"--vision-unified: text_config has no 3-axis mrope_section ({rope!r})"
        )
    if rope.get("mrope_interleaved") is False:
        raise VisionUnifiedError(
            "--vision-unified supports interleaved MRoPE only "
            "(mrope_interleaved is false)"
        )
    return model_type, [int(s) for s in section]


def load_text_tokenizer(model_path: str, model_name: str, trust_remote_code: bool):
    """The tokenizer a ``--text-only`` route gets for this checkpoint: the
    call ``utils.tokenizer._load_strict_false`` makes for a VLM-shaped
    checkpoint, then the same template installs (custom template,
    ``VLLM_MLX_CHAT_TEMPLATE_FILE``). A separate object from the processor's
    tokenizer."""
    from mlx_lm.utils import load_config, load_tokenizer

    from .utils.tokenizer import (
        _install_chat_template_file,
        _install_custom_chat_template,
    )

    local = _local_model_dir(model_path)
    tokenizer_config: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if "qwen3" in model_name.lower() or "Qwen3" in model_name:
        tokenizer_config["eos_token"] = "<|im_end|>"
    config = load_config(local)
    tokenizer = load_tokenizer(
        local, tokenizer_config, eos_token_ids=config.get("eos_token_id", None)
    )
    tokenizer = _install_custom_chat_template(str(local), tokenizer)
    return _install_chat_template_file(tokenizer)


def wrap_as_text_only_model(text_model: Any, config: dict) -> Any:
    """Put ``build_text_model``'s inner ``TextModel`` in the outer
    ``mlx_lm.models.qwen3_5.Model`` class a ``--text-only`` load builds, so
    every fork hook that walks the model tree (text drafter, MoE kernels)
    sees the same shape. No second set of weights is created."""
    import mlx.nn as nn
    from mlx_lm.models import qwen3_5

    args = qwen3_5.ModelArgs.from_dict(config)
    outer = qwen3_5.Model.__new__(qwen3_5.Model)
    nn.Module.__init__(outer)
    outer.args = args
    outer.model_type = args.model_type
    outer.language_model = text_model
    outer.train(False)
    return outer


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class VisionUnifiedEngine(BatchedEngine):
    """A ``--text-only`` BatchedEngine whose scheduler also takes images."""

    vision_unified = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["force_text_only"] = True
        kwargs.pop("force_mllm", None)
        super().__init__(*args, **kwargs)
        self._vlm_model: Any = None
        self._vlm_processor: Any = None
        self._mrope_section: list = []
        self._text_model_type = ""
        self._media_rows: Optional[MediaRows] = None
        self._pending_media: dict[int, MediaInputs] = {}

    # -- load ----------------------------------------------------------------
    def _prepare_llm_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        from mlx_vlm import load as vlm_load

        from .lazy_realize import realize_module_arrays
        from .text_model_from_vlm import build_text_model

        local = _local_model_dir(self._model_path)
        config = read_checkpoint_config(str(local))
        model_type, section = checkpoint_support(config)
        vlm_model, processor = vlm_load(
            str(local), trust_remote_code=self._trust_remote_code
        )
        if not callable(getattr(vlm_model, "get_input_embeddings", None)):
            raise VisionUnifiedError(
                "--vision-unified: the mlx-vlm model has no get_input_embeddings"
            )
        text_model = build_text_model(vlm_model, str(local), enable_mtp=False)
        if text_model is None:
            raise VisionUnifiedError(
                f"--vision-unified: no text model can be built from this "
                f"checkpoint (model_type={model_type!r}; see build_text_model's "
                f"log line above). Serve it with --text-only, or as an MLLM route."
            )
        model = wrap_as_text_only_model(text_model, config)
        vu.install_mrope(model, section)
        vu.install_batch_cache_delta()
        realize_module_arrays(model)
        self._tokenizer = load_text_tokenizer(
            self._model_path, self._model_name, self._trust_remote_code
        )
        self._model = model
        self._vlm_model = vlm_model
        self._vlm_processor = processor
        self._mrope_section = section
        self._text_model_type = model_type
        self._configure_metal_memory_limits()
        logger.info(
            "[vision_unified] loaded %s once through mlx-vlm: text model "
            "model_type=%s (zero-copy), MRoPE section %s",
            self._model_name,
            model_type,
            section,
        )

    async def _start_llm(self) -> None:
        await super()._start_llm()
        core = self._engine.engine
        self._media_rows = MediaRows(self._vlm_model, self._mrope_section)
        core.scheduler._vision_unified = self._media_rows
        original = core.add_request

        async def add_request(prompt, sampling_params=None, request_id=None, **kw):
            media = self._pending_media.pop(id(prompt), None)
            if media is not None:
                kw["images"] = [media]
            return await original(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                **kw,
            )

        core.add_request = add_request

    # -- prompt --------------------------------------------------------------
    def _apply_chat_template(
        self,
        messages: list,
        tools: Optional[list] = None,
        num_images: int = 0,
        num_audios: int = 0,
        num_videos: int = 0,
        chat_template_kwargs: Optional[dict] = None,
        enable_thinking: Optional[bool] = None,
    ) -> str:
        if num_images or num_videos or num_audios:
            messages = self._prepare_mllm_messages(messages)
        return super()._apply_chat_template(
            messages,
            tools,
            num_images=num_images,
            num_audios=num_audios,
            num_videos=num_videos,
            chat_template_kwargs=chat_template_kwargs,
            enable_thinking=enable_thinking,
        )

    def _preprocess_media(self, prompt: str, images: list) -> MediaInputs:
        """Load the images and run mlx-vlm's processor on the rendered
        prompt. Host-side only (numpy out); runs off the event loop."""
        from mlx_vlm.utils import prepare_inputs

        from .mllm_media_key import image_digest
        from .models.mllm import process_image_input

        tic = time.perf_counter()
        cfg = self._vlm_model.config
        image_token = getattr(cfg, "image_token_id", None) or getattr(
            cfg, "image_token_index", None
        )
        tok = self._tokenizer
        rendered_ids = tok.encode(prompt) if hasattr(tok, "encode") else []
        placeholders = sum(1 for t in rendered_ids if t == image_token)
        if placeholders != len(images):
            raise MediaNotSupported(
                f"the chat template rendered {placeholders} image "
                f"placeholder(s) for {len(images)} image(s); this route's "
                f"template cannot place images"
            )
        paths = [process_image_input(img) for img in images]
        inputs = prepare_inputs(
            self._vlm_processor,
            images=paths,
            prompts=prompt,
            image_token_index=image_token,
        )
        ids = _np(inputs["input_ids"]).reshape(-1).astype(np.int64).tolist()
        grid = inputs.get("image_grid_thw")
        pixel_values = inputs.get("pixel_values")
        if grid is None or pixel_values is None:
            raise MediaNotSupported("the processor returned no image grid")
        n_tokens = sum(1 for t in ids if t == image_token)
        return MediaInputs(
            input_ids=ids,
            pixel_values=_np(pixel_values),
            image_grid_thw=_np(grid),
            image_digests=tuple(image_digest(img) for img in images),
            n_images=len(images),
            image_tokens=n_tokens,
            preprocess_s=time.perf_counter() - tic,
        )

    async def _media_prompt(self, prompt: str, images, videos, audio) -> list:
        if videos or audio:
            raise MediaNotSupported(
                "this --vision-unified route serves images only; "
                "video/audio content is not supported here"
            )
        media = await asyncio.to_thread(self._preprocess_media, prompt, list(images))
        ids = list(media.input_ids)
        self._pending_media[id(ids)] = media
        return ids

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list] = None,
        images: Optional[list] = None,
        videos: Optional[list] = None,
        audio: Optional[list] = None,
        **kwargs,
    ) -> GenerationOutput:
        if not (images or videos or audio):
            return await super().generate(
                prompt, max_tokens, temperature, top_p, stop, **kwargs
            )
        if not self._loaded:
            await self.start()
        ids = await self._media_prompt(prompt, images, videos, audio)
        try:
            return await super().generate(
                ids, max_tokens, temperature, top_p, stop, **kwargs
            )
        finally:
            self._pending_media.pop(id(ids), None)

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list] = None,
        images: Optional[list] = None,
        videos: Optional[list] = None,
        audio: Optional[list] = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        if not (images or videos or audio):
            async for out in super().stream_generate(
                prompt, max_tokens, temperature, top_p, stop, **kwargs
            ):
                yield out
            return
        if not self._loaded:
            await self.start()
        ids = await self._media_prompt(prompt, images, videos, audio)
        try:
            async for out in super().stream_generate(
                ids, max_tokens, temperature, top_p, stop, **kwargs
            ):
                yield out
        finally:
            self._pending_media.pop(id(ids), None)

    # -- stats ---------------------------------------------------------------
    def get_stats(self) -> dict:
        stats = super().get_stats()
        rows = self._media_rows
        stats["vision_unified"] = {
            "text_model_type": self._text_model_type,
            "mrope_section": self._mrope_section,
            **(rows.stats() if rows is not None else {}),
        }
        return stats
