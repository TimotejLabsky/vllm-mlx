# SPDX-License-Identifier: Apache-2.0
"""(#181) ``--vision-split``: one route serves text AND images from one load.

A checkpoint that ships a vision tower (Qwen3.8-27B, Qwen3.6-35B-A3B with
vision weights) used to need a choice per route: ``--text-only`` keeps every
fork text feature (batched system-KV + SSD tier, relief, measured-bytes
admission, the text MTP drafter, ...) and 400s images (#65); the MLLM branch
answers images but runs text through ``MLLMScheduler``, which has none of
those. This mode runs both in one process over ONE copy of the weights:

* the checkpoint loads once, through mlx-vlm (model + processor), exactly as
  an MLLM route loads it;
* :func:`~vllm_mlx.text_model_from_vlm.build_text_model` builds an mlx-lm
  ``TextModel`` from it that shares every backbone/lm_head array zero-copy;
* :class:`VisionSplitEngine` owns TWO stock :class:`BatchedEngine` children:
  ``text`` is configured exactly as a ``--text-only`` route (it IS one, with
  the model injected instead of loaded), ``vision`` exactly as an MLLM route
  (with a small explicit prefix-cache budget, no SSD tier, no MTP);
* requests route per request: media present -> ``vision``; otherwise
  ``text``, through the unchanged text preparation path in ``server.py``;
* the two schedulers take turns on the GPU (``vllm_mlx/gpu_gate.py``).

Everything here is opt-in. Without the flag nothing in this module runs: the
server builds a plain ``BatchedEngine`` and the two seams in ``server.py``
(:func:`request_is_mllm`, :func:`engine_serves_media`) reduce to the old
``engine.is_mllm`` reads.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from functools import partial
from pathlib import Path
from typing import Any, AsyncIterator

from .engine.base import BaseEngine, GenerationOutput, cleanup_startup_cancellation
from .gpu_gate import GatedWorkerExecutor, GpuGate

logger = logging.getLogger(__name__)

# Content part types that make a request a media request. Same set the
# server's MLLM-branch media scan has always used.
MEDIA_PART_TYPES = frozenset(
    {"image_url", "image", "video", "video_url", "audio", "audio_url"}
)

DEFAULT_VISION_CACHE_MB = 1024
DEFAULT_YIELD_TIMEOUT_S = 0.5


class VisionSplitError(RuntimeError):
    """A ``--vision-split`` route cannot start as configured. Never degraded
    silently: the route refuses to start and the message says why."""


# ---------------------------------------------------------------------------
# Request classification (server seams)
# ---------------------------------------------------------------------------


def _part_type(part: Any) -> str:
    if isinstance(part, dict):
        return str(part.get("type", "") or "")
    return str(getattr(part, "type", "") or "")


def messages_have_media(messages: Any) -> bool:
    """True when any message carries an image/video/audio content part.

    Accepts the server's pydantic ``Message`` objects and plain dicts.
    """
    for msg in messages or ():
        if isinstance(msg, dict):
            content = msg.get("content")
        else:
            content = getattr(msg, "content", None)
        if isinstance(content, list):
            for part in content:
                if _part_type(part) in MEDIA_PART_TYPES:
                    return True
    return False


def is_vision_split(engine: Any) -> bool:
    return getattr(engine, "vision_split", False) is True


def request_is_mllm(engine: Any, request_messages: Any) -> bool:
    """Whether the server prepares THIS request MLLM-style.

    On every route but a split one this is ``engine.is_mllm``, unchanged. On
    a split route only a media request takes the MLLM preparation (images
    stay embedded in the messages for the processor); a text request takes
    the exact text path a ``--text-only`` route takes.
    """
    if is_vision_split(engine):
        return messages_have_media(request_messages)
    return bool(getattr(engine, "is_mllm", False))


def engine_serves_media(engine: Any) -> bool:
    """The #65 media guard's question: may this route answer media at all?"""
    return bool(getattr(engine, "is_mllm", False)) or is_vision_split(engine)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def check_vision_split_config(
    *,
    use_batching: bool,
    force_text_only: bool = False,
    force_mllm: bool = False,
    auto_unload_idle_seconds: float = 0.0,
    lazy_load_model: bool = False,
    mllm_draft_model: str | None = None,
    enable_mtp: bool = False,
    compile: bool = False,
    models_config: str | None = None,
) -> None:
    """Raise :class:`VisionSplitError` naming every reason the flag cannot
    apply. Called from the CLI before any download and again in
    ``load_model``."""
    problems = []
    if not use_batching:
        problems.append("requires --continuous-batching")
    if force_text_only:
        problems.append("is mutually exclusive with --text-only")
    if force_mllm:
        problems.append("is mutually exclusive with --mllm")
    if auto_unload_idle_seconds and auto_unload_idle_seconds > 0:
        problems.append("does not support --auto-unload-idle-seconds yet")
    if lazy_load_model:
        problems.append("does not support --lazy-load-model yet")
    if mllm_draft_model:
        problems.append("does not support an MLLM draft model")
    if enable_mtp:
        problems.append(
            "does not support --enable-mtp (the text side would not run the "
            "--text-only code path; the text drafter env works)"
        )
    if compile:
        problems.append("does not support --compile")
    if models_config:
        problems.append("serves one model; it does not support --models-config")
    from .moe_fusion import fusion_enabled

    if fusion_enabled():
        problems.append(
            "cannot run with VLLM_MLX_MOE_GATEUP_FUSION armed: the fused "
            "gate/up tensor would be a second copy of the expert weights next "
            "to the vision model's (VLLM_MLX_MOE_DECODE_KERNELS alone is fine)"
        )
    if problems:
        raise VisionSplitError("--vision-split " + "; ".join(problems))


def vision_scheduler_config(scheduler_config: Any, cache_mb: int) -> Any:
    """The vision child's copy of the route's SchedulerConfig.

    * ``cache_memory_mb`` -> the split budget: on the MLLM branch this field
      sizes its prefix cache, and the route's own value is the TEXT budget;
    * no SSD tier: the text side owns the route's SSD directory;
    * no MTP injection into the shared vlm model.
    """
    if scheduler_config is None:
        from .scheduler import SchedulerConfig

        scheduler_config = SchedulerConfig()
    return dataclasses.replace(
        scheduler_config,
        cache_memory_mb=int(cache_mb),
        ssd_cache_dir=None,
        enable_mtp=False,
    )


def _local_model_dir(model_path: str) -> Path:
    from .text_model_from_vlm import _resolve_model_path

    local = _resolve_model_path(model_path)
    if local is None:
        try:
            from mlx_lm.utils import _download

            local = Path(_download(model_path))
        except Exception as e:  # noqa: BLE001
            raise VisionSplitError(
                f"--vision-split: cannot resolve a local snapshot for {model_path!r}"
            ) from e
    return Path(local)


def text_model_type(model_path: str) -> str:
    try:
        config = json.loads((_local_model_dir(model_path) / "config.json").read_text())
    except Exception:  # noqa: BLE001
        return ""
    text_config = config.get("text_config", config)
    return str(text_config.get("model_type") or config.get("model_type") or "")


def load_text_tokenizer(model_path: str, model_name: str, trust_remote_code: bool):
    """The tokenizer a ``--text-only`` route gets for this checkpoint.

    Same call ``utils.tokenizer._load_strict_false`` makes for a VLM-shaped
    checkpoint (mlx-lm ``load_tokenizer``, eos ids from ``load_config``, the
    Qwen3 ``<|im_end|>`` eos override), then the same template installs
    (DeepSeek-V4 encoder, ``VLLM_MLX_CHAT_TEMPLATE_FILE``). A SEPARATE object
    from the processor's tokenizer: the MLLM batch generator rewrites its own
    processor's chat template for prefix stability, and that must not reach
    the text side.
    """
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


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class VisionSplitEngine(BaseEngine):
    """Two stock BatchedEngines over one model; see the module docstring."""

    vision_split = True

    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = False,
        scheduler_config: Any | None = None,
        stream_interval: int = 1,
        gpu_memory_utilization: float = 0.90,
        model_path: str | None = None,
        vision_cache_mb: int = DEFAULT_VISION_CACHE_MB,
        yield_timeout_s: float = DEFAULT_YIELD_TIMEOUT_S,
        specprefill_enabled: bool = False,
        specprefill_threshold: int = 8192,
        specprefill_keep_pct: float = 0.3,
        specprefill_backbone_pct: float = 0.0,
        specprefill_draft_model: str | None = None,
    ) -> None:
        from .engine.batched import BatchedEngine

        if vision_cache_mb <= 0:
            raise VisionSplitError("--vision-split-cache-mb must be positive")
        self._model_name = model_name
        self._model_path = model_path or model_name
        self._trust_remote_code = trust_remote_code
        self._vision_cache_mb = int(vision_cache_mb)
        self.gate = GpuGate(yield_timeout_s=yield_timeout_s)
        # The text child: a --text-only BatchedEngine in every respect but
        # the load (injected in start()).
        self.text = BatchedEngine(
            model_name=model_name,
            trust_remote_code=trust_remote_code,
            scheduler_config=scheduler_config,
            stream_interval=stream_interval,
            force_text_only=True,
            gpu_memory_utilization=gpu_memory_utilization,
            model_path=model_path,
        )
        # Its worker thread, gated: every MLX call there is one text turn.
        self.text._generation_executor = GatedWorkerExecutor(
            self.gate, initializer=self.text._record_generation_thread
        )
        self.vision = BatchedEngine(
            model_name=model_name,
            trust_remote_code=trust_remote_code,
            scheduler_config=vision_scheduler_config(
                scheduler_config, self._vision_cache_mb
            ),
            stream_interval=stream_interval,
            force_mllm=True,
            gpu_memory_utilization=gpu_memory_utilization,
            specprefill_enabled=specprefill_enabled,
            specprefill_threshold=specprefill_threshold,
            specprefill_keep_pct=specprefill_keep_pct,
            specprefill_backbone_pct=specprefill_backbone_pct,
            specprefill_draft_model=specprefill_draft_model,
            model_path=model_path,
        )
        self._loaded = False
        self._text_route_info: dict[str, Any] = {}

    # -- identity ------------------------------------------------------------
    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_mllm(self) -> bool:
        # Text-flavoured on purpose: every server path that reads is_mllm
        # (message prep, template choice, health model_type) must behave as
        # on a --text-only route for text. Media is decided per request via
        # request_is_mllm / engine_serves_media.
        return False

    @property
    def tokenizer(self) -> Any:
        return self.text.tokenizer

    @property
    def preserve_native_tool_format(self) -> bool:  # type: ignore[override]
        return getattr(self.text, "preserve_native_tool_format", False)

    @preserve_native_tool_format.setter
    def preserve_native_tool_format(self, value: bool) -> None:
        self.text.preserve_native_tool_format = value
        self.vision.preserve_native_tool_format = value

    @property
    def use_harmony_rendering(self) -> bool:  # type: ignore[override]
        return getattr(self.text, "use_harmony_rendering", False)

    @use_harmony_rendering.setter
    def use_harmony_rendering(self, value: bool) -> None:
        self.text.use_harmony_rendering = value
        self.vision.use_harmony_rendering = value

    def __getattr__(self, name: str) -> Any:
        # Anything the server reads that is not defined here (e.g.
        # _apply_chat_template for the thinking probe, _tokenizer) answers
        # as the --text-only route would. Only reached for missing names.
        if name in ("text", "vision", "gate"):
            raise AttributeError(name)
        return getattr(self.text, name)

    # -- lifecycle -----------------------------------------------------------
    def _build_text_side(self, vlm_model: Any) -> tuple[Any, Any]:
        """On the text worker (one gated turn): build, realize, arm."""
        from .lazy_realize import realize_module_arrays
        from .moe_fusion import maybe_fuse
        from .text_model_from_vlm import build_text_model

        model_type = text_model_type(self._model_path)
        try:
            text_model = build_text_model(vlm_model, self._model_path, enable_mtp=False)
        except Exception as e:  # noqa: BLE001
            raise VisionSplitError(
                f"--vision-split: building the text model from the vision "
                f"checkpoint raised ({type(e).__name__}: {e}); model_type="
                f"{model_type!r}. Serve it with --text-only instead."
            ) from e
        if text_model is None:
            raise VisionSplitError(
                f"--vision-split: no text model can be built from this "
                f"checkpoint (model_type={model_type!r}; see the log above for "
                f"build_text_model's reason). Serve it with --text-only, or as "
                f"an MLLM route."
            )
        # What _prepare_llm_model does after a --text-only load: realize on
        # the stepping thread, then the env-gated MoE hooks (#96 is refused
        # by check_vision_split_config; #179 kernels arm per instance on the
        # text model's own modules, never the vlm's).
        realize_module_arrays(text_model)
        maybe_fuse(text_model, self._model_name)
        tokenizer = load_text_tokenizer(
            self._model_path, self._model_name, self._trust_remote_code
        )
        self._text_route_info = {
            "text_model_class": f"{type(text_model).__module__}."
            f"{type(text_model).__qualname__}",
            "text_model_type": model_type,
        }
        return text_model, tokenizer

    async def start(self) -> None:
        if self._loaded:
            return
        try:
            # 1. The one load: mlx-vlm model + processor, on the event loop
            #    (where MLLMScheduler steps), MLLM scheduler started.
            await self.vision.start()
            scheduler = self.vision._mllm_scheduler
            if scheduler is None or self.vision._model is None:
                raise VisionSplitError(
                    "--vision-split: the vision side did not start an MLLM scheduler"
                )
            # read by MLLMScheduler._process_loop via getattr (#181)
            setattr(scheduler, "_gpu_gate", self.gate)
            # 2. The text model from the same arrays, built on the text
            #    worker (its stepping thread), then a --text-only start.
            text_model, tokenizer = await self.text._run_on_generation_worker(
                partial(self._build_text_side, self.vision._model)
            )
            self.text._model = text_model
            self.text._tokenizer = tokenizer
            self.text._configure_metal_memory_limits()
            await self.text.start()
            self._loaded = True
            logger.info(
                "[vision-split] %s: text -> AsyncEngineCore (%s), media -> "
                "MLLMScheduler (prefix cache %d MB); GPU gate: %s",
                self._model_name,
                self._text_route_info.get("text_model_class"),
                self._vision_cache_mb,
                self.gate.stats()["policy"],
            )
        except BaseException as e:
            import asyncio

            if isinstance(e, asyncio.CancelledError):
                await cleanup_startup_cancellation(self.stop)
            else:
                try:
                    await self.stop()
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "[vision-split] cleanup after failed start", exc_info=True
                    )
            raise

    async def stop(self) -> None:
        # Text first: its worker drains its last turn through the gate.
        try:
            await self.text.stop()
        finally:
            await self.vision.stop()
            self._loaded = False

    def is_healthy(self) -> bool:
        return self.text.is_healthy() and self.vision.is_healthy()

    # -- routing ---------------------------------------------------------------
    # Signatures and defaults are BatchedEngine's: every value is forwarded
    # explicitly, so the chosen child sees exactly what the caller sent.
    def _side_for(
        self, messages: Any = None, images=None, videos=None, audio=None
    ) -> Any:
        if images or videos or audio or messages_have_media(messages):
            return self.vision
        return self.text

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> GenerationOutput:
        side = self._side_for(
            images=kwargs.get("images"),
            videos=kwargs.get("videos"),
            audio=kwargs.get("audio"),
        )
        output: GenerationOutput = await side.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            **kwargs,
        )
        return output

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationOutput]:
        side = self._side_for(
            images=kwargs.get("images"),
            videos=kwargs.get("videos"),
            audio=kwargs.get("audio"),
        )
        async for output in side.stream_generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            **kwargs,
        ):
            yield output

    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: list[dict] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        **kwargs: Any,
    ) -> GenerationOutput:
        side = self._side_for(messages, images=images, videos=videos)
        output: GenerationOutput = await side.chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            tools=tools,
            images=images,
            videos=videos,
            **kwargs,
        )
        return output

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: list[dict] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationOutput]:
        side = self._side_for(messages, images=images, videos=videos)
        async for output in side.stream_chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            tools=tools,
            images=images,
            videos=videos,
            **kwargs,
        ):
            yield output

    async def abort_request(self, request_id: str) -> bool:
        if await self.text.abort_request(request_id):
            return True
        return bool(await self.vision.abort_request(request_id))

    def raise_if_serialized_busy(self, request_id: str = "stream") -> None:
        # The pre-SSE probe has no request body; the text queue is the
        # route's queue. The vision scheduler enforces its own cap at add.
        self.text.raise_if_serialized_busy(request_id)

    # -- stats -----------------------------------------------------------------
    def get_stats(self) -> dict[str, Any]:
        """The text engine's stats, unchanged (dashboards and the exporter
        keep reading a --text-only route), with the vision side and the gate
        under ``vision_split``. Request counts are summed and the request
        lists merged, so an idle check sees both sides."""
        stats = dict(self.text.get_stats())
        vstats = self.vision.get_stats()
        for key in ("num_running", "num_waiting"):
            stats[key] = int(stats.get(key, 0) or 0) + int(vstats.get(key, 0) or 0)
        stats["requests"] = list(stats.get("requests") or []) + [
            {**r, "side": "vision"} for r in (vstats.get("requests") or [])
        ]
        if "vision_embedding_cache" in vstats:
            stats["vision_embedding_cache"] = vstats["vision_embedding_cache"]
        stats["vision_split"] = {
            "enabled": True,
            "gate": self.gate.stats(),
            "vision_prefix_cache_budget_mb": self._vision_cache_mb,
            **self._text_route_info,
            "vision": vstats,
        }
        return stats

    def get_cache_stats(self) -> dict[str, Any] | None:
        text = self.text.get_cache_stats()
        out: dict[str, Any] = dict(text) if isinstance(text, dict) else {}
        out["vision"] = self.vision.get_cache_stats()
        return out

    # -- caches ----------------------------------------------------------------
    def clear_runtime_caches(self) -> dict[str, Any] | None:
        return {
            "text": self.text.clear_runtime_caches(),
            "vision": self.vision.clear_runtime_caches(),
        }

    async def _clear_runtime_caches_on_owner(self) -> dict[str, Any] | None:
        return {
            "text": await self.text._clear_runtime_caches_on_owner(),
            "vision": await self.vision._clear_runtime_caches_on_owner(),
        }

    def clear_prefix_cache(self) -> dict[str, Any] | None:
        out = dict(self.text.clear_prefix_cache() or {})
        for key, value in (self.vision.clear_prefix_cache() or {}).items():
            out[f"vision_{key}"] = value
        return out

    async def _clear_prefix_cache_on_owner(self) -> dict[str, Any] | None:
        out = dict(await self.text._clear_prefix_cache_on_owner() or {})
        vision = await self.vision._clear_prefix_cache_on_owner() or {}
        for key, value in vision.items():
            out[f"vision_{key}"] = value
        return out

    # Persistence is the TEXT cache only. The vision prefix cache is small,
    # in memory, and rebuilt by use: saving it into the same directory would
    # interleave two caches' files, and media-keyed entries (#142) have never
    # been round-tripped through save_to_disk.
    def save_cache_to_disk(self, cache_dir: str) -> bool:
        return self.text.save_cache_to_disk(cache_dir)

    async def _save_cache_to_disk_on_owner(self, cache_dir: str) -> bool:
        return await self.text._save_cache_to_disk_on_owner(cache_dir)

    def load_cache_from_disk(self, cache_dir: str) -> int:
        return self.text.load_cache_from_disk(cache_dir)

    async def _load_cache_from_disk_on_owner(self, cache_dir: str) -> int:
        return await self.text._load_cache_from_disk_on_owner(cache_dir)
