# SPDX-License-Identifier: Apache-2.0
"""PATCHES.md #181 stage 2 — the ``--vision-unified`` engine, config and
server seams (no weights downloaded)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from tests.test_vision_unified_181 import _tiny
from vllm_mlx import vision_unified as vu


def test_181_config_refuses_every_unsupported_combination(monkeypatch):
    from vllm_mlx.vision_unified_engine import (
        VisionUnifiedError,
        check_vision_unified_config,
    )

    monkeypatch.delenv("VLLM_MLX_MOE_GATEUP_FUSION", raising=False)
    check_vision_unified_config(use_batching=True)  # the supported shape
    cases = {
        "continuous-batching": dict(use_batching=False),
        "text-only": dict(use_batching=True, force_text_only=True),
        "--mllm": dict(use_batching=True, force_mllm=True),
        "auto-unload": dict(use_batching=True, auto_unload_idle_seconds=5),
        "lazy-load": dict(use_batching=True, lazy_load_model=True),
        "MLLM draft": dict(use_batching=True, mllm_draft_model="x"),
        "enable-mtp": dict(use_batching=True, enable_mtp=True),
        "compile": dict(use_batching=True, compile=True),
        "models-config": dict(use_batching=True, models_config="m.yaml"),
        "max-kv-size": dict(use_batching=True, max_kv_size=4096),
        "chunked-prefill": dict(use_batching=True, chunked_prefill_tokens=512),
    }
    for needle, kw in cases.items():
        with pytest.raises(VisionUnifiedError, match=needle):
            check_vision_unified_config(**kw)
    monkeypatch.setenv("VLLM_MLX_MOE_GATEUP_FUSION", "1")
    with pytest.raises(VisionUnifiedError, match="GATEUP_FUSION"):
        check_vision_unified_config(use_batching=True)


def _cfg(text_type="qwen3_5_text", rope=None, vision=True):
    rope = (
        {"mrope_section": [11, 11, 10], "mrope_interleaved": True}
        if rope is None
        else rope
    )
    cfg = {
        "model_type": "qwen3_5",
        "text_config": {"model_type": text_type, "rope_parameters": rope},
    }
    if vision:
        cfg["vision_config"] = {"depth": 2}
    return cfg


def test_181_checkpoint_support():
    from vllm_mlx.vision_unified_engine import VisionUnifiedError, checkpoint_support

    assert checkpoint_support(_cfg()) == ("qwen3_5_text", [11, 11, 10])
    assert checkpoint_support(_cfg("qwen3_5_moe_text"))[0] == "qwen3_5_moe_text"
    for bad, needle in (
        (_cfg(vision=False), "vision tower"),
        (_cfg("gemma4_text"), "supports"),
        (_cfg("qwen3_vl"), "supports"),
        (_cfg(rope={}), "mrope_section"),
        (
            _cfg(rope={"mrope_section": [1, 1, 1], "mrope_interleaved": False}),
            "interleaved",
        ),
    ):
        with pytest.raises(VisionUnifiedError, match=needle):
            checkpoint_support(bad)


def test_181_server_seams_are_engine_is_mllm_off_a_unified_route():
    from vllm_mlx.vision_unified_engine import engine_serves_media, request_is_mllm

    img = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}
    ]
    txt = [{"role": "user", "content": "hi"}]
    for is_mllm in (False, True):
        plain = SimpleNamespace(is_mllm=is_mllm)
        assert request_is_mllm(plain, img) is is_mllm
        assert request_is_mllm(plain, txt) is is_mllm
        assert engine_serves_media(plain) is is_mllm
    unified = SimpleNamespace(is_mllm=False, vision_unified=True)
    assert request_is_mllm(unified, img) is True
    assert request_is_mllm(unified, txt) is False
    assert engine_serves_media(unified) is True
    # a truthy non-True attribute (MagicMock) is not the flag
    assert (
        engine_serves_media(SimpleNamespace(is_mllm=False, vision_unified=1)) is False
    )


def test_181_text_only_shell_shares_weights_and_logits():
    from mlx.utils import tree_flatten
    from mlx_lm.models import qwen3_5

    from vllm_mlx.vision_unified_engine import wrap_as_text_only_model

    inner = _tiny()
    cfg = {"model_type": "qwen3_5", "text_config": dict(vars(inner.args))}
    outer = wrap_as_text_only_model(inner, cfg)
    assert type(outer) is qwen3_5.Model and outer.language_model is inner
    a = dict(tree_flatten(outer.parameters()))
    b = dict(tree_flatten(inner.parameters()))
    assert len(a) == len(b)
    assert all(a[f"language_model.{k}"] is v for k, v in b.items())
    toks = mx.array([[1, 2, 3, 4]])
    assert mx.array_equal(outer(toks), inner(toks))


def _bare_engine():
    from vllm_mlx.vision_unified_engine import VisionUnifiedEngine

    eng = VisionUnifiedEngine.__new__(VisionUnifiedEngine)
    eng._pending_media = {}
    eng._loaded = True
    eng._media_rows = None
    return eng


def test_181_video_and_audio_are_refused_with_media_not_supported():
    from vllm_mlx.engine.base import MediaNotSupported

    eng = _bare_engine()
    for videos, audio in ((["v"], None), (None, ["a"])):
        with pytest.raises(MediaNotSupported, match="images only"):
            asyncio.run(eng._media_prompt("p", [], videos, audio))


def test_181_template_without_image_placeholders_is_refused():
    from vllm_mlx.engine.base import MediaNotSupported

    eng = _bare_engine()
    eng._vlm_model = SimpleNamespace(config=SimpleNamespace(image_token_id=7))
    eng._tokenizer = SimpleNamespace(encode=lambda s: [1, 2, 3])  # no 7
    with pytest.raises(MediaNotSupported, match="0 image placeholder"):
        eng._preprocess_media("rendered prompt", ["data:image/png;base64,AAAA"])


def test_181_text_generate_is_the_parent_call(monkeypatch):
    from vllm_mlx.engine.batched import BatchedEngine

    seen = {}

    async def parent_generate(self, prompt, *a, **kw):
        seen["prompt"] = prompt
        seen["kw"] = kw
        return "out"

    monkeypatch.setattr(BatchedEngine, "generate", parent_generate)
    eng = _bare_engine()
    assert asyncio.run(eng.generate("hello", 8, top_k=3)) == "out"
    assert seen == {"prompt": "hello", "kw": {"top_k": 3}}
    assert eng._pending_media == {}


def test_181_media_attaches_to_the_request_by_prompt_identity(monkeypatch):
    from vllm_mlx.engine.batched import BatchedEngine

    eng = _bare_engine()
    media = vu.MediaInputs(
        input_ids=[1, 2], pixel_values=np.zeros(1), image_grid_thw=np.zeros(3)
    )
    seen = {}

    async def original(prompt, sampling_params=None, request_id=None, **kw):
        seen.clear()
        seen.update(kw)
        return "rid"

    async def parent_start(self):
        return None

    core = SimpleNamespace(add_request=original, scheduler=SimpleNamespace())
    eng._engine = SimpleNamespace(engine=core)
    eng._vlm_model = SimpleNamespace(config=None)
    eng._mrope_section = [1, 1, 1]
    monkeypatch.setattr(BatchedEngine, "_start_llm", parent_start)
    asyncio.run(eng._start_llm())
    ids = [1, 2]
    eng._pending_media[id(ids)] = media
    asyncio.run(core.add_request(prompt=[1, 2]))  # an equal list, other object
    assert "images" not in seen
    asyncio.run(core.add_request(prompt=ids))
    assert seen["images"] == [media] and eng._pending_media == {}
    assert core.scheduler._vision_unified is eng._media_rows


def test_181_cli_refuses_before_any_download(monkeypatch, capsys):
    from tests.test_cli import _serve_args
    from vllm_mlx import cli
    from vllm_mlx.utils import download

    called = []
    monkeypatch.setattr(
        download, "ensure_model_downloaded", lambda *a, **k: called.append(a)
    )
    with pytest.raises(SystemExit):
        cli.serve_command(
            _serve_args(continuous_batching=True, text_only=True, vision_unified=True)
        )
    assert "--vision-unified is mutually exclusive with --text-only" in (
        capsys.readouterr().out
    )
    assert called == []


def test_181_cli_flag_defaults_off():
    from vllm_mlx.cli import create_parser

    args = create_parser().parse_args(["serve", "m"])
    assert args.vision_unified is False
    args = create_parser().parse_args(
        ["serve", "m", "--continuous-batching", "--vision-unified"]
    )
    assert args.vision_unified is True
