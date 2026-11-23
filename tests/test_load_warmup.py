# SPDX-License-Identifier: Apache-2.0
"""Fork #112 probe: env-gated load-time warm-up."""

import asyncio
from types import SimpleNamespace

from vllm_mlx.load_warmup import load_warmup_tokens, run_load_warmup


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_MLX_LOAD_WARMUP", raising=False)
    assert load_warmup_tokens() == 0
    monkeypatch.setenv("VLLM_MLX_LOAD_WARMUP", "junk")
    assert load_warmup_tokens() == 0
    monkeypatch.setenv("VLLM_MLX_LOAD_WARMUP", "4")
    assert load_warmup_tokens() == 4


def test_runs_one_request_and_swallows_errors():
    calls = []

    class Engine:
        async def stream_chat(self, **kw):
            calls.append(kw)
            yield SimpleNamespace(finished=True)

    asyncio.run(run_load_warmup(Engine(), 2))
    assert calls[0]["max_tokens"] == 2

    class Broken:
        async def stream_chat(self, **kw):
            raise RuntimeError("boom")
            yield

    asyncio.run(run_load_warmup(Broken(), 1))  # must not raise
