"""DRY sampler wiring on the batched path (PATCHES.md #31).

Before this patch the dry_* request fields reached BatchedEngine.chat/
stream_chat in **kwargs and silently vanished — never popped into
SamplingParams, never forwarded to either scheduler. These tests pin the
wiring on all four entry paths without loading a model or MLX scheduler.
"""

import asyncio
from types import SimpleNamespace

import pytest

from vllm_mlx.dry_sampler import DRYLogitsProcessor
from vllm_mlx.engine.batched import BatchedEngine


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [sum(ord(c) for c in text) % 997]


class _FakeOutput(SimpleNamespace):
    pass


def _output(**overrides):
    base = dict(
        output_text="ok",
        new_text="ok",
        output_token_ids=[1],
        prompt_tokens=1,
        completion_tokens=1,
        finished=True,
        finish_reason="stop",
        mtp_drafts=0,
        mtp_accepted=0,
    )
    base.update(overrides)
    return _FakeOutput(**base)


def _make_engine(is_mllm: bool):
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._loaded = True
    engine._is_mllm = is_mllm
    engine._mllm_scheduler = None
    engine._engine = None
    engine._tokenizer = _FakeTokenizer()
    engine._processor = SimpleNamespace(tokenizer=_FakeTokenizer())
    return engine


class _CaptureLLMEngine:
    def __init__(self):
        self.sampling_params = None

    async def generate(self, prompt, sampling_params):
        self.sampling_params = sampling_params
        return _output()

    async def add_request(self, prompt, sampling_params, prefix_boundary=0):
        self.sampling_params = sampling_params
        return "req-1"

    async def stream_outputs(self, request_id):
        yield _output()


class _CaptureMLLMScheduler:
    def __init__(self):
        self.kwargs = None

    async def generate(self, **kwargs):
        self.kwargs = kwargs
        return _output()

    async def add_request_async(self, **kwargs):
        self.kwargs = kwargs
        return "req-1"

    async def stream_outputs(self, request_id):
        yield _output()


def _dry_procs(processors):
    return [p for p in (processors or []) if isinstance(p, DRYLogitsProcessor)]


def test_llm_generate_attaches_dry_processor():
    engine = _make_engine(is_mllm=False)
    engine._engine = _CaptureLLMEngine()

    asyncio.run(
        engine.generate(prompt="x", dry_multiplier=0.8, dry_allowed_length=4)
    )

    procs = _dry_procs(engine._engine.sampling_params.logits_processors)
    assert len(procs) == 1
    assert procs[0].multiplier == pytest.approx(0.8)
    assert procs[0].allowed_length == 4


def test_llm_stream_generate_attaches_dry_processor():
    engine = _make_engine(is_mllm=False)
    engine._engine = _CaptureLLMEngine()

    async def consume():
        async for _ in engine.stream_generate(prompt="x", dry_multiplier=0.8):
            pass

    asyncio.run(consume())
    assert len(_dry_procs(engine._engine.sampling_params.logits_processors)) == 1


def test_mllm_generate_forwards_dry_processor():
    engine = _make_engine(is_mllm=True)
    engine._mllm_scheduler = _CaptureMLLMScheduler()

    asyncio.run(engine.generate(prompt="x", dry_multiplier=0.8))

    assert len(_dry_procs(engine._mllm_scheduler.kwargs["logits_processors"])) == 1


def test_mllm_stream_generate_forwards_dry_processor():
    engine = _make_engine(is_mllm=True)
    engine._mllm_scheduler = _CaptureMLLMScheduler()

    async def consume():
        async for _ in engine.stream_generate(prompt="x", dry_multiplier=0.8):
            pass

    asyncio.run(consume())
    assert len(_dry_procs(engine._mllm_scheduler.kwargs["logits_processors"])) == 1


def test_dry_off_keeps_processors_none():
    engine = _make_engine(is_mllm=False)
    engine._engine = _CaptureLLMEngine()

    asyncio.run(engine.generate(prompt="x"))

    assert engine._engine.sampling_params.logits_processors is None


def test_external_processors_survive_alongside_dry():
    engine = _make_engine(is_mllm=False)
    engine._engine = _CaptureLLMEngine()
    external = lambda tokens, logits: logits  # noqa: E731

    asyncio.run(
        engine.generate(prompt="x", logits_processors=[external], dry_multiplier=0.8)
    )

    procs = engine._engine.sampling_params.logits_processors
    assert external in procs
    assert len(_dry_procs(procs)) == 1
