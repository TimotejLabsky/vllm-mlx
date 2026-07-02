"""Stop-string enforcement on the batched path (PATCHES.md #32).

The batched schedulers only honor stop TOKEN IDS; ``sampling_params.stop``
was never read, so API/parser stop strings silently ran through. The engine
layer now scans generated text: streaming cuts the final chunk at the match
start and aborts the underlying request; non-stream truncates the full text.
"""

import asyncio
from types import SimpleNamespace

from vllm_mlx.engine.batched import BatchedEngine
from vllm_mlx.stop_strings import StopStringScanner, truncate_at_stop

# ---------------------------------------------------------------- unit level


def test_truncate_at_stop_earliest_match():
    text, hit = truncate_at_stop("a STOP b HALT c", ["HALT", "STOP"])
    assert (text, hit) == ("a ", True)


def test_truncate_at_stop_no_match():
    assert truncate_at_stop("abc", ["STOP"]) == ("abc", False)
    assert truncate_at_stop("abc", None) == ("abc", False)


def test_scanner_match_within_chunk():
    s = StopStringScanner(["<|im_end|>"])
    assert s.scan("Hello") == ("Hello", False)
    assert s.scan(" world<|im_end|> tail") == (" world", True)


def test_scanner_match_spanning_chunks():
    s = StopStringScanner(["ABC"])
    emit, hit = s.scan("xA")
    assert (emit, hit) == ("xA", False)
    emit, hit = s.scan("BCy")
    # Match started inside the already-emitted tail: nothing more to emit.
    assert (emit, hit) == ("", True)


def test_scanner_single_char_stop():
    s = StopStringScanner(["\n"])
    assert s.scan("ab\ncd") == ("ab", True)


def test_scanner_inactive_passthrough():
    s = StopStringScanner([])
    assert not s.active
    assert s.scan("anything") == ("anything", False)


# ------------------------------------------------------------- wiring level


def _output(**overrides):
    base = dict(
        output_text="",
        new_text="",
        output_token_ids=[1],
        prompt_tokens=1,
        completion_tokens=1,
        finished=False,
        finish_reason=None,
        mtp_drafts=0,
        mtp_accepted=0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class _StreamLLMEngine:
    """Yields three chunks; records whether the request got aborted."""

    def __init__(self):
        self.aborted = None

    async def add_request(self, prompt, sampling_params, prefix_boundary=0):
        return "req-1"

    async def stream_outputs(self, request_id):
        for out in (
            _output(new_text="Hello ", output_text="Hello "),
            _output(new_text="world<|im_end|>", output_text="Hello world<|im_end|>"),
            _output(
                new_text=" leaked",
                output_text="Hello world<|im_end|> leaked",
                finished=True,
                finish_reason="length",
            ),
        ):
            yield out

    async def abort_request(self, request_id):
        self.aborted = request_id
        return True

    async def generate(self, prompt, sampling_params):
        return _output(
            output_text="Hello world<|im_end|> leaked",
            finished=True,
            finish_reason="length",
        )


def _make_llm_engine():
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._loaded = True
    engine._is_mllm = False
    engine._mllm_scheduler = None
    engine._engine = _StreamLLMEngine()
    engine._tokenizer = SimpleNamespace(encode=lambda s, **k: [1])
    return engine


def test_stream_generate_stops_and_aborts_on_stop_string():
    engine = _make_llm_engine()
    chunks = []

    async def consume():
        async for out in engine.stream_generate(prompt="x", stop=["<|im_end|>"]):
            chunks.append(out)

    asyncio.run(consume())

    assert "".join(c.new_text for c in chunks) == "Hello world"
    assert chunks[-1].finished is True
    assert chunks[-1].finish_reason == "stop"
    # The third chunk (post-stop) must never be yielded.
    assert len(chunks) == 2
    assert engine._engine.aborted == "req-1"


def test_stream_generate_no_stop_passthrough():
    engine = _make_llm_engine()
    chunks = []

    async def consume():
        async for out in engine.stream_generate(prompt="x"):
            chunks.append(out)

    asyncio.run(consume())
    assert "".join(c.new_text for c in chunks) == "Hello world<|im_end|> leaked"
    assert chunks[-1].finish_reason == "length"
    assert engine._engine.aborted is None


def test_generate_truncates_at_stop_string():
    engine = _make_llm_engine()
    out = asyncio.run(engine.generate(prompt="x", stop=["<|im_end|>"]))
    assert out.text == "Hello world"
    assert out.finish_reason == "stop"


class _StreamMLLMScheduler:
    def __init__(self):
        self.aborted = None

    async def add_request_async(self, **kwargs):
        return "req-2"

    async def stream_outputs(self, request_id):
        for out in (
            _output(new_text="A", output_text="A"),
            _output(new_text="B<stop>C", output_text="AB<stop>C"),
        ):
            yield out

    def abort_request(self, request_id):
        self.aborted = request_id
        return True

    async def generate(self, **kwargs):
        return _output(
            output_text="AB<stop>C", finished=True, finish_reason="length"
        )


def _make_mllm_engine():
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._loaded = True
    engine._is_mllm = True
    engine._mllm_scheduler = _StreamMLLMScheduler()
    engine._engine = None
    engine._processor = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda s, **k: [1]))
    engine._tokenizer = None
    return engine


def test_mllm_stream_generate_stops_on_stop_string():
    engine = _make_mllm_engine()
    chunks = []

    async def consume():
        async for out in engine.stream_generate(prompt="x", stop=["<stop>"]):
            chunks.append(out)

    asyncio.run(consume())
    assert "".join(c.new_text for c in chunks) == "AB"
    assert chunks[-1].finish_reason == "stop"
    assert engine._mllm_scheduler.aborted == "req-2"


def test_mllm_generate_truncates_at_stop_string():
    engine = _make_mllm_engine()
    out = asyncio.run(engine.generate(prompt="x", stop=["<stop>"]))
    assert out.text == "AB"
    assert out.finish_reason == "stop"
