# SPDX-License-Identifier: Apache-2.0
"""PATCHES.md #91 — a mid-stream failure must not look like an empty answer.

A streaming response commits HTTP 200 before generation starts, so an
exception raised afterwards cannot become an error status. Before this patch
the body simply stopped: llama-swap logged "no valid JSON data found in
stream" and still returned 200, LiteLLM rewrote it to finish_reason=stop, and
an agent client read a well-formed empty turn and quietly stopped — with no
traceback logged anywhere and nothing counted.
"""

import json

import pytest

import vllm_mlx.server as srv
from vllm_mlx.engine.base import GenerationOutput


@pytest.fixture
def anyio_backend():
    return "asyncio"


class _Boom(Exception):
    """Carries a distinctive marker so we can assert it never reaches a client."""


BOOM_TEXT = "SENSITIVE-INTERNAL-DETAIL-42"


class _RecordingTracker:
    def __init__(self):
        self.calls = []

    def observe_ttft(self):
        pass

    def finish(self, **kwargs):
        self.calls.append(kwargs)


def _sse_payloads(chunks):
    """Parse `data: {...}` JSON objects out of raw SSE text."""
    out = []
    for line in "".join(chunks).splitlines():
        if line.startswith("data: "):
            body = line[len("data: ") :]
            if body.strip() != "[DONE]":
                out.append(json.loads(body))
    return out


# ---------------------------------------------------------------- helpers


class TestErrorChunkHelpers:
    def test_openai_chunk_is_valid_sse_json(self):
        raw = srv._stream_error_chunk()
        assert raw.startswith("data: ") and raw.endswith("\n\n")
        err = json.loads(raw[len("data: ") :])["error"]
        assert err["type"] == "internal_error"
        assert err["code"] == "stream_failed"
        assert err["message"]
        assert "request_id" not in err

    def test_openai_chunk_carries_request_id_when_known(self):
        err = json.loads(srv._stream_error_chunk("chatcmpl-abc123")[len("data: ") :])
        assert err["error"]["request_id"] == "chatcmpl-abc123"

    def test_anthropic_event_uses_its_own_dialect(self):
        raw = srv._anthropic_stream_error_event()
        assert raw.startswith("event: error\ndata: ")
        payload = json.loads(raw.split("data: ", 1)[1])
        assert payload["type"] == "error"
        assert payload["error"]["type"] == "api_error"


# ------------------------------------------------- chat completions stream


def _chat_request():
    return srv.ChatCompletionRequest(
        model="test-model",
        messages=[srv.Message(role="user", content="Hello")],
        stream=True,
    )


def _prepare(monkeypatch):
    monkeypatch.setattr(srv, "_model_name", "test-model")
    monkeypatch.setattr(srv, "_reasoning_parser_name", None)
    monkeypatch.setattr(srv, "_reasoning_parser", None)
    monkeypatch.setattr(srv, "_enable_auto_tool_choice", False)
    monkeypatch.setattr(srv, "_tool_call_parser", None)


class _FailingChatEngine:
    """Raises after ``emit`` successful tokens."""

    model_name = "test-model"

    def __init__(self, emit=0):
        self.emit = emit

    async def stream_chat(self, messages, **kwargs):
        for i in range(self.emit):
            yield GenerationOutput(
                text=f"tok{i}",
                new_text=f"tok{i}",
                finished=False,
                finish_reason=None,
                prompt_tokens=4,
                completion_tokens=i + 1,
            )
        raise _Boom(BOOM_TEXT)


@pytest.mark.anyio
async def test_chat_stream_emits_error_chunk_before_first_token(monkeypatch, caplog):
    """The 2026-09-01 shape: raise before any token."""
    _prepare(monkeypatch)
    request = _chat_request()
    tracker = _RecordingTracker()
    chunks = []

    with caplog.at_level("ERROR"):
        with pytest.raises(_Boom):
            async for chunk in srv.stream_chat_completion(
                _FailingChatEngine(emit=0),
                request.messages,
                request,
                metrics_tracker=tracker,
            ):
                chunks.append(chunk)

    body = "".join(chunks)
    errors = [p for p in _sse_payloads(chunks) if "error" in p]
    assert len(errors) == 1, "client must be told the stream failed"
    assert errors[0]["error"]["code"] == "stream_failed"
    assert body.rstrip().endswith("data: [DONE]"), "stream must still terminate"

    # the traceback is logged (this path used to be completely silent)
    assert BOOM_TEXT in caplog.text
    assert "_Boom" in caplog.text

    # ...but never leaks to the client
    assert BOOM_TEXT not in body

    assert tracker.calls and tracker.calls[0]["result"] == "error"
    assert tracker.calls[0]["completion_tokens"] == 0


@pytest.mark.anyio
async def test_chat_stream_emits_error_chunk_mid_stream(monkeypatch):
    _prepare(monkeypatch)
    request = _chat_request()
    tracker = _RecordingTracker()
    chunks = []

    with pytest.raises(_Boom):
        async for chunk in srv.stream_chat_completion(
            _FailingChatEngine(emit=2),
            request.messages,
            request,
            metrics_tracker=tracker,
        ):
            chunks.append(chunk)

    assert any("error" in p for p in _sse_payloads(chunks))
    assert tracker.calls[0]["result"] == "error"
    assert tracker.calls[0]["completion_tokens"] > 0


@pytest.mark.anyio
async def test_chat_stream_cancellation_emits_no_error_chunk(monkeypatch):
    """A client hanging up is not a server error, and yielding during
    GeneratorExit would raise "async generator ignored GeneratorExit"."""
    _prepare(monkeypatch)
    request = _chat_request()
    tracker = _RecordingTracker()

    class _Slow:
        model_name = "test-model"

        async def stream_chat(self, messages, **kwargs):
            for i in range(100):
                yield GenerationOutput(
                    text="x",
                    new_text="x",
                    finished=False,
                    finish_reason=None,
                    prompt_tokens=1,
                    completion_tokens=i + 1,
                )

    agen = srv.stream_chat_completion(
        _Slow(), request.messages, request, metrics_tracker=tracker
    )
    seen = [await agen.__anext__(), await agen.__anext__()]
    await agen.aclose()  # must not raise

    assert not any("error" in p for p in _sse_payloads(seen))
    assert tracker.calls and tracker.calls[0]["result"] == "cancelled"


# --------------------------------------------------- text completions stream


class _FailingCompletionEngine:
    model_name = "test-model"

    async def stream_generate(self, **kwargs):
        raise _Boom(BOOM_TEXT)
        yield  # pragma: no cover — makes this an async generator


@pytest.mark.anyio
async def test_completion_stream_emits_error_chunk(monkeypatch):
    request = srv.CompletionRequest(model="test-model", prompt="hi", stream=True)
    tracker = _RecordingTracker()
    chunks = []

    with pytest.raises(_Boom):
        async for chunk in srv.stream_completion(
            _FailingCompletionEngine(),
            "hi",
            request,
            max_tokens=16,
            metrics_tracker=tracker,
        ):
            chunks.append(chunk)

    body = "".join(chunks)
    assert any("error" in p for p in _sse_payloads(chunks))
    assert "[DONE]" in body
    assert BOOM_TEXT not in body
    assert tracker.calls[0]["result"] == "error"


@pytest.mark.anyio
async def test_completion_stream_close_does_not_lose_metrics(monkeypatch):
    """#91 regression: the [DONE] yield lived in `finally`, so closing the
    generator raised "async generator ignored GeneratorExit" and skipped
    metrics_tracker.finish() with it — aborted completions went uncounted."""
    tracker = _RecordingTracker()
    request = srv.CompletionRequest(model="test-model", prompt="hi", stream=True)

    class _Slow:
        model_name = "test-model"

        async def stream_generate(self, **kwargs):
            for i in range(100):
                yield GenerationOutput(
                    text="x",
                    new_text="x",
                    finished=False,
                    finish_reason=None,
                    prompt_tokens=1,
                    completion_tokens=i + 1,
                )

    agen = srv.stream_completion(
        _Slow(), "hi", request, max_tokens=16, metrics_tracker=tracker
    )
    await agen.__anext__()
    await agen.aclose()  # used to raise RuntimeError

    assert tracker.calls, "finish() must still run when the client hangs up"
    assert tracker.calls[0]["result"] == "cancelled"


# ------------------------------------------------------- #91 metrics counter


class TestStreamAbortCounter:
    def _collector(self):
        pytest.importorskip("prometheus_client")
        from vllm_mlx.metrics import MetricsCollector

        collector = MetricsCollector()
        collector.configure(enabled=True)
        collector._init_prometheus()
        return collector

    def _render(self, collector):
        return collector.render_metrics(engine=None, mcp_manager=None)[0].decode()

    def test_raise_before_first_token_is_the_alertable_signature(self):
        collector = self._collector()
        collector.observe_inference(
            endpoint="chat",
            stream=True,
            result="error",
            duration=0.1,
            prompt_tokens=11050,
            completion_tokens=0,
        )
        assert (
            'vllm_mlx_stream_aborts_total{endpoint="chat",phase="before_first_token"'
            ',result="error"} 1.0' in self._render(collector)
        )

    def test_mid_stream_abort_is_a_separate_phase(self):
        collector = self._collector()
        collector.observe_inference(
            endpoint="chat",
            stream=True,
            result="error",
            duration=0.1,
            prompt_tokens=10,
            completion_tokens=7,
        )
        assert (
            'vllm_mlx_stream_aborts_total{endpoint="chat",phase="mid_stream"'
            ',result="error"} 1.0' in self._render(collector)
        )

    def test_cancelled_is_counted_but_distinguishable(self):
        collector = self._collector()
        collector.observe_inference(
            endpoint="chat",
            stream=True,
            result="cancelled",
            duration=0.1,
            prompt_tokens=10,
            completion_tokens=0,
        )
        text = self._render(collector)
        assert 'result="cancelled"' in text
        assert 'result="error"' not in text

    def test_success_and_non_streaming_are_not_counted(self):
        collector = self._collector()
        collector.observe_inference(
            endpoint="chat",
            stream=True,
            result="success",
            duration=0.1,
            prompt_tokens=10,
            completion_tokens=0,
        )
        collector.observe_inference(
            endpoint="chat",
            stream=False,
            result="error",
            duration=0.1,
            prompt_tokens=10,
            completion_tokens=0,
        )
        # the bare name also appears in the HELP line — assert on sample lines
        assert "vllm_mlx_stream_aborts_total{" not in self._render(collector)

    def test_87_still_only_counts_successes(self):
        """Guard the split: #87 keeps its meaning, #91 covers the rest."""
        collector = self._collector()
        collector.observe_inference(
            endpoint="chat",
            stream=True,
            result="error",
            duration=0.1,
            prompt_tokens=10,
            completion_tokens=0,
        )
        text = self._render(collector)
        assert 'vllm_mlx_empty_completions_total{endpoint="chat"}' not in text
        assert "vllm_mlx_stream_aborts_total{" in text
