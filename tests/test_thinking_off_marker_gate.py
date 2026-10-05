# SPDX-License-Identifier: Apache-2.0
"""Fork #123: a reasoning marker split across deltas never leaks as content."""

import pytest

import vllm_mlx.server as srv
from vllm_mlx.reasoning import get_parser


def _gate(name="deepseek_r1", thinking_off=True):
    return srv._ThinkingOffMarkerGate(get_parser(name)(), thinking_off)


def _run(gate, deltas):
    out = []
    for i, d in enumerate(deltas):
        gate.observe(d)
        out.append(gate.hold(d, finished=i == len(deltas) - 1))
    return out


def test_split_start_marker_is_held_then_released_to_the_parser():
    gate = _gate()
    out = _run(gate, ["<think", ">\\nOkay", " so", "</think>Hi"])
    assert out[0] == ""  # the fragment is not emitted as content
    assert out[1] == "<think>\\nOkay"  # released whole once the latch fires
    assert gate.latched


def test_false_alarm_is_released_unchanged():
    gate = _gate()
    out = _run(gate, ["a <th", "ings> b", " c"])
    assert "".join(out) == "a <things> b c" and not gate.latched
    assert out[0] == "a "  # only the ambiguous tail was held


def test_held_tail_flushes_at_end_of_stream():
    gate = _gate()
    assert _run(gate, ["text <thi"]) == ["text <thi"]


def test_thinking_on_is_untouched():
    gate = _gate(thinking_off=False)
    assert _run(gate, ["<think", ">x"]) == ["<think", ">x"]


def test_special_token_marker_latches_on_raw_text():
    gate = _gate("harmony")
    gate.observe("<|channel|>analysis<|message|>")
    assert gate.latched


@pytest.mark.parametrize("name", ["glm4", "qwen3", "deepseek_r1"])
def test_end_only_marker_split(name):
    gate = _gate(name)
    out = _run(gate, ["reasoning </thi", "nk>Answer"])
    assert out[0] == "reasoning " and out[1] == "</think>Answer" and gate.latched


# --- end to end through the three stream endpoints --------------------------

import json  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

SPLIT = ("<think", ">\nOkay so", " the user", "</think>", "Hello")
HARMONY = (
    "<|channel|>analysis<|message|>",
    "We need to answer.",
    "<|end|><|start|>assistant<|channel|>final<|message|>",
    "Red",
)


def _engine(deltas):
    async def fake_stream_chat(messages, **kwargs):
        last = len(deltas) - 1
        for i, piece in enumerate(deltas):
            yield SimpleNamespace(
                new_text=piece,
                prompt_tokens=4,
                completion_tokens=i + 1,
                finished=i == last,
                finish_reason="stop" if i == last else None,
            )

    return MagicMock(stream_chat=fake_stream_chat)


def _chat_request():
    return srv.ChatCompletionRequest(
        model="test-model",
        messages=[srv.Message(role="user", content="hi")],
        max_tokens=16,
    )


def _with_parser(name):
    saved = (srv._reasoning_parser, srv._model_name)
    srv._reasoning_parser, srv._model_name = get_parser(name)(), "test-model"
    return saved


def _restore(saved):
    srv._reasoning_parser, srv._model_name = saved


@pytest.mark.anyio
async def test_chat_completions_split_marker_never_leaks():
    saved = _with_parser("deepseek_r1")
    try:
        request = _chat_request()
        body = "".join(
            [
                c
                async for c in srv.stream_chat_completion(
                    _engine(SPLIT),
                    request.messages,
                    request,
                    chat_template_kwargs={"enable_thinking": False},
                )
            ]
        )
    finally:
        _restore(saved)
    content = ""
    for line in body.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            for ch in json.loads(line[6:]).get("choices", []):
                content += ch.get("delta", {}).get("content") or ""
    assert content == "Hello"


async def _anthropic_body(deltas, parser):
    msgs = [{"role": "user", "content": "hi"}]
    prepared = srv.PreparedChatInvocation(
        messages=msgs,
        chat_kwargs={"chat_template_kwargs": {"enable_thinking": False}},
        response_format=None,
        json_logits_processor=None,
    )
    saved = _with_parser(parser)
    try:
        return "".join(
            [
                c
                async for c in srv._stream_anthropic_messages(
                    _engine(deltas),
                    srv.ChatCompletionRequest(
                        model="test-model", messages=[srv.Message(**msgs[0])]
                    ),
                    srv.AnthropicRequest(
                        model="test-model", max_tokens=16, messages=msgs
                    ),
                    prepared,
                )
            ]
        )
    finally:
        _restore(saved)


def _anthropic_text(body):
    text = ""
    for line in body.splitlines():
        if line.startswith("data: "):
            delta = json.loads(line[6:]).get("delta", {})
            if delta.get("type") == "text_delta":
                text += delta.get("text", "")
    return text


@pytest.mark.anyio
async def test_anthropic_split_marker_never_leaks():
    body = await _anthropic_body(SPLIT, "deepseek_r1")
    assert _anthropic_text(body) == "Hello"
    assert "thinking_delta" not in body


@pytest.mark.anyio
async def test_anthropic_harmony_thinking_off_keeps_analysis_out():
    """The latch used to read special-token-stripped text, so harmony's
    <|channel|> could never fire it on /v1/messages."""
    body = await _anthropic_body(HARMONY, "harmony")
    assert _anthropic_text(body) == "Red"


@pytest.mark.anyio
async def test_responses_split_marker_never_leaks():
    request = srv.ResponsesRequest(model="test-model", input="hi", stream=True)
    msgs = [{"role": "user", "content": "hi"}]
    saved = _with_parser("deepseek_r1")
    try:
        with patch.object(
            srv,
            "_prepare_streaming_responses_request",
            return_value=(
                _engine(SPLIT),
                _chat_request(),
                msgs,
                {"chat_template_kwargs": {"enable_thinking": False}},
            ),
        ):
            body = "".join([c async for c in srv._stream_responses_request(request)])
    finally:
        _restore(saved)
    text = ""
    for line in body.splitlines():
        if line.startswith("data: "):
            event = json.loads(line[6:])
            if event.get("type") == "response.output_text.delta":
                text += event["delta"]
    assert text == "Hello"
