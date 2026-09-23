# SPDX-License-Identifier: Apache-2.0
"""gpt-oss streamed tool calls through the real OpenAI streaming loop (fork #112).

The delta sequences below were captured from gpt-oss-20b-MXFP4-Q4 on the Mac
Studio (2026-09-23, BatchedEngine, T=0): control tokens arrive as their own
deltas, the channel name is split (``comment`` + ``ary``), and ``<|call|>`` —
the stop token — never appears in the text. Before the fix, the harmony
reasoning parser only noticed a channel switch when ``<|channel|>`` and the
channel name shared a delta, so the commentary message (the tool call's
arguments) streamed out as *reasoning*; and the harmony tool parser only saw
the reasoning parser's marker-stripped content, so no tool call was emitted
(``finish_reason: stop``). Non-streaming was unaffected.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import vllm_mlx.server as srv
from vllm_mlx.api.models import ChatCompletionRequest
from vllm_mlx.reasoning.harmony_parser import HarmonyReasoningParser
from vllm_mlx.tool_parsers.harmony_tool_parser import HarmonyToolParser

# fmt: off
TOOL_CALL_DELTAS = [
    "<|channel|>", "analysis", "<|message|>", "We", " need", " to", " call",
    " the", " function", " read", " with", " path", ' "', "a", ".txt", '".',
    "<|end|>", "<|start|>", "assistant", "<|channel|>", "comment", "ary", " to",
    "=", "functions", ".read", " ", "<|constrain|>", "json", "<|message|>",
    '{"', "path", '":"', "a", ".txt", '"}', "",
]
ANALYSIS = 'We need to call the function read with path "a.txt".'

FINAL_ANSWER_DELTAS = [
    "<|channel|>", "analysis", "<|message|>", "User", " wants", " a", " fact",
    ".", "<|end|>", "<|start|>", "assistant", "<|channel|>", "final",
    "<|message|>", "The", " sky", " is", " blue", ".", "",
]
# fmt: on

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]


@pytest.fixture()
def harmony_route(monkeypatch):
    monkeypatch.setattr(srv, "_reasoning_parser_name", "harmony", raising=False)
    monkeypatch.setattr(srv, "_reasoning_parser", None, raising=False)
    monkeypatch.setattr(
        srv, "_detect_implicit_thinking", lambda *a, **k: False, raising=False
    )
    monkeypatch.setattr(
        srv,
        "_get_streaming_tool_parser",
        # Like the real one: no tools on the request, no tool parser.
        lambda request, *a, **k: (
            HarmonyToolParser(None) if getattr(request, "tools", None) else None
        ),
        raising=False,
    )


def _engine(deltas):
    engine = MagicMock()
    engine.model_name = "gpt-oss-20b"
    engine.preserve_native_tool_format = False

    async def _stream_chat(**kwargs):
        text = ""
        for i, delta in enumerate(deltas):
            text += delta
            last = i == len(deltas) - 1
            yield SimpleNamespace(
                new_text=delta,
                text=text,
                prompt_tokens=115,
                completion_tokens=i + 1,
                finish_reason="stop" if last else None,
                finished=last,
            )

    engine.stream_chat = _stream_chat
    return engine


async def _run(deltas, **request_kwargs):
    request = ChatCompletionRequest(
        model="gpt-oss-20b",
        messages=[{"role": "user", "content": "Read the file a.txt."}],
        stream=True,
        **request_kwargs,
    )
    chunks = [
        c
        async for c in srv.stream_chat_completion(
            _engine(deltas), request.messages, request
        )
    ]
    out = {"content": "", "reasoning": "", "calls": {}, "finish": []}
    for raw in chunks:
        for line in raw.splitlines():
            body = line.removeprefix("data: ").strip()
            if not line.startswith("data: ") or body == "[DONE]":
                continue
            for choice in json.loads(body).get("choices") or []:
                delta = choice.get("delta") or {}
                out["content"] += delta.get("content") or ""
                out["reasoning"] += (
                    delta.get("reasoning") or delta.get("reasoning_content") or ""
                )
                for tc in delta.get("tool_calls") or []:
                    call = out["calls"].setdefault(tc["index"], ["", ""])
                    call[0] += tc["function"].get("name") or ""
                    call[1] += tc["function"].get("arguments") or ""
                if choice.get("finish_reason"):
                    out["finish"].append(choice["finish_reason"])
    return out


@pytest.mark.anyio
async def test_streamed_tool_call_is_emitted(harmony_route):
    out = await _run(TOOL_CALL_DELTAS, tools=TOOLS)

    assert list(out["calls"]) == [0]
    name, arguments = out["calls"][0]
    assert name == "read"
    assert json.loads(arguments) == {"path": "a.txt"}
    assert out["finish"] == ["tool_calls"]


@pytest.mark.anyio
async def test_commentary_does_not_leak_into_reasoning_or_content(harmony_route):
    out = await _run(TOOL_CALL_DELTAS, tools=TOOLS)

    assert out["reasoning"] == ANALYSIS
    assert "path" not in out["content"]
    assert "<|" not in out["content"] + out["reasoning"]


@pytest.mark.anyio
async def test_final_answer_streams_as_content(harmony_route):
    """The same route with tools offered but a plain answer: final channel is
    content, analysis is reasoning, no tool call."""
    out = await _run(FINAL_ANSWER_DELTAS, tools=TOOLS)

    assert out["content"] == "The sky is blue."
    assert out["reasoning"] == "User wants a fact."
    assert out["calls"] == {}
    assert out["finish"] == ["stop"]


@pytest.mark.anyio
async def test_final_answer_without_tools(harmony_route):
    """No tools offered: no tool parser at all; the reasoning parser alone."""
    out = await _run(FINAL_ANSWER_DELTAS)

    assert out["content"] == "The sky is blue."
    assert out["reasoning"] == "User wants a fact."


# ----------------------------------------- the reasoning parser on its own
def _parse(deltas):
    parser = HarmonyReasoningParser()
    parser.reset_state()
    text = ""
    reasoning = content = ""
    for delta in deltas:
        previous = text
        text += delta
        msg = parser.extract_reasoning_streaming(previous, text, delta)
        if msg is not None:
            reasoning += msg.reasoning or ""
            content += msg.content or ""
    return reasoning, content


def test_reasoning_parser_follows_split_channel_names():
    reasoning, content = _parse(TOOL_CALL_DELTAS)
    assert reasoning == ANALYSIS  # the commentary arguments are not reasoning
    assert content == ""


def test_reasoning_parser_final_channel():
    reasoning, content = _parse(FINAL_ANSWER_DELTAS)
    assert reasoning == "User wants a fact."
    assert content == "The sky is blue."


# ------------------------------------ the Anthropic and Responses front ends
@pytest.fixture()
def harmony_globals(monkeypatch):
    """Both paths build their final tool calls through the module-level
    parser config (`_parse_tool_calls_with_parser`), like the real route."""
    monkeypatch.setattr(srv, "_reasoning_parser_name", "harmony", raising=False)
    monkeypatch.setattr(srv, "_reasoning_parser", None, raising=False)
    monkeypatch.setattr(srv, "_enable_auto_tool_choice", True, raising=False)
    monkeypatch.setattr(srv, "_tool_call_parser", "harmony", raising=False)
    monkeypatch.setattr(srv, "_tool_parser_instance", None, raising=False)
    monkeypatch.setattr(srv, "_model_name", "gpt-oss-20b", raising=False)
    monkeypatch.setattr(
        srv, "_detect_implicit_thinking", lambda *a, **k: False, raising=False
    )


async def _anthropic(deltas):
    msgs = [{"role": "user", "content": "Read a.txt"}]
    openai_request = ChatCompletionRequest(
        model="m", messages=msgs, max_tokens=400, tools=TOOLS
    )
    anthropic_request = srv.AnthropicRequest(
        model="m",
        max_tokens=400,
        messages=msgs,
        tools=[
            {
                "name": "read",
                "description": "Read a file",
                "input_schema": TOOLS[0]["function"]["parameters"],
            }
        ],
    )
    prepared = srv.PreparedChatInvocation(
        messages=msgs, chat_kwargs={}, response_format=None, json_logits_processor=None
    )
    body = "".join(
        [
            c
            async for c in srv._stream_anthropic_messages(
                _engine(deltas), openai_request, anthropic_request, prepared
            )
        ]
    )
    events = [json.loads(x[6:]) for x in body.splitlines() if x.startswith("data: ")]
    deltas_ = [e["delta"] for e in events if e.get("type") == "content_block_delta"]
    return {
        "text": "".join(d.get("text", "") for d in deltas_),
        "thinking": "".join(d.get("thinking", "") for d in deltas_),
        "tool_use": [
            e["content_block"]["name"]
            for e in events
            if e.get("type") == "content_block_start"
            and e["content_block"]["type"] == "tool_use"
        ],
        "args": "".join(d.get("partial_json", "") for d in deltas_),
        "stop": [
            e["delta"].get("stop_reason")
            for e in events
            if e.get("type") == "message_delta"
        ],
    }


async def _responses(deltas):
    from unittest.mock import patch

    request = srv.ResponsesRequest(
        model="m",
        input="Read a.txt",
        stream=True,
        tools=[
            {
                "type": "function",
                "name": "read",
                "description": "Read a file",
                "parameters": TOOLS[0]["function"]["parameters"],
            }
        ],
    )
    chat_request = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "Read a.txt"}],
        max_tokens=400,
        tools=TOOLS,
    )
    prepared = (
        _engine(deltas),
        chat_request,
        [{"role": "user", "content": "Read a.txt"}],
        {},
    )
    with patch.object(
        srv, "_prepare_streaming_responses_request", return_value=prepared
    ):
        body = "".join([c async for c in srv._stream_responses_request(request)])
    events = [json.loads(x[6:]) for x in body.splitlines() if x.startswith("data: ")]
    return {
        "text": "".join(
            e.get("delta", "")
            for e in events
            if e.get("type") == "response.output_text.delta"
        ),
        "reasoning": "".join(
            e.get("delta", "")
            for e in events
            if e.get("type", "").startswith("response.reasoning")
            and e.get("type", "").endswith(".delta")
        ),
        "calls": [
            (e["item"]["name"], json.loads(e["item"]["arguments"]))
            for e in events
            if e.get("type") == "response.output_item.done"
            and e["item"].get("type") == "function_call"
        ],
    }


@pytest.mark.anyio
async def test_anthropic_stream_tool_call(harmony_globals):
    # Before the fix: SPECIAL_TOKENS_PATTERN stripped the control tokens ahead
    # of the reasoning parser, so NOTHING streamed — no thinking, no tool_use.
    out = await _anthropic(TOOL_CALL_DELTAS)
    assert out["thinking"] == ANALYSIS
    assert out["tool_use"] == ["read"]
    assert json.loads(out["args"]) == {"path": "a.txt"}
    assert out["stop"] == ["tool_use"]
    assert out["text"] == ""


@pytest.mark.anyio
async def test_anthropic_stream_final_answer(harmony_globals):
    out = await _anthropic(FINAL_ANSWER_DELTAS)
    assert out["text"] == "The sky is blue."
    assert out["thinking"] == "User wants a fact."
    assert out["tool_use"] == []


@pytest.mark.anyio
async def test_responses_stream_tool_call(harmony_globals):
    out = await _responses(TOOL_CALL_DELTAS)
    assert out["reasoning"] == ANALYSIS  # args used to be appended here
    assert out["calls"] == [("read", {"path": "a.txt"})]
    assert out["text"] == ""


@pytest.mark.anyio
async def test_responses_stream_final_answer(harmony_globals):
    out = await _responses(FINAL_ANSWER_DELTAS)
    assert out["text"] == "The sky is blue."  # used to be ""
    assert out["reasoning"] == "User wants a fact."
