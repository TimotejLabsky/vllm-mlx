# SPDX-License-Identifier: Apache-2.0
"""Streaming parsers emit every tool call exactly once (fork #110).

Each parser re-parses the whole accumulated text when a call block closes, so
``result.tool_calls`` holds every call so far. Emitting all of them again —
at index 0 with a fresh id — makes OpenAI clients (which accumulate by index)
append the arguments onto the first call. Upstream #774 fixed this in
qwen_tool_parser only; the fork routes glm47, gemma4, nemotron, harmony and
auto, so every parser gets the same guarantee through
``ToolParser._stream_new_tool_calls``.
"""

import json

import pytest

from vllm_mlx.tool_parsers import ToolParserManager

A = {"path": "a"}
B = {"path": "b"}

# Two calls per parser in its native streaming shape: (first, second).
FORMATS = {
    "glm47": (
        "<tool_call>read<arg_key>path</arg_key><arg_value>a</arg_value></tool_call>",
        "<tool_call>read<arg_key>path</arg_key><arg_value>b</arg_value></tool_call>",
    ),
    "gemma4": (
        '<|tool_call>call:read{path:<|"|>a<|"|>}<tool_call|>',
        '<|tool_call>call:read{path:<|"|>b<|"|>}<tool_call|>',
    ),
    "nemotron": (
        "<tool_call><function=read><parameter=path>a</parameter></function></tool_call>",
        "<tool_call><function=read><parameter=path>b</parameter></function></tool_call>",
    ),
    "harmony": (
        "<|channel|>commentary to=functions.read <|constrain|>json"
        '<|message|>{"path": "a"}<|call|>',
        "<|start|>assistant<|channel|>commentary to=functions.read "
        '<|constrain|>json<|message|>{"path": "b"}<|call|>',
    ),
    "auto": (
        '<tool_call>{"name": "read", "arguments": {"path": "a"}}</tool_call>',
        '<tool_call>{"name": "read", "arguments": {"path": "b"}}</tool_call>',
    ),
    "kimi": (
        "<|tool_calls_section_begin|><|tool_call_begin|>functions.read:0"
        '<|tool_call_argument_begin|>{"path": "a"}<|tool_call_end|>',
        "<|tool_call_begin|>functions.read:1<|tool_call_argument_begin|>"
        '{"path": "b"}<|tool_call_end|><|tool_calls_section_end|>',
    ),
    "minimax": (
        '<minimax:tool_call>\n<invoke name="read">\n'
        '<parameter name="path">a</parameter>\n</invoke>\n</minimax:tool_call>',
        '<minimax:tool_call>\n<invoke name="read">\n'
        '<parameter name="path">b</parameter>\n</invoke>\n</minimax:tool_call>',
    ),
    "deepseek": (
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>read\n"
        '```json\n{"path": "a"}\n```<｜tool▁call▁end｜>',
        "<｜tool▁call▁begin｜>function<｜tool▁sep｜>read\n"
        '```json\n{"path": "b"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
    ),
    "deepseek_v4": (
        '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="read">\n'
        '<｜DSML｜parameter name="path" string="true">a</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n",
        '<｜DSML｜invoke name="read">\n'
        '<｜DSML｜parameter name="path" string="true">b</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
    ),
    "functionary": (
        '<function=read>{"path": "a"}</function>',
        '<function=read>{"path": "b"}</function>',
    ),
    "granite": (
        '<|tool_call|>[{"name": "read", "arguments": {"path": "a"}}, ',
        '{"name": "read", "arguments": {"path": "b"}}]',
    ),
    "xlam": (
        '[{"name": "read", "arguments": {"path": "a"}}, ',
        '{"name": "read", "arguments": {"path": "b"}}]',
    ),
}


def _stream(name, deltas):
    """Feed deltas like the server does; collect every emitted tool call."""
    parser = ToolParserManager.get_tool_parser(name)(None)
    text = ""
    calls = []
    for delta in deltas:
        previous = text
        text += delta
        result = parser.extract_tool_calls_streaming(previous, text, delta)
        calls.extend((result or {}).get("tool_calls") or [])
    finalize = getattr(parser, "finalize_streaming", None)
    if finalize is not None:
        calls.extend((finalize(text) or {}).get("tool_calls") or [])
    return calls, parser


@pytest.mark.parametrize("name", sorted(FORMATS))
def test_each_call_is_emitted_once_with_increasing_indexes(name):
    first, second = FORMATS[name]
    # Two calls completing in sequence, each followed by trailing whitespace
    # and the engine's final empty delta.
    calls, _ = _stream(name, [first, "\n", "", second, "\n", ""])

    assert [c["index"] for c in calls] == [0, 1]
    assert len({c["id"] for c in calls}) == 2
    assert [c["function"]["name"] for c in calls] == ["read", "read"]
    assert [json.loads(c["function"]["arguments"]) for c in calls] == [A, B]


@pytest.mark.parametrize("name", sorted(FORMATS))
def test_reset_starts_indexes_over(name):
    first, second = FORMATS[name]
    _, parser = _stream(name, [first, second])
    parser.reset()
    text = first + second
    result = parser.extract_tool_calls_streaming("", text, text)
    calls = (result or {}).get("tool_calls") or []
    if not calls and getattr(parser, "finalize_streaming", None):
        calls = (parser.finalize_streaming(text) or {}).get("tool_calls") or []
    assert [c["index"] for c in calls] == [0, 1]


def test_harmony_two_identical_calls_are_two_invocations():
    # The old (name, arguments) signature dedupe swallowed the second one.
    first, _ = FORMATS["harmony"]
    again = "<|start|>assistant" + first
    calls, _ = _stream("harmony", [first, "\n", again, ""])
    assert [c["index"] for c in calls] == [0, 1]
    assert [json.loads(c["function"]["arguments"]) for c in calls] == [A, A]


def test_gemma4_non_streaming_reads_every_block():
    # The Gemma 4 template renders each parallel call as its own block; the
    # parser used to stop after the first.
    parser = ToolParserManager.get_tool_parser("gemma4")(None)
    first, second = FORMATS["gemma4"]
    result = parser.extract_tool_calls("Checking both. " + first + second)
    assert result.tools_called
    assert [json.loads(c["arguments"]) for c in result.tool_calls] == [A, B]
    assert result.content == "Checking both."


def test_minimax_second_wrapper_block_fires():
    # "</minimax:tool_call> not in previous" only ever fired for the first block.
    first, second = FORMATS["minimax"]
    calls, _ = _stream("minimax", [first, second])
    assert [json.loads(c["function"]["arguments"]) for c in calls] == [A, B]


def test_deepseek_v4_finalize_does_not_resend_streamed_calls():
    # Text after the closed block made finalize_streaming re-parse and send
    # every call a second time.
    first, second = FORMATS["deepseek_v4"]
    calls, _ = _stream("deepseek_v4", [first + second, "\n"])
    assert [c["index"] for c in calls] == [0, 1]


@pytest.mark.anyio
@pytest.mark.parametrize("split_second_close", [False, True])
@pytest.mark.parametrize("name", ["glm47", "gemma4", "nemotron", "auto"])
async def test_openai_stream_carries_each_call_once(
    name, split_second_close, monkeypatch
):
    """Through the real server loop: the SSE stream an OpenAI client
    accumulates by index gets call 0 and call 1, once each."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import vllm_mlx.server as srv
    from vllm_mlx.api.models import ChatCompletionRequest

    parser = ToolParserManager.get_tool_parser(name)(None)
    monkeypatch.setattr(srv, "_get_streaming_tool_parser", lambda *a, **k: parser)
    monkeypatch.setattr(srv, "_reasoning_parser", None, raising=False)

    first, second = FORMATS[name]
    if split_second_close:
        # #111: the first call streamed, so the server's end-of-stream
        # fallback is skipped — the parser itself has to see the split close.
        deltas = [first, "\n", second[:-3], second[-3:], ""]
    else:
        deltas = [first, "\n", second, "\n", ""]

    def _out(i, text):
        last = i == len(deltas) - 1
        return SimpleNamespace(
            new_text=text,
            text=text,
            prompt_tokens=7,
            completion_tokens=i + 1,
            finish_reason="stop" if last else None,
            finished=last,
        )

    engine = MagicMock()
    engine.model_name = "test-model"
    engine.preserve_native_tool_format = False

    async def _stream_chat(**kwargs):
        for i, text in enumerate(deltas):
            yield _out(i, text)

    engine.stream_chat = _stream_chat
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "read a and b"}],
        stream=True,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                },
            }
        ],
    )
    chunks = [
        c async for c in srv.stream_chat_completion(engine, request.messages, request)
    ]
    calls = []
    for raw in chunks:
        for line in raw.splitlines():
            body = line.removeprefix("data: ").strip()
            if not line.startswith("data: ") or body == "[DONE]":
                continue
            payload = json.loads(body)
            for choice in payload.get("choices") or []:
                calls.extend((choice.get("delta") or {}).get("tool_calls") or [])

    assert [c["index"] for c in calls] == [0, 1]
    assert [json.loads(c["function"]["arguments"]) for c in calls] == [A, B]


# ------------------------------------------- #111 split end markers are seen
def _chunked(text, size):
    return [text[i : i + size] for i in range(0, len(text), size)]


@pytest.mark.parametrize("size", [1, 2, 3, 5, 7])
@pytest.mark.parametrize("name", sorted(FORMATS))
def test_calls_survive_any_chunking(name, size):
    """A closing marker split across deltas never appears in one delta_text;
    triggering on it lost the call (fork #111)."""
    first, second = FORMATS[name]
    calls, _ = _stream(name, _chunked(first + "\n" + second, size) + ["\n", ""])

    assert [c["index"] for c in calls] == [0, 1]
    assert [json.loads(c["function"]["arguments"]) for c in calls] == [A, B]


@pytest.mark.parametrize("name", sorted(FORMATS))
def test_split_close_on_the_second_call_only(name):
    """The case the server's end-of-stream fallback cannot rescue: the first
    call already streamed, so the fallback is skipped."""
    first, second = FORMATS[name]
    calls, _ = _stream(name, [first, "\n", second[:-3], second[-3:], ""])

    assert [c["index"] for c in calls] == [0, 1]
    assert [json.loads(c["function"]["arguments"]) for c in calls] == [A, B]


@pytest.mark.parametrize(
    "previous, current, expected",
    [
        ("ab</tool", "ab</tool_call>", True),  # completed across the boundary
        ("ab</tool_call>", "ab</tool_call>\n", False),  # already complete before
        ("", "</tool_call>", True),  # whole marker in one delta
        ("<tool_call>x", "<tool_call>xy", False),  # no marker
        ("a</tool_call>b</tool_", "a</tool_call>b</tool_call>", True),  # 2nd
    ],
)
def test_marker_completed_window(previous, current, expected):
    from vllm_mlx.tool_parsers.abstract_tool_parser import ToolParser

    assert (
        ToolParser._marker_completed(previous, current, ("</tool_call>",)) is expected
    )
