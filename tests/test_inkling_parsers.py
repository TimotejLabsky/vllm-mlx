# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 message-block parsers (reasoning + tool calls), fork #182."""

import json

import pytest

from vllm_mlx.inkling_blocks import InklingBlockScanner, scan
from vllm_mlx.reasoning import get_parser
from vllm_mlx.tool_parsers import ToolParserManager

THINK = "<|message_model|><|content_thinking|>Check the forecast.<|end_message|>"
TEXT = "<|message_model|><|content_text|>It is sunny.<|end_message|>"
CALL = (
    "<|message_model|>get_weather<|content_invoke_tool_json|>"
    '{"name":"get_weather","args":{"city":"Paris","days":2}}<|end_message|>'
)
CALL2 = (
    "<|message_model|>get_time<|content_invoke_tool_json|>"
    '{"name":"get_time","args":{}}<|end_message|>'
)
EOS = "<|content_model_end_sampling|>"


def _tool_parser():
    parser = ToolParserManager.get_tool_parser("inkling")()
    parser.reset()
    return parser


def _reasoning_parser():
    parser = get_parser("inkling")()
    parser.reset_state()
    return parser


def _chunks(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def _stream_tools(text: str, size: int):
    """Drive the tool parser the way the server does; return (content, calls)."""
    parser = _tool_parser()
    content, calls, seen = "", [], ""
    for delta in _chunks(text, size):
        result = parser.extract_tool_calls_streaming(seen, seen + delta, delta)
        seen += delta
        if result:
            content += result.get("content", "")
            calls += result.get("tool_calls", [])
    final = parser.finalize_streaming(seen)
    if final:
        content += final.get("content", "")
        calls += final.get("tool_calls", [])
    return content, calls


def _stream_reasoning(text: str, size: int):
    parser = _reasoning_parser()
    reasoning, content, seen = "", "", ""
    for delta in _chunks(text, size):
        msg = parser.extract_reasoning_streaming(seen, seen + delta, delta)
        seen += delta
        if msg:
            reasoning += msg.reasoning or ""
            content += msg.content or ""
    final = parser.finalize_stream()
    if final:
        reasoning += final.reasoning or ""
        content += final.content or ""
    return reasoning, content


class TestRegistration:
    def test_both_parsers_registered(self):
        assert ToolParserManager.get_tool_parser("inkling") is (
            ToolParserManager.get_tool_parser("inkling2")
        )
        assert get_parser("inkling").__name__ == "InklingReasoningParser"

    def test_both_read_the_raw_stream(self):
        assert ToolParserManager.get_tool_parser("inkling").CONSUMES_RAW_STREAM
        assert get_parser("inkling").CONSUMES_RAW_STREAM


class TestNonStreaming:
    def test_thinking_then_text(self):
        reasoning, content = _reasoning_parser().extract_reasoning(THINK + TEXT + EOS)
        assert reasoning == "Check the forecast."
        assert content == "It is sunny."

    def test_tool_call(self):
        info = _tool_parser().extract_tool_calls(THINK + CALL + EOS)
        assert info.tools_called
        assert info.content is None
        (call,) = info.tool_calls
        assert call["name"] == "get_weather"
        assert json.loads(call["arguments"]) == {"city": "Paris", "days": 2}

    def test_text_and_two_calls(self):
        info = _tool_parser().extract_tool_calls(TEXT + CALL + CALL2 + EOS)
        assert info.content == "It is sunny."
        assert [c["name"] for c in info.tool_calls] == ["get_weather", "get_time"]
        assert info.tool_calls[1]["arguments"] == "{}"

    def test_reasoning_parser_leaves_calls_out_of_content(self):
        reasoning, content = _reasoning_parser().extract_reasoning(THINK + CALL + EOS)
        assert reasoning == "Check the forecast."
        assert content is None

    def test_header_is_optional(self):
        text = "<|content_thinking|>hm<|end_message|><|content_text|>ok<|end_message|>"
        assert _reasoning_parser().extract_reasoning(text) == ("hm", "ok")

    def test_repeated_blocks_join(self):
        text = (
            "<|message_model|><|content_thinking|>a<|end_message|>"
            "<|message_model|><|content_text|>one <|end_message|>"
            "<|message_model|><|content_thinking|>b<|end_message|>"
            "<|message_model|><|content_text|>two<|end_message|>"
        )
        assert _reasoning_parser().extract_reasoning(text) == ("ab", "one two")

    def test_name_from_header_when_json_has_none(self):
        text = (
            "<|message_model|>lookup<|content_invoke_tool_json|>"
            '{"args":{"q":"x"}}<|end_message|>'
        )
        (call,) = _tool_parser().extract_tool_calls(text).tool_calls
        assert call["name"] == "lookup"
        assert json.loads(call["arguments"]) == {"q": "x"}

    def test_truncated_call_is_not_a_call(self):
        text = (
            THINK + "<|message_model|>get_weather<|content_invoke_tool_json|>"
            '{"name":"get_weather","args":{"city":"Par'
        )
        info = _tool_parser().extract_tool_calls(text)
        assert not info.tools_called
        assert info.content is None

    def test_unclosed_but_complete_call_is_recovered(self):
        text = CALL.removesuffix("<|end_message|>")
        (call,) = _tool_parser().extract_tool_calls(text).tool_calls
        assert call["name"] == "get_weather"

    def test_closed_block_with_bad_json_keeps_raw_arguments(self):
        text = (
            "<|message_model|>run<|content_invoke_tool_json|>"
            '{"name":"run","args":{oops}<|end_message|>'
        )
        (call,) = _tool_parser().extract_tool_calls(text).tool_calls
        assert call["name"] == "run"
        assert call["arguments"] == '{"name":"run","args":{oops}'

    def test_plain_text_without_markers_is_content(self):
        assert _reasoning_parser().extract_reasoning("just text") == (None, "just text")
        info = _tool_parser().extract_tool_calls("just text")
        assert info.content == "just text" and not info.tools_called

    def test_unknown_marker_and_angle_brackets_stay_in_text(self):
        body = "if a < b: print('<|x|>') # <tag>"
        text = f"<|message_model|><|content_text|>{body}<|end_message|>"
        assert _tool_parser().extract_tool_calls(text).content == body

    def test_tool_error_and_tool_text_blocks_are_content(self):
        text = (
            "<|message_model|><|content_tool_error|>bad<|end_message|>"
            "<|message_model|><|content_invoke_tool_text|>raw<|end_message|>"
        )
        assert _reasoning_parser().extract_reasoning(text) == (None, "badraw")

    def test_unicode_arguments_are_not_escaped(self):
        text = (
            "<|message_model|>say<|content_invoke_tool_json|>"
            '{"name":"say","args":{"text":"dobrý deň"}}<|end_message|>'
        )
        (call,) = _tool_parser().extract_tool_calls(text).tool_calls
        assert "dobrý deň" in call["arguments"]


@pytest.mark.parametrize("size", [1, 3, 7, 16, 1000])
class TestStreaming:
    def test_matches_non_streaming(self, size):
        text = THINK + TEXT + CALL + CALL2 + EOS
        content, calls = _stream_tools(text, size)
        assert content == "It is sunny."
        assert [c["function"]["name"] for c in calls] == ["get_weather", "get_time"]
        assert [c["index"] for c in calls] == [0, 1]
        assert json.loads(calls[0]["function"]["arguments"]) == {
            "city": "Paris",
            "days": 2,
        }
        assert _stream_reasoning(text, size) == ("Check the forecast.", "It is sunny.")

    def test_no_marker_text_leaks(self, size):
        content, _ = _stream_tools(THINK + TEXT + CALL + EOS, size)
        reasoning, reply = _stream_reasoning(THINK + TEXT + CALL + EOS, size)
        for piece in (content, reasoning, reply):
            assert "<|" not in piece and "|>" not in piece
        assert "get_weather" not in content + reasoning + reply

    def test_truncated_call_emits_nothing(self, size):
        text = TEXT + "<|message_model|>get_weather<|content_invoke_tool_json|>{"
        content, calls = _stream_tools(text, size)
        assert content == "It is sunny." and calls == []

    def test_angle_bracket_text_survives(self, size):
        body = "a < b and <|not_ours|> and x <"
        text = f"<|message_model|><|content_text|>{body}"
        content, _ = _stream_tools(text, size)
        assert content == body
        assert _stream_reasoning(text, size) == ("", body)

    def test_identical_calls_are_two_calls(self, size):
        _, calls = _stream_tools(CALL + CALL + EOS, size)
        assert [c["index"] for c in calls] == [0, 1]
        assert calls[0]["id"] != calls[1]["id"]


class TestScanner:
    def test_marker_split_across_deltas_is_held(self):
        scanner = InklingBlockScanner()
        assert scanner.feed("<|message_model|><|content_te") == ("", "", [])
        assert scanner.feed("xt|>hi<|end_mess") == ("", "hi", [])
        assert scanner.feed("age|>") == ("", "", [])

    def test_each_delta_emits_only_its_own_text(self):
        scanner = InklingBlockScanner()
        scanner.feed("<|message_model|><|content_thinking|>")
        assert scanner.feed("ab") == ("ab", "", [])
        assert scanner.feed("cd") == ("cd", "", [])

    def test_scan_equals_feed_plus_finish(self):
        text = THINK + TEXT + CALL
        reasoning, content, tools = scan(text)
        assert (reasoning, content) == ("Check the forecast.", "It is sunny.")
        assert tools[0].name == "get_weather" and tools[0].closed

    def test_parser_resyncs_when_not_fed_from_the_start(self):
        parser = _tool_parser()
        seen = THINK + "<|message_model|><|content_text|>It is"
        result = parser.extract_tool_calls_streaming(seen, seen + " sunny.", " sunny.")
        assert result == {"content": " sunny."}
