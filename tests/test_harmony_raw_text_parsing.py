# SPDX-License-Identifier: Apache-2.0
"""
Regression tests for fork patch #72: harmony tool calls must be parsed from the
engine's PRE-clean text.

Both engines run ``clean_output_text`` over their final text before returning a
``GenerationOutput``. For harmony models ``_clean_gpt_oss_output`` deletes whole
structural blocks — ``<|channel|>commentary to=functions.X<|constrain|>json
<|message|>`` and ``<|start|>assistant`` — leaving only the bare argument JSON
(and, tellingly, a stray ``<|end|>``). Handing that text to ``HarmonyToolParser``
can never yield a tool call, so non-streaming gpt-oss tool calling silently
returned reasoning prose as content with ``tool_calls`` absent.

The fix carries the pre-clean text on ``GenerationOutput.raw_text`` and has the
server parse from it. ``clean_output_text`` itself is upstream's no-reasoning-
parser fallback and is deliberately left untouched.

Usage:
    pytest tests/test_harmony_raw_text_parsing.py -v
"""

from types import SimpleNamespace

from vllm_mlx.api.utils import clean_output_text
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.server import _parse_source_text
from vllm_mlx.tool_parsers.harmony_tool_parser import HarmonyToolParser

RAW_TOOL_OUTPUT = (
    "<|channel|>analysis<|message|>We should call the tool.<|end|>"
    "<|start|>assistant<|channel|>commentary to=functions.get_weather "
    '<|constrain|>json<|message|>{"location":"San Francisco"}<|call|>'
)


def test_clean_output_text_destroys_the_commentary_block():
    """Pin the exact corruption the fix works around."""
    cleaned = clean_output_text(RAW_TOOL_OUTPUT)

    assert "<|channel|>" not in cleaned
    assert "to=functions.get_weather" not in cleaned
    # The stray <|end|> survives — this is the live signature of the bug.
    assert "<|end|>" in cleaned
    assert '{"location":"San Francisco"}' in cleaned


def test_parser_extracts_from_raw_but_not_from_cleaned():
    parser = HarmonyToolParser(None)
    request = {"tools": [{"type": "function", "function": {"name": "get_weather"}}]}

    from_raw = parser.extract_tool_calls(RAW_TOOL_OUTPUT, request)
    assert from_raw.tools_called is True
    assert from_raw.tool_calls[0]["name"] == "get_weather"

    parser.reset()
    from_cleaned = parser.extract_tool_calls(clean_output_text(RAW_TOOL_OUTPUT), request)
    assert from_cleaned.tools_called is False


def test_parse_source_prefers_raw_text():
    output = GenerationOutput(
        text=clean_output_text(RAW_TOOL_OUTPUT),
        raw_text=RAW_TOOL_OUTPUT,
    )
    assert _parse_source_text(output) == RAW_TOOL_OUTPUT


def test_parse_source_falls_back_to_text_when_raw_absent():
    """Engines/paths that don't populate raw_text keep the old behaviour."""
    assert _parse_source_text(GenerationOutput(text="hello")) == "hello"
    assert _parse_source_text(SimpleNamespace(text="hello")) == "hello"
    assert _parse_source_text(SimpleNamespace()) == ""


def test_generation_output_raw_text_defaults_empty():
    assert GenerationOutput(text="x").raw_text == ""
