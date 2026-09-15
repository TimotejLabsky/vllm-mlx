# SPDX-License-Identifier: Apache-2.0
"""
'<' immediately followed by another '<' must not stall Qwen3XMLToolParser.

``_find_next_complete_element`` held back any buffer fragment that "looks
like the start of a tool tag" so partial ``<function=...`` heads are not
emitted as text. When a second ``<`` arrived before any ``>``, the fragment
it checked was just ``"<"`` — a prefix of every tool tag — so the splitter
waited for more data forever. Everything after the first ``<`` was lost:
``cat > f << 'EOF' ... EOF`` reached the client as ``cat > f ``, in both the
non-streaming and the streaming path. Found live 2026-09-15 on the Qwen3.8 /
Qwen3.6 routes (openclaw heredoc state writes, C++ ``std::cout <<``).

A fragment that is followed by another ``<`` can never complete into a tool
tag head (none contains ``<`` after its first character), so these cases
must parse exactly — at every chunk size.
"""

import json

import pytest

pytest.importorskip("transformers")

from vllm_mlx.tool_parsers.qwen3_xml_tool_parser import Qwen3XMLToolParser

REQUEST = {
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "exec",
                "parameters": {
                    "type": "object",
                    "required": ["command"],
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "integer"},
                    },
                },
            },
        }
    ]
}

VALUES = {
    "heredoc_quoted": 'cat > /tmp/state.json << \'ENDSTATE\'\n{"date": "2026-09-13", "open_issues": []}\nENDSTATE',
    "heredoc_bare": 'cat > /tmp/s.json <<EOF\n{"a": 1}\nEOF',
    "heredoc_dash": "cat <<-EOF\n\tindented\n\tEOF",
    "here_string": 'grep -c x <<< "$VAR"',
    "cpp_shift": "printf '%s' 'std::cout << \"Hello\" << std::endl;' > /tmp/a.cpp",
    "shift_assign": "python3 -c 'x = 1; x <<= 3; print(x)'",
    "trailing_lt": "echo 'a <'",
    "lt_then_tag_like": "echo '<<x>>'",
    "single_lt_control": "python3 -c 'print(1 < 3)'",
}

CHUNK_SIZES = [1, 2, 3, 5, 7, 13]


def _tool_call_text(command: str, timeout: int | None = None) -> str:
    extra = (
        f"<parameter=timeout>\n{timeout}\n</parameter>\n" if timeout is not None else ""
    )
    return (
        "<tool_call>\n<function=exec>\n"
        f"<parameter=command>\n{command}\n</parameter>\n"
        f"{extra}"
        "</function>\n</tool_call>"
    )


def _stream(text: str, size: int) -> tuple[list[tuple[str, str]], str]:
    parser = Qwen3XMLToolParser(None)
    calls: dict[int, dict[str, str]] = {}
    content = ""
    previous = ""
    for i in range(0, len(text), size):
        delta = text[i : i + size]
        current = previous + delta
        out = parser.extract_tool_calls_streaming(
            previous, current, delta, [], [], [], REQUEST
        )
        previous = current
        if not out:
            continue
        content += out.get("content") or ""
        for tc in out.get("tool_calls") or []:
            entry = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
            fn = tc.get("function") or {}
            entry["name"] += fn.get("name") or ""
            entry["arguments"] += fn.get("arguments") or ""
    return [(c["name"], c["arguments"]) for _, c in sorted(calls.items())], content


@pytest.mark.parametrize("name", list(VALUES))
def test_non_streaming_keeps_double_lt_argument(name):
    value = VALUES[name]
    result = Qwen3XMLToolParser(None).extract_tool_calls(
        _tool_call_text(value), REQUEST
    )
    assert result.tools_called
    assert len(result.tool_calls) == 1
    assert json.loads(result.tool_calls[0]["arguments"]) == {"command": value}


@pytest.mark.parametrize("size", CHUNK_SIZES)
@pytest.mark.parametrize("name", list(VALUES))
def test_streaming_keeps_double_lt_argument(name, size):
    value = VALUES[name]
    calls, _ = _stream(_tool_call_text(value), size)
    assert len(calls) == 1
    assert calls[0][0] == "exec"
    assert json.loads(calls[0][1]) == {"command": value}


@pytest.mark.parametrize("size", [1, 3, 7])
def test_parameter_after_heredoc_still_parses_and_coerces(size):
    value = VALUES["heredoc_quoted"]
    text = _tool_call_text(value, timeout=30)
    expected = {"command": value, "timeout": 30}
    non_stream = Qwen3XMLToolParser(None).extract_tool_calls(text, REQUEST)
    assert json.loads(non_stream.tool_calls[0]["arguments"]) == expected
    calls, _ = _stream(text, size)
    assert json.loads(calls[0][1]) == expected


@pytest.mark.parametrize("size", [1, 2, 5])
def test_double_lt_in_plain_content_is_not_swallowed(size):
    text = "Use a heredoc: cat <<EOF, and in C++ write std::cout << x."
    non_stream = Qwen3XMLToolParser(None).extract_tool_calls(text, REQUEST)
    assert not non_stream.tools_called
    assert non_stream.content == text
    calls, content = _stream(text, size)
    assert calls == []
    assert content == text


@pytest.mark.parametrize("size", [1, 2, 3])
def test_partial_tool_tag_head_is_still_held_back(size):
    """The original purpose of the hold-back must survive: a split
    ``<function=`` head is not leaked as content before it completes."""
    text = "<function=exec>\n<parameter=command>\nls <<EOF\nx\nEOF\n</parameter>\n</function>"
    calls, content = _stream(text, size)
    assert len(calls) == 1
    assert json.loads(calls[0][1]) == {"command": "ls <<EOF\nx\nEOF"}
    assert "<function" not in content and "<parameter" not in content
