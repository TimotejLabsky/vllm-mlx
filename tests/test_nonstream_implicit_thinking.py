# SPDX-License-Identifier: Apache-2.0
"""Fork #121: non-streaming replies honour implicit <think> too.

Templates such as GLM-4.7's (and Qwen3's thinking-on prompt) end the prompt
with an open ``<think>``. Streaming knew (``_detect_implicit_thinking`` ->
``reset_state(implicit_mode=True)``); the non-stream extractor used the shared
parser without that signal, so output cut off by ``max_tokens`` before
``</think>`` came back as *content* (seen live on GLM-4.7-Flash).
"""

import pytest

import vllm_mlx.server as srv
from vllm_mlx.reasoning import get_parser

TRUNCATED = "The user wants a one-word greeting.\nCommon one-word greetings: Hello,"


@pytest.mark.parametrize("name", ["glm4", "qwen3", "deepseek_r1"])
def test_parser_implicit_mode_classes_unclosed_output_as_reasoning(name):
    parser = get_parser(name)()
    parser.reset_state(implicit_mode=True)
    assert parser.extract_reasoning(TRUNCATED) == (TRUNCATED, None)
    # Closed blocks are unaffected.
    assert parser.extract_reasoning("x</think>Hello") == ("x", "Hello")


@pytest.mark.parametrize("name", ["glm4", "qwen3"])
def test_parser_explicit_mode_unchanged(name):
    assert get_parser(name)().extract_reasoning(TRUNCATED) == (None, TRUNCATED)


@pytest.fixture
def glm_server(monkeypatch):
    shared = get_parser("glm4")()
    monkeypatch.setattr(srv, "_reasoning_parser", shared)
    monkeypatch.setattr(srv, "_reasoning_parser_name", "glm4")
    return shared


def _engine():
    class E:
        tokenizer = None

    return E()


def test_nonstream_truncated_implicit_thinking_is_reasoning(glm_server, monkeypatch):
    monkeypatch.setattr(srv, "_detect_implicit_thinking", lambda e, k: True)
    reasoning, content, _ = srv._extract_reasoning_and_tool_calls(
        TRUNCATED, None, engine=_engine(), chat_kwargs={}
    )
    assert reasoning == TRUNCATED and not content
    # The shared module parser was not switched into implicit mode.
    assert glm_server._implicit_mode is False


def test_nonstream_without_implicit_template_keeps_content(glm_server, monkeypatch):
    monkeypatch.setattr(srv, "_detect_implicit_thinking", lambda e, k: False)
    reasoning, content, _ = srv._extract_reasoning_and_tool_calls(
        TRUNCATED, None, engine=_engine(), chat_kwargs={}
    )
    assert (reasoning, content) == (None, TRUNCATED)


def test_nonstream_thinking_off_never_probes(glm_server, monkeypatch):
    def boom(e, k):
        raise AssertionError("probed with thinking off")

    monkeypatch.setattr(srv, "_detect_implicit_thinking", boom)
    reasoning, content, _ = srv._extract_reasoning_and_tool_calls(
        "Hello", None, allow_reasoning=False, engine=_engine(), chat_kwargs={}
    )
    assert (reasoning, content) == (None, "Hello")
