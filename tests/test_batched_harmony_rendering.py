# SPDX-License-Identifier: Apache-2.0
"""Patch #75: BatchedEngine honors ``use_harmony_rendering``.

Parity with SimpleEngine's #581/#568 harmony branch — previously the flag was
a no-op on every batched route, so gpt-oss multi-turn tool conversations were
rendered by the Jinja template (which flattens structural ``tool_calls``
history) instead of the canonical openai-harmony renderer.
"""

from unittest.mock import MagicMock, patch

import pytest

TOOL_CONVO = [
    {"role": "user", "content": "Find the bug in foo.py."},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "run_command",
                    "arguments": '{"cmd": "cat foo.py"}',
                },
            }
        ],
    },
    {"role": "user", "content": "Continue."},
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        },
    }
]


def _engine():
    with patch("vllm_mlx.engine.batched.is_mllm_model", return_value=False):
        from vllm_mlx.engine.batched import BatchedEngine

        engine = BatchedEngine("test-model")
    engine._tokenizer = MagicMock()
    engine._tokenizer.apply_chat_template.return_value = "jinja-prompt"
    engine.use_harmony_rendering = True
    return engine


class TestBatchedHarmonyRendering:
    def test_harmony_replaces_jinja_and_keeps_tool_call_structure(self):
        from vllm_mlx.utils.harmony_render import HAS_HARMONY

        if not HAS_HARMONY:
            pytest.skip("openai-harmony not installed")

        engine = _engine()
        prompt = engine._apply_chat_template(TOOL_CONVO, tools=TOOLS)

        assert "<|start|>system" in prompt
        assert "to=functions.run_command" in prompt
        assert "<|channel|>commentary" in prompt
        # The lossy bracket-text flattening must not appear.
        assert "[Calling tool:" not in prompt
        engine._tokenizer.apply_chat_template.assert_not_called()

    def test_harmony_forwards_reasoning_effort_and_tools(self):
        engine = _engine()
        with patch(
            "vllm_mlx.utils.harmony_render.render_messages",
            return_value="harmony-prompt",
        ) as render:
            prompt = engine._apply_chat_template(
                [{"role": "user", "content": "Hi"}],
                tools=TOOLS,
                chat_template_kwargs={"reasoning_effort": "low"},
            )
        assert prompt == "harmony-prompt"
        assert render.call_args.kwargs["reasoning_effort"] == "low"
        assert render.call_args.kwargs["tools"] == TOOLS
        engine._tokenizer.apply_chat_template.assert_not_called()

    def test_media_requests_fall_through_to_template(self):
        engine = _engine()
        prompt = engine._apply_chat_template(
            [{"role": "user", "content": "Describe."}],
            num_images=1,
        )
        assert prompt == "jinja-prompt"
        engine._tokenizer.apply_chat_template.assert_called_once()

    def test_flag_off_uses_template(self):
        engine = _engine()
        engine.use_harmony_rendering = False
        prompt = engine._apply_chat_template([{"role": "user", "content": "Hi"}])
        assert prompt == "jinja-prompt"
        engine._tokenizer.apply_chat_template.assert_called_once()
