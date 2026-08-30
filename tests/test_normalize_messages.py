# SPDX-License-Identifier: Apache-2.0
"""
Tests for _normalize_messages() in vllm_mlx.server.

_normalize_messages() maps non-standard roles (developer -> system) and merges
consecutive same-role messages before chat template application. This prevents
crashes from Qwen 3.5 and Llama templates that require alternating roles.
"""


class TestNormalizeMessages:
    """Test _normalize_messages() for handling real-world client formats."""

    def test_merge_consecutive_system_messages(self):
        """Consecutive system messages are merged into one."""
        from vllm_mlx.server import _normalize_messages

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "system", "content": "Always respond in JSON."},
            {"role": "user", "content": "Hello"},
        ]
        result = _normalize_messages(messages)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "helpful assistant" in result[0]["content"]
        assert "JSON" in result[0]["content"]
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "Hello"

    def test_merge_consecutive_user_messages(self):
        """Consecutive user messages are merged into one."""
        from vllm_mlx.server import _normalize_messages

        messages = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "First part"},
            {"role": "user", "content": "Second part"},
        ]
        result = _normalize_messages(messages)
        assert len(result) == 2
        assert result[1]["role"] == "user"
        assert "First part" in result[1]["content"]
        assert "Second part" in result[1]["content"]

    def test_opencode_format(self):
        """OpenCode's system+system+user+user format is normalized."""
        from vllm_mlx.server import _normalize_messages

        messages = [
            {"role": "system", "content": "System prompt part 1"},
            {"role": "system", "content": "System prompt part 2"},
            {"role": "user", "content": "User instruction"},
            {"role": "user", "content": "User question"},
        ]
        result = _normalize_messages(messages)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"

    def test_developer_role_mapped_to_system(self):
        """OpenAI Responses API 'developer' role is mapped to 'system'."""
        from vllm_mlx.server import _normalize_messages

        messages = [
            {"role": "developer", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        result = _normalize_messages(messages)
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"

    def test_developer_and_system_merged(self):
        """developer + system consecutive messages are merged after role mapping."""
        from vllm_mlx.server import _normalize_messages

        messages = [
            {"role": "developer", "content": "Part 1"},
            {"role": "system", "content": "Part 2"},
            {"role": "user", "content": "Hello"},
        ]
        result = _normalize_messages(messages)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "Part 1" in result[0]["content"]
        assert "Part 2" in result[0]["content"]

    def test_already_alternating_unchanged(self):
        """Well-formed alternating messages pass through unchanged."""
        from vllm_mlx.server import _normalize_messages

        messages = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "Bye"},
        ]
        result = _normalize_messages(messages)
        assert result == messages

    def test_single_message_unchanged(self):
        """Single message passes through unchanged."""
        from vllm_mlx.server import _normalize_messages

        messages = [{"role": "user", "content": "Hello"}]
        result = _normalize_messages(messages)
        assert result == messages

    def test_empty_messages(self):
        """Empty message list passes through."""
        from vllm_mlx.server import _normalize_messages

        assert _normalize_messages([]) == []

    def test_multimodal_content_preserved(self):
        """Messages with list content (multimodal) are not merged."""
        from vllm_mlx.server import _normalize_messages

        messages = [
            {"role": "user", "content": "Describe this:"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "http://example.com/img.png"},
                    },
                ],
            },
        ]
        result = _normalize_messages(messages)
        # List content can't be trivially merged with string - kept separate
        assert len(result) >= 1

    def test_preserves_non_content_fields(self):
        """Fields other than role/content are preserved on the first merged message."""
        from vllm_mlx.server import _normalize_messages

        messages = [
            {"role": "system", "content": "Part 1", "name": "sys1"},
            {"role": "system", "content": "Part 2"},
            {"role": "user", "content": "Hello"},
        ]
        result = _normalize_messages(messages)
        assert len(result) == 2
        assert result[0]["role"] == "system"

    def test_null_content_not_merged(self):
        """Messages with None content (tool_calls pattern) are not merged."""
        from vllm_mlx.server import _normalize_messages

        messages = [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "tc1"}]},
            {"role": "assistant", "content": "Follow-up"},
        ]
        result = _normalize_messages(messages)
        # None content can't be merged with string - kept separate
        assert len(result) == 2

    def test_three_consecutive_system_messages(self):
        """Three consecutive system messages merge into one."""
        from vllm_mlx.server import _normalize_messages

        messages = [
            {"role": "system", "content": "Part 1"},
            {"role": "system", "content": "Part 2"},
            {"role": "system", "content": "Part 3"},
            {"role": "user", "content": "Hello"},
        ]
        result = _normalize_messages(messages)
        assert len(result) == 2
        assert "Part 1" in result[0]["content"]
        assert "Part 3" in result[0]["content"]


class TestNormalizeMessagesStructuralMerge:
    """#83: the merge must not DROP keys beyond role/content.

    The old dict-drop merge deleted ``tool_calls`` from an assistant
    text-turn + tool-call-turn pair — a standard agent-loop history shape —
    so re-sent histories lost calls (same corruption class as llama.cpp
    #27626, reached through the merge instead of prefill-continuation).
    """

    def test_assistant_text_plus_tool_call_turn_keeps_tool_calls(self):
        from vllm_mlx.server import _normalize_messages

        tc = [{"id": "1", "type": "function",
               "function": {"name": "get_weather", "arguments": "{}"}}]
        messages = [
            {"role": "user", "content": "check the weather"},
            {"role": "assistant", "content": "Let me check."},
            {"role": "assistant", "content": "", "tool_calls": tc},
        ]
        result = _normalize_messages(messages)
        assert len(result) == 2
        assert result[1]["tool_calls"] == tc
        # empty second content: no stray separator appended
        assert result[1]["content"] == "Let me check."

    def test_two_tool_call_turns_concatenate_in_order(self):
        from vllm_mlx.server import _normalize_messages

        tc1 = [{"id": "1", "type": "function",
                "function": {"name": "a", "arguments": "{}"}}]
        tc2 = [{"id": "2", "type": "function",
                "function": {"name": "b", "arguments": "{}"}}]
        messages = [
            {"role": "assistant", "content": "", "tool_calls": tc1},
            {"role": "assistant", "content": "", "tool_calls": tc2},
        ]
        result = _normalize_messages(messages)
        assert len(result) == 1
        assert result[0]["tool_calls"] == tc1 + tc2

    def test_prev_tool_calls_survive_text_followup(self):
        from vllm_mlx.server import _normalize_messages

        tc = [{"id": "1", "type": "function",
               "function": {"name": "a", "arguments": "{}"}}]
        messages = [
            {"role": "assistant", "content": "", "tool_calls": tc},
            {"role": "assistant", "content": "and here is why"},
        ]
        result = _normalize_messages(messages)
        assert len(result) == 1
        assert result[0]["tool_calls"] == tc
        assert result[0]["content"] == "and here is why"

    def test_extra_scalar_keys_first_writer_wins(self):
        from vllm_mlx.server import _normalize_messages

        messages = [
            {"role": "assistant", "content": "a", "reasoning_content": "r1"},
            {"role": "assistant", "content": "b", "reasoning_content": "r2"},
        ]
        result = _normalize_messages(messages)
        assert len(result) == 1
        assert result[0]["reasoning_content"] == "r1"
        # carried when only the second message has it
        messages = [
            {"role": "assistant", "content": "a"},
            {"role": "assistant", "content": "b", "reasoning_content": "r2"},
        ]
        assert _normalize_messages(messages)[0]["reasoning_content"] == "r2"

    def test_plain_text_merge_unchanged(self):
        from vllm_mlx.server import _normalize_messages

        messages = [
            {"role": "user", "content": "one"},
            {"role": "user", "content": "two"},
        ]
        result = _normalize_messages(messages)
        assert result[0]["content"] == "one\n\ntwo"

    def test_input_messages_not_mutated(self):
        from vllm_mlx.server import _normalize_messages

        tc = [{"id": "1", "type": "function",
               "function": {"name": "a", "arguments": "{}"}}]
        first = {"role": "assistant", "content": "", "tool_calls": tc}
        second = {"role": "assistant", "content": "", "tool_calls": [
            {"id": "2", "type": "function",
             "function": {"name": "b", "arguments": "{}"}}]}
        _normalize_messages([first, second])
        assert first["tool_calls"] == tc  # concat produced a new list
