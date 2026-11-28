# SPDX-License-Identifier: Apache-2.0
"""
Tool call parser for Inkling 2 models.

A call is its own message block, addressed by the tool name in the header:

    <|message_model|>get_weather<|content_invoke_tool_json|>
    {"name":"get_weather","args":{"city":"Paris"}}<|end_message|>

Several calls in one turn are several blocks. The reply text is in
``<|content_text|>`` blocks and reasoning in ``<|content_thinking|>`` blocks
(see ``vllm_mlx/inkling_blocks.py``). Pair with the ``inkling`` reasoning
parser, which reads the same raw stream.

A block cut off by ``max_tokens`` is a call only if its JSON is already
complete; a closed block with unparseable JSON keeps its raw text as the
arguments, under the header's name, so the client sees what the model wrote.
"""

import json
import uuid
from collections.abc import Sequence
from typing import Any

from ..inkling_blocks import TOOL_JSON, InklingBlockScanner, ToolBlock, scan
from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)


def _generate_tool_id() -> str:
    """Generate a unique tool call ID."""
    return f"call_{uuid.uuid4().hex[:8]}"


def _to_call(block: ToolBlock) -> dict[str, Any] | None:
    """The OpenAI-shaped call for one block, or None if it is not a call."""
    raw = block.body.strip()
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        payload = None
    if isinstance(payload, dict):
        name = payload.get("name") or block.name
        arguments = payload.get("args", payload.get("arguments", {}))
        if isinstance(name, str) and name:
            return {
                "id": _generate_tool_id(),
                "name": name,
                "arguments": (
                    json.dumps(arguments, ensure_ascii=False)
                    if isinstance(arguments, (dict, list))
                    else str(arguments)
                ),
            }
    if block.closed and block.name and raw:
        return {"id": _generate_tool_id(), "name": block.name, "arguments": raw}
    return None


@ToolParserManager.register_module(["inkling", "inkling2"])
class InklingToolParser(ToolParser):
    """Tool call parser for Inkling 2's ``invoke_tool_json`` message blocks."""

    STREAMING_MARKERS = (TOOL_JSON,)
    # The reply itself is framed in control tokens, so every delta has to pass
    # through this parser, not only the ones after a call marker.
    REQUIRES_EAGER_STREAMING = True
    # The streaming loops hand this parser the raw delta (control tokens
    # intact); it splits the blocks itself.
    CONSUMES_RAW_STREAM = True
    # A call is never written inside a thinking block in this format.
    TOOL_MARKUP_IN_REASONING = False
    # The chat template renders tool_calls and role="tool" messages natively.
    SUPPORTS_NATIVE_TOOL_FORMAT = True

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._scanner = InklingBlockScanner()
        self._calls: list[dict[str, Any]] = []

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        _, content, blocks = scan(model_output)
        tool_calls = [call for call in map(_to_call, blocks) if call]
        return ExtractedToolCallInformation(
            tools_called=bool(tool_calls),
            tool_calls=tool_calls,
            content=content.strip() or None,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int] | None = None,
        current_token_ids: Sequence[int] | None = None,
        delta_token_ids: Sequence[int] | None = None,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Stream text-block bodies as content and each call when its block closes."""
        if self._scanner.fed != len(previous_text):
            # Not fed from the start of this request: rebuild from the
            # accumulated text. Its content already went out; a call that
            # closed in it has not, so it is emitted with this delta.
            self._scanner = InklingBlockScanner()
            _, _, missed = self._scanner.feed(previous_text)
            self._calls = [call for call in map(_to_call, missed) if call]
        _, content, blocks = self._scanner.feed(delta_text)
        return self._emit(content, blocks)

    def finalize_streaming(self, current_text: str) -> dict[str, Any] | None:
        """Release held text and a call whose block never closed but parses."""
        _, content, blocks = self._scanner.finish()
        return self._emit(content, blocks)

    def _emit(self, content: str, blocks: list[ToolBlock]) -> dict[str, Any] | None:
        self._calls.extend(call for call in map(_to_call, blocks) if call)
        result: dict[str, Any] = {}
        if content:
            result["content"] = content
        if emitted := self._stream_new_tool_calls(self._calls):
            result["tool_calls"] = emitted["tool_calls"]
        return result or None

    def reset(self) -> None:
        """Reset parser state for a new request."""
        super().reset()
        self._scanner = InklingBlockScanner()
        self._calls = []
