# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for GPT-OSS models using Harmony format.

Harmony uses channels for reasoning vs final content:

    <|channel|>analysis
    <|message|>Let me think about this...
    <|end|>
    <|channel|>final
    <|message|>The answer is 42.
    <|return|>

The analysis channel contains reasoning, and the final channel
contains the user-facing response.
"""

import re

from .base import DeltaMessage, ReasoningParser

# Analysis channel blocks: <|channel|>analysis<|message|>...<|end|>
_ANALYSIS_PATTERN = re.compile(
    r"<\|channel\|>analysis\s*<\|message\|>(.*?)<\|end\|>",
    re.DOTALL,
)

# Final channel content: <|channel|>final<|message|>...<|return|>
_FINAL_PATTERN = re.compile(
    r"<\|channel\|>final\s*<\|message\|>(.*?)<\|return\|>",
    re.DOTALL,
)


_MESSAGE_END_TOKENS = ("<|end|>", "<|return|>", "<|call|>", "<|start|>")


def _channel_state(text: str) -> tuple[str | None, bool]:
    """(channel, inside its message) at the end of the accumulated text.

    The channel is whatever follows the last ``<|channel|>`` — "analysis",
    "final" or "commentary" (possibly followed by `` to=functions.x``); a name
    still being streamed is None. The message is open once ``<|message|>``
    follows that header and no end token has closed it since.
    """
    header = text.rfind("<|channel|>")
    if header == -1:
        return None, False
    rest = text[header + len("<|channel|>") :]
    channel = next(
        (name for name in ("analysis", "final", "commentary") if rest.startswith(name)),
        None,
    )
    start = rest.find("<|message|>")
    if start == -1:
        return channel, False
    body = rest[start + len("<|message|>") :]
    return channel, not any(token in body for token in _MESSAGE_END_TOKENS)


class HarmonyReasoningParser(ReasoningParser):
    """
    Reasoning parser for GPT-OSS models using Harmony format.

    Extracts reasoning from the 'analysis' channel and content from
    the 'final' channel. Commentary channels (tool calls) are ignored
    since they are handled by the tool parser.

    Example:
        Input: "<|channel|>analysis<|message|>Thinking...<|end|>
                <|channel|>final<|message|>Result.<|return|>"
        Output: reasoning="Thinking...", content="Result."
    """

    # Streaming loops must hand this parser the raw delta: the channel
    # structure lives in control tokens a special-token filter would delete
    # (fork #112 — the Anthropic path did exactly that).
    CONSUMES_RAW_STREAM = True

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._current_channel: str | None = None
        self._in_message: bool = False

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning from complete Harmony output.

        Collects all analysis channel blocks as reasoning and the
        final channel block as content.

        Args:
            model_output: Complete model output text.

        Returns:
            (reasoning, content) tuple. Either may be None.
        """
        # Collect all analysis blocks
        analysis_blocks = _ANALYSIS_PATTERN.findall(model_output)
        reasoning = "\n".join(block.strip() for block in analysis_blocks) or None

        # Extract final channel content
        final_match = _FINAL_PATTERN.search(model_output)
        content = final_match.group(1).strip() if final_match else None

        return reasoning, content

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """
        Extract reasoning from streaming Harmony output.

        Tracks the current channel and emits reasoning deltas for
        analysis channel content and content deltas for final channel.

        Args:
            previous_text: Accumulated text before this delta.
            current_text: Accumulated text including this delta.
            delta_text: The new text in this streaming chunk.

        Returns:
            DeltaMessage with reasoning and/or content, or None.
        """
        # Control tokens arrive as their own deltas and the channel name is
        # split across tokens ("comment" + "ary" on gpt-oss-20b), so the old
        # per-delta checks ("<|channel|>" and the name in ONE delta) never
        # saw a switch: the parser stayed on "analysis" and streamed the
        # final answer and the tool-call arguments as reasoning (fork #112).
        # The state is derived from the accumulated text instead — only on a
        # control-token delta, the only thing that can change it (the channel
        # name is complete by the time <|message|> arrives), so plain text
        # deltas stay O(1) instead of re-slicing the message every token.
        if "<|" in delta_text:
            # Control-token delta (or a merged delta containing one): never
            # user-visible. Text riding along after the token is dropped, as
            # before; token-level streaming does not produce that shape.
            self._current_channel, self._in_message = _channel_state(current_text)
            return None

        if not self._in_message:
            return None
        if self._current_channel == "analysis":
            return DeltaMessage(reasoning=delta_text)
        if self._current_channel == "final":
            return DeltaMessage(content=delta_text)
        # Commentary (tool calls) belongs to the tool parser, which reads the
        # raw stream; an unknown/partial channel name is withheld.
        return None

    def reset_state(self, implicit_mode: bool = False):  # noqa: ARG002
        """Reset streaming state for a new request."""
        self._current_channel = None
        self._in_message = False
