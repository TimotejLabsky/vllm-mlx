# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for Inkling 2 models.

Inkling 2 writes its reasoning as ``<|content_thinking|>`` message blocks and
the reply as ``<|content_text|>`` blocks (see ``vllm_mlx/inkling_blocks.py``
for the format). Thinking blocks become ``reasoning``, text blocks become
``content``; tool-call blocks are left to the ``inkling`` tool parser, which
reads the same raw stream.
"""

from ..inkling_blocks import HEADER, InklingBlockScanner, scan
from .base import DeltaMessage, ReasoningParser


class InklingReasoningParser(ReasoningParser):
    """
    Reasoning parser for Inkling 2's message-block format.

    Example:
        Input: "<|message_model|><|content_thinking|>Let me see.<|end_message|>
                <|message_model|><|content_text|>42.<|end_message|>"
        Output: reasoning="Let me see.", content="42."
    """

    # The block structure lives in control tokens a special-token filter would
    # delete, so streaming loops must hand this parser the raw delta.
    CONSUMES_RAW_STREAM = True

    # Read by the server's thinking-disabled latch: a model header means block
    # structure follows. Minimum reasoning effort does not switch Inkling 2's
    # thinking off entirely, so thinking blocks can appear on any request.
    start_token = HEADER

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._scanner = InklingBlockScanner()

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        reasoning, content, _ = scan(model_output)
        return reasoning.strip() or None, content.strip() or None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        if self._scanner.fed != len(previous_text):
            # Not fed from the start of this request (or reused without a
            # reset): rebuild from the accumulated text, emitting nothing for
            # the part that is already behind us.
            self._scanner = InklingBlockScanner()
            self._scanner.feed(previous_text)
        reasoning, content, _ = self._scanner.feed(delta_text)
        if not reasoning and not content:
            return None
        return DeltaMessage(reasoning=reasoning or None, content=content or None)

    def finalize_stream(self) -> DeltaMessage | None:
        reasoning, content, _ = self._scanner.finish()
        if not reasoning and not content:
            return None
        return DeltaMessage(reasoning=reasoning or None, content=content or None)

    def reset_state(self, implicit_mode: bool = False):  # noqa: ARG002
        """Reset streaming state for a new request."""
        self._scanner = InklingBlockScanner()
