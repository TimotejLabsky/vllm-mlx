# SPDX-License-Identifier: Apache-2.0
"""
Incremental scanner for the Inkling 2 message-block format.

An Inkling 2 turn is a run of typed blocks, each closed by ``<|end_message|>``
and the turn by ``<|content_model_end_sampling|>`` (the EOS token):

    <|message_model|><|content_thinking|>...<|end_message|>
    <|message_model|><|content_text|>...<|end_message|>
    <|message_model|>get_weather<|content_invoke_tool_json|>
        {"name":"get_weather","args":{"city":"Paris"}}<|end_message|>
    <|content_model_end_sampling|>

The ``<|message_model|>`` header is optional in front of a content marker and
carries the tool name for a call. This is the grammar the checkpoint's own
``response_template`` (tokenizer_config.json) declares: thinking blocks are
reasoning, ``text`` / ``tool_error`` / ``invoke_tool_text`` blocks are content,
``invoke_tool_json`` blocks are tool calls, every kind may repeat.

The reasoning parser and the tool parser both read the raw token stream
(``CONSUMES_RAW_STREAM``) and both use this scanner, so streaming and
non-streaming, and the two parsers, cannot disagree about where a block ends.
It is fed deltas and costs O(delta): a marker split across deltas is held back
until it completes or can no longer be one.
"""

import re
from dataclasses import dataclass

HEADER = "<|message_model|>"
TOOL_JSON = "<|content_invoke_tool_json|>"

_OPENERS = {
    "<|content_thinking|>": "thinking",
    "<|content_text|>": "text",
    "<|content_tool_error|>": "text",
    "<|content_invoke_tool_text|>": "text",
    TOOL_JSON: "tool",
}
# Anything that ends the current block without opening a model block: the
# block close, the turn close, and role headers the model should never write.
_CLOSERS = frozenset(
    {
        "<|end_message|>",
        "<|content_model_end_sampling|>",
        "<|endoftext|>",
        "<|message_user|>",
        "<|message_system|>",
        "<|message_tool|>",
    }
)
_MARKER_RE = re.compile(r"<\|[a-z0-9_]+\|>")
# A tail that can still grow into one of the markers above.
_PARTIAL_RE = re.compile(r"<(?:\|[a-z0-9_]*\|?)?")
_MAX_MARKER = max(len(m) for m in (*_OPENERS, *_CLOSERS, HEADER))


@dataclass
class ToolBlock:
    """One ``invoke_tool_json`` block: header name, raw JSON body, closed or cut."""

    name: str
    body: str
    closed: bool


class InklingBlockScanner:
    """Split an Inkling 2 token stream into reasoning, content and tool blocks."""

    def __init__(self) -> None:
        self._kind: str | None = None  # None | "header" | "thinking" | "text" | "tool"
        self._pending = ""
        self._name = ""
        self._tool_body = ""
        self.fed = 0

    def feed(self, delta: str) -> tuple[str, str, list[ToolBlock]]:
        """Consume ``delta``; return (reasoning, content, completed tool blocks)."""
        self.fed += len(delta)
        out: dict[str, list] = {"thinking": [], "text": [], "tool": []}
        buf = self._pending + delta
        self._pending = ""
        pos = 0
        while True:
            start = buf.find("<", pos)
            if start == -1:
                self._body(buf[pos:], out)
                break
            self._body(buf[pos:start], out)
            match = _MARKER_RE.match(buf, start)
            if match:
                self._marker(match.group(), out)
                pos = match.end()
                continue
            tail = buf[start:]
            if len(tail) < _MAX_MARKER and _PARTIAL_RE.fullmatch(tail):
                self._pending = tail  # may still become a marker
                break
            self._body("<", out)
            pos = start + 1
        return "".join(out["thinking"]), "".join(out["text"]), out["tool"]

    def finish(self) -> tuple[str, str, list[ToolBlock]]:
        """End of generation: release held text and report an unclosed call."""
        out: dict[str, list] = {"thinking": [], "text": [], "tool": []}
        pending, self._pending = self._pending, ""
        self._body(pending, out)
        if self._kind == "tool":
            out["tool"].append(ToolBlock(self._name, self._tool_body, closed=False))
        self._kind = None
        return "".join(out["thinking"]), "".join(out["text"]), out["tool"]

    def _body(self, text: str, out: dict[str, list]) -> None:
        if not text:
            return
        if self._kind == "header":
            self._name += text
        elif self._kind == "tool":
            self._tool_body += text
        elif self._kind == "thinking":
            out["thinking"].append(text)
        else:
            # A text block, or text outside any block (a stream whose control
            # tokens were filtered out before it got here): both are content.
            out["text"].append(text)

    def _close(self, out: dict[str, list]) -> None:
        if self._kind == "tool":
            out["tool"].append(ToolBlock(self._name, self._tool_body, closed=True))
        self._kind = None

    def _marker(self, marker: str, out: dict[str, list]) -> None:
        if marker == HEADER:
            self._close(out)
            self._kind = "header"
            self._name = ""
        elif marker in _OPENERS:
            if self._kind == "header":
                name = self._name.strip()
            else:
                self._close(out)
                name = ""
            self._kind = _OPENERS[marker]
            self._name = name
            self._tool_body = ""
        elif marker in _CLOSERS:
            self._close(out)
        else:
            # Not one of ours (e.g. the model quoting some other "<|x|>").
            self._body(marker, out)


def scan(text: str) -> tuple[str, str, list[ToolBlock]]:
    """Scan a complete model output: (reasoning, content, tool blocks)."""
    scanner = InklingBlockScanner()
    reasoning, content, tools = scanner.feed(text)
    more_reasoning, more_content, more_tools = scanner.finish()
    return reasoning + more_reasoning, content + more_content, tools + more_tools
