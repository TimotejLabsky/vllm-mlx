# SPDX-License-Identifier: Apache-2.0
"""(#141) XML tool-call parameter values keep their whitespace.

Qwen3.5/3.6/3.8, Qwen3-Coder and Nemotron emit
``<parameter=name>\\nvalue\\n</parameter>``: ONE newline on each side frames
the value, and everything between is argument data. The hermes, auto and
nemotron parsers matched ``\\s*(.*?)\\s*`` and ``.strip()``-ed the value, so an
``edit`` call's ``oldString`` of ``"    return x\\n"`` reached the client as
``"return x"`` - the indentation and trailing newline the edit has to match
were gone. This removes exactly one framing newline per side, the rule the
vendored ``qwen3_xml`` parser (vLLM's) and upstream's qwen parser (#808)
already follow.
"""

from __future__ import annotations


def strip_param_framing(value: str) -> str:
    """``value`` minus one leading and one trailing framing newline.

    A lone ``"\\n"`` is kept: one newline cannot be both delimiters, so it is
    the value itself (upstream #808's guard). ``"\\n\\n"`` is an empty value
    framed on both sides.
    """
    if value == "\n":
        return value
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value
