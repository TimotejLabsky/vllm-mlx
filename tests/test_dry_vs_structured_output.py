# SPDX-License-Identifier: Apache-2.0
"""PATCHES.md #93 — DRY must not police declared structure.

DRY penalises exact repeated token sequences to break prose loops. Structured
output legitimately repeats: a JSON array re-emits `{"content": "` per element.
Its sequence breakers cannot prevent this because they resolve to
single-character token ids while BPE merges JSON punctuation into tokens like
`' {"'`, `'":'` and `'",'` — so the breaker-free run stays long and the
penalty becomes a prohibition rather than a nudge.
"""

import pytest

import vllm_mlx.server as srv


class _Req:
    """Minimal stand-in for ChatCompletionRequest's duck-typed surface."""

    def __init__(self, **kw):
        self.tools = kw.get("tools")
        self.tool_choice = kw.get("tool_choice")
        self.response_format = kw.get("response_format")


TOOLS = [{"type": "function", "function": {"name": "todowrite"}}]


# ------------------------------------------------- when DRY must be off


def test_declared_tools_suppress_dry():
    assert srv._dry_suppressed_for_structure(_Req(tools=TOOLS)) == "tools"


def test_json_schema_response_format_suppresses_dry():
    fmt = {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}
    assert srv._dry_suppressed_for_structure(_Req(response_format=fmt)) == (
        "response_format"
    )


def test_json_object_response_format_suppresses_dry():
    assert (
        srv._dry_suppressed_for_structure(_Req(response_format={"type": "json_object"}))
        == "response_format"
    )


def test_response_format_as_object_not_dict():
    class _Fmt:
        type = "json_schema"

    assert srv._dry_suppressed_for_structure(_Req(response_format=_Fmt())) == (
        "response_format"
    )


# ------------------------------------------------- when DRY must stay on


def test_plain_prose_request_keeps_dry():
    """The loop-breaking case DRY exists for must not regress."""
    assert srv._dry_suppressed_for_structure(_Req()) is None


def test_tool_choice_none_keeps_dry():
    """Tools declared but explicitly disabled — no structure will be emitted."""
    assert (
        srv._dry_suppressed_for_structure(_Req(tools=TOOLS, tool_choice="none")) is None
    )


def test_empty_tool_list_keeps_dry():
    assert srv._dry_suppressed_for_structure(_Req(tools=[])) is None


def test_text_response_format_keeps_dry():
    assert (
        srv._dry_suppressed_for_structure(_Req(response_format={"type": "text"}))
        is None
    )


# ------------------------------------------------- the in-place application


def test_suppression_zeroes_the_multiplier():
    kwargs = {"dry_multiplier": 0.8, "dry_allowed_length": 4}
    srv._apply_dry_structure_suppression(kwargs, _Req(tools=TOOLS))
    assert kwargs["dry_multiplier"] == 0.0


def test_suppression_overrides_gateway_injected_values():
    """LiteLLM injects dry_* via extra_body; the request value must lose."""
    kwargs = {"dry_multiplier": 0.8, "dry_base": 1.75, "dry_allowed_length": 4}
    srv._apply_dry_structure_suppression(kwargs, _Req(tools=TOOLS))
    assert kwargs["dry_multiplier"] == 0.0


def test_suppression_beats_env_defaults_too():
    """A None multiplier would defer to VLLM_MLX_DRY_* env; 0.0 does not."""
    kwargs = {"dry_multiplier": None}
    srv._apply_dry_structure_suppression(kwargs, _Req(tools=TOOLS))
    assert kwargs["dry_multiplier"] == 0.0


def test_prose_request_is_left_untouched():
    kwargs = {"dry_multiplier": 0.8, "dry_allowed_length": 4}
    srv._apply_dry_structure_suppression(kwargs, _Req())
    assert kwargs["dry_multiplier"] == 0.8


def test_already_off_is_not_rewritten():
    kwargs = {"dry_multiplier": 0}
    srv._apply_dry_structure_suppression(kwargs, _Req(tools=TOOLS))
    assert kwargs["dry_multiplier"] == 0


# ---------------------------------------- the mechanism, pinned numerically


def test_zero_multiplier_disables_the_processor():
    """multiplier <= 0 is DRY's own off switch — the contract #93 relies on."""
    from vllm_mlx.dry_sampler import build_dry_processor

    class _Tok:
        def encode(self, s, **kw):
            return [1]

    assert build_dry_processor(_Tok(), multiplier=0.0) is None


def test_penalty_at_the_deployed_settings_is_a_prohibition():
    """Why suppression, not re-tuning: the penalty dwarfs any logit.

    Deployed gateway values were multiplier=0.8, base=1.75, allowed_length=4.
    A JSON element measured 16 breaker-free tokens on Qwen3.8.
    """
    multiplier, base, allowed, run = 0.8, 1.75, 4, 16
    penalty = multiplier * base ** (run - allowed)
    assert penalty > 500, penalty
    # A 2-element list stays under water — which is why short lists worked
    # and the failure began at the third element.
    assert multiplier * base ** (6 - allowed) < 3


@pytest.mark.parametrize(
    "breakers,merged_token",
    [
        (("\n", ":", '"', "*"), ' {"'),
        (("\n", ":", '"', "*"), '":'),
        (("\n", ":", '"', "*"), '",'),
    ],
)
def test_single_char_breakers_cannot_match_merged_json_tokens(breakers, merged_token):
    """The reason re-tuning breakers is not a real fix.

    Breakers resolve to token ids. BPE merges JSON punctuation, so a
    single-character breaker id never equals the id of `' {"'` / `'":'` /
    `'",'`, and the breaker-free run stays long.
    """
    assert merged_token not in breakers
    assert len(merged_token) > 1
