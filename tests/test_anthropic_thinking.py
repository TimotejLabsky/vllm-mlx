# SPDX-License-Identifier: Apache-2.0
"""Fork #120: the Anthropic ``thinking`` field reaches the engine.

``AnthropicRequest`` did not declare ``thinking``, so pydantic dropped it and
``{"type": "disabled"}`` had no effect on thinking routes (Claude Code and
other Anthropic clients send it on every request).
"""

import pytest

import vllm_mlx.server as srv
from vllm_mlx.api.anthropic_adapter import anthropic_to_openai
from vllm_mlx.api.anthropic_models import AnthropicRequest


def _req(thinking):
    body = {
        "model": "m",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
    }
    if thinking is not None:
        body["thinking"] = thinking
    return AnthropicRequest(**body)


@pytest.mark.parametrize(
    ("thinking", "expected"),
    [
        (None, (None, None)),
        ({"type": "disabled"}, (False, None)),
        ({"type": "enabled", "budget_tokens": 2048}, (True, 2048)),
        ({"type": "enabled"}, (True, None)),
        ({"type": "enabled", "budget_tokens": 0}, (True, None)),
        ({"type": "adaptive"}, (None, None)),
        ({"type": "something-new"}, (None, None)),
    ],
)
def test_thinking_maps_onto_openai_fields(thinking, expected):
    out = anthropic_to_openai(_req(thinking))
    assert (out.enable_thinking, out.thinking_token_budget) == expected


def test_disabled_reaches_thinking_disabled_resolution():
    out = anthropic_to_openai(_req({"type": "disabled"}))
    assert srv._thinking_disabled(out, {}) is True


@pytest.mark.parametrize(
    ("route_default", "requested", "expected"),
    [
        (6144, 31999, 6144),  # never lifts the route cap
        (6144, 2048, 2048),  # may tighten it
        (None, 31999, 31999),  # no route cap: the client's budget applies
        (6144, None, None),  # no budget asked: route default applies later
    ],
)
def test_budget_only_tightens_route_default(
    monkeypatch, route_default, requested, expected
):
    monkeypatch.setattr(srv, "_default_thinking_token_budget", route_default)
    thinking = {"type": "enabled"}
    if requested is not None:
        thinking["budget_tokens"] = requested
    out = anthropic_to_openai(_req(thinking))
    srv._clamp_anthropic_thinking_budget(out)
    assert out.thinking_token_budget == expected


class _Engine:
    is_mllm = False
    preserve_native_tool_format = False
    use_harmony_rendering = False


@pytest.fixture
def budget_route(monkeypatch):
    """A route with --default-thinking-token-budget 6144 and a stub builder."""
    built = []

    class _Proc(srv._ThinkingAwareLogitsProcessor):
        def __init__(self, budget):  # no tokenizer needed for the stub
            self.budget = budget

    def fake_build(engine, budget, *, inner=None, prompt_has_think_tag=True):
        built.append(budget)
        return _Proc(budget)

    monkeypatch.setattr(srv, "_default_thinking_token_budget", 6144)
    monkeypatch.setattr(srv, "_build_thinking_processor", fake_build)
    return built


def _prepared(thinking):
    out = anthropic_to_openai(_req(thinking))
    srv._clamp_anthropic_thinking_budget(out)
    return srv._prepare_anthropic_invocation(_Engine(), out, 64)


def test_disabled_reaches_engine_kwargs_and_skips_budget(budget_route):
    prepared = _prepared({"type": "disabled"})
    assert prepared.chat_kwargs["enable_thinking"] is False
    assert prepared.thinking_processor is None and budget_route == []


def test_route_budget_now_enforced_on_messages(budget_route):
    """Before #120 /v1/messages never built the budget processor at all."""
    prepared = _prepared(None)
    assert budget_route == [6144]
    assert prepared.thinking_processor in prepared.chat_kwargs["logits_processors"]
    # ...and the streaming reasoning parser stays allowed with it installed.
    assert srv._anthropic_stream_reasoning_allowed(prepared, prepared.chat_kwargs)


def test_client_budget_tightens(budget_route):
    _prepared({"type": "enabled", "budget_tokens": 2048})
    assert budget_route == [2048]


def test_structured_output_still_disables_stream_reasoning():
    prepared = srv.PreparedChatInvocation(
        messages=[],
        chat_kwargs={"logits_processors": [object()]},
        response_format=None,
        json_logits_processor=object(),
    )
    assert not srv._anthropic_stream_reasoning_allowed(prepared, prepared.chat_kwargs)
