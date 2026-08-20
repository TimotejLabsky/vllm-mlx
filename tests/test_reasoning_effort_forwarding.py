# SPDX-License-Identifier: Apache-2.0
"""Patch #76: forward OpenAI ``reasoning_effort`` into the chat template.

Before this patch the parameter was accepted by the API model and then
silently dropped for every value except ``"none"``, so per-request effort
switching was impossible and Qwen3.8 ran at its template default (``xhigh``)
on every request.

The fail-safe requirement is the interesting half: Qwen3.8's template calls
``raise_exception()`` for anything outside ``xhigh|medium|low``, and clients
(Claude Code, opencode) send ``high``. An unsupported value must never reach
the template.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_mlx.utils.reasoning_effort import (
    EFFORT_FALLBACK_KEY,
    HARMONY_EFFORT_LEVELS,
    normalize_effort_in_template_kwargs,
    normalize_reasoning_effort,
    render_with_effort_fallback,
    strip_effort_fallback,
    template_effort_vocabulary,
)

# Verbatim shape of Qwen/Qwen3.8-27B chat_template.jinja lines 45-51 — the
# family whose raise_exception this patch exists to avoid.
QWEN38_TEMPLATE = """\
{%- set reasoning_instructions = '' %}
{%- if enable_thinking is undefined or enable_thinking is true %}
    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
        {{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ '. Supported types are xhigh (default), medium, and low.') }}
    {%- endif %}
    {%- if resolved_reasoning_effort == 'xhigh' %}
        {%- set reasoning_instructions = 'Reasoning effort is set to xhigh.' %}
    {%- elif resolved_reasoning_effort == 'low' %}
        {%- set reasoning_instructions = 'Reasoning effort is set to low.' %}
    {%- endif %}
{%- endif %}
{{- reasoning_instructions }}
{%- for message in messages %}{{- message['content'] }}{%- endfor %}
"""

# gpt-oss: interpolated with a default, never validated. Nothing can raise, so
# the right verdict is "unknown vocabulary, pass it through".
GPT_OSS_TEMPLATE = """\
{%- if reasoning_effort is not defined %}
    {%- set reasoning_effort = "medium" %}
{%- endif %}
{{- "Reasoning: " + reasoning_effort + "\\n\\n" }}
{%- for message in messages %}{{- message['content'] }}{%- endfor %}
"""

# Qwen3.6 / GLM / most of the fleet: no reasoning_effort variable at all.
PLAIN_TEMPLATE = "{%- for message in messages %}{{- message['content'] }}{%- endfor %}"


class TestTemplateVocabulary:
    """Value-vocabulary introspection (the part no upstream engine does)."""

    def test_qwen38_vocabulary_is_extracted(self):
        assert template_effort_vocabulary(QWEN38_TEMPLATE) == frozenset(
            {"xhigh", "medium", "low"}
        )

    def test_template_without_the_variable_reports_empty(self):
        # Empty frozenset == "drop the kwarg", distinct from None == "unknown".
        assert template_effort_vocabulary(PLAIN_TEMPLATE) == frozenset()

    def test_unvalidated_template_reports_unknown(self):
        # One literal is a default, not a vocabulary; nothing here can raise.
        assert template_effort_vocabulary(GPT_OSS_TEMPLATE) is None

    def test_assignment_only_mention_does_not_count_as_a_read(self):
        # jinja2.meta distinguishes a declared variable from an undeclared one,
        # which a substring match cannot.
        source = (
            "{%- set reasoning_effort = 'medium' %}"
            "{%- for m in messages %}{{- m['content'] }}{%- endfor %}"
        )
        assert template_effort_vocabulary(source) == frozenset()

    def test_unparseable_source_is_unknown_not_empty(self):
        # vLLM #36907: a bare template *name* parses as trivial Jinja with an
        # empty variable set, which must not be read as "declares nothing".
        assert template_effort_vocabulary("some_template_name") is None


class TestNormalization:
    def test_high_maps_to_xhigh(self):
        vocab = template_effort_vocabulary(QWEN38_TEMPLATE)
        assert normalize_reasoning_effort("high", vocab) == "xhigh"

    def test_supported_value_passes_through(self):
        vocab = template_effort_vocabulary(QWEN38_TEMPLATE)
        assert normalize_reasoning_effort("medium", vocab) == "medium"

    def test_case_and_whitespace_are_tolerated(self):
        vocab = template_effort_vocabulary(QWEN38_TEMPLATE)
        assert normalize_reasoning_effort("  HIGH ", vocab) == "xhigh"

    def test_minimal_folds_onto_low(self):
        vocab = template_effort_vocabulary(QWEN38_TEMPLATE)
        assert normalize_reasoning_effort("minimal", vocab) == "low"

    def test_medium_is_dropped_rather_than_guessed(self):
        # No unambiguous neighbour; the template default beats a coin flip.
        assert normalize_reasoning_effort("medium", {"low", "high"}) is None

    def test_unknown_vocabulary_passes_the_value_through(self):
        assert normalize_reasoning_effort("high", None) == "high"

    def test_empty_vocabulary_drops_the_value(self):
        assert normalize_reasoning_effort("high", frozenset()) is None

    def test_non_string_values_are_dropped(self):
        # An explicit chat_template_kwargs {"reasoning_effort": None} would
        # otherwise reach `None not in (...)` and raise.
        assert normalize_reasoning_effort(None, {"low"}) is None
        assert normalize_reasoning_effort(3, {"low"}) is None

    def test_xhigh_folds_onto_high_for_harmony(self):
        assert normalize_reasoning_effort("xhigh", HARMONY_EFFORT_LEVELS) == "high"

    def test_kwargs_helper_rewrites_in_place(self):
        kwargs = {"tokenize": False, "reasoning_effort": "high"}
        normalize_effort_in_template_kwargs(kwargs, QWEN38_TEMPLATE)
        assert kwargs == {"tokenize": False, "reasoning_effort": "xhigh"}

    def test_kwargs_helper_drops_unsupported(self):
        kwargs = {"tokenize": False, "reasoning_effort": "high"}
        normalize_effort_in_template_kwargs(kwargs, PLAIN_TEMPLATE)
        assert kwargs == {"tokenize": False}


class TestFloorBeatsNeighbour:
    """Resolution order: exact -> operator floor -> neighbour -> drop.

    vLLM-aligned: no engine invents a level for an unsupported request, the
    operator's --default-chat-template-kwargs decides. Without the floor rung,
    an unsupported value lands on the TEMPLATE default, which on Qwen3.8 is
    xhigh -- so garbage would buy more thinking than sending nothing, and
    Claude Code's `high` would resolve to the runaway-thinking mode itself.
    """

    def setup_method(self):
        self.vocab = template_effort_vocabulary(QWEN38_TEMPLATE)

    def test_unsupported_value_takes_the_floor_not_the_neighbour(self):
        # Without a floor this is "xhigh" (the neighbour). With one, medium.
        assert normalize_reasoning_effort("high", self.vocab) == "xhigh"
        assert normalize_reasoning_effort("high", self.vocab, "medium") == "medium"

    def test_garbage_takes_the_floor_instead_of_the_template_default(self):
        assert normalize_reasoning_effort("banana", self.vocab) is None
        assert normalize_reasoning_effort("banana", self.vocab, "medium") == "medium"

    def test_supported_value_still_beats_the_floor(self):
        assert normalize_reasoning_effort("low", self.vocab, "medium") == "low"
        assert normalize_reasoning_effort("xhigh", self.vocab, "medium") == "xhigh"

    def test_floor_the_template_rejects_falls_through_to_neighbour(self):
        # gpt-oss vocabulary; a "xhigh" floor is not renderable there.
        assert (
            normalize_reasoning_effort("xhigh", HARMONY_EFFORT_LEVELS, "xhigh")
            == "high"
        )

    def test_floor_is_case_normalized(self):
        assert normalize_reasoning_effort("high", self.vocab, " MEDIUM ") == "medium"

    def test_kwargs_helper_consumes_the_reserved_key(self):
        kwargs = {
            "tokenize": False,
            "reasoning_effort": "high",
            EFFORT_FALLBACK_KEY: "medium",
        }
        normalize_effort_in_template_kwargs(kwargs, QWEN38_TEMPLATE)
        assert kwargs == {"tokenize": False, "reasoning_effort": "medium"}

    def test_reserved_key_never_survives_even_with_no_effort(self):
        # It is fork-internal plumbing; it must not reach the Jinja context.
        kwargs = {"tokenize": False, EFFORT_FALLBACK_KEY: "medium"}
        normalize_effort_in_template_kwargs(kwargs, QWEN38_TEMPLATE)
        assert kwargs == {"tokenize": False}

    def test_strip_helper_leaves_the_original_untouched(self):
        original = {"reasoning_effort": "low", EFFORT_FALLBACK_KEY: "medium"}
        stripped = strip_effort_fallback(original)
        assert stripped == {"reasoning_effort": "low"}
        assert EFFORT_FALLBACK_KEY in original


class TestRenderFallback:
    """Backstop for templates whose vocabulary could not be introspected."""

    def test_retry_without_effort_when_the_template_rejects_it(self):
        calls = []

        def render(**kwargs):
            calls.append(dict(kwargs))
            if "reasoning_effort" in kwargs:
                raise ValueError("Unexpected reasoning effort high.")
            return "prompt"

        kwargs = {"tokenize": False, "reasoning_effort": "high"}
        assert render_with_effort_fallback(render, kwargs, model_name="m") == "prompt"
        assert len(calls) == 2
        # The caller's dict is cleaned too: SimpleEngine re-renders it for the
        # system-prefix probe and a mismatch is a silent KV cache miss.
        assert "reasoning_effort" not in kwargs

    def test_unrelated_failures_are_not_masked(self):
        def render(**_kwargs):
            raise RuntimeError("model is on fire")

        with pytest.raises(RuntimeError, match="on fire"):
            render_with_effort_fallback(
                render, {"reasoning_effort": "high"}, model_name="m"
            )

    def test_no_retry_when_effort_was_not_passed(self):
        calls = []

        def render(**kwargs):
            calls.append(kwargs)
            raise ValueError("something else")

        with pytest.raises(ValueError):
            render_with_effort_fallback(render, {"tokenize": False}, model_name="m")
        assert len(calls) == 1

    def test_warns_once_per_model(self, caplog):
        import vllm_mlx.utils.reasoning_effort as mod

        mod._warned_models.discard("noisy-model")

        def render(**kwargs):
            if "reasoning_effort" in kwargs:
                raise ValueError("nope")
            return "prompt"

        with caplog.at_level("WARNING"):
            for _ in range(3):
                render_with_effort_fallback(
                    render, {"reasoning_effort": "high"}, model_name="noisy-model"
                )
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1


def _qwen38_tokenizer():
    """A tokenizer whose template behaves like Qwen3.8's: it raises."""
    from jinja2 import Environment

    env = Environment()

    def _raise_exception(message):
        raise ValueError(message)

    def apply_chat_template(messages, **kwargs):
        kwargs.pop("tokenize", None)
        kwargs.pop("add_generation_prompt", None)
        template = env.from_string(QWEN38_TEMPLATE)
        return template.render(
            messages=messages, raise_exception=_raise_exception, **kwargs
        )

    return SimpleNamespace(
        chat_template=QWEN38_TEMPLATE,
        apply_chat_template=apply_chat_template,
    )


class TestBatchedEngineIntegration:
    """End-to-end through the real render path, against a raising template."""

    def _engine(self, tokenizer):
        with patch("vllm_mlx.engine.batched.is_mllm_model", return_value=False):
            from vllm_mlx.engine.batched import BatchedEngine

            engine = BatchedEngine("qwen3.8-27b")
        engine._tokenizer = tokenizer
        return engine

    def test_high_is_normalized_instead_of_raising(self):
        engine = self._engine(_qwen38_tokenizer())
        prompt = engine._apply_chat_template(
            [{"role": "user", "content": "hi"}],
            chat_template_kwargs={"reasoning_effort": "high"},
        )
        # Rendered, not raised — and it picked the template's top level.
        assert "Reasoning effort is set to xhigh." in prompt

    def test_unsupported_value_is_dropped_not_raised(self):
        engine = self._engine(_qwen38_tokenizer())
        prompt = engine._apply_chat_template(
            [{"role": "user", "content": "hi"}],
            chat_template_kwargs={"reasoning_effort": "ludicrous"},
        )
        # Dropped -> the template's own default (xhigh) applies.
        assert "Reasoning effort is set to xhigh." in prompt

    def test_backstop_catches_a_template_the_probe_misreads(self):
        """Introspection is defeated; the render/retry backstop must still hold."""
        tokenizer = _qwen38_tokenizer()
        # Hide the source so no vocabulary can be extracted: the value goes
        # through unnormalized and the template raises.
        tokenizer.chat_template = None
        engine = self._engine(tokenizer)
        prompt = engine._apply_chat_template(
            [{"role": "user", "content": "hi"}],
            chat_template_kwargs={"reasoning_effort": "high"},
        )
        assert "Reasoning effort is set to xhigh." in prompt

    def test_supported_value_reaches_the_template(self):
        engine = self._engine(_qwen38_tokenizer())
        prompt = engine._apply_chat_template(
            [{"role": "user", "content": "hi"}],
            chat_template_kwargs={"reasoning_effort": "low"},
        )
        assert "Reasoning effort is set to xhigh." not in prompt

    def test_floor_wins_over_neighbour_end_to_end(self):
        """Claude Code's `high` must not resolve to Qwen3.8's xhigh default."""
        engine = self._engine(_qwen38_tokenizer())
        prompt = engine._apply_chat_template(
            [{"role": "user", "content": "hi"}],
            chat_template_kwargs={
                "reasoning_effort": "high",
                EFFORT_FALLBACK_KEY: "medium",
            },
        )
        # medium emits no instruction at all; xhigh would have emitted one.
        assert "Reasoning effort is set to" not in prompt

    def test_garbage_falls_to_the_floor_end_to_end(self):
        engine = self._engine(_qwen38_tokenizer())
        prompt = engine._apply_chat_template(
            [{"role": "user", "content": "hi"}],
            chat_template_kwargs={
                "reasoning_effort": "banana",
                EFFORT_FALLBACK_KEY: "low",
            },
        )
        assert "Reasoning effort is set to low." in prompt

    def test_reserved_key_never_reaches_the_template(self):
        """A leaked fork-internal kwarg would land in the Jinja context."""
        seen = {}
        tokenizer = _qwen38_tokenizer()
        real = tokenizer.apply_chat_template

        def spy(messages, **kwargs):
            seen.update(kwargs)
            return real(messages, **kwargs)

        tokenizer.apply_chat_template = spy
        engine = self._engine(tokenizer)
        engine._apply_chat_template(
            [{"role": "user", "content": "hi"}],
            chat_template_kwargs={
                "reasoning_effort": "low",
                EFFORT_FALLBACK_KEY: "medium",
            },
        )
        assert EFFORT_FALLBACK_KEY not in seen

    def test_backstop_ladder_lands_on_the_floor(self):
        """Unparseable template: retry the floor before dropping outright."""
        tokenizer = _qwen38_tokenizer()
        tokenizer.chat_template = None  # defeat introspection
        engine = self._engine(tokenizer)
        prompt = engine._apply_chat_template(
            [{"role": "user", "content": "hi"}],
            chat_template_kwargs={
                "reasoning_effort": "high",
                EFFORT_FALLBACK_KEY: "low",
            },
        )
        # Floor rendered, rather than dropping to the xhigh template default.
        assert "Reasoning effort is set to low." in prompt

    def test_harmony_branch_folds_xhigh_onto_high(self):
        engine = self._engine(MagicMock())
        engine.use_harmony_rendering = True
        with patch(
            "vllm_mlx.utils.harmony_render.render_messages",
            return_value="harmony-prompt",
        ) as render:
            engine._apply_chat_template(
                [{"role": "user", "content": "Hi"}],
                chat_template_kwargs={"reasoning_effort": "xhigh"},
            )
        assert render.call_args.kwargs["reasoning_effort"] == "high"


class TestServerForwarding:
    """`_prepare_chat_completion_invocation` builds chat_template_kwargs."""

    def _prepare(self, monkeypatch, **request_fields):
        import vllm_mlx.server as srv

        monkeypatch.setattr(
            srv, "_prepare_chat_messages", lambda _e, _m: ([], [], [], [], False)
        )
        monkeypatch.setattr(
            srv,
            "_prepare_json_logits_processor",
            lambda *_a, **_k: ([], None),
        )

        fields = {
            "messages": [],
            "max_tokens": None,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "min_p": None,
            "presence_penalty": None,
            "repetition_penalty": None,
            "response_format": None,
            "tools": None,
            "tool_choice": None,
            "enable_thinking": None,
            "reasoning_effort": None,
            "chat_template_kwargs": None,
            "specprefill": None,
            "specprefill_keep_pct": None,
            "stop": None,
            "thinking_token_budget": None,
        }
        fields.update(request_fields)
        request = SimpleNamespace(**fields)
        prepared = srv._prepare_chat_completion_invocation(
            SimpleNamespace(is_mllm=False), request, 128
        )
        return prepared.chat_kwargs.get("chat_template_kwargs") or {}

    def test_plain_value_is_forwarded(self, monkeypatch):
        ctk = self._prepare(monkeypatch, reasoning_effort="high")
        assert ctk["reasoning_effort"] == "high"

    def test_value_is_normalized_case_insensitively(self, monkeypatch):
        ctk = self._prepare(monkeypatch, reasoning_effort="High")
        assert ctk["reasoning_effort"] == "high"

    def test_explicit_chat_template_kwargs_wins(self, monkeypatch):
        ctk = self._prepare(
            monkeypatch,
            reasoning_effort="high",
            chat_template_kwargs={"reasoning_effort": "low"},
        )
        assert ctk["reasoning_effort"] == "low"

    def test_explicit_null_in_chat_template_kwargs_also_wins(self, monkeypatch):
        ctk = self._prepare(
            monkeypatch,
            reasoning_effort="high",
            chat_template_kwargs={"reasoning_effort": None},
        )
        assert ctk["reasoning_effort"] is None

    def test_none_still_disables_thinking_and_is_not_forwarded(self, monkeypatch):
        # Home Assistant's conversation route depends on this mapping.
        ctk = self._prepare(monkeypatch, reasoning_effort="none")
        assert ctk == {"enable_thinking": False}

    def test_absent_parameter_adds_nothing(self, monkeypatch):
        assert self._prepare(monkeypatch) == {}

    def test_beats_the_server_default(self, monkeypatch):
        # A --default-chat-template-kwargs value is a fallback; a per-request
        # effort has to be able to override it or per-request switching is
        # still impossible.
        import vllm_mlx.server as srv

        monkeypatch.setattr(
            srv, "_default_chat_template_kwargs", {"reasoning_effort": "medium"}
        )
        ctk = self._prepare(monkeypatch, reasoning_effort="high")
        assert ctk["reasoning_effort"] == "high"

    def test_server_default_applies_when_the_request_is_silent(self, monkeypatch):
        import vllm_mlx.server as srv

        monkeypatch.setattr(
            srv, "_default_chat_template_kwargs", {"reasoning_effort": "medium"}
        )
        assert self._prepare(monkeypatch)["reasoning_effort"] == "medium"

    def test_route_floor_is_handed_to_the_engine(self, monkeypatch):
        """The engine can't see --default-chat-template-kwargs; ship it along."""
        import vllm_mlx.server as srv

        monkeypatch.setattr(
            srv, "_default_chat_template_kwargs", {"reasoning_effort": "medium"}
        )
        ctk = self._prepare(monkeypatch, reasoning_effort="high")
        assert ctk["reasoning_effort"] == "high"
        assert ctk[EFFORT_FALLBACK_KEY] == "medium"

    def test_no_floor_configured_ships_no_reserved_key(self, monkeypatch):
        import vllm_mlx.server as srv

        monkeypatch.setattr(srv, "_default_chat_template_kwargs", None)
        ctk = self._prepare(monkeypatch, reasoning_effort="high")
        assert EFFORT_FALLBACK_KEY not in ctk

    def test_none_does_not_ship_a_floor(self, monkeypatch):
        # "none" means enable_thinking=False, not "pick a level".
        import vllm_mlx.server as srv

        monkeypatch.setattr(
            srv, "_default_chat_template_kwargs", {"reasoning_effort": "medium"}
        )
        ctk = self._prepare(monkeypatch, reasoning_effort="none")
        assert ctk["enable_thinking"] is False


class TestResponsesApiForwarding:
    """Responses `reasoning.effort` rides the same path (deliberate, not skipped)."""

    def test_effort_reaches_the_chat_request(self):
        import vllm_mlx.server as srv
        from vllm_mlx.api.responses_models import (
            ResponseReasoningConfig,
            ResponsesRequest,
        )

        request = ResponsesRequest(
            model="m", input="hi", reasoning=ResponseReasoningConfig(effort="high")
        )
        chat_request = srv._responses_request_to_chat_request(request)
        assert chat_request.reasoning_effort == "high"

    def test_absent_reasoning_config_leaves_it_unset(self):
        import vllm_mlx.server as srv
        from vllm_mlx.api.responses_models import ResponsesRequest

        chat_request = srv._responses_request_to_chat_request(
            ResponsesRequest(model="m", input="hi")
        )
        assert chat_request.reasoning_effort is None
