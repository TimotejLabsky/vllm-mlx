"""#114: a thinking budget must not arm on a request whose template renders
thinking OFF through chat_template_kwargs alone (no top-level enable_thinking).
"""

from unittest.mock import patch

import pytest

from vllm_mlx import server
from vllm_mlx.api.models import ChatCompletionRequest, Message


class _Engine:
    is_mllm = False
    preserve_native_tool_format = False


def _built(**request_fields):
    request = ChatCompletionRequest(
        model="m", messages=[Message(role="user", content="hi")], **request_fields
    )
    with (
        patch.object(server, "_default_thinking_token_budget", 6144),
        patch.object(server, "_build_thinking_processor", return_value=None) as build,
    ):
        server._prepare_chat_completion_invocation(_Engine(), request, 16)
    return build


@pytest.mark.parametrize("effort", ["none", "None", " NONE "])
def test_reasoning_effort_none_does_not_arm_the_budget(effort):
    assert not _built(reasoning_effort=effort).called


def test_template_kwargs_thinking_off_does_not_arm_the_budget():
    build = _built(chat_template_kwargs={"enable_thinking": False})
    assert not build.called


def test_thinking_on_still_arms_the_budget():
    build = _built()
    assert build.called
    assert build.call_args.kwargs["prompt_has_think_tag"] is True
