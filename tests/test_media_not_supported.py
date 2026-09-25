# SPDX-License-Identifier: Apache-2.0
"""400-on-media guard for text-only routes (vision series).

Text-only engines used to answer media requests with the media silently
STRIPPED out of the messages (extract_multimodal_content side list,
dropped on the batched LLM branch): a 200 with a hallucinated answer
about an image the model never saw. The server now rejects pre-engine
(and pre-StreamingResponse, so streams get a real 400 too); the batched
LLM branches raise as defense in depth.
"""

import asyncio

import pytest

from vllm_mlx.engine.base import MediaNotSupported, PromptTooLong
from vllm_mlx.engine.batched import BatchedEngine
from vllm_mlx.server import (
    ChatCompletionRequest,
    _prepare_anthropic_invocation,
    _prepare_chat_completion_invocation,
)


class _TextOnlyEngine:
    is_mllm = False
    preserve_native_tool_format = False
    use_harmony_rendering = False


class _MllmEngine:
    is_mllm = True
    preserve_native_tool_format = False
    use_harmony_rendering = False


def _media_request(**kwargs):
    return ChatCompletionRequest(
        model="m",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,aGk="},
                    },
                ],
            }
        ],
        **kwargs,
    )


def _text_request():
    return ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "hello"}]
    )


class TestExceptionShape:
    def test_code_and_hierarchy(self):
        exc = MediaNotSupported("nope")
        assert exc.code == "media_not_supported"
        assert isinstance(exc, ValueError)
        # Must share PromptTooLong's non-retryable 400 translation path.
        assert not isinstance(exc, type(PromptTooLong))


class TestServerGuard:
    def test_chat_prep_rejects_media_on_text_only_engine(self):
        with pytest.raises(MediaNotSupported):
            _prepare_chat_completion_invocation(
                _TextOnlyEngine(), _media_request(), effective_max_tokens=16
            )

    def test_chat_prep_admits_media_on_mllm_engine(self):
        # No raise; MLLM engines keep media parts inside the messages
        # (the engine extracts them itself — side lists stay empty).
        prepared = _prepare_chat_completion_invocation(
            _MllmEngine(), _media_request(), effective_max_tokens=16
        )
        parts = prepared.messages[-1]["content"]
        assert any(
            isinstance(p, dict) and p.get("type") == "image_url" for p in parts
        )

    def test_chat_prep_admits_text_on_text_only_engine(self):
        prepared = _prepare_chat_completion_invocation(
            _TextOnlyEngine(), _text_request(), effective_max_tokens=16
        )
        assert "images" not in prepared.chat_kwargs

    def test_anthropic_prep_rejects_media_on_text_only_engine(self):
        with pytest.raises(MediaNotSupported):
            _prepare_anthropic_invocation(
                _TextOnlyEngine(), _media_request(), effective_max_tokens=16
            )


class TestEngineDefenseInDepth:
    def _llm_engine(self):
        engine = BatchedEngine.__new__(BatchedEngine)
        engine._loaded = True
        engine._is_mllm = False
        engine._mllm_scheduler = None
        engine._engine = None
        return engine

    def test_generate_raises_on_media(self):
        engine = self._llm_engine()

        with pytest.raises(MediaNotSupported):
            asyncio.run(engine.generate(prompt="p", images=["x.png"]))

    def test_stream_generate_raises_on_media(self):
        engine = self._llm_engine()

        async def _consume():
            async for _ in engine.stream_generate(prompt="p", audio=["x.wav"]):
                pass

        with pytest.raises(MediaNotSupported):
            asyncio.run(_consume())


class TestAnthropicImageBlocksReachTheGuard:
    """#116 (upstream #776): Anthropic image blocks used to be dropped by the
    adapter, so /v1/messages never reached the guard above and a text route
    answered about an image it never saw. They now convert to image_url parts
    and hit the same 400 as chat completions; vision routes receive them.
    """

    def _converted(self):
        from vllm_mlx.api.anthropic_adapter import anthropic_to_openai
        from vllm_mlx.api.anthropic_models import AnthropicRequest

        return anthropic_to_openai(
            AnthropicRequest(
                model="m",
                max_tokens=16,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "iVBORw0KGgo=",
                                },
                            },
                            {"type": "text", "text": "What is this?"},
                        ],
                    }
                ],
            )
        )

    def test_text_route_rejects_anthropic_image(self):
        with pytest.raises(MediaNotSupported):
            _prepare_anthropic_invocation(
                _TextOnlyEngine(), self._converted(), effective_max_tokens=16
            )

    def test_vision_route_accepts_anthropic_image(self):
        _prepare_anthropic_invocation(
            _MllmEngine(), self._converted(), effective_max_tokens=16
        )
