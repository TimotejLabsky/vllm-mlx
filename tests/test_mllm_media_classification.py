# SPDX-License-Identifier: Apache-2.0
"""Media classification fixes on the batched MLLM path.

Two holes, one class of bug:

- Audio-bearing requests were treated as text-only by the mid-batch-extend
  and chunked-prefill interleave filters (``not r.images and not r.videos``),
  which route rows through the language model alone — silently dropping the
  audio. ``MLLMBatchRequest.has_media`` now counts audio.
- ``video_url`` content parts were never converted to the HuggingFace
  ``{"type": "video"}`` form (the conversion gate only counted images and
  audio), so a video-only request reached the processor with raw OpenAI
  parts and no video placeholder tokens.
"""

from vllm_mlx.engine.batched import BatchedEngine
from vllm_mlx.mllm_batch_generator import MLLMBatchRequest


def _request(**kwargs) -> MLLMBatchRequest:
    return MLLMBatchRequest(uid=1, request_id="req-1", prompt="hi", **kwargs)


class TestHasMedia:
    def test_text_only(self):
        assert not _request().has_media

    def test_images_are_media(self):
        assert _request(images=["img.png"]).has_media

    def test_videos_are_media(self):
        assert _request(videos=["clip.mp4"]).has_media

    def test_audio_is_media(self):
        assert _request(audio=["speech.wav"]).has_media

    def test_empty_lists_are_text_only(self):
        assert not _request(images=[], videos=[], audio=[]).has_media


class TestPrepareMllmMessages:
    def test_video_url_converted(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "video_url", "video_url": {"url": "file://clip.mp4"}},
                ],
            }
        ]
        prepared = BatchedEngine._prepare_mllm_messages(messages)
        assert prepared[0]["content"] == [
            {"type": "text", "text": "describe"},
            {"type": "video"},
        ]

    def test_image_and_audio_still_converted(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "x.png"}},
                    {"type": "audio_url", "audio_url": {"url": "x.wav"}},
                ],
            }
        ]
        prepared = BatchedEngine._prepare_mllm_messages(messages)
        assert prepared[0]["content"] == [{"type": "image"}, {"type": "audio"}]


class TestTemplateConversionGate:
    """The gate deciding whether _prepare_mllm_messages runs must count videos."""

    def _engine_with_recording_processor(self):
        recorded = {}

        class _Processor:
            def apply_chat_template(self, messages, **_kwargs):
                recorded["messages"] = messages
                return "PROMPT"

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._is_mllm = True
        engine._processor = _Processor()
        engine._model_name = "test-vlm"
        # The `tokenizer` property resolves via `_processor.tokenizer`,
        # falling back to the processor itself — fine for this test.
        return engine, recorded

    def test_video_only_request_triggers_conversion(self):
        engine, recorded = self._engine_with_recording_processor()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": "file://clip.mp4"}}
                ],
            }
        ]
        engine._apply_chat_template(messages, num_videos=1)
        assert recorded["messages"][0]["content"] == [{"type": "video"}]

    def test_no_media_no_conversion(self):
        engine, recorded = self._engine_with_recording_processor()
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello"}],
            }
        ]
        engine._apply_chat_template(messages)
        # Content parts pass through untouched.
        assert recorded["messages"][0]["content"] == [{"type": "text", "text": "hello"}]
