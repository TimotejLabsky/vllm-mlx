# SPDX-License-Identifier: Apache-2.0
"""Tests for SimpleEngine concurrency handling."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

pytestmark = pytest.mark.anyio


class _FakeCacheLayer:
    """Stand-in for one mlx_lm KVCache layer in the fork's snapshot path.

    The system-KV MISS branch snapshots ``c.state`` (a tuple of arrays),
    ``mx.eval``s each element, and logs ``sum(c.nbytes for c in cache)``,
    so the fake needs a real-array tuple state and an ``nbytes`` int.
    """

    nbytes = 8

    def __init__(self):
        self.state = (mx.zeros(1), mx.zeros(1))
        # Pre-evaluate: the serialized worker rebinds to a fresh MLX stream,
        # so lazy arrays scheduled on the test thread's default stream would
        # fail to evaluate there.
        mx.eval(*self.state)


class TestSimpleEngineConcurrency:
    """Test SimpleEngine lock behavior with concurrent requests."""

    @pytest.fixture
    def anyio_backend(self):
        return "asyncio"

    @pytest.fixture
    def mock_model(self):
        """Create a mock model that tracks concurrent calls."""
        model = MagicMock()
        model.tokenizer = MagicMock()
        model.tokenizer.encode = MagicMock(return_value=[1, 2, 3])

        # Track concurrent executions
        model._concurrent_count = 0
        model._max_concurrent = 0

        def generate_side_effect(**kwargs):
            model._concurrent_count += 1
            model._max_concurrent = max(model._max_concurrent, model._concurrent_count)
            # Simulate some work
            import time

            time.sleep(0.05)
            model._concurrent_count -= 1
            result = MagicMock()
            result.text = "test response"
            result.tokens = [1, 2, 3]
            result.finish_reason = "stop"
            return result

        model.generate = MagicMock(side_effect=generate_side_effect)

        # stream_generate tracks concurrency the same way so tests that
        # exercise SimpleEngine.generate() (which is now an accumulator
        # over stream_generate) see the same serialization behavior.
        def stream_generate_side_effect(**kwargs):
            model._concurrent_count += 1
            model._max_concurrent = max(model._max_concurrent, model._concurrent_count)
            import time

            time.sleep(0.05)
            model._concurrent_count -= 1
            chunk = MagicMock()
            chunk.text = "test response"
            chunk.tokens = [1, 2, 3]
            chunk.finished = True
            chunk.finish_reason = "stop"
            chunk.prompt_tokens = 3
            chunk.completion_tokens = 3
            yield chunk

        model.stream_generate = MagicMock(side_effect=stream_generate_side_effect)
        return model

    @pytest.fixture
    def mock_llm_model(self):
        """Create a mock LLM model."""
        model = MagicMock()
        model.tokenizer = MagicMock()
        model.tokenizer.encode = MagicMock(return_value=[1, 2, 3])

        # Track concurrent executions
        model._concurrent_count = 0
        model._max_concurrent = 0

        def chat_side_effect(**kwargs):
            model._concurrent_count += 1
            model._max_concurrent = max(model._max_concurrent, model._concurrent_count)
            import time

            time.sleep(0.05)
            model._concurrent_count -= 1
            result = MagicMock()
            result.text = "test response"
            result.tokens = [1, 2, 3]
            result.finish_reason = "stop"
            return result

        model.chat = MagicMock(side_effect=chat_side_effect)
        return model

    @pytest.mark.anyio
    async def test_lock_prevents_concurrent_generate(self, mock_model):
        """Test that the lock prevents concurrent generate calls."""
        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = mock_model
            engine._loaded = True
            engine._generation_lock_admission = "wait"

            # Launch multiple concurrent generate calls
            tasks = [
                engine.generate(prompt=f"test prompt {i}", max_tokens=10)
                for i in range(5)
            ]

            await asyncio.gather(*tasks)

            # With the lock, max concurrent should be 1
            assert mock_model._max_concurrent == 1, (
                f"Expected max concurrent to be 1, but got {mock_model._max_concurrent}. "
                "The lock is not working correctly."
            )

    @pytest.mark.anyio
    async def test_lock_prevents_concurrent_chat(self, mock_llm_model):
        """Test that the lock prevents concurrent chat calls.

        The fork routes plain-text non-MLLM chat through
        ``_stream_generate_text`` (system-KV text route, patch #4), which
        tokenizes for real and cannot run against a MagicMock model. Media-
        shaped messages keep ``chat()`` on the legacy blocking
        ``self._model.chat`` path (the real ``has_media_content`` gate), which
        is the seam this test mocks; the serialization property under test —
        ``_run_blocking_serialized`` holding ``_generation_lock`` — is the
        same lock every route uses.
        """
        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = mock_llm_model
            engine._loaded = True
            engine._generation_lock_admission = "wait"
            # chat() recomputes prompt accounting via apply_chat_template
            # (tokenize=True) on the legacy path.
            mock_llm_model.tokenizer.apply_chat_template = MagicMock(
                return_value=[1, 2, 3]
            )

            # Launch multiple concurrent chat calls (media part pins the
            # legacy mlx blocking-chat path on a non-MLLM engine).
            tasks = [
                engine.chat(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": "x.png"}},
                                {"type": "text", "text": f"test {i}"},
                            ],
                        }
                    ],
                    max_tokens=10,
                )
                for i in range(5)
            ]

            await asyncio.gather(*tasks)

            # With the lock, max concurrent should be 1
            assert mock_llm_model._max_concurrent == 1, (
                f"Expected max concurrent to be 1, but got {mock_llm_model._max_concurrent}. "
                "The lock is not working correctly."
            )

    def test_default_admission_waits_and_respects_env(self, monkeypatch):
        """Fork default admission is "wait"; the env var is respected.

        Diverges from upstream PR #540 (patch #15): upstream defaults to
        fail_fast and has a bug where VLLM_MLX_SIMPLE_ENGINE_LOCK_ADMISSION
        is validated but then unconditionally overwritten with fail_fast.
        The fork defaults to "wait" (OpenCode fires title + main request
        simultaneously) and honors the env var as an opt-in to load shedding.
        """
        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            monkeypatch.delenv(
                "VLLM_MLX_SIMPLE_ENGINE_LOCK_ADMISSION", raising=False
            )
            engine = SimpleEngine("test-model")
            assert engine._generation_lock_admission == "wait"

            monkeypatch.setenv(
                "VLLM_MLX_SIMPLE_ENGINE_LOCK_ADMISSION", "fail_fast"
            )
            engine_env = SimpleEngine("test-model")
            assert engine_env._generation_lock_admission == "fail_fast"

    @pytest.mark.anyio
    async def test_fail_fast_admission_rejects_second_serialized_request(self):
        """fail_fast admission (opt-in; fork default is "wait") rejects a
        second serialized request with EngineBusy instead of queueing."""
        from vllm_mlx.engine.base import EngineBusy
        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._loaded = True
            engine._generation_lock_admission = "fail_fast"

            started = threading.Event()
            release = threading.Event()

            def slow_call():
                started.set()
                release.wait(timeout=1.0)
                return "ok"

            first = asyncio.create_task(
                engine._run_blocking_serialized(
                    slow_call,
                    request_id="first-serialized-request",
                )
            )
            await asyncio.to_thread(started.wait, 1.0)

            with pytest.raises(EngineBusy) as excinfo:
                await engine._run_blocking_serialized(
                    lambda: "late",
                    request_id="second-serialized-request",
                )

            assert "text_generation_busy" == excinfo.value.code
            assert "active=first-serialized-request" in str(excinfo.value)
            assert engine._generation_busy_rejections == 1

            release.set()
            await first

    def test_lock_admission_env_wait_is_preserved(self, monkeypatch):
        """VLLM_MLX_SIMPLE_ENGINE_LOCK_ADMISSION=wait keeps queued behavior."""
        from vllm_mlx.engine.simple import SimpleEngine

        monkeypatch.setenv("VLLM_MLX_SIMPLE_ENGINE_LOCK_ADMISSION", "wait")

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")

        assert engine._generation_lock_admission == "wait"
        assert engine.get_stats()["generation_lock"]["admission"] == "wait"

    @pytest.mark.anyio
    async def test_blocking_serialized_tracks_active_request(self):
        """Busy errors include the current blocking serialized holder."""
        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._loaded = True

            def slow_call():
                import time

                time.sleep(0.05)
                return "ok"

            first = asyncio.create_task(
                engine._run_blocking_serialized(
                    slow_call,
                    request_id="probe-active-holder",
                )
            )
            for _ in range(100):
                if engine._generation_lock.locked():
                    break
                await asyncio.sleep(0.001)

            assert "probe-active-holder" in engine._generation_lock_holder_summary()

            await first
            assert engine._generation_lock_holder_summary() == "none"

    async def test_chat_with_tools_aggregates_streaming_path(self, mock_llm_model):
        """Tool-enabled non-stream chat should use the streaming path."""
        from vllm_mlx.engine.simple import SimpleEngine

        async def fake_stream_chat(*args, **kwargs):
            yield MagicMock(
                text="partial",
                tokens=[1],
                prompt_tokens=11,
                completion_tokens=1,
                finish_reason=None,
                finished=False,
            )
            yield MagicMock(
                text='<|im_end|><tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>',
                tokens=[7, 8, 9],
                prompt_tokens=11,
                completion_tokens=4,
                finish_reason="stop",
                finished=True,
            )

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = mock_llm_model
            engine._loaded = True
            engine.stream_chat = fake_stream_chat  # type: ignore[method-assign]

            output = await engine.chat(
                messages=[{"role": "user", "content": "run pwd"}],
                max_tokens=16,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            )

            assert (
                output.text
                == '<tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>'
            )
            assert output.tokens == [7, 8, 9]
            assert output.prompt_tokens == 11
            assert output.completion_tokens == 4
            assert output.finish_reason == "stop"
            mock_llm_model.chat.assert_not_called()

    @pytest.mark.skip(
        reason="upstream #523/#541 cache API; fork carries the patches "
        "#9/#12/#13 system-KV stack — see PATCHES.md. The fork routes "
        "non-MLLM text chat through _stream_generate_text, whose contract "
        "is plain-dict messages (server.py normalizes Pydantic Message "
        "objects via model_dump before the engine boundary); the upstream "
        "in-stream_chat normalization gate this test exercised is only "
        "reachable on the non-MLLM media path."
    )
    @pytest.mark.anyio
    async def test_stream_chat_cache_path_accepts_pydantic_message_objects(self):
        """`stream_chat`'s declared signature is ``list[dict]`` but real callers
        (``server.py``'s streaming endpoint, ``test_server.py``'s direct
        invocations) pass Pydantic ``Message`` objects. The system-prefix
        KV-cache eligibility check on this path uses ``.get('role')`` /
        ``dict(m)`` semantics; without normalisation the iteration raises
        ``'Message' object has no attribute 'get'`` before the call ever
        reaches the underlying ``stream_generate``."""
        from vllm_mlx.api.models import Message
        from vllm_mlx.engine.simple import SimpleEngine

        # ``apply_chat_template`` returns identical strings for both
        # probe-divergence renders → boundary stays at 0 → the cache path
        # is correctly skipped and execution falls through to
        # ``self.stream_generate``. The test's value is asserting no
        # ``AttributeError`` leaks out of the message-normalisation step.
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = (
            "<|im_start|>system\nyou are an assistant<|im_end|>\n"
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        )
        tokenizer.bos_token = None
        tokenizer.encode = MagicMock(return_value=[1, 2, 3])

        model = MagicMock()
        model.tokenizer = tokenizer

        captured_stream_generate = []

        async def fake_stream_generate(*, prompt, **kwargs):
            captured_stream_generate.append({"prompt": prompt, "kwargs": kwargs})
            out = MagicMock(
                text="hi back",
                new_text="hi back",
                prompt_tokens=3,
                completion_tokens=1,
                finished=True,
                finish_reason="stop",
            )
            yield out

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = model
            engine._loaded = True
            engine._supports_system_kv_cache = True
            engine.stream_generate = fake_stream_generate  # type: ignore[method-assign]

            messages = [
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="hi"),
            ]

            chunks = [c async for c in engine.stream_chat(messages=messages)]

        # No AttributeError was raised → normalisation worked.
        # apply_chat_template was called at least 3 times: once for the
        # initial ``prompt`` build and twice more for the Alpha/Bravo
        # probe-divergence renders.
        assert tokenizer.apply_chat_template.call_count >= 3
        assert len(captured_stream_generate) == 1
        assert chunks and chunks[0].text == "hi back"

    @pytest.mark.anyio
    async def test_stream_chat_skips_cache_path_when_no_system_message(self):
        """If the message list has no system role, the fork's text route
        (``_stream_generate_text``, which non-MLLM stream_chat dispatches to)
        must short-circuit ``has_system = False``: no system-prefix
        tokenization, no snapshot store, no LRU mutation — the raw rendered
        prompt goes straight to ``mlx_lm.stream_generate`` uncached."""
        from vllm_mlx.engine.simple import SimpleEngine

        rendered = "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = rendered
        tokenizer.bos_token = None
        tokenizer.encode = MagicMock(return_value=[1, 2, 3])
        tokenizer.eos_token_id = 99

        model = MagicMock()
        model.tokenizer = tokenizer
        # The text route drives the raw mlx_lm model (self._model.model).
        model.model = MagicMock()

        captured: list[dict] = []

        def fake_stream_generate(model_arg, tokenizer_arg, prompt, **kw):
            captured.append({"prompt": prompt, **kw})
            yield SimpleNamespace(text="hi", finish_reason="stop")

        with (
            patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False),
            patch("vllm_mlx.engine.simple._bind_worker_generation_streams"),
            patch("mlx_lm.stream_generate", fake_stream_generate),
        ):
            engine = SimpleEngine("test-model")
            engine._model = model
            engine._loaded = True

            chunks = [
                c
                async for c in engine.stream_chat(
                    messages=[{"role": "user", "content": "hello"}],
                )
            ]

        # No system → system-KV machinery never engages: the prompt render is
        # the only template call, the system prefix is never tokenized, and
        # nothing is stored or counted.
        assert tokenizer.apply_chat_template.call_count == 1
        assert tokenizer.encode.call_count == 0
        assert engine._system_kv_snapshot is None
        assert len(engine._system_kv_lru) == 0
        assert engine._system_kv_misses == 0
        assert engine._system_kv_hits == 0
        # Uncached: the full rendered prompt string, no prompt_cache kwarg.
        assert captured and captured[0]["prompt"] == rendered
        assert "prompt_cache" not in captured[0]
        assert chunks and chunks[0].text == "hi"

    @pytest.mark.anyio
    async def test_stream_chat_cache_path_surfaces_error_when_mlx_raises(self):
        """Fork semantics (patches #4/#9): the text route's serialized worker
        has no uncached in-request fallback — a failure before the first
        generated token propagates to the caller as the original exception
        (upstream #523's silent stream_generate fallback only exists on the
        fork's non-MLLM media path). The failed request must also not poison
        engine state: nothing may be stored in the snapshot slot or LRU."""
        from vllm_mlx.engine.simple import SimpleEngine

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\nhello<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        tokenizer.bos_token = None
        # Long enough token lists that the system-prefix slice is a proper
        # prefix of the full sequence, so the MISS prefill branch is entered.
        tokenizer.encode = MagicMock(
            side_effect=[
                list(range(50)),  # full prompt
                list(range(20)),  # system prefix (proper prefix of above)
                list(range(40)),  # extended prefix (up to gen-prompt marker)
            ]
        )
        tokenizer.eos_token_id = 99

        model = MagicMock()
        model.tokenizer = tokenizer
        model.model = MagicMock()

        # Force the cache-aware worker to raise before the first emit.
        def make_prompt_cache_raises(*args, **kwargs):
            raise RuntimeError("simulated mlx-lm failure")

        with (
            patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False),
            patch("vllm_mlx.engine.simple._bind_worker_generation_streams"),
            patch(
                "mlx_lm.models.cache.make_prompt_cache",
                side_effect=make_prompt_cache_raises,
            ),
            patch("mlx_lm.sample_utils.make_sampler", return_value=MagicMock()),
        ):
            engine = SimpleEngine("test-model")
            engine._model = model
            engine._loaded = True

            chunks = []
            with pytest.raises(RuntimeError, match="simulated mlx-lm failure"):
                async for c in engine.stream_chat(
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "hello"},
                    ],
                ):
                    chunks.append(c)

        # Error surfaced before any token was streamed, and the failed MISS
        # stored nothing.
        assert chunks == []
        assert engine._system_kv_snapshot is None
        assert len(engine._system_kv_lru) == 0

    @pytest.mark.anyio
    async def test_stream_chat_cache_path_honors_decode_controls(self):
        """Fork semantics (patches #4/#9, inverting upstream #523's gate):
        active decode controls do NOT skip the system-KV cache path. The
        fork's text route builds the sampler from ``top_k``/``min_p``, layers
        penalty processors via ``make_logits_processors``, forwards request
        ``logits_processors`` into ``mlx_lm.stream_generate``, and matches
        ``stop`` sequences in the consumer loop — all while still storing and
        reusing the system-prefix KV snapshot."""
        from vllm_mlx.engine.simple import SimpleEngine

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = (
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            "<|im_start|>user\nhello<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        tokenizer.bos_token = None
        tokenizer.encode = MagicMock(
            side_effect=[
                list(range(50)),  # full prompt
                list(range(20)),  # system prefix (proper prefix of above)
                list(range(40)),  # extended prefix (up to gen-prompt marker)
            ]
        )
        tokenizer.eos_token_id = 99

        model = MagicMock()
        model.tokenizer = tokenizer
        model.model = MagicMock()

        gen_calls: list[dict] = []
        sampler_calls: list[dict] = []
        penalty_calls: list[dict] = []

        def sentinel_processor(tokens, logits):
            return logits

        def penalty_processor(tokens, logits):
            return logits

        def fake_stream_generate(model_arg, tokenizer_arg, prompt, **kw):
            gen_calls.append({"prompt": prompt, **kw})
            # Text containing the stop sequence: the consumer loop must
            # detect it and finish with reason "stop".
            yield SimpleNamespace(text="ok<|im_end|>", finish_reason=None)

        def fake_make_sampler(**kw):
            sampler_calls.append(kw)
            return MagicMock()

        def fake_make_logits_processors(**kw):
            penalty_calls.append(kw)
            return [penalty_processor]

        with (
            patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False),
            patch(
                "mlx_lm.models.cache.make_prompt_cache",
                return_value=[_FakeCacheLayer()],
            ),
            patch("mlx_lm.sample_utils.make_sampler", side_effect=fake_make_sampler),
            patch(
                "mlx_lm.sample_utils.make_logits_processors",
                side_effect=fake_make_logits_processors,
            ),
            patch("mlx_lm.stream_generate", fake_stream_generate),
        ):
            engine = SimpleEngine("test-model")
            engine._model = model
            engine._loaded = True

            chunks = [
                c
                async for c in engine.stream_chat(
                    messages=[
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "hello"},
                    ],
                    stop=["<|im_end|>"],
                    logits_processors=[sentinel_processor],
                    top_k=40,
                    min_p=0.1,
                    presence_penalty=0.5,
                    repetition_penalty=1.2,
                )
            ]

        # The cache path ran despite active decode controls: the MISS stored
        # a snapshot and only the post-prefix suffix went to mlx_lm.
        assert engine._system_kv_misses == 1
        assert engine._system_kv_snapshot is not None
        assert engine._system_kv_token_count == 40
        assert gen_calls, "mlx_lm.stream_generate was not invoked"
        assert gen_calls[0]["prompt"].tolist() == list(range(40, 50))
        assert "prompt_cache" in gen_calls[0]
        # ...and every decode control was threaded through, not dropped.
        assert sampler_calls == [
            {"temp": 0.7, "top_p": 0.9, "top_k": 40, "min_p": 0.1}
        ]
        assert penalty_calls == [
            {"repetition_penalty": 1.2, "presence_penalty": 0.5}
        ]
        assert gen_calls[0]["logits_processors"] == [
            sentinel_processor,
            penalty_processor,
        ]
        # stop sequences are enforced in the consumer loop.
        assert chunks and chunks[-1].finished
        assert chunks[-1].finish_reason == "stop"

    @pytest.mark.anyio
    async def test_stream_chat_takes_cache_path_when_decode_controls_are_no_ops(self):
        """server.py always sets ``top_k=0``, ``min_p=0.0``, ``presence_penalty=0.0``,
        ``repetition_penalty=1.0`` (no-ops) in ``chat_kwargs``; the common
        path must still hit the system-KV cache. Fork semantics: the text
        route locates the system prefix via the ChatML marker on the single
        prompt render (no probe-divergence re-renders), prefills it once, and
        stores the snapshot in the active slot."""
        from vllm_mlx.engine.simple import SimpleEngine

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = (
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            "<|im_start|>user\nhello<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        tokenizer.bos_token = None
        tokenizer.encode = MagicMock(
            side_effect=[
                list(range(50)),  # full prompt
                list(range(20)),  # system prefix (proper prefix of above)
                list(range(40)),  # extended prefix (up to gen-prompt marker)
            ]
        )
        tokenizer.eos_token_id = 99

        model = MagicMock()
        model.tokenizer = tokenizer
        model.model = MagicMock()

        gen_calls: list[dict] = []

        def fake_stream_generate(model_arg, tokenizer_arg, prompt, **kw):
            gen_calls.append({"prompt": prompt, **kw})
            yield SimpleNamespace(text="ok", finish_reason="stop")

        with (
            patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False),
            patch(
                "mlx_lm.models.cache.make_prompt_cache",
                return_value=[_FakeCacheLayer()],
            ),
            patch("mlx_lm.sample_utils.make_sampler", return_value=MagicMock()),
            patch("mlx_lm.stream_generate", fake_stream_generate),
        ):
            engine = SimpleEngine("test-model")
            engine._model = model
            engine._loaded = True

            chunks = [
                c
                async for c in engine.stream_chat(
                    messages=[
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "hello"},
                    ],
                    top_k=0,
                    min_p=0.0,
                    presence_penalty=0.0,
                    repetition_penalty=1.0,
                )
            ]

        # Cache path engaged: one MISS stored an extended-prefix snapshot in
        # the active slot, and only the gen-prompt tail went to mlx_lm.
        assert engine._system_kv_misses == 1
        assert engine._system_kv_snapshot is not None
        assert engine._system_kv_token_count == 40
        assert engine._system_kv_token_ids == list(range(40))
        assert gen_calls and gen_calls[0]["prompt"].tolist() == list(range(40, 50))
        assert "prompt_cache" in gen_calls[0]
        # Marker-based prefix detection: exactly one template render, no
        # probe-divergence re-renders.
        assert tokenizer.apply_chat_template.call_count == 1
        assert chunks and chunks[0].text == "ok"

    @pytest.mark.anyio
    async def test_stream_chat_cache_path_layers_mtp_on_top(self):
        """Fork semantics (patches #4/#9, inverting upstream #523's gate):
        ``self._mtp`` does NOT skip the system-KV cache path. The text route
        stacks a fresh MTP cache on top of the snapshotted backbone cache and
        signals speculative decoding to mlx_lm via ``num_draft_tokens`` —
        cache-eligible and uncached turns share the same MTP semantics."""
        from vllm_mlx.engine.simple import SimpleEngine

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = (
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            "<|im_start|>user\nhello<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        tokenizer.bos_token = None
        tokenizer.encode = MagicMock(
            side_effect=[
                list(range(50)),  # full prompt
                list(range(20)),  # system prefix (proper prefix of above)
                list(range(40)),  # extended prefix (up to gen-prompt marker)
            ]
        )
        tokenizer.eos_token_id = 99

        backbone_layer = _FakeCacheLayer()
        model = MagicMock()
        model.tokenizer = tokenizer
        model.model = MagicMock()
        # The text route checks model.mtp and stacks make_mtp_cache() output.
        model.model.mtp = MagicMock(name="mtp_head")
        model.model.make_mtp_cache = MagicMock(return_value=["mtp-cache"])

        gen_calls: list[dict] = []

        def fake_stream_generate(model_arg, tokenizer_arg, prompt, **kw):
            gen_calls.append({"prompt": prompt, **kw})
            yield SimpleNamespace(text="ok", finish_reason="stop")

        with (
            patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False),
            patch(
                "mlx_lm.models.cache.make_prompt_cache",
                return_value=[backbone_layer],
            ),
            patch("mlx_lm.sample_utils.make_sampler", return_value=MagicMock()),
            patch("mlx_lm.stream_generate", fake_stream_generate),
        ):
            engine = SimpleEngine("test-model", mtp=True, mtp_num_draft_tokens=4)
            engine._model = model
            engine._loaded = True

            chunks = [
                c
                async for c in engine.stream_chat(
                    messages=[
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "hello"},
                    ],
                )
            ]

        # Cache path engaged despite MTP.
        assert engine._system_kv_misses == 1
        assert engine._system_kv_snapshot is not None
        # MTP cache stacked on the snapshotted backbone; speculative decode
        # signalled via num_draft_tokens (the text route never passes mtp=).
        assert gen_calls, "mlx_lm.stream_generate was not invoked"
        assert gen_calls[0]["prompt_cache"] == [backbone_layer, "mtp-cache"]
        assert gen_calls[0]["num_draft_tokens"] == 4
        assert "mtp" not in gen_calls[0]
        assert gen_calls[0]["prompt"].tolist() == list(range(40, 50))
        assert chunks and chunks[0].text == "ok"

    @pytest.mark.anyio
    async def test_stream_chat_cache_path_runs_with_specprefill_loaded(self):
        """Fork semantics (patches #4/#9, inverting upstream #523's gate): a
        loaded SpecPrefill draft model does NOT skip the system-KV cache path.
        The text route integrates the two — when SpecPrefill engages it scores
        only the post-prefix suffix on top of the snapshotted backbone cache —
        and below ``specprefill_threshold`` (suffix too short) the request
        stays on the cached normal decode path without touching the scorer."""
        from vllm_mlx.engine.simple import SimpleEngine

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = (
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            "<|im_start|>user\nhello<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        tokenizer.bos_token = None
        tokenizer.encode = MagicMock(
            side_effect=[
                list(range(50)),  # full prompt
                list(range(20)),  # system prefix (proper prefix of above)
                list(range(40)),  # extended prefix (up to gen-prompt marker)
            ]
        )
        tokenizer.eos_token_id = 99

        model = MagicMock()
        model.tokenizer = tokenizer
        model.model = MagicMock()

        gen_calls: list[dict] = []

        def fake_stream_generate(model_arg, tokenizer_arg, prompt, **kw):
            gen_calls.append({"prompt": prompt, **kw})
            yield SimpleNamespace(text="ok", finish_reason="stop")

        score_tokens_mock = MagicMock(name="score_tokens")

        with (
            patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False),
            patch(
                "mlx_lm.models.cache.make_prompt_cache",
                return_value=[_FakeCacheLayer()],
            ),
            patch("mlx_lm.sample_utils.make_sampler", return_value=MagicMock()),
            patch("mlx_lm.stream_generate", fake_stream_generate),
            patch("vllm_mlx.specprefill.score_tokens", score_tokens_mock),
        ):
            # Default specprefill_threshold (8192) far exceeds the 10-token
            # suffix, so SpecPrefill must not engage for this request.
            engine = SimpleEngine("test-model")
            engine._model = model
            engine._draft_model = MagicMock(name="specprefill_draft_model")
            engine._loaded = True

            chunks = [
                c
                async for c in engine.stream_chat(
                    messages=[
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "hello"},
                    ],
                )
            ]

        # Cache path engaged despite the loaded draft model.
        assert engine._system_kv_misses == 1
        assert engine._system_kv_snapshot is not None
        assert gen_calls, "mlx_lm.stream_generate was not invoked"
        assert gen_calls[0]["prompt"].tolist() == list(range(40, 50))
        assert "prompt_cache" in gen_calls[0]
        # Suffix below threshold → SpecPrefill scorer untouched.
        score_tokens_mock.assert_not_called()
        assert chunks and chunks[0].text == "ok"

    @pytest.mark.anyio
    async def test_stream_chat_cache_path_forwards_max_kv_size(self):
        """Fork semantics (patches #4/#9, inverting upstream #523's gate): a
        non-zero engine ``max_kv_size`` does NOT force the uncached path.
        The text route forwards the bound into ``make_prompt_cache(model,
        max_kv_size=N)`` when it builds the snapshot cache, so bounded-KV
        serving and the system-KV cache compose instead of excluding each
        other."""
        from vllm_mlx.engine.simple import SimpleEngine

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = (
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            "<|im_start|>user\nhello<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        tokenizer.bos_token = None
        tokenizer.encode = MagicMock(
            side_effect=[
                list(range(50)),  # full prompt
                list(range(20)),  # system prefix (proper prefix of above)
                list(range(40)),  # extended prefix (up to gen-prompt marker)
            ]
        )
        tokenizer.eos_token_id = 99

        model = MagicMock()
        model.tokenizer = tokenizer
        model.model = MagicMock()

        gen_calls: list[dict] = []
        cache_calls: list[dict] = []

        def fake_make_prompt_cache(*args, **kw):
            cache_calls.append(kw)
            return [_FakeCacheLayer()]

        def fake_stream_generate(model_arg, tokenizer_arg, prompt, **kw):
            gen_calls.append({"prompt": prompt, **kw})
            yield SimpleNamespace(text="ok", finish_reason="stop")

        with (
            patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False),
            patch(
                "mlx_lm.models.cache.make_prompt_cache",
                side_effect=fake_make_prompt_cache,
            ),
            patch("mlx_lm.sample_utils.make_sampler", return_value=MagicMock()),
            patch("mlx_lm.stream_generate", fake_stream_generate),
        ):
            engine = SimpleEngine("test-model", max_kv_size=4096)
            engine._model = model
            engine._loaded = True

            chunks = [
                c
                async for c in engine.stream_chat(
                    messages=[
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "hello"},
                    ],
                )
            ]

        # Cache path engaged under bounded KV, with the bound forwarded into
        # the snapshot cache constructor.
        assert engine._system_kv_misses == 1
        assert engine._system_kv_snapshot is not None
        assert cache_calls and cache_calls[0].get("max_kv_size") == 4096
        assert gen_calls and gen_calls[0]["prompt"].tolist() == list(range(40, 50))
        assert "prompt_cache" in gen_calls[0]
        assert chunks and chunks[0].text == "ok"

    @pytest.mark.anyio
    async def test_system_kv_probe_denylists_rotating_kv_cache_only(self):
        """Fork semantics (patches #6/#9): the ``start()`` snapshot-safety
        probe is a DENYLIST, not upstream #541's all-KVCache allowlist.
        Only ``RotatingKVCache`` entries (sliding-window models — gemma3_text,
        olmo3, recurrent_gemma — whose ``.state`` aliases in-place-mutated
        ring buffers) disable ``_supports_system_kv_cache``. An
        ``ArraysCache`` + ``KVCache`` hybrid (Gated DeltaNet layers in
        Qwen3.6 etc.) is ALLOWED — patch #6 shallow-copies the list state at
        capture/restore."""
        from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            # RotatingKVCache anywhere in the probe cache → snapshot-unsafe.
            engine = SimpleEngine("test-model")
            engine._model = MagicMock()  # pre-set so start() skips loading
            with patch(
                "mlx_lm.models.cache.make_prompt_cache",
                return_value=[KVCache(), RotatingKVCache(max_size=8)],
            ):
                await engine.start()
            assert engine._supports_system_kv_cache is False

            # ArraysCache+KVCache hybrid passes the denylist (upstream's
            # allowlist would have rejected it).
            engine_hybrid = SimpleEngine("test-model")
            engine_hybrid._model = MagicMock()
            with patch(
                "mlx_lm.models.cache.make_prompt_cache",
                return_value=[KVCache(), ArraysCache(2)],
            ):
                await engine_hybrid.start()
            assert engine_hybrid._supports_system_kv_cache is True

            # Probe failure fails closed.
            engine_err = SimpleEngine("test-model")
            engine_err._model = MagicMock()
            with patch(
                "mlx_lm.models.cache.make_prompt_cache",
                side_effect=RuntimeError("probe boom"),
            ):
                await engine_err.start()
            assert engine_err._supports_system_kv_cache is False

    @pytest.mark.anyio
    async def test_stream_generate_text_env_disable_skips_snapshot_persistence(
        self, monkeypatch
    ):
        """Fork semantics: the text route's snapshot kill-switch is
        ``VLLM_MLX_DISABLE_SYSTEM_KV=1`` (``_is_system_kv_safe``), used for
        models that drift/loop on cache replay (hybrid Qwen3.5/3.6 family —
        mlx-lm#1162); the ``_supports_system_kv_cache`` probe only gates the
        separate non-MLLM media-path branch. When disabled, the request still
        prefills its own prompt cache, but NOTHING may persist: no active-slot
        snapshot, no LRU entry, and no future HIT."""
        from vllm_mlx.engine.simple import SimpleEngine

        monkeypatch.setenv("VLLM_MLX_DISABLE_SYSTEM_KV", "1")

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = (
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            "<|im_start|>user\nhello<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        tokenizer.bos_token = None
        tokenizer.encode = MagicMock(
            side_effect=[
                list(range(50)),  # full prompt
                list(range(20)),  # system prefix (proper prefix of above)
                list(range(40)),  # extended prefix (up to gen-prompt marker)
            ]
        )
        tokenizer.decode = MagicMock(return_value="")
        tokenizer.eos_token_id = 99

        text_model = MagicMock()
        text_model.mtp = None

        engine = SimpleEngine("test-model", force_mllm=True)
        engine._loaded = True
        engine._text_model = text_model
        engine._text_tokenizer = tokenizer

        def fake_stream_generate(*args, **kw):
            # _run_all iterates this synchronously (`for resp in ...`), so the
            # mock must be a sync generator, not an async one.
            yield SimpleNamespace(
                text="ok",
                new_text="ok",
                prompt_tokens=3,
                completion_tokens=1,
                finished=True,
                finish_reason="stop",
                token=99,
            )

        with (
            patch(
                "mlx_lm.models.cache.make_prompt_cache",
                return_value=[_FakeCacheLayer()],
            ),
            patch("mlx_lm.sample_utils.make_sampler", return_value=MagicMock()),
            patch(
                "mlx_lm.sample_utils.make_logits_processors",
                return_value=[],
            ),
            patch("mlx_lm.stream_generate", side_effect=fake_stream_generate),
        ):
            chunks = [
                c
                async for c in engine._stream_generate_text(
                    messages=[
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "hello"},
                    ],
                    max_tokens=4,
                    temperature=0.7,
                    top_p=0.95,
                )
            ]

        assert chunks, "expected the request to still stream output"
        # The miss is counted (the prefill genuinely happened)…
        assert engine._system_kv_misses == 1
        # …but nothing was persisted for replay: no active slot, no LRU
        # entry, no hit accounting.
        assert engine._system_kv_snapshot is None
        assert engine._system_kv_hash is None
        assert engine._system_kv_token_ids is None
        assert len(engine._system_kv_lru) == 0
        assert engine._system_kv_hits == 0
        assert engine._system_kv_tokens_saved == 0

    @pytest.mark.anyio
    async def test_stream_generate_text_forwards_max_kv_size_under_bounded_kv(self):
        """Bounded-KV contract for the fork's text route (patches #4/#9,
        diverging from upstream #541 which skipped the cache branch entirely
        under ``_max_kv_size > 0``): ``_stream_generate_text`` builds its
        snapshot cache via ``make_prompt_cache(model, max_kv_size=N)`` — the
        engine bound is forwarded, not silently dropped — and the system-KV
        snapshot still stores and serves."""
        from vllm_mlx.engine.simple import SimpleEngine

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = (
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            "<|im_start|>user\nhello<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        tokenizer.bos_token = None
        tokenizer.encode = MagicMock(
            side_effect=[
                list(range(50)),  # full prompt
                list(range(20)),  # system prefix (proper prefix of above)
                list(range(40)),  # extended prefix (up to gen-prompt marker)
            ]
        )
        tokenizer.decode = MagicMock(return_value="")
        tokenizer.eos_token_id = 99

        text_model = MagicMock()
        text_model.mtp = None

        engine = SimpleEngine("test-model", force_mllm=True, max_kv_size=2048)
        engine._loaded = True
        engine._text_model = text_model
        engine._text_tokenizer = tokenizer

        cache_calls: list[dict] = []

        def fake_make_prompt_cache(*args, **kw):
            cache_calls.append(kw)
            return [_FakeCacheLayer()]

        def fake_stream_generate(*args, **kw):
            yield SimpleNamespace(
                text="ok",
                new_text="ok",
                prompt_tokens=3,
                completion_tokens=1,
                finished=True,
                finish_reason="stop",
                token=99,
            )

        with (
            patch(
                "mlx_lm.models.cache.make_prompt_cache",
                side_effect=fake_make_prompt_cache,
            ),
            patch("mlx_lm.sample_utils.make_sampler", return_value=MagicMock()),
            patch(
                "mlx_lm.sample_utils.make_logits_processors",
                return_value=[],
            ),
            patch("mlx_lm.stream_generate", side_effect=fake_stream_generate),
        ):
            chunks = [
                c
                async for c in engine._stream_generate_text(
                    messages=[
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "hello"},
                    ],
                    max_tokens=4,
                    temperature=0.7,
                    top_p=0.95,
                )
            ]

        assert chunks, "expected the cached path to emit at least one chunk"
        # The snapshot cache was built WITH the engine bound.
        assert cache_calls and cache_calls[0].get("max_kv_size") == 2048
        # And the cache path genuinely engaged under bounded KV.
        assert engine._system_kv_misses == 1
        assert engine._system_kv_snapshot is not None
        assert engine._system_kv_token_count == 40

    @pytest.mark.anyio
    async def test_stream_chat_system_cache_copies_arrays_cache_state(self):
        """Hybrid cache snapshots must not alias ``ArraysCache.state`` lists."""
        import mlx.core as mx

        from vllm_mlx.engine.simple import SimpleEngine

        def apply_chat_template_side_effect(messages, **kwargs):
            user_content = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    user_content = m.get("content") or ""
                    break
            return (
                "<|im_start|>system\nYou are helpful.<|im_end|>\n"
                f"<|im_start|>user\n{user_content}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.side_effect = apply_chat_template_side_effect
        tokenizer.bos_token = None
        tokenizer.encode = MagicMock(side_effect=[list(range(10)), list(range(5))])

        model = MagicMock()
        model.tokenizer = tokenizer

        class MutableListCache:
            def __init__(self):
                self.cache = [None]

            @property
            def state(self):
                return self.cache

            @state.setter
            def state(self, value):
                self.cache = value

            @property
            def nbytes(self):
                return sum(c.nbytes for c in self.cache if c is not None)

        cache_entry = MutableListCache()

        def fake_model(tokens, cache):
            cache[0].cache[0] = mx.array([[1, 2, 3]])
            return MagicMock()

        model.model = fake_model

        def fake_stream_generate(model_arg, tokenizer_arg, prompt, **kwargs):
            # Simulate suffix/decode state advancing after the system-prefix
            # snapshot has been stored. The saved snapshot must not follow this
            # mutable list change.
            prompt_cache = kwargs["prompt_cache"]
            prompt_cache[0].cache[0] = mx.array([[9, 9, 9]])
            yield SimpleNamespace(text="ok", token=7, finish_reason="stop")

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = model
            engine._loaded = True
            engine._supports_system_kv_cache = True

            with (
                patch("mlx_lm.stream_generate", side_effect=fake_stream_generate),
                patch(
                    "mlx_lm.models.cache.make_prompt_cache",
                    return_value=[cache_entry],
                ),
                patch("mlx_lm.sample_utils.make_sampler"),
            ):
                chunks = [
                    c
                    async for c in engine.stream_chat(
                        messages=[
                            {"role": "system", "content": "You are helpful."},
                            {"role": "user", "content": "hello"},
                        ],
                    )
                ]

        assert chunks[-1].text == "ok"
        assert engine._system_kv_cache
        saved_snapshot, saved_token_count = next(iter(engine._system_kv_cache.values()))
        assert saved_token_count == 5
        saved = saved_snapshot[0][0]
        assert saved.tolist() == [[1, 2, 3]]
        assert cache_entry.cache[0].tolist() == [[9, 9, 9]]

    def test_system_cache_probe_allows_arrays_cache_and_rejects_rotating(self):
        """Hybrid ArraysCache is snapshot-safe; rotating cache still is not."""
        from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

        from vllm_mlx.engine.simple import SimpleEngine

        assert SimpleEngine._cache_class_is_system_snapshot_safe(KVCache())
        assert SimpleEngine._cache_class_is_system_snapshot_safe(ArraysCache(size=2))
        assert not SimpleEngine._cache_class_is_system_snapshot_safe(
            RotatingKVCache(max_size=128)
        )

    @pytest.mark.anyio
    async def test_stream_chat_uses_gate_time_snapshot_under_concurrent_mutation(
        self,
    ):
        """A concurrent MISS that reassigns ``self._system_kv_snapshot``
        between the cache-hit gate (which runs outside
        ``_run_blocking_serialized``) and the snapshot restore (which runs
        inside the serialized worker) must not corrupt the HIT.

        Fork semantics (patches #9/#12): the HIT path passes the snapshot to
        the worker as an explicit ``_run_blocking_serialized`` argument bound
        at gate time; the worker must restore from that reference, never
        re-read the active-slot ivar.

        Simulates the race by reassigning the active-slot snapshot inside the
        ``_run_blocking_serialized`` hook (executed after the gate but before
        the worker restores), then asserts the restore loop wrote the
        gate-time entries, not the post-gate intruder."""
        from vllm_mlx.engine.simple import SimpleEngine

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = (
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            "<|im_start|>user\nhello<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        tokenizer.bos_token = None
        tokenizer.encode = MagicMock(
            side_effect=[
                list(range(50)),  # full prompt
                list(range(20)),  # system prefix (proper prefix of above)
                list(range(40)),  # HIT grow probe == cached ids → no grow
            ]
        )
        tokenizer.eos_token_id = 99

        model = MagicMock()
        model.tokenizer = tokenizer
        model.model = MagicMock()

        original_snapshot = [("ORIGINAL_K", "ORIGINAL_V")]
        intruder_snapshot = [("INTRUDER_K", "INTRUDER_V")]

        captured_states: list = []

        class MockCacheEntry:
            def __init__(self) -> None:
                self._state = None

            @property
            def state(self):
                return self._state

            @state.setter
            def state(self, value) -> None:
                captured_states.append(value)
                self._state = value

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = model
            engine._loaded = True
            # Pre-seed an active-slot HIT: the cached extended prefix
            # (range(40)) is a proper prefix of the new request's full token
            # list (range(50)) — the fork's longest-prefix-match gate.
            engine._system_kv_snapshot = original_snapshot
            engine._system_kv_hash = "seeded-hash"
            engine._system_kv_token_count = 40
            engine._system_kv_token_ids = list(range(40))

            async def serialized_with_race(func, *args, on_cancel=None, **kw):
                # Simulate a concurrent MISS reassigning the active slot AFTER
                # the gate's HIT decision but BEFORE the worker restores. The
                # gate passed the original snapshot as an argument;
                # reassignment here must not affect that binding.
                engine._system_kv_snapshot = intruder_snapshot
                return await asyncio.to_thread(func, *args, **kw)

            engine._run_blocking_serialized = (
                serialized_with_race  # type: ignore[method-assign]
            )

            with (
                patch("mlx_lm.stream_generate", return_value=iter([])),
                patch(
                    "mlx_lm.models.cache.make_prompt_cache",
                    return_value=[MockCacheEntry()],
                ),
                patch("mlx_lm.sample_utils.make_sampler"),
                patch("mlx_lm.sample_utils.make_logits_processors", return_value=[]),
            ):
                _ = [
                    c
                    async for c in engine.stream_chat(
                        messages=[
                            {"role": "system", "content": "You are helpful."},
                            {"role": "user", "content": "hello"},
                        ],
                    )
                ]

        # The gate decided HIT…
        assert engine._system_kv_hits == 1
        # …and the restore wrote the gate-time snapshot exactly once.
        # If the worker had re-read ``self._system_kv_snapshot`` we would see
        # ``("INTRUDER_K", "INTRUDER_V")`` instead — that's the TOCTOU bug.
        assert captured_states == [("ORIGINAL_K", "ORIGINAL_V")], (
            "Snapshot restore did not use the gate-time reference; "
            f"captured={captured_states}"
        )

    @pytest.mark.anyio
    async def test_lock_serializes_stream_generate(self, mock_model):
        """Test that stream_generate uses the same lock as other methods."""
        from vllm_mlx.engine.simple import SimpleEngine

        def stream_generate_side_effect(**kwargs):
            # Yield a few chunks
            for i in range(3):
                chunk = MagicMock()
                chunk.text = f"chunk{i}"
                chunk.prompt_tokens = 5
                chunk.finished = i == 2
                chunk.finish_reason = "stop" if i == 2 else None
                yield chunk

        mock_model.stream_generate = MagicMock(side_effect=stream_generate_side_effect)

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = mock_model
            engine._loaded = True
            engine._generation_lock_admission = "wait"

            # Test that stream_generate acquires the lock
            # by checking if it blocks when lock is already held
            lock_acquired = asyncio.Event()
            stream_started = asyncio.Event()

            async def hold_lock():
                async with engine._generation_lock:
                    lock_acquired.set()
                    # Wait until stream tries to start
                    await asyncio.sleep(0.1)

            async def try_stream():
                # Wait for lock to be held
                await lock_acquired.wait()
                stream_started.set()
                # This should block until hold_lock releases
                result = []
                async for chunk in engine.stream_generate(prompt="test", max_tokens=10):
                    result.append(chunk)
                return result

            # Start both tasks
            hold_task = asyncio.create_task(hold_lock())
            stream_task = asyncio.create_task(try_stream())

            # Wait a bit for stream to try to acquire lock
            await asyncio.sleep(0.05)

            # Stream should have started but be blocked on the lock
            assert stream_started.is_set(), "Stream should have attempted to start"

            # Stream task should not be done yet (blocked on lock)
            assert not stream_task.done(), "Stream should be blocked waiting for lock"

            # Let hold_lock finish
            await hold_task

            # Now stream should complete
            result = await stream_task
            assert len(result) == 3, f"Expected 3 chunks, got {len(result)}"

    @pytest.mark.anyio
    async def test_engine_initialization_creates_lock(self):
        """Test that SimpleEngine creates a lock on initialization."""
        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")

            assert hasattr(engine, "_generation_lock")
            assert isinstance(engine._generation_lock, asyncio.Lock)

    @pytest.mark.anyio
    async def test_run_blocking_serialized_rebinds_worker_generation_streams(self):
        """Worker-thread MLX generation should get fresh thread-local streams."""
        import importlib

        from vllm_mlx.engine.simple import SimpleEngine

        mlx_lm_generate = importlib.import_module("mlx_lm.generate")
        sentinel_stream = object()

        with (
            patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False),
            patch("vllm_mlx.mlx_streams.mx.default_device", return_value="gpu"),
            patch(
                "vllm_mlx.mlx_streams.mx.new_stream",
                return_value=sentinel_stream,
            ),
            patch("vllm_mlx.mlx_streams.mx.set_default_stream"),
        ):
            engine = SimpleEngine("test-model")
            observed = await engine._run_blocking_serialized(
                lambda: mlx_lm_generate.generation_stream
            )

        assert observed is sentinel_stream

    @pytest.mark.anyio
    async def test_llm_stream_generate_stays_on_model_load_thread(self):
        """SimpleEngine must load and stream on the same thread for MLX streams."""
        from vllm_mlx.engine.simple import SimpleEngine

        class FakeLLMModel:
            def __init__(self, *_args, **_kwargs):
                self._load_thread = None
                self.tokenizer = MagicMock()
                self.tokenizer.encode.return_value = [1, 2, 3]

            def load(self):
                self._load_thread = threading.get_ident()

            def stream_generate(self, **_kwargs):
                if threading.get_ident() != self._load_thread:
                    raise RuntimeError("There is no Stream(gpu, 0) in current thread.")
                yield SimpleNamespace(
                    text="ok",
                    prompt_tokens=3,
                    finished=True,
                    finish_reason="stop",
                )

        with (
            patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False),
            patch("vllm_mlx.models.llm.MLXLanguageModel", FakeLLMModel),
        ):
            engine = SimpleEngine("test-model")

            outputs = [
                chunk
                async for chunk in engine.stream_generate(
                    prompt="hello",
                    max_tokens=1,
                    temperature=0.0,
                    top_p=1.0,
                )
            ]

        assert outputs
        assert outputs[-1].new_text == "ok"
        assert outputs[-1].finished is True

    @pytest.mark.anyio
    async def test_start_keeps_text_routing_for_mllm_without_mtp(self):
        """MLLM text-only routing must stay available when MTP is disabled."""
        from vllm_mlx.engine.simple import SimpleEngine

        event_loop_thread = threading.get_ident()
        captured = {}
        text_model = MagicMock()
        text_model.mtp = None
        tokenizer = MagicMock()
        tokenizer.convert_tokens_to_ids.return_value = 42

        mock_mllm = MagicMock()
        mock_mllm.model = MagicMock()
        mock_mllm.get_tokenizer.return_value = tokenizer

        def build_text_model(*_args, **kwargs):
            captured["build_thread"] = threading.get_ident()
            captured["enable_mtp"] = kwargs["enable_mtp"]
            return text_model

        with (
            patch(
                "vllm_mlx.models.mllm.MLXMultimodalLM",
                return_value=mock_mllm,
            ),
            patch(
                "vllm_mlx.text_model_from_vlm.build_text_model",
                side_effect=build_text_model,
            ),
        ):
            engine = SimpleEngine("qwen3.6-27b", force_mllm=True, mtp=False)
            try:
                await engine.start()
                worker_thread = await asyncio.get_running_loop().run_in_executor(
                    engine._generation_worker(), threading.get_ident
                )

                assert engine._text_model is text_model
                # The text-route tokenizer is wrapped in mlx_lm's
                # TokenizerWrapper so the model's FULL eos set
                # (generation_config.json) terminates generation (gemma-4
                # declares three eos ids; the bare HF tokenizer only carries
                # one). The wrapper must delegate to the original.
                from mlx_lm.tokenizer_utils import TokenizerWrapper

                wrapped = engine._text_tokenizer
                if isinstance(wrapped, TokenizerWrapper):
                    assert wrapped._tokenizer is tokenizer
                else:
                    assert wrapped is tokenizer
                assert captured["build_thread"] == worker_thread
                assert captured["build_thread"] != event_loop_thread
                assert captured["enable_mtp"] is False
            finally:
                await engine.stop()

        assert engine._text_model_initialization_attempted is False

    @pytest.mark.anyio
    async def test_mllm_media_stream_stays_on_owner_thread_with_text_route(self):
        """Media requests must not move mlx_vlm generation to a worker thread."""
        from vllm_mlx.engine.simple import SimpleEngine

        class FakeMllmModel:
            def __init__(self):
                self._owner_thread = threading.get_ident()

            def stream_chat(self, **_kwargs):
                if threading.get_ident() != self._owner_thread:
                    raise RuntimeError("There is no Stream(gpu, 3) in current thread.")
                yield SimpleNamespace(
                    text="image described",
                    finish_reason="stop",
                    prompt_tokens=5,
                )

        async def fail_if_called(*_args, **_kwargs):
            raise AssertionError("MLLM requests must not use worker routing")

        engine = SimpleEngine("test-model", force_mllm=True, mtp=False)
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._text_tokenizer = MagicMock()
        # lifecycle._prepare_engine_start runs prepare_for_start on the
        # generation worker, so the model's owner thread is that worker rather
        # than the event loop. Build the fake there to match.
        engine._model = engine._generation_worker().submit(FakeMllmModel).result()
        engine._run_blocking_serialized = fail_if_called  # type: ignore[method-assign]

        outputs = [
            chunk
            async for chunk in engine.stream_chat(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe this"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,AAAA"},
                            },
                        ],
                    }
                ],
                max_tokens=16,
            )
        ]

        assert outputs[-1].text == "image described"
        assert outputs[-1].finish_reason == "stop"

    @pytest.mark.anyio
    async def test_mllm_draft_stream_stays_on_owner_thread_with_text_route(self):
        """Text-only MLLM draft requests share mlx_vlm's owner thread."""
        from vllm_mlx.engine.simple import SimpleEngine

        captured = {}

        class FakeMllmModel:
            def __init__(self):
                self._owner_thread = threading.get_ident()

            def stream_chat(self, **kwargs):
                if threading.get_ident() != self._owner_thread:
                    raise RuntimeError("There is no Stream(gpu, 3) in current thread.")
                captured.update(kwargs)
                yield SimpleNamespace(
                    text="drafted",
                    finish_reason="stop",
                    prompt_tokens=3,
                )

        engine = SimpleEngine(
            "test-model",
            force_mllm=True,
            mllm_draft_model="assistant",
            mllm_draft_kind="mtp",
        )
        engine._loaded = True
        engine._text_model = MagicMock()
        # Same ownership rule as the media route: the model is built on the
        # generation worker, so that is the thread stream_chat must run on.
        engine._model = engine._generation_worker().submit(FakeMllmModel).result()

        outputs = [
            chunk
            async for chunk in engine.stream_chat(
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=8,
                mllm_draft=True,
            )
        ]

        assert captured["mllm_draft"] is True
        assert outputs[-1].text == "drafted"

    @pytest.mark.anyio
    async def test_mllm_media_stream_uses_fail_fast_admission(self):
        """A concurrent media request must receive EngineBusy instead of queueing."""
        from vllm_mlx.engine.base import EngineBusy
        from vllm_mlx.engine.simple import SimpleEngine

        class FakeMllmModel:
            def stream_chat(self, **_kwargs):
                yield SimpleNamespace(
                    text="first",
                    finish_reason=None,
                    prompt_tokens=5,
                )
                yield SimpleNamespace(
                    text=" second",
                    finish_reason="stop",
                    prompt_tokens=5,
                )

        def media_messages(label):
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": label},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,AAAA"},
                        },
                    ],
                }
            ]

        engine = SimpleEngine("test-model", force_mllm=True, mtp=False)
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._model = FakeMllmModel()

        first_started = asyncio.Event()
        release_first = asyncio.Event()

        async def consume_first():
            outputs = []
            async for output in engine.stream_chat(
                messages=media_messages("first"),
                request_id="first-media",
            ):
                outputs.append(output)
                if len(outputs) == 1:
                    first_started.set()
                    await release_first.wait()
            return outputs

        first_task = asyncio.create_task(consume_first())
        await asyncio.wait_for(first_started.wait(), timeout=1.0)

        try:
            with pytest.raises(EngineBusy) as excinfo:
                _ = [
                    output
                    async for output in engine.stream_chat(
                        messages=media_messages("second"),
                        request_id="second-media",
                    )
                ]

            assert excinfo.value.code == "text_generation_busy"
            assert "request_id=second-media" in str(excinfo.value)
            assert "active=none" not in str(excinfo.value)
            assert engine._generation_busy_rejections == 1
        finally:
            release_first.set()

        outputs = await first_task
        assert outputs[-1].finish_reason == "stop"

    @pytest.mark.anyio
    async def test_mllm_native_video_stream_stays_off_event_loop(self):
        """Native video generation must not block the asyncio event loop."""
        from vllm_mlx.engine.simple import SimpleEngine

        owner_thread = threading.get_ident()
        release = threading.Event()
        worker_thread = None

        class FakeMllmModel:
            _video_native = True

            def _collect_video_inputs(self, _messages):
                return {0: ["video.mp4"]}

            def stream_chat(self, **_kwargs):
                nonlocal worker_thread
                worker_thread = threading.get_ident()
                release.wait(timeout=0.2)
                yield SimpleNamespace(
                    text="video described",
                    finish_reason="stop",
                    prompt_tokens=5,
                )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": "video.mp4"}},
                    {"type": "text", "text": "describe this"},
                ],
            }
        ]
        engine = SimpleEngine("test-model", force_mllm=True, mtp=False)
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._model = FakeMllmModel()

        async def consume():
            return [
                output
                async for output in engine.stream_chat(
                    messages=messages,
                    request_id="native-video",
                )
            ]

        release_timer = threading.Timer(0.2, release.set)
        release_timer.start()
        started_at = asyncio.get_running_loop().time()
        stream_task = asyncio.create_task(consume())
        try:
            await asyncio.sleep(0.02)
            loop_delay = asyncio.get_running_loop().time() - started_at
        finally:
            release.set()
            release_timer.cancel()

        outputs = await stream_task
        assert loop_delay < 0.1
        assert worker_thread != owner_thread
        assert outputs[-1].text == "video described"

    @pytest.mark.anyio
    async def test_start_defers_text_model_when_mllm_draft_is_configured(
        self, monkeypatch
    ):
        """Draft-backed MLLM startup must not build an unused TextModel."""
        from vllm_mlx.engine.simple import SimpleEngine

        class FakeMllmModel:
            def __init__(self):
                self.model = object()
                self.loaded = False

            def load(self):
                self.loaded = True

        model = FakeMllmModel()

        monkeypatch.setattr(
            "vllm_mlx.models.mllm.MLXMultimodalLM", lambda *args, **kwargs: model
        )

        def unexpected_text_model_build(*args, **kwargs):
            raise AssertionError(
                "draft-backed startup must defer TextModel construction"
            )

        monkeypatch.setattr(
            "vllm_mlx.text_model_from_vlm.build_text_model",
            unexpected_text_model_build,
        )

        engine = SimpleEngine(
            "laguna-test",
            force_mllm=True,
            mllm_draft_model="/models/laguna-dflash",
            mllm_draft_kind="dflash",
            mllm_draft_block_size=8,
        )
        await engine.start()

        assert model.loaded is True
        assert engine._text_model is None

    @pytest.mark.anyio
    async def test_non_draft_request_lazily_initializes_mllm_text_model(self):
        """An explicit draft opt-out retains the existing text-only route."""
        from vllm_mlx.engine.simple import SimpleEngine

        engine = SimpleEngine(
            "laguna-test",
            force_mllm=True,
            mllm_draft_model="/models/laguna-dflash",
        )
        owner_thread = threading.get_ident()
        initialized: list[int] = []

        def initialize_text_model():
            initialized.append(threading.get_ident())

        engine._initialize_text_model = initialize_text_model  # type: ignore[method-assign]

        await engine._ensure_text_model_for_request(mllm_draft_requested=True)
        assert initialized == []

        await engine._ensure_text_model_for_request(mllm_draft_requested=False)
        assert len(initialized) == 1
        assert initialized[0] != owner_thread

        engine._text_model_initialization_attempted = True
        await engine._ensure_text_model_for_request(mllm_draft_requested=False)
        assert len(initialized) == 1

    @pytest.mark.anyio
    async def test_mllm_nonstream_text_only_routes_without_mtp(self):
        """Non-stream text-only MLLM chat must aggregate the TextModel route."""
        from vllm_mlx.engine.simple import SimpleEngine

        async def fake_stream_chat(*args, **kwargs):
            yield MagicMock(
                text="Hello",
                tokens=[1],
                prompt_tokens=5,
                completion_tokens=1,
                finish_reason="stop",
                finished=True,
            )

        engine = SimpleEngine("test-model", force_mllm=True, mtp=False)
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._model = MagicMock()
        engine.stream_chat = fake_stream_chat  # type: ignore[method-assign]

        output = await engine.chat(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=16,
        )

        assert output.text == "Hello"
        assert output.tokens == [1]
        assert output.prompt_tokens == 5
        assert output.completion_tokens == 1
        assert output.finish_reason == "stop"
        engine._model.chat.assert_not_called()

    @pytest.mark.anyio
    async def test_mllm_nonstream_text_only_without_text_model_uses_stream_path(self):
        """When TextModel is unavailable, text-only MLLM non-stream chat should
        aggregate stream_chat to avoid mlx_vlm chat thread-stream mismatches.
        """
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        class FakeMllmModel:
            def chat(self, **kwargs):
                raise RuntimeError("There is no Stream(gpu, 3) in current thread.")

            def stream_chat(self, **kwargs):
                yield SimpleNamespace(text="one, two, three", finish_reason="stop")

        engine = SimpleEngine("test-model", force_mllm=True, mtp=False)
        engine._loaded = True
        engine._text_model = None
        engine._model = FakeMllmModel()

        output = await engine.chat(
            messages=[{"role": "user", "content": "Count: one, two, three"}],
            max_tokens=16,
        )

        assert output.text == "one, two, three"
        assert output.finish_reason == "stop"

    @pytest.mark.anyio
    async def test_mllm_nonstream_text_only_without_text_model_keeps_stream_thread_owner(
        self,
    ):
        """MLLM text-only non-stream path must keep stream_chat on model thread.

        Regression: aggregate_stream_chat -> stream_chat moved mlx_vlm stream
        generation off the thread that owns the model and could raise
        "There is no Stream(gpu, N) in current thread".

        The owner is now the engine's pinned generation worker rather than the
        caller's thread, so the model is built there — which is what ``start()``
        does. The invariant under test is unchanged: whichever thread builds the
        model must be the one that generates from it.
        """
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        class FakeMllmModel:
            def __init__(self):
                self._owner_thread = threading.get_ident()

            def stream_chat(self, **kwargs):
                if threading.get_ident() != self._owner_thread:
                    raise RuntimeError("There is no Stream(gpu, 3) in current thread.")
                yield SimpleNamespace(
                    text="one, two, three",
                    finish_reason="stop",
                    prompt_tokens=3,
                )

        engine = SimpleEngine("test-model", force_mllm=True, mtp=False)
        engine._loaded = True
        engine._text_model = None
        loop = asyncio.get_running_loop()
        engine._model = await loop.run_in_executor(
            engine._generation_worker(), FakeMllmModel
        )

        output = await engine.chat(
            messages=[{"role": "user", "content": "Count: one, two, three"}],
            max_tokens=16,
        )

        assert output.text == "one, two, three"
        assert output.finish_reason == "stop"

    @pytest.mark.anyio
    async def test_llm_nonstream_with_logits_processors_uses_stream_path(self):
        """Constrained non-stream chat must not call the blocking chat API.

        ``response_format`` is implemented by request-local logits processors.
        If a non-stream request goes through the blocking model.chat() path,
        the server cannot observe token progress or cancel at token boundaries
        when a client/proxy disconnects.  Aggregating stream_chat keeps the
        constrained and unconstrained chat paths on the same cancellable stream
        implementation.
        """
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        captured_stream_kwargs = {}

        class FakeTokenizer:
            bos_token = None

            def apply_chat_template(self, messages, **kwargs):
                return "<|im_start|>user\nhello"

            def encode(self, text, **kwargs):
                return [1, 2, 3]

        class FakeModel:
            tokenizer = FakeTokenizer()

            def chat(self, **kwargs):
                raise AssertionError("blocking chat path should not be used")

            def stream_generate(self, **kwargs):
                captured_stream_kwargs.update(kwargs)
                yield SimpleNamespace(
                    text="{}",
                    finish_reason="stop",
                    finished=True,
                    prompt_tokens=3,
                )

        engine = SimpleEngine("test-model", force_mllm=False, mtp=False)
        engine._loaded = True
        engine._model = FakeModel()
        sentinel_processor = object()

        output = await engine.chat(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=16,
            logits_processors=[sentinel_processor],
        )

        assert output.text == "{}"
        assert output.finish_reason == "stop"
        assert captured_stream_kwargs["logits_processors"] == [sentinel_processor]

    @pytest.mark.anyio
    async def test_requests_complete_in_order(self, mock_model):
        """Test that concurrent requests complete (may be in any order due to lock).

        Uses "wait" admission explicitly. This test is about queued requests all
        completing, not about the admission policy — the default "fail_fast"
        rejects the second and third outright, which
        ``test_default_admission_rejects_second_serialized_request`` covers.
        Before generation was pinned to its own thread, the distinction was
        invisible here: generation blocked the event loop, so the later requests
        could not reach admission control until the first had already finished.
        """
        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = mock_model
            engine._loaded = True
            engine._generation_lock_admission = "wait"

            # Launch multiple concurrent generate calls
            results = await asyncio.gather(
                *[
                    engine.generate(prompt=f"test prompt {i}", max_tokens=10)
                    for i in range(3)
                ]
            )

            # All requests should complete
            assert len(results) == 3
            for result in results:
                assert result.text == "test response"

    @pytest.mark.anyio
    async def test_generate_accumulates_over_stream_generate(self):
        """generate() should iterate stream_generate() and return the last
        yielded GenerationOutput, forwarding per-request kwargs (including
        SpecPrefill overrides) through so they reach _stream_generate_specprefill.
        """
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.engine.simple import SimpleEngine

        captured_kwargs = {}

        async def fake_stream_generate(**kwargs):
            captured_kwargs.update(kwargs)
            # First chunk: mid-generation
            yield GenerationOutput(
                text="partial",
                new_text="partial",
                tokens=[1, 2],
                prompt_tokens=11,
                completion_tokens=2,
                finished=False,
                finish_reason=None,
            )
            # Final chunk: finished
            yield GenerationOutput(
                text="partial final",
                new_text=" final",
                tokens=[1, 2, 3],
                prompt_tokens=11,
                completion_tokens=3,
                finished=True,
                finish_reason="stop",
            )

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._loaded = True
            engine.stream_generate = fake_stream_generate  # type: ignore[method-assign]

            output = await engine.generate(
                prompt="say hi",
                max_tokens=16,
                temperature=0.6,
                top_p=0.95,
                specprefill=True,
                specprefill_keep_pct=0.2,
            )

        # Accumulator returns the last GenerationOutput's fields
        assert output.text == "partial final"
        assert output.tokens == [1, 2, 3]
        assert output.prompt_tokens == 11
        assert output.completion_tokens == 3
        assert output.finish_reason == "stop"
        assert output.finished is True

        # Per-request SpecPrefill overrides reach stream_generate
        assert captured_kwargs.get("prompt") == "say hi"
        assert captured_kwargs.get("max_tokens") == 16
        assert captured_kwargs.get("specprefill") is True
        assert captured_kwargs.get("specprefill_keep_pct") == 0.2

    @pytest.mark.anyio
    async def test_generate_empty_stream_returns_safe_default(self):
        """If stream_generate yields nothing, generate() returns an empty
        stop-reason GenerationOutput rather than raising.
        """
        from vllm_mlx.engine.simple import SimpleEngine

        async def empty_stream_generate(**kwargs):
            return
            yield  # unreachable; makes this a generator

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._loaded = True
            engine.stream_generate = empty_stream_generate  # type: ignore[method-assign]

            output = await engine.generate(prompt="anything", max_tokens=5)

        assert output.text == ""
        assert output.finish_reason == "stop"

    def test_seed_logits_processors_prepends_prompt_tokens(self):
        """Continuation decode processors must see the original prompt prefix."""
        from vllm_mlx.engine.simple import _seed_logits_processors

        seen = {}

        def processor(tokens, logits):
            seen["tokens"] = tokens.tolist()
            return logits

        seeded = _seed_logits_processors(
            mx.array([10, 11], dtype=mx.uint32), [processor]
        )

        logits = mx.zeros((1, 8), dtype=mx.float32)
        seeded[0](mx.array([12, 13], dtype=mx.uint32), logits)

        assert seen["tokens"] == [10, 11, 12, 13]

    @pytest.mark.anyio
    async def test_specprefill_success_preserves_mtp_path(self):
        """Successful sparse prefill should continue through the normal MTP path.

        Fork guard (patch #17): the content-phase resume forwards ``mtp=``
        only when ``inspect.signature(stream_generate)`` exposes an explicit
        ``mtp`` parameter (VAR_KEYWORD is not enough), so the fake must
        declare ``mtp`` in its signature for the kwarg to arrive.
        """
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        captured = {}

        def fake_make_sampler(**kwargs):
            captured["sampler_kwargs"] = kwargs

            def _sample(_logprobs):
                return mx.array([17], dtype=mx.uint32)

            return _sample

        def fake_stream_generate(model, tokenizer, prompt, *, mtp=None, **kwargs):
            captured["prompt"] = prompt.tolist()
            captured["kwargs"] = {**kwargs, "mtp": mtp}
            yield SimpleNamespace(text="B", finish_reason="stop")

        def fake_select_chunks(_importance, **kwargs):
            captured["select_chunks_kwargs"] = kwargs
            return mx.array([0, 1, 2], dtype=mx.int32)

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|im_start|>user\nhello"
        tokenizer.bos_token = None
        tokenizer.eos_token_id = 99
        tokenizer.encode.return_value = [5, 6, 7]
        tokenizer.decode.side_effect = lambda ids: "".join(
            {17: "A", 99: ""}.get(tok, f"<{tok}>") for tok in ids
        )

        text_model = MagicMock()
        text_model.mtp = object()
        text_model.make_mtp_cache.return_value = ["mtp-cache"]

        engine = SimpleEngine(
            "test-model",
            force_mllm=True,
            mtp=True,
            mtp_num_draft_tokens=4,
            specprefill_enabled=True,
            specprefill_threshold=1,
        )
        engine._loaded = True
        engine._text_model = text_model
        engine._text_tokenizer = tokenizer
        engine._draft_model = object()

        with (
            patch("vllm_mlx.engine.simple._bind_worker_generation_streams"),
            patch(
                "mlx_lm.models.cache.make_prompt_cache",
                return_value=["backbone-cache"],
            ),
            patch("mlx_lm.sample_utils.make_sampler", side_effect=fake_make_sampler),
            patch(
                "mlx_lm.sample_utils.make_logits_processors",
                return_value=[],
            ),
            # Replace (not side_effect-wrap) so inspect.signature() sees the
            # fake's explicit ``mtp`` parameter and the guard forwards it.
            patch("mlx_lm.stream_generate", fake_stream_generate),
            patch(
                "vllm_mlx.specprefill.score_tokens",
                return_value=mx.array([1.0, 0.9, 0.8], dtype=mx.float32),
            ),
            patch(
                "vllm_mlx.specprefill.select_chunks",
                side_effect=fake_select_chunks,
            ),
            patch(
                "vllm_mlx.specprefill.sparse_prefill",
                return_value=mx.zeros((1, 3, 32), dtype=mx.float32),
            ),
            patch("vllm_mlx.specprefill.cleanup_rope"),
        ):
            outputs = [
                chunk
                async for chunk in engine._stream_generate_text(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=4,
                    temperature=0.6,
                    top_p=0.95,
                    specprefill_backbone_pct=0.25,
                )
            ]

        assert [chunk.new_text for chunk in outputs] == ["A", "B"]
        assert captured["sampler_kwargs"] == {
            "temp": 0.6,
            "top_p": 0.95,
            "top_k": 0,
            "min_p": 0.0,
        }
        assert captured["prompt"] == [17]
        assert captured["kwargs"]["mtp"] is True
        assert captured["kwargs"]["prompt_cache"] == ["backbone-cache", "mtp-cache"]
        assert captured["kwargs"]["max_tokens"] == 3
        assert captured["kwargs"]["logits_processors"] is None
        assert captured["select_chunks_kwargs"]["backbone_pct"] == 0.25

    @pytest.mark.anyio
    async def test_specprefill_drops_mtp_when_stream_generate_lacks_param(self):
        """Patch #17 guard, degraded side: when the installed mlx_lm
        ``stream_generate`` does NOT expose an explicit ``mtp`` parameter
        (deployed mlx_lm 0.31.3 — VAR_KEYWORD doesn't count), the SpecPrefill
        content-phase resume must drop the kwarg and continue without native
        MTP instead of raising TypeError."""
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        captured = {}

        def fake_make_sampler(**kwargs):
            def _sample(_logprobs):
                return mx.array([17], dtype=mx.uint32)

            return _sample

        def fake_stream_generate(model, tokenizer, prompt, **kwargs):
            # No explicit ``mtp`` parameter — like mlx_lm 0.31.3. A forwarded
            # mtp= would land in **kwargs here; the guard must not send it.
            captured["kwargs"] = kwargs
            yield SimpleNamespace(text="B", finish_reason="stop")

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|im_start|>user\nhello"
        tokenizer.bos_token = None
        tokenizer.eos_token_id = 99
        tokenizer.encode.return_value = [5, 6, 7]
        tokenizer.decode.side_effect = lambda ids: "".join(
            {17: "A", 99: ""}.get(tok, f"<{tok}>") for tok in ids
        )

        text_model = MagicMock()
        text_model.mtp = object()
        text_model.make_mtp_cache.return_value = ["mtp-cache"]

        engine = SimpleEngine(
            "test-model",
            force_mllm=True,
            mtp=True,
            mtp_num_draft_tokens=4,
            specprefill_enabled=True,
            specprefill_threshold=1,
        )
        engine._loaded = True
        engine._text_model = text_model
        engine._text_tokenizer = tokenizer
        engine._draft_model = object()

        with (
            patch("vllm_mlx.engine.simple._bind_worker_generation_streams"),
            patch(
                "mlx_lm.models.cache.make_prompt_cache",
                return_value=["backbone-cache"],
            ),
            patch("mlx_lm.sample_utils.make_sampler", side_effect=fake_make_sampler),
            patch(
                "mlx_lm.sample_utils.make_logits_processors",
                return_value=[],
            ),
            patch("mlx_lm.stream_generate", fake_stream_generate),
            patch(
                "vllm_mlx.specprefill.score_tokens",
                return_value=mx.array([1.0, 0.9, 0.8], dtype=mx.float32),
            ),
            patch(
                "vllm_mlx.specprefill.select_chunks",
                return_value=mx.array([0, 1, 2], dtype=mx.int32),
            ),
            patch(
                "vllm_mlx.specprefill.sparse_prefill",
                return_value=mx.zeros((1, 3, 32), dtype=mx.float32),
            ),
            patch("vllm_mlx.specprefill.cleanup_rope"),
        ):
            outputs = [
                chunk
                async for chunk in engine._stream_generate_text(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=4,
                    temperature=0.6,
                    top_p=0.95,
                )
            ]

        # No TypeError, generation completed, and the kwarg was dropped.
        assert [chunk.new_text for chunk in outputs] == ["A", "B"]
        assert "mtp" not in captured["kwargs"]

    @pytest.mark.anyio
    async def test_stream_generate_text_forwards_logits_processors_and_sampler_args(
        self,
    ):
        """Text routing must preserve request-local decoding controls."""
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        captured_kwargs = {}
        sampler_calls = []
        penalty_calls = []
        user_processor = MagicMock()
        penalty_processor = MagicMock()

        def fake_stream_generate(model, tokenizer, prompt, **kwargs):
            captured_kwargs.update(kwargs)
            yield SimpleNamespace(text="Hello", finish_reason="stop")

        def fake_make_sampler(**kwargs):
            sampler_calls.append(kwargs)
            return MagicMock()

        def fake_make_logits_processors(**kwargs):
            penalty_calls.append(kwargs)
            return [penalty_processor]

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|im_start|>user\nhello"
        tokenizer.bos_token = None
        tokenizer.eos_token_id = 42

        engine = SimpleEngine("test-model", force_mllm=True, mtp=False)
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._text_model.mtp = None
        engine._text_tokenizer = tokenizer

        with (
            patch("mlx_lm.stream_generate", side_effect=fake_stream_generate),
            patch("mlx_lm.sample_utils.make_sampler", side_effect=fake_make_sampler),
            patch(
                "mlx_lm.sample_utils.make_logits_processors",
                side_effect=fake_make_logits_processors,
            ),
        ):
            outputs = [
                chunk
                async for chunk in engine._stream_generate_text(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=16,
                    temperature=0.3,
                    top_p=0.8,
                    top_k=40,
                    min_p=0.1,
                    presence_penalty=1.5,
                    repetition_penalty=1.2,
                    logits_processors=[user_processor],
                )
            ]

        assert outputs[-1].text == "Hello"
        assert sampler_calls == [{"temp": 0.3, "top_p": 0.8, "top_k": 40, "min_p": 0.1}]
        assert penalty_calls == [{"repetition_penalty": 1.2, "presence_penalty": 1.5}]
        assert captured_kwargs["logits_processors"] == [
            user_processor,
            penalty_processor,
        ]

    @pytest.mark.anyio
    async def test_stream_generate_text_skips_system_cache_when_text_model_not_safe(
        self,
    ):
        """MLLM TextModel routing must not snapshot hybrid ArraysCache state."""
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = (
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            "<|im_start|>user\nhello<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        tokenizer.bos_token = None
        tokenizer.eos_token_id = 42
        tokenizer.encode = MagicMock(
            side_effect=[
                list(range(10)),  # full prompt
                list(range(5)),  # system prefix
            ]
        )

        captured_prompts = []

        def fake_stream_generate(model, tokenizer, prompt, **kwargs):
            captured_prompts.append(prompt)
            yield SimpleNamespace(text="Hello", finish_reason="stop")

        def fail_if_manual_cache_path_runs(*args, **kwargs):
            raise AssertionError("manual system-cache path must be skipped")

        engine = SimpleEngine("test-model", force_mllm=True, mtp=False)
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._text_model.mtp = None
        engine._text_tokenizer = tokenizer
        engine._supports_system_kv_cache = False

        with (
            patch("mlx_lm.stream_generate", side_effect=fake_stream_generate),
            patch(
                "mlx_lm.models.cache.make_prompt_cache",
                side_effect=fail_if_manual_cache_path_runs,
            ),
        ):
            outputs = [
                chunk
                async for chunk in engine._stream_generate_text(
                    messages=[
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "hello"},
                    ],
                    max_tokens=16,
                    temperature=0.3,
                    top_p=0.8,
                )
            ]

        assert outputs[-1].text == "Hello"
        assert captured_prompts == [tokenizer.apply_chat_template.return_value]
        tokenizer.encode.assert_not_called()

    @pytest.mark.anyio
    async def test_stream_generate_text_normal_path_uses_generation_worker(self):
        """VLM-derived TextModel generation must run on the pinned worker."""
        from vllm_mlx.engine.simple import SimpleEngine

        event_loop_thread = threading.get_ident()
        generation_threads = []

        def fake_stream_generate(_model, _tokenizer, **_kwargs):
            generation_threads.append(threading.get_ident())
            yield SimpleNamespace(text="Hello", finish_reason="stop")

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|im_start|>user\nhello"
        tokenizer.bos_token = None
        tokenizer.eos_token_id = 42
        tokenizer.encode.return_value = [1, 2, 3]

        engine = SimpleEngine("test-model", force_mllm=True, mtp=False)
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._text_model.mtp = None
        engine._text_tokenizer = tokenizer
        worker_thread = await asyncio.get_running_loop().run_in_executor(
            engine._generation_worker(), threading.get_ident
        )

        try:
            with patch("mlx_lm.stream_generate", side_effect=fake_stream_generate):
                outputs = [
                    chunk
                    async for chunk in engine._stream_generate_text(
                        messages=[{"role": "user", "content": "hello"}],
                        max_tokens=16,
                        temperature=0.7,
                        top_p=0.9,
                    )
                ]
        finally:
            await engine.stop()

        assert outputs[-1].text == "Hello"
        assert generation_threads == [worker_thread]
        assert generation_threads[0] != event_loop_thread

    @pytest.mark.anyio
    async def test_stream_generate_text_disables_mtp_when_logits_processors_active(
        self,
    ):
        """Custom logits processors must fail closed to non-MTP decoding."""
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        captured_kwargs = {}
        user_processor = MagicMock()

        def fake_stream_generate(model, tokenizer, prompt, **kwargs):
            captured_kwargs.update(kwargs)
            yield SimpleNamespace(text="Hello", finish_reason="stop")

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|im_start|>user\nhello"
        tokenizer.bos_token = None
        tokenizer.eos_token_id = 42

        engine = SimpleEngine("test-model", force_mllm=True, mtp=True)
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._text_model.mtp = MagicMock()
        engine._text_tokenizer = tokenizer

        with patch("mlx_lm.stream_generate", side_effect=fake_stream_generate):
            outputs = [
                chunk
                async for chunk in engine._stream_generate_text(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=16,
                    temperature=0.7,
                    top_p=0.9,
                    logits_processors=[user_processor],
                )
            ]

        assert outputs[-1].text == "Hello"
        assert "mtp" not in captured_kwargs
        assert captured_kwargs["logits_processors"][0] is user_processor

    @pytest.mark.anyio
    async def test_stream_generate_text_disables_mtp_for_thinking_processor(
        self,
    ):
        """Thinking-budget processors must fail closed to non-MTP decoding."""
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        captured_kwargs = {}

        def fake_stream_generate(model, tokenizer, prompt, **kwargs):
            captured_kwargs.update(kwargs)
            yield SimpleNamespace(text="Hello", finish_reason="stop")

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|im_start|>user\nhello"
        tokenizer.bos_token = None
        tokenizer.eos_token_id = 42

        engine = SimpleEngine("test-model", force_mllm=True, mtp=True)
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._text_model.mtp = MagicMock()
        engine._text_tokenizer = tokenizer

        thinking_proc = MagicMock()

        with patch("mlx_lm.stream_generate", side_effect=fake_stream_generate):
            outputs = [
                chunk
                async for chunk in engine._stream_generate_text(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=16,
                    temperature=0.7,
                    top_p=0.9,
                    logits_processors=[thinking_proc],
                )
            ]

        assert outputs[-1].text == "Hello"
        assert "mtp" not in captured_kwargs
        assert captured_kwargs["logits_processors"][0] is thinking_proc

    @pytest.mark.anyio
    async def test_stream_generate_text_passes_num_draft_tokens(self):
        """Text routing should forward configured MTP draft depth."""
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        captured_kwargs = {}

        def fake_stream_generate(model, tokenizer, prompt, **kwargs):
            captured_kwargs.update(kwargs)
            yield SimpleNamespace(text="Hello", finish_reason="stop")

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|im_start|>user\nhello"
        tokenizer.bos_token = None
        tokenizer.eos_token_id = 42

        engine = SimpleEngine(
            "test-model",
            force_mllm=True,
            mtp=True,
            mtp_num_draft_tokens=4,
        )
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._text_model.mtp = MagicMock()
        engine._text_tokenizer = tokenizer

        with patch("mlx_lm.stream_generate", side_effect=fake_stream_generate):
            outputs = [
                chunk
                async for chunk in engine._stream_generate_text(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=16,
                    temperature=0.7,
                    top_p=0.9,
                )
            ]

        assert outputs[-1].text == "Hello"
        assert "mtp" not in captured_kwargs
        assert captured_kwargs["num_draft_tokens"] == 4

    @pytest.mark.anyio
    async def test_stream_generate_text_reenables_mtp_after_retired_processor_when_enabled(
        self,
    ):
        """Retired thinking processor handoff is an explicit opt-in path."""
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        calls = []

        class RetiringProcessor:
            def __init__(self):
                self.is_retired = False

            def __call__(self, tokens, logits):
                return logits

        processor = RetiringProcessor()

        def fake_stream_generate(model, tokenizer, prompt, **kwargs):
            calls.append({"prompt": prompt, **kwargs})
            if len(calls) == 1:
                processor.is_retired = True
                yield SimpleNamespace(token=11, text="Hello", finish_reason=None)
            else:
                yield SimpleNamespace(token=12, text=" world", finish_reason="stop")

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|im_start|>user\nhello"
        tokenizer.bos_token = None
        tokenizer.eos_token_id = 42
        tokenizer.encode.return_value = [11]

        engine = SimpleEngine(
            "test-model",
            force_mllm=True,
            mtp=True,
            mtp_num_draft_tokens=4,
        )
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._text_model.mtp = MagicMock()
        engine._text_model.make_mtp_cache.return_value = []
        engine._text_tokenizer = tokenizer

        with (
            patch.dict(
                "os.environ",
                {"VLLM_MLX_ENABLE_THINKING_RETIREMENT_RESUME": "1"},
            ),
            patch("mlx_lm.stream_generate", side_effect=fake_stream_generate),
            patch("mlx_lm.models.cache.make_prompt_cache", return_value=[]),
            patch("mlx_lm.models.cache.trim_prompt_cache"),
        ):
            outputs = [
                chunk
                async for chunk in engine._stream_generate_text(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=16,
                    temperature=0.7,
                    top_p=0.9,
                    logits_processors=[processor],
                )
            ]

        assert outputs[-1].text == "Hello world"
        assert len(calls) == 2
        assert "mtp" not in calls[0]
        assert calls[0]["logits_processors"][0] is processor
        assert "mtp" not in calls[1]
        assert calls[1]["num_draft_tokens"] == 4
        assert "prompt_cache" in calls[1]
        assert "logits_processors" not in calls[1]

    @pytest.mark.anyio
    async def test_stream_generate_text_specprefill_reenables_mtp_after_retirement(
        self,
    ):
        """SpecPrefill retirement-to-MTP continuation is explicit opt-in."""
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        calls = []

        class RetiringProcessor:
            def __init__(self):
                self.is_retired = False

            def __call__(self, tokens, logits):
                return logits

        processor = RetiringProcessor()

        def fake_stream_generate(model, tokenizer, prompt, **kwargs):
            calls.append({"prompt": prompt, **kwargs})
            yield SimpleNamespace(token=12, text=" world", finish_reason="stop")

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|im_start|>user\nhello"
        tokenizer.bos_token = None
        tokenizer.eos_token_id = 42
        tokenizer.encode.return_value = [1, 2, 3, 4]
        tokenizer.decode.side_effect = lambda toks: "Hello" if toks == [11] else ""

        engine = SimpleEngine(
            "test-model",
            force_mllm=True,
            mtp=True,
            mtp_num_draft_tokens=4,
            specprefill_enabled=True,
        )
        engine._loaded = True
        engine._draft_model = MagicMock()
        engine._text_model = MagicMock()
        engine._text_model.mtp = MagicMock()
        engine._text_model.make_mtp_cache.return_value = []
        engine._text_tokenizer = tokenizer

        def fake_sample(tokens, logits, sampler, logits_processors):
            processor.is_retired = True
            return mx.array(11, dtype=mx.uint32), logits

        with (
            patch.dict(
                "os.environ",
                {"VLLM_MLX_ENABLE_THINKING_RETIREMENT_RESUME": "1"},
            ),
            patch("mlx_lm.stream_generate", side_effect=fake_stream_generate),
            patch("mlx_lm.models.cache.make_prompt_cache", return_value=[]),
            patch(
                "vllm_mlx.specprefill.score_tokens", return_value=mx.array([0.1, 0.2])
            ),
            patch("vllm_mlx.specprefill.select_chunks", return_value=mx.array([0, 1])),
            patch(
                "vllm_mlx.specprefill.sparse_prefill",
                return_value=mx.zeros((1, 1, 32)),
            ),
            patch("vllm_mlx.specprefill.cleanup_rope"),
            patch(
                "vllm_mlx.engine.simple._sample_with_processors",
                side_effect=fake_sample,
            ),
        ):
            outputs = [
                chunk
                async for chunk in engine._stream_generate_text(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=16,
                    temperature=0.7,
                    top_p=0.9,
                    specprefill=True,
                    logits_processors=[processor],
                )
            ]

        assert outputs[-1].text == "Hello world"
        assert len(calls) == 1
        assert "mtp" not in calls[0]
        assert calls[0]["num_draft_tokens"] == 4
        assert "prompt_cache" in calls[0]
        assert "logits_processors" not in calls[0]

    @pytest.mark.anyio
    async def test_cancellation_does_not_release_lock_before_worker_finishes(
        self, mock_llm_model
    ):
        """A cancelled blocking chat call must not overlap the next worker.

        Media-shaped messages pin ``chat()`` on the legacy blocking
        ``self._model.chat`` path (the fork routes plain-text non-MLLM chat
        through ``_stream_generate_text``, which tokenizes for real and can't
        run on a MagicMock model); the property under test —
        ``_run_blocking_serialized`` keeping ``_generation_lock`` held until
        the worker thread actually finishes — is the same lock and runner
        every route uses.
        """
        from threading import Event, Lock

        from vllm_mlx.engine.simple import SimpleEngine

        first_started = Event()
        release_workers = Event()
        call_count = 0
        call_lock = Lock()

        def chat_side_effect(**kwargs):
            nonlocal call_count
            with call_lock:
                call_count += 1
                current_call = call_count
                mock_llm_model._concurrent_count += 1
                mock_llm_model._max_concurrent = max(
                    mock_llm_model._max_concurrent,
                    mock_llm_model._concurrent_count,
                )
                if current_call == 1:
                    first_started.set()

            try:
                release_workers.wait(timeout=1.0)
                result = MagicMock()
                result.text = f"response-{current_call}"
                result.tokens = [1, 2, 3]
                result.finish_reason = "stop"
                return result
            finally:
                with call_lock:
                    mock_llm_model._concurrent_count -= 1

        mock_llm_model.chat = MagicMock(side_effect=chat_side_effect)

        def media_messages(text: str) -> list[dict]:
            # Media part pins the legacy mlx blocking-chat path on a
            # non-MLLM engine (see docstring).
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "x.png"}},
                        {"type": "text", "text": text},
                    ],
                }
            ]

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = mock_llm_model
            engine._loaded = True
            engine._generation_lock_admission = "wait"
            # chat() recomputes prompt accounting via apply_chat_template
            # (tokenize=True) on the legacy path.
            mock_llm_model.tokenizer.apply_chat_template = MagicMock(
                return_value=[1, 2, 3]
            )

            task1 = asyncio.create_task(
                engine.chat(messages=media_messages("first"), max_tokens=8)
            )
            await asyncio.to_thread(first_started.wait, 1.0)

            task1.cancel()
            task2 = asyncio.create_task(
                engine.chat(messages=media_messages("second"), max_tokens=8)
            )

            await asyncio.sleep(0.05)
            release_workers.set()

            with pytest.raises(asyncio.CancelledError):
                await task1
            result2 = await task2

            assert result2.text == "response-2"
            assert mock_llm_model._max_concurrent == 1

    @pytest.mark.anyio
    async def test_specprefill_path_does_not_prelock_serialized_runner(self):
        """SpecPrefill should let _run_blocking_serialized own the lock."""
        from vllm_mlx.engine.simple import SimpleEngine

        async def fake_serialized(func, *args, **kwargs):
            assert not engine._generation_lock.locked()
            return []

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._loaded = True
            engine._model = MagicMock()
            engine._model.model = MagicMock()
            engine._model.tokenizer = MagicMock()
            engine._draft_model = MagicMock()
            engine._run_blocking_serialized = fake_serialized  # type: ignore[method-assign]

            outputs = []
            async for chunk in engine._stream_generate_specprefill(
                prompt="hello",
                tokens=[1, 2, 3, 4],
                max_tokens=4,
                temperature=0.7,
                top_p=0.9,
            ):
                outputs.append(chunk)

            assert len(outputs) == 1
            assert outputs[0].finished
            assert outputs[0].completion_tokens == 0

    @pytest.mark.anyio
    async def test_text_mtp_path_does_not_prelock_serialized_runner(self):
        """Text-only MTP path should let _run_blocking_serialized own the lock."""
        from vllm_mlx.engine.simple import SimpleEngine

        async def fake_serialized(func, *args, **kwargs):
            assert not engine._generation_lock.locked()
            return []

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=True):
            engine = SimpleEngine("test-model")
            engine._loaded = True
            engine._text_model = MagicMock()
            engine._text_model.make_mtp_cache = MagicMock(return_value=[])
            engine._text_tokenizer = MagicMock()
            engine._text_tokenizer.apply_chat_template = MagicMock(return_value="hello")
            engine._text_tokenizer.bos_token = None
            engine._draft_model = None
            engine._run_blocking_serialized = fake_serialized  # type: ignore[method-assign]

            outputs = []
            async for chunk in engine._stream_generate_text(
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=4,
                temperature=0.7,
                top_p=0.9,
            ):
                outputs.append(chunk)

            assert len(outputs) == 1
            assert outputs[0].finished
            assert outputs[0].completion_tokens == 0

    @pytest.mark.anyio
    async def test_specprefill_threads_same_cancel_check_to_helpers(self):
        """SpecPrefill worker should pass one cooperative cancel hook through both phases."""
        from vllm_mlx.engine.simple import SimpleEngine

        captured = {}

        def fake_score_tokens(*args, cancel_check=None, **kwargs):
            captured["score"] = cancel_check
            return mx.array([0.5], dtype=mx.float32)

        def fake_sparse_prefill(*args, cancel_check=None, **kwargs):
            captured["prefill"] = cancel_check
            return mx.zeros((1, 1, 8), dtype=mx.float32)

        def fake_select_chunks(_importance, **kwargs):
            captured["select_chunks_kwargs"] = kwargs
            return mx.array([0], dtype=mx.int32)

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._loaded = True
            engine._draft_model = MagicMock()
            engine._model = MagicMock()
            engine._model.model = MagicMock()
            engine._model.tokenizer = MagicMock()
            engine._model.tokenizer.decode = MagicMock(return_value="A")
            engine._model.tokenizer.eos_token_id = 0

            outputs = []
            with (
                patch("mlx_lm.models.cache.make_prompt_cache", return_value=[]),
                patch(
                    "mlx_lm.sample_utils.make_sampler",
                    return_value=lambda logits: mx.array([0], dtype=mx.int32),
                ),
                patch(
                    "vllm_mlx.specprefill.score_tokens", side_effect=fake_score_tokens
                ),
                patch(
                    "vllm_mlx.specprefill.select_chunks",
                    side_effect=fake_select_chunks,
                ),
                patch(
                    "vllm_mlx.specprefill.sparse_prefill",
                    side_effect=fake_sparse_prefill,
                ),
                patch("vllm_mlx.specprefill.cleanup_rope"),
            ):
                async for chunk in engine._stream_generate_specprefill(
                    prompt="hello",
                    tokens=[1, 2, 3, 4],
                    max_tokens=4,
                    temperature=0.7,
                    top_p=0.9,
                    specprefill_backbone_pct=0.25,
                ):
                    outputs.append(chunk.new_text)

        assert outputs == ["A"]
        assert callable(captured["score"])
        assert captured["score"] is captured["prefill"]
        assert captured["select_chunks_kwargs"]["backbone_pct"] == 0.25

    @pytest.mark.anyio
    async def test_cancelling_specprefill_request_stops_during_scoring(self):
        """Cancelling SpecPrefill should signal the blocking scorer and exit without output."""
        import time
        from threading import Event

        from vllm_mlx.engine.simple import SimpleEngine, _SpecPrefillCancelled

        score_started = Event()
        score_cancelled = Event()

        def fake_score_tokens(*args, cancel_check=None, **kwargs):
            assert callable(cancel_check)
            score_started.set()
            while True:
                try:
                    cancel_check()
                except _SpecPrefillCancelled:
                    score_cancelled.set()
                    raise
                time.sleep(0.01)

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._loaded = True
            engine._draft_model = MagicMock()
            engine._model = MagicMock()
            engine._model.model = MagicMock()
            engine._model.tokenizer = MagicMock()

            async def consume():
                async for _chunk in engine._stream_generate_specprefill(
                    prompt="hello",
                    tokens=[1, 2, 3, 4],
                    max_tokens=4,
                    temperature=0.7,
                    top_p=0.9,
                ):
                    pytest.fail("Cancelled SpecPrefill request should not emit output")

            with (
                patch("mlx_lm.models.cache.make_prompt_cache", return_value=[]),
                patch(
                    "vllm_mlx.specprefill.score_tokens",
                    side_effect=fake_score_tokens,
                ),
                patch("vllm_mlx.specprefill.cleanup_rope"),
            ):
                task = asyncio.create_task(consume())
                assert await asyncio.to_thread(score_started.wait, 1.0)
                task.cancel()

                with pytest.raises(asyncio.CancelledError):
                    await task

        assert await asyncio.to_thread(score_cancelled.wait, 1.0)

    @pytest.mark.anyio
    async def test_cancelling_specprefill_request_stops_during_sparse_prefill(self):
        """Cancelling SpecPrefill should signal the sparse-prefill loop and exit without output."""
        import time
        from threading import Event

        from vllm_mlx.engine.simple import SimpleEngine, _SpecPrefillCancelled

        prefill_started = Event()
        prefill_cancelled = Event()

        def fake_sparse_prefill(*args, cancel_check=None, **kwargs):
            assert callable(cancel_check)
            prefill_started.set()
            while True:
                try:
                    cancel_check()
                except _SpecPrefillCancelled:
                    prefill_cancelled.set()
                    raise
                time.sleep(0.01)

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._loaded = True
            engine._draft_model = MagicMock()
            engine._model = MagicMock()
            engine._model.model = MagicMock()
            engine._model.tokenizer = MagicMock()

            async def consume():
                async for _chunk in engine._stream_generate_specprefill(
                    prompt="hello",
                    tokens=[1, 2, 3, 4],
                    max_tokens=4,
                    temperature=0.7,
                    top_p=0.9,
                ):
                    pytest.fail("Cancelled SpecPrefill request should not emit output")

            with (
                patch("mlx_lm.models.cache.make_prompt_cache", return_value=[]),
                patch(
                    "vllm_mlx.specprefill.score_tokens",
                    return_value=mx.array([0.5], dtype=mx.float32),
                ),
                patch(
                    "vllm_mlx.specprefill.select_chunks",
                    return_value=mx.array([0], dtype=mx.int32),
                ),
                patch(
                    "vllm_mlx.specprefill.sparse_prefill",
                    side_effect=fake_sparse_prefill,
                ),
                patch("vllm_mlx.specprefill.cleanup_rope"),
            ):
                task = asyncio.create_task(consume())
                assert await asyncio.to_thread(prefill_started.wait, 1.0)
                task.cancel()

                with pytest.raises(asyncio.CancelledError):
                    await task
        assert await asyncio.to_thread(prefill_cancelled.wait, 1.0)


class TestSimpleEngineNaturalStop:
    @staticmethod
    def _engine(stream_generate):
        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
        engine._loaded = True
        engine._model = MagicMock()
        engine._model.tokenizer.encode.return_value = [1, 2, 3]
        engine._model.stream_generate.side_effect = stream_generate
        return engine

    @pytest.mark.anyio
    async def test_generator_exhaustion_emits_one_stop(self):
        def generate(**kwargs):
            yield SimpleNamespace(text="Hel", prompt_tokens=3, finish_reason=None)
            yield SimpleNamespace(text="lo", prompt_tokens=3, finish_reason="stop")

        outputs = [
            output
            async for output in self._engine(generate).stream_generate(
                prompt="hi", max_tokens=50
            )
        ]

        finished = [output for output in outputs if output.finished]
        assert len(finished) == 1
        assert finished[0] is outputs[-1]
        assert finished[0].new_text == ""
        assert finished[0].finish_reason == "stop"

    @pytest.mark.anyio
    async def test_token_limit_keeps_length_reason(self):
        def generate(**kwargs):
            yield SimpleNamespace(text="a", prompt_tokens=3, finish_reason=None)
            yield SimpleNamespace(text="b", prompt_tokens=3, finish_reason=None)
            yield SimpleNamespace(text="c", prompt_tokens=3, finish_reason=None)

        outputs = [
            output
            async for output in self._engine(generate).stream_generate(
                prompt="hi", max_tokens=3
            )
        ]

        finished = [output for output in outputs if output.finished]
        assert len(finished) == 1
        assert finished[0] is outputs[-1]
        assert finished[0].finish_reason == "length"

    @pytest.mark.anyio
    async def test_exception_is_not_reported_as_stop(self):
        def generate(**kwargs):
            yield SimpleNamespace(text="partial", prompt_tokens=3, finish_reason=None)
            raise RuntimeError("backend failed")

        engine = self._engine(generate)
        outputs = []
        with pytest.raises(RuntimeError, match="backend failed"):
            async for output in engine.stream_generate(prompt="hi", max_tokens=50):
                outputs.append(output)

        assert not any(output.finished for output in outputs)
        assert not engine._active_requests

    @pytest.mark.anyio
    async def test_empty_generator_emits_stop(self):
        def generate(**kwargs):
            yield from ()

        outputs = [
            output
            async for output in self._engine(generate).stream_generate(
                prompt="hi", max_tokens=50
            )
        ]

        assert len(outputs) == 1
        assert outputs[0].finished
        assert outputs[0].finish_reason == "stop"


class TestSimpleEngineStreamClose:
    @staticmethod
    def _engine(generate):
        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
        engine._loaded = True
        engine._model = MagicMock()
        engine._model.stream_generate.side_effect = generate
        engine._model.tokenizer.apply_chat_template.return_value = "prompt"
        return engine

    @staticmethod
    async def _close_and_assert_clean(engine, stream, backend_closed, loop_errors):
        from vllm_mlx.engine.simple import _in_tracker

        first = await anext(stream)
        await stream.aclose()
        await asyncio.sleep(0)

        assert not first.finished
        assert backend_closed()
        assert not engine._active_requests
        assert engine._num_running == 0
        assert not engine._generation_lock.locked()
        assert not _in_tracker.get()
        assert not loop_errors

    @pytest.mark.anyio
    async def test_public_stream_generate_close_cleans_inner_state(self):
        backend_closed = False

        def generate(**kwargs):
            nonlocal backend_closed
            try:
                yield SimpleNamespace(
                    text="partial", prompt_tokens=3, finish_reason=None
                )
                yield SimpleNamespace(text="ignored", prompt_tokens=3)
            finally:
                backend_closed = True

        engine = self._engine(generate)

        loop_errors = []
        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
        try:
            stream = engine.stream_generate(prompt="hi", max_tokens=50)
            await self._close_and_assert_clean(
                engine, stream, lambda: backend_closed, loop_errors
            )
        finally:
            loop.set_exception_handler(previous_handler)

    @pytest.mark.anyio
    async def test_public_stream_chat_close_cleans_nested_generate(self):
        backend_closed = False

        def generate(**kwargs):
            nonlocal backend_closed
            try:
                yield SimpleNamespace(
                    text="partial", prompt_tokens=3, finish_reason=None
                )
                yield SimpleNamespace(text="ignored", prompt_tokens=3)
            finally:
                backend_closed = True

        engine = self._engine(generate)
        loop_errors = []
        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
        try:
            stream = engine.stream_chat(
                messages=[{"role": "user", "content": "hi"}], max_tokens=50
            )
            await self._close_and_assert_clean(
                engine, stream, lambda: backend_closed, loop_errors
            )
        finally:
            loop.set_exception_handler(previous_handler)

    @pytest.mark.anyio
    async def test_mllm_text_route_close_reaches_inner_cleanup(self):
        from vllm_mlx.engine.base import GenerationOutput

        inner_closed = False

        async def text_stream(*args, **kwargs):
            nonlocal inner_closed
            try:
                yield GenerationOutput(
                    text="partial",
                    new_text="partial",
                    prompt_tokens=3,
                    completion_tokens=1,
                    finished=False,
                )
                yield GenerationOutput(text="ignored")
            finally:
                inner_closed = True

        engine = self._engine(lambda **kwargs: iter(()))
        engine._is_mllm = True
        engine._text_model = MagicMock()
        engine._stream_generate_text = text_stream

        stream = engine.stream_chat(
            messages=[{"role": "user", "content": "hi"}], max_tokens=50
        )
        first = await anext(stream)
        await stream.aclose()

        assert not first.finished
        assert inner_closed
        assert not engine._active_requests
        assert engine._num_running == 0


class TestSimpleEngineClearRuntimeCaches:
    """Releasing the multi-slot system-prompt KV cache state (patches
    #9/#12/#13: active-slot ivars + ``_system_kv_lru`` bag).

    Fork semantics differ from upstream #523/#541: ``clear_runtime_caches``
    is the MLLM model-cache hook only (returns None for non-MLLM engines);
    the system-KV snapshot stack — multi-GB Metal-heap state — is released
    by ``stop()``.
    """

    @pytest.fixture
    def anyio_backend(self):
        return "asyncio"

    def _seed_system_kv_state(self, engine):
        # Active slot + LRU bag + non-zero counters, as if the engine had
        # been serving traffic with three distinct system prefixes.
        engine._system_kv_snapshot = [(b"active_k", b"active_v")]
        engine._system_kv_hash = "hash_active"
        engine._system_kv_token_count = 28000
        engine._system_kv_token_ids = [1, 2, 3]
        engine._system_kv_lru["hash_a"] = {
            "snapshot": [(b"snap_a_k", b"snap_a_v")],
            "token_count": 6500,
            "token_ids": [4, 5, 6],
        }
        engine._system_kv_lru["hash_b"] = {
            "snapshot": [(b"snap_b_k", b"snap_b_v")],
            "token_count": 900,
            "token_ids": [7, 8, 9],
        }
        engine._system_kv_hits = 5
        engine._system_kv_misses = 2
        engine._system_kv_tokens_saved = 140000
        engine._system_kv_evictions = 1
        engine._supports_system_kv_cache = True

    @pytest.mark.anyio
    async def test_stop_drops_lru_and_resets_counters(self):
        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
        engine._loaded = True
        self._seed_system_kv_state(engine)

        # Fork divergence: clear_runtime_caches has no system-KV semantics
        # for non-MLLM engines — it must not touch the snapshot stack.
        assert engine.clear_runtime_caches() is None
        assert len(engine._system_kv_lru) == 2
        assert engine._system_kv_snapshot is not None

        # stop() is the operational release path for the snapshot stack.
        await engine.stop()

        assert engine._system_kv_snapshot is None
        assert engine._system_kv_hash is None
        assert engine._system_kv_token_count == 0
        assert engine._system_kv_token_ids is None
        assert len(engine._system_kv_lru) == 0, "LRU bag must be empty after stop"
        assert engine._system_kv_hits == 0
        assert engine._system_kv_misses == 0
        assert engine._system_kv_tokens_saved == 0
        assert engine._system_kv_evictions == 0
        assert engine._supports_system_kv_cache is False

    def test_clear_runtime_caches_no_op_for_non_mllm(self):
        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")

        result = engine.clear_runtime_caches()

        # Non-MLLM, empty LRU, zeroed counters → nothing to report.
        assert result is None


class TestSimpleEngineStop:
    """stop() must actually release MLX's Metal buffer cache, not just drop
    Python references — otherwise idle-unload frees objects but not memory.
    """

    async def test_stop_calls_mx_clear_cache(self, monkeypatch):
        from vllm_mlx.engine import simple as simple_mod
        from vllm_mlx.engine.simple import SimpleEngine

        calls = {"count": 0}
        monkeypatch.setattr(
            simple_mod.mx,
            "clear_cache",
            lambda: calls.__setitem__("count", calls["count"] + 1),
        )

        engine = SimpleEngine("test-model")
        engine._model = object()
        engine._loaded = True

        await engine.stop()

        assert calls["count"] == 1
        assert engine._model is None
        assert engine._loaded is False
