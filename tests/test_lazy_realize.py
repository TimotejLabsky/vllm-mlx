"""Tests for vllm_mlx/lazy_realize.py and its BatchedEngine wiring (PATCHES.md #29).

The batched LLM path loads the model on the event-loop thread but steps it on
the engine-core executor thread. A lazy init-time array (gpt-oss's attention
`sinks`) recorded at load and first evaluated on the worker dies with
"There is no Stream(gpu, N) in current thread" — engine_core then burns a
failed step + self-heal cycle. The realize at load time prevents that.
"""

import threading

import mlx.core as mx
import mlx.nn as nn

from vllm_mlx.lazy_realize import realize_module_arrays


class _LazyBufferModule(nn.Module):
    def __init__(self):
        super().__init__()
        # Lazy graph over a private buffer — excluded from parameters(),
        # like rope_utils' scaled-RoPE _freqs or gpt-oss's sinks.
        self._sinks = mx.exp(mx.arange(0, 8, dtype=mx.float32) * -0.5)


class _HostModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _LazyBufferModule()


def _assert_cross_thread_evaluable(arr):
    errors = []

    def cross_thread_eval():
        try:
            mx.eval(arr)
        except RuntimeError as e:  # pragma: no cover - the regression itself
            errors.append(e)

    t = threading.Thread(target=cross_thread_eval)
    t.start()
    t.join()
    assert not errors, f"lazy array not realized on the load thread: {errors[0]}"


def test_realizes_private_lazy_arrays():
    model = _HostModel()
    realize_module_arrays(model)
    _assert_cross_thread_evaluable(model.attn._sinks)


def test_object_without_modules_is_noop():
    realize_module_arrays(object())  # must not raise


def test_batched_prepare_llm_model_realizes_on_load_thread(monkeypatch):
    """_prepare_llm_model must realize module arrays right after load."""
    from vllm_mlx.engine.batched import BatchedEngine
    from vllm_mlx.utils import tokenizer as tokenizer_mod

    model = _HostModel()

    monkeypatch.setattr(
        tokenizer_mod,
        "load_model_with_fallback",
        lambda name, tokenizer_config=None: (model, object()),
    )

    engine = BatchedEngine.__new__(BatchedEngine)
    engine._model = None
    engine._tokenizer = None
    engine._model_name = "fake-model"
    engine._trust_remote_code = False
    engine._scheduler_config = None
    engine._configure_metal_memory_limits = lambda: None

    engine._prepare_llm_model()

    assert engine._model is model
    _assert_cross_thread_evaluable(model.attn._sinks)
