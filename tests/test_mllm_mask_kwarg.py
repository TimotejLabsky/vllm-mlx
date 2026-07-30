# SPDX-License-Identifier: Apache-2.0
"""Required-``mask`` arch quirk on the batched MLLM vision-encode call.

mlx-vlm's mistral3 ``Model.__call__`` declares ``mask`` as a required
positional (and never uses it in the body), while every other arch makes
it ``Optional[...] = None``. The fork's ``_run_vision_encoding`` calls
``self.model(input_ids, cache=cache, **kwargs)`` — without a mask that
raised ``TypeError: Model.__call__() missing 1 required positional
argument: 'mask'`` before the first vision token, failing every image
request on mistral3-family routes (Devstral, Mistral-Small).

Fix: detect ``mask`` in the model's call signature once at init and pass
``mask=None`` only when the parameter exists — arches without it never
see the kwarg.
"""

import inspect
from types import SimpleNamespace

import mlx.core as mx
import pytest

from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator, MLLMBatchRequest

pytestmark = pytest.mark.skipif(mx is None, reason="MLX not available")


class _RequiredMaskModel:
    """Mirrors mlx-vlm mistral3: mask required positional, unused."""

    def __init__(self):
        self.calls = []
        self.config = SimpleNamespace()

    def __call__(self, input_ids, pixel_values, mask, cache=None, **kwargs):
        self.calls.append({"mask": mask, "kwargs": dict(kwargs)})
        return mx.zeros((1, 1, 8))


class _NoMaskModel:
    """Typical arch shape without any mask parameter."""

    def __init__(self):
        self.calls = []
        self.config = SimpleNamespace()

    def __call__(self, input_ids, pixel_values=None, cache=None, **kwargs):
        self.calls.append({"kwargs": dict(kwargs)})
        return mx.zeros((1, 1, 8))


def _bare_generator(model) -> MLLMBatchGenerator:
    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen.model = model
    gen._model_call_takes_mask = "mask" in inspect.signature(
        type(model).__call__
    ).parameters
    gen.maybe_relieve_pressure = lambda: 0
    return gen


def _request() -> MLLMBatchRequest:
    req = MLLMBatchRequest(uid=1, request_id="req-1", prompt="hi")
    req.input_ids = mx.array([[1, 2, 3, 4]])
    req.pixel_values = mx.zeros((1, 3, 4, 4))
    return req


class TestMaskKwarg:
    def test_required_mask_arch_gets_mask_none(self):
        model = _RequiredMaskModel()
        gen = _bare_generator(model)
        out = gen._run_vision_encoding(_request())
        assert out is not None
        assert len(model.calls) == 1
        assert model.calls[0]["mask"] is None

    def test_no_mask_arch_never_sees_kwarg(self):
        model = _NoMaskModel()
        gen = _bare_generator(model)
        gen._run_vision_encoding(_request())
        assert len(model.calls) == 1
        assert "mask" not in model.calls[0]["kwargs"]

    def test_explicit_mask_in_extra_kwargs_not_clobbered(self):
        model = _RequiredMaskModel()
        gen = _bare_generator(model)
        req = _request()
        sentinel = object()
        req.extra_kwargs["mask"] = sentinel
        gen._run_vision_encoding(req)
        assert model.calls[0]["mask"] is sentinel

    def test_init_detects_mask_parameter(self):
        assert (
            "mask"
            in inspect.signature(_RequiredMaskModel.__call__).parameters
        )
        gen_with = _bare_generator(_RequiredMaskModel())
        gen_without = _bare_generator(_NoMaskModel())
        assert gen_with._model_call_takes_mask is True
        assert gen_without._model_call_takes_mask is False
