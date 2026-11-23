"""#105: the MLLM first-token path must not keep a prefill's full logits.

``logits[:, -1, :]`` is a view of the whole ``(B, T, vocab)`` forward output.
Held through cache admission and the next row's prefill it pins ~1 GB per
2K-token chunk on a 248K vocab (the mlx-vlm #2806 pattern).
"""

import gc

import mlx.core as mx

from vllm_mlx.mllm_batch_generator import _detached_last_row

MB = 2**20


def _held_after_dropping_logits(take_row):
    logits = mx.random.normal((1, 256, 65536))  # 64 MiB
    mx.eval(logits)
    mx.clear_cache()
    before = mx.get_active_memory()
    row = take_row(logits)
    mx.eval(row)
    logits = None
    gc.collect()
    mx.clear_cache()
    return (mx.get_active_memory() - before + 64 * MB) / MB, row


def test_105_a_plain_last_row_view_pins_the_whole_forward_output():
    held, _row = _held_after_dropping_logits(lambda x: x[:, -1, :])
    assert held > 60  # the control: this is the hold


def test_105_detached_last_row_lets_the_forward_output_go():
    held, row = _held_after_dropping_logits(_detached_last_row)
    assert held < 1
    assert row.shape == (1, 65536)


def test_105_detached_last_row_values_are_the_last_position():
    x = mx.random.normal((2, 5, 11))
    assert mx.array_equal(_detached_last_row(x), x[:, -1, :]).item()
