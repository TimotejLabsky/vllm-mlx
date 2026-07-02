"""Realize lazy init-time MLX arrays on the calling (load) thread.

MLX lazy graphs are tagged to the stream of the thread that recorded them.
A model loaded on one thread but first stepped on another (SimpleEngine's
serialized worker, BatchedEngine's engine-core executor) crashes with
"There is no Stream(gpu, N) in current thread" if any init-time array is
still lazy at that first forward — gpt-oss's attention ``sinks =
mx.zeros((num_heads,))`` is the canonical case (PATCHES.md #28); gemma-4's
RoPE tables hit the same class on the VLM->text route (#21 / upstream #614).

Call this immediately after weight loading, on the thread that loaded the
model. Harmless no-op when every array is already realized.
"""

import mlx.core as mx


def realize_module_arrays(model) -> None:
    """mx.eval every mx.array reachable from the model's module tree.

    Walks ``modules()`` and evals raw ``values()`` — unlike
    ``parameters()``, this covers underscore-private buffers (e.g.
    ``RoPE._freqs``) that never appear in the parameter tree.
    """
    if not hasattr(model, "modules"):
        return
    mx.eval(
        [
            v
            for module in model.modules()
            for v in module.values()
            if isinstance(v, mx.array)
        ]
    )
