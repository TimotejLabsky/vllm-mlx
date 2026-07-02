"""Reusable cross-thread MLX stream harness — the engine_core executor shape.

Five fork bugs to date share one mechanism (#21, #28, #29, and two live-only
finds in #34): an MLX lazy graph recorded on one thread is first evaluated on
another, and dies with "There is no Stream(gpu, N) in current thread". The
dangerous shape is specifically a thread whose default stream was REBOUND to
a fresh stream (``bind_generation_streams`` does this for engine_core's
executor and SimpleEngine's serialized worker) — graphs recorded there are
tagged to a stream no other thread has.

Single-threaded tests are structurally blind to this class. Any test that
moves cache state between "scheduler executor" and "event loop" roles should
run the executor side through :class:`StreamBoundWorker`.
"""

import queue
import threading

import mlx.core as mx


class StreamBoundWorker:
    """A dedicated thread whose MLX default stream is a fresh non-default
    stream, exactly like ``bind_generation_streams()`` gives the engine-core
    executor. Use ``run(fn)`` to execute code "on the executor"; exceptions
    propagate to the caller.
    """

    def __init__(self):
        self._q: queue.Queue = queue.Queue()
        self._started = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        if not self._started.wait(timeout=10):  # pragma: no cover
            raise RuntimeError("StreamBoundWorker failed to start")

    def _loop(self):
        stream = mx.new_stream(mx.default_device())
        mx.set_default_stream(stream)
        self._started.set()
        while True:
            item = self._q.get()
            if item is None:
                return
            fn, box, done = item
            try:
                box["result"] = fn()
            except BaseException as e:  # noqa: BLE001 - propagated to caller
                box["error"] = e
            done.set()

    def run(self, fn, timeout: float = 60):
        """Run ``fn()`` on the stream-bound thread; return its result or
        re-raise its exception in the calling thread."""
        box: dict = {}
        done = threading.Event()
        self._q.put((fn, box, done))
        if not done.wait(timeout):  # pragma: no cover
            raise TimeoutError("StreamBoundWorker call timed out")
        if "error" in box:
            raise box["error"]
        return box.get("result")

    def close(self):
        self._q.put(None)
        self._thread.join(timeout=10)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
