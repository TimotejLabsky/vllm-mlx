# SPDX-License-Identifier: Apache-2.0
"""(#181) One GPU, two schedulers: the turn gate of a ``--vision-split`` route.

A split route runs the text scheduler (``AsyncEngineCore``, stepping on the
``engine-core`` worker thread) and the vision scheduler (``MLLMScheduler``,
stepping on the event loop) over ONE model's weights. Nothing else stops the
two from evaluating at the same time on two threads: MLX makes no promise
about that, both sides' memory guards would see half the picture mid-step,
and the module-level ``generation_stream`` handles that mlx-lm / mlx-vlm read
are process-global while the streams they name exist only on the thread that
made them (PATCHES.md #28: a stream used off its thread is "There is no
Stream(gpu, N) in current thread").

Policy: **one side at a time, one turn = one unit of work, alternate when
both want it.**

* A text turn is ONE call on the text worker thread: a scheduler step, a
  cache clear, a cache save/load, the model build. :class:`GatedWorkerExecutor`
  wraps every function submitted to that thread, so no MLX work there can run
  outside a turn.
* A vision turn is ONE ``MLLMScheduler.step()`` (plus its first-step stream
  bind), taken with :func:`vision_turn` around the call in ``_process_loop``.
  The step is synchronous on the event loop, so a vision turn never spans an
  ``await`` and always ends.
* Alternation: after a side's turn, if the other side is waiting, the side
  that just went may not go again until the other has had one turn. The
  vision side waits asynchronously (an ``asyncio.Event`` per waiter), so the
  event loop keeps serving while text holds the GPU. The text side waits on a
  condition variable on its own worker thread.
* The text side's yield is BOUNDED (``yield_timeout_s``): a vision waiter
  that cannot run because the event loop itself is blocked (a synchronous
  call waiting on the text worker, e.g. ``clear_runtime_caches``) must not
  deadlock the worker. After the bound the text turn proceeds and the
  timeout is counted. The vision side's yield needs no bound: text turns
  always end, and the vision coroutine waits without blocking anyone.
* Turns are per step, not per request: an abort, a client disconnect or a
  timeout never holds the gate. The longest wait either side can see is one
  turn of the other side (a prefill chunk, or one vision step - which can be
  long: an image encode plus its prefill).
* Every turn re-points the generation-stream globals at a stream owned by
  the thread taking the turn (the one that side used last), so whatever the
  other thread bound last can never be read across threads.

The gate holds no MLX state of its own and is inert unless a split route
builds one: a normal route never constructs it.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, Optional

logger = logging.getLogger(__name__)

TEXT = "text"
VISION = "vision"
_SIDES = (TEXT, VISION)

# A turn longer than this is logged: it is how long the other side waited.
_SLOW_TURN_S = 5.0


def _current_generation_stream() -> Any:
    """The stream the mlx-lm generation global names right now (or None)."""
    import sys

    module = sys.modules.get("mlx_lm.generate")
    return getattr(module, "generation_stream", None) if module else None


class GpuGate:
    """Mutual exclusion with alternation between the text and vision sides.

    Thread-safe; the text side calls :meth:`acquire_text` on its worker
    thread, the vision side awaits :meth:`acquire_vision` on the event loop.
    """

    def __init__(
        self,
        yield_timeout_s: float = 0.5,
        rebind_streams: bool = True,
    ) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._holder: Optional[str] = None
        self._last: Optional[str] = None
        self._waiting = {TEXT: 0, VISION: 0}
        self._vision_waiters: list[tuple[asyncio.AbstractEventLoop, asyncio.Event]] = []
        self._yield_timeout_s = float(yield_timeout_s)
        self._rebind_streams = rebind_streams
        # The stream each side's thread used last (set at release).
        self._streams: dict[str, Any] = {}
        self._turn_started = 0.0
        self.counters = {
            "text_turns": 0,
            "vision_turns": 0,
            "text_wait_s": 0.0,
            "vision_wait_s": 0.0,
            "text_hold_s": 0.0,
            "vision_hold_s": 0.0,
            "max_text_wait_s": 0.0,
            "max_vision_wait_s": 0.0,
            "text_yield_timeouts": 0,
            "vision_cancelled_waits": 0,
            "slow_turns": 0,
        }

    # -- helpers (call with the condition held) ------------------------------
    def _may_take(self, side: str) -> bool:
        if self._holder is not None:
            return False
        other = VISION if side == TEXT else TEXT
        return not (self._waiting[other] and self._last == side)

    def _grant(self, side: str, waited: float) -> None:
        self._holder = side
        self._turn_started = time.perf_counter()
        self.counters[f"{side}_turns"] += 1
        self.counters[f"{side}_wait_s"] += waited
        if waited > self.counters[f"max_{side}_wait_s"]:
            self.counters[f"max_{side}_wait_s"] = waited

    def _wake_vision_waiters(self) -> None:
        waiters, self._vision_waiters = self._vision_waiters, []
        for loop, event in waiters:
            try:
                loop.call_soon_threadsafe(event.set)
            except RuntimeError:  # loop closed: nobody left to wake
                pass

    # -- stream ownership ------------------------------------------------------
    def _enter_streams(self, side: str) -> None:
        """Point the generation globals at a stream this thread owns."""
        if not self._rebind_streams:
            return
        try:
            import mlx.core as mx

            from .mlx_streams import bind_generation_streams

            stream = self._streams.get(side)
            if stream is None:
                stream = mx.new_stream(mx.default_device())
                self._streams[side] = stream
            bind_generation_streams(stream=stream)
        except Exception:  # noqa: BLE001 - a missing MLX must not wedge the gate
            logger.debug("[gpu-gate] stream re-point failed", exc_info=True)

    def _leave_streams(self, side: str) -> None:
        if not self._rebind_streams:
            return
        # The side may have bound its own stream during the turn (the
        # schedulers bind once at their first step). Whatever the global
        # names now was set on this thread during this turn: remember it.
        current = _current_generation_stream()
        if current is not None:
            self._streams[side] = current

    # -- text side (blocking, on the worker thread) ---------------------------
    def acquire_text(self) -> None:
        start = time.perf_counter()
        with self._cond:
            self._waiting[TEXT] += 1
            deadline: Optional[float] = None
            try:
                while True:
                    if self._may_take(TEXT):
                        break
                    if self._holder is None:
                        # Free, but vision is waiting and text went last:
                        # yield for a bounded time (see the module doc).
                        now = time.perf_counter()
                        if deadline is None:
                            deadline = now + self._yield_timeout_s
                        remaining = deadline - now
                        if remaining <= 0:
                            self.counters["text_yield_timeouts"] += 1
                            break
                        self._cond.wait(remaining)
                    else:
                        self._cond.wait()
            finally:
                self._waiting[TEXT] -= 1
            self._grant(TEXT, time.perf_counter() - start)
        self._enter_streams(TEXT)

    # -- vision side (async, on the event loop) -------------------------------
    async def acquire_vision(self) -> None:
        loop = asyncio.get_running_loop()
        start = time.perf_counter()
        registered = False
        try:
            while True:
                with self._cond:
                    if self._may_take(VISION):
                        if registered:
                            self._waiting[VISION] -= 1
                            registered = False
                        self._grant(VISION, time.perf_counter() - start)
                        break
                    if not registered:
                        self._waiting[VISION] += 1
                        registered = True
                    event = asyncio.Event()
                    self._vision_waiters.append((loop, event))
                await event.wait()
        except BaseException:
            with self._cond:
                if registered:
                    self._waiting[VISION] -= 1
                    self.counters["vision_cancelled_waits"] += 1
                    # A text waiter may be yielding to us: let it go.
                    self._cond.notify_all()
            raise
        self._enter_streams(VISION)

    # -- both ------------------------------------------------------------------
    def release(self, side: str) -> None:
        self._leave_streams(side)
        with self._cond:
            if self._holder != side:
                raise RuntimeError(
                    f"gpu gate released by {side!r} while held by {self._holder!r}"
                )
            held = time.perf_counter() - self._turn_started
            self.counters[f"{side}_hold_s"] += held
            if held > _SLOW_TURN_S:
                self.counters["slow_turns"] += 1
                logger.info(
                    "[gpu-gate] %s turn held the GPU %.1fs (waiting: text=%d vision=%d)",
                    side,
                    held,
                    self._waiting[TEXT],
                    self._waiting[VISION],
                )
            self._holder = None
            self._last = side
            self._cond.notify_all()
            self._wake_vision_waiters()

    def run_text_turn(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run ``fn`` as one text turn (on the calling thread)."""
        self.acquire_text()
        try:
            return fn(*args, **kwargs)
        finally:
            self.release(TEXT)

    def stats(self) -> dict[str, Any]:
        with self._cond:
            out: dict[str, Any] = {
                k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in self.counters.items()
            }
            out["holder"] = self._holder
            out["waiting"] = dict(self._waiting)
            out["yield_timeout_s"] = self._yield_timeout_s
            out["policy"] = "alternate-per-step"
        return out


class GatedWorkerExecutor(ThreadPoolExecutor):
    """The text side's single worker thread, every submitted call one turn.

    A drop-in for the ``ThreadPoolExecutor(max_workers=1)`` BatchedEngine
    builds in ``_generation_worker``: same thread name, same initializer, so
    ``_run_on_generation_worker_sync``'s same-thread shortcut keeps working
    (a call made ON the worker runs inline, inside the turn already held).
    """

    def __init__(
        self,
        gate: GpuGate,
        *,
        thread_name_prefix: str = "engine-core",
        initializer: Optional[Callable[[], Any]] = None,
    ) -> None:
        super().__init__(
            max_workers=1,
            thread_name_prefix=thread_name_prefix,
            initializer=initializer,
        )
        self.gate = gate

    def submit(self, fn, /, *args, **kwargs) -> Future:  # type: ignore[override]
        return super().submit(self.gate.run_text_turn, fn, *args, **kwargs)


@asynccontextmanager
async def vision_turn(gate: Optional[GpuGate]) -> AsyncIterator[None]:
    """One vision turn around a synchronous MLLM step; no-op without a gate
    (every route that is not ``--vision-split``)."""
    if gate is None:
        yield
        return
    await gate.acquire_vision()
    try:
        yield
    finally:
        gate.release(VISION)
