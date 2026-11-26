# SPDX-License-Identifier: Apache-2.0
"""PATCHES.md #181: ``--vision-split`` — one route serves text and images.

Mocked suite only (no weights). The real-model check is
``scripts/fork/e2e_vision_split.py``.
"""

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from vllm_mlx.engine.base import MediaNotSupported
from vllm_mlx.engine.batched import BatchedEngine
from vllm_mlx.gpu_gate import TEXT, VISION, GatedWorkerExecutor, GpuGate, vision_turn
from vllm_mlx.vision_split import (
    VisionSplitEngine,
    VisionSplitError,
    check_vision_split_config,
    engine_serves_media,
    messages_have_media,
    request_is_mllm,
    vision_scheduler_config,
)

IMAGE_MSG = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
        ],
    }
]
TEXT_MSG = [{"role": "user", "content": "hello"}]


# ---------------------------------------------------------------------------
# GPU gate
# ---------------------------------------------------------------------------


def _wait_until(pred, timeout=5.0):
    deadline = time.time() + timeout
    while not pred():
        if time.time() > deadline:
            raise AssertionError("condition not reached")
        time.sleep(0.002)


class TestGpuGate:
    def test_text_and_vision_never_overlap(self):
        gate = GpuGate(yield_timeout_s=0.05, rebind_streams=False)
        active = []
        overlaps = []
        lock = threading.Lock()

        def enter(side):
            with lock:
                if active:
                    overlaps.append((side, list(active)))
                active.append(side)

        def leave(side):
            with lock:
                active.remove(side)

        def text_worker():
            for _ in range(60):
                gate.acquire_text()
                enter(TEXT)
                time.sleep(0.001)
                leave(TEXT)
                gate.release(TEXT)

        async def vision():
            for _ in range(60):
                await gate.acquire_vision()
                enter(VISION)
                time.sleep(0.001)  # synchronous, like MLLMScheduler.step()
                leave(VISION)
                gate.release(VISION)
                await asyncio.sleep(0)

        t = threading.Thread(target=text_worker)
        t.start()
        asyncio.run(vision())
        t.join(10)
        assert not overlaps
        stats = gate.stats()
        assert stats["text_turns"] == 60 and stats["vision_turns"] == 60
        assert stats["holder"] is None and stats["waiting"] == {TEXT: 0, VISION: 0}

    def test_alternates_when_both_wait(self):
        """Text that just had a turn waits for the vision waiter's turn."""
        gate = GpuGate(yield_timeout_s=5.0, rebind_streams=False)
        order = []

        def text_worker():
            gate.acquire_text()
            order.append("text-1")
            _wait_until(lambda: gate._waiting[VISION] == 1)
            gate.release(TEXT)
            gate.acquire_text()  # must not barge ahead of the vision waiter
            order.append("text-2")
            gate.release(TEXT)

        async def vision():
            t = threading.Thread(target=text_worker)
            t.start()
            await asyncio.to_thread(_wait_until, lambda: gate._holder == TEXT)
            await gate.acquire_vision()
            order.append("vision")
            time.sleep(0.02)
            gate.release(VISION)
            await asyncio.to_thread(t.join, 10)

        asyncio.run(vision())
        assert order == ["text-1", "vision", "text-2"]

    def test_vision_that_went_last_yields_to_a_waiting_text(self):
        gate = GpuGate(rebind_streams=False)

        async def scenario():
            await gate.acquire_vision()
            got = []
            t = threading.Thread(target=lambda: (gate.acquire_text(), got.append(1)))
            t.start()
            await asyncio.to_thread(_wait_until, lambda: gate._waiting[TEXT] == 1)
            gate.release(VISION)
            await gate.acquire_vision()  # queued behind the text turn
            assert got == [1]
            gate.release(VISION)
            t.join(5)

        def release_text_soon():
            _wait_until(lambda: gate._holder == TEXT)
            time.sleep(0.02)
            gate.release(TEXT)

        helper = threading.Thread(target=release_text_soon)
        helper.start()
        asyncio.run(scenario())
        helper.join(5)
        assert gate.stats()["vision_turns"] == 2 and gate.stats()["text_turns"] == 1

    def test_text_yield_is_bounded(self):
        """A vision waiter that cannot run (event loop blocked in a sync call
        to the text worker) must not deadlock the text side."""
        gate = GpuGate(yield_timeout_s=0.05, rebind_streams=False)
        with gate._cond:
            gate._waiting[VISION] = 1
            gate._last = TEXT
        t0 = time.perf_counter()
        gate.acquire_text()
        elapsed = time.perf_counter() - t0
        gate.release(TEXT)
        assert 0.04 <= elapsed < 2.0
        assert gate.stats()["text_yield_timeouts"] == 1

    def test_cancelled_vision_wait_unregisters(self):
        gate = GpuGate(rebind_streams=False)
        gate.acquire_text()

        async def scenario():
            task = asyncio.ensure_future(gate.acquire_vision())
            await asyncio.sleep(0.01)
            assert gate._waiting[VISION] == 1
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(scenario())
        assert gate._waiting[VISION] == 0
        assert gate.stats()["vision_cancelled_waits"] == 1
        gate.release(TEXT)
        t0 = time.perf_counter()
        gate.acquire_text()  # nobody waiting: immediate
        assert time.perf_counter() - t0 < 0.05
        gate.release(TEXT)

    def test_release_by_the_wrong_side_raises(self):
        gate = GpuGate(rebind_streams=False)
        gate.acquire_text()
        with pytest.raises(RuntimeError):
            gate.release(VISION)
        gate.release(TEXT)

    def test_each_turn_repoints_the_generation_stream_at_its_own_thread(self):
        """#28: the generation_stream globals are process-wide; a turn must
        never run with the other thread's stream there."""
        import importlib

        from vllm_mlx.mlx_streams import (
            restore_generation_streams,
            snapshot_generation_streams,
        )

        # mlx_lm re-exports a generate() FUNCTION under the submodule's name
        mlg = importlib.import_module("mlx_lm.generate")
        snap = snapshot_generation_streams()
        gate = GpuGate()
        ex = GatedWorkerExecutor(gate, thread_name_prefix="t181")
        try:
            seen_text = [ex.submit(lambda: mlg.generation_stream).result()]

            async def vision():
                async with vision_turn(gate):
                    return mlg.generation_stream

            seen_vision = asyncio.run(vision())
            seen_text.append(ex.submit(lambda: mlg.generation_stream).result())
            assert seen_text[0] is seen_text[1]
            assert seen_vision is not seen_text[0]
            assert gate._streams[TEXT] is seen_text[0]
            assert gate._streams[VISION] is seen_vision
        finally:
            ex.shutdown(wait=True)
            restore_generation_streams(snap)

    def test_gated_executor_runs_every_call_as_a_text_turn(self):
        gate = GpuGate(rebind_streams=False)
        inits = []
        ex = GatedWorkerExecutor(gate, initializer=lambda: inits.append(1))
        try:
            assert ex.submit(lambda: gate._holder).result() == TEXT
            assert ex.submit(lambda a, b=0: a + b, 2, b=3).result() == 5
            assert gate._holder is None and inits == [1]
            assert gate.stats()["text_turns"] == 2
        finally:
            ex.shutdown(wait=True)

    def test_vision_turn_without_a_gate_is_a_noop(self):
        async def go():
            async with vision_turn(None):
                return "ran"

        assert asyncio.run(go()) == "ran"


def test_mllm_process_loop_steps_inside_a_vision_turn():
    from vllm_mlx.mllm_scheduler import MLLMScheduler

    gate = GpuGate(rebind_streams=False)
    sched = MLLMScheduler.__new__(MLLMScheduler)
    sched._running = True
    sched.batch_generator = None
    sched.waiting, sched.running = [], {}
    seen = []

    def step():
        seen.append(gate._holder)
        sched._running = False

    sched.has_requests = lambda: True
    sched.step = step
    sched._gpu_gate = gate
    asyncio.run(sched._process_loop())
    assert seen == [VISION]
    assert gate._holder is None and gate.stats()["vision_turns"] == 1

    # no gate (every other route): the step runs as before
    sched._running = True
    sched._gpu_gate = None
    asyncio.run(sched._process_loop())
    assert seen == [VISION, None]


# ---------------------------------------------------------------------------
# Request classification
# ---------------------------------------------------------------------------


class TestClassification:
    def test_media_detection(self):
        assert messages_have_media(IMAGE_MSG)
        assert not messages_have_media(TEXT_MSG)
        assert not messages_have_media(
            [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
        )
        for kind in ("image", "video", "video_url", "audio", "audio_url"):
            assert messages_have_media([{"role": "user", "content": [{"type": kind}]}])
        assert messages_have_media(
            [SimpleNamespace(content=[SimpleNamespace(type="image_url")])]
        )
        assert not messages_have_media(None)

    def test_flag_off_engines_keep_is_mllm(self):
        from unittest.mock import MagicMock

        for is_mllm in (True, False):
            for engine in (
                SimpleNamespace(is_mllm=is_mllm),
                MagicMock(is_mllm=is_mllm),
            ):
                assert request_is_mllm(engine, IMAGE_MSG) is is_mllm
                assert request_is_mllm(engine, TEXT_MSG) is is_mllm
                assert engine_serves_media(engine) is is_mllm

    def test_split_engine_decides_per_request(self):
        engine = SimpleNamespace(is_mllm=False, vision_split=True)
        assert request_is_mllm(engine, IMAGE_MSG) is True
        assert request_is_mllm(engine, TEXT_MSG) is False
        assert engine_serves_media(engine) is True


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfig:
    def test_valid(self, monkeypatch):
        monkeypatch.delenv("VLLM_MLX_MOE_GATEUP_FUSION", raising=False)
        check_vision_split_config(use_batching=True)

    @pytest.mark.parametrize(
        "kwargs,needle",
        [
            ({"use_batching": False}, "--continuous-batching"),
            ({"force_text_only": True}, "--text-only"),
            ({"force_mllm": True}, "--mllm"),
            ({"auto_unload_idle_seconds": 60}, "--auto-unload-idle-seconds"),
            ({"lazy_load_model": True}, "--lazy-load-model"),
            ({"mllm_draft_model": "x"}, "draft"),
            ({"enable_mtp": True}, "--enable-mtp"),
            ({"compile": True}, "--compile"),
            ({"models_config": "m.yaml"}, "--models-config"),
        ],
    )
    def test_refusals(self, monkeypatch, kwargs, needle):
        monkeypatch.delenv("VLLM_MLX_MOE_GATEUP_FUSION", raising=False)
        args = {"use_batching": True, **kwargs}
        with pytest.raises(VisionSplitError, match=needle.replace("-", r"\-")):
            check_vision_split_config(**args)

    def test_moe_fusion_refused(self, monkeypatch):
        monkeypatch.setenv("VLLM_MLX_MOE_GATEUP_FUSION", "1")
        with pytest.raises(VisionSplitError, match="second copy"):
            check_vision_split_config(use_batching=True)

    def test_vision_scheduler_config_copy(self):
        from vllm_mlx.scheduler import SchedulerConfig

        cfg = SchedulerConfig(
            cache_memory_mb=16384, ssd_cache_dir="/x", enable_mtp=True
        )
        v = vision_scheduler_config(cfg, 512)
        assert (v.cache_memory_mb, v.ssd_cache_dir, v.enable_mtp) == (512, None, False)
        assert (cfg.cache_memory_mb, cfg.ssd_cache_dir, cfg.enable_mtp) == (
            16384,
            "/x",
            True,
        )
        assert v.max_num_seqs == cfg.max_num_seqs
        assert vision_scheduler_config(None, 7).cache_memory_mb == 7


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def _split(tmp_path, **kw):
    from vllm_mlx.scheduler import SchedulerConfig

    cfg = SchedulerConfig(cache_memory_mb=2048, ssd_cache_dir=str(tmp_path / "ssd"))
    return VisionSplitEngine(
        "org/Some-VL", scheduler_config=cfg, model_path=str(tmp_path), **kw
    )


class TestEngineShape:
    def test_children(self, tmp_path):
        eng = _split(tmp_path, vision_cache_mb=300)
        try:
            assert isinstance(eng.text, BatchedEngine) and isinstance(
                eng.vision, BatchedEngine
            )
            assert eng.text.is_mllm is False and eng.vision.is_mllm is True
            assert isinstance(eng.text._generation_executor, GatedWorkerExecutor)
            assert eng.text._generation_executor.gate is eng.gate
            # The text child keeps the route's config object itself.
            assert eng.text._scheduler_config.cache_memory_mb == 2048
            assert eng.text._scheduler_config.ssd_cache_dir.endswith("ssd")
            v = eng.vision._scheduler_config
            assert (v.cache_memory_mb, v.ssd_cache_dir, v.enable_mtp) == (
                300,
                None,
                False,
            )
            assert eng.is_mllm is False and eng.vision_split is True
            assert engine_serves_media(eng) and request_is_mllm(eng, IMAGE_MSG)
            assert not request_is_mllm(eng, TEXT_MSG)
        finally:
            eng.text._generation_executor.shutdown(wait=True)

    def test_server_set_attributes_reach_both_children(self, tmp_path):
        eng = _split(tmp_path)
        try:
            eng.preserve_native_tool_format = True
            eng.use_harmony_rendering = True
            for child in (eng.text, eng.vision):
                assert child.preserve_native_tool_format is True
                assert child.use_harmony_rendering is True
            # unknown reads answer as the --text-only child
            assert eng._apply_chat_template.__self__ is eng.text
        finally:
            eng.text._generation_executor.shutdown(wait=True)

    def test_bad_budget(self, tmp_path):
        with pytest.raises(VisionSplitError):
            _split(tmp_path, vision_cache_mb=0)


class _Tok:
    chat_template = "tmpl"

    def apply_chat_template(self, messages, **kw):
        return "TEXT|" + "|".join(str(m.get("content")) for m in messages)

    def encode(self, s):
        return list(range(len(s)))


class _Proc:
    chat_template = "vtmpl"
    tokenizer = _Tok()

    def apply_chat_template(self, messages, **kw):
        return "VISION|" + repr(messages)


def _out(text="ok", finished=True):
    return SimpleNamespace(
        output_text=text,
        new_text=text,
        prompt_tokens=3,
        completion_tokens=1,
        cached_tokens=0,
        finished=finished,
        finish_reason="stop",
        mtp_drafts=0,
        mtp_accepted=0,
        output_token_ids=[1],
    )


class _FakeCore:
    def __init__(self):
        self.calls = []

    async def add_request(self, prompt, sampling_params, prefix_boundary=0):
        self.calls.append(("add", prompt, sampling_params))
        return "t-1"

    async def stream_outputs(self, rid):
        yield _out()

    async def generate(self, prompt, sampling_params):
        self.calls.append(("gen", prompt, sampling_params))
        return _out()


class _FakeMllm:
    def __init__(self):
        self.calls = []

    async def add_request_async(self, prompt, **kw):
        self.calls.append(("add", prompt, kw))
        return "v-1"

    async def stream_outputs(self, rid):
        yield _out()

    async def generate(self, prompt, **kw):
        self.calls.append(("gen", prompt, kw))
        return _out()


def _wired(tmp_path):
    eng = _split(tmp_path)
    eng.text._generation_executor.shutdown(wait=True)
    eng.text._generation_executor = None
    eng.text._loaded = eng.vision._loaded = eng._loaded = True
    eng.text._tokenizer = _Tok()
    eng.text._engine = _FakeCore()
    eng.vision._processor = _Proc()
    eng.vision._mllm_scheduler = _FakeMllm()
    return eng


def _drain(agen):
    async def go():
        return [o async for o in agen]

    return asyncio.run(go())


class TestRouting:
    def test_text_request_reaches_the_text_core_only(self, tmp_path):
        eng = _wired(tmp_path)
        _drain(eng.stream_chat(TEXT_MSG, max_tokens=8, temperature=0.0))
        asyncio.run(eng.chat(TEXT_MSG, max_tokens=8, temperature=0.0))
        assert [c[0] for c in eng.text._engine.calls] == ["add", "gen"]
        assert eng.text._engine.calls[0][1].startswith("TEXT|")
        assert eng.vision._mllm_scheduler.calls == []

    def test_media_request_reaches_the_mllm_scheduler_only(self, tmp_path):
        eng = _wired(tmp_path)
        _drain(eng.stream_chat(IMAGE_MSG, max_tokens=8, temperature=0.0))
        asyncio.run(eng.chat(IMAGE_MSG, max_tokens=8))
        calls = eng.vision._mllm_scheduler.calls
        assert [c[0] for c in calls] == ["add", "gen"]
        assert calls[0][1].startswith("VISION|")
        assert calls[0][2]["images"] == ["data:image/png;base64,aGk="]
        assert eng.text._engine.calls == []

    def test_generate_routes_on_media_kwargs(self, tmp_path):
        eng = _wired(tmp_path)
        asyncio.run(eng.generate(prompt="p", max_tokens=4))
        asyncio.run(eng.generate(prompt="p", max_tokens=4, images=["x.png"]))
        _drain(eng.stream_generate(prompt="p", max_tokens=4, videos=["v.mp4"]))
        assert len(eng.text._engine.calls) == 1
        assert len(eng.vision._mllm_scheduler.calls) == 2

    def test_text_side_is_the_text_only_code_path(self, tmp_path):
        """A text request on the split route builds the same prompt and the
        same SamplingParams a --text-only BatchedEngine builds."""
        eng = _wired(tmp_path)
        ref = BatchedEngine(
            "org/Some-VL", force_text_only=True, model_path=str(tmp_path)
        )
        ref._loaded = True
        ref._tokenizer = _Tok()
        ref._engine = _FakeCore()
        kwargs = dict(max_tokens=9, temperature=0.0, top_p=0.5, top_k=3, stop=["x"])
        _drain(eng.stream_chat(TEXT_MSG, **kwargs))
        _drain(ref.stream_chat(TEXT_MSG, **kwargs))
        (_, p1, s1), (_, p2, s2) = eng.text._engine.calls[0], ref._engine.calls[0]
        assert p1 == p2
        for f in ("max_tokens", "temperature", "top_p", "top_k", "min_p", "stop"):
            assert getattr(s1, f) == getattr(s2, f), f

    def test_text_child_still_refuses_media(self, tmp_path):
        """Defense in depth stays: media can never reach the text core."""
        eng = _wired(tmp_path)
        with pytest.raises(MediaNotSupported):
            asyncio.run(eng.text.generate(prompt="p", images=["x.png"]))


class TestStatsAndCaches:
    def _stubbed(self, tmp_path):
        eng = _split(tmp_path)
        eng.text._generation_executor.shutdown(wait=True)
        calls = []

        def child(name, stats, cache):
            async def aret(*a, **k):
                calls.append((name, "async", a))
                return (
                    {"prefix_cache": True} if name == "text" else {"prefix_cache": True}
                )

            ns = SimpleNamespace(
                get_stats=lambda: stats,
                get_cache_stats=lambda: cache,
                clear_runtime_caches=lambda: calls.append((name, "clear")) or {name: 1},
                _clear_runtime_caches_on_owner=aret,
                clear_prefix_cache=lambda: calls.append((name, "prefix"))
                or {"prefix_cache": True},
                _clear_prefix_cache_on_owner=aret,
                save_cache_to_disk=lambda d: calls.append((name, "save", d)) or True,
                _save_cache_to_disk_on_owner=aret,
                load_cache_from_disk=lambda d: calls.append((name, "load", d)) or 3,
                _load_cache_from_disk_on_owner=aret,
                abort_request=AsyncMock(return_value=(name == "vision")),
                stop=AsyncMock(),
                is_healthy=lambda: True,
                raise_if_serialized_busy=lambda rid: calls.append((name, "busy")),
            )
            return ns

        eng.text = child(
            "text",
            {
                "engine_type": "batched",
                "num_running": 1,
                "num_waiting": 2,
                "requests": [{"request_id": "t"}],
                "system_kv_cache": {"hits": 4},
            },
            {"system_kv_cache": {"hits": 4}},
        )
        eng.vision = child(
            "vision",
            {
                "num_running": 1,
                "num_waiting": 0,
                "requests": [{"request_id": "v"}],
                "memory_aware_cache": {"hits": 1},
                "vision_embedding_cache": {"hits": 2},
            },
            {"prefix_cache": {"hits": 1}},
        )
        return eng, calls

    def test_stats(self, tmp_path):
        eng, _ = self._stubbed(tmp_path)
        st = eng.get_stats()
        assert st["system_kv_cache"] == {"hits": 4}  # text block unchanged, top level
        assert "memory_aware_cache" not in st  # vision cache never masks it
        assert (st["num_running"], st["num_waiting"]) == (2, 2)
        assert [r["request_id"] for r in st["requests"]] == ["t", "v"]
        assert st["requests"][1]["side"] == "vision"
        assert st["vision_embedding_cache"] == {"hits": 2}
        vs = st["vision_split"]
        assert vs["enabled"] and vs["vision"]["memory_aware_cache"] == {"hits": 1}
        assert vs["gate"]["policy"] == "alternate-per-step"
        assert vs["vision_prefix_cache_budget_mb"] == 1024
        cs = eng.get_cache_stats()
        assert cs["system_kv_cache"] == {"hits": 4} and cs["vision"] == {
            "prefix_cache": {"hits": 1}
        }

    def test_clears_cover_both_and_persistence_is_text_only(self, tmp_path):
        eng, calls = self._stubbed(tmp_path)
        assert eng.clear_runtime_caches() == {
            "text": {"text": 1},
            "vision": {"vision": 1},
        }
        assert eng.clear_prefix_cache() == {
            "prefix_cache": True,
            "vision_prefix_cache": True,
        }
        assert eng.save_cache_to_disk("/d") is True
        assert eng.load_cache_from_disk("/d") == 3
        assert ("text", "save", "/d") in calls and ("text", "load", "/d") in calls
        assert not any(c[0] == "vision" and c[1] in ("save", "load") for c in calls)
        out = asyncio.run(eng._clear_runtime_caches_on_owner())
        assert set(out) == {"text", "vision"}
        out = asyncio.run(eng._clear_prefix_cache_on_owner())
        assert out == {"prefix_cache": True, "vision_prefix_cache": True}
        asyncio.run(eng._save_cache_to_disk_on_owner("/d"))
        asyncio.run(eng._load_cache_from_disk_on_owner("/d"))
        owners = [c for c in calls if c[1] == "async"]
        assert [c[0] for c in owners] == [
            "text",
            "vision",
            "text",
            "vision",
            "text",
            "text",
        ]

    def test_abort_stop_health_busy(self, tmp_path):
        eng, calls = self._stubbed(tmp_path)
        assert asyncio.run(eng.abort_request("x")) is True  # found on vision
        eng.text.abort_request.assert_awaited_once_with("x")
        eng.vision.abort_request.assert_awaited_once_with("x")
        asyncio.run(eng.stop())
        eng.text.stop.assert_awaited_once()
        eng.vision.stop.assert_awaited_once()
        assert eng.is_healthy()
        eng.raise_if_serialized_busy("r")
        assert ("text", "busy") in calls


class TestStart:
    def _engine(self, tmp_path, monkeypatch, built):
        eng = _split(tmp_path)
        sched = SimpleNamespace()
        vlm = object()

        async def vision_start():
            eng.vision._mllm_scheduler = sched
            eng.vision._model = vlm
            eng.vision._loaded = True

        monkeypatch.setattr(eng.vision, "start", vision_start)
        monkeypatch.setattr(eng.vision, "stop", AsyncMock())
        monkeypatch.setattr(eng.text, "start", AsyncMock())
        monkeypatch.setattr(eng.text, "stop", AsyncMock())
        monkeypatch.setattr(eng.text, "_configure_metal_memory_limits", lambda: None)
        import vllm_mlx.text_model_from_vlm as tmv
        import vllm_mlx.vision_split as vs

        seen = {}

        def fake_build(model, path, enable_mtp=True):
            seen["thread"] = threading.current_thread().name
            seen["holder"] = eng.gate._holder
            seen["args"] = (model, path, enable_mtp)
            if isinstance(built, Exception):
                raise built
            return built

        monkeypatch.setattr(tmv, "build_text_model", fake_build)
        monkeypatch.setattr(vs, "load_text_tokenizer", lambda *a: "TOK")
        monkeypatch.setattr(vs, "text_model_type", lambda p: "qwen3_5_text")
        return eng, sched, vlm, seen

    def test_success_wires_the_gate_and_the_shared_model(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VLLM_MLX_MOE_DECODE_KERNELS", raising=False)
        monkeypatch.delenv("VLLM_MLX_MOE_GATEUP_FUSION", raising=False)
        import mlx.nn as nn

        import mlx.core as mx

        tm = nn.Linear(2, 2)
        mx.eval(tm.parameters())  # as the vlm load leaves its arrays (#28)
        eng, sched, vlm, seen = self._engine(tmp_path, monkeypatch, tm)
        try:
            asyncio.run(eng.start())
            assert sched._gpu_gate is eng.gate
            assert eng.text._model is tm and eng.text._tokenizer == "TOK"
            assert seen["args"] == (vlm, str(tmp_path), False)
            # built on the text worker, inside a text turn
            assert seen["thread"].startswith("engine-core") and seen["holder"] == TEXT
            eng.text.start.assert_awaited_once()
            assert eng._loaded
        finally:
            eng.text._generation_executor.shutdown(wait=True)

    @pytest.mark.parametrize("built", [None, ValueError("shape mismatch")])
    def test_no_text_model_is_a_startup_error(self, tmp_path, monkeypatch, built):
        eng, _, _, _ = self._engine(tmp_path, monkeypatch, built)
        try:
            with pytest.raises(VisionSplitError, match="qwen3_5_text"):
                asyncio.run(eng.start())
            eng.text.start.assert_not_awaited()
            eng.text.stop.assert_awaited()
            eng.vision.stop.assert_awaited()
            assert not eng._loaded
        finally:
            eng.text._generation_executor.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Server and CLI seams
# ---------------------------------------------------------------------------


class _SplitLike:
    is_mllm = False
    vision_split = True
    preserve_native_tool_format = False
    use_harmony_rendering = False


class TestServerSeams:
    def _req(self, messages):
        from vllm_mlx.server import ChatCompletionRequest

        return ChatCompletionRequest(model="m", messages=messages)

    def test_text_on_split_takes_the_text_prep_path(self):
        from vllm_mlx.server import _prepare_chat_messages

        class _TextOnly(_SplitLike):
            vision_split = False

        a = _prepare_chat_messages(_SplitLike(), self._req(TEXT_MSG).messages)
        b = _prepare_chat_messages(_TextOnly(), self._req(TEXT_MSG).messages)
        assert a == b and a[4] is False

    def test_media_on_split_takes_the_mllm_prep_path(self):
        from vllm_mlx.server import _prepare_chat_messages

        class _Mllm(_SplitLike):
            is_mllm = True
            vision_split = False

        a = _prepare_chat_messages(_SplitLike(), self._req(IMAGE_MSG).messages)
        b = _prepare_chat_messages(_Mllm(), self._req(IMAGE_MSG).messages)
        assert a == b and a[4] is True
        assert any(p.get("type") == "image_url" for p in a[0][-1]["content"])

    def test_guards(self):
        from vllm_mlx.server import (
            _prepare_anthropic_invocation,
            _prepare_chat_completion_invocation,
        )

        class _TextOnly(_SplitLike):
            vision_split = False

        for prep in (
            _prepare_chat_completion_invocation,
            _prepare_anthropic_invocation,
        ):
            with pytest.raises(MediaNotSupported):
                prep(_TextOnly(), self._req(IMAGE_MSG), effective_max_tokens=8)
            prep(_SplitLike(), self._req(IMAGE_MSG), effective_max_tokens=8)

    def test_load_model_builds_the_split_engine_only_with_the_flag(self, monkeypatch):
        import vllm_mlx.server as server
        import vllm_mlx.vision_split as vs

        built = []

        class _Stub:
            def __init__(self, **kw):
                built.append(kw)

        monkeypatch.setattr(vs, "VisionSplitEngine", _Stub)
        monkeypatch.setattr(server, "_engine", None)
        monkeypatch.setattr(server, "_residency_manager", None)
        monkeypatch.delenv("VLLM_MLX_MOE_GATEUP_FUSION", raising=False)
        with pytest.raises(VisionSplitError, match="continuous-batching"):
            server.load_model("org/M", use_batching=False, vision_split=True)
        server.load_model(
            "org/M",
            use_batching=True,
            vision_split=True,
            vision_split_cache_mb=333,
            model_path="/p",
        )
        assert isinstance(server._engine, _Stub)
        assert built[0]["vision_cache_mb"] == 333 and built[0]["model_path"] == "/p"
        monkeypatch.setattr(server, "_engine", None)

    def test_cli_flags(self):
        from vllm_mlx.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["serve", "m"])
        assert args.vision_split is False and args.vision_split_cache_mb == 1024
        args = parser.parse_args(
            ["serve", "m", "--vision-split", "--vision-split-cache-mb", "64"]
        )
        assert args.vision_split is True and args.vision_split_cache_mb == 64

    def test_cli_refuses_vision_split_with_text_only(self, monkeypatch, capsys):
        from vllm_mlx import cli

        parser = cli.create_parser()
        args = parser.parse_args(
            ["serve", "m", "--continuous-batching", "--text-only", "--vision-split"]
        )
        monkeypatch.delenv("VLLM_MLX_MOE_GATEUP_FUSION", raising=False)
        with pytest.raises(SystemExit):
            cli.serve_command(args)
        assert (
            "--vision-split is mutually exclusive with --text-only"
            in capsys.readouterr().out
        )
