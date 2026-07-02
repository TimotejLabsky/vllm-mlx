"""Flip-enablement guards for the batched engine (PATCHES.md #39):
length-aware co-batching (padded-KV waste), queue-cap overload shedding,
and observability plumbing for the new cache counters.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_mlx import batched_system_kv as bkv
from vllm_mlx.batched_system_kv import BatchedSystemKV
from vllm_mlx.engine.base import EngineBusy

from tests.test_batched_system_kv import TOKENS, _donor_at, _FakeModel

# ------------------------------------------------------- pad-waste guard


def _req(rid, prompt_tokens, out_tokens=0):
    return SimpleNamespace(
        request_id=rid,
        num_prompt_tokens=prompt_tokens,
        output_token_ids=[0] * out_tokens,
    )


def _guard_scheduler(kv, running_lengths):
    running = {
        f"run-{i}": _req(f"run-{i}", n) for i, n in enumerate(running_lengths)
    }
    return SimpleNamespace(hybrid_kv=kv, running=running)


def test_pad_guard_inert_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_MLX_BATCHED_PAD_WASTE_MB", raising=False)
    kv = BatchedSystemKV(_FakeModel())
    kv.store("r1", TOKENS, _donor_at(len(TOKENS)))
    scheduler = _guard_scheduler(kv, [500_000])
    assert bkv.should_defer_cobatch(scheduler, _req("c", 100)) is False


def test_pad_guard_inert_on_cold_cache(monkeypatch):
    monkeypatch.setenv("VLLM_MLX_BATCHED_PAD_WASTE_MB", "1")
    kv = BatchedSystemKV(_FakeModel())  # no entries -> no bytes/token estimate
    scheduler = _guard_scheduler(kv, [500_000])
    assert bkv.should_defer_cobatch(scheduler, _req("c", 100)) is False


def test_pad_guard_defers_extreme_length_mix(monkeypatch):
    monkeypatch.setenv("VLLM_MLX_BATCHED_PAD_WASTE_MB", "1")
    kv = BatchedSystemKV(_FakeModel())
    kv.store("r1", TOKENS, _donor_at(len(TOKENS)))
    assert kv.bytes_per_token() > 0

    # ~500K-token difference at the tiny test bytes/token still crosses 1MB
    scheduler = _guard_scheduler(kv, [500_000])
    candidate = _req("c", 100)
    assert bkv.should_defer_cobatch(scheduler, candidate) is True
    assert candidate._pad_defer_logged is True

    # similar lengths -> negligible waste -> admit
    scheduler = _guard_scheduler(kv, [520])
    assert bkv.should_defer_cobatch(scheduler, _req("c2", 500)) is False


def test_pad_guard_inert_without_running(monkeypatch):
    monkeypatch.setenv("VLLM_MLX_BATCHED_PAD_WASTE_MB", "1")
    kv = BatchedSystemKV(_FakeModel())
    kv.store("r1", TOKENS, _donor_at(len(TOKENS)))
    scheduler = SimpleNamespace(hybrid_kv=kv, running={})
    assert bkv.should_defer_cobatch(scheduler, _req("c", 100)) is False


def test_schedule_waiting_defers_and_preserves_fcfs(monkeypatch):
    from vllm_mlx.request import Request, SamplingParams
    from vllm_mlx.scheduler import Scheduler, SchedulerConfig

    monkeypatch.setenv("VLLM_MLX_BATCHED_SYSTEM_KV", "1")
    monkeypatch.setenv("VLLM_MLX_BATCHED_PAD_WASTE_MB", "1")
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 0
    scheduler = Scheduler(
        model, tokenizer, SchedulerConfig(max_num_seqs=4, enable_prefix_cache=True)
    )
    # real cache with one entry so bytes_per_token() > 0
    scheduler.hybrid_kv = BatchedSystemKV(_FakeModel())
    scheduler.hybrid_kv.store("seed", TOKENS, _donor_at(len(TOKENS)))
    scheduler.running["long"] = _req("long", 500_000)
    scheduler.batch_generator = MagicMock()

    request = Request(
        request_id="short-1",
        prompt="x",
        sampling_params=SamplingParams(max_tokens=8),
        prompt_token_ids=[1, 2, 3],
        num_prompt_tokens=3,
    )
    scheduler.waiting.append(request)

    scheduled = scheduler._schedule_waiting()

    assert scheduled == []
    assert scheduler.waiting[0] is request  # deferred, FCFS preserved
    scheduler.batch_generator.insert_segments.assert_not_called()
    scheduler.batch_generator.insert.assert_not_called()


# ------------------------------------------------------------ queue cap


def _capped_scheduler(monkeypatch, cap):
    from vllm_mlx.scheduler import Scheduler, SchedulerConfig

    if cap is None:
        monkeypatch.delenv("VLLM_MLX_BATCHED_MAX_QUEUE", raising=False)
    else:
        monkeypatch.setenv("VLLM_MLX_BATCHED_MAX_QUEUE", str(cap))
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 0
    return Scheduler(
        model, tokenizer, SchedulerConfig(max_num_seqs=4, enable_prefix_cache=False)
    )


def _mk_request(rid):
    from vllm_mlx.request import Request, SamplingParams

    return Request(
        request_id=rid,
        prompt="x",
        sampling_params=SamplingParams(max_tokens=8),
        prompt_token_ids=[1, 2, 3],
        num_prompt_tokens=3,
    )


def test_queue_cap_rejects_over_capacity(monkeypatch):
    scheduler = _capped_scheduler(monkeypatch, 2)
    scheduler.add_request(_mk_request("r1"))
    scheduler.add_request(_mk_request("r2"))
    with pytest.raises(EngineBusy):
        scheduler.add_request(_mk_request("r3"))
    assert scheduler.queue_rejections == 1
    assert scheduler.get_stats()["queue_rejections"] == 1


def test_queue_cap_unbounded_by_default(monkeypatch):
    scheduler = _capped_scheduler(monkeypatch, None)
    for i in range(10):
        scheduler.add_request(_mk_request(f"r{i}"))
    assert scheduler.queue_rejections == 0


def test_engine_pre_stream_probe_raises_when_capped():
    from vllm_mlx.engine.batched import BatchedEngine

    engine = BatchedEngine.__new__(BatchedEngine)
    engine._engine = SimpleNamespace(
        scheduler=SimpleNamespace(queue_cap=1, waiting=[object()], queue_rejections=0)
    )
    with pytest.raises(EngineBusy):
        # SERVER CONTRACT: _probe_engine_busy passes the request id
        # positionally — call it the way the server does (a no-arg call
        # here masked the production TypeError->500 on every stream).
        engine.raise_if_serialized_busy("req-abc")
    assert engine._engine.scheduler.queue_rejections == 1


def test_engine_pre_stream_probe_noop_without_cap():
    from vllm_mlx.engine.batched import BatchedEngine

    engine = BatchedEngine.__new__(BatchedEngine)
    engine._engine = SimpleNamespace(
        scheduler=SimpleNamespace(queue_cap=0, waiting=[object()] * 5)
    )
    engine.raise_if_serialized_busy("req-abc")  # must not raise
    engine._engine = None
    engine.raise_if_serialized_busy("req-abc")  # no engine -> no-op


def test_pre_stream_probe_signature_matches_server_contract():
    """server._probe_engine_busy calls probe(request_id) positionally on
    BOTH engines; a signature drift is a 500 on every streaming request."""
    import inspect

    from vllm_mlx.engine.batched import BatchedEngine
    from vllm_mlx.engine.simple import SimpleEngine

    for cls in (BatchedEngine, SimpleEngine):
        sig = inspect.signature(cls.raise_if_serialized_busy)
        params = [p for p in sig.parameters.values() if p.name != "self"]
        assert params, f"{cls.__name__} probe takes no request_id"
        assert params[0].kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ), f"{cls.__name__} probe cannot take request_id positionally"


# --------------------------------------------------------- observability


def test_stats_carry_type_and_new_counters():
    kv = BatchedSystemKV(_FakeModel())
    s = kv.stats()
    assert s["type"] == "batched_system_kv"
    for key in ("grown_stores", "boundary_stores", "ssd_promotes"):
        assert key in s
