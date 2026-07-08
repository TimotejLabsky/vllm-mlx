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
    for key in (
        "grown_stores",
        "boundary_stores",
        "ssd_promotes",
        "pressure_evictions",
        "pressure_skipped_stores",
    ):
        assert key in s


# ------------------------------------------------- dynamic concurrency (#40)


def test_kv_budget_defers_when_total_over(monkeypatch):
    monkeypatch.delenv("VLLM_MLX_BATCHED_PAD_WASTE_MB", raising=False)
    monkeypatch.setenv("VLLM_MLX_BATCHED_KV_BUDGET_MB", "1")
    kv = BatchedSystemKV(_FakeModel())
    kv.store("r1", TOKENS, _donor_at(len(TOKENS)))

    # two seats at ~500K padded length blows a 1MB budget at test bytes/token
    scheduler = _guard_scheduler(kv, [500_000])
    assert bkv.should_defer_cobatch(scheduler, _req("c", 400_000)) is True
    assert kv.admission_deferrals == 1
    assert kv.stats()["admission_deferrals"] == 1

    # short contexts fit the same budget -> seats float upward
    scheduler = _guard_scheduler(kv, [500, 400, 300])
    assert bkv.should_defer_cobatch(scheduler, _req("c2", 450)) is False


def test_kv_budget_inert_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_MLX_BATCHED_KV_BUDGET_MB", raising=False)
    monkeypatch.delenv("VLLM_MLX_BATCHED_PAD_WASTE_MB", raising=False)
    kv = BatchedSystemKV(_FakeModel())
    kv.store("r1", TOKENS, _donor_at(len(TOKENS)))
    scheduler = _guard_scheduler(kv, [500_000])
    assert bkv.should_defer_cobatch(scheduler, _req("c", 400_000)) is False


def test_memory_watermark_backstop(monkeypatch):
    import mlx.core as mx

    monkeypatch.setenv("VLLM_MLX_BATCHED_MEM_WATERMARK_PCT", "90")
    kv = BatchedSystemKV(_FakeModel())  # cold cache: watermark must still work
    scheduler = _guard_scheduler(kv, [500])

    monkeypatch.setattr(mx, "get_active_memory", lambda: 95)
    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": 100}
    )
    assert bkv.should_defer_cobatch(scheduler, _req("c", 100)) is True

    monkeypatch.setattr(mx, "get_active_memory", lambda: 50)
    assert bkv.should_defer_cobatch(scheduler, _req("c2", 100)) is False


# --------------------------------------------------- pressure relief (#48)


DISJOINT = list(range(5000, 5800))  # no shared prefix with TOKENS


def _mb(n):
    return n * 1024 * 1024


def _watermarked(monkeypatch, active_mb, ceiling_mb=100):
    """A cache with the watermark at 90% of a ``ceiling_mb`` ceiling and a
    mocked allocator: returns ``(kv, mem)`` where ``mem['active']`` /
    ``mem['peak']`` (bytes) can be mutated between calls. reset_peak
    collapses peak to active, mirroring the real semantics."""
    import mlx.core as mx

    monkeypatch.setenv("VLLM_MLX_BATCHED_MEM_WATERMARK_PCT", "90")
    mem = {"active": _mb(active_mb), "peak": _mb(active_mb)}
    monkeypatch.setattr(mx, "get_active_memory", lambda: mem["active"])
    monkeypatch.setattr(mx, "get_peak_memory", lambda: mem["peak"])
    monkeypatch.setattr(
        mx, "reset_peak_memory", lambda: mem.__setitem__("peak", mem["active"])
    )
    monkeypatch.setattr(
        mx,
        "device_info",
        lambda: {"max_recommended_working_set_size": _mb(ceiling_mb)},
    )
    return BatchedSystemKV(_FakeModel()), mem


def test_relief_inert_without_watermark(monkeypatch):
    monkeypatch.delenv("VLLM_MLX_BATCHED_MEM_WATERMARK_PCT", raising=False)
    kv = BatchedSystemKV(_FakeModel())
    kv.store("r1", TOKENS, _donor_at(len(TOKENS)))
    assert kv.relieve_pressure() == 0
    assert kv.stats()["entry_count"] == 1
    assert kv.pressure_evictions == 0


def test_relief_needs_a_peak_spike(monkeypatch):
    # active and peak both under the 90 MB threshold -> nothing happens
    kv, mem = _watermarked(monkeypatch, active_mb=50)
    kv.store("r1", TOKENS, _donor_at(len(TOKENS)))
    assert kv.relieve_pressure() == 0
    assert kv.stats()["entry_count"] == 1


def test_relief_triggers_on_intra_step_peak(monkeypatch):
    """The 2026-07-09 live finding: prefill-chunk transients spike PAST the
    threshold and are freed before the inter-chunk check — the trigger
    must be the peak since the last check, not instantaneous active."""
    kv, mem = _watermarked(monkeypatch, active_mb=50)
    kv.store("r1", TOKENS, _donor_at(len(TOKENS)))
    kv.store("r2", DISJOINT, _donor_at(len(DISJOINT)))

    mem["peak"] = _mb(95)  # a chunk spiked over; active already back at 50
    assert kv.relieve_pressure() == 1  # active <= threshold stops after one
    assert kv.pressure_evictions == 1
    assert kv.stats()["entry_count"] == 1
    # LRU-first: the older TOKENS entry went, the newer chain survives
    assert next(iter(kv._entries.values()))["tokens"] == DISJOINT
    assert kv.stats()["evictions"] == 1  # counted in the existing gauge too
    assert mem["peak"] == mem["active"]  # window reset for the next step


def test_relief_empties_bag_under_sustained_pressure(monkeypatch):
    kv, mem = _watermarked(monkeypatch, active_mb=50)
    kv.store("r1", TOKENS, _donor_at(len(TOKENS)))
    kv.store("r2", DISJOINT, _donor_at(len(DISJOINT)))

    mem["active"] = _mb(95)
    mem["peak"] = _mb(96)
    assert kv.relieve_pressure() == 2  # active never clears -> whole bag
    assert kv.stats()["entry_count"] == 0
    assert kv.stats()["pressure_evictions"] == 2


def test_boundary_store_skipped_under_pressure(monkeypatch):
    kv, mem = _watermarked(monkeypatch, active_mb=95)
    assert kv.store_prompt_boundary("r1", TOKENS, _donor_at(len(TOKENS))) is False
    assert kv.boundary_stores == 0
    assert kv.stats()["entry_count"] == 0
    assert kv.stats()["pressure_skipped_stores"] == 1


def test_final_store_skipped_under_pressure_without_donor(monkeypatch):
    kv, mem = _watermarked(monkeypatch, active_mb=95)
    assert kv.store("r1", TOKENS, _donor_at(len(TOKENS))) is False
    assert kv.stats()["entry_count"] == 0
    assert kv.stats()["pressure_skipped_stores"] == 1


def test_final_store_skipped_when_the_copy_would_overshoot(monkeypatch):
    """The 2026-07-09 live finding (second half): a deep final store passed
    the instantaneous gate at 48.9 GB — after the batch KV was freed — and
    then materialized a 7 GB copy. The gate must price the copy in."""
    kv, mem = _watermarked(monkeypatch, active_mb=5, ceiling_mb=10)
    kv.store("r1", TOKENS, _donor_at(len(TOKENS)))  # seeds bytes/token
    bpt = kv.bytes_per_token()
    assert bpt > 0

    # a chain whose snapshot alone crosses the 9 MB threshold from 5 MB
    need = int(_mb(5) / bpt)
    big = list(range(100_000, 100_000 + need))
    assert kv.store("rbig", big, _donor_at(need)) is False
    assert kv.stats()["pressure_skipped_stores"] == 1
    assert kv.stats()["entry_count"] == 1  # the seed entry is untouched


def test_overshoot_gate_survives_an_emptied_bag(monkeypatch):
    """Live 2026-07-09 round-2 finding: relief evicts the WHOLE bag mid-
    prefill, so the deep final store that follows priced itself against a
    bytes/token of 0 and landed a 7 GB copy. The estimate must persist."""
    kv, mem = _watermarked(monkeypatch, active_mb=5, ceiling_mb=10)
    kv.store("r1", TOKENS, _donor_at(len(TOKENS)))
    # learned at INSERT, not lazily on read — a serial workload never reads
    # bytes/token before relief empties the bag (live round 3)
    assert kv._bpt_hint > 0
    bpt = kv.bytes_per_token()

    mem["peak"] = _mb(15)
    mem["active"] = _mb(9.5)  # stays over -> relief empties the bag
    assert kv.relieve_pressure() == 1
    assert kv.stats()["entry_count"] == 0

    mem["active"] = _mb(5)  # batch KV freed; instantaneous gate would pass
    mem["peak"] = _mb(5)
    need = int(_mb(5) / bpt)
    big = list(range(100_000, 100_000 + need))
    assert kv.store("rbig", big, _donor_at(need)) is False  # hint held
    assert kv.stats()["pressure_skipped_stores"] == 1


def test_final_store_grows_under_pressure_with_donor(monkeypatch):
    # seed + restore while clear; pressure arrives before the final store
    kv, mem = _watermarked(monkeypatch, active_mb=50)
    kv.store("r1", TOKENS, _donor_at(len(TOKENS)))
    assert kv.fetch(TOKENS + [7, 8], request_id="r2") is not None

    mem["active"] = _mb(95)
    grown_tokens = TOKENS + [7, 8, 9, 10]
    assert kv.store("r2", grown_tokens, _donor_at(len(grown_tokens))) is True
    assert kv.stats()["grown_stores"] == 1  # O(delta), safe at peak
    assert kv.stats()["pressure_skipped_stores"] == 0


def test_step_hook_wiring_and_none_safety():
    bkv.maybe_relieve_pressure(SimpleNamespace(hybrid_kv=None))  # no-op

    kv = MagicMock()
    bkv.maybe_relieve_pressure(SimpleNamespace(hybrid_kv=kv))
    kv.relieve_pressure.assert_called_once()

    kv.relieve_pressure.side_effect = RuntimeError("metal")
    bkv.maybe_relieve_pressure(SimpleNamespace(hybrid_kv=kv))  # swallowed


def test_scheduler_step_relieves_pressure(monkeypatch):
    scheduler = _capped_scheduler(monkeypatch, None)
    scheduler.hybrid_kv = MagicMock()
    scheduler.hybrid_kv.has_ssd = False
    scheduler.step()
    scheduler.hybrid_kv.relieve_pressure.assert_called_once()
