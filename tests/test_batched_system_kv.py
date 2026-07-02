"""Hybrid-safe checkpoint prefix cache for the batched LLM scheduler
(PATCHES.md #34 — the item-B port).

Unit level uses REAL mlx-lm cache classes (KVCache + ArraysCache) so the
slice/apply semantics are the ones production hits. Wiring level drives the
scheduler hooks with mocks.
"""

from unittest.mock import MagicMock

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache

from vllm_mlx.batched_system_kv import BatchedSystemKV, batched_system_kv_enabled

ENABLE_ENV = "VLLM_MLX_BATCHED_SYSTEM_KV"


class _FakeModel:
    """make_prompt_cache defers to model.make_cache — 2 KV + 1 recurrent."""

    def make_cache(self):
        return [KVCache(), ArraysCache(size=2), KVCache()]


def _fill_kv(cache: KVCache, n: int, seed: float = 0.0):
    keys = mx.arange(n, dtype=mx.float32).reshape(1, 1, n, 1) + seed
    values = keys + 0.5
    cache.update_and_fetch(keys, values)


def _fill_recurrent(cache: ArraysCache, pos: int, seed: float = 0.0):
    cache[0] = mx.full((1, 4), float(pos) + seed)
    cache[1] = mx.full((1, 2, 3), float(pos) * 2 + seed)


def _donor_at(pos: int, seed: float = 0.0):
    """A hybrid per-layer cache list as it would look after `pos` tokens."""
    kv1, rec, kv2 = KVCache(), ArraysCache(size=2), KVCache()
    _fill_kv(kv1, pos, seed)
    _fill_kv(kv2, pos, seed + 100)
    _fill_recurrent(rec, pos, seed)
    return [kv1, rec, kv2]


def _make_cache(monkeypatch=None, **env):
    if monkeypatch is not None:
        for k, v in env.items():
            monkeypatch.setenv(k, str(v))
    return BatchedSystemKV(_FakeModel())


TOKENS = list(range(1000, 1800))  # 800-token entry sequence


def test_enable_env_gate(monkeypatch):
    monkeypatch.delenv(ENABLE_ENV, raising=False)
    assert not batched_system_kv_enabled()
    monkeypatch.setenv(ENABLE_ENV, "1")
    assert batched_system_kv_enabled()


def test_full_prefix_extension_restores_whole_snapshot():
    """New prompt exactly extends the stored chain -> restore at donor_len."""
    kv = _make_cache()
    kv.store("r1", TOKENS, _donor_at(len(TOKENS)))

    result = kv.fetch(TOKENS + [7, 8, 9])
    assert result is not None
    cache, remaining, pos = result
    assert pos == len(TOKENS)
    assert remaining == [7, 8, 9]
    assert isinstance(cache[0], KVCache) and cache[0].offset == len(TOKENS)
    assert isinstance(cache[1], ArraysCache)
    assert mx.array_equal(cache[1][0], mx.full((1, 4), float(len(TOKENS))))
    assert kv.hits == 1 and kv.misses == 0 and kv.partial_hits == 0


def test_divergent_chain_restores_at_checkpoint():
    """Divergence mid-chain -> nearest recurrent checkpoint <= divergence,
    attention KV sliced to the same position."""
    kv = _make_cache()
    kv.note_scheduled("r1", 0)
    kv.capture_segment("r1", 448, _donor_at(448))
    kv.store("r1", TOKENS, _donor_at(len(TOKENS)))

    divergent = TOKENS[:600] + [1, 2, 3, 4]
    result = kv.fetch(divergent)
    assert result is not None
    cache, remaining, pos = result
    assert pos == 448
    assert remaining == divergent[448:]
    assert cache[0].offset == 448
    assert cache[0].state[0].shape[2] == 448  # sliced, not trimmed
    assert mx.array_equal(cache[1][0], mx.full((1, 4), 448.0))
    assert kv.partial_hits == 1
    assert kv.partial_tokens_saved == 448


def test_hybrid_without_checkpoint_misses_on_divergence():
    """No ladder -> a divergent match cannot restore recurrent state."""
    kv = _make_cache()
    kv.store("r1", TOKENS, _donor_at(len(TOKENS)))

    assert kv.fetch(TOKENS[:600] + [1, 2, 3]) is None
    assert kv.misses == 1


def test_short_lcp_below_floor_misses():
    kv = _make_cache()
    kv.store("r1", TOKENS, _donor_at(len(TOKENS)))
    assert kv.fetch(TOKENS[:100] + [5] * 500) is None


def test_lru_eviction_by_slots(monkeypatch):
    monkeypatch.setenv("VLLM_MLX_SYSTEM_KV_SLOTS", "2")
    kv = _make_cache()
    kv.store("r1", TOKENS, _donor_at(300))
    kv.store("r2", [2] * 700, _donor_at(300))
    kv.store("r3", [3] * 700, _donor_at(300))
    assert kv.evictions == 1
    assert kv.stats()["entry_count"] == 2
    # r1 (oldest) evicted: its exact extension now misses
    assert kv.fetch(TOKENS + [1]) is None


def test_capture_segment_skips_pure_attention():
    kv = _make_cache()
    kv.note_scheduled("r1", 0)
    pure = [KVCache(), KVCache()]
    _fill_kv(pure[0], 300)
    _fill_kv(pure[1], 300)
    kv.capture_segment("r1", 300, pure)
    assert kv._pending == {}


def test_pure_attention_entry_restores_at_any_divergence():
    """All-trim donors need no ladder — any position slices."""

    class _PureModel:
        def make_cache(self):
            return [KVCache(), KVCache()]

    kv = BatchedSystemKV(_PureModel())
    donor = [KVCache(), KVCache()]
    _fill_kv(donor[0], 800)
    _fill_kv(donor[1], 800, 100)
    kv.store("r1", TOKENS, donor)

    result = kv.fetch(TOKENS[:600] + [1, 2, 3])
    assert result is not None
    cache, remaining, pos = result
    assert pos == 600
    assert cache[0].offset == 600


def test_base_pos_offsets_checkpoint_positions():
    """A restored request's segment positions are relative to its inserted
    tokens; note_scheduled anchors them to absolute positions."""
    kv = _make_cache()
    kv.note_scheduled("r1", 500)
    kv.capture_segment("r1", 100, _donor_at(600))
    assert kv._pending["r1"][0]["pos"] == 600


def test_restored_request_inherits_donor_ladder():
    """A hit seeds the new request's ladder with the donor's checkpoints,
    so the chain it eventually stores stays divergence-restorable even when
    it prefills almost nothing itself (exact re-send)."""
    kv = _make_cache()
    kv.note_scheduled("r1", 0)
    kv.capture_segment("r1", 448, _donor_at(448))
    kv.store("r1", TOKENS, _donor_at(len(TOKENS)))

    result = kv.fetch(TOKENS + [7, 8], request_id="r2")
    assert result is not None
    _, _, pos = result
    ladder = kv._pending["r2"]
    assert [cp["pos"] for cp in ladder] == [448, pos]

    # r2's stored chain (a superset) restores a divergent fetch at 448.
    kv.store("r2", TOKENS + [7, 8, 9], _donor_at(len(TOKENS) + 3))
    result = kv.fetch(TOKENS[:600] + [1, 2, 3], request_id="r3")
    assert result is not None
    assert result[2] == 448


def test_duplicate_chain_store_replaces_and_merges_ladders():
    """Re-running an identical chain must replace the old entry (no slot
    burn) and keep the union of both ladders — an exact re-send whose own
    ladder is poor must not shadow the checkpoint-rich original."""
    kv = _make_cache()
    kv.note_scheduled("r1", 0)
    kv.capture_segment("r1", 256, _donor_at(256))
    kv.store("r1", TOKENS, _donor_at(len(TOKENS)))

    kv.note_scheduled("r2", 0)
    kv.capture_segment("r2", 512, _donor_at(512))
    kv.store("r2", TOKENS, _donor_at(len(TOKENS)))

    assert kv.stats()["entry_count"] == 1
    entry = next(iter(kv._entries.values()))
    assert [cp["pos"] for cp in entry["checkpoints"]] == [256, 512]


def test_prompt_boundary_store_keeps_pending_and_final_store_absorbs():
    """The boundary store copies the ladder (generation still owns it) and
    the finished chain absorbs the boundary entry via prefix subsumption —
    one entry at the end, carrying the merged ladder."""
    kv = _make_cache()
    prompt = TOKENS[:600]
    kv.note_scheduled("r1", 0)
    kv.capture_segment("r1", 448, _donor_at(448))

    assert kv.store_prompt_boundary("r1", prompt, _donor_at(600)) is True
    assert kv.stats()["entry_count"] == 1
    assert [cp["pos"] for cp in kv._pending["r1"]] == [448]  # not popped

    kv.store("r1", TOKENS, _donor_at(len(TOKENS)))
    assert kv.stats()["entry_count"] == 1  # boundary entry absorbed
    entry = next(iter(kv._entries.values()))
    assert entry["tokens"] == TOKENS
    assert 448 in [cp["pos"] for cp in entry["checkpoints"]]
    assert "r1" not in kv._pending


def test_prompt_boundary_entry_serves_aborted_chain():
    """An aborted request (no final store) still leaves its prompt prefill
    behind — the point of the feature."""
    kv = _make_cache()
    prompt = TOKENS[:600]
    kv.note_scheduled("r1", 0)
    kv.capture_segment("r1", 448, _donor_at(448))
    kv.store_prompt_boundary("r1", prompt, _donor_at(600))
    kv.discard_pending("r1")  # abort path

    result = kv.fetch(prompt + [55, 56, 57, 58])
    assert result is not None
    _, remaining, pos = result
    assert pos == 600
    assert remaining == [55, 56, 57, 58]
    assert kv.stats()["boundary_stores"] == 1


def test_prompt_boundary_store_skipped_below_min_new_content():
    """A near-full restore adds nothing worth an entry — the donor already
    covers the chain."""
    kv = _make_cache()
    kv.note_scheduled("r1", 590)
    assert kv.store_prompt_boundary("r1", TOKENS[:600], _donor_at(600)) is False
    assert kv.stats()["entry_count"] == 0


def test_discard_pending_on_abort():
    kv = _make_cache()
    kv.note_scheduled("r1", 0)
    kv.capture_segment("r1", 448, _donor_at(448))
    kv.discard_pending("r1")
    assert kv._pending == {}
    assert kv._base_pos == {}


def test_fetch_returns_cross_thread_evaluable_arrays():
    """fetch() runs on the event-loop thread; the batch evaluates on the
    executor thread. The restored cache must carry REALIZED buffers, not
    lazy slice graphs — patch #28's crash class, found live in the Studio
    A/B (the worker step died, engine_core self-healed to model-thread
    stepping, and the request silently re-prefilled cold)."""
    import threading

    kv = _make_cache()
    kv.note_scheduled("r1", 0)
    kv.capture_segment("r1", 448, _donor_at(448))
    kv.store("r1", TOKENS, _donor_at(len(TOKENS)))

    result = kv.fetch(TOKENS[:600] + [1, 2, 3])
    assert result is not None
    cache = result[0]

    errors = []

    def cross_thread_eval():
        try:
            for c in cache:
                st = c.state
                items = st if isinstance(st, (list, tuple)) else [st]
                mx.eval([a for a in items if a is not None])
        except RuntimeError as e:  # pragma: no cover - the regression itself
            errors.append(e)

    t = threading.Thread(target=cross_thread_eval)
    t.start()
    t.join()
    assert not errors, f"restored cache not realized on fetch thread: {errors[0]}"


def test_rotating_kv_checkpoints_restore_and_realize():
    """Sliding-window models (gpt-oss, gemma text) carry RotatingKVCache —
    ckpt-class like recurrent layers, but with TUPLE states. capture must
    eval tuples too (found live: lazy Rotating states crossed to the fetch
    thread and died), and restore must round-trip state+meta at checkpoint
    positions."""
    import threading

    from mlx_lm.models.cache import RotatingKVCache

    class _RotModel:
        def make_cache(self):
            return [KVCache(), RotatingKVCache(max_size=128)]

    def rot_donor(pos):
        kv, rot = KVCache(), RotatingKVCache(max_size=128)
        _fill_kv(kv, pos)
        rot.update_and_fetch(
            mx.arange(pos, dtype=mx.float32).reshape(1, 1, pos, 1),
            mx.arange(pos, dtype=mx.float32).reshape(1, 1, pos, 1) + 0.5,
        )
        return [kv, rot]

    kv = BatchedSystemKV(_RotModel())
    kv.note_scheduled("r1", 0)
    kv.capture_segment("r1", 448, rot_donor(448))
    kv.store("r1", TOKENS, rot_donor(len(TOKENS)))

    result = kv.fetch(TOKENS[:600] + [1, 2, 3])
    assert result is not None
    cache, remaining, pos = result
    assert pos == 448
    assert isinstance(cache[1], RotatingKVCache)

    errors = []

    def cross_thread_eval():
        try:
            for c in cache:
                st = c.state
                items = st if isinstance(st, (list, tuple)) else [st]
                mx.eval([a for a in items if a is not None])
        except RuntimeError as e:  # pragma: no cover - the regression itself
            errors.append(e)

    t = threading.Thread(target=cross_thread_eval)
    t.start()
    t.join()
    assert not errors, f"Rotating checkpoint state not realized: {errors[0]}"


def test_stats_shape():
    kv = _make_cache()
    kv.store("r1", TOKENS, _donor_at(300))
    s = kv.stats()
    for key in (
        "enabled", "hits", "misses", "hit_rate", "partial_hits",
        "tokens_saved", "partial_tokens_saved", "evictions",
        "entry_count", "capacity", "memory_mb",
    ):
        assert key in s
    assert s["enabled"] is True
    assert s["entry_count"] == 1
    assert s["memory_mb"] > 0


# ------------------------------------------------------------ wiring level


def _make_scheduler(monkeypatch):
    from vllm_mlx.scheduler import Scheduler, SchedulerConfig

    monkeypatch.setenv(ENABLE_ENV, "1")
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.encode = lambda x: list(range(len(x.split())))
    tokenizer.eos_token_id = 0
    config = SchedulerConfig(max_num_seqs=4, enable_prefix_cache=True)
    return Scheduler(model, tokenizer, config)


def test_scheduler_init_replaces_memory_aware_cache(monkeypatch):
    scheduler = _make_scheduler(monkeypatch)
    assert scheduler.hybrid_kv is not None
    assert scheduler.memory_aware_cache is None
    assert scheduler.block_aware_cache is None


def test_add_request_fetch_sets_restore_fields(monkeypatch):
    from vllm_mlx.request import Request, SamplingParams

    scheduler = _make_scheduler(monkeypatch)
    sentinel_cache = ["cache"]
    scheduler.hybrid_kv = MagicMock()
    scheduler.hybrid_kv.fetch.return_value = (sentinel_cache, [42, 43], 598)

    request = Request(
        request_id="req-1",
        prompt=list(range(600)),
        sampling_params=SamplingParams(max_tokens=8),
    )
    scheduler.add_request(request)

    assert request.cache_hit_type == "system_kv"
    assert request.prompt_cache is sentinel_cache
    assert request.cached_tokens == 598
    assert request.remaining_tokens == [42, 43]


def test_schedule_waiting_uses_insert_segments(monkeypatch):
    from vllm_mlx.request import Request, SamplingParams

    scheduler = _make_scheduler(monkeypatch)
    scheduler.hybrid_kv.ckpt_interval = 256

    params = SamplingParams(max_tokens=8)
    request = Request(
        request_id="req-1",
        prompt="x",
        sampling_params=params,
        prompt_token_ids=list(range(600)),
        num_prompt_tokens=600,
    )
    scheduler.waiting.append(request)
    scheduler._current_sampler_params = (
        params.temperature, params.top_p, params.min_p,
    )
    scheduler.batch_generator = MagicMock()
    scheduler.batch_generator.insert_segments.return_value = [11]

    scheduled = scheduler._schedule_waiting()

    assert len(scheduled) == 1
    scheduler.batch_generator.insert.assert_not_called()
    segments = scheduler.batch_generator.insert_segments.call_args.args[0][0]
    assert [len(s) for s in segments] == [256, 256, 88]
    assert scheduler.hybrid_kv._base_pos["req-1"] == 0


def test_capture_hybrid_checkpoints_extracts_at_segment_end(monkeypatch):
    from types import SimpleNamespace

    scheduler = _make_scheduler(monkeypatch)
    scheduler.hybrid_kv = MagicMock()
    scheduler.uid_to_request_id[11] = "req-1"
    cache_list = ["layer0"]
    scheduler.batch_generator = MagicMock()
    scheduler.batch_generator.extract_cache.return_value = {
        11: (cache_list, [1, 2, 3])
    }

    responses = [
        SimpleNamespace(uid=11, end_of_segment=False, end_of_prompt=False,
                        progress=(128, 600)),
        SimpleNamespace(uid=11, end_of_segment=True, end_of_prompt=False,
                        progress=(256, 600)),
    ]
    scheduler._capture_hybrid_checkpoints(responses)

    scheduler.hybrid_kv.capture_segment.assert_called_once_with(
        "req-1", 256, cache_list
    )
    scheduler.hybrid_kv.store_prompt_boundary.assert_not_called()


def test_capture_hybrid_checkpoints_boundary_store_at_end_of_prompt(monkeypatch):
    """The end_of_prompt response also persists the prompt-boundary entry
    (patch #35 abort resilience)."""
    from types import SimpleNamespace

    from vllm_mlx.request import Request, SamplingParams

    scheduler = _make_scheduler(monkeypatch)
    scheduler.hybrid_kv = MagicMock()
    scheduler.uid_to_request_id[11] = "req-1"
    request = Request(
        request_id="req-1",
        prompt="x",
        sampling_params=SamplingParams(max_tokens=8),
        prompt_token_ids=list(range(600)),
        num_prompt_tokens=600,
    )
    scheduler.requests["req-1"] = request
    cache_list = ["layer0"]
    scheduler.batch_generator = MagicMock()
    scheduler.batch_generator.extract_cache.return_value = {
        11: (cache_list, [1, 2, 3])
    }

    scheduler._capture_hybrid_checkpoints(
        [SimpleNamespace(uid=11, end_of_segment=True, end_of_prompt=True,
                         progress=(600, 600))]
    )

    scheduler.hybrid_kv.capture_segment.assert_called_once_with(
        "req-1", 600, cache_list
    )
    scheduler.hybrid_kv.store_prompt_boundary.assert_called_once_with(
        "req-1", request.prompt_token_ids, cache_list
    )


def test_boundary_store_skipped_under_concurrent_load(monkeypatch):
    """Under a concurrent burst the boundary snapshot's cost lands inside
    the busy step loop (bench-measured TTFT inflation at conc=4) — solo
    requests only; concurrent chains still store at finish."""
    from types import SimpleNamespace

    scheduler = _make_scheduler(monkeypatch)
    scheduler.hybrid_kv = MagicMock()
    scheduler.uid_to_request_id[11] = "req-1"
    scheduler.running["req-1"] = object()
    scheduler.running["req-2"] = object()
    scheduler.batch_generator = MagicMock()
    scheduler.batch_generator.extract_cache.return_value = {11: (["l0"], [1])}

    scheduler._capture_hybrid_checkpoints(
        [SimpleNamespace(uid=11, end_of_segment=True, end_of_prompt=True,
                         progress=(600, 600))]
    )

    scheduler.hybrid_kv.capture_segment.assert_called_once()
    scheduler.hybrid_kv.store_prompt_boundary.assert_not_called()


def test_cleanup_finished_stores_into_hybrid_kv(monkeypatch):
    from vllm_mlx.request import Request, RequestStatus, SamplingParams

    scheduler = _make_scheduler(monkeypatch)
    scheduler.hybrid_kv = MagicMock()
    scheduler.hybrid_kv.store.return_value = True

    request = Request(
        request_id="req-1",
        prompt="x",
        sampling_params=SamplingParams(max_tokens=8),
        prompt_token_ids=[1, 2, 3],
        num_prompt_tokens=3,
    )
    request.output_token_ids = [4, 5]
    request._extracted_cache = ["layer0"]
    request.status = RequestStatus.FINISHED_STOPPED
    scheduler.running["req-1"] = request
    scheduler.requests["req-1"] = request

    scheduler._cleanup_finished({"req-1"})

    scheduler.hybrid_kv.store.assert_called_once_with(
        "req-1", [1, 2, 3, 4, 5], ["layer0"]
    )
    assert request._extracted_cache is None


def test_get_stats_emits_system_kv_cache_block(monkeypatch):
    scheduler = _make_scheduler(monkeypatch)
    stats = scheduler.get_stats()
    assert "system_kv_cache" in stats
    assert stats["system_kv_cache"]["enabled"] is True
    assert "memory_aware_cache" not in stats
