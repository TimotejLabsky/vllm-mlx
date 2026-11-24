"""Hybrid-safe checkpoint prefix cache for the batched LLM scheduler
(PATCHES.md #34 — the item-B port).

Unit level uses REAL mlx-lm cache classes (KVCache + ArraysCache) so the
slice/apply semantics are the ones production hits. Wiring level drives the
scheduler hooks with mocks.
"""

from unittest.mock import MagicMock

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache

from vllm_mlx.batched_system_kv import BatchedSystemKV, batched_system_kv_enabled
from vllm_mlx.cache_state_compat import legacy_state

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


# ------------------------------------------------------------ grow-on-HIT


def test_grown_store_shares_donor_arrays():
    """A chain that continues its donor must reference the donor's trim
    segments (zero copy for the shared prefix) instead of re-materializing
    the whole context — SimpleEngine's grow-on-HIT economics (patch #37)."""
    kv = _make_cache()
    kv.store("r1", TOKENS, _donor_at(len(TOKENS)))
    donor_k = next(iter(kv._entries.values()))["snapshot"][0][0][0]

    assert kv.fetch(TOKENS + [7, 8], request_id="r2") is not None
    grown_tokens = TOKENS + [7, 8, 9, 10]
    kv.store("r2", grown_tokens, _donor_at(len(grown_tokens)))

    assert kv.stats()["grown_stores"] == 1
    assert kv.stats()["entry_count"] == 1  # donor absorbed
    entry = next(iter(kv._entries.values()))
    segments = entry["snapshot"][0]
    assert len(segments) == 2
    assert segments[0][0] is donor_k  # shared BY REFERENCE, not copied
    assert segments[1][0].shape[2] == 4  # O(delta) slice only

    # The segmented entry restores correctly at a checkpoint-free position
    # (pure attention slice across segment boundary).
    result = kv.fetch(grown_tokens + [99])
    assert result is not None
    cache, remaining, pos = result
    assert pos == len(grown_tokens)
    assert cache[0].offset == len(grown_tokens)


def test_grow_after_divergent_restore_reuses_prefix_only():
    """A divergent hit grows from the donor's common prefix: whole donor
    segments by reference up to the divergence, then the new delta."""
    kv = _make_cache()
    kv.note_scheduled("r1", 0)
    kv.capture_segment("r1", 448, _donor_at(448))
    kv.store("r1", TOKENS, _donor_at(len(TOKENS)))

    divergent = TOKENS[:600] + [1, 2, 3, 4]
    assert kv.fetch(divergent, request_id="r2") is not None  # restores at 448
    kv.store("r2", divergent, _donor_at(len(divergent)))

    assert kv.stats()["grown_stores"] == 1
    assert kv.stats()["entry_count"] == 2  # divergent chain != prefix, no absorb
    grown = next(
        e for e in kv._entries.values() if e["tokens"] == divergent
    )
    segments = grown["snapshot"][0]
    # prefix reuse bounded by the common prefix (600), delta covers the rest
    assert sum(k.shape[2] for k, _ in segments) == len(divergent)
    assert segments[-1][0].shape[2] == len(divergent) - 600

    result = kv.fetch(divergent + [55])
    assert result is not None
    assert result[2] == len(divergent)
    # KV content across the segment seam must match a straight donor
    control = _donor_at(len(divergent))
    assert mx.array_equal(legacy_state(result[0][0])[0], legacy_state(control[0])[0])


def test_boundary_final_grow_cascade():
    """fetch -> boundary store (absorbs donor) -> final store must cascade
    the donor linkage so the final store still grows (from the boundary
    entry), sharing arrays end to end."""
    kv = _make_cache()
    kv.store("r1", TOKENS[:400], _donor_at(400))
    donor_k = next(iter(kv._entries.values()))["snapshot"][0][0][0]

    assert kv.fetch(TOKENS, request_id="r3") is not None  # restore at 400
    kv.note_scheduled("r3", 400)
    # prompt boundary at 700: 300 new tokens >= partial_min -> stored + grown
    assert kv.store_prompt_boundary("r3", TOKENS[:700], _donor_at(700)) is True
    kv.store("r3", TOKENS, _donor_at(len(TOKENS)))

    assert kv.stats()["grown_stores"] == 2  # boundary grew AND final grew
    assert kv.stats()["entry_count"] == 1
    entry = next(iter(kv._entries.values()))
    segments = entry["snapshot"][0]
    assert segments[0][0] is donor_k  # original donor array survives the cascade
    assert [k.shape[2] for k, _ in segments] == [400, 300, 100]
    assert sum(k.shape[2] for k, _ in segments) == len(TOKENS)


def test_grown_entries_skip_spill(monkeypatch, tmp_path):
    """Grown entries don't re-spill (SimpleEngine policy: the stored prefix
    promotes on restart and re-grows cheaply)."""
    import os

    kv = _make_ssd_cache(monkeypatch, tmp_path)
    kv.store("r1", TOKENS, _donor_at(len(TOKENS)))
    assert kv.fetch(TOKENS + [7, 8], request_id="r2") is not None
    kv.store("r2", TOKENS + [7, 8, 9], _donor_at(len(TOKENS) + 3))
    kv.close()  # drain writer

    data_dir = None
    for root, dirs, _files in os.walk(tmp_path):
        if root.endswith("/data"):
            data_dir = root
            break
    assert data_dir is not None
    assert len(os.listdir(data_dir)) == 1  # only the original chain spilled


# --------------------------------------------------------------- ssd tier


def _make_ssd_cache(monkeypatch, tmp_path):
    from types import SimpleNamespace as NS

    monkeypatch.setenv("VLLM_MLX_SSD_SYSTEM_KV_DIR", str(tmp_path))
    return BatchedSystemKV(
        _FakeModel(), tokenizer=NS(name_or_path="unit/test-model")
    )


def test_ssd_disabled_without_env(monkeypatch):
    monkeypatch.delenv("VLLM_MLX_SSD_SYSTEM_KV_DIR", raising=False)
    kv = _make_cache()
    assert kv.has_ssd is False
    assert kv.check_ssd(TOKENS) is None


def test_ssd_spill_and_promote_across_restart(monkeypatch, tmp_path):
    """Full chain spills write-through; a fresh instance (restart) promotes
    it from disk and restores bit-exactly."""
    kv1 = _make_ssd_cache(monkeypatch, tmp_path)
    assert kv1.has_ssd
    kv1.note_scheduled("r1", 0)
    kv1.capture_segment("r1", 448, _donor_at(448))
    kv1.store("r1", TOKENS, _donor_at(len(TOKENS)))
    kv1.close()  # drain writer — simulates process exit

    kv2 = _make_ssd_cache(monkeypatch, tmp_path)
    assert kv2.stats()["entry_count"] == 0  # RAM is cold
    candidate = kv2.check_ssd(TOKENS + [7, 8])
    assert candidate is not None
    assert kv2.promote_ssd(candidate) is True

    result = kv2.fetch(TOKENS + [7, 8])
    assert result is not None
    cache, remaining, pos = result
    assert pos == len(TOKENS)
    assert remaining == [7, 8]
    assert mx.array_equal(cache[1][0], mx.full((1, 4), float(len(TOKENS))))
    assert kv2.stats()["ssd_promotes"] == 1


def test_ssd_shared_prefix_promote_restores_at_checkpoint(monkeypatch, tmp_path):
    """A divergent chain promotes via the shared-prefix index path and
    restores at the checkpoint — partial restore works across restarts."""
    kv1 = _make_ssd_cache(monkeypatch, tmp_path)
    kv1.note_scheduled("r1", 0)
    kv1.capture_segment("r1", 448, _donor_at(448))
    kv1.store("r1", TOKENS, _donor_at(len(TOKENS)))
    kv1.close()

    kv2 = _make_ssd_cache(monkeypatch, tmp_path)
    divergent = TOKENS[:600] + [1, 2, 3, 4]
    candidate = kv2.check_ssd(divergent)
    assert candidate is not None
    assert kv2.promote_ssd(candidate) is True

    result = kv2.fetch(divergent)
    assert result is not None
    assert result[2] == 448


def test_ssd_boundary_store_survives_restart(monkeypatch, tmp_path):
    """An aborted request's prompt-boundary entry persists — abort
    resilience holds ACROSS restarts, not just within a process."""
    kv1 = _make_ssd_cache(monkeypatch, tmp_path)
    prompt = TOKENS[:600]
    kv1.note_scheduled("r1", 0)
    kv1.capture_segment("r1", 448, _donor_at(448))
    kv1.store_prompt_boundary("r1", prompt, _donor_at(600))
    kv1.discard_pending("r1")  # abort — no final store
    kv1.close()

    kv2 = _make_ssd_cache(monkeypatch, tmp_path)
    candidate = kv2.check_ssd(prompt + [55, 56])
    assert candidate is not None
    assert kv2.promote_ssd(candidate) is True
    result = kv2.fetch(prompt + [55, 56])
    assert result is not None
    assert result[2] == 600


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
        "req-1", [1, 2, 3, 4, 5], ["layer0"], gen_start=3
    )
    assert request._extracted_cache is None


def test_concurrent_hybrid_requests_each_store_at_finish(monkeypatch):
    """Regression (#100): two hybrid requests finishing in the same step must
    BOTH leave an entry. The boundary store is solo-only, so under concurrency
    the finish store is the only path — and upstream #683's
    ``_prompt_output_entry_is_useless`` gate (non-trimmable => drop the
    prompt+output cache) silently killed it for every hybrid model, because
    ArraysCache is never trimmable. The hybrid bag restores via slices +
    checkpoints and never trims, so the extraction must be kept. Drives the
    REAL response path (the pre-existing cleanup test set _extracted_cache by
    hand and could not see the gate)."""
    from types import SimpleNamespace

    from vllm_mlx.request import Request, SamplingParams

    scheduler = _make_scheduler(monkeypatch)
    kv = scheduler.hybrid_kv
    assert isinstance(kv, BatchedSystemKV)
    scheduler._decode_tokens = lambda ids: ""

    responses = []
    for n, (rid, uid, base) in enumerate([("req-a", 11, 1000), ("req-b", 12, 5000)]):
        prompt = list(range(base, base + 800))
        request = Request(
            request_id=rid,
            prompt="x",
            sampling_params=SamplingParams(max_tokens=8),
            prompt_token_ids=prompt,
            num_prompt_tokens=len(prompt),
        )
        request.first_token_time = 0.0
        request.append_output_token(base + 900)  # past the prompt-only hook
        scheduler.running[rid] = request
        scheduler.requests[rid] = request
        scheduler.uid_to_request_id[uid] = rid
        cache = _donor_at(len(prompt) + 2, seed=float(n))
        responses.append(
            SimpleNamespace(
                uid=uid, token=base + 901, finish_reason="stop", prompt_cache=cache
            )
        )
    assert len(scheduler.running) == 2  # concurrent: boundary store is off

    _outputs, finished = scheduler._process_batch_responses(responses)
    assert finished == {"req-a", "req-b"}
    for rid in finished:
        assert (
            getattr(scheduler.running[rid], "_extracted_cache", None) is not None
        ), rid

    scheduler._cleanup_finished(finished)

    assert len(kv._entries) == 2
    stored = sorted(e["tokens"][0] for e in kv._entries.values())
    assert stored == [1000, 5000]
    assert kv.boundary_stores == 0 and kv.pressure_skipped_stores == 0


def test_add_request_marks_ssd_pending_on_miss(monkeypatch):
    from vllm_mlx.request import Request, SamplingParams

    scheduler = _make_scheduler(monkeypatch)
    scheduler.hybrid_kv = MagicMock()
    scheduler.hybrid_kv.fetch.return_value = None
    scheduler.hybrid_kv.check_ssd.return_value = {"tokens": (1,), "file_path": "x"}

    request = Request(
        request_id="req-1",
        prompt=list(range(600)),
        sampling_params=SamplingParams(max_tokens=8),
    )
    scheduler.add_request(request)

    assert request.cache_hit_type == "ssd_pending"
    assert request._ssd_candidate == {"tokens": (1,), "file_path": "x"}


def test_try_promote_hybrid_ssd_pending_restores_request(monkeypatch):
    from vllm_mlx.request import Request, SamplingParams

    scheduler = _make_scheduler(monkeypatch)
    scheduler.hybrid_kv = MagicMock()
    scheduler.hybrid_kv.has_ssd = True
    scheduler.hybrid_kv.promote_ssd.return_value = True
    scheduler.hybrid_kv.fetch.return_value = (["cache"], [42], 599)

    request = Request(
        request_id="req-1",
        prompt="x",
        sampling_params=SamplingParams(max_tokens=8),
        prompt_token_ids=list(range(600)),
        num_prompt_tokens=600,
    )
    request.cache_hit_type = "ssd_pending"
    request._ssd_candidate = {"tokens": (1,), "file_path": "x"}
    scheduler.waiting.append(request)

    scheduler._try_promote_hybrid_ssd_pending()

    assert request.cache_hit_type == "system_kv"
    assert request.prompt_cache == ["cache"]
    assert request.cached_tokens == 599
    assert request.remaining_tokens == [42]


def test_try_promote_hybrid_ssd_pending_falls_back_to_miss(monkeypatch):
    from vllm_mlx.request import Request, SamplingParams

    scheduler = _make_scheduler(monkeypatch)
    scheduler.hybrid_kv = MagicMock()
    scheduler.hybrid_kv.has_ssd = True
    scheduler.hybrid_kv.promote_ssd.return_value = False

    request = Request(
        request_id="req-1",
        prompt="x",
        sampling_params=SamplingParams(max_tokens=8),
        prompt_token_ids=list(range(600)),
        num_prompt_tokens=600,
    )
    request.cache_hit_type = "ssd_pending"
    request._ssd_candidate = {"tokens": (1,), "file_path": "x"}
    scheduler.waiting.append(request)

    scheduler._try_promote_hybrid_ssd_pending()

    assert request.cache_hit_type == "miss"
    scheduler.hybrid_kv.fetch.assert_not_called()


def test_get_stats_emits_system_kv_cache_block(monkeypatch):
    scheduler = _make_scheduler(monkeypatch)
    stats = scheduler.get_stats()
    assert "system_kv_cache" in stats
    assert stats["system_kv_cache"]["enabled"] is True
    assert "memory_aware_cache" not in stats


# ── message-boundary checkpoints (PATCHES.md #88) ─────────────────────────


class _FakeTok:
    """Marker strings encode to distinctive id runs (100+char ords)."""

    bos_token_id = 1

    def encode(self, text, add_special_tokens=True):
        return [100 + (ord(c) % 50) for c in text[:4]]


def _boundary_kv(monkeypatch, min_step=None):
    if min_step is not None:
        monkeypatch.setenv("VLLM_MLX_BATCHED_KV_BOUNDARY_MIN_STEP", str(min_step))
    else:
        monkeypatch.delenv("VLLM_MLX_BATCHED_KV_BOUNDARY_MIN_STEP", raising=False)
    return BatchedSystemKV(_FakeModel())


def test_find_message_boundaries_hits_and_min_step(monkeypatch):
    from vllm_mlx.batched_system_kv import find_message_boundaries

    marker = (7, 8, 9)
    toks = [0] * 100 + [7, 8, 9] + [0] * 50 + [7, 8, 9] + [0] * 300 + [7, 8, 9] + [0] * 20
    # hits at 100, 153, 456. min_step measures from the previous accepted
    # cut INCLUDING position 0: with 120, 100 is too close to the start,
    # 153 accepted, 456 accepted (303 past 153).
    got = find_message_boundaries(toks, [marker], 120)
    assert got == (153, 456)
    # min_step 0 keeps all
    assert find_message_boundaries(toks, [marker], 0) == (100, 153, 456)
    # the LAST boundary bypasses min_step (llama.cpp #24176 rule)
    got = find_message_boundaries(toks, [marker], 10_000)
    assert got == (456,)
    # no markers / no hits
    assert find_message_boundaries(toks, [], 0) == ()
    assert find_message_boundaries([0] * 50, [marker], 0) == ()


def test_split_segments_boundary_cuts_with_interval_bound(monkeypatch):
    kv = _boundary_kv(monkeypatch)
    kv.ckpt_interval = 256
    toks = list(range(1000))
    segs = kv.split_segments(toks, boundaries=(300, 900))
    # cuts at 300 and 900, interval subdivision between them
    lens = [len(s) for s in segs]
    assert sum(lens) == 1000
    assert all(ln <= 256 for ln in lens)
    # reassembles exactly
    flat = [t for s in segs for t in s]
    assert flat == toks
    # boundary positions are segment starts
    starts = [0]
    for ln in lens[:-1]:
        starts.append(starts[-1] + ln)
    assert 300 in starts and 900 in starts
    # no boundaries -> legacy uniform split
    legacy = kv.split_segments(toks)
    assert [len(s) for s in legacy] == [256, 256, 256, 232]


def test_note_scheduled_stores_absolute_boundaries_and_capture_flags(monkeypatch):
    kv = _boundary_kv(monkeypatch)
    kv.note_scheduled("r1", 500, boundaries=(300,))
    assert kv._boundary_pos["r1"] == frozenset({800})
    kv.discard_pending("r1")
    assert "r1" not in kv._boundary_pos


def test_boundary_marker_ids_detects_family_and_caches(monkeypatch):
    kv = _boundary_kv(monkeypatch)
    prompt = "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nhi"
    tok = _FakeTok()
    seqs = kv.boundary_marker_ids(prompt, tok)
    assert seqs and all(isinstance(s, tuple) and s for s in seqs)
    assert kv.boundary_marker_ids(prompt, tok) is seqs  # cached per family
    assert kv.boundary_marker_ids("no markers here", tok) == ()
    assert kv.boundary_marker_ids(prompt, None) == ()


# ------------------------------------------------- #164 grown chains spill


def _grown_ssd_kv(monkeypatch, tmp_path, delta):
    from types import SimpleNamespace as NS

    monkeypatch.setenv("VLLM_MLX_SSD_SYSTEM_KV_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_MLX_SSD_SPILL_GROWN_DELTA", str(delta))
    return BatchedSystemKV(_FakeModel(), tokenizer=NS(name_or_path="unit/test-model"))


def _grow_turns(kv, turns):
    """Turn 0 stores TOKENS[:400]; each later turn restores and grows by 100."""
    kv.store("t0", TOKENS[:400], _donor_at(400))
    length = 400
    for n in range(1, turns + 1):
        rid = f"t{n}"
        assert kv.fetch(TOKENS[: length + 100], request_id=rid) is not None
        length += 100
        kv.store(rid, TOKENS[:length], _donor_at(length))
    return length


def test_164_a_grown_chain_spills_past_the_delta_and_restores_exactly(
    monkeypatch, tmp_path
):
    kv = _grown_ssd_kv(monkeypatch, tmp_path, delta=250)
    length = _grow_turns(kv, 4)  # 500, 600, 700, 800
    st = kv.stats()
    assert st["grown_stores"] == 4
    # Disk holds 400 (non-grown). 500 and 600 are under 250 past it, 700 is
    # not; 800 is 100 past 700 once that write has landed, 400 past otherwise.
    assert st["grown_spills"] in (1, 2)
    kv.close()  # drain: every queued spill lands

    kv2 = _grown_ssd_kv(monkeypatch, tmp_path, delta=250)
    try:
        candidate = kv2.check_ssd(TOKENS[:length] + [7])
        assert candidate is not None and len(candidate["tokens"]) >= 700
        assert kv2.promote_ssd(candidate)
        cache, remaining, pos = kv2.fetch(TOKENS[:length] + [7])
        assert pos == len(candidate["tokens"])
        control = _donor_at(pos)
        for layer in (0, 2):
            got, want = legacy_state(cache[layer]), legacy_state(control[layer])
            assert mx.array_equal(got[0], want[0])
            assert mx.array_equal(got[1], want[1])
    finally:
        kv2.close()


def test_164_off_by_default_grown_chains_do_not_spill(monkeypatch, tmp_path):
    kv = _grown_ssd_kv(monkeypatch, tmp_path, delta=0)
    try:
        _grow_turns(kv, 4)
        assert kv.stats()["grown_spills"] == 0
    finally:
        kv.close()


# ---------------------------------- #169 small chains stay off the SSD tier


_BIG = list(range(5000, 5600))


def _spill_small_and_big(monkeypatch, tmp_path, min_tokens):
    if min_tokens:
        monkeypatch.setenv("VLLM_MLX_SSD_SPILL_MIN_TOKENS", str(min_tokens))
    else:
        monkeypatch.delenv("VLLM_MLX_SSD_SPILL_MIN_TOKENS", raising=False)
    kv = _grown_ssd_kv(monkeypatch, tmp_path, delta=0)
    kv.store("small", TOKENS[:400], _donor_at(400))
    kv.store("big", _BIG, _donor_at(600))
    assert kv.fetch(TOKENS[:400] + [7]) is not None  # RAM still has it
    skipped = kv.stats()["spill_skipped_small"]
    kv.close()  # drain the spill queue
    kv2 = _grown_ssd_kv(monkeypatch, tmp_path, delta=0)
    try:
        small = kv2.check_ssd(TOKENS[:400] + [7])
        big = kv2.check_ssd(_BIG + [7])
    finally:
        kv2.close()
    return small, big, skipped


def test_169_chains_under_the_minimum_stay_ram_only(monkeypatch, tmp_path):
    small, big, skipped = _spill_small_and_big(monkeypatch, tmp_path, 450)
    assert small is None and skipped == 1
    assert big is not None and len(big["tokens"]) == 600


def test_169_off_by_default_every_chain_spills(monkeypatch, tmp_path):
    small, big, skipped = _spill_small_and_big(monkeypatch, tmp_path, 0)
    assert small is not None and big is not None and skipped == 0


def test_164_segmented_layer_round_trips_through_the_ssd_format():
    from vllm_mlx.system_kv import state_arrays
    from vllm_mlx.system_kv_ssd import flatten_snapshot, unflatten_snapshot

    a, b = _donor_at(300), _donor_at(500)
    bk, bv = legacy_state(b[0])
    seg = [legacy_state(a[0]), (bk[..., 300:, :] + 0, bv[..., 300:, :] + 0)]
    snapshot = [seg, legacy_state(a[1]), legacy_state(a[2])]
    tensors, meta = flatten_snapshot(snapshot)
    assert meta[0] == {"i": 0, "kind": "kvseg", "n": 2}
    back = unflatten_snapshot(tensors, meta)
    assert [k.shape[2] for k, _ in back[0]] == [300, 200]
    pairs = zip(state_arrays(back[0]), state_arrays(seg))
    assert all(mx.array_equal(x, y) for x, y in pairs)
    assert len(state_arrays(seg)) == 4  # byte accounting sees the segments


# --------------------------------------- #170 same-conversation supersede


def _lineage_kv(monkeypatch, tmp_path, slack="512"):
    from types import SimpleNamespace as NS

    monkeypatch.setenv("VLLM_MLX_SSD_SYSTEM_KV_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_MLX_SSD_SUPERSEDE_SLACK", slack)
    return BatchedSystemKV(_FakeModel(), tokenizer=NS(name_or_path="unit/test-model"))


def _with_donor(kv, rid, disk):
    kv._entries[99] = {"tokens": [], "disk": disk}
    kv._restore_source[rid] = 99


def test_170_next_turn_inherits_the_disk_copy_of_its_conversation(
    monkeypatch, tmp_path
):
    """The previous turn's copy = its prompt (350) + generated answer. The next
    turn contains that prompt up to the re-rendered generation header."""
    kv = _lineage_kv(monkeypatch, tmp_path)
    try:
        old = tuple(TOKENS[:350] + [9] * 50)
        _with_donor(kv, "r1", (old, 350))
        nxt = TOKENS[:345] + [5] * 200  # parts at 345: inside the header slack
        assert kv._lineage_disk_locked("r1", nxt) == (old, 350)
        # A different job that only shares the system prefix does not.
        other = TOKENS[:200] + [4] * 300
        assert kv._lineage_disk_locked("r1", other) is None
        # Unknown prompt length (an entry spilled before #170): no inheritance.
        _with_donor(kv, "r2", (old, None))
        assert kv._lineage_disk_locked("r2", nxt) is None
    finally:
        kv.close()


def test_170_supersede_pick_needs_a_checkpoint_near_the_split(monkeypatch, tmp_path):
    kv = _lineage_kv(monkeypatch, tmp_path)
    try:
        old = tuple(TOKENS[:350] + [9] * 50)
        new = tuple(TOKENS[:345] + [5] * 200)
        near = {"disk": (old, 350), "checkpoints": [{"pos": 300}, {"pos": 345}]}
        assert kv._supersede_pick(new, near) == old
        far = {"disk": (old, 350), "checkpoints": [{"pos": 540}]}
        assert kv._supersede_pick(new, far) is None
        # A strict prefix is left to the store's own rule.
        pre = tuple(TOKENS[:300])
        strict = {"disk": (pre, 290), "checkpoints": [{"pos": 299}]}
        assert kv._supersede_pick(new, strict) is None
    finally:
        kv.close()


def test_170_off_without_slack(monkeypatch, tmp_path):
    kv = _lineage_kv(monkeypatch, tmp_path, slack="0")
    try:
        old = tuple(TOKENS[:350] + [9] * 50)
        new = tuple(TOKENS[:345] + [5] * 200)
        e = {"disk": (old, 350), "checkpoints": [{"pos": 345}]}
        assert kv._supersede_pick(new, e) is None
    finally:
        kv.close()


def test_170_grown_spill_delta_counts_from_the_lineage_copy(monkeypatch, tmp_path):
    """The previous turn's copy ends in the model's raw output, so it is not a
    strict prefix of the next chain; the delta is measured from it anyway
    (before: from the longest strict prefix, i.e. every turn re-spilled)."""
    kv = _grown_ssd_kv(monkeypatch, tmp_path, delta=250)
    try:
        chain = list(TOKENS)
        n = len(chain)
        assert n > 400
        prev = tuple(chain[: n - 100] + [-1] * 50)  # chain is 100 past it
        assert kv._grown_spill_due(chain) is True  # nothing on disk
        assert kv._grown_spill_due(chain, {"disk": (prev, n - 100)}) is False
        far = tuple(chain[: n - 300] + [-1] * 50)  # 300 past: due
        assert kv._grown_spill_due(chain, {"disk": (far, n - 300)}) is True
    finally:
        kv.close()


# ------------------------------------- #171 promote only usable SSD candidates


class _StubSSD:
    """Index-level stand-in: rows and per-entry checkpoint positions."""

    def __init__(self, prefix_row=None, shared=(), ckpts=None):
        self.prefix_row, self.shared, self.ckpts = prefix_row, list(shared), ckpts or {}

    def lookup_prefix(self, toks):
        return self.prefix_row

    def lookup_shared(self, toks):
        return self.shared

    def read_meta(self, fp):
        return {"kinds": ["trim", "ckpt"],
                "checkpoints": [{"pos": p} for p in self.ckpts.get(fp, [])]}  # fmt: skip


def _shared(fp, n, common):
    return {"file_path": fp, "tokens": tuple(range(n)), "common_len": common}


def test_171_best_usable_candidate_wins_not_longest_common(monkeypatch, tmp_path):
    kv = _lineage_kv(monkeypatch, tmp_path)
    real = kv._ssd
    try:
        # a: longer common prefix but its ladder stops at 300; b: lands at 3000.
        kv._ssd = _StubSSD(
            shared=[_shared("a", 9000, 8000), _shared("b", 5000, 4000)],
            ckpts={"a": [300], "b": [2048, 3000]},
        )
        c = kv.check_ssd(list(range(10000)))
        assert c["file_path"] == "b" and c["usable"] == 3000
    finally:
        kv._ssd = real
        kv.close()


def test_171_unusable_candidate_is_not_promoted(monkeypatch, tmp_path):
    """09:30 2026-11-21: a 13K entry shared 4000 tokens but had no checkpoint
    a restore could use; it was promoted (1 GB read) and then nothing came of
    it. Now check_ssd returns None and counts it."""
    kv = _lineage_kv(monkeypatch, tmp_path)
    real = kv._ssd
    try:
        kv._ssd = _StubSSD(
            shared=[_shared("a", 13114, 4000)], ckpts={"a": [8000, 12000]}
        )
        assert kv.check_ssd(list(range(11583))) is None
        assert kv.ssd_unusable_skipped == 1
        # A full-prefix hit is still taken.
        kv._ssd = _StubSSD(
            prefix_row={"num_tokens": 600, "file_path": "p"}, ckpts={"p": [512]}
        )
        c = kv.check_ssd(list(range(900)))
        assert c is not None and c["usable"] == 600
    finally:
        kv._ssd = real
        kv.close()


def test_171_json_parameter_values_compare_in_sorted_key_order():
    """2026-11-21 02:02: the model wrote a ``calls`` parameter (a JSON list of
    nested calls) and the client sent it back with the nested keys in another
    order; #166 compared parameter values as raw text and re-prefilled."""
    from vllm_mlx.batched_system_kv import _canonical_call

    model = ('<function=tool_call>\n<parameter=calls>\n[{"arguments": {"datasourceUid": '
             '"loki", "format": "compact", "limit": 40}, "name": "query_loki"}]\n'
             "</parameter>\n</function>")  # fmt: skip
    client = ('<function=tool_call>\n<parameter=calls>\n[{"name": "query_loki", "arguments": '
              '{"limit": 40, "datasourceUid": "loki", "format": "compact"}}]\n'
              "</parameter>\n</function>")  # fmt: skip
    assert _canonical_call(model) == _canonical_call(client)
    changed = client.replace("40", "41")
    assert _canonical_call(model) != _canonical_call(changed)
    # Values that are not JSON compare as text, as before.
    a = "<function=f>\n<parameter=q>\n{not json\n</parameter>\n</function>"
    b = "<function=f>\n<parameter=q>\n{not  json\n</parameter>\n</function>"
    assert _canonical_call(a) == _canonical_call(b)


# ----------------------------------------- #173 free-space floor refusals


def test_173_a_refused_spill_leaves_no_disk_lineage(monkeypatch, tmp_path):
    """Under the floor the chain must not be recorded as on disk: a phantom
    ``disk`` key would make #170 supersede the phantom (leaking the real older
    copy) and #164 skip grown spills it still owes once space frees up."""
    import collections

    from vllm_mlx import system_kv_ssd

    DU = collections.namedtuple("DU", "total used free")
    free = {"v": 5}
    monkeypatch.setattr(
        system_kv_ssd.shutil, "disk_usage", lambda p: DU(1000, 0, free["v"])
    )
    monkeypatch.setenv("VLLM_MLX_SSD_MIN_FREE_PCT", "10")
    kv = _grown_ssd_kv(monkeypatch, tmp_path, delta=0)
    ssd = kv._ssd
    try:
        kv.store("t0", TOKENS[:400], _donor_at(400))
        (entry,) = kv._entries.values()
        assert entry.get("disk") is None
        assert not entry.get("spill_pending")
        st = ssd.get_stats()
        assert st["spill_refused_low_disk"] == 1 and st["queued_bytes"] == 0
        free["v"] = 10**15  # space freed: the next chain spills normally
        kv.store("t1", _BIG, _donor_at(600))
        assert any(e.get("disk") is not None for e in kv._entries.values())
    finally:
        kv.close()
    assert ssd.get_stats()["spill_count"] == 1
