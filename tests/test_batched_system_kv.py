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
    assert mx.array_equal(result[0][0].state[0], control[0].state[0])


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
        "req-1", [1, 2, 3, 4, 5], ["layer0"]
    )
    assert request._extracted_cache is None


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
