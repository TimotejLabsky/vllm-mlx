"""Fork invariants — the guarantees upstream never sees and a rebase can
silently undo (precedent: upstream #683 disabled the hybrid finish store for a
month, PATCHES.md #100). Each test is named for the PATCHES.md entry it pins
and asserts the CONDITION, not an incidental effect, so a reverted hunk goes
red here even when the feature-level suites still pass.

A rebase is not done until this module is green AND its assertions were
re-read against upstream's diff of the touched functions.
"""

import shutil
import sys
import tempfile
import time
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm_mlx import batched_system_kv as bkv
from vllm_mlx.batched_system_kv import BatchedSystemKV
from vllm_mlx.system_kv_ssd import SystemKVSSDConfig, SystemKVSSDStore

from tests.test_batched_flip_enablement import _mb, _watermarked
from tests.test_batched_system_kv import (
    TOKENS,
    _donor_at,
    _FakeModel,
    _make_scheduler,
)
from tests.test_system_kv_ssd import _make_hybrid_snapshot

DISJOINT = list(range(9000, 9800))


def _wait_for(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


# ------------------------------------------------- #103 recovery must not leak


def _running_request(scheduler, rid, uid, base):
    from vllm_mlx.request import Request, SamplingParams

    prompt = list(range(base, base + 800))
    request = Request(
        request_id=rid,
        prompt="x",
        sampling_params=SamplingParams(max_tokens=8),
        prompt_token_ids=prompt,
        num_prompt_tokens=len(prompt),
    )
    scheduler.running[rid] = request
    scheduler.requests[rid] = request
    scheduler.uid_to_request_id[uid] = rid
    scheduler.request_id_to_uid[rid] = uid
    return request


def _seed_ladder(kv, rid, pos=512):
    """What capture_segment / a restoring fetch leave behind for a live row."""
    kv._base_pos[rid] = 0
    kv._restore_source[rid] = 1
    kv.capture_segment(rid, pos, _donor_at(pos))
    assert rid in kv._pending


def test_103_recovery_discards_pending_ladders(monkeypatch):
    """generation_error_recovery skips _cleanup_finished, so it must drop each
    aborted row's in-flight ladder itself (as _do_abort_request does). Found
    live 2026-09-16: every Metal-OOM recovery kept up to 8 recurrent-state
    copies per row alive — and after a HIT, the evicted donor's — so active
    memory ratcheted ~1 GB per recovery (55.9 -> 71.2 GB at running=1) while
    relief evicted entries and freed nothing."""
    scheduler = _make_scheduler(monkeypatch)
    kv = scheduler.hybrid_kv
    assert isinstance(kv, BatchedSystemKV)
    for rid, uid, base in [("oom-a", 21, 1000), ("oom-b", 22, 5000)]:
        request = _running_request(scheduler, rid, uid, base)
        request.prompt_cache = _donor_at(16)
        request._extracted_cache = _donor_at(16)
        _seed_ladder(kv, rid)
    assert kv.stats()["pending_ladders"] == 2
    assert kv.stats()["pending_ladder_mb"] > 0

    aborted = scheduler._recover_from_generation_error()

    assert aborted == {"oom-a", "oom-b"}
    assert kv._pending == {} and kv._base_pos == {} and kv._restore_source == {}
    assert kv.stats()["pending_ladders"] == 0
    assert kv.stats()["pending_ladder_mb"] == 0
    for rid in aborted:
        request = scheduler.requests[rid]
        assert request.prompt_cache is None
        assert request._extracted_cache is None


def test_103_step_error_path_leaves_no_pending_ladder(monkeypatch):
    """The same guarantee through the real ``step()`` except-branch — the
    branch that ``break``s past _cleanup_finished."""
    scheduler = _make_scheduler(monkeypatch)
    kv = scheduler.hybrid_kv
    _running_request(scheduler, "oom-step", 31, 1000)
    _seed_ladder(kv, "oom-step")
    generator = MagicMock()
    generator.next.side_effect = RuntimeError(
        "[METAL] Command buffer execution failed: Insufficient Memory"
    )
    generator.prompt_cache_nbytes = 0
    scheduler.batch_generator = generator

    output = scheduler.step()

    assert "oom-step" in output.finished_request_ids
    assert [o.finish_reason for o in output.outputs] == ["error"]
    assert kv.stats()["pending_ladders"] == 0
    assert not scheduler.running


# ------------------------------------ #103 waiting deque is mutated cross-thread


def test_103_promote_ssd_pending_tolerates_a_mutating_waiting_queue():
    """add_request appends to ``waiting`` on the event loop while the promote
    hook iterates it on the executor. Iterating the live deque raised
    ``deque mutated during iteration`` 15x on 2026-09-16 — each one killed the
    batch generator and aborted every in-flight row."""
    waiting = deque()

    def _request(rid):
        return SimpleNamespace(
            request_id=rid,
            cache_hit_type="ssd_pending",
            _ssd_candidate={"tokens": (1, 2, 3), "file_path": rid},
            prompt_token_ids=[1, 2, 3],
        )

    waiting.extend([_request("w1"), _request("w2")])
    hybrid_kv = MagicMock()

    def _promote(_candidate):
        waiting.append(_request(f"late-{len(waiting)}"))  # the racing append
        return False

    hybrid_kv.promote_ssd.side_effect = _promote
    scheduler = SimpleNamespace(hybrid_kv=hybrid_kv, waiting=waiting)

    bkv.promote_ssd_pending(scheduler)  # must not raise

    assert [r.cache_hit_type for r in list(waiting)[:2]] == ["miss", "miss"]
    assert len(waiting) == 4  # the late arrivals wait for the next pass


# ------------------------------------------ #103 the SSD spill backlog is memory


def _busy_store(tmpdir, **config):
    busy = {"flag": True, "may_write": False}
    store = SystemKVSSDStore(
        SystemKVSSDConfig(cache_dir=tmpdir, **config),
        idle_check=lambda: not busy["flag"],
        can_write_busy=lambda: busy["may_write"],
    )
    store.start_writer()
    return store, busy


def test_103_drop_backlog_releases_queued_and_held_spills():
    """A never-idle engine leaves every spill queued (and one held by the
    writer in its idle-wait): resident unified memory that serves no request.
    drop_backlog must release ALL of it, and the writer must keep working."""
    d = tempfile.mkdtemp(prefix="skv-backlog-")
    try:
        store, busy = _busy_store(d)
        for n in range(3):
            assert store.enqueue_spill(
                tuple(range(n * 100, n * 100 + 48)), _make_hybrid_snapshot(seq=32)
            )
        # the writer dequeues one and parks in the idle-wait holding it
        assert _wait_for(lambda: store.get_stats()["queue_depth"] == 2)
        queued = store.get_stats()["queued_bytes"]
        assert queued > 0

        assert store.drop_backlog() == queued

        assert _wait_for(lambda: store.get_stats()["queued_bytes"] == 0)
        assert _wait_for(lambda: store.get_stats()["backlog_drops"] == 3)
        stats = store.get_stats()
        assert stats["queue_depth"] == 0 and stats["spill_count"] == 0
        assert store.lookup_prefix(tuple(range(60))) is None  # nothing written

        # still alive: an idle engine lands the next spill
        busy["flag"] = False
        tokens = tuple(range(7000, 7048))
        assert store.enqueue_spill(tokens, _make_hybrid_snapshot(seq=32))
        assert _wait_for(lambda: store.lookup_prefix(tokens) is not None, 10.0)
        store.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_103_busy_write_lands_after_a_bounded_wait_when_memory_allows():
    """Defer-until-idle assumes idle windows. After ``busy_write_after_s`` a
    queued spill is written while busy — but only once the caller's memory
    check agrees (2026-09-17: 4 writes vs 30 drops in 3 h of CI)."""
    d = tempfile.mkdtemp(prefix="skv-busywrite-")
    try:
        store, busy = _busy_store(d, busy_write_after_s=0.3)
        tokens = tuple(range(48))
        assert store.enqueue_spill(tokens, _make_hybrid_snapshot(seq=32))
        time.sleep(1.6)
        assert store.lookup_prefix(tokens) is None, "wrote under memory pressure"

        busy["may_write"] = True
        assert _wait_for(lambda: store.lookup_prefix(tokens) is not None, 10.0)
        assert store.get_stats()["busy_writes"] == 1
        store.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_103_never_writes_while_busy_by_default():
    """SimpleEngine keeps the 2026-06-12 contract: no escape hatch unless
    configured."""
    assert SystemKVSSDConfig(cache_dir="x").busy_write_after_s == 0.0


def test_103_writer_releases_the_snapshot_after_writing():
    """Loop locals outlive an iteration: the writer kept the last-written
    snapshot's arrays referenced until the NEXT spill arrived."""
    d = tempfile.mkdtemp(prefix="skv-refs-")
    try:
        store = SystemKVSSDStore(SystemKVSSDConfig(cache_dir=d))
        store.start_writer()
        tokens = tuple(range(48))
        assert store.enqueue_spill(tokens, _make_hybrid_snapshot(seq=32))
        assert _wait_for(lambda: store.lookup_prefix(tokens) is not None, 10.0)

        def _writer_locals():
            frame = sys._current_frames().get(store._writer_thread.ident)
            while frame is not None and frame.f_code.co_name != "_writer_loop":
                frame = frame.f_back
            return dict(frame.f_locals) if frame is not None else None

        def _released():
            held = _writer_locals()
            return held is not None and all(
                held.get(name) is None for name in ("item", "tensors")
            )

        assert _wait_for(_released)
        store.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_103_relief_drops_the_spill_backlog_before_any_entry(monkeypatch):
    """Relief used to evict the chains agents were about to extend while the
    backlog stayed pinned. A breach the backlog can absorb costs no entry."""
    kv, mem = _watermarked(monkeypatch, active_mb=50)
    kv.store("r1", TOKENS, _donor_at(len(TOKENS)))
    kv.store("r2", DISJOINT, _donor_at(len(DISJOINT)))
    ssd = MagicMock()

    def _drop():
        mem["active"] = _mb(40)
        return _mb(30)

    ssd.drop_backlog.side_effect = _drop
    kv._ssd = ssd
    mem["active"] = _mb(95)
    mem["peak"] = _mb(95)

    assert kv.relieve_pressure() == 0
    ssd.drop_backlog.assert_called_once()
    assert kv.pressure_backlog_drops == 1
    assert kv.stats()["pressure_backlog_drops"] == 1
    assert kv.stats()["entry_count"] == 2 and kv.pressure_evictions == 0


def test_103_relief_still_evicts_when_the_backlog_is_not_enough(monkeypatch):
    kv, mem = _watermarked(monkeypatch, active_mb=50)
    kv.store("r1", TOKENS, _donor_at(len(TOKENS)))
    kv.store("r2", DISJOINT, _donor_at(len(DISJOINT)))
    ssd = MagicMock()
    ssd.drop_backlog.return_value = 0  # empty backlog
    kv._ssd = ssd
    mem["peak"] = _mb(95)  # a chunk spiked; active already back under

    assert kv.relieve_pressure() == 1
    assert kv.pressure_backlog_drops == 0
    assert kv.stats()["entry_count"] == 1


def test_103_batched_route_caps_the_backlog_and_arms_busy_writes(monkeypatch, tmp_path):
    """The store's 12 GB / idle-only defaults were sized for SimpleEngine's
    turn gaps; the batched bag must not inherit them."""
    monkeypatch.setenv("VLLM_MLX_SSD_SYSTEM_KV_DIR", str(tmp_path))
    kv = BatchedSystemKV(_FakeModel())
    try:
        assert kv._ssd._config.max_queued_gb == 4.0
        assert kv._ssd._config.busy_write_after_s == 30.0
        assert kv._ssd._can_write_busy() is True  # no watermark armed
        assert "queued_bytes" in kv.stats()["ssd"]
    finally:
        kv._ssd.close()

    monkeypatch.setenv("VLLM_MLX_SSD_SYSTEM_KV_MAX_QUEUED_GB", "1.5")
    monkeypatch.setenv("VLLM_MLX_SSD_SYSTEM_KV_BUSY_WRITE_S", "0")
    kv = BatchedSystemKV(_FakeModel())
    try:
        assert kv._ssd._config.max_queued_gb == 1.5
        assert kv._ssd._config.busy_write_after_s == 0.0
    finally:
        kv._ssd.close()


def test_103_metrics_export_the_unaccounted_memory():
    """Both holders were invisible to every gauge while they pinned the 27B
    route at its watermark with an EMPTY bag (cache_memory_bytes read 0)."""
    import pytest

    pytest.importorskip("prometheus_client")
    from vllm_mlx.metrics import MetricsCollector

    collector = MetricsCollector()
    collector.configure(enabled=True)
    collector._init_prometheus()
    engine = MagicMock()
    engine.get_stats.return_value = {
        "engine_type": "batched",
        "system_kv_cache": {
            "hits": 1,
            "misses": 0,
            "evictions": 0,
            "hit_rate": 1.0,
            "tokens_saved": 0,
            "partial_hits": 0,
            "pending_ladder_mb": 2.0,
            "ssd": {"queued_bytes": 12345},
        },
    }
    body, _ = collector.render_metrics(engine=engine, mcp_manager=None)
    values = {
        name: float(value)
        for name, _, value in (
            line.rpartition(" ") for line in body.decode().splitlines()
        )
        if name.startswith("vllm_mlx_cache_")
    }
    assert values["vllm_mlx_cache_pending_ladder_bytes"] == _mb(2)
    assert values["vllm_mlx_cache_ssd_queued_bytes"] == 12345
