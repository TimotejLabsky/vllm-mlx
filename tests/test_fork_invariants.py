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


# ----------------------- #104 an engine-side abort must not read as an answer

import asyncio  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402

import pytest  # noqa: E402

# opencode's retry policy (session/retry.ts): status >= 500, or a message
# matching one of these. A frame that matches neither is treated as fatal.
_AGENT_RETRY_RE = re.compile(
    r"429|500|502|503|504|524|overloaded|service unavailable|internal error",
    re.IGNORECASE,
)


def _aborted_output(**overrides):
    base = dict(
        output_text="",
        new_text="",
        output_token_ids=[],
        prompt_tokens=7,
        completion_tokens=0,
        finished=True,
        finish_reason="error",
        error_kind="oom",
        mtp_drafts=0,
        mtp_accepted=0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_104_only_engine_aborts_become_generation_aborted():
    from vllm_mlx.engine.base import GenerationAborted, raise_if_generation_aborted

    with pytest.raises(GenerationAborted) as caught:
        raise_if_generation_aborted(_aborted_output())
    assert caught.value.kind == "oom" and caught.value.code == "generation_aborted"
    assert _AGENT_RETRY_RE.search(str(caught.value))

    with pytest.raises(GenerationAborted) as caught:
        raise_if_generation_aborted(_aborted_output(error_kind=None))
    assert caught.value.kind == "generation_error"

    # prompt_too_long keeps its non-retryable 400 path; normal finishes pass
    raise_if_generation_aborted(_aborted_output(error_kind="prompt_too_long"))
    raise_if_generation_aborted(_aborted_output(finish_reason="stop"))
    raise_if_generation_aborted(_aborted_output(finish_reason="length"))


def test_104_recovery_names_the_cause_and_counts_itself(monkeypatch):
    scheduler = _make_scheduler(monkeypatch)
    _running_request(scheduler, "oom-kind", 41, 1000)
    generator = MagicMock()
    generator.next.side_effect = RuntimeError(
        "[METAL] Command buffer execution failed: Insufficient Memory "
        "(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)."
    )
    generator.prompt_cache_nbytes = 0
    scheduler.batch_generator = generator

    output = scheduler.step()

    assert [(o.finish_reason, o.error_kind) for o in output.outputs] == [
        ("error", "oom")
    ]
    stats = scheduler.get_stats()
    assert stats["generation_recoveries"] == 1
    assert stats["recovery_aborted_requests"] == 1

    # anything that is not a Metal OOM is still an abort, just named honestly
    _running_request(scheduler, "other-kind", 42, 5000)
    generator = MagicMock()
    generator.next.side_effect = RuntimeError("deque mutated during iteration")
    generator.prompt_cache_nbytes = 0
    scheduler.batch_generator = generator
    output = scheduler.step()
    assert [o.error_kind for o in output.outputs] == ["generation_error"]
    assert scheduler.get_stats()["generation_recoveries"] == 2


class _AbortingCore:
    """An engine core whose request is taken down by recovery after ``emit``
    good chunks — what BatchedEngine sees from engine_core today."""

    def __init__(self, emit=0):
        self.emit = emit

    async def add_request(self, prompt, sampling_params, prefix_boundary=0):
        return "req-abort"

    async def stream_outputs(self, request_id):
        for i in range(self.emit):
            yield _aborted_output(
                output_text="tok" * (i + 1),
                new_text="tok",
                output_token_ids=[i],
                completion_tokens=i + 1,
                finished=False,
                finish_reason=None,
                error_kind=None,
            )
        yield _aborted_output()

    async def generate(self, prompt, sampling_params):
        return _aborted_output()

    async def abort_request(self, request_id):
        return True


def _batched_engine(emit=0):
    from vllm_mlx.engine.batched import BatchedEngine

    engine = BatchedEngine.__new__(BatchedEngine)
    engine._loaded = True
    engine._is_mllm = False
    engine._mllm_scheduler = None
    engine._engine = _AbortingCore(emit)
    engine._tokenizer = SimpleNamespace(encode=lambda s, **k: [1])
    return engine


def test_104_batched_engine_raises_instead_of_returning_an_empty_turn():
    from vllm_mlx.engine.base import GenerationAborted

    with pytest.raises(GenerationAborted):
        asyncio.run(_batched_engine().generate(prompt="x"))

    seen = []

    async def consume():
        async for out in _batched_engine(emit=2).stream_generate(prompt="x"):
            seen.append(out)

    with pytest.raises(GenerationAborted):
        asyncio.run(consume())
    # the good chunks still reached the caller; the abort never did as text
    assert [o.new_text for o in seen] == ["tok", "tok"]
    assert all(o.finish_reason != "error" for o in seen)


def test_104_non_stream_abort_is_a_retryable_503():
    import vllm_mlx.server as srv
    from fastapi import HTTPException
    from vllm_mlx.engine.base import GenerationAborted

    with pytest.raises(HTTPException) as caught:
        srv._raise_generation_aborted(GenerationAborted("oom"))
    assert caught.value.status_code == 503
    assert caught.value.headers["Retry-After"] == "15"
    assert caught.value.detail["error"] == "generation_aborted"
    assert caught.value.detail["kind"] == "oom"

    # every non-stream endpoint that maps EngineBusy must map this too
    source = open(srv.__file__).read()
    assert source.count("except GenerationAborted as exc:\n") >= 6
    assert (
        source.count("_raise_generation_aborted(exc)")
        == source.count("_raise_engine_busy(exc)") - 1
    )  # the pre-stream probe has no generation to abort


class _AbortingChatEngine:
    model_name = "test-model"

    def __init__(self, emit=0):
        self.emit = emit

    async def _stream(self):
        from vllm_mlx.engine.base import GenerationAborted, GenerationOutput

        for i in range(self.emit):
            yield GenerationOutput(
                text=f"tok{i}",
                new_text=f"tok{i}",
                finished=False,
                finish_reason=None,
                prompt_tokens=4,
                completion_tokens=i + 1,
            )
        raise GenerationAborted("oom")

    def stream_chat(self, messages, **kwargs):
        return self._stream()

    def stream_generate(self, **kwargs):
        return self._stream()


def _sse(chunks):
    out = []
    for line in "".join(chunks).splitlines():
        if line.startswith("data: ") and line[6:].strip() != "[DONE]":
            out.append(json.loads(line[6:]))
    return out


class _Tracker:
    def __init__(self):
        self.calls = []

    def observe_ttft(self):
        pass

    def finish(self, **kwargs):
        self.calls.append(kwargs)


@pytest.mark.parametrize("emit", [0, 2])
def test_104_chat_stream_abort_is_a_503_shaped_frame(monkeypatch, emit, caplog):
    """Headers are long gone, so the frame has to say "retryable": the 09-16
    recoveries all ended as 200 + an empty delta + finish_reason="error"."""
    import vllm_mlx.server as srv

    monkeypatch.setattr(srv, "_model_name", "test-model")
    monkeypatch.setattr(srv, "_reasoning_parser_name", None)
    monkeypatch.setattr(srv, "_reasoning_parser", None)
    monkeypatch.setattr(srv, "_enable_auto_tool_choice", False)
    monkeypatch.setattr(srv, "_tool_call_parser", None)
    request = srv.ChatCompletionRequest(
        model="test-model",
        messages=[srv.Message(role="user", content="Hello")],
        stream=True,
    )
    tracker = _Tracker()
    chunks = []

    async def consume():
        async for chunk in srv.stream_chat_completion(
            _AbortingChatEngine(emit),
            request.messages,
            request,
            metrics_tracker=tracker,
        ):
            chunks.append(chunk)

    with caplog.at_level("WARNING"):
        asyncio.run(consume())  # expected condition: must NOT raise

    payloads = _sse(chunks)
    errors = [p["error"] for p in payloads if "error" in p]
    assert len(errors) == 1
    assert errors[0]["code"] == 503 and errors[0]["kind"] == "oom"
    assert errors[0]["error"] == "generation_aborted"
    assert _AGENT_RETRY_RE.search(errors[0]["message"])
    assert "".join(chunks).rstrip().endswith("data: [DONE]")
    # never a clean finish: no chunk may carry a finish_reason
    finishes = [
        c.get("finish_reason")
        for p in payloads
        for c in p.get("choices", [])
        if c.get("finish_reason")
    ]
    assert finishes == []
    # counted as an error (feeds stream_aborts_total), phase from the tokens
    assert tracker.calls[0]["result"] == "error"
    assert tracker.calls[0]["completion_tokens"] == emit
    assert "aborted by the engine" in caplog.text and "Traceback" not in caplog.text


def test_104_completion_stream_abort_is_a_503_shaped_frame():
    import vllm_mlx.server as srv

    request = srv.CompletionRequest(model="test-model", prompt="hi", stream=True)
    tracker = _Tracker()
    chunks = []

    async def consume():
        async for chunk in srv.stream_completion(
            _AbortingChatEngine(),
            "hi",
            request,
            max_tokens=16,
            metrics_tracker=tracker,
        ):
            chunks.append(chunk)

    asyncio.run(consume())
    errors = [p["error"] for p in _sse(chunks) if "error" in p]
    assert len(errors) == 1 and errors[0]["code"] == 503
    assert "[DONE]" in "".join(chunks)
    assert tracker.calls[0]["result"] == "error"


def test_104_anthropic_dialect_uses_its_retryable_error_type():
    import vllm_mlx.server as srv
    from vllm_mlx.engine.base import GenerationAborted

    event = srv._anthropic_stream_error_event(GenerationAborted("oom"))
    payload = json.loads(event.split("data: ", 1)[1])
    assert payload["error"]["type"] == "overloaded_error"
    assert _AGENT_RETRY_RE.search(payload["error"]["message"])
    # the generic (#91) shapes are untouched
    assert (
        json.loads(srv._stream_error_chunk().split("data: ", 1)[1])["error"]["code"]
        == "stream_failed"
    )
    assert "api_error" in srv._anthropic_stream_error_event()


# ------------------------------ #100 the hybrid bag's finish store vs upstream #683


def _finish_one(scheduler, rid="fin-a", uid=51, base=1000):
    """Drive one finished hybrid row through the REAL response path."""
    request = _running_request(scheduler, rid, uid, base)
    request.first_token_time = 0.0
    request.append_output_token(base + 900)
    scheduler._decode_tokens = lambda ids: ""
    response = SimpleNamespace(
        uid=uid,
        token=base + 901,
        finish_reason="stop",
        prompt_cache=_donor_at(len(request.prompt_token_ids) + 2),
    )
    scheduler._process_batch_responses([response])
    return request


def test_100_hybrid_bag_keeps_the_extraction_upstream_calls_useless(monkeypatch):
    """The condition itself, not its downstream effect: upstream #683 drops
    the finish-time cache whenever it cannot be trimmed, and ArraysCache never
    can — so on every hybrid model the bag's only concurrent store path went
    dark for a month. The bag restores by slicing + checkpoints and never
    trims; with it active the verdict of ``_prompt_output_entry_is_useless``
    must not matter."""
    from vllm_mlx.scheduler import Scheduler

    monkeypatch.setattr(
        Scheduler, "_prompt_output_entry_is_useless", staticmethod(lambda cache: True)
    )
    scheduler = _make_scheduler(monkeypatch)
    assert scheduler.hybrid_kv is not None

    request = _finish_one(scheduler)

    assert getattr(request, "_extracted_cache", None) is not None


def test_100_upstream_gate_still_governs_every_other_cache(monkeypatch):
    """The carve-out is for the hybrid bag only. Without it, #683's protection
    (45 dead full-length entries -> Metal resource limit 499000) must hold."""
    from vllm_mlx.scheduler import Scheduler

    scheduler = _make_scheduler(monkeypatch)
    scheduler.hybrid_kv = None

    monkeypatch.setattr(
        Scheduler, "_prompt_output_entry_is_useless", staticmethod(lambda cache: True)
    )
    dropped = _finish_one(scheduler, "fin-useless", 52, 1000)
    assert getattr(dropped, "_extracted_cache", None) is None

    monkeypatch.setattr(
        Scheduler, "_prompt_output_entry_is_useless", staticmethod(lambda cache: False)
    )
    kept = _finish_one(scheduler, "fin-useful", 53, 5000)
    assert getattr(kept, "_extracted_cache", None) is not None


def test_100_a_real_hybrid_cache_is_what_upstream_calls_useless():
    """Pins the premise: if mlx-lm ever makes ArraysCache trimmable, the
    carve-out stops mattering and this goes red to say so."""
    from vllm_mlx.scheduler import Scheduler

    assert Scheduler._prompt_output_entry_is_useless(_donor_at(64)) is True


# ------------------------------------- #34/#37 the bag's LRU and donor linkage


def test_034_a_hit_protects_its_entry_from_the_next_eviction(monkeypatch):
    """Eviction is LRU with hit-touch, not FIFO: the chain an agent just
    extended must outlive an idle one. (The baseline any smarter policy —
    plan P1-1 — has to keep.)"""
    monkeypatch.setenv("VLLM_MLX_SYSTEM_KV_SLOTS", "2")
    kv = BatchedSystemKV(_FakeModel())
    kv.store("r1", TOKENS, _donor_at(len(TOKENS)))
    kv.store("r2", DISJOINT, _donor_at(len(DISJOINT)))

    assert kv.fetch(TOKENS + [1, 2, 3]) is not None  # touches r1's entry

    third = list(range(20000, 20800))
    kv.store("r3", third, _donor_at(len(third)))

    kept = sorted(e["tokens"][0] for e in kv._entries.values())
    assert kept == [TOKENS[0], third[0]]  # the untouched DISJOINT entry went


def test_037_a_store_whose_donor_was_evicted_still_lands_as_a_full_copy(monkeypatch):
    """fetch links the request to its donor entry for the O(delta) grow path.
    If that donor is evicted before the request finishes — routine with more
    live chains than slots — the store must fall back to a full copy, not
    fail or alias a dropped entry."""
    monkeypatch.setenv("VLLM_MLX_SYSTEM_KV_SLOTS", "1")
    kv = BatchedSystemKV(_FakeModel())
    kv.store("donor", TOKENS, _donor_at(len(TOKENS)))

    grown = TOKENS + list(range(30000, 30400))
    assert kv.fetch(grown, request_id="grower") is not None
    assert "grower" in kv._restore_source

    kv.store("other", DISJOINT, _donor_at(len(DISJOINT)))  # evicts the donor
    assert [e["tokens"][0] for e in kv._entries.values()] == [DISJOINT[0]]

    assert kv.store("grower", grown, _donor_at(len(grown))) is True

    entries = list(kv._entries.values())
    assert len(entries) == 1 and entries[0]["tokens"] == grown
    assert kv.grown_stores == 0  # a full copy, not a grow from a dead donor
    assert "grower" not in kv._restore_source and "grower" not in kv._pending
    # and it is a usable entry: an extension restores from it
    assert kv.fetch(grown + [7, 8, 9]) is not None


# ---------------------------------- #105 a request that runs alone is examined too

_ENTRY_MB = 20  # what each resident entry "costs" in the mocked allocator


def _solo_kv(monkeypatch, base_mb, ceiling_mb=100, transient_mb=10, floor_kb=64):
    """Guard armed; mocked allocator where active = base + 20 MB per entry,
    so evicting the bag visibly makes room. Limit = 95% of the ceiling."""
    import mlx.core as mx

    monkeypatch.setenv("VLLM_MLX_BATCHED_SOLO_TRANSIENT_MB", str(transient_mb))
    monkeypatch.setenv("VLLM_MLX_BATCHED_BPT_FLOOR_KB", str(floor_kb))
    kv = BatchedSystemKV(_FakeModel())
    monkeypatch.setattr(
        mx, "get_active_memory", lambda: _mb(base_mb + _ENTRY_MB * len(kv._entries))
    )
    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": _mb(ceiling_mb)}
    )
    monkeypatch.setattr(mx, "clear_cache", lambda: None)
    return kv


def test_105_guard_is_inert_unless_armed(monkeypatch):
    import mlx.core as mx

    monkeypatch.delenv("VLLM_MLX_BATCHED_SOLO_TRANSIENT_MB", raising=False)
    kv = BatchedSystemKV(_FakeModel())
    monkeypatch.setattr(mx, "get_active_memory", lambda: _mb(10_000))
    assert kv.solo_prefill_verdict(1_000_000) is None
    assert kv.stats()["solo_rejections"] == 0


def test_105_a_request_that_fits_is_admitted_untouched(monkeypatch):
    kv = _solo_kv(monkeypatch, base_mb=20)
    kv.store("r1", TOKENS, _donor_at(len(TOKENS)))
    # active 40 (20 + one entry) + 400 tok x 64 KB (25) + 10 transient = 75 <= 95
    assert kv.solo_prefill_verdict(400) is None
    assert kv.stats()["entry_count"] == 1
    assert kv.solo_relief_passes == 0 and kv.solo_rejections == 0


def test_105_makes_room_before_it_rejects(monkeypatch):
    """The bag and the spill backlog are pure cache: give them back, then
    re-check. Rejecting while holding droppable memory would be a false 503."""
    kv = _solo_kv(monkeypatch, base_mb=20)
    kv.store("r1", TOKENS, _donor_at(len(TOKENS)))
    kv.store("r2", DISJOINT, _donor_at(len(DISJOINT)))
    ssd = MagicMock()
    ssd.drop_backlog.return_value = _mb(5)
    kv._ssd = ssd
    # active 60 (20 + 2 entries); need 60 -> 120 > 95. One eviction -> 100,
    # still over; two -> 80 <= 95.
    assert kv.solo_prefill_verdict(800) is None
    ssd.drop_backlog.assert_called_once()  # backlog first
    assert kv.stats()["entry_count"] == 0
    assert kv.solo_relief_passes == 1 and kv.solo_rejections == 0
    assert kv.pressure_backlog_drops == 1


def test_105_rejects_only_what_cannot_fit_with_the_cache_emptied(monkeypatch):
    kv = _solo_kv(monkeypatch, base_mb=50)
    kv.store("r1", TOKENS, _donor_at(len(TOKENS)))
    # need 60 on a 50 MB floor of weights -> 110 > 95 even with an empty bag
    assert kv.solo_prefill_verdict(800) == "insufficient_memory"
    assert kv.stats()["entry_count"] == 0  # it did try
    assert kv.stats()["solo_rejections"] == 1


def test_105_prices_only_the_tokens_still_to_prefill(monkeypatch):
    """A restored prefix is already resident (it is inside ``active``)."""
    kv = _solo_kv(monkeypatch, base_mb=50)
    scheduler = SimpleNamespace(hybrid_kv=kv, running={})
    deep_but_cached = SimpleNamespace(
        remaining_tokens=[0] * 100, num_prompt_tokens=50_000
    )
    assert bkv.solo_prefill_verdict(scheduler, deep_but_cached) is None
    cold = SimpleNamespace(remaining_tokens=None, num_prompt_tokens=50_000)
    assert bkv.solo_prefill_verdict(scheduler, cold) == "insufficient_memory"


def test_105_is_not_consulted_when_something_is_running(monkeypatch):
    """Co-batching is the #39/#40/#102 gates' job; this one is for the hole
    they leave (``if not scheduler.running: return False``)."""
    kv = _solo_kv(monkeypatch, base_mb=50)
    scheduler = SimpleNamespace(hybrid_kv=kv, running={"busy": object()})
    cold = SimpleNamespace(remaining_tokens=None, num_prompt_tokens=50_000)
    assert bkv.solo_prefill_verdict(scheduler, cold) is None
    assert kv.solo_rejections == 0


def test_105_a_rejection_reaches_the_client_and_does_not_block_the_queue(monkeypatch):
    import mlx.core as mx
    from vllm_mlx.request import Request, SamplingParams

    monkeypatch.setenv("VLLM_MLX_BATCHED_SOLO_TRANSIENT_MB", "10")
    monkeypatch.setenv("VLLM_MLX_BATCHED_BPT_FLOOR_KB", "64")
    scheduler = _make_scheduler(monkeypatch)
    kv = scheduler.hybrid_kv
    monkeypatch.setattr(mx, "get_active_memory", lambda: _mb(50))
    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": _mb(100)}
    )
    # never reach the real generator: the request that passes the guard is
    # put back by the "no batch generator yet" branch
    monkeypatch.setattr(scheduler, "_ensure_batch_generator", lambda params: None)

    def _waiting(rid, n):
        request = Request(
            request_id=rid,
            prompt="x",
            sampling_params=SamplingParams(max_tokens=8),
            prompt_token_ids=list(range(n)),
            num_prompt_tokens=n,
        )
        scheduler.requests[rid] = request
        scheduler.waiting.append(request)
        return request

    too_big = _waiting("too-big", 800)  # 50 + 50 + 10 = 110 > 95
    fits = _waiting("fits", 100)  # 50 + 6.25 + 10 <= 95
    kv.note_scheduled("too-big", 0)
    kv.capture_segment("too-big", 64, _donor_at(64))
    assert "too-big" in kv._pending

    output = scheduler.step()

    assert [(o.request_id, o.finish_reason, o.error_kind) for o in output.outputs] == [
        ("too-big", "error", "insufficient_memory")
    ]
    assert output.finished_request_ids == {"too-big"}
    assert too_big.prompt_cache is None and "too-big" not in kv._pending
    assert "too-big" not in scheduler.running
    # the queue behind it was still examined — and passed the guard
    assert list(scheduler.waiting) == [fits]
    assert kv.stats()["solo_rejections"] == 1

    # ...and #104 turns that output into a retryable, named error
    from vllm_mlx.engine.base import GenerationAborted, raise_if_generation_aborted

    with pytest.raises(GenerationAborted) as caught:
        raise_if_generation_aborted(output.outputs[0])
    assert caught.value.kind == "insufficient_memory"
