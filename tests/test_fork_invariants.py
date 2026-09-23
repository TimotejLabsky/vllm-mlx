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


# ------------------- #106 queued restores and co-batch admission see real memory

GROWN = TOKENS + list(range(40000, 40300))  # a follow-up turn on TOKENS


def _lazy_kv(monkeypatch, **env):
    monkeypatch.setenv("VLLM_MLX_BATCHED_LAZY_RESTORE", "1")
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    kv = BatchedSystemKV(_FakeModel())
    kv.store("seed", TOKENS, _donor_at(len(TOKENS)))
    return kv


def _queued(rid, tokens):
    return SimpleNamespace(
        request_id=rid,
        prompt_token_ids=list(tokens),
        num_prompt_tokens=len(tokens),
        output_token_ids=[],
        prompt_cache=None,
        cached_tokens=0,
        remaining_tokens=None,
        cache_hit_type=None,
    )


def test_106_peek_matches_fetch_but_builds_and_counts_nothing(monkeypatch):
    kv = _lazy_kv(monkeypatch)
    kv.store("other", DISJOINT, _donor_at(len(DISJOINT)))  # now the MRU entry
    before = (kv.hits, kv.misses, kv.tokens_saved, dict(kv._pending))

    pos = kv.peek(GROWN)

    assert pos > 0
    assert (kv.hits, kv.misses, kv.tokens_saved, dict(kv._pending)) == before
    # ...but it LRU-touched the matched chain, so it outlives the idle one
    assert list(kv._entries.values())[-1]["tokens"] == TOKENS
    eager = kv.fetch(GROWN, request_id="eager")
    assert eager is not None and eager[2] == pos  # same position fetch uses
    assert kv.peek(list(range(70000, 70400))) == 0


def test_106_lazy_add_request_pins_no_memory_until_admission(monkeypatch):
    kv = _lazy_kv(monkeypatch)
    request = _queued("lazy", GROWN)

    bkv.fetch_for_request(kv, request)

    assert request.cache_hit_type == "system_kv_pending"
    assert request.prompt_cache is None  # the multi-GB copy does not exist yet
    assert request.cached_tokens > 0
    assert request.remaining_tokens == GROWN[request.cached_tokens :]
    assert kv.hits == 0 and "lazy" not in kv._pending

    bkv.materialize_pending_restore(SimpleNamespace(hybrid_kv=kv), request)

    assert request.cache_hit_type == "system_kv"
    assert request.prompt_cache is not None
    assert kv.hits == 1 and kv.lazy_restores == 1
    assert "lazy" in kv._restore_source  # grow-on-HIT linkage intact


def test_106_a_lazy_restore_is_byte_identical_to_an_eager_one(monkeypatch):
    import mlx.core as mx

    lazy_kv = _lazy_kv(monkeypatch)
    lazy = _queued("lazy", GROWN)
    bkv.fetch_for_request(lazy_kv, lazy)
    bkv.materialize_pending_restore(SimpleNamespace(hybrid_kv=lazy_kv), lazy)

    monkeypatch.setenv("VLLM_MLX_BATCHED_LAZY_RESTORE", "0")
    eager_kv = BatchedSystemKV(_FakeModel())
    eager_kv.store("seed", TOKENS, _donor_at(len(TOKENS)))
    eager = _queued("eager", GROWN)
    bkv.fetch_for_request(eager_kv, eager)

    assert eager.cache_hit_type == "system_kv"  # inert by default = old path
    assert lazy.cached_tokens == eager.cached_tokens
    assert lazy.remaining_tokens == eager.remaining_tokens
    for lazy_layer, eager_layer in zip(lazy.prompt_cache, eager.prompt_cache):
        lazy_state, eager_state = lazy_layer.state, eager_layer.state
        assert len(lazy_state) == len(eager_state)
        for a, b in zip(lazy_state, eager_state):
            if a is None or b is None:
                assert a is b
            else:
                assert a.shape == b.shape and bool(mx.array_equal(a, b))


def test_106_an_entry_evicted_during_the_wait_degrades_to_a_miss(monkeypatch):
    kv = _lazy_kv(monkeypatch, VLLM_MLX_SYSTEM_KV_SLOTS=1)
    request = _queued("waited", GROWN)
    bkv.fetch_for_request(kv, request)
    assert request.cache_hit_type == "system_kv_pending"

    kv.store("other", DISJOINT, _donor_at(len(DISJOINT)))  # evicts the match

    bkv.materialize_pending_restore(SimpleNamespace(hybrid_kv=kv), request)

    assert request.cache_hit_type == "miss" and request.prompt_cache is None
    assert request.cached_tokens == 0 and request.remaining_tokens == GROWN
    assert kv.lazy_restore_misses == 1


def _projected_kv(monkeypatch, base_mb, ram_floor_mb=0, armed=True):
    """Mocked allocator: active = base + 20 MB per resident entry; limit is
    95% of a 100 MB ceiling; bytes/token floor 16 KB; chunk transient 8 MB."""
    import mlx.core as mx

    monkeypatch.setenv("VLLM_MLX_BATCHED_SOLO_TRANSIENT_MB", "8")
    monkeypatch.setenv("VLLM_MLX_BATCHED_BPT_FLOOR_KB", "16")
    monkeypatch.setenv("VLLM_MLX_SYSTEM_KV_RAM_MB", str(ram_floor_mb))
    if armed:
        monkeypatch.setenv("VLLM_MLX_BATCHED_PROJECTED_ADMISSION", "1")
    kv = BatchedSystemKV(_FakeModel())
    monkeypatch.setattr(
        mx, "get_active_memory", lambda: _mb(base_mb + _ENTRY_MB * len(kv._entries))
    )
    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": _mb(100)}
    )
    monkeypatch.setattr(mx, "clear_cache", lambda: None)
    return kv


def _charge_entries(kv):
    """Tell the bag its entries cost what the mocked allocator charges."""
    for entry in kv._entries.values():
        entry["bytes"] = _mb(_ENTRY_MB)


def _replay_0917(kv):
    """One deep row decoding + a deep candidate whose restore is still only
    matched: 2 x 700 tok merged (21.9 MB) + 50 tok growth + 650 tok restore
    (10.2 MB) + 8 MB transient ~= 41 MB on top of active."""
    running = {"deep": _queued("deep", range(700))}
    candidate = _queued("second", range(700))
    candidate.cache_hit_type = "system_kv_pending"
    candidate.cached_tokens = 650
    candidate.remaining_tokens = list(range(50))
    return SimpleNamespace(hybrid_kv=kv, running=running), candidate


def test_106_projection_is_inert_unless_armed(monkeypatch):
    kv = _projected_kv(monkeypatch, base_mb=90, armed=False)
    scheduler, candidate = _replay_0917(kv)
    assert bkv.should_defer_cobatch(scheduler, candidate) is False
    assert kv.projected_defers == 0


def test_106_the_0917_signature_makes_room_instead_of_running_out(monkeypatch):
    """Three OOMs at 63.0-63.6 GB: a ~65-70K row decoding, the bag grown into
    idle headroom, a second deep row co-batched, a 67-72K restore realised for
    a request still waiting. Every gate passed - none looks at what the
    process already holds. Projected: 60 active + 41 = 101 > 95; shedding one
    cache entry (-20) fits, so the row is admitted WITH room made."""
    kv = _projected_kv(monkeypatch, base_mb=20)
    kv.store("r1", TOKENS, _donor_at(len(TOKENS)))
    kv.store("r2", DISJOINT, _donor_at(len(DISJOINT)))
    _charge_entries(kv)
    scheduler, candidate = _replay_0917(kv)

    assert bkv.should_defer_cobatch(scheduler, candidate) is False

    assert kv.stats()["entry_count"] == 1  # one shed, not the whole bag
    assert kv.projected_relief_passes == 1 and kv.projected_defers == 0


def test_106_what_cannot_fit_waits_and_is_never_rejected(monkeypatch, caplog):
    kv = _projected_kv(monkeypatch, base_mb=60)
    kv.store("r1", TOKENS, _donor_at(len(TOKENS)))
    scheduler, candidate = _replay_0917(kv)

    with caplog.at_level("INFO"):
        assert bkv.should_defer_cobatch(scheduler, candidate) is True

    assert kv.projected_defers == 1 and kv.admission_deferrals == 1
    assert kv.solo_rejections == 0
    # shedding could not have closed the gap, so NOTHING was shed: the cache
    # (incl. the entry this request will restore from) is intact
    assert kv.stats()["entry_count"] == 1 and kv.projected_relief_passes == 0
    assert "projected peak" in caplog.text and "restore" in caplog.text


def test_106_room_is_made_only_down_to_the_ram_floor(monkeypatch):
    """A second seat is not worth an empty cache: shedding stops at the RAM
    floor, and the row waits instead."""
    kv = _projected_kv(monkeypatch, base_mb=20, ram_floor_mb=25)
    kv.store("r1", TOKENS, _donor_at(len(TOKENS)))
    kv.store("r2", DISJOINT, _donor_at(len(DISJOINT)))
    third = list(range(50000, 50800))
    kv.store("r3", third, _donor_at(len(third)))
    _charge_entries(kv)
    scheduler, candidate = _replay_0917(kv)

    # 80 active + 41 = 121 > 95. Bag 60 -> 40 -> 20 MB; 20 <= the 25 MB floor
    # stops it with one entry left: 40 + 41 = 81 fits.
    assert bkv.should_defer_cobatch(scheduler, candidate) is False
    assert kv.stats()["entry_count"] == 1


def test_106_a_deferred_request_is_not_materialised_an_admitted_one_is(monkeypatch):
    """The scheduler hook sits past every gate: deferral must not build the
    copy (that was the unbudgeted memory), admission must."""
    import mlx.core as mx
    from vllm_mlx.request import Request, SamplingParams

    monkeypatch.setenv("VLLM_MLX_BATCHED_LAZY_RESTORE", "1")
    monkeypatch.setenv("VLLM_MLX_BATCHED_PROJECTED_ADMISSION", "1")
    monkeypatch.setenv("VLLM_MLX_BATCHED_SOLO_TRANSIENT_MB", "8")
    monkeypatch.setenv("VLLM_MLX_BATCHED_BPT_FLOOR_KB", "16")
    scheduler = _make_scheduler(monkeypatch)
    kv = scheduler.hybrid_kv
    kv.store("seed", TOKENS, _donor_at(len(TOKENS)))
    active = {"mb": 90}
    monkeypatch.setattr(mx, "get_active_memory", lambda: _mb(active["mb"]))
    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": _mb(100)}
    )
    monkeypatch.setattr(scheduler, "_ensure_batch_generator", lambda params: None)

    request = Request(
        request_id="queued",
        prompt="x",
        sampling_params=SamplingParams(max_tokens=8),
        prompt_token_ids=list(GROWN),
        num_prompt_tokens=len(GROWN),
    )
    bkv.fetch_for_request(kv, request)
    scheduler.requests["queued"] = request
    scheduler.waiting.append(request)
    _running_request(scheduler, "busy", 61, 1000)

    scheduler._schedule_waiting()  # 90 MB active: the projection defers it
    assert request.cache_hit_type == "system_kv_pending"
    assert request.prompt_cache is None and kv.lazy_restores == 0

    active["mb"] = 10
    scheduler._schedule_waiting()  # fits now: past the gates -> materialised
    assert request.cache_hit_type == "system_kv"
    assert request.prompt_cache is not None and kv.lazy_restores == 1


# ------------------- #107 a queued request keeps its entry, or gets it from SSD


def test_107_a_lazy_miss_falls_back_to_the_ssd_tier(monkeypatch, tmp_path):
    """2026-09-21 stress test: a deep follow-up matched its chain at enqueue,
    waited ~10 min behind the KV budget, lost the RAM entry - and re-prefilled
    60K tokens cold (18 min) although the entry sat on SSD, 0.5 s away. The
    eager miss path probes SSD; the lazy one did not."""
    from tests.test_batched_system_kv import _make_ssd_cache

    monkeypatch.setenv("VLLM_MLX_BATCHED_LAZY_RESTORE", "1")
    writer = _make_ssd_cache(monkeypatch, tmp_path)
    writer.store("seed", TOKENS, _donor_at(len(TOKENS)))
    writer.close()  # drain the spill to disk

    kv = _make_ssd_cache(monkeypatch, tmp_path)
    kv.store("seed", TOKENS, _donor_at(len(TOKENS)))  # resident copy
    request = _queued("waited", GROWN)
    bkv.fetch_for_request(kv, request)
    assert request.cache_hit_type == "system_kv_pending"
    peeked = request.cached_tokens

    with kv._lock:  # the wait: RAM loses the entry (relief takes pins too)
        kv._entries.clear()
    assert kv.peek(GROWN) == 0

    bkv.materialize_pending_restore(SimpleNamespace(hybrid_kv=kv), request)

    assert request.cache_hit_type == "system_kv"
    assert request.cached_tokens == peeked and request.prompt_cache is not None
    assert kv.lazy_ssd_fallbacks == 1 and kv.lazy_restore_misses == 0
    assert kv.stats()["ssd_promotes"] == 1
    kv.close()


def test_107_no_ssd_probe_when_the_entry_is_still_resident(monkeypatch, tmp_path):
    from tests.test_batched_system_kv import _make_ssd_cache

    monkeypatch.setenv("VLLM_MLX_BATCHED_LAZY_RESTORE", "1")
    kv = _make_ssd_cache(monkeypatch, tmp_path)
    kv.store("seed", TOKENS, _donor_at(len(TOKENS)))
    request = _queued("prompt", GROWN)
    bkv.fetch_for_request(kv, request)
    kv.check_ssd = MagicMock(side_effect=AssertionError("disk probe on a RAM hit"))

    bkv.materialize_pending_restore(SimpleNamespace(hybrid_kv=kv), request)

    assert request.cache_hit_type == "system_kv" and kv.lazy_ssd_fallbacks == 0
    kv.close()


def test_107_a_queued_requests_entry_is_pinned_until_it_is_admitted(monkeypatch):
    kv = _lazy_kv(monkeypatch)
    request = _queued("waiting", GROWN)
    bkv.fetch_for_request(kv, request)
    assert kv.stats()["pinned_entries"] == 1

    bkv.materialize_pending_restore(SimpleNamespace(hybrid_kv=kv), request)
    assert kv.stats()["pinned_entries"] == 0  # released at admission

    aborted = _queued("aborted", GROWN)
    bkv.fetch_for_request(kv, aborted)
    assert kv.stats()["pinned_entries"] == 1
    kv.discard_pending("aborted")  # every abort path goes through this
    assert kv.stats()["pinned_entries"] == 0 and kv._peeked == {}


def test_107_budget_eviction_prefers_entries_nobody_is_waiting_for(monkeypatch):
    """The 09-21 mechanism: an SSD promote for one chain pushed out the LRU
    entry - which a queued request had matched minutes earlier."""
    monkeypatch.setenv("VLLM_MLX_SYSTEM_KV_SLOTS", "2")
    kv = _lazy_kv(monkeypatch)  # holds TOKENS
    kv.store("other", DISJOINT, _donor_at(len(DISJOINT)))
    bkv.fetch_for_request(kv, _queued("waiting", GROWN))  # pins TOKENS...
    assert kv.fetch(DISJOINT + [1, 2, 3]) is not None  # ...then DISJOINT is MRU

    third = list(range(50000, 50800))
    kv.store("third", third, _donor_at(len(third)))

    kept = sorted(e["tokens"][0] for e in kv._entries.values())
    assert kept == [TOKENS[0], third[0]]  # the pinned LRU entry survived

    # with every older entry pinned the budget still holds - and the entry
    # just inserted is never its own victim
    bkv.fetch_for_request(kv, _queued("waiting-2", third + [9, 9, 9]))
    fourth = list(range(60000, 60800))
    kv.store("fourth", fourth, _donor_at(len(fourth)))
    assert kv.stats()["entry_count"] == 2
    assert fourth[0] in [e["tokens"][0] for e in kv._entries.values()]


def test_107_making_room_spares_pinned_entries_but_relief_does_not(monkeypatch):
    kv = _projected_kv(monkeypatch, base_mb=20)
    monkeypatch.setenv("VLLM_MLX_BATCHED_LAZY_RESTORE", "1")
    kv.lazy_restore = True
    kv.store("r1", TOKENS, _donor_at(len(TOKENS)))
    kv.store("r2", DISJOINT, _donor_at(len(DISJOINT)))
    _charge_entries(kv)
    bkv.fetch_for_request(kv, _queued("queued", GROWN))  # pins r1 (the LRU)
    assert kv.fetch(DISJOINT + [1]) is not None  # r2 becomes MRU, r1 LRU
    scheduler, candidate = _replay_0917(kv)

    # 60 active + 41 > 95: one entry must go. LRU is r1 - but it is pinned.
    assert bkv.should_defer_cobatch(scheduler, candidate) is False
    assert [e["tokens"][0] for e in kv._entries.values()] == [TOKENS[0]]

    # only a pinned entry left, still over -> wait, do not take it
    kv2 = _projected_kv(monkeypatch, base_mb=45)
    kv2.lazy_restore = True
    kv2.store("r1", TOKENS, _donor_at(len(TOKENS)))
    _charge_entries(kv2)
    bkv.fetch_for_request(kv2, _queued("queued", GROWN))
    scheduler2, candidate2 = _replay_0917(kv2)
    assert bkv.should_defer_cobatch(scheduler2, candidate2) is True  # 65+41 > 95
    assert kv2.stats()["entry_count"] == 1 and kv2.projected_relief_passes == 0

    # ...but against a real OOM everything is fair game
    assert kv2._drop_lru_entry() is True and kv2.stats()["entry_count"] == 0


def test_107_solo_guard_takes_pinned_entries_rather_than_reject(monkeypatch):
    """A false 503 is worse than a queued request's cold prefill."""
    kv = _solo_kv(monkeypatch, base_mb=20)
    kv.store("r1", TOKENS, _donor_at(len(TOKENS)))
    kv.peek(GROWN, request_id="queued")  # pinned
    # active 40; need 60 -> 100 > 95. Sparing the pin would reject; it must not.
    assert kv.solo_prefill_verdict(800) is None
    assert kv.stats()["entry_count"] == 0 and kv.solo_rejections == 0


# ---------- #108 an entry whose spill is still queued is not worth evicting


def _busy_ssd_kv(monkeypatch, tmp_path, **env):
    """A bag whose SSD writer never gets an idle window and may not write
    while busy: every spill stays queued until the test says otherwise."""
    monkeypatch.setenv("VLLM_MLX_SSD_SYSTEM_KV_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_MLX_SSD_SYSTEM_KV_BUSY_WRITE_S", "0")
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    busy = {"flag": True}
    kv = BatchedSystemKV(
        _FakeModel(),
        tokenizer=SimpleNamespace(name_or_path="unit/test-model"),
        idle_check=lambda: not busy["flag"],
    )
    return kv, busy


def _entry_for(kv, first_token):
    return next(e for e in kv._entries.values() if e["tokens"][0] == first_token)


def test_108_spill_pending_tracks_the_queue_written_or_dropped(monkeypatch, tmp_path):
    kv, busy = _busy_ssd_kv(monkeypatch, tmp_path)
    try:
        kv.store("a", TOKENS, _donor_at(len(TOKENS)))
        entry = _entry_for(kv, TOKENS[0])
        assert entry["spill_pending"] is True
        assert kv.stats()["spill_pending_entries"] == 1

        busy["flag"] = False  # an idle window: the write lands
        assert _wait_for(lambda: entry["spill_pending"] is False, 10.0)
        assert kv._ssd.lookup_prefix(tuple(TOKENS)) is not None

        busy["flag"] = True
        kv.store("b", DISJOINT, _donor_at(len(DISJOINT)))
        other = _entry_for(kv, DISJOINT[0])
        assert other["spill_pending"] is True
        assert kv._ssd.drop_backlog() > 0  # relief takes the backlog
        assert _wait_for(lambda: other["spill_pending"] is False, 5.0)
        assert kv.stats()["spill_pending_entries"] == 0
    finally:
        kv.close()


def test_108_a_spill_the_queue_refused_is_not_marked_pending(monkeypatch, tmp_path):
    kv, _busy = _busy_ssd_kv(
        monkeypatch, tmp_path, VLLM_MLX_SSD_SYSTEM_KV_MAX_QUEUED_GB="0.0000001"
    )
    try:
        kv.store("a", TOKENS, _donor_at(len(TOKENS)))  # empty queue admits one
        kv.store("b", DISJOINT, _donor_at(len(DISJOINT)))  # over the cap: dropped
        assert _entry_for(kv, TOKENS[0])["spill_pending"] is True
        assert _entry_for(kv, DISJOINT[0])["spill_pending"] is False
        assert kv.stats()["ssd"]["spill_drops"] == 1
    finally:
        kv.close()


def test_108_eviction_spares_the_entry_whose_spill_is_still_queued(
    monkeypatch, tmp_path
):
    """2026-09-21 stress re-run: the last chain's follow-up arrived ~10 s after
    its turn finished. The dynamic budget had just evicted that chain's RAM
    entry while its spill was still queued - so it was in neither RAM nor the
    SSD index, and 60K tokens were re-prefilled cold. The eviction had freed
    nothing: the write queue holds the same arrays."""
    kv, _busy = _busy_ssd_kv(monkeypatch, tmp_path, VLLM_MLX_SYSTEM_KV_SLOTS="2")
    try:
        kv.store("a", TOKENS, _donor_at(len(TOKENS)))  # LRU, spill queued
        kv.store("b", DISJOINT, _donor_at(len(DISJOINT)))
        _entry_for(kv, DISJOINT[0])["spill_pending"] = False  # b's write landed

        third = list(range(50000, 50800))
        kv.store("c", third, _donor_at(len(third)))

        kept = sorted(e["tokens"][0] for e in kv._entries.values())
        assert kept == [TOKENS[0], third[0]]  # b went, the queued LRU survived
        assert kv.peek(GROWN) > 0  # ...so its follow-up restores instead of 18 min
    finally:
        kv.close()


def test_108_a_pinned_entry_goes_before_a_spill_pending_one(monkeypatch, tmp_path):
    """With everything spoken for the budget still has to hold: a pinned entry
    at least frees memory, a spill-pending one frees nothing."""
    kv, _busy = _busy_ssd_kv(monkeypatch, tmp_path, VLLM_MLX_SYSTEM_KV_SLOTS="2")
    try:
        kv.store("a", TOKENS, _donor_at(len(TOKENS)))  # LRU, spill queued
        kv.store("b", DISJOINT, _donor_at(len(DISJOINT)))
        _entry_for(kv, DISJOINT[0])["spill_pending"] = False
        kv.peek(DISJOINT + [1, 2, 3], request_id="queued")  # b is pinned (and MRU)
        assert kv.fetch(GROWN) is not None  # a becomes MRU, b LRU

        third = list(range(50000, 50800))
        kv.store("c", third, _donor_at(len(third)))

        kept = sorted(e["tokens"][0] for e in kv._entries.values())
        assert kept == [TOKENS[0], third[0]]  # the pinned one went, not the queued
    finally:
        kv.close()


def test_108_make_room_does_not_count_spill_pending_bytes_as_sheddable(monkeypatch):
    kv = _projected_kv(monkeypatch, base_mb=20)
    kv.store("r1", TOKENS, _donor_at(len(TOKENS)))
    kv.store("r2", DISJOINT, _donor_at(len(DISJOINT)))
    _charge_entries(kv)
    for entry in kv._entries.values():
        entry["spill_pending"] = True  # evicting them would free nothing
    scheduler, candidate = _replay_0917(kv)

    # 60 active + 41 > 95 and nothing sheddable -> wait, cache untouched
    assert bkv.should_defer_cobatch(scheduler, candidate) is True
    assert kv.stats()["entry_count"] == 2 and kv.projected_relief_passes == 0

    # relief is a different matter: against a real OOM it still takes them
    assert kv._drop_lru_entry() is True


# ------------------------- #83/#109 _normalize_messages vs upstream #774 guard
def test_83_109_normalize_merges_assistant_pairs_never_tool_results():
    """Upstream #774 refuses any merge where either side has tool_calls; the
    fork keeps #83's assistant text-turn + tool-call-turn merge (templates
    need alternating roles) and shares only the role allowlist (#109). A
    rebase that takes #774's condition wholesale goes red on the first half."""
    from vllm_mlx.server import _normalize_messages

    tc = [
        {"id": "c1", "type": "function", "function": {"name": "a", "arguments": "{}"}}
    ]
    merged = _normalize_messages(
        [
            {"role": "assistant", "content": "Let me check."},
            {"role": "assistant", "content": "", "tool_calls": tc},
        ]
    )
    assert len(merged) == 1 and merged[0]["tool_calls"] == tc

    results = [
        {"role": "tool", "tool_call_id": "c1", "content": "AAA"},
        {"role": "tool", "tool_call_id": "c2", "content": "BBB"},
    ]
    assert _normalize_messages(results) == results


# ---------------------- #110 streaming tool calls are emitted once, by position
def test_110_routed_parsers_emit_each_streamed_call_once():
    """Upstream owns these parsers and fixed the re-emit only in qwen (#774).
    A rebase that takes an upstream rewrite of any routed parser's streaming
    block — or reverts gemma4 to first-block-only parsing, or harmony to its
    (name, arguments) dedupe — goes red here."""
    from tests.test_tool_parser_stream_call_identity import FORMATS, _stream

    for name in ("glm47", "gemma4", "nemotron", "harmony", "auto"):
        first, second = FORMATS[name]
        calls, _ = _stream(name, [first, "\n", second, "\n", ""])
        assert [c["index"] for c in calls] == [0, 1], name
    # Two identical calls are two invocations, not one.
    first, _ = FORMATS["harmony"]
    calls, _ = _stream("harmony", [first, "<|start|>assistant" + first])
    assert [c["index"] for c in calls] == [0, 1]


# ------------------- #111 a closing marker split across deltas still fires
def test_111_routed_parsers_see_split_closing_markers():
    """Upstream's parsers trigger on `END in delta_text`; a rebase that takes
    an upstream rewrite of a routed parser's trigger goes red here."""
    from tests.test_tool_parser_stream_call_identity import A, B, FORMATS, _stream

    for name in ("glm47", "gemma4", "nemotron", "harmony", "auto"):
        first, second = FORMATS[name]
        text = first + "\n" + second
        calls, _ = _stream(name, [text[i : i + 1] for i in range(len(text))])
        assert [c["index"] for c in calls] == [0, 1], name
        assert [json.loads(c["function"]["arguments"]) for c in calls] == [A, B]
