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
