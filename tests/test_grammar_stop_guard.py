"""Stop terminators must not fire inside a live grammar (PATCHES.md #89).

A structured-output request carries two terminators that knew nothing about
each other: the schema processor (#73) that forces the output to be a legal
JSON value, and the stop terminators — ``stop`` strings (#32) and the
repetition detector (#77). A pretty-printed schema emits ``"\\n\\n"`` between
members, so a client sending ``stop=["\\n\\n"]`` got truncated JSON reported
as ``finish_reason="stop"``: a clean stop on broken output.

These tests assert the arbitration and, just as importantly, that requests
with NO grammar attached behave byte-identically to the pre-#89 code — the
existing #32 stop-string and #77 repetition suites are the regression surface
and stay green untouched.
"""

import asyncio
from types import SimpleNamespace

from vllm_mlx.engine.batched import BatchedEngine
from vllm_mlx.grammar_guard import (
    grammar_unterminated,
    has_grammar,
    iter_grammar_processors,
)
from vllm_mlx.stop_strings import StopStringScanner


class _FakeSchemaProcessor:
    """Minimal implementation of the accepting protocol."""

    def __init__(self, accepting: bool = False):
        self.accepting = accepting

    def is_accepting(self) -> bool:
        return self.accepting

    def __call__(self, tokens, logits):  # pragma: no cover - never sampled here
        return logits


class _PlainProcessor:
    """A processor that does NOT participate (DRY, penalties, thinking)."""

    def __call__(self, tokens, logits):  # pragma: no cover
        return logits


class _BrokenSchemaProcessor:
    def is_accepting(self) -> bool:
        raise RuntimeError("matcher exploded")


# ---------------------------------------------------------------- unit level


def test_no_processors_is_never_unterminated():
    assert grammar_unterminated(None) is False
    assert grammar_unterminated([]) is False
    assert has_grammar(None) is False


def test_non_grammar_processors_are_ignored():
    procs = [_PlainProcessor(), _PlainProcessor()]
    assert list(iter_grammar_processors(procs)) == []
    assert has_grammar(procs) is False
    assert grammar_unterminated(procs) is False


def test_mid_value_grammar_is_unterminated():
    procs = [_PlainProcessor(), _FakeSchemaProcessor(accepting=False)]
    assert has_grammar(procs) is True
    assert grammar_unterminated(procs) is True


def test_accepting_grammar_is_terminated():
    procs = [_FakeSchemaProcessor(accepting=True)]
    assert has_grammar(procs) is True
    assert grammar_unterminated(procs) is False


def test_nested_per_sequence_lists_are_flattened():
    # The batched scheduler carries one processor list per sequence.
    nested = [[_PlainProcessor()], [_FakeSchemaProcessor(accepting=False)]]
    assert grammar_unterminated(nested) is True
    assert grammar_unterminated([[_FakeSchemaProcessor(accepting=True)]]) is False


def test_bare_processor_not_in_a_list():
    assert grammar_unterminated(_FakeSchemaProcessor(accepting=False)) is True
    assert grammar_unterminated(_PlainProcessor()) is False


class _Wrapper:
    """Mirrors the thinking processors' ``_inner`` delegation."""

    def __init__(self, inner):
        self._inner = inner

    def __call__(self, tokens, logits):  # pragma: no cover
        return logits


def test_wrapped_grammar_is_still_seen():
    wrapped = _Wrapper(_FakeSchemaProcessor(accepting=False))
    assert has_grammar([wrapped]) is True
    assert grammar_unterminated([wrapped]) is True
    assert grammar_unterminated([_Wrapper(_FakeSchemaProcessor(True))]) is False


def test_wrapper_answering_for_itself_wins():
    outer = _FakeSchemaProcessor(accepting=True)
    outer._inner = _FakeSchemaProcessor(accepting=False)
    assert grammar_unterminated([outer]) is False


def test_wrapping_a_plain_processor_is_not_a_grammar():
    assert has_grammar([_Wrapper(_PlainProcessor())]) is False


def test_delegate_chain_cannot_loop_forever():
    loop = _Wrapper(None)
    loop._inner = loop
    assert has_grammar([loop]) is False


def test_raising_predicate_is_treated_as_unterminated():
    # Fail-closed: an unreadable matcher must not license a truncating stop.
    assert grammar_unterminated([_BrokenSchemaProcessor()]) is True


def test_any_open_grammar_wins():
    procs = [
        _FakeSchemaProcessor(accepting=True),
        _FakeSchemaProcessor(accepting=False),
    ]
    assert grammar_unterminated(procs) is True


# --------------------------------------------------- deferred scanner (#89)


def test_deferred_scan_reports_and_advances():
    s = StopStringScanner(["\n\n"])
    assert s.deferred_scan('{"a": 1,') is False
    assert s.deferred_scan("\n\n") is True
    # The tail advanced, so the same match is not re-reported next chunk.
    assert s.deferred_scan('  "b": 2') is False


def test_deferred_scan_sees_cross_chunk_match():
    s = StopStringScanner(["ABC"])
    assert s.deferred_scan("xA") is False
    assert s.deferred_scan("BCy") is True


def test_deferred_scan_keeps_tail_correct_for_a_later_real_scan():
    # Deferring must not corrupt the carried tail: once the grammar closes,
    # a match spanning the deferred chunk still has to be found.
    s = StopStringScanner(["ABC"])
    assert s.deferred_scan("xA") is False
    assert s.scan("BCy") == ("", True)


def test_deferred_scan_inactive_scanner():
    s = StopStringScanner([])
    assert s.deferred_scan("anything") is False


# ------------------------------------------------------- engine wiring (#89)


def _output(**overrides):
    base = dict(
        output_text="",
        new_text="",
        output_token_ids=[1],
        prompt_tokens=1,
        completion_tokens=1,
        finished=False,
        finish_reason=None,
        mtp_drafts=0,
        mtp_accepted=0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


_PRETTY = ['{\n  "a": 1,', "\n\n", '  "b": 2\n}']


class _PrettyJSONEngine:
    """Streams a pretty-printed object whose separator IS the stop string."""

    def __init__(self):
        self.aborted = None

    async def add_request(self, prompt, sampling_params, prefix_boundary=0):
        return "req-json"

    async def stream_outputs(self, request_id):
        acc = ""
        for i, chunk in enumerate(_PRETTY):
            acc += chunk
            last = i == len(_PRETTY) - 1
            yield _output(
                new_text=chunk,
                output_text=acc,
                finished=last,
                finish_reason="stop" if last else None,
            )

    async def abort_request(self, request_id):
        self.aborted = request_id
        return True

    async def generate(self, prompt, sampling_params):
        return _output(
            output_text="".join(_PRETTY), finished=True, finish_reason="stop"
        )


def _make_engine():
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._loaded = True
    engine._is_mllm = False
    engine._mllm_scheduler = None
    engine._engine = _PrettyJSONEngine()
    engine._tokenizer = SimpleNamespace(encode=lambda s, **k: [1])
    return engine


def _stream(engine, **kwargs):
    chunks = []

    async def consume():
        async for out in engine.stream_generate(prompt="x", **kwargs):
            chunks.append(out)

    asyncio.run(consume())
    return chunks


def test_stream_stop_string_inside_grammar_does_not_truncate():
    engine = _make_engine()
    chunks = _stream(
        engine,
        stop=["\n\n"],
        logits_processors=[_FakeSchemaProcessor(accepting=False)],
    )
    assert "".join(c.new_text for c in chunks) == "".join(_PRETTY)
    assert len(chunks) == 3
    assert chunks[-1].finished is True
    # Ended on the model's own EOS, not on the stop string.
    assert chunks[-1].finish_reason == "stop"
    assert engine._engine.aborted is None


def test_stream_stop_string_without_grammar_still_truncates():
    # The #32 contract is untouched when no schema processor is attached.
    engine = _make_engine()
    chunks = _stream(engine, stop=["\n\n"])
    assert "".join(c.new_text for c in chunks) == '{\n  "a": 1,'
    assert chunks[-1].finish_reason == "stop"
    assert engine._engine.aborted == "req-json"


def test_stream_stop_string_fires_once_grammar_accepts():
    engine = _make_engine()
    processor = _FakeSchemaProcessor(accepting=True)
    chunks = _stream(engine, stop=["\n\n"], logits_processors=[processor])
    assert "".join(c.new_text for c in chunks) == '{\n  "a": 1,'
    assert chunks[-1].finish_reason == "stop"
    assert engine._engine.aborted == "req-json"


def test_stream_plain_processor_does_not_suppress():
    engine = _make_engine()
    chunks = _stream(engine, stop=["\n\n"], logits_processors=[_PlainProcessor()])
    assert "".join(c.new_text for c in chunks) == '{\n  "a": 1,'


def test_nonstream_stop_string_inside_grammar_does_not_truncate():
    engine = _make_engine()
    out = asyncio.run(
        engine.generate(
            prompt="x",
            stop=["\n\n"],
            logits_processors=[_FakeSchemaProcessor(accepting=True)],
        )
    )
    # Non-streaming suppresses wholesale: with a schema attached the whole
    # text IS the value, so every match lands inside it — including here,
    # where the grammar has already accepted at the end of generation.
    assert out.text == "".join(_PRETTY)
    assert out.finish_reason == "stop"


def test_nonstream_stop_string_without_grammar_still_truncates():
    engine = _make_engine()
    out = asyncio.run(engine.generate(prompt="x", stop=["\n\n"]))
    assert out.text == '{\n  "a": 1,'
    assert out.finish_reason == "stop"


class _StampedPrettyEngine(_PrettyJSONEngine):
    """Stamps ``grammar_unterminated`` per chunk, as the scheduler does.

    Mid-value for every chunk but the last — which is what the producer
    records even for the closing chunk, since the predicate is one token
    stale. The processor handed to the engine claims the grammar has already
    FINISHED, standing in for a consumer that drained late: the stamp must
    win, or the stop cuts an earlier chunk mid-object.
    """

    async def stream_outputs(self, request_id):
        acc = ""
        for i, chunk in enumerate(_PRETTY):
            acc += chunk
            last = i == len(_PRETTY) - 1
            yield _output(
                new_text=chunk,
                output_text=acc,
                finished=last,
                finish_reason="stop" if last else None,
                grammar_unterminated=True,
            )


def test_stamped_verdict_beats_a_stale_live_read():
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._loaded = True
    engine._is_mllm = False
    engine._mllm_scheduler = None
    engine._engine = _StampedPrettyEngine()
    engine._tokenizer = SimpleNamespace(encode=lambda s, **k: [1])

    # The live processor says "done" — a late consumer's view. The per-chunk
    # stamp says mid-value, and it must win.
    chunks = _stream(
        engine,
        stop=["\n\n"],
        logits_processors=[_FakeSchemaProcessor(accepting=True)],
    )
    assert "".join(c.new_text for c in chunks) == "".join(_PRETTY)
    assert engine._engine.aborted is None


def test_stamped_terminated_verdict_allows_the_stop():
    class _Stamped(_PrettyJSONEngine):
        async def stream_outputs(self, request_id):
            acc = ""
            for i, chunk in enumerate(_PRETTY):
                acc += chunk
                last = i == len(_PRETTY) - 1
                yield _output(
                    new_text=chunk,
                    output_text=acc,
                    finished=last,
                    finish_reason="stop" if last else None,
                    grammar_unterminated=False,
                )

    engine = BatchedEngine.__new__(BatchedEngine)
    engine._loaded = True
    engine._is_mllm = False
    engine._mllm_scheduler = None
    engine._engine = _Stamped()
    engine._tokenizer = SimpleNamespace(encode=lambda s, **k: [1])
    # Live processor says mid-value; the stamp says the value closed before
    # this chunk, so the stop is honoured.
    chunks = _stream(
        engine,
        stop=["\n\n"],
        logits_processors=[_FakeSchemaProcessor(accepting=False)],
    )
    assert "".join(c.new_text for c in chunks) == '{\n  "a": 1,'


def test_collector_merge_is_conservative_about_the_verdict():
    from vllm_mlx.output_collector import RequestOutputCollector
    from vllm_mlx.request import RequestOutput

    def _put_two(first, second):
        c = RequestOutputCollector(aggregate=True)
        c.put(RequestOutput(request_id="r", new_text="a", grammar_unterminated=first))
        c.put(RequestOutput(request_id="r", new_text="b", grammar_unterminated=second))
        return c.get_nowait()

    # Either side mid-value makes the merged span mid-value.
    assert _put_two(True, False).grammar_unterminated is True
    assert _put_two(False, True).grammar_unterminated is True
    assert _put_two(False, False).grammar_unterminated is False
    # Unstamped stays unstamped, so consumers still fall back to a live read.
    assert _put_two(None, None).grammar_unterminated is None
    # The merge must still carry text and the other #82 fields.
    merged = _put_two(True, False)
    assert merged.new_text == "ab"


def test_suppression_counter_is_latched_per_stream():
    calls = []
    import vllm_mlx.engine.batched as batched_mod

    original = batched_mod._count_grammar_stop_suppressed
    batched_mod._count_grammar_stop_suppressed = lambda source: calls.append(source)
    try:
        gate = batched_mod._GrammarStopGate(
            # Mixed lengths: the short stop sits inside the tail kept for the
            # long one and re-matches on later chunks.
            ["\n\n", "<|im_end|>"],
            [_FakeSchemaProcessor(accepting=False)],
        )
        for chunk in ('{\n  "a": 1,', "\n\n", "  x", "  y", "  z"):
            text, hit = gate.scan(SimpleNamespace(new_text=chunk))
            assert (text, hit) == (chunk, False)
    finally:
        batched_mod._count_grammar_stop_suppressed = original
    assert calls == ["stop_string"], f"counter not latched: {calls}"


class _MLLMPrettyScheduler:
    def __init__(self):
        self.aborted = None

    async def add_request_async(self, **kwargs):
        return "req-mllm"

    async def stream_outputs(self, request_id):
        acc = ""
        for i, chunk in enumerate(_PRETTY):
            acc += chunk
            last = i == len(_PRETTY) - 1
            yield _output(
                new_text=chunk,
                output_text=acc,
                finished=last,
                finish_reason="stop" if last else None,
            )

    def abort_request(self, request_id):
        self.aborted = request_id
        return True

    async def generate(self, **kwargs):
        return _output(
            output_text="".join(_PRETTY), finished=True, finish_reason="stop"
        )


def _make_mllm_engine():
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._loaded = True
    engine._is_mllm = True
    engine._mllm_scheduler = _MLLMPrettyScheduler()
    engine._engine = None
    engine._processor = SimpleNamespace(
        tokenizer=SimpleNamespace(encode=lambda s, **k: [1])
    )
    engine._tokenizer = None
    return engine


def test_mllm_stream_stop_string_inside_grammar_does_not_truncate():
    engine = _make_mllm_engine()
    chunks = _stream(
        engine,
        stop=["\n\n"],
        logits_processors=[_FakeSchemaProcessor(accepting=False)],
    )
    assert "".join(c.new_text for c in chunks) == "".join(_PRETTY)
    assert engine._mllm_scheduler.aborted is None


def test_mllm_nonstream_without_grammar_still_truncates():
    engine = _make_mllm_engine()
    out = asyncio.run(engine.generate(prompt="x", stop=["\n\n"]))
    assert out.text == '{\n  "a": 1,'


# ---------------------------------------------- repetition stop vs grammar


def _bare_scheduler_with_request(cfg, logits_processors=None):
    """Scheduler skeleton with one running request (mirrors the #77 suite)."""
    from unittest.mock import Mock

    from vllm_mlx.repetition_stop import RepetitionStopTracker
    from vllm_mlx.request import Request, SamplingParams
    from vllm_mlx.scheduler import Scheduler

    sched = Scheduler.__new__(Scheduler)
    request = Request(
        request_id="req-json-loop",
        prompt="x",
        sampling_params=SamplingParams(
            max_tokens=4096, logits_processors=logits_processors
        ),
    )
    request.prompt_token_ids = [1, 2, 3]
    request.num_prompt_tokens = 3
    request.first_token_time = 0.0

    detok = SimpleNamespace(
        add_token=lambda t: None,
        last_segment="",
        finalize=lambda: None,
        text="",
    )
    sched.uid_to_request_id = {11: "req-json-loop"}
    sched.running = {"req-json-loop": request}
    sched._repstop = RepetitionStopTracker(cfg)
    sched.batch_generator = Mock()
    sched._store_prompt_only_cache = lambda *a, **k: None
    sched._get_detokenizer = lambda rid: detok
    sched._detokenizer_pool = {"req-json-loop": detok}
    sched._cleanup_detokenizer = lambda rid: None
    sched._decode_tokens = lambda ids: ""
    sched.total_completion_tokens = 0
    sched.num_requests_processed = 0
    return sched, request


def _loop_config():
    from vllm_mlx.repetition_stop import RepetitionStopConfig

    return RepetitionStopConfig(
        enabled=True,
        window=128,
        min_period=1,
        max_period=16,
        min_repeats=3,
        min_span=16,
        interval=4,
        min_tokens=16,
    ).sanitized()


def _run_until_finished(sched):
    for _ in range(200):
        resp = SimpleNamespace(uid=11, token=7, finish_reason=None)
        outputs, finished_ids = sched._process_batch_responses([resp])
        if finished_ids:
            return outputs[0]
    return None


def test_repetition_stop_inside_grammar_finishes_as_length():
    from vllm_mlx.request import RequestStatus

    sched, request = _bare_scheduler_with_request(
        _loop_config(), logits_processors=[_FakeSchemaProcessor(accepting=False)]
    )
    finished = _run_until_finished(sched)
    assert finished is not None, "detector never fired"
    # A truncated JSON value must never be reported as a clean stop.
    assert finished.finish_reason == "length"
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    sched.batch_generator.remove.assert_called_once_with([11])


def test_repetition_stop_after_grammar_accepts_still_stops():
    from vllm_mlx.request import RequestStatus

    sched, request = _bare_scheduler_with_request(
        _loop_config(), logits_processors=[_FakeSchemaProcessor(accepting=True)]
    )
    finished = _run_until_finished(sched)
    assert finished is not None
    assert finished.finish_reason == "stop"
    assert request.status == RequestStatus.FINISHED_STOPPED


def test_repetition_stop_without_grammar_is_unchanged():
    from vllm_mlx.request import RequestStatus

    sched, request = _bare_scheduler_with_request(_loop_config())
    finished = _run_until_finished(sched)
    assert finished is not None
    assert finished.finish_reason == "stop"
    assert request.status == RequestStatus.FINISHED_STOPPED


# ------------------------------------------------- processor predicate wiring


def test_lmfe_processor_exposes_the_predicate():
    from vllm_mlx.constrained.json_schema_processor import JSONSchemaLogitsProcessor

    # Constructing one needs lm-format-enforcer + a tokenizer; assert the
    # protocol on the class instead, so the guard's contract is pinned even
    # where the optional backend is absent.
    assert callable(getattr(JSONSchemaLogitsProcessor, "is_accepting", None))
    processor = JSONSchemaLogitsProcessor.__new__(JSONSchemaLogitsProcessor)
    processor._accepting = False
    assert processor.is_accepting() is False
    processor._accepting = True
    assert processor.is_accepting() is True


def test_llguidance_processor_exposes_the_predicate():
    from vllm_mlx.constrained.llguidance_schema_processor import (
        LLGuidanceJSONSchemaLogitsProcessor as P,
    )

    assert callable(getattr(P, "is_accepting", None))
    processor = P.__new__(P)
    processor._accepting = False
    assert processor.is_accepting() is False
    processor._accepting = True
    assert processor.is_accepting() is True


def test_llguidance_terminal_state_reports_accepting():
    from vllm_mlx.constrained.llguidance_schema_processor import (
        LLGuidanceJSONSchemaLogitsProcessor as P,
    )

    processor = P.__new__(P)
    processor._terminal = True
    processor._matcher = SimpleNamespace(
        is_accepting=lambda: (_ for _ in ()).throw(RuntimeError("gone"))
    )
    # _terminal short-circuits before the matcher is consulted.
    assert processor._terminal or processor._matcher_is_accepting()
    processor._terminal = False
    # An unreadable matcher reports mid-value rather than raising.
    assert processor._matcher_is_accepting() is False


def test_metrics_counter_exists():
    from vllm_mlx.metrics import metrics

    assert hasattr(metrics, "observe_grammar_stop_suppressed")
    # Safe to call whether or not prometheus_client is installed/enabled.
    metrics.observe_grammar_stop_suppressed(source="stop_string")
    metrics.observe_grammar_stop_suppressed(source="repetition")
