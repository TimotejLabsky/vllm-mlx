"""Per-request samplers on the batched LLM path (PATCHES.md #33).

Before this patch the LLM scheduler baked the FIRST request's
(temperature, top_p, min_p) into a generator-level sampler — top_k was
dropped, presence_penalty never became a processor, and a request whose
params differed from the running batch silently sampled with stale
settings. mlx-lm's BatchGenerator applies per-sequence samplers row-wise;
these tests pin that every insert now carries the request's own sampler.
"""

from unittest.mock import MagicMock

import vllm_mlx.scheduler as scheduler_mod
from vllm_mlx.request import Request, SamplingParams
from vllm_mlx.scheduler import Scheduler, SchedulerConfig


def _make_scheduler() -> Scheduler:
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.encode = lambda x: list(range(len(x.split())))
    tokenizer.eos_token_id = 0
    config = SchedulerConfig(max_num_seqs=4, enable_prefix_cache=False)
    return Scheduler(model, tokenizer, config)


def _queue_request(scheduler, params: SamplingParams, rid="req-000000000001"):
    request = Request(
        request_id=rid,
        prompt="a b c",
        sampling_params=params,
        prompt_token_ids=[1, 2, 3],
        num_prompt_tokens=3,
    )
    scheduler.waiting.append(request)
    # Match _current_sampler_params so _ensure_batch_generator keeps the mock.
    scheduler._current_sampler_params = (
        params.temperature,
        params.top_p,
        params.min_p,
    )
    scheduler.batch_generator = MagicMock()
    scheduler.batch_generator.insert.return_value = [7]
    return request


def test_insert_carries_per_request_sampler(monkeypatch):
    captured = {}

    def fake_make_sampler(**kwargs):
        captured.update(kwargs)
        return "sampler-sentinel"

    monkeypatch.setattr(scheduler_mod, "make_sampler", fake_make_sampler)

    scheduler = _make_scheduler()
    params = SamplingParams(
        max_tokens=32, temperature=0.3, top_p=0.85, top_k=40, min_p=0.05
    )
    _queue_request(scheduler, params)

    scheduled = scheduler._schedule_waiting()

    assert len(scheduled) == 1
    assert captured == {"temp": 0.3, "top_p": 0.85, "min_p": 0.05, "top_k": 40}
    insert_kwargs = scheduler.batch_generator.insert.call_args.kwargs
    assert insert_kwargs["samplers"] == ["sampler-sentinel"]


def test_presence_penalty_becomes_processor(monkeypatch):
    calls = []

    def fake_make_logits_processors(**kwargs):
        calls.append(kwargs)
        return [lambda tokens, logits: logits]

    monkeypatch.setattr(
        scheduler_mod, "make_logits_processors", fake_make_logits_processors
    )

    scheduler = _make_scheduler()
    params = SamplingParams(max_tokens=32, presence_penalty=0.5)
    _queue_request(scheduler, params)

    scheduled = scheduler._schedule_waiting()

    assert len(scheduled) == 1
    assert {"presence_penalty": 0.5} in calls
    insert_kwargs = scheduler.batch_generator.insert.call_args.kwargs
    assert len(insert_kwargs["logits_processors"][0]) == 1


def test_no_penalties_keeps_empty_processor_list():
    scheduler = _make_scheduler()
    params = SamplingParams(max_tokens=32)
    _queue_request(scheduler, params)

    scheduled = scheduler._schedule_waiting()

    assert len(scheduled) == 1
    insert_kwargs = scheduler.batch_generator.insert.call_args.kwargs
    assert insert_kwargs["logits_processors"] == [[]]
    assert len(insert_kwargs["samplers"]) == 1
