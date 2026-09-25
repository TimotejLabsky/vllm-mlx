"""#115 (port of upstream #796): MLLM usage.prompt_tokens reports the
processor-expanded prompt (vision tokens included), not add_request's
text-only estimate.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx

from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator, MLLMBatchResponse
from vllm_mlx.mllm_scheduler import MLLMScheduler
from vllm_mlx.request import RequestStatus


def _resp(token, finish=None, prompt_tokens=None, uid=0):
    return MLLMBatchResponse(
        uid=uid,
        request_id="req-1",
        token=token,
        logprobs=mx.array([0.0]),
        finish_reason=finish,
        prompt_tokens=prompt_tokens,
    )


def _scheduler(estimate):
    sched = MLLMScheduler.__new__(MLLMScheduler)
    sched._detokenizer_pool = {}
    sched.uid_to_request_id = {0: "req-1"}
    sched.total_prompt_tokens = estimate  # accrued at schedule time
    sched.total_completion_tokens = 0
    sched.num_requests_processed = 0
    tok = MagicMock()
    tok.decode.return_value = "x"
    sched.processor = SimpleNamespace(tokenizer=tok)
    req = MagicMock()
    req.request_id = "req-1"
    req.output_tokens = []
    req.num_output_tokens = 0
    req.num_prompt_tokens = estimate
    req.status = RequestStatus.RUNNING
    req.first_token_time = None
    req.mtp_drafts = 0
    req.mtp_accepted = 0
    sched.running = {"req-1": req}
    return sched, req


def test_expanded_prompt_length_replaces_text_estimate_once():
    sched, req = _scheduler(estimate=12)

    outputs, _ = sched._process_batch_responses([_resp(100, prompt_tokens=1036)])
    assert outputs[0].prompt_tokens == 1036
    assert req.num_prompt_tokens == 1036
    assert sched.total_prompt_tokens == 1036

    outputs, _ = sched._process_batch_responses(
        [_resp(101, finish="stop", prompt_tokens=1036)]
    )
    assert outputs[0].prompt_tokens == 1036
    assert sched.total_prompt_tokens == 1036  # corrected once, not re-added


def test_missing_prompt_tokens_keeps_the_estimate():
    sched, req = _scheduler(estimate=12)
    outputs, _ = sched._process_batch_responses([_resp(100)])
    assert outputs[0].prompt_tokens == 12
    assert sched.total_prompt_tokens == 12


def test_next_stamps_rows_from_before_and_after_the_step():
    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    leaving = SimpleNamespace(
        uids=[0], requests=[SimpleNamespace(input_ids=mx.zeros((1, 7)))]
    )
    joining = SimpleNamespace(
        uids=[1], requests=[SimpleNamespace(input_ids=mx.zeros((1, 900)))]
    )
    gen.active_batch = joining
    responses = [_resp(1, uid=0), _resp(2, uid=1), _resp(3, uid=9)]

    gen._stamp_prompt_tokens(responses, leaving)

    assert [r.prompt_tokens for r in responses] == [7, 900, None]
