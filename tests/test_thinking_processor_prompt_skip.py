# SPDX-License-Identifier: Apache-2.0
"""The thinking phase machine must walk GENERATED tokens only, never the prompt.

mlx-lm hands logits processors the full sequence (prompt + generated). Walking
the prompt is not a cosmetic miscount: any prompt replaying an earlier
``<think>...</think>`` span — i.e. every multi-turn agent conversation — drove
the machine to CONTENT before the first generated token. CONTENT masks
``</think>`` to -inf, so the model could never close its own think block; the
reasoning parser found no closer and dumped the whole answer into ``content``
with ``reasoning_content`` empty, the budget never engaged, and generation ran
to ``max_tokens``.

Measured on Qwen3.8-27B-4bit before the fix: a 448-token prompt carried
``</think>`` at index 103, so the machine reached CONTENT with
``thinking_tokens=103`` before generation began.
"""

import mlx.core as mx

from vllm_mlx.constrained.thinking_processor import (
    Phase,
    ThinkingAwareLogitsProcessor,
)

START = [100]
END = [101]
VOCAB = 128


def _proc(budget=8, prompt_has_think_tag=True):
    return ThinkingAwareLogitsProcessor(
        start_token_ids=list(START),
        end_token_ids=list(END),
        thinking_token_budget=budget,
        vocab_size=VOCAB,
        prompt_has_think_tag=prompt_has_think_tag,
    )


def _logits():
    return mx.zeros((VOCAB,))


def _feed(proc, ids):
    proc(mx.array(ids), _logits())


# A prompt that replays a prior reasoning span and then opens a new one —
# exactly the multi-turn agent shape.
PROMPT_WITH_PRIOR_SPAN = [1, 2, START[0], 3, 4, END[0], 5, 6, START[0]]


def test_prompt_with_prior_span_does_not_reach_content():
    """The regression: prior </think> in the PROMPT must not end the span."""
    proc = _proc()
    _feed(proc, PROMPT_WITH_PRIOR_SPAN)
    assert proc.state is Phase.THINKING
    assert proc.thinking_tokens == 0, "prompt tokens must not count as thinking"


def test_close_token_is_not_masked_after_a_prompt_with_prior_span():
    """The damage the old behaviour did: the model could not emit </think>."""
    proc = _proc()
    _feed(proc, PROMPT_WITH_PRIOR_SPAN)
    out = proc(mx.array(PROMPT_WITH_PRIOR_SPAN), _logits())
    assert out[END[0]].item() != float("-inf"), "</think> must remain emittable"


def test_budget_counts_generated_tokens_only():
    proc = _proc(budget=3)
    _feed(proc, PROMPT_WITH_PRIOR_SPAN)
    for n in range(1, 3):
        _feed(proc, PROMPT_WITH_PRIOR_SPAN + [7] * n)
        assert proc.thinking_tokens == n
        assert proc.state is Phase.THINKING
    # third generated token reaches the budget
    _feed(proc, PROMPT_WITH_PRIOR_SPAN + [7, 7, 7])
    assert proc.state is Phase.TRANSITIONING


def test_generated_close_token_still_ends_the_span():
    proc = _proc()
    _feed(proc, PROMPT_WITH_PRIOR_SPAN)
    _feed(proc, PROMPT_WITH_PRIOR_SPAN + [7, END[0]])
    assert proc.state is Phase.CONTENT


def test_budget_is_not_shortened_by_prompt_length():
    """A long prompt must not eat the budget."""
    proc = _proc(budget=5)
    long_prompt = list(range(2, 400))
    _feed(proc, long_prompt)
    _feed(proc, long_prompt + [7, 7, 7, 7])
    assert proc.thinking_tokens == 4
    assert proc.state is Phase.THINKING


def test_idle_start_still_waits_for_a_generated_open_tag():
    """With no prompt-side think tag the machine still starts IDLE."""
    proc = _proc(prompt_has_think_tag=False)
    _feed(proc, [1, 2, 3])
    assert proc.state is Phase.IDLE
    _feed(proc, [1, 2, 3, START[0]])
    assert proc.state is Phase.THINKING


def test_empty_first_call_selects_the_generated_only_convention():
    """An empty first call means the caller feeds GENERATED tokens only.

    The MLLM scheduler applies processors before the first completion token and
    never sends the prompt (``mllm_batch_generator`` passes ``output_tokens``),
    so nothing must be skipped on that path — a ``</think>`` in the tokens it
    sends is genuinely generated and must end the span. The batched/mlx-lm path
    is the opposite: ``BatchGenerator.prompt()`` folds prompt tokens into the
    sequence it hands processors, so its first call IS the prompt.
    """
    proc = _proc()
    proc(mx.array([], dtype=mx.int32), _logits())
    _feed(proc, PROMPT_WITH_PRIOR_SPAN)
    assert proc.state is Phase.CONTENT, "generated </think> must end the span"


def test_batched_convention_skips_the_prompt_without_an_empty_call():
    """No empty first call => batched convention => first batch is the prompt."""
    proc = _proc()
    _feed(proc, PROMPT_WITH_PRIOR_SPAN)
    assert proc.state is Phase.THINKING
    assert proc.thinking_tokens == 0


def test_zero_budget_still_forces_transition_immediately():
    proc = _proc(budget=0)
    assert proc.state is Phase.TRANSITIONING
    out = proc(mx.array(PROMPT_WITH_PRIOR_SPAN), _logits())
    assert out[END[0]].item() == 0.0, "the forced closer must be the only choice"


def test_rollback_never_re_enters_the_prompt():
    """A rewind past the prompt boundary must not re-walk prompt tokens.

    Within one request the prompt is fixed, so this is defensive — but the
    fork's cache machinery does rewind sequences, and walking the prompt is
    precisely the bug this module guards against.
    """
    proc = _proc()
    _feed(proc, PROMPT_WITH_PRIOR_SPAN)
    _feed(proc, PROMPT_WITH_PRIOR_SPAN + [7, 8])
    assert proc.thinking_tokens == 2

    # Diverge inside the prompt region: the clamp must hold the boundary.
    _feed(proc, PROMPT_WITH_PRIOR_SPAN[:4] + [99])
    assert proc.state is not Phase.CONTENT, "must not have walked the prompt's </think>"
    assert proc.thinking_tokens == 0
