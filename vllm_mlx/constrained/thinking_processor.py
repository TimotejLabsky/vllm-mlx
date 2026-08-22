"""Thinking-aware logits processor for reasoning models.

Manages the full thinking lifecycle: budget enforcement, phase transitions,
and content-phase constrained decoding delegation.
"""

from __future__ import annotations

import enum
from collections import deque
from typing import Callable

import mlx.core as mx


class BoundedSuffixMatcher:
    """Detect a target token sequence in a stream using a rolling suffix buffer.

    Unlike a naive sequential matcher that resets to position 0 on mismatch,
    this uses a bounded buffer that catches overlapping prefixes.
    """

    __slots__ = ("target", "_buf", "_max_len")

    def __init__(self, target_ids: list[int]) -> None:
        if not target_ids:
            raise ValueError("target_ids must be non-empty")
        self.target = tuple(target_ids)
        self._max_len = len(target_ids)
        self._buf: deque[int] = deque(maxlen=self._max_len)

    def feed(self, token_id: int) -> bool:
        """Feed one token. Returns True when the buffer suffix equals the target."""
        self._buf.append(token_id)
        return len(self._buf) == self._max_len and tuple(self._buf) == self.target

    def reset(self) -> None:
        """Clear the buffer."""
        self._buf.clear()

    def snapshot(self) -> tuple[int, ...]:
        """Return a serializable copy of the current suffix buffer."""
        return tuple(self._buf)

    def restore(self, state: tuple[int, ...]) -> None:
        """Restore the suffix buffer from a previous snapshot."""
        self._buf.clear()
        self._buf.extend(state)


class Phase(enum.Enum):
    """Thinking lifecycle phases."""

    IDLE = "idle"
    THINKING = "thinking"
    TRANSITIONING = "transitioning"
    CONTENT = "content"


class ThinkingAwareLogitsProcessor:
    """Unified logits processor for thinking-model lifecycle management.

    Manages a four-phase state machine:
      IDLE -> THINKING -> TRANSITIONING -> CONTENT

    - IDLE: before reasoning start tokens. Pass through.
    - THINKING: inside reasoning span. Count tokens, pass through.
    - TRANSITIONING: forcing reasoning end sequence via logits masking.
    - CONTENT: after reasoning closed. Delegate to inner processor.

    No re-entry into THINKING after CONTENT is reached.
    """

    __slots__ = (
        "_start_matcher",
        "_end_matcher",
        "_end_token_ids",
        "_content_phase_mask_ids",
        "_thinking_token_budget",
        "_inner",
        "_vocab_size",
        "_state",
        "_thinking_tokens",
        "_transition_index",
        "_processed_len",
        "_processed_token_ids",
        "_snapshots",
        "watchdog_was_enforced",
        "_no_final_content_token_limit",
        "_prompt_len",
    )

    def __init__(
        self,
        start_token_ids: list[int],
        end_token_ids: list[int],
        thinking_token_budget: int,
        inner: Callable[[mx.array, mx.array], mx.array] | None = None,
        vocab_size: int = 152064,
        prompt_has_think_tag: bool = False,
        no_final_content_token_limit: int | None = None,
    ) -> None:
        self._start_matcher = BoundedSuffixMatcher(start_token_ids)
        self._end_matcher = BoundedSuffixMatcher(end_token_ids)
        self._end_token_ids = list(end_token_ids)
        # Mask only the first token of each sequence: sufficient because most
        # tokenizers encode <think>/<|think|> as a single special token.
        self._content_phase_mask_ids = tuple(
            dict.fromkeys([start_token_ids[0], end_token_ids[0]])
        )
        self._thinking_token_budget = thinking_token_budget
        self._inner = inner
        self._vocab_size = vocab_size
        self._thinking_tokens = 0
        self._transition_index = 0
        self.watchdog_was_enforced = False
        self._no_final_content_token_limit = no_final_content_token_limit
        # When the chat template already injected <think> into the prompt,
        # the first generated token is already inside the thinking span.
        # Start in THINKING (or TRANSITIONING if budget=0) instead of IDLE.
        if prompt_has_think_tag:
            if thinking_token_budget == 0:
                self._state = Phase.TRANSITIONING
            else:
                self._state = Phase.THINKING
        else:
            self._state = Phase.IDLE
        # Set on the first sync: mlx-lm hands processors the FULL sequence,
        # so we have to learn where the prompt ends. See _sync_to_tokens.
        self._prompt_len: int | None = None
        self._processed_len = 0
        self._processed_token_ids: list[int] = []
        self._snapshots = [self._snapshot_state()]

    @property
    def state(self) -> Phase:
        return self._state

    @property
    def thinking_tokens(self) -> int:
        return self._thinking_tokens

    @property
    def is_retired(self) -> bool:
        """True when the processor is in CONTENT with no inner constraint.

        The engine can use this signal to drop the processor and re-enable
        MTP for the remaining content generation (Phase 2 optimization).
        """
        return self._state == Phase.CONTENT and self._inner is None

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        # The MLLM scheduler applies processors before the first completion
        # token is emitted, so ``tokens`` can be empty on step 0.
        if tokens.size == 0:
            # An empty first call marks the generated-only convention (the MLLM
            # scheduler applies processors before the first completion token).
            # Callers on that path never send the prompt, so there is nothing to
            # skip; recording 0 stops _sync_to_tokens treating their first real
            # batch of GENERATED tokens as prompt.
            if self._prompt_len is None:
                self._prompt_len = 0
            if self._state == Phase.TRANSITIONING:
                return self._force_transition(logits)
            if self._state == Phase.CONTENT:
                return self._call_inner(tokens, logits)
            return logits

        self._sync_to_tokens(tokens)

        if self._state == Phase.TRANSITIONING:
            return self._force_transition(logits)

        # Phase.CONTENT
        if self._state == Phase.CONTENT:
            return self._call_inner(tokens, logits)
        return logits

    def _force_transition(self, logits: mx.array) -> mx.array:
        """Force the next token in the reasoning end sequence."""
        target_id = self._end_token_ids[self._transition_index]
        # Mask all logits to -inf, then set the target token to 0.
        # Handle both 1-D (vocab,) and 2-D (1, vocab) logits shapes.
        masked = mx.full(logits.shape, float("-inf"))
        if masked.ndim == 1:
            masked[target_id] = 0.0
        else:
            masked[..., target_id] = 0.0
        return masked

    def _call_inner(self, tokens: mx.array, logits: mx.array) -> mx.array:
        """Delegate to inner processor if present."""
        if self._inner is not None:
            logits = self._inner(tokens, logits)
        return self._mask_content_phase_control_tokens(logits)

    def _mask_content_phase_control_tokens(self, logits: mx.array) -> mx.array:
        """Prevent reserved think-tag starts from leaking into final content."""
        for token_id in self._content_phase_mask_ids:
            if logits.ndim == 1:
                logits[token_id] = float("-inf")
            else:
                logits[..., token_id] = float("-inf")
        return logits

    def _snapshot_state(
        self,
    ) -> tuple[Phase, int, int, tuple[int, ...], tuple[int, ...], bool]:
        return (
            self._state,
            self._thinking_tokens,
            self._transition_index,
            self._start_matcher.snapshot(),
            self._end_matcher.snapshot(),
            self.watchdog_was_enforced,
        )

    def _restore_snapshot(self, processed_len: int) -> None:
        # In CONTENT phase, snapshots stop growing (see _sync_to_tokens).
        # If rollback targets a CONTENT position beyond the snapshot list,
        # use the last available snapshot -- the state is identical since
        # _advance_with_token is a no-op in CONTENT.
        # Snapshots are indexed by GENERATED position: snapshots[0] is the state
        # at the prompt boundary, since the prompt is skipped wholesale in
        # _sync_to_tokens. Subtract the prompt length before indexing, or a
        # rollback restores a snapshot from the wrong point in the generation.
        generated_len = max(processed_len - (self._prompt_len or 0), 0)
        snap_idx = min(generated_len, len(self._snapshots) - 1)
        (
            self._state,
            self._thinking_tokens,
            self._transition_index,
            start_state,
            end_state,
            self.watchdog_was_enforced,
        ) = self._snapshots[snap_idx]
        self._start_matcher.restore(start_state)
        self._end_matcher.restore(end_state)
        self._processed_len = processed_len
        self._processed_token_ids = self._processed_token_ids[:processed_len]
        self._snapshots = self._snapshots[: snap_idx + 1]

    def _sync_to_tokens(self, tokens: mx.array) -> None:
        target_len = int(tokens.size)
        token_ids = tokens.tolist()

        # mlx-lm passes the FULL sequence (prompt + generated) to a logits
        # processor. The phase machine must only ever walk GENERATED tokens.
        #
        # Walking the prompt is not a cosmetic miscount: any prompt replaying an
        # earlier ``<think>...</think>`` span -- i.e. every multi-turn agent
        # conversation -- drove the machine to CONTENT before the first token
        # was generated. CONTENT masks ``<think>``/``</think>`` to -inf, so the
        # model was then physically unable to close its own think block: the
        # reasoning parser found no closer, dumped the whole answer into
        # ``content`` with ``reasoning_content`` empty, the budget never
        # engaged (already CONTENT), and generation ran to ``max_tokens``.
        # Measured on Qwen3.8-27B-4bit: a 448-token prompt carried </think> at
        # index 103, so the machine reached CONTENT with thinking_tokens=103
        # before generation began.
        #
        # A freshly-built processor has by definition seen no generated tokens,
        # so whatever is present on the first sync is prompt.
        if self._prompt_len is None:
            self._prompt_len = target_len
            self._processed_len = target_len
            self._processed_token_ids = list(token_ids)
            return
        common_len = 0
        max_common = min(target_len, self._processed_len)
        while (
            common_len < max_common
            and self._processed_token_ids[common_len] == token_ids[common_len]
        ):
            common_len += 1
        if common_len < self._processed_len:
            # Never roll back INTO the prompt. Those tokens were deliberately
            # skipped above and must never be walked by the phase machine; a
            # rewind past the prompt boundary would re-run the exact bug this
            # guard exists to prevent. Within one request the prompt is fixed,
            # so this clamp is defensive rather than load-bearing.
            self._restore_snapshot(max(common_len, self._prompt_len or 0))
        if target_len == self._processed_len:
            return
        for token_id in token_ids[self._processed_len :]:
            self._advance_with_token(token_id)
            self._processed_token_ids.append(token_id)
            self._processed_len += 1
            # Skip snapshots in CONTENT -- _advance_with_token is a no-op
            # there, so snapshots would just waste memory on long generations.
            if self._state != Phase.CONTENT:
                self._snapshots.append(self._snapshot_state())

    def _advance_with_token(self, token_id: int) -> None:
        if self._state == Phase.IDLE:
            if self._start_matcher.feed(token_id):
                self._state = Phase.THINKING
                if self._thinking_token_budget == 0:
                    self._state = Phase.TRANSITIONING
                    self._transition_index = 0
            return

        if self._state == Phase.THINKING:
            if self._end_matcher.feed(token_id):
                self._state = Phase.CONTENT
                return
            self._thinking_tokens += 1
            if self._thinking_tokens >= self._thinking_token_budget:
                self._state = Phase.TRANSITIONING
                self._transition_index = 0
            elif (
                self._no_final_content_token_limit is not None
                and self._thinking_tokens >= self._no_final_content_token_limit
            ):
                self._state = Phase.TRANSITIONING
                self._transition_index = 0
                self.watchdog_was_enforced = True
            return

        if self._state == Phase.TRANSITIONING:
            expected = self._end_token_ids[self._transition_index]
            if token_id == expected:
                self._transition_index += 1
                if self._transition_index >= len(self._end_token_ids):
                    self._state = Phase.CONTENT
                    self._end_matcher.reset()
            return
