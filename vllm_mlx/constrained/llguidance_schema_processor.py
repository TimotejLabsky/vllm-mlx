# SPDX-License-Identifier: Apache-2.0
"""Strict JSON Schema token masking backed by ``llguidance``."""

from __future__ import annotations

from typing import Any, cast

import mlx.core as mx
import numpy as np

from .json_schema_processor import ConstrainedDecodingError

_MAX_NONPROGRESS_WHITESPACE_CHARS = 256
_tokenizer_cache: dict[int, tuple[Any, Any, frozenset[int], Any]] = {}


def is_available() -> bool:
    """Return whether the strict JSON Schema backend is importable."""
    try:
        import llguidance  # noqa: F401
        import llguidance.hf  # noqa: F401
        import llguidance.mlx  # noqa: F401
    except ImportError:
        return False
    return True


def _token_list(tokens: mx.array) -> list[int]:
    values = cast(
        int | list[int] | list[list[int]],
        tokens.tolist() if hasattr(tokens, "tolist") else list(tokens),
    )
    if isinstance(values, int):
        return [values]
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(token) for token in cast(list[int], values)]


def _ll_tokenizer(tokenizer: Any):
    import llguidance.hf

    candidates = [tokenizer]
    for attribute in ("tokenizer", "_tokenizer"):
        candidate = getattr(tokenizer, attribute, None)
        if candidate is not None and candidate not in candidates:
            candidates.append(candidate)

    errors: list[str] = []
    for candidate in candidates:
        cached = _tokenizer_cache.get(id(candidate))
        if cached is not None and cached[0] is candidate:
            return cached
        try:
            ll_tokenizer = llguidance.hf.from_tokenizer(candidate)
        except (TypeError, ValueError) as exc:
            errors.append(f"{type(candidate).__name__}: {exc}")
            continue
        whitespace_tokens = frozenset(
            token_id
            for token_id in range(ll_tokenizer.vocab_size)
            if _decoded_token_is_whitespace(candidate, token_id)
        )
        result = (
            candidate,
            ll_tokenizer,
            whitespace_tokens,
            _whitespace_prefilter(candidate, ll_tokenizer.vocab_size),
        )
        _tokenizer_cache[id(candidate)] = result
        return result
    detail = "; ".join(errors)
    raise ConstrainedDecodingError(
        "strict JSON Schema requires a fast Hugging Face tokenizer"
        + (f" ({detail})" if detail else "")
    )


def _decoded_token_is_whitespace(tokenizer: Any, token_id: int) -> bool:
    try:
        text = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        text = tokenizer.decode([token_id])
    except Exception:
        return False
    return bool(text) and text.isspace()


def _whitespace_prefilter(
    tokenizer: Any, vocab_size: int
) -> tuple[frozenset[int], int] | None:
    """Tokens that could end a 256-character whitespace run, and how many.

    A trailing run of ``_MAX_NONPROGRESS_WHITESPACE_CHARS`` whitespace
    characters in the decoded suffix needs at least ``run`` consecutive
    trailing tokens from ``maybe``: tokens whose own decode is empty (special
    tokens, a stripped leading space), whitespace, or holds a replacement
    character (a partial multi-byte sequence that can complete into U+0085
    or U+00A0 with its neighbour). Any other token puts a non-whitespace
    character at the end of the text. Returns None, meaning "always decode",
    when the tokenizer cannot give token lengths.
    """
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if convert is None:
        return None
    maybe: set[int] = set()
    longest = 1
    try:
        for token_id in range(vocab_size):
            try:
                text = tokenizer.decode(
                    [token_id],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            except TypeError:
                text = tokenizer.decode([token_id])
            if text and not text.isspace() and "\ufffd" not in text:
                continue
            piece = convert(token_id)
            maybe.add(token_id)
            longest = max(longest, len(text), len(piece) if piece else 0)
    except Exception:
        return None
    run = -(-_MAX_NONPROGRESS_WHITESPACE_CHARS // longest)
    return frozenset(maybe), max(1, run)


def _trailing_json_whitespace(text: str) -> int:
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
    if in_string:
        return 0
    return len(text) - len(text.rstrip())


def _clear_mask_token(mask: np.ndarray, token_id: int) -> None:
    word_index, bit_index = divmod(token_id, 32)
    if word_index >= mask.shape[1]:
        return
    value = int(np.uint32(mask[0, word_index])) & ~(1 << bit_index)
    mask[0, word_index] = np.array(value, dtype=np.uint32).view(np.int32)


class LLGuidanceJSONSchemaLogitsProcessor:
    """Apply a request-local JSON Schema grammar to MLX logits."""

    def __init__(self, schema: dict, tokenizer: Any) -> None:
        if not is_available():
            raise ConstrainedDecodingError(
                "llguidance is required for strict JSON Schema response_format"
            )

        import llguidance

        self._tokenizer, ll_tokenizer, whitespace_tokens, prefilter = _ll_tokenizer(
            tokenizer
        )
        self._ws_prefilter = prefilter
        self._vocab_size = ll_tokenizer.vocab_size
        self._whitespace_tokens = whitespace_tokens
        eos_tokens = ll_tokenizer.eos_tokens
        self._eos_tokens = {int(token) for token in eos_tokens}
        try:
            grammar = llguidance.LLMatcher.grammar_from_json_schema(schema)
            self._matcher = llguidance.LLMatcher(ll_tokenizer, grammar)
        except Exception as exc:
            raise ConstrainedDecodingError(
                f"failed to compile strict JSON Schema grammar: {exc}"
            ) from exc

        # llguidance can return an invalid matcher without raising.
        if self._matcher.is_error():
            detail = self._matcher.get_error() or self._matcher.stop_reason()
            raise ConstrainedDecodingError(
                f"failed to compile strict JSON Schema grammar: {detail}"
            )

        self._prompt_len: int | None = None
        self._consumed_suffix: list[int] = []
        self._terminal = False
        # Grammar state as of the last mask application, cached as a plain
        # bool for the stop paths to read from another thread (PATCHES.md
        # #89 — see vllm_mlx/grammar_guard.py for the protocol).
        self._accepting = False

    def _consume_suffix(self, suffix: list[int]) -> None:
        previous = self._consumed_suffix
        if len(suffix) >= len(previous) and suffix[: len(previous)] == previous:
            delta = suffix[len(previous) :]
        else:
            self._matcher.reset()
            delta = suffix

        if delta and all(token in self._eos_tokens for token in delta):
            if not self._matcher.is_accepting():
                raise ConstrainedDecodingError(
                    "received an EOS token before the JSON Schema was complete"
                )
            self._terminal = True
            self._consumed_suffix = list(suffix)
            return
        if delta and not self._matcher.consume_tokens(delta):
            try:
                decoded_tail = self._tokenizer.decode(suffix[-12:])
            except Exception:
                decoded_tail = "<decode-error>"
            raise ConstrainedDecodingError(
                "generated token prefix violated the declared JSON Schema: "
                f"delta={delta!r}, suffix_tail={suffix[-12:]!r}, "
                f"decoded_tail={decoded_tail!r}"
            )
        if self._matcher.is_error():
            detail = self._matcher.get_error() or self._matcher.stop_reason()
            raise ConstrainedDecodingError(
                f"strict JSON Schema matcher entered an error state: {detail}"
            )
        self._consumed_suffix = list(suffix)

    def _whitespace_run_reached(self, suffix: list[int]) -> bool:
        """Whether the decoded suffix ends in a non-progress whitespace run.

        Same answer as decoding the whole suffix every step, but the decode
        only runs when the last few tokens could form such a run (see
        ``_whitespace_prefilter``), so a normal step costs O(1), not O(n).
        """
        prefilter = self._ws_prefilter
        if prefilter is not None:
            maybe, run = prefilter
            if len(suffix) < run:
                return False
            vocab_size = self._vocab_size
            for token in suffix[-run:]:
                if token not in maybe and 0 <= token < vocab_size:
                    return False
        decoded = self._tokenizer.decode(
            suffix,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return _trailing_json_whitespace(decoded) >= _MAX_NONPROGRESS_WHITESPACE_CHARS

    def _matcher_is_accepting(self) -> bool:
        try:
            return bool(self._matcher.is_accepting())
        except Exception:
            # An unreadable matcher is not a reason to fail a generation that
            # is otherwise progressing; report mid-value and let the stop
            # guard stay conservative (PATCHES.md #89).
            return False

    def is_accepting(self) -> bool:
        """Grammar state as of the last mask application (PATCHES.md #89).

        Part of the stop-vs-grammar protocol in ``vllm_mlx.grammar_guard``:
        ``False`` means the declared JSON value is still open, so no stop
        terminator may end the request here.
        """
        return self._accepting

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        import llguidance.mlx

        tokens_list = _token_list(tokens)
        if self._prompt_len is None:
            self._prompt_len = len(tokens_list)
        suffix = tokens_list[self._prompt_len :]
        self._consume_suffix(suffix)
        self._accepting = self._terminal or self._matcher_is_accepting()
        if self._terminal:
            mask = mx.full(logits.shape, -float("inf"))
            for token in self._eos_tokens:
                if token < logits.shape[-1]:
                    if logits.ndim == 1:
                        mask[token] = 0.0
                    else:
                        mask[..., token] = 0.0
            return logits + mask

        try:
            mask = llguidance.mlx.allocate_token_bitmask(1, self._vocab_size)
            llguidance.mlx.fill_next_token_bitmask(self._matcher, mask)
            if self._whitespace_run_reached(suffix):
                for token_id in self._whitespace_tokens:
                    _clear_mask_token(mask, token_id)
            masked: mx.array = llguidance.mlx.apply_token_bitmask(logits, mask)
        except Exception as exc:
            raise ConstrainedDecodingError(
                f"failed to construct strict JSON Schema token mask: {exc}"
            ) from exc
        return masked[0] if logits.ndim == 1 else masked
