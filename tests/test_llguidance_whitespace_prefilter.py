# SPDX-License-Identifier: Apache-2.0
"""PATCHES.md #176: the llguidance processor stops re-decoding the suffix.

The non-progress whitespace check used to decode the whole generated suffix
on every step (O(n) per step, O(n^2) per request). It now decodes only when
the last few tokens could form a 256-character whitespace run, and must give
exactly the old answer.
"""

import glob
import os
import random
from types import SimpleNamespace

import pytest

from vllm_mlx.constrained import llguidance_schema_processor as m

PIECES = (
    ["a", "bc", ' "', '"', "\\", "x y", "{", "}", ":"]  # progress text
    + [" ", "  ", "\n", " " * 40, "\n" + " " * 7]  # whitespace
    + ["", "�", "��"]  # special/empty and partial bytes
)
SPECIAL = {PIECES.index(""), PIECES.index("�"), PIECES.index("��")}


class FakeTokenizer:
    def __init__(self):
        self.decodes = 0

    def convert_ids_to_tokens(self, token_id):
        return PIECES[token_id]

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=True):
        self.decodes += 1
        return "".join(PIECES[i] for i in ids)


def _processor(tokenizer, prefilter=True):
    proc = m.LLGuidanceJSONSchemaLogitsProcessor.__new__(
        m.LLGuidanceJSONSchemaLogitsProcessor
    )
    proc._tokenizer = tokenizer
    proc._vocab_size = len(PIECES)
    proc._ws_prefilter = (
        m._whitespace_prefilter(tokenizer, len(PIECES)) if prefilter else None
    )
    return proc


def _reference(tokenizer, suffix):
    decoded = tokenizer.decode(
        suffix, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return m._trailing_json_whitespace(decoded) >= m._MAX_NONPROGRESS_WHITESPACE_CHARS


def test_176_prefilter_matches_full_decode_on_random_suffixes():
    tok = FakeTokenizer()
    proc = _processor(tok)
    rng = random.Random(176)
    ws = [i for i, p in enumerate(PIECES) if p and p.isspace()]
    hits = 0
    for _ in range(3000):
        pool = [i for i in range(len(PIECES)) if '"' not in PIECES[i]]
        if rng.random() < 0.3:
            pool = list(range(len(PIECES)))  # sometimes leave a string open
        body = [rng.choice(pool) for _ in range(rng.randrange(0, 40))]
        quiet = ws + [PIECES.index("")]
        tail = [rng.choice(quiet) for _ in range(rng.randrange(0, 30))]
        suffix = body + tail
        expected = _reference(tok, suffix)
        hits += expected
        assert proc._whitespace_run_reached(suffix) == expected, suffix
    assert hits > 100  # the long-run branch was actually exercised


def test_176_in_string_whitespace_run_is_not_non_progress():
    tok = FakeTokenizer()
    proc = _processor(tok)
    opened = [PIECES.index(' "')]
    suffix = opened + [PIECES.index(" " * 40)] * 10
    assert _reference(tok, suffix) is False
    assert proc._whitespace_run_reached(suffix) is False
    suffix = [PIECES.index("a")] + [PIECES.index(" " * 40)] * 10
    assert proc._whitespace_run_reached(suffix) is True


def test_176_normal_steps_do_not_decode_the_suffix():
    tok = FakeTokenizer()
    proc = _processor(tok)
    tok.decodes = 0
    suffix = []
    for i in range(2000):
        suffix.append(PIECES.index("bc") if i % 3 else PIECES.index(" "))
        assert proc._whitespace_run_reached(suffix) is False
    assert tok.decodes == 0


def test_176_without_token_lengths_it_always_decodes():
    class Bare:
        def decode(self, ids, **kw):
            return "".join(PIECES[i] for i in ids)

    assert m._whitespace_prefilter(Bare(), len(PIECES)) is None
    proc = _processor(Bare(), prefilter=False)
    assert proc._whitespace_run_reached([PIECES.index(" " * 40)] * 7) is True
    assert proc._whitespace_run_reached([PIECES.index("a")] * 7) is False


def test_176_out_of_vocab_token_forces_the_full_decode():
    tok = FakeTokenizer()
    proc = _processor(tok)
    proc._vocab_size = 3  # tokens past it are unknown to the prefilter
    tok.decodes = 0
    proc._whitespace_run_reached([PIECES.index(" " * 40)] * 10)
    assert tok.decodes == 1


def test_176_matches_real_tokenizer_if_cached():
    transformers = pytest.importorskip("transformers")
    paths = glob.glob(
        os.path.expanduser("~")
        + "/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-8bit/snapshots/*"
    )
    if not paths:
        pytest.skip("Qwen3 tokenizer not cached")
    tok = transformers.AutoTokenizer.from_pretrained(paths[0])
    vocab = len(tok)
    pre = m._whitespace_prefilter(tok, vocab)
    assert pre is not None
    proc = SimpleNamespace(_ws_prefilter=pre, _vocab_size=vocab, _tokenizer=tok)
    check = m.LLGuidanceJSONSchemaLogitsProcessor._whitespace_run_reached
    ws_ids = [tok.encode(s, add_special_tokens=False) for s in (" " * 64, "\n" * 8)]
    text_ids = tok.encode('{"a": "hello world"', add_special_tokens=False)
    for reps in (1, 4, 5, 9, 20):
        for ws in ws_ids:
            suffix = text_ids + ws * reps
            assert check(proc, suffix) == _reference(tok, suffix)
    rng = random.Random(1)
    for _ in range(300):
        suffix = [rng.randrange(vocab) for _ in range(rng.randrange(1, 30))]
        assert check(proc, suffix) == _reference(tok, suffix)
