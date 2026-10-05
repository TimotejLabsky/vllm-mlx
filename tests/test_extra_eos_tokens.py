# SPDX-License-Identifier: Apache-2.0
"""Fork #122: VLLM_MLX_EXTRA_EOS_TOKENS adds per-route turn terminators."""

from types import SimpleNamespace

import pytest

from vllm_mlx.utils.tokenizer import (
    EXTRA_EOS_ENV,
    collect_eos_token_ids,
    extra_eos_token_ids,
)

VOCAB = {"<|end|>": 200020, "<|endoftext|>": 199999, "<unk>": 0}


def _tok():
    return SimpleNamespace(
        eos_token_id=199999,
        unk_token_id=0,
        name_or_path=None,
        convert_tokens_to_ids=lambda t: VOCAB.get(t, 0),
    )


def test_unset_adds_nothing(monkeypatch):
    monkeypatch.delenv(EXTRA_EOS_ENV, raising=False)
    assert extra_eos_token_ids(_tok()) == set()
    assert collect_eos_token_ids(_tok()) == {199999}


def test_token_strings_and_ids(monkeypatch):
    monkeypatch.setenv(EXTRA_EOS_ENV, "<|end|>, 42")
    assert extra_eos_token_ids(_tok()) == {200020, 42}
    assert collect_eos_token_ids(_tok()) == {199999, 200020, 42}


def test_unknown_token_is_skipped(monkeypatch, caplog):
    monkeypatch.setenv(EXTRA_EOS_ENV, "<|nope|>,<|end|>")
    assert extra_eos_token_ids(_tok()) == {200020}
    assert "unknown token" in caplog.text


def test_batched_llm_scheduler_stops_on_extra(monkeypatch):
    from vllm_mlx.scheduler import Scheduler

    monkeypatch.setenv(EXTRA_EOS_ENV, "<|end|>")
    sched = Scheduler.__new__(Scheduler)
    sched.tokenizer = _tok()
    sched._actual_tokenizer = _tok()
    assert sched._get_stop_tokens() == {199999, 200020}


@pytest.mark.parametrize("value", ["", "  ", ","])
def test_blank_values(monkeypatch, value):
    monkeypatch.setenv(EXTRA_EOS_ENV, value)
    assert extra_eos_token_ids(_tok()) == set()
