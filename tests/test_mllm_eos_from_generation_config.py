# SPDX-License-Identifier: Apache-2.0
"""EOS union from generation_config.json must work for HF repo ids.

MLLMScheduler._get_stop_tokens read generation_config.json only from a
LOCAL path, but tokenizer.name_or_path is usually the HF repo id (what
llama-swap passes) — so the multi-EOS union silently never happened
(the gemma multi-eos leak class: <turn|> runs to max_tokens). Repo ids
now resolve through the HF cache (the patch-#14 trick).
"""

import json
from types import SimpleNamespace

from vllm_mlx.mllm_scheduler import MLLMScheduler


def _scheduler_with_tokenizer(tokenizer):
    sched = MLLMScheduler.__new__(MLLMScheduler)
    sched.processor = SimpleNamespace(tokenizer=tokenizer)
    return sched


def _tokenizer(name_or_path, eos_token_id=1):
    return SimpleNamespace(name_or_path=name_or_path, eos_token_id=eos_token_id)


def test_local_dir_still_read(tmp_path):
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [1, 106, 50]})
    )
    sched = _scheduler_with_tokenizer(_tokenizer(str(tmp_path)))

    assert sched._get_stop_tokens() == {1, 106, 50}


def test_repo_id_resolves_via_hf_cache(tmp_path, monkeypatch):
    gc_file = tmp_path / "generation_config.json"
    gc_file.write_text(json.dumps({"eos_token_id": [1, 106]}))

    import huggingface_hub

    calls = {}

    def fake_try_load(repo_id, filename=None):
        calls["repo_id"] = repo_id
        calls["filename"] = filename
        return str(gc_file)

    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache", fake_try_load
    )
    sched = _scheduler_with_tokenizer(_tokenizer("mlx-community/Some-VLM-4bit"))

    stop = sched._get_stop_tokens()

    assert calls == {
        "repo_id": "mlx-community/Some-VLM-4bit",
        "filename": "generation_config.json",
    }
    assert stop == {1, 106}


def test_repo_id_not_cached_falls_back_to_tokenizer_eos(monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        lambda repo_id, filename=None: None,
    )
    sched = _scheduler_with_tokenizer(_tokenizer("owner/never-downloaded"))

    assert sched._get_stop_tokens() == {1}


def test_plain_name_without_slash_skips_hf_lookup():
    # No slash → not a repo id → no HF lookup attempted (would only
    # matter if huggingface_hub were broken; the tokenizer EOS remains).
    sched = _scheduler_with_tokenizer(_tokenizer("local-model", eos_token_id=[7, 8]))

    assert sched._get_stop_tokens() == {7, 8}
