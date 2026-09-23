# SPDX-License-Identifier: Apache-2.0
"""The fork's changed-lines lint gate across a rebase force-push.

After a rebase, ``github.event.before`` (the old tip) is not an ancestor of
HEAD. Diffing ``before...HEAD`` measured from the merge-base — the OLD
upstream base — so every fork patch line counted as changed and the job went
red with drift nobody added (2026-09-23 rebase: 43 files). The gate now
compares the two trees in that case, which still catches drift the rebase
itself introduces.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/fork/black_changed_lines.py"

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None
    or subprocess.run(
        [sys.executable, "-m", "black", "--version"], capture_output=True
    ).returncode
    != 0,
    reason="needs git and black",
)

CLEAN = "def f():\n    return 1\n"
DRIFTED = "def g( ):\n    return   2\n"  # pre-existing fork drift


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _commit(repo, files, message):
    for name, text in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _gate(repo, base):
    return subprocess.run(
        [sys.executable, str(SCRIPT), base],
        cwd=repo,
        capture_output=True,
        text=True,
    )


@pytest.fixture()
def rebased(tmp_path):
    """upstream v1 <- fork patch (drifted file) = old tip; then upstream v2
    and the fork patch replayed on it = new tip (history rewritten)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    upstream_v1 = _commit(repo, {"vllm_mlx/up.py": CLEAN}, "upstream v1")
    old_tip = _commit(repo, {"vllm_mlx/fork.py": DRIFTED}, "fork patch")
    _git(repo, "checkout", "-q", "-b", "rebased", upstream_v1)
    _commit(repo, {"vllm_mlx/up.py": CLEAN + "\n\ndef h():\n    return 3\n"}, "v2")
    _commit(repo, {"vllm_mlx/fork.py": DRIFTED}, "fork patch (replayed)")
    return repo, old_tip


def test_rebase_force_push_does_not_flag_pre_existing_drift(rebased):
    repo, old_tip = rebased
    result = _gate(repo, old_tip)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "history rewritten" in result.stdout


def test_rebase_force_push_still_catches_drift_the_rebase_adds(rebased):
    repo, old_tip = rebased
    _commit(repo, {"vllm_mlx/up.py": CLEAN + "\nx = {  'a':1 }\n"}, "bad resolution")
    result = _gate(repo, old_tip)
    assert result.returncode != 0
    assert "vllm_mlx/up.py" in result.stdout


def test_ordinary_push_unchanged(rebased):
    repo, _ = rebased
    base = _git(repo, "rev-parse", "HEAD")
    _commit(repo, {"vllm_mlx/new.py": "y = {  'b':2 }\n"}, "drifted new file")
    result = _gate(repo, base)
    assert result.returncode != 0
    assert "history rewritten" not in result.stdout
