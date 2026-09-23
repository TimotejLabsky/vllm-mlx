#!/usr/bin/env python
"""CI lint gate for the fork: black-check only the LINES a change adds.

Fork policy (PATCHES.md) is "no NEW black findings in touched files": ~44
files carry formatting drift from upstream rebases, and reformatting them
wholesale would turn every one of their lines into rebase-conflict surface.
``black --check vllm_mlx/ tests/`` therefore failed on every PR, which made
the lint job a permanent red X that told nobody anything — and trained
everyone to merge past a failing check.

This runs ``black --line-ranges`` over exactly the added/modified line ranges
of each changed Python file, so the job is green unless THIS change
introduces drift. New files are checked whole.

Uses ``--diff`` (empty output = clean), never ``--check``: black's cache
records a file as formatted after a passing ``--check --line-ranges`` run,
and a later whole-file ``--check`` then reports it clean when it is not.

    python scripts/fork/black_changed_lines.py <base-sha> [paths...]
"""

import os
import re
import subprocess
import sys
import tempfile

DEFAULT_PATHS = ("vllm_mlx", "tests")
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
_ZERO = re.compile(r"^0+$")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def _resolve_base(base: str) -> str | None:
    """PRs pass the base sha; pushes pass ``github.event.before`` (all zeros
    for a brand-new branch). Fall back to the previous commit."""
    for candidate in (base, "HEAD~1"):
        if not candidate or _ZERO.match(candidate):
            continue
        try:
            _git("rev-parse", "--verify", "--quiet", f"{candidate}^{{commit}}")
            return candidate
        except subprocess.CalledProcessError:
            continue
    return None


def diff_spec(base: str) -> str:
    """``base...HEAD`` (from the merge-base) for an ordinary PR or push.

    After a force-push (a rebase onto upstream) ``github.event.before`` is not
    an ancestor of HEAD, and the merge-base is the OLD upstream base — so
    ``...`` counted every line of every fork patch as "changed" and the job
    went red on each rebase with drift nobody added. Compare the two trees
    instead (``base..HEAD``): exactly what the push changed in content —
    upstream's window plus the conflict resolutions.
    """
    try:
        _git("merge-base", "--is-ancestor", base, "HEAD")
        return f"{base}...HEAD"
    except subprocess.CalledProcessError:
        return f"{base}..HEAD"


def changed_ranges(spec: str, path: str) -> list[tuple[int, int]]:
    """(start, end) line ranges ``path`` gained in the diff ``spec``."""
    ranges = []
    for line in _git("diff", "-U0", spec, "--", path).splitlines():
        match = _HUNK.match(line)
        if not match:
            continue
        start, count = int(match.group(1)), int(match.group(2) or "1")
        if count:  # count == 0 is a pure deletion
            ranges.append((start, start + count - 1))
    return ranges


def own_hunks(black_diff: str, ranges: list[tuple[int, int]]) -> list[str]:
    """The hunks of black's diff that rewrite a line this change touched.

    ``--line-ranges`` expands to whole statements: adding one key to a long
    dict literal makes black reformat the entire literal, including drift at
    its far end that predates the change. A hunk counts only if one of the
    ORIGINAL lines it removes lies inside the change's own ranges."""

    def touched(line_no: int) -> bool:
        return any(a <= line_no <= b for a, b in ranges)

    kept, current, mine, old_no = [], [], False, 0
    for line in black_diff.splitlines():
        header = re.match(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@", line)
        if header:
            if current and mine:
                kept.append("\n".join(current))
            current, mine, old_no = [line], False, int(header.group(1))
            continue
        if not current:
            continue  # file header lines
        current.append(line)
        if line.startswith("-"):
            mine = mine or touched(old_no)
            old_no += 1
        elif not line.startswith("+"):
            old_no += 1
    if current and mine:
        kept.append("\n".join(current))
    return kept


def main(argv: list[str]) -> int:
    base = _resolve_base(argv[1] if len(argv) > 1 else "")
    paths = argv[2:] or list(DEFAULT_PATHS)
    if base is None:
        print(
            "black_changed_lines: no usable base commit — nothing to compare, skipping"
        )
        return 0

    spec = diff_spec(base)
    files = [
        f
        for f in _git(
            "diff", "--name-only", "--diff-filter=AM", spec, "--", *paths
        ).splitlines()
        if f.endswith(".py")
    ]
    if not files:
        print(f"black_changed_lines: no Python changes under {paths} vs {base[:12]}")
        return 0
    if not spec.endswith("...HEAD"):
        print(
            f"black_changed_lines: {base[:12]} is not an ancestor of HEAD "
            "(history rewritten) — checking the tree diff, not the merge-base"
        )

    env = dict(os.environ, BLACK_CACHE_DIR=tempfile.mkdtemp(prefix="black-cache-"))
    failed = []
    for path in files:
        ranges = changed_ranges(spec, path)
        if not ranges:
            continue
        cmd = [sys.executable, "-m", "black", "--quiet", "--diff"]
        cmd += [f"--line-ranges={a}-{b}" for a, b in ranges]
        result = subprocess.run(cmd + [path], capture_output=True, text=True, env=env)
        if result.returncode != 0:
            print(f"ERROR  {path}: black failed\n{result.stderr}")
            failed.append(path)
            continue
        hunks = own_hunks(result.stdout, ranges)
        if hunks:
            print(f"DRIFT  {path}: lines this change touches are not black-clean")
            print("\n".join(hunks))
            failed.append(path)
        else:
            print(f"ok     {path} ({len(ranges)} changed range(s))")

    if failed:
        print(
            f"\n{len(failed)} file(s) gained formatting drift. Fix only the lines "
            "shown (do not reformat the whole file — see the module docstring)."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
