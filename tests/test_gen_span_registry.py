"""Fork #163 - the generated-span registry on its own."""

import json

from vllm_mlx.gen_span_registry import GenSpanRegistry

HEAD = list(range(100, 160))
GEN = [7, 8, 9, 10]
CHAIN = HEAD + GEN + [1, 2, 3]


def test_a_chain_sharing_the_prefix_finds_the_span():
    reg = GenSpanRegistry()
    assert reg.add(CHAIN, [(60, 64)]) == 1
    assert reg.add(CHAIN, [(60, 64)]) == 0  # idempotent
    assert reg.spans_for(HEAD + GEN) == [(60, 64)]
    assert reg.spans_for(CHAIN + [99, 98]) == [(60, 64)]
    assert reg.spans_for(HEAD + [7, 8, 9]) == []  # does not reach the end
    assert reg.spans_for(HEAD[:-1] + [0] + GEN) == []  # different history


def test_capacity_evicts_least_recently_used():
    reg = GenSpanRegistry(capacity=16)
    chains = [[i] * 10 + GEN for i in range(20)]
    for c in chains:
        reg.add(c, [(10, 14)])
    assert len(reg) == 16
    assert reg.spans_for(chains[0]) == []
    assert reg.spans_for(chains[-1]) == [(10, 14)]


def test_persists_across_instances_and_tolerates_a_bad_file(tmp_path):
    path = tmp_path / "gen_spans.json"
    GenSpanRegistry(str(path)).add(CHAIN, [(60, 64), (64, 67)])
    again = GenSpanRegistry(str(path))
    assert again.loaded == 2
    assert again.spans_for(CHAIN) == [(60, 64), (64, 67)]

    path.write_text("{not json")
    broken = GenSpanRegistry(str(path))
    assert len(broken) == 0
    broken.add(CHAIN, [(60, 64)])
    assert json.loads(path.read_text())["format"] == 1


def test_out_of_range_spans_are_ignored():
    reg = GenSpanRegistry()
    assert reg.add(CHAIN, [(5, 5), (-1, 3), (60, 999)]) == 0
    assert len(reg) == 0
