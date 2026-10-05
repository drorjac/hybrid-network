"""The four block diagrams render, with no text outside its box and no overlaps."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from hybrid_network import diagrams


@pytest.mark.parametrize("name", list(diagrams.DIAGRAMS))
def test_layout_is_clean(name):
    c = diagrams.DIAGRAMS[name]()
    try:
        assert diagrams.layout_problems(c) == []
    finally:
        plt.close(c.fig)


def test_draw_all_writes_four_pngs(tmp_path):
    paths = diagrams.draw_all(tmp_path)
    assert sorted(p.name for p in paths) == [
        "evaluation.png",
        "hybrid.png",
        "pipeline.png",
        "training.png",
    ]
    for p in paths:
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert p.stat().st_size > 10_000
