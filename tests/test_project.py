"""Configuration, checks, the command line, the cache key and the results page."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from hybrid_network import cli, config, doc, study
from hybrid_network.checks import require
from hybrid_network.seeds import REPORT_SEEDS, TUNE_SEEDS


def test_require_raises_value_error():
    require(True, "never raised")
    with pytest.raises(ValueError, match="bad input"):
        require(False, "bad input")


def test_seeds_are_disjoint():
    assert REPORT_SEEDS == (11, 23, 42)
    assert not set(TUNE_SEEDS) & set(REPORT_SEEDS)


def test_root_holds_pyproject():
    assert (config.project_root() / "pyproject.toml").is_file()


def test_env_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("HYBRID_NETWORK_RESULTS_DIR", str(tmp_path / "r"))
    monkeypatch.setenv("HYBRID_NETWORK_CACHE_DIR", str(tmp_path / "c"))
    config.reset_settings()
    try:
        s = config.get_settings()
        assert s.results_dir == (tmp_path / "r").resolve()
        assert s.cache_dir == (tmp_path / "c").resolve()
        assert s.results("cml").is_dir()
    finally:
        config.reset_settings()


def test_cache_key_is_unchanged():
    """Job keys equal those of the code that produced the cached results, so
    a copied .cache/cml/ is reused instead of retrained."""
    assert study.VERSION == 9
    assert study.Job("openmrg", 11).key() == "d2ffa087062d0e35483f"
    assert study.Job("openmrg", 11, quick=True).key() == "e3d7d30aa37ede5c5249"


def test_cli_parses():
    p = cli._parser()
    a = p.parse_args(["run", "--stage", "main", "noise", "--workers", "2", "--quick"])
    assert a.stage == ["main", "noise"] and a.workers == 2 and a.quick
    assert p.parse_args(["doc"]).command == "doc"
    assert p.parse_args(["diagrams"]).command == "diagrams"
    assert p.parse_args(["build"]).only == list(cli.SOURCES)


def _fake_runs() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    rows = []
    for d in ("openmrg", "sim_iid"):
        for s in REPORT_SEEDS:
            for a in doc.LABEL:
                rows.append(
                    {
                        "stage": "main",
                        "dataset": d,
                        "seed": s,
                        "budget": 1.0,
                        "noise_db": np.nan,
                        "arm": a,
                        "nrmse": float(rng.uniform(1, 3)),
                        "nrmse_link_median": float(rng.uniform(1, 3)),
                        "nbias": float(rng.normal()),
                        "corr": float(rng.uniform()),
                        "nrmse_wet": float(rng.uniform(0.5, 1)),
                        "gate_mean": float(rng.uniform()),
                        "gate_wet": float(rng.uniform()),
                    }
                )
    info = pd.DataFrame(
        {
            "dataset": ["openmrg", "sim_iid"],
            "n_links": [10, 3],
            "n_samples": [100, 30],
            "wet_fraction": [0.05, 0.07],
            "mean_rain_mmh": [0.1, 0.2],
            "freq_min": [28.0, 18.0],
            "freq_max": [39.0, 38.0],
            "len_min": [0.5, 1.0],
            "len_max": [7.0, 4.0],
        }
    )
    return pd.DataFrame(rows), info


def test_render_doc_writes_results_md(tmp_path, monkeypatch):
    settings = replace(
        config.get_settings(),
        root=tmp_path,
        results_dir=tmp_path / "results",
        figures_dir=tmp_path / "figures",
        cache_dir=tmp_path / ".cache",
        docs_dir=tmp_path / "docs",
    )
    monkeypatch.setattr(doc, "get_settings", lambda: settings)
    runs, info = _fake_runs()
    out = settings.results("cml")
    runs.to_csv(out / "runs.csv", index=False)
    info.to_csv(out / "datasets.csv", index=False)
    text = doc.render_doc()
    page = tmp_path / "docs" / "RESULTS.md"
    assert page.read_text() == text
    assert not (tmp_path / "README.md").exists()
    assert "](../figures/cml/main_nrmse.png)" in text
    assert (tmp_path / "figures" / "cml" / "main_nrmse.png").is_file()
