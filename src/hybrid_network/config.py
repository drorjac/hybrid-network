"""Where the package reads from and writes to.

The project root is the folder that holds `pyproject.toml`: the checkout when
the package runs from `src/`, and the working directory otherwise, so an
installed copy never writes inside site-packages. Under the root:

    results/   tables written by the study and the analysis
    figures/   figures written by `doc` and `diagrams`
    .cache/    per-job results and cached samples
    docs/      generated pages

Two environment variables move the outputs, for example to a scratch
directory in a smoke test:

    HYBRID_NETWORK_RESULTS_DIR   default <root>/results
    HYBRID_NETWORK_CACHE_DIR     default <root>/.cache

The raw archives are found through CML_DATA_ROOT (see `sources.common`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent


def project_root() -> Path:
    """The folder containing pyproject.toml, or the working directory."""
    # <root>/src/hybrid_network/config.py -> <root>
    candidate = _PACKAGE_DIR.parent.parent
    if (candidate / "pyproject.toml").is_file():
        return candidate
    return Path.cwd()


def _path_from_env(var: str, default: Path) -> Path:
    raw = os.environ.get(var)
    return Path(raw).expanduser().resolve() if raw else default


@dataclass(frozen=True)
class Settings:
    """Resolved paths. Immutable; call `reset_settings()` to re-read them."""

    root: Path
    results_dir: Path
    figures_dir: Path
    cache_dir: Path
    docs_dir: Path

    def results(self, name: str) -> Path:
        """`results/<name>/`, created if needed."""
        path = self.results_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def figures(self, name: str) -> Path:
        """`figures/<name>/`, created if needed."""
        path = self.figures_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The process-wide settings, read once from the environment."""
    root = project_root()
    return Settings(
        root=root,
        results_dir=_path_from_env("HYBRID_NETWORK_RESULTS_DIR", root / "results"),
        figures_dir=root / "figures",
        cache_dir=_path_from_env("HYBRID_NETWORK_CACHE_DIR", root / ".cache"),
        docs_dir=root / "docs",
    )


def reset_settings() -> None:
    """Forget the cached settings, so a test can change the environment."""
    get_settings.cache_clear()
