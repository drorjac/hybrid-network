"""Input checks that stay active under `python -O`."""

from __future__ import annotations


def require(condition: bool, message: str) -> None:
    """Raise ValueError with `message` unless `condition` holds."""
    if not condition:
        raise ValueError(message)
