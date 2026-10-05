"""Figure style: a pinned light surface, ink colours and the arm palette.

The figures are rasterised on an explicit light surface, so the palette is
read against that surface whatever the viewer's theme. The four categorical
colours were checked for all pairs on the light surface, for normal and
colour-deficient vision.
"""

from __future__ import annotations

import matplotlib as mpl

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e3e2de"

#: The four competing arms: blue, orange, aqua, violet.
ARM_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]


def use_style() -> None:
    """Set the rcParams every figure in the package uses."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.dpi": 160,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 7,
            "figure.constrained_layout.use": True,
        }
    )


def _style(ax, title=None, xlabel=None, ylabel=None):
    """Title on the left in ink, axis labels in the secondary ink."""
    if title:
        ax.set_title(title, color=INK, loc="left", fontweight="normal")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2)
    return ax
