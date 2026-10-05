"""Block diagrams of the pipeline, the hybrid model, training and evaluation.

Four figures are drawn with matplotlib on the light surface of `style` and
saved at 150 dpi to figures/diagrams/:

    pipeline.png     archives and simulator to the results page
    hybrid.png       the two branches, the gate, the blend and the loss
    training.png     the baselines and the three hybrid training schemes
    evaluation.png   test predictions to metrics, classes, detection, events

Coordinates are in tenths of an inch, so one data unit is the same length on
both axes and a 9 pt line of text is about 1.7 units tall. Every box records
the text drawn inside it, and `layout_problems` reports any text that leaves
its box, any two boxes that overlap and any free label that touches a box.
The tests require that list to be empty for each figure.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.text import Text

from .config import get_settings
from .style import ARM_PALETTE, INK, INK_2, INK_MUTED, SURFACE

DPI = 150
TITLE_PT = 10.0
BODY_PT = 9.0
LEADING = 1.38  # line height over font size
UNIT_PT = 7.2  # points per data unit (a tenth of an inch)

PL = ARM_PALETTE[0]  # power law: blue
NN = ARM_PALETTE[1]  # GRU: orange
HYB = ARM_PALETTE[3]  # gate and blend: violet
PHASED = ARM_PALETTE[2]  # phased hybrid: aqua
DATA_FILL = "#f1f0ec"


@dataclass
class Canvas:
    """A figure whose axes span it, in tenths of an inch."""

    fig: Figure
    ax: plt.Axes
    boxes: list[tuple[FancyBboxPatch, list[Text]]] = field(default_factory=list)
    labels: list[Text] = field(default_factory=list)


def _canvas(width: float, height: float) -> Canvas:
    fig = plt.figure(figsize=(width / 10, height / 10), dpi=DPI, layout="none")
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_facecolor(SURFACE)
    ax.axis("off")
    return Canvas(fig, ax)


def _line_height(pt: float) -> float:
    return pt * LEADING / UNIT_PT


def box(
    c: Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: Sequence[str] = (),
    edge: str = INK_2,
    fill: str = SURFACE,
    align: str = "center",
) -> tuple[float, float, float, float]:
    """A rounded box with a bold title and body lines, centred vertically.

    Returns (x, y, w, h) so callers can attach arrows to its edges."""
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0,rounding_size=1.0",
        facecolor=fill,
        edgecolor=edge,
        linewidth=1.5,
        zorder=2,
    )
    c.ax.add_patch(patch)
    heights = [_line_height(TITLE_PT)] + [_line_height(BODY_PT)] * len(lines)
    top = y + h / 2 + sum(heights) / 2
    texts = []
    tx = x + w / 2 if align == "center" else x + 1.2
    for i, (s, lh) in enumerate(zip([title, *lines], heights, strict=True)):
        texts.append(
            c.ax.text(
                tx if i else x + w / 2,
                top - lh / 2,
                s,
                ha=align if i else "center",
                va="center",
                fontsize=TITLE_PT if i == 0 else BODY_PT,
                fontweight="bold" if i == 0 else "normal",
                color=INK if i == 0 else INK_2,
                zorder=3,
            )
        )
        top -= lh
    c.boxes.append((patch, texts))
    return x, y, w, h


def label(
    c: Canvas,
    x: float,
    y: float,
    s: str,
    ha: str = "center",
    va: str = "center",
    color: str = INK_2,
    size: float = BODY_PT,
    style: str = "normal",
) -> Text:
    """Free text, such as an arrow label; checked against every box."""
    t = c.ax.text(
        x,
        y,
        s,
        ha=ha,
        va=va,
        fontsize=size,
        color=color,
        style=style,
        zorder=3,
        bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.6},
    )
    c.labels.append(t)
    return t


def arrow(
    c: Canvas,
    points: Sequence[tuple[float, float]],
    color: str = INK_MUTED,
    head: bool = True,
) -> None:
    """A polyline through `points`, with an arrow head on the last segment."""
    xs, ys = zip(*points, strict=True)
    if len(points) > 2:
        c.ax.plot(
            xs[:-1], ys[:-1], color=color, lw=1.4, zorder=1, solid_capstyle="butt"
        )
    if head:
        c.ax.add_patch(
            FancyArrowPatch(
                points[-2],
                points[-1],
                arrowstyle="-|>",
                mutation_scale=12,
                lw=1.4,
                color=color,
                shrinkA=0,
                shrinkB=0,
                zorder=1,
            )
        )
    else:
        c.ax.plot(xs[-2:], ys[-2:], color=color, lw=1.4, zorder=1)


def dot(c: Canvas, x: float, y: float, color: str = INK_MUTED) -> None:
    """A junction where one line splits into two."""
    c.ax.plot([x], [y], "o", ms=4, color=color, zorder=1)


def heading(c: Canvas, x: float, y: float, s: str) -> None:
    """A section heading above a group of boxes."""
    label(c, x, y, s, ha="left", color=INK, size=TITLE_PT, style="italic")


def layout_problems(c: Canvas) -> list[str]:
    """Text outside its box, overlapping boxes, labels touching boxes."""
    c.fig.canvas.draw()
    r = c.fig.canvas.get_renderer()  # type: ignore[attr-defined]
    fig_bb = c.fig.bbox
    out = []
    pboxes = [(p.get_window_extent(r), ts) for p, ts in c.boxes]
    for i, (bb, texts) in enumerate(pboxes):
        name = texts[0].get_text()
        for t in texts:
            tb = t.get_window_extent(r)
            if (
                tb.x0 < bb.x0 + 2
                or tb.x1 > bb.x1 - 2
                or tb.y0 < bb.y0 + 1
                or tb.y1 > bb.y1 - 1
            ):
                out.append(f"text {t.get_text()!r} leaves box {name!r}")
        if (
            bb.x0 < fig_bb.x0
            or bb.x1 > fig_bb.x1
            or bb.y0 < fig_bb.y0
            or bb.y1 > fig_bb.y1
        ):
            out.append(f"box {name!r} leaves the figure")
        for bb2, texts2 in pboxes[i + 1 :]:
            if bb.overlaps(bb2):
                out.append(f"boxes {name!r} and {texts2[0].get_text()!r} overlap")
    lbs = [(t, t.get_window_extent(r)) for t in c.labels]
    for t, tb in lbs:
        if (
            tb.x0 < fig_bb.x0
            or tb.x1 > fig_bb.x1
            or tb.y0 < fig_bb.y0
            or tb.y1 > fig_bb.y1
        ):
            out.append(f"label {t.get_text()!r} leaves the figure")
        for bb, texts in pboxes:
            if tb.overlaps(bb):
                out.append(
                    f"label {t.get_text()!r} touches box {texts[0].get_text()!r}"
                )
    for i, (t, tb) in enumerate(lbs):
        for t2, tb2 in lbs[i + 1 :]:
            if tb.overlaps(tb2):
                out.append(f"labels {t.get_text()!r} and {t2.get_text()!r} overlap")
    return out


# ---------------------------------------------------------------------------


def pipeline() -> Canvas:
    """From the archives and the simulator to docs/RESULTS.md."""
    c = _canvas(132, 84)
    heading(c, 2, 81.5, "inputs (archives under CML_DATA_ROOT, default ~/data/cml)")
    arch = [
        ("OpenMRG", "Gothenburg, TSL − RSL, gauges"),
        ("OpenRainER", "Emilia-Romagna, radar RADadj"),
        ("Netherlands", "min/max RSL, KNMI gauges"),
        ("OpenMesh", "New York, −RSL, PWS + ASOS"),
    ]
    ys = [70.0, 61.4, 52.8, 44.2]
    mid = (ys[0] + 7 + ys[-1]) / 2
    for (t, s), y in zip(arch, ys, strict=True):
        box(c, 2, y, 29, 7, t, [s], fill=DATA_FILL)
        arrow(c, [(31, y + 3.5), (34, y + 3.5)], head=False)
    arrow(c, [(34, ys[0] + 3.5), (34, ys[-1] + 3.5)], head=False)
    arrow(c, [(34, mid), (38, mid)])
    box(
        c,
        2,
        32.5,
        29,
        9,
        "sim.simulate",
        ["ITU law, noise, 0.3 dB steps", "links (f, L) from OpenMRG"],
        fill=DATA_FILL,
    )

    box(
        c,
        38,
        mid - 8.5,
        25,
        17,
        "sources/<name>.build()",
        [
            "≤ 40 sublinks near",
            "a reference instrument,",
            "attenuation in dB,",
            "reference rain in mm/h",
        ],
    )
    arrow(c, [(63, mid), (68, mid)])
    box(
        c,
        68,
        mid - 8.5,
        26,
        17,
        "common format",
        [
            "links.csv",
            "series.parquet",
            "reference.parquet",
            "meta.json",
            "in <name>/derived/hybrid/",
        ],
        fill=DATA_FILL,
    )
    arrow(c, [(94, mid), (99, mid)])
    box(
        c,
        99,
        mid - 12.5,
        31,
        25,
        "data.make_samples",
        [
            "baseline: causal 24 h",
            "rolling median per link",
            "x = A − baseline (excess)",
            "input: 30-min history",
            "target: mean rain over a bin",
            "of max(10 min, reference step)",
            "split by day 70/15/15,",
            "stratified by daily rain",
            "study.samples: noise screen,",
            "drop links with robust sd",
            "> max(0.6 dB, 1.5 steps)",
        ],
    )
    # the simulator skips the builders: it returns the same four tables
    arrow(c, [(31, 37.0), (112, 37.0), (112, mid - 12.5)])
    label(c, 70, 37.0, "the same four tables, built in memory", size=8.5)

    # bottom band: training, study, analysis, page
    arrow(c, [(124, mid - 12.5), (124, 28.5), (16.5, 28.5), (16.5, 24)])
    yb, hb = 2, 22
    box(
        c,
        2,
        yb,
        29,
        hb,
        "train.fit_arms",
        [
            "one dataset, seed, budget",
            "pl_itu: ITU k, α",
            "pl_cal: k, α, τ per link",
            "gru: 2 × GRU 64",
            "hybrid_joint",
            "hybrid_gate",
            "hybrid_phased",
            "test metrics, predictions",
        ],
        edge=HYB,
    )
    arrow(c, [(31, yb + hb / 2), (37, yb + hb / 2)])
    box(
        c,
        37,
        yb,
        33,
        hb,
        "study.run",
        [
            "main: all arms, seeds 11/23/42",
            "budget: 5/15/40 % of train days",
            "noise: sim, sd 0.05−0.8 dB",
            "cache: .cache/cml/<key>.json",
            "results/cml/runs.csv,",
            "datasets.csv, dropped_links.csv",
        ],
    )
    arrow(c, [(70, yb + hb / 2), (76, yb + hb / 2)])
    box(
        c,
        76,
        yb,
        28,
        hb,
        "analysis.run",
        [
            "main-stage test predictions,",
            "seeds pooled",
            "intensity classes",
            "detection at 0.1 mm/h",
            "rain events, 60-min gap",
            "intensity.csv, detection.csv,",
            "events.csv",
        ],
    )
    arrow(c, [(104, yb + hb / 2), (109, yb + hb / 2)])
    box(
        c,
        109,
        yb,
        21,
        hb,
        "doc.render_doc",
        [
            "hybrid-network doc",
            "tables from",
            "results/cml/",
            "figures/cml/",
            "docs/RESULTS.md",
        ],
        fill=DATA_FILL,
    )
    return c


def hybrid() -> Canvas:
    """The two branches, the gate and the blend, as `models.Hybrid` computes
    them, with the training loss of `train._loss`."""
    c = _canvas(126, 66)
    box(
        c,
        2,
        23,
        21,
        14,
        "input x",
        ["excess attenuation, dB", "30-min window", "series time steps"],
        fill=DATA_FILL,
    )
    # power-law branch
    box(
        c,
        30,
        40,
        35,
        17,
        "power-law branch $r_M$",
        [
            r"dead zone: $\max(x-\tau,\ 0)$",
            r"per minute: $(\,\cdot\,/\,kL)^{1/\alpha}$",
            "mean over the minutes of the bin",
            "k, α: ITU-R P.838-3 at f, pol",
            "per link: log k, α, log τ trained",
            "τ starts at 3 noise sd, ≥ 1 step",
        ],
        edge=PL,
    )
    # GRU branch
    box(
        c,
        30,
        4,
        35,
        15,
        "GRU branch $r_D$",
        [
            "per step: [x / L, x / 10]",
            "GRU, 64 units",
            "GRU, 64 units",
            "linear read-out, last step",
            "softplus, so $r_D \\geq 0$",
        ],
        edge=NN,
    )
    arrow(c, [(23, 33), (26.5, 33), (26.5, 48.5), (30, 48.5)])
    arrow(c, [(23, 27), (26.5, 27), (26.5, 11.5), (30, 11.5)])
    arrow(c, [(23, 30), (70, 30)])
    label(c, 46, 30, "sd(x), mean(x)", size=8.5)
    # gate
    box(
        c,
        70,
        20,
        30,
        20,
        "gate g",
        [
            "s = [sd x, mean x, log1p $r_M$,",
            "log1p $r_D$, $r_M - r_D$]",
            "standardised on training bins",
            r"$z = (w + w_\ell)\cdot s + b + b_\ell$",
            r"w, b shared; $w_\ell$, $b_\ell$ per link",
            r"$g = \mathrm{sigmoid}(z)$",
        ],
        edge=HYB,
    )
    # branch outputs to the gate and to the blend
    arrow(c, [(65, 48.5), (113, 48.5), (113, 35)], color=PL)
    arrow(c, [(85, 48.5), (85, 40)], color=PL)
    dot(c, 85, 48.5, PL)
    label(c, 72, 48.5, "$r_M$", color=PL)
    arrow(c, [(65, 11.5), (113, 11.5), (113, 25)], color=NN)
    arrow(c, [(85, 11.5), (85, 20)], color=NN)
    dot(c, 85, 11.5, NN)
    label(c, 72, 11.5, "$r_D$", color=NN)
    # blend
    box(c, 104, 25, 20, 10, "blend", [r"$\hat r = g\,r_M + (1-g)\,r_D$"], edge=HYB)
    arrow(c, [(100, 30), (104, 30)], color=HYB)
    label(c, 102, 32.2, "g", color=HYB)
    # loss
    box(
        c,
        70,
        53,
        54,
        11,
        "training loss",
        [
            r"$\mathrm{MSE}(\hat r, r) + 0.1\,\mathrm{MSE}(r_M, r)"
            r" + 0.1\,\mathrm{MSE}(r_D, r)$",
            "r: reference rain of the bin (gauge or radar)",
        ],
    )
    return c


def training() -> Canvas:
    """The three baselines and the three ways `train.fit_arms` trains a hybrid."""
    c = _canvas(132, 90)
    cols = {"joint": 2.0, "gate": 47.0, "phased": 92.0}
    cw = 38.0
    heading(c, 2, 87, "baselines, each also reported as an arm")
    yb = 74.0
    box(
        c,
        cols["joint"],
        yb,
        cw,
        9,
        "pl_itu",
        ["ITU k, α; τ from the noise", "no training"],
        edge=INK_MUTED,
    )
    box(
        c,
        cols["gate"],
        yb,
        cw,
        9,
        "pl_cal",
        ["log k, α, log τ per link", "Adam, lr 1e-2"],
        edge=PL,
    )
    box(
        c,
        cols["phased"],
        yb,
        cw,
        9,
        "gru",
        ["2 × GRU 64, linear, softplus", "Adam, lr 1e-3"],
        edge=NN,
    )

    step_h, gap, top = 7.5, 3.0, 64.5

    def steps(col: str, items, edge: str) -> list[float]:
        x = cols[col]
        ys = []
        for i, (title, lines) in enumerate(items):
            y = top - step_h - i * (step_h + gap)
            box(c, x, y, cw, step_h, title, lines, edge=edge)
            if i:
                arrow(
                    c,
                    [(x + cw / 2, y + step_h + gap), (x + cw / 2, y + step_h)],
                    color=edge,
                )
            ys.append(y)
        return ys

    start = ("start", ["copies of pl_cal and gru,", "both frozen"])
    init = (
        "per-link gate start",
        ["g* = least-squares blend of", "the frozen branches, per link"],
    )
    steps(
        "joint",
        [
            ("start", ["PL at ITU k, α, τ;", "new GRU and gate"]),
            ("train everything together", ["PL lr 1e-2;", "GRU and gate lr 1e-3"]),
        ],
        HYB,
    )
    steps(
        "gate",
        [
            start,
            init,
            ("train the gate", ["shared and per-link w, b", "lr 1e-2"]),
        ],
        HYB,
    )
    steps(
        "phased",
        [
            start,
            init,
            ("phase 1: gate", ["shared and per-link w, b", "lr 1e-2"]),
            ("phase 2: gate + GRU", ["PL still frozen", "lr 1e-3"]),
            ("phase 3: everything", ["PL lr 1e-3,", "GRU and gate lr 1e-4"]),
        ],
        PHASED,
    )
    heading(c, cols["joint"] + 0.5, 67.3, "hybrid_joint")
    heading(c, cols["gate"] + 0.5, 67.3, "hybrid_gate")
    heading(c, cols["phased"] + 0.5, 67.3, "hybrid_phased")

    # pl_itu gives the joint hybrid its starting law; pl_cal and gru are
    # copied into both pretrained hybrids
    xj, xg, xp = (cols[k] + cw / 2 for k in ("joint", "gate", "phased"))
    arrow(c, [(xj, yb), (xj, top)], color=INK_MUTED)
    bus = 70.5
    arrow(c, [(xg, yb), (xg, top)], color=PL)
    arrow(c, [(xp, yb), (xp, top)], color=NN)
    arrow(c, [(xg, bus), (xp, bus)], color=INK_MUTED, head=False)
    dot(c, xg, bus)
    dot(c, xp, bus)
    label(c, (xg + xp) / 2, bus, "both branches into both", size=8.5)

    box(
        c,
        2,
        2,
        128,
        10.5,
        "every stage",
        [
            "Adam, weight decay 1e-5, at most 15 epochs, early stop on the "
            "validation loss with patience 3, best state kept (the starting "
            "state included)",
            "each epoch draws up to 20 000 wet bins and as many dry ones, "
            "weighted back to the natural wet/dry mix",
            "loss: MSE of the output, plus 0.1 × the MSE of each branch for "
            "the hybrids",
        ],
    )
    return c


def evaluation() -> Canvas:
    """What is computed on the test days, and where it is written."""
    c = _canvas(132, 62)
    groups = [
        (
            "metrics per job: train.metrics",
            [
                "NRMSE = RMSE / mean(r)",
                r"NBIAS = mean($\hat r - r$) / mean(r)",
                r"correlation of $\hat r$ and r",
                "wet NRMSE: bins with r > 0.1 mm/h",
                "median over links of the per-link NRMSE",
            ],
            12.5,
        ),
        (
            "intensity classes: analysis",
            [
                "by reference rate, mm/h: dry < 0.1,",
                "light 0.1−1, moderate 1−5,",
                "heavy 5−20, very heavy > 20;",
                "mean estimate, relative bias, RMSE",
            ],
            10.8,
        ),
        (
            "detection at 0.1 mm/h: analysis",
            [
                "POD: share of wet bins called wet",
                "FAR: share of wet calls that were dry",
                "share of dry bins called wet",
            ],
            9.0,
        ),
        (
            "rain events: analysis",
            [
                "wet reference bins on one link,",
                "gaps of at most 60 min;",
                "error in event total and peak,",
                "by total: < 1, 1−5, 5−20, > 20 mm",
            ],
            10.8,
        ),
    ]
    gap = 3.0
    total = sum(h for *_, h in groups) + gap * (len(groups) - 1)
    y = 2.0 + total
    xm, wm = 42.0, 44.0
    mids = []
    for title, lines, h in groups:
        y -= h
        box(c, xm, y, wm, h, title, lines, edge=HYB if "metrics" in title else INK_2)
        mids.append(y + h / 2)
        y -= gap
    ymid = 2.0 + total / 2
    heading(c, 2, 59, "evaluation on the test days, the same bins for every arm")
    box(
        c,
        2,
        ymid - 9,
        30,
        18,
        "test predictions",
        [
            "15 % of days, stratified",
            "by daily rain, one split",
            "for every arm",
            "one estimate per bin",
            "seeds 11, 23, 42",
        ],
        fill=DATA_FILL,
    )
    arrow(c, [(32, ymid), (37, ymid)], head=False)
    arrow(c, [(37, mids[0]), (37, mids[-1])], head=False)
    for ym in mids:
        arrow(c, [(37, ym), (42, ym)])
    xr = 96.0
    for ym in mids:
        arrow(c, [(86, ym), (91, ym)], head=False)
    arrow(c, [(91, mids[0]), (91, mids[-1])], head=False)
    yres = ymid + 4
    arrow(c, [(91, yres), (xr, yres)])
    box(
        c,
        xr,
        yres - 9,
        34,
        18,
        "results/cml/",
        [
            "runs.csv: metrics per job, arm",
            "intensity.csv",
            "detection.csv",
            "events.csv",
            "analysis_meta.json",
        ],
        fill=DATA_FILL,
    )
    arrow(c, [(xr + 17, yres - 9), (xr + 17, 12)])
    box(
        c,
        xr,
        2,
        34,
        10,
        "hybrid-network doc",
        ["docs/RESULTS.md, figures/cml/", "every number read from the CSVs"],
    )
    return c


def save(c: Canvas, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    c.fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(c.fig)
    return path


DIAGRAMS: dict[str, Callable[[], Canvas]] = {
    "pipeline": pipeline,
    "hybrid": hybrid,
    "training": training,
    "evaluation": evaluation,
}


def draw_all(out_dir: Path | None = None) -> list[Path]:
    """Draw every diagram; return the PNG paths."""
    out = out_dir or get_settings().figures("diagrams")
    return [save(f(), Path(out) / f"{name}.png") for name, f in DIAGRAMS.items()]
