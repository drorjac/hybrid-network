"""Command line: build the datasets, run the study, write the page, draw diagrams.

    hybrid-network build                       reduce every archive to the common format
    hybrid-network run [--stage S ...] [--workers N] [--quick]
                                          train and score every arm, then analyse
    hybrid-network doc                         docs/RESULTS.md and figures from results/
    hybrid-network diagrams                    the block diagrams in figures/diagrams/

`run` without `--quick` also writes the page. With `--quick` it writes to
results/cml_quick/ and leaves the page alone.
"""

from __future__ import annotations

import argparse
import importlib
import logging
from collections.abc import Sequence

SOURCES = ("openmrg", "openrainer", "netherlands", "openmesh")
STAGES = ("main", "budget", "noise")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hybrid-network",
        description="Rain from commercial microwave links: power law, GRU and "
        "gated hybrid.",
    )
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="reduce the archives under CML_DATA_ROOT")
    b.add_argument(
        "--only",
        nargs="+",
        choices=SOURCES,
        default=list(SOURCES),
        help="datasets to build (default: all)",
    )
    r = sub.add_parser("run", help="the experiments, then the page")
    r.add_argument(
        "--stage",
        nargs="+",
        choices=STAGES,
        default=list(STAGES),
        help="stages to run (default: all)",
    )
    r.add_argument("--workers", type=int, default=4, help="parallel processes")
    r.add_argument("--quick", action="store_true", help="short runs, one seed")
    sub.add_parser("doc", help="docs/RESULTS.md and figures/cml/ from results/cml/")
    sub.add_parser("diagrams", help="block diagrams in figures/diagrams/")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.command == "build":
        for name in args.only:
            mod = importlib.import_module(f"hybrid_network.sources.{name}")
            print(f"{name}: {mod.build()}")
        return 0
    if args.command == "run":
        from .study import run

        run(stages=tuple(args.stage), workers=args.workers, quick=args.quick)
        if not args.quick:
            from . import doc

            doc.render_doc()
        return 0
    if args.command == "doc":
        from . import doc

        doc.render_doc()
        print(doc.output_path())
        return 0
    if args.command == "diagrams":
        from .diagrams import draw_all

        for path in draw_all():
            print(path)
        return 0
    return 2
