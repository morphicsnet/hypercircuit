#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from hypercircuit.discovery.aggregate import run_week2_screening
from hypercircuit.utils.io import read_json


def _parse(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Week 2: run multi-family synergy screening aggregator.")
    p.add_argument(
        "--config",
        nargs="+",
        default=["configs/base.yaml", "configs/discovery.yaml"],
        help="Base + discovery config files (merged in order).",
    )
    p.add_argument(
        "--overlay",
        nargs="+",
        default=None,
        help="Dataset overlay YAMLs (task families). If omitted, uses built-in default list.",
    )
    p.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        help="Config overrides in dotted.key=value form (e.g., discovery.week2_screening.replicates_k=3).",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse(argv)
    report_path = run_week2_screening(
        config_paths=args.config,
        overlay_paths=args.overlay,
        overrides=args.override,
    )
    payload = read_json(Path(report_path))
    families = payload.get("families", {})

    # Print per-family summaries
    for fam in sorted(families.keys()):
        block = families.get(fam, {})
        counts = block.get("counts", {})
        stab = block.get("stability", {})
        print(
            "[week2] {fam} total={tot} synergy={syn} stability={stab_count} stability_multi={sm:.3f}".format(
                fam=fam,
                tot=int(counts.get("n_candidates_total", 0)),
                syn=int(counts.get("n_passed_synergy", 0)),
                stab_count=int(counts.get("n_passed_stability", 0)),
                sm=float(stab.get("stability_multi", 0.0)),
            )
        )


if __name__ == "__main__":
    main()