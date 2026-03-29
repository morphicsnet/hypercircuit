from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from hypercircuit.blt_io import run_blt_analysis


def _parse(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Hypercircuit BLT downstream analysis")
    parser.add_argument("manifest_path")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse(argv)
    path = run_blt_analysis(args.manifest_path, output_dir=args.output_dir)
    print(Path(path))


if __name__ == "__main__":
    main()
