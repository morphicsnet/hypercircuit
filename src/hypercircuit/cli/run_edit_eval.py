from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np

from hypercircuit.steering.edits import compute_edit_map, apply_edits
from hypercircuit.utils.config import Config, load_config, stage_path
from hypercircuit.utils.io import load_jsonl, read_json, write_json
from hypercircuit.utils.registry import start_run, log_artifact, finalize_run


def _parse(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulate safety edits.")
    p.add_argument("--config", nargs="+", required=False, default=["configs/base.yaml", "configs/editing.yaml"])
    p.add_argument("-o", "--override", action="append", default=[])
    p.add_argument("--run-dir", default=None, help="Legacy path form runs/<run_id> (overrides run.output_dir/run_id).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse(argv)
    cfg: Config = load_config(args.config, args.override)

    cfg_dict = cfg.model_dump()
    run_sec = cfg_dict.setdefault("run", {})
    if args.run_dir:
        p = Path(args.run_dir)
        run_sec["output_dir"] = str(p.parent)
        run_sec["run_id"] = p.name
    else:
        legacy = run_sec.get("run_dir")
        if legacy and not run_sec.get("run_id"):
            lp = Path(legacy)
            run_sec["output_dir"] = str(lp.parent)
            run_sec["run_id"] = lp.name

    run_id, run_dir = start_run(cfg_dict, stage_name="editing", config_paths=args.config)

    events = load_jsonl(stage_path(run_dir, "logs.jsonl"))
    n_samples = len(events)
    n_features = cfg.dataset.n_features
    X = np.zeros((n_samples, n_features), dtype=float)
    for ev in events:
        i = int(ev["sample_id"])
        for j in ev.get("active", []):  # type: ignore[assignment]
            X[i, int(j)] = 1.0

    surrogate = read_json(stage_path(run_dir, "surrogate.json"))
    weights = np.array(surrogate.get("weights") or np.zeros(n_features), dtype=float)

    edit_map = compute_edit_map(weights, scale=cfg.editing.scale)
    X_ed = apply_edits(X, edit_map)

    deltas = float(np.mean(X.sum(axis=1) - X_ed.sum(axis=1)))
    payload = {"avg_activation_delta": deltas, "scale": cfg.editing.scale}
    out_path = stage_path(run_dir, "edits.json")
    write_json(out_path, payload)
    log_artifact(out_path, kind="edits", metadata={})
    finalize_run(status="success", metrics_dict=payload)


if __name__ == "__main__":
    main()
