from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from hypercircuit.causal.harness import CausalHarness
from hypercircuit.utils.config import Config, stage_path
from hypercircuit.utils.io import load_jsonl, save_jsonl, write_json
from hypercircuit.utils.registry import log_artifact
from hypercircuit.utils.seed import expand_seeds


def _median(vals: Sequence[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(float(x) for x in vals)
    n = len(s)
    m = n // 2
    if n % 2 == 1:
        return float(s[m])
    return float(0.5 * (s[m - 1] + s[m]))


def _resolve_families(cfg: Config, families_cli: Optional[Sequence[str]] = None) -> List[str]:
    if families_cli:
        return list(dict.fromkeys(families_cli))
    mx = getattr(cfg, "matrix", None)
    if mx and getattr(mx, "families", None):
        return list(mx.families or [])
    dic = getattr(cfg, "dictionary", None)
    if dic and getattr(dic, "families", None):
        return list(dic.families or [])
    wk2 = getattr(cfg.discovery, "week2_screening", None)
    if wk2 and getattr(wk2, "top_families", None):
        return list(wk2.top_families or [])
    fam = cfg.run.task_family or cfg.dataset.task_family or cfg.dataset.name
    return [str(fam)]


def _resolve_methods(cfg: Config, methods_cli: Optional[Sequence[str]] = None) -> List[str]:
    if methods_cli:
        return list(dict.fromkeys(methods_cli))
    mx = getattr(cfg, "matrix", None)
    if mx and getattr(mx, "methods", None):
        return list(mx.methods or [])
    return [
        "pairwise_baseline",
        "no_ensembles_patching",
        "auto_circuit_no_hyper",
        "std_steering",
        "hypercircuit_full",
    ]


def run_robustness_evaluation(
    *,
    cfg: Config,
    run_dir: Path,
    families: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
    use_remaining: bool = True,
) -> Dict[str, Any]:
    """
    Week 6: Robustness evaluation over matrix cells (mock/deterministic).

    Inputs:
      - Loads matrix cell effects from remaining_matrix_results.jsonl (default) or matrix_results.jsonl.
      - Uses harness.evaluate_paraphrase_and_adversarial to simulate paraphrase/adversarial variants.

    Writes:
      - matrix_robustness.jsonl: per-cell robustness rows
      - robustness_summary.json: aggregate medians and counts

    Returns:
      {
        "robustness_path": Path,
        "summary_path": Path,
        "summary": {...}
      }
    """
    fams = _resolve_families(cfg, families)
    meths = _resolve_methods(cfg, methods)

    # Locate matrix input
    candidate_names = ["remaining_matrix_results.jsonl", "matrix_results.jsonl"] if use_remaining else ["matrix_results.jsonl", "remaining_matrix_results.jsonl"]
    matrix_path: Optional[Path] = None
    for nm in candidate_names:
        p = stage_path(run_dir, nm)
        if p.exists():
            matrix_path = p
            break
    if matrix_path is None:
        # Emit empty artifacts to keep deterministic behavior
        rob_path = stage_path(run_dir, "matrix_robustness.jsonl")
        save_jsonl(rob_path, [])
        log_artifact(rob_path, kind="matrix_robustness", metadata={"cells": 0})
        summ = {
            "n_cells": 0,
            "n_seeds": 0,
            "effect_persistence_median": 0.0,
            "seed_stability_median": 0.0,
            "ood_delta_median": 0.0,
            "source": None,
        }
        summ_path = stage_path(run_dir, "robustness_summary.json")
        write_json(summ_path, summ)
        log_artifact(summ_path, kind="robustness_summary")
        return {"robustness_path": rob_path, "summary_path": summ_path, "summary": summ}

    rows = load_jsonl(matrix_path)
    # Filter cell set
    rows = [r for r in rows if (str(r.get("family")) in fams) and (str(r.get("method")) in meths)]

    # Resolve robustness knobs
    g4 = getattr(getattr(cfg.causal, "gate4", None), "robustness", None)
    n_paraphrases = int(getattr(g4, "n_paraphrases", 2) if g4 else 2)
    n_adversarial = int(getattr(g4, "n_adversarial", 2) if g4 else 2)
    n_seed_cap = int(getattr(g4, "seeds", 5) if g4 else 5)
    seeds_all = expand_seeds(cfg.run.seeds)[: max(1, n_seed_cap)]

    harness = CausalHarness(ablation_strength=cfg.causal.ablation_strength)

    out_rows: List[Dict[str, Any]] = []
    for r in rows:
        fam = str(r.get("family"))
        method = str(r.get("method"))
        checksum = str(r.get("checksum_id", ""))

        expected_effect = float(r.get("effect_size", 0.0))
        rob = harness.evaluate_paraphrase_and_adversarial(
            expected_effect=expected_effect,
            n_paraphrases=n_paraphrases,
            n_adversarial=n_adversarial,
            seeds=seeds_all,
            seed=seeds_all[0] if seeds_all else 0,
        )
        baseline_med = _median([float(x) for x in rob.get("baseline_effects", [])])

        out_rows.append(
            {
                "family": fam,
                "method": method,
                "checksum_id": checksum,
                "n_paraphrases": int(n_paraphrases),
                "n_adversarial": int(n_adversarial),
                "n_seeds": int(len(seeds_all)),
                "effect_persistence": float(rob.get("effect_persistence", 0.0)),
                "seed_stability": float(rob.get("seed_stability", 0.0)),
                "ood_delta": float(rob.get("ood_delta", 0.0)),
                "baseline_effect_median": float(baseline_med),
            }
        )

    # Write per-cell robustness results
    rob_path = stage_path(run_dir, "matrix_robustness.jsonl")
    save_jsonl(rob_path, out_rows)
    log_artifact(rob_path, kind="matrix_robustness", metadata={"cells": len(out_rows)})

    # Aggregate summary
    ep_vals = [float(x.get("effect_persistence", 0.0)) for x in out_rows]
    ss_vals = [float(x.get("seed_stability", 0.0)) for x in out_rows]
    od_vals = [float(x.get("ood_delta", 0.0)) for x in out_rows]
    summary = {
        "source": str(matrix_path.name),
        "n_cells": int(len(out_rows)),
        "n_seeds": int(len(seeds_all)),
        "effect_persistence_median": _median(ep_vals),
        "seed_stability_median": _median(ss_vals),
        "ood_delta_median": _median(od_vals),
    }

    summ_path = stage_path(run_dir, "robustness_summary.json")
    write_json(summ_path, summary)
    log_artifact(summ_path, kind="robustness_summary")

    return {"robustness_path": rob_path, "summary_path": summ_path, "summary": summary}


__all__ = ["run_robustness_evaluation"]