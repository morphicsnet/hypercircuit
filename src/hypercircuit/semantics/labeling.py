from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from hypercircuit.utils.io import save_jsonl, write_json
from hypercircuit.utils.seed import seed_context
from hypercircuit.utils.config import Config


class LabelSchema(BaseModel):
    """Mapping of feature id to human-readable label."""
    labels: Dict[int, str] = {}


def select_exemplars(activations: np.ndarray, k: int = 3) -> Dict[int, List[int]]:
    """Pick top-k samples per feature based on activation magnitude."""
    n_samples, n_features = activations.shape
    out: Dict[int, List[int]] = {}
    for j in range(n_features):
        topk = np.argsort(-activations[:, j])[:k].tolist()
        out[j] = topk
    return out


def agreement_jaccard(labels_a: Dict[int, str], labels_b: Dict[int, str]) -> float:
    """Simple label agreement using Jaccard over label sets."""
    sa, sb = set(labels_a.values()), set(labels_b.values())
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb) or 1
    return inter / union


# ----------------------------
# Week 7: Label finalization
# ----------------------------

def _stable_ids_for_family(family: str, n: int) -> List[str]:
    return [f"{family}::ens_{i:02d}" for i in range(n)]


def _deterministic_exemplars(n_samples: int, k: int, seed: int) -> List[int]:
    with seed_context(seed):
        rng = np.random.default_rng(seed)
        if n_samples <= 0:
            return []
        idxs = rng.integers(0, n_samples, size=max(0, k))
        return sorted(set(int(x) for x in idxs.tolist()))


def compute_agreement_metrics(labels_by_family: Mapping[str, List[Mapping[str, Any]]], seed: int) -> Mapping[str, Any]:
    """
    Stubbed inter-rater agreement and drift checks (deterministic).
    Returns per-family coverage and a global mock kappa.
    """
    per_family_cov: Dict[str, float] = {}
    for fam, rows in labels_by_family.items():
        # coverage as proportion of labels to a small target (mock)
        target = max(1, len(rows))
        per_family_cov[fam] = float(len(rows)) / float(target)
    # deterministic mock kappa
    with seed_context(seed + 777):
        rng = np.random.default_rng(seed + 777)
        kappa = float(np.round(0.60 + 0.10 * rng.random(), 2))  # ~0.60-0.70
    return {
        "per_family_coverage": per_family_cov,
        "kappa_mock": kappa,
        "drift_checks": {fam: {"drift_detected": False} for fam in labels_by_family.keys()},
    }


def finalize_labels(cfg: Config, run_dir: Path, families: Optional[List[str]] = None) -> Mapping[str, str]:
    """
    Deterministic label finalization for Week 7.
    Produces:
      - labels.jsonl with fields:
        ensemble_id, family, label_text, evidence_tokens, context_summary, confidence,
        uncertainty, exemplars, annotator_stub, seed, schema_version
      - label_report.json with per-family coverage, mock agreement metrics, and drift checks.
    """
    # Resolve families
    fams: List[str]
    if families:
        fams = list(dict.fromkeys(families))
    elif getattr(cfg.dictionary, "families", None):
        fams = list(dict.fromkeys(cfg.dictionary.families or []))
    else:
        default = getattr(cfg.discovery.week2_screening, "top_families", []) or []
        if not default:
            default = [getattr(cfg.dataset, "task_family", None) or cfg.dataset.name]
        fams = list(dict.fromkeys(default))

    fams = sorted(fams)
    run_dir.mkdir(parents=True, exist_ok=True)
    labels_path = run_dir / "labels.jsonl"
    report_path = run_dir / "label_report.json"

    # Deterministic knobs
    seed0 = (cfg.run.seeds[0] if getattr(cfg.run, "seeds", None) else cfg.run.seed) or 0
    top_k = int(getattr(cfg.labeling, "exemplars_top_k", 3))
    n_samples = int(getattr(cfg.dataset, "n_samples", 64))

    rows: List[Mapping[str, Any]] = []
    by_family: Dict[str, List[Mapping[str, Any]]] = {}

    for fam in fams:
        # mock accepted ensembles per family
        n_accepted = max(3, min(10, int(getattr(cfg.dictionary, "min_passed_per_top_family", 5))))
        ens_ids = _stable_ids_for_family(fam, n_accepted)
        fam_rows: List[Mapping[str, Any]] = []

        for j, ens_id in enumerate(ens_ids):
            s = seed0 + hash((fam, j)) % (2**31 - 1)
            exemplars = _deterministic_exemplars(n_samples=n_samples, k=top_k, seed=s)
            evidence = exemplars[: max(1, min(2, len(exemplars)))]
            confidence = float(0.75 + 0.05 * ((j % 3) - 1))  # 0.70, 0.75, 0.80 pattern
            uncertainty = float(max(0.0, 1.0 - confidence))

            rec: Dict[str, Any] = {
                "ensemble_id": ens_id,
                "family": fam,
                "label_text": f"{fam} pattern {j:02d}",
                "evidence_tokens": evidence,
                "context_summary": f"Mock context for {fam} ensemble {j}",
                "confidence": confidence,
                "uncertainty": uncertainty,
                "exemplars": exemplars,
                "annotator_stub": "mock_v1",
                "seed": int(s),
                "schema_version": "0.1.0",
            }
            rows.append(rec)
            fam_rows.append(rec)

        by_family[fam] = fam_rows

    save_jsonl(labels_path, rows)

    # Agreement + drift (mock, deterministic)
    agr = compute_agreement_metrics(by_family, seed=seed0)
    report = {
        "schema_version": "0.1.0",
        "families": fams,
        "coverage": agr.get("per_family_coverage", {}),
        "agreement": {"kappa_mock": agr.get("kappa_mock", 0.0)},
        "drift": agr.get("drift_checks", {}),
        "counts": {"total_labels": len(rows)},
    }
    write_json(report_path, report)

    return {"labels_path": str(labels_path), "report_path": str(report_path)}
