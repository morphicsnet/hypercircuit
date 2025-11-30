from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from hypercircuit.causal.harness import CausalHarness
from hypercircuit.utils.config import Config, stage_path
from hypercircuit.utils.io import load_jsonl, save_jsonl
from hypercircuit.utils.registry import log_artifact
from hypercircuit.utils.seed import seed_context


def _presence_index(events: Sequence[Mapping[str, Any]]) -> Tuple[Dict[int, set], int]:
    by_sample: Dict[int, set] = {}
    max_id = -1
    for ev in events:
        sid = int(ev.get("sample_id", 0))
        nt = str(ev.get("node_type", ""))
        by_sample.setdefault(sid, set()).add(nt)
        if sid > max_id:
            max_id = sid
    n_samples = max(0, max_id + 1)
    return by_sample, n_samples


def _presence_matrix(members: List[str], by_sample: Mapping[int, set], n_samples: int) -> np.ndarray:
    X = np.zeros((n_samples, len(members)), dtype=float)
    for i in range(n_samples):
        present = by_sample.get(i, set())
        for j, m in enumerate(members):
            X[i, j] = 1.0 if m in present else 0.0
    return X


def _index_params_by_eid(params: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    idx: Dict[str, Mapping[str, Any]] = {}
    for r in params:
        eid = r.get("ensemble_id")
        if eid is not None:
            idx[str(eid)] = r
    return idx


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


@dataclass(frozen=True)
class _NecessityKnobs:
    disable_higher_order: bool
    retrain_on_subset: bool
    top_k_per_family: int


def _select_top_by_cv_for_family(
    fam: str,
    params_idx: Mapping[str, Mapping[str, Any]],
    ensembles: Sequence[Mapping[str, Any]],
    top_k: int,
) -> List[Mapping[str, Any]]:
    fam_ens = [e for e in ensembles if str(e.get("family")) == fam]
    def _cv(e: Mapping[str, Any]) -> float:
        pr = params_idx.get(str(e.get("id")))
        return float(pr.get("cv_score", 0.0)) if pr else 0.0
    fam_ens.sort(key=lambda d: -_cv(d))
    return fam_ens[: top_k]


def _variant_weights(
    base_w: np.ndarray,
    *,
    disable_higher_order: bool,
    retrain_on_subset: bool,
) -> np.ndarray:
    if not disable_higher_order:
        return base_w.copy()
    if base_w.size == 0:
        return base_w.copy()
    w = base_w.copy()
    # Keep only strongest term as first-order surrogate for necessity test
    top = int(np.argmax(w))
    mask = np.zeros_like(w)
    mask[top] = 1.0
    if retrain_on_subset:
        # Simple renormalization: concentrate total weight mass on top feature
        total = float(np.sum(w))
        w2 = np.zeros_like(w)
        w2[top] = max(0.0, total)
        return w2
    else:
        # Drop all but the top weight (no retraining)
        w2 = np.zeros_like(w)
        w2[top] = float(w[top])
        return w2


def _median(vals: Sequence[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(float(x) for x in vals)
    n = len(s)
    m = n // 2
    if n % 2 == 1:
        return float(s[m])
    return float(0.5 * (s[m - 1] + s[m]))


def run_higher_order_necessity(
    *,
    cfg: Config,
    run_dir: Path,
    families: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Week 6: Higher-order necessity ablation (mock/deterministic).

    For each family, selects top-K ensembles by CV. For each ensemble:
      - Reconstruct synthetic target from presence matrix.
      - Compute baseline predictive MSE and causal delta (using current weights).
      - Disable higher-order terms per config (keep top-1 weight); optionally "retrain" by redistributing mass.
      - Recompute predictive MSE and causal delta with variant weights.
      - Report drops and an irreducibility score from subset ablations.

    Writes:
      - necessity_results.jsonl (per-ensemble rows)
    """
    fams = _resolve_families(cfg, families)

    # Load artifacts
    events_path = stage_path(run_dir, "logs.jsonl")
    ensembles_path = stage_path(run_dir, "ensembles.jsonl")
    params_path = stage_path(run_dir, "surrogates_params.jsonl")
    events = load_jsonl(events_path) if events_path.exists() else []
    ensembles = load_jsonl(ensembles_path) if ensembles_path.exists() else []
    params = load_jsonl(params_path) if params_path.exists() else []

    by_sample, n_samples = _presence_index(events)
    params_idx = _index_params_by_eid(params)

    knobs = _NecessityKnobs(
        disable_higher_order=bool(getattr(getattr(cfg.surrogate, "necessity", None), "disable_higher_order", True)),
        retrain_on_subset=bool(getattr(getattr(cfg.surrogate, "necessity", None), "retrain_on_subset", False)),
        top_k_per_family=int(getattr(cfg.causal, "n_ensembles_per_family", 10)),
    )
    harness = CausalHarness(ablation_strength=cfg.causal.ablation_strength)

    out_rows: List[Dict[str, Any]] = []

    for fam in fams:
        selected = _select_top_by_cv_for_family(fam, params_idx, ensembles, knobs.top_k_per_family)
        for e in selected:
            eid = str(e.get("id"))
            members = list(e.get("members", []))
            pr = params_idx.get(eid) or {}
            state = pr.get("model_state") or {}
            weights = np.array(state.get("weights") or [0.0] * len(members), dtype=float)
            intercept = float(state.get("intercept", 0.0))
            X = _presence_matrix(members, by_sample, n_samples)

            # Synthetic target (deterministic tiny noise)
            with seed_context(0):
                rng = np.random.default_rng(0)
                y = X.sum(axis=1) + 0.01 + (rng.normal(0.0, 0.001, size=n_samples) if n_samples > 0 else 0.0)

            # Baseline predictions and causal effect
            y_pred = X @ weights + intercept
            nz = [i for i, w in enumerate(weights) if w > 0]
            X_abl = harness.ablate(X, nz) if nz else X
            y_pred_abl = X_abl @ weights + intercept
            mse_base = float(np.mean((y - y_pred) ** 2))
            delta_base = float(np.mean((y - y_pred_abl) ** 2) - mse_base)

            # Variant weights (higher-order disabled)
            w_var = _variant_weights(weights, disable_higher_order=knobs.disable_higher_order, retrain_on_subset=knobs.retrain_on_subset)
            y_pred_var = X @ w_var + intercept
            nz_var = [i for i, w in enumerate(w_var) if w > 0]
            X_abl_var = harness.ablate(X, nz_var) if nz_var else X
            y_pred_abl_var = X_abl_var @ w_var + intercept
            mse_var = float(np.mean((y - y_pred_var) ** 2))
            delta_var = float(np.mean((y - y_pred_abl_var) ** 2) - np.mean((y - y_pred_var) ** 2))

            # Drops
            pred_drop = float(max(0.0, mse_var - mse_base))
            causal_drop = float(max(0.0, delta_base - delta_var))

            # Subset ablation irreducibility (singleton approximation)
            single_deltas: List[float] = []
            for i in range(len(members)):
                Xi = harness.ablate(X, [i])
                yi_pred = Xi @ weights + intercept
                d_i = float(np.mean((y - yi_pred) ** 2) - mse_base)
                single_deltas.append(max(0.0, d_i))
            denom = float(delta_base + 1e-12)
            irr = float(max(0.0, min(1.0, (delta_base - max(single_deltas) if single_deltas else 0.0) / denom)))

            out_rows.append(
                {
                    "family": fam,
                    "ensemble_id": eid,
                    "n_features": int(len(members)),
                    "top1_weight": float(weights[int(np.argmax(weights))]) if len(weights) > 0 else 0.0,
                    "disable_higher_order": bool(knobs.disable_higher_order),
                    "retrain_on_subset": bool(knobs.retrain_on_subset),
                    "baseline_mse": mse_base,
                    "variant_mse": mse_var,
                    "predictive_alignment_drop": pred_drop,
                    "baseline_causal_delta": delta_base,
                    "variant_causal_delta": delta_var,
                    "causal_impact_drop": causal_drop,
                    "irreducibility_score": irr,
                }
            )

    out_path = stage_path(run_dir, "necessity_results.jsonl")
    save_jsonl(out_path, out_rows)
    log_artifact(out_path, kind="necessity_results", metadata={"rows": len(out_rows)})

    # Summary (medians)
    pred_drops = [float(r.get("predictive_alignment_drop", 0.0)) for r in out_rows]
    causal_drops = [float(r.get("causal_impact_drop", 0.0)) for r in out_rows]
    irr_vals = [float(r.get("irreducibility_score", 0.0)) for r in out_rows]

    return {
        "necessity_path": out_path,
        "summary": {
            "n_rows": int(len(out_rows)),
            "predictive_alignment_drop_median": _median(pred_drops),
            "causal_impact_drop_median": _median(causal_drops),
            "irreducibility_median": _median(irr_vals),
        },
    }


__all__ = ["run_higher_order_necessity"]