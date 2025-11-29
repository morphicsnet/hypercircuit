from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from hypercircuit.surrogate.model import MonotoneCombiner
from hypercircuit.utils.config import Config, stage_path
from hypercircuit.utils.io import load_jsonl, save_jsonl, write_json
from hypercircuit.utils.seed import active_seed, seed_context


def _median(vals: List[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    m = n // 2
    if n % 2 == 1:
        return float(s[m])
    return float(0.5 * (s[m - 1] + s[m]))


def _corrcoef_safe(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    va = float(np.var(a))
    vb = float(np.var(b))
    if va == 0.0 or vb == 0.0:
        return 0.0
    try:
        r = float(np.corrcoef(a, b)[0, 1])
        # map [-1,1] -> [0,1] for consistency with other mock metrics
        r01 = (r + 1.0) / 2.0
        return float(max(0.0, min(1.0, r01)))
    except Exception:
        return 0.0


def _ece_like(y_true: np.ndarray, y_pred: np.ndarray, bins: int = 10) -> Tuple[float, Dict[str, Any]]:
    """Mock ECE: average |E[y|bin] - E[ŷ|bin]| weighted by bin frequency.

    Also returns a simple calibration mapping state with bin edges and targets.
    """
    if y_true.size == 0 or y_pred.size == 0:
        return 0.0, {"bins": None, "targets": None}
    # Guard constant predictions
    if float(np.std(y_pred)) == 0.0:
        t = float(abs(np.mean(y_true) - np.mean(y_pred)))
        return t, {"bins": None, "targets": None}
    # Quantile bins
    qs = np.linspace(0.0, 1.0, num=bins + 1)
    try:
        edges = np.quantile(y_pred, qs)
    except Exception:
        edges = np.linspace(float(np.min(y_pred)), float(np.max(y_pred)), num=bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf
    err = 0.0
    targets: List[float] = []
    counts: List[int] = []
    means_pred: List[float] = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_pred >= lo) & (y_pred < hi)
        if not np.any(mask):
            targets.append(0.0)
            counts.append(0)
            means_pred.append(0.0)
            continue
        yt = float(np.mean(y_true[mask]))
        yp = float(np.mean(y_pred[mask]))
        w = int(np.sum(mask))
        err += float(abs(yt - yp)) * (w / max(1, y_pred.size))
        targets.append(yt)
        counts.append(w)
        means_pred.append(yp)
    state = {
        "bins": [float(x) if np.isfinite(x) else (-1e30 if i == 0 else 1e30) for i, x in enumerate(edges.tolist())],
        "targets": targets,
        "counts": counts,
        "means_pred": means_pred,
    }
    return float(err), state


def assemble_mock_training_data(
    *,
    run_dir: Path,
    ensembles: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    seed: Optional[int] = 0,
) -> Dict[str, Dict[str, Any]]:
    """
    Build per-ensemble training data from logging events (mock).

    For each ensemble:
      - Features: one column per member; X[sample, j]=1 if member event present for that sample.
      - Target: y = sum_j X[:, j] + small seed-bound noise (deterministic via seed_context()).
    """
    # Precompute per-sample presence sets by node_type
    by_sample: Dict[int, set] = {}
    sample_ids: set[int] = set()
    for ev in events:
        s = int(ev.get("sample_id", 0))
        sample_ids.add(s)
        # Prefer structured logging events with node_type
        nt = ev.get("node_type")
        if nt is not None:
            by_sample.setdefault(s, set()).add(str(nt))
        # Back-compat: ignore 'active' legacy format here (dictionary members are node_type strings)

    n_samples = (max(sample_ids) + 1) if sample_ids else 0

    data: Dict[str, Dict[str, Any]] = {}
    with seed_context(seed):
        rng = np.random.default_rng(seed if seed is not None else 0)
        for e in ensembles:
            eid = str(e.get("id"))
            members: List[str] = [str(m) for m in e.get("members", [])]
            if not members or not eid:
                continue
            X = np.zeros((n_samples, len(members)), dtype=float)
            for i in range(n_samples):
                present = by_sample.get(i, set())
                for j, m in enumerate(members):
                    X[i, j] = 1.0 if m in present else 0.0
            # Synthetic target with tiny deterministic noise
            base = X.sum(axis=1).astype(float)
            noise = rng.normal(loc=0.0, scale=0.01, size=n_samples) if n_samples > 0 else np.zeros((0,), dtype=float)
            y = base + noise
            data[eid] = {
                "X": X,
                "y": y,
                "family": e.get("family"),
                "members": members,
                "size": int(e.get("size", len(members))),
            }
    return data


def cross_validate_and_calibrate(
    X: np.ndarray,
    y: np.ndarray,
    *,
    folds: int,
    nonneg: bool = True,
    calibration: str = "isotonic",
    seed: Optional[int] = 0,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    K-fold CV with correlation score and mock calibration error (ECE-like).
    Returns (cv_score_mean, calibration_error, calibration_state).
    """
    n = X.shape[0]
    if n == 0 or folds <= 1:
        # Fit once and compute single score
        model = MonotoneCombiner(nonneg=nonneg).fit(X, y)
        yhat = model.predict(X)
        cv = _corrcoef_safe(y, yhat)
        ece, state = _ece_like(y, yhat)
        return float(cv), float(ece), state

    idx = np.arange(n)
    with seed_context(seed):
        rng = np.random.default_rng(seed if seed is not None else 0)
        rng.shuffle(idx)

    # Split into folds
    chunks: List[np.ndarray] = np.array_split(idx, folds)

    scores: List[float] = []
    y_val_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []

    for f in range(folds):
        val_idx = chunks[f]
        train_idx = np.concatenate([chunks[i] for i in range(folds) if i != f]) if folds > 1 else val_idx
        if train_idx.size == 0 or val_idx.size == 0:
            continue
        m = MonotoneCombiner(nonneg=nonneg).fit(X[train_idx], y[train_idx])
        yhat_val = m.predict(X[val_idx])
        score = _corrcoef_safe(y[val_idx], yhat_val)
        scores.append(score)
        y_val_all.append(y[val_idx].astype(float))
        y_pred_all.append(yhat_val.astype(float))

    if not scores:
        cv = 0.0
        ece_state = {"bins": None, "targets": None}
        return float(cv), 0.0, ece_state

    # Aggregate calibration across out-of-fold predictions
    y_val_cat = np.concatenate(y_val_all, axis=0)
    y_pred_cat = np.concatenate(y_pred_all, axis=0)

    if calibration == "isotonic":
        ece, state = _ece_like(y_val_cat, y_pred_cat, bins=10)
    else:
        # No-op calibration still computes ECE-like for reporting
        ece, state = _ece_like(y_val_cat, y_pred_cat, bins=10)

    return float(np.mean(scores)), float(ece), state


def serialize_surrogate(
    *,
    run_dir: Path,
    records: Sequence[Mapping[str, Any]],
    per_family_summary: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Path]:
    """Write surrogates_params.jsonl and surrogates_index.json artifacts."""
    params_path = stage_path(run_dir, "surrogates_params.jsonl")
    index_path = stage_path(run_dir, "surrogates_index.json")
    save_jsonl(params_path, records)
    write_json(
        index_path,
        {
            "created_at": None,
            "summary": {k: dict(v) for k, v in per_family_summary.items()},
            "total": int(len(records)),
        },
    )
    return {"params": params_path, "index": index_path}


def fit_surrogates_for_family(
    *,
    run_dir: Path,
    cfg: Config,
    families: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Orchestrate surrogate training across a set of families using ensembles.jsonl and logs.jsonl.

    Writes:
      - surrogates_params.jsonl (one record per ensemble)
      - surrogates_index.json (per-family summary stats)
    Returns:
      {
        "paths": {"params": Path, "index": Path},
        "summary": {...}
      }
    """
    # Resolve families to use
    fam_set: Optional[set[str]] = None
    if families:
        fam_set = set(families)
    elif getattr(cfg.discovery.week2_screening, "top_families", None):
        fam_set = set(cfg.discovery.week2_screening.top_families)

    # Load artifacts
    ensembles_path = stage_path(run_dir, "ensembles.jsonl")
    events_path = stage_path(run_dir, "logs.jsonl")
    ensembles = load_jsonl(ensembles_path) if ensembles_path.exists() else []
    events = load_jsonl(events_path) if events_path.exists() else []

    # Filter ensembles by requested families
    if fam_set is not None:
        ensembles = [e for e in ensembles if e.get("family") in fam_set]

    # Build per-ensemble training data (reuse events for all)
    seed_val = active_seed(cfg.run.seeds, stage_idx=0)
    data_by_eid = assemble_mock_training_data(run_dir=run_dir, ensembles=ensembles, events=events, seed=seed_val)

    # Train one surrogate per ensemble with CV + calibration
    recs: List[Dict[str, Any]] = []
    # Family aggregates
    fam_cv: Dict[str, List[float]] = {}
    fam_cal: Dict[str, List[float]] = {}
    for e in ensembles:
        eid = str(e.get("id"))
        fam = str(e.get("family"))
        block = data_by_eid.get(eid)
        if not block:
            continue
        X: np.ndarray = block["X"]  # type: ignore[assignment]
        y: np.ndarray = block["y"]  # type: ignore[assignment]

        cv_score, cal_err, cal_state = cross_validate_and_calibrate(
            X,
            y,
            folds=int(cfg.surrogate.cv_folds),
            nonneg=bool(cfg.surrogate.nonneg),
            calibration=str(getattr(cfg.surrogate, "calibration", "isotonic") or "isotonic"),
            seed=seed_val,
        )

        # Fit final model on all data
        model = MonotoneCombiner(nonneg=bool(cfg.surrogate.nonneg)).fit(X, y)
        recs.append(
            {
                "ensemble_id": eid,
                "family": fam,
                "cv_score": float(cv_score),
                "calibration_error": float(cal_err),
                "model_state": model.state_dict(),
                "hyperparams": {
                    "nonneg": bool(cfg.surrogate.nonneg),
                    "cv_folds": int(cfg.surrogate.cv_folds),
                    "calibration": str(getattr(cfg.surrogate, "calibration", "isotonic") or "isotonic"),
                },
            }
        )
        fam_cv.setdefault(fam, []).append(float(cv_score))
        fam_cal.setdefault(fam, []).append(float(cal_err))

    # Per-family summary
    per_family_summary: Dict[str, Dict[str, Any]] = {}
    for fam in sorted({str(e.get("family")) for e in ensembles}):
        cv_vals = fam_cv.get(fam, [])
        cal_vals = fam_cal.get(fam, [])
        per_family_summary[fam] = {
            "n_surrogates_trained": int(len(cv_vals)),
            "median_cv_score": _median(cv_vals),
            "median_calibration_error": _median(cal_vals),
        }

    # Write artifacts
    paths = serialize_surrogate(run_dir=run_dir, records=recs, per_family_summary=per_family_summary)

    # Orchestrator summary for registry
    summary = {
        "families": {k: dict(v) for k, v in per_family_summary.items()},
        "total_trained": int(len(recs)),
    }
    return {"paths": paths, "summary": summary}


__all__ = [
    "fit_surrogates_for_family",
    "assemble_mock_training_data",
    "cross_validate_and_calibrate",
    "serialize_surrogate",
]