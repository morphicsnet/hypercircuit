from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from hypercircuit.causal.harness import CausalHarness
from hypercircuit.utils.config import Config, stage_path
from hypercircuit.utils.io import load_jsonl, save_jsonl
from hypercircuit.utils.registry import log_artifact
from hypercircuit.utils.seed import seed_context, expand_seeds


@dataclass(frozen=True)
class MatrixKnobs:
    n_prompts: int = 500
    n_seeds: int = 5
    ci_method: str = "bootstrap"
    alpha: float = 0.05
    fdr: float = 0.10


def _resolve_families(cfg: Config, families_cli: Optional[Sequence[str]] = None) -> List[str]:
    if families_cli:
        return list(families_cli)
    mx = getattr(cfg, "matrix", None)
    if mx and getattr(mx, "families", None):
        return list(mx.families or [])
    dic = getattr(cfg, "dictionary", None)
    if dic and getattr(dic, "families", None):
        return list(dic.families or [])
    wk2 = getattr(cfg.discovery, "week2_screening", None)
    if wk2 and getattr(wk2, "top_families", None):
        return list(wk2.top_families or [])
    # fallback single family from dataset
    fam = cfg.run.task_family or cfg.dataset.task_family or cfg.dataset.name
    return [fam]


def _resolve_methods(cfg: Config, methods_cli: Optional[Sequence[str]] = None) -> List[str]:
    if methods_cli:
        return list(methods_cli)
    mx = getattr(cfg, "matrix", None)
    if mx and getattr(mx, "methods", None):
        return list(mx.methods)
    return [
        "pairwise_baseline",
        "no_ensembles_patching",
        "auto_circuit_no_hyper",
        "std_steering",
        "hypercircuit_full",
    ]


def _stable_parity(family: str, method: str) -> int:
    h = hashlib.md5(f"{family}|{method}".encode("utf-8")).hexdigest()
    return int(h[-1], 16) % 2


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


def _index_params_by_eid(params: Iterable[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    idx: Dict[str, Mapping[str, Any]] = {}
    for r in params:
        eid = r.get("ensemble_id")
        if eid is not None:
            idx[str(eid)] = r
    return idx


def _select_ensembles_for_family(
    fam: str,
    ensembles: Sequence[Mapping[str, Any]],
    params_idx: Mapping[str, Mapping[str, Any]],
    method: str,
    top_k: int,
) -> List[Mapping[str, Any]]:
    fam_ens = [e for e in ensembles if str(e.get("family")) == fam]
    # Sort by surrogate CV score desc when available
    def _score(e: Mapping[str, Any]) -> float:
        pid = str(e.get("id"))
        pr = params_idx.get(pid)
        return float(pr.get("cv_score", 0.0)) if pr else 0.0

    if method == "pairwise_baseline":
        fam_ens = [e for e in fam_ens if int(e.get("size", 0)) == 2]
        fam_ens.sort(key=lambda e: -_score(e))
        return fam_ens[:top_k]
    # other methods: general top-k per family
    fam_ens.sort(key=lambda e: -_score(e))
    return fam_ens[:top_k]


def _bootstrap_ci(effects: List[float], alpha: float, seed: int) -> Tuple[float, float]:
    if not effects:
        return 0.0, 0.0
    B = max(20, 5 * len(effects))  # tiny but deterministic
    with seed_context(seed):
        rng = np.random.default_rng(seed)
        boots: List[float] = []
        n = len(effects)
        arr = np.array(effects, dtype=float)
        for _ in range(B):
            idx = rng.integers(0, n, size=n)
            boots.append(float(np.mean(arr[idx])))
    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return lo, hi


def _perm_p_value(effects: List[float], seed: int) -> float:
    if not effects:
        return 1.0
    with seed_context(seed):
        rng = np.random.default_rng(seed)
        T_obs = float(np.mean(effects))
        B = max(50, 10 * len(effects))
        ge = 0
        for _ in range(B):
            signs = rng.choice([-1.0, 1.0], size=len(effects))
            T_perm = float(np.mean(signs * np.array(effects)))
            if T_perm >= T_obs:
                ge += 1
        return float(ge / B)


def _checksum_for_cell(fam: str, method: str, ensemble_ids: Sequence[str], n_seeds: int, n_prompts: int) -> str:
    payload = f"{fam}|{method}|{','.join(sorted(ensemble_ids))}|{n_seeds}|{n_prompts}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def run_matrix_evaluation(
    *,
    cfg: Config,
    run_dir: Path,
    families: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
    half_matrix: Optional[bool] = None,
    remaining_only: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Evaluate a deterministic matrix over (families x methods).

    Modes:
      - Default: half-matrix parity=0 to produce matrix_results.jsonl
      - remaining_only=True: half-matrix parity=1 to produce remaining_matrix_results.jsonl

    Returns a summary dict with coverage and top-line method medians.
    """
    fams = _resolve_families(cfg, families)
    meths = _resolve_methods(cfg, methods)

    mx = getattr(cfg, "matrix", None)
    half = bool(half_matrix if half_matrix is not None else (getattr(mx, "half_matrix", True) if mx else True))
    per = getattr(mx, "per_cell", None)
    knobs = MatrixKnobs(
        n_prompts=int(getattr(per, "n_prompts", 500) if per else 500),
        n_seeds=int(getattr(per, "n_seeds", 5) if per else 5),
        ci_method=str(getattr(per, "ci_method", "bootstrap") if per else "bootstrap"),
        alpha=float(getattr(per, "alpha", 0.05) if per else 0.05),
        fdr=float(getattr(per, "fdr", 0.10) if per else 0.10),
    )

    # Determine which parity to evaluate when half-matrix is on
    target_parity = 0
    if half:
        target_parity = 1 if bool(remaining_only) else 0

    # Load artifacts
    events_path = stage_path(run_dir, "logs.jsonl")
    ensembles_path = stage_path(run_dir, "ensembles.jsonl")
    params_path = stage_path(run_dir, "surrogates_params.jsonl")
    events = load_jsonl(events_path) if events_path.exists() else []
    ensembles = load_jsonl(ensembles_path) if ensembles_path.exists() else []
    params = load_jsonl(params_path) if params_path.exists() else []

    params_idx = _index_params_by_eid(params)
    by_sample, n_samples = _presence_index(events)

    harness = CausalHarness(ablation_strength=cfg.causal.ablation_strength)
    top_k = int(getattr(cfg.causal, "n_ensembles_per_family", 10))

    # Build grid
    grid: List[Tuple[str, str]] = []
    for fam in fams:
        for m in meths:
            if half:
                if _stable_parity(fam, m) != target_parity:
                    continue
            grid.append((fam, m))

    # Evaluate cells
    seeds_all = expand_seeds(cfg.run.seeds)[: max(1, knobs.n_seeds)]
    results: List[Dict[str, Any]] = []

    def _ensemble_delta(
        eid: str, members: List[str], method: str, ablation_strength_scale: float, seed: int
    ) -> float:
        pr = params_idx.get(eid) or {}
        state = pr.get("model_state") or {}
        weights = np.array(state.get("weights") or [0.0] * len(members), dtype=float)
        intercept = float(state.get("intercept", 0.0))
        X = _presence_matrix(members, by_sample, n_samples)
        # synthetic target
        with seed_context(seed):
            rng = np.random.default_rng(seed)
            y = X.sum(axis=1) + 0.01 + (rng.normal(0.0, 0.001, size=n_samples) if n_samples > 0 else 0.0)
        y_pred = X @ weights + intercept
        nz = [i for i, w in enumerate(weights) if w > 0]
        if method == "auto_circuit_no_hyper":
            # only strongest direction
            if len(nz) > 1:
                top = int(np.argmax(weights))
                nz = [top] if top in nz else (nz[:1] if nz else [])
        if method == "std_steering":
            # mock steering gain proportional to active directions (no ablation)
            return float(0.02 * max(0, len(nz)))
        scale = max(0.0, float(ablation_strength_scale))
        if scale != 1.0:
            h = CausalHarness(ablation_strength=harness.ablation_strength * scale)
        else:
            h = harness
        X_abl = h.ablate(X, nz) if nz else X
        y_pred_abl = X_abl @ weights + intercept
        delta = float(np.mean((y - y_pred_abl) ** 2) - np.mean((y - y_pred) ** 2))
        return delta

    for fam, method in grid:
        selected = _select_ensembles_for_family(fam, ensembles, params_idx, method, top_k)
        if not selected:
            # skip empty cells
            continue
        ens_ids = [str(e.get("id")) for e in selected]
        # determine ablation scale per method
        if method == "no_ensembles_patching":
            scale = 0.5
        else:
            scale = 1.0

        per_seed_effects: List[float] = []
        for si, seed in enumerate(seeds_all):
            deltas: List[float] = []
            for e in selected:
                eid = str(e.get("id"))
                members = list(e.get("members", []))
                d = _ensemble_delta(eid, members, method, scale, seed)
                # Make pairwise_baseline more conservative if not exactly size-2
                if method == "pairwise_baseline":
                    if int(e.get("size", len(members))) != 2:
                        d *= 0.5
                deltas.append(d)
            # aggregate across ensembles
            per_seed_effects.append(float(np.mean(deltas) if deltas else 0.0))

        effect = float(np.median(per_seed_effects) if per_seed_effects else 0.0)
        ci_lo, ci_hi = _bootstrap_ci(per_seed_effects, knobs.alpha, seed=seeds_all[0] if seeds_all else 0)
        p_val = _perm_p_value(per_seed_effects, seed=seeds_all[0] if seeds_all else 0)
        # stability mock: 1 - std
        stab = float(max(0.0, 1.0 - float(np.std(per_seed_effects)) if per_seed_effects else 0.0))
        passed = bool((effect > 0.0) and (p_val < knobs.alpha))
        checksum = _checksum_for_cell(fam, method, ens_ids, n_seeds=len(seeds_all), n_prompts=knobs.n_prompts)

        results.append(
            {
                "family": fam,
                "method": method,
                "n_prompts": int(knobs.n_prompts),
                "n_seeds": int(len(seeds_all)),
                "effect_size": effect,
                "ci_low": float(ci_lo),
                "ci_high": float(ci_hi),
                "p_value": float(p_val),
                "stability": float(stab),
                "passed": bool(passed),
                "checksum_id": checksum,
            }
        )

    # Write results
    out_name = "matrix_results.jsonl"
    kind = "matrix_results"
    if half and bool(remaining_only):
        out_name = "remaining_matrix_results.jsonl"
        kind = "remaining_matrix_results"
    out_path = stage_path(run_dir, out_name)
    save_jsonl(out_path, results)
    log_artifact(out_path, kind=kind, metadata={"cells": len(results)})

    # Aggregate summary
    configured_total = len(fams) * len(meths)
    expected_cells = (configured_total + 1) // 2 if half else configured_total
    n_cells = len(results)
    coverage_ratio = float(0.0 if expected_cells == 0 else n_cells / expected_cells)

    # median effect for hypercircuit_full
    hc_effects = [float(r.get("effect_size", 0.0)) for r in results if r.get("method") == "hypercircuit_full"]
    med_hc = float(np.median(hc_effects) if hc_effects else 0.0)

    # improvement rate: families where HC beats all baselines
    fam_set = sorted(set([str(r.get("family")) for r in results]))
    improved = 0
    for fam in fam_set:
        fam_rows = [r for r in results if r.get("family") == fam]
        if not fam_rows:
            continue
        hc_vals = [float(r.get("effect_size", 0.0)) for r in fam_rows if r.get("method") == "hypercircuit_full"]
        base_vals = [float(r.get("effect_size", 0.0)) for r in fam_rows if r.get("method") != "hypercircuit_full"]
        if hc_vals and base_vals and (np.median(hc_vals) > np.median(base_vals)):
            improved += 1
    pass_rate_improvement = float(0.0 if not fam_set else improved / len(fam_set))

    return {
        "matrix_path": out_path,
        "n_cells_evaluated": n_cells,
        "coverage_ratio": coverage_ratio,
        "median_effect_hypercircuit": med_hc,
        "pass_rate_improvement": pass_rate_improvement,
        "fdr_level": knobs.fdr,
    }


__all__ = ["run_matrix_evaluation"]