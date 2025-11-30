from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import hashlib
import math

import numpy as np

from hypercircuit.utils.config import Config
from hypercircuit.utils.io import load_jsonl
from hypercircuit.utils.seed import active_seed, seed_context


@dataclass(frozen=True)
class EditPlan:
    families: List[str]
    proposed_scales: Dict[str, float]
    ranking: List[str]
    seed: int


def _family_hash_score(fam: str, seed: int) -> float:
    """Stable [0,1] score from family name and seed for deterministic fallback."""
    h = hashlib.md5(f"{fam}|{seed}".encode("utf-8")).hexdigest()
    # take first 8 hex chars as int
    v = int(h[:8], 16)
    return (v % 10_000_000) / 10_000_000.0


def _sensitivity_from_surrogates(run_dir: Path) -> Dict[str, float]:
    """
    Derive a per-family sensitivity proxy from surrogates_params.jsonl if available.
    Uses average normalized L1 of nonnegative weights across records per family.
    """
    path = run_dir / "surrogates_params.jsonl"
    if not path.exists():
        return {}
    try:
        rows = load_jsonl(path)
    except Exception:
        return {}
    acc: Dict[str, List[float]] = {}
    for r in rows:
        fam = str(r.get("family")) if r.get("family") is not None else None
        if not fam:
            continue
        state = r.get("model_state") or {}
        weights = state.get("weights")
        if not isinstance(weights, (list, tuple)):
            continue
        w = np.array(weights, dtype=float)
        w = np.clip(w, 0.0, None)
        if w.size == 0:
            continue
        norm = float(w.max() + 1e-8)
        s = float(np.sum(w) / (norm * max(1, w.size)))
        acc.setdefault(fam, []).append(max(0.0, min(1.0, s)))
    out: Dict[str, float] = {}
    for fam, vals in acc.items():
        out[fam] = float(np.median(np.array(vals, dtype=float))) if vals else 0.0
    return out


def propose_edit_plan(*, cfg: Config, run_dir: Path, families: Optional[List[str]] = None) -> EditPlan:
    """
    Select risky-family ensembles and assign initial scales based on surrogate sensitivities,
    falling back to deterministic hash-based scores. Ranking is deterministic given the active seed.
    """
    fams = list(families or cfg.editing.families or [])
    fams = sorted(set(fams))
    if not fams:
        # final fallback to a small default set to keep pipeline alive in mocks
        fams = ["sycophancy", "jailbreak"]

    seed = int(active_seed(cfg.run.seeds, stage_idx=4))
    sens_from_params = _sensitivity_from_surrogates(run_dir)

    scores: Dict[str, float] = {}
    for fam in fams:
        if fam in sens_from_params:
            scores[fam] = float(sens_from_params[fam])
        else:
            scores[fam] = _family_hash_score(fam, seed)

    # normalize scores to [0,1]
    if scores:
        vals = list(scores.values())
        lo, hi = min(vals), max(vals)
        rng = hi - lo
        if rng <= 1e-12:
            scores = {k: 0.5 for k in scores.keys()}
        else:
            scores = {k: (v - lo) / rng for k, v in scores.items()}

    max_edit = float(getattr(cfg.editing, "max_edit_scale", 0.3))
    # proposed base scales proportional to sensitivity, with a gentle floor for exploration
    proposed = {fam: float(min(max_edit, max(0.05 * max_edit, (0.5 + 0.5 * scores[fam]) * max_edit))) for fam in fams}

    # deterministic ranking: score desc, name asc
    ranking = sorted(fams, key=lambda f: (-scores[f], f))

    return EditPlan(families=fams, proposed_scales=proposed, ranking=ranking, seed=seed)


def apply_edit_schedule(
    *,
    proposed_scales: Mapping[str, float],
    step_schedule: List[float],
    max_edit_scale: float,
) -> Dict[str, List[float]]:
    """
    Stage scales per step schedule (assumed non-decreasing), honoring max_edit_scale and monotonic constraints.
    For each family, the applied step scale is min(step_value, proposed_scale, max_edit_scale).
    """
    # ensure schedule is non-decreasing
    sched: List[float] = []
    cur = 0.0
    for s in step_schedule:
        s2 = float(max(cur, s))
        sched.append(s2)
        cur = s2

    applied: Dict[str, List[float]] = {}
    for fam, base in proposed_scales.items():
        b = float(min(max_edit_scale, max(0.0, base)))
        applied[fam] = [float(min(b, v, max_edit_scale)) for v in sched]
        # reinforce monotonic nondecreasing in case of FP artifacts
        for i in range(1, len(applied[fam])):
            if applied[fam][i] < applied[fam][i - 1]:
                applied[fam][i] = applied[fam][i - 1]
    return applied


def rollback_point(run_dir: Path) -> Dict[str, Any]:
    """
    Produce checkpoint metadata to revert to pre-edit state.
    This is a light-weight manifest of current artifacts with sizes and mtimes.
    """
    entries: List[Dict[str, Any]] = []
    for p in sorted(run_dir.glob("*")):
        if p.is_file() and p.name not in {"artifacts.jsonl"}:
            try:
                st = p.stat()
                entries.append(
                    {
                        "path": p.name,
                        "bytes": int(st.st_size),
                        "mtime": float(st.st_mtime),
                    }
                )
            except Exception:
                # best-effort
                pass
    return {
        "checkpoint_kind": "pre_edit",
        "artifact_snapshot": entries,
    }