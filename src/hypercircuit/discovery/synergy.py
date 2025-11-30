from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Tuple
from math import isnan

Member = str


def _max_subset_ws(members: List[Member], ws_index: Mapping[frozenset[Member], float]) -> float:
    """Max weighted support over proper subsets."""
    s = set(members)
    if len(s) <= 1:
        return 0.0
    best = 0.0
    # for pairs, subsets are singles; for triples, subsets are pairs
    if len(s) == 2:
        for m in s:
            best = max(best, float(ws_index.get(frozenset([m]), 0.0)))
    elif len(s) == 3:
        ml = list(s)
        pairs = [
            frozenset([ml[0], ml[1]]),
            frozenset([ml[0], ml[2]]),
            frozenset([ml[1], ml[2]]),
        ]
        for p in pairs:
            best = max(best, float(ws_index.get(p, 0.0)))
    return best


def _spearman_rho(series_a: List[float], series_b: List[float]) -> float:
    """Compute Spearman rank correlation (simple implementation)."""
    n = len(series_a)
    if n == 0 or len(series_b) != n:
        return 0.0
    # Ranks (stable)
    def ranks(vals: List[float]) -> List[float]:
        order = sorted(range(n), key=lambda i: (vals[i], i))
        r = [0.0] * n
        cur = 1
        for idx in order:
            r[idx] = float(cur)
            cur += 1
        return r

    ra = ranks(series_a)
    rb = ranks(series_b)
    mean_a = sum(ra) / n
    mean_b = sum(rb) / n
    num = sum((ra[i] - mean_a) * (rb[i] - mean_b) for i in range(n))
    den_a = sum((ra[i] - mean_a) ** 2 for i in range(n)) or 1.0
    den_b = sum((rb[i] - mean_b) ** 2 for i in range(n)) or 1.0
    rho = num / (den_a**0.5 * den_b**0.5)
    try:
        return float(max(0.0, min(1.0, (rho + 1.0) / 2.0)))  # map [-1,1] -> [0,1]
    except Exception:
        return 0.0


def compute_stability_multi(replicate_vectors: List[List[float]]) -> float:
    """
    Compute average pairwise Spearman rank correlation across k replicates.

    Args:
        replicate_vectors: list of length k; each item is a list of scores
            aligned to the same candidate ordering.
    Returns:
        Stability in [0,1], via average of pairwise Spearman rhos mapped from [-1,1].
    """
    k = len(replicate_vectors)
    if k < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(k):
        for j in range(i + 1, k):
            total += _spearman_rho(replicate_vectors[i], replicate_vectors[j])
            pairs += 1
    return float(total / pairs) if pairs else 0.0


def score_candidates(
    candidates: Iterable[Mapping[str, object]],
    ws_index: Mapping[frozenset[Member], float],
    replicate_ws: Mapping[str, Mapping[frozenset[Member], float]],
    *,
    replicates_k: int = 2,
) -> List[Mapping[str, object]]:
    """Annotate candidates with synergy_score, redundancy_flag, stability_score.

    If replicates_k > 2 and replicate_ws provides >= k replicate maps, compute
    a multi-replicate stability score via average pairwise Spearman rho.
    Otherwise, fall back to the two-replicate A/B stability."""
    ann: List[Mapping[str, object]] = []
    # Precompute replicate vectors for stability
    keys: List[frozenset[Member]] = [frozenset(c["members"]) for c in candidates]  # type: ignore[index]

    if replicates_k > 2 and len(replicate_ws) >= 3:
        # Deterministic ordering of replicate names
        rep_names = sorted(replicate_ws.keys())[:replicates_k]
        vectors = [[float(replicate_ws.get(n, {}).get(k, 0.0)) for k in keys] for n in rep_names]
        global_stability = compute_stability_multi(vectors)
    else:
        a_vals = [float(replicate_ws.get("A", {}).get(k, 0.0)) for k in keys]
        b_vals = [float(replicate_ws.get("B", {}).get(k, 0.0)) for k in keys]
        global_stability = _spearman_rho(a_vals, b_vals)

    for c in candidates:
        members: List[Member] = list(c["members"])  # type: ignore[index]
        ws = float(c.get("weighted_support", 0.0))
        subset_best = _max_subset_ws(members, ws_index)
        synergy = max(0.0, ws - subset_best)
        redundancy = ws <= subset_best + 1e-12
        ann.append(
            {
                **c,
                "synergy_score": float(synergy),
                "redundancy_flag": bool(redundancy),
                "stability_score": float(global_stability),
            }
        )
    # Deterministic ordering
    ann.sort(key=lambda d: (-float(d["weighted_support"]), -float(d["synergy_score"]), tuple(d["members"])))  # type: ignore[index]
    return ann


def filter_candidates(
    scored: Iterable[Mapping[str, object]],
    synergy_threshold: float,
    stability_score_min: float,
) -> List[Mapping[str, object]]:
    """Filter candidates by thresholds (mock: do not drop redundancy; dedup occurs later)."""
    out: List[Mapping[str, object]] = []
    for c in scored:
        if float(c.get("synergy_score", 0.0)) < synergy_threshold:  # type: ignore[arg-type]
            continue
        if float(c.get("stability_score", 0.0)) < stability_score_min:  # type: ignore[arg-type]
            continue
        out.append(c)
    out.sort(key=lambda d: (-float(d["weighted_support"]), -float(d["synergy_score"]), tuple(d["members"])))  # type: ignore[index]
    return out
