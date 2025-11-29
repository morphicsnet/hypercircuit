from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, Tuple, Set


Member = str
Key = Tuple[int, int, int]  # (sample_id, layer, window_index)


def _member_from_event(ev: Mapping[str, object]) -> Member:
    """Normalize an event into a member string (coarse in mock: by node_type only)."""
    if "node_type" in ev:
        return f"{ev['node_type']}"
    # Back-compat: events with 'active' feature indices will be handled elsewhere
    raise KeyError("Event missing node_type")


def _build_transactions_from_logging(events: Iterable[Mapping[str, object]], temporal_span: int) -> Dict[Key, Set[Member]]:
    """Group logging events into transactions keyed by (sample, layer, window_index)."""
    tx: Dict[Key, Set[Member]] = defaultdict(set)
    for ev in events:
        sid = int(ev.get("sample_id", 0))  # type: ignore[arg-type]
        layer = int(ev.get("layer", 0))  # type: ignore[arg-type]
        tok = int(ev.get("token_index", 0))  # type: ignore[arg-type]
        w = tok // max(1, temporal_span)
        try:
            m = _member_from_event(ev)
        except KeyError:
            # If this is an 'active' style record, skip here (handled by alt path)
            continue
        tx[(sid, layer, w)].add(m)
    return tx


def _build_transactions_from_active(events: Iterable[Mapping[str, object]], temporal_span: int) -> Dict[Key, Set[Member]]:
    """Back-compat builder: per-sample active list into a single-window transaction."""
    tx: Dict[Key, Set[Member]] = defaultdict(set)
    for ev in events:
        sid = int(ev.get("sample_id", 0))  # type: ignore[arg-type]
        act = ev.get("active", [])
        if isinstance(act, (list, tuple)):
            for a in act:
                tx[(sid, 0, 0)].add(f"feat:{int(a)}")
    return tx


def _ensure_transactions(events: Iterable[Mapping[str, object]], temporal_span: int) -> Dict[Key, Set[Member]]:
    # Peek first item to decide format
    events = list(events)
    for ev in events:
        if "active" in ev and "node_id" not in ev:
            return _build_transactions_from_active(events, temporal_span)
    return _build_transactions_from_logging(events, temporal_span)


def _mine_supports(
    tx: Dict[Key, Set[Member]],
    max_set_size: int,
) -> Tuple[Dict[frozenset[Member], int], Dict[frozenset[Member], Tuple[int, int]]]:
    """Count raw supports across transactions and track (min_w, max_w) per set."""
    support: Dict[frozenset[Member], int] = defaultdict(int)
    window_bounds: Dict[frozenset[Member], Tuple[int, int]] = {}

    for (sid, layer, w), members in tx.items():
        if not members:
            continue
        sorted_members = sorted(members)
        # Singles
        for m in sorted_members:
            k = frozenset([m])
            support[k] += 1
            lo, hi = window_bounds.get(k, (w, w))
            window_bounds[k] = (min(lo, w), max(hi, w))
        # Pairs / Triples
        for ksize in (2, 3):
            if ksize > max_set_size:
                break
            for comb in combinations(sorted_members, ksize):
                k = frozenset(comb)
                support[k] += 1
                lo, hi = window_bounds.get(k, (w, w))
                window_bounds[k] = (min(lo, w), max(hi, w))
    return support, window_bounds


def _to_weighted_support(
    raw_support: Dict[frozenset[Member], int],
    n_tx: int,
) -> Dict[frozenset[Member], float]:
    return {k: (v / max(1, n_tx)) for k, v in raw_support.items()}


def _jaccard(a: Set[Member], b: Set[Member]) -> float:
    inter = len(a & b)
    union = len(a | b)
    return 0.0 if union == 0 else inter / union


def mine_rank_weighted_coactivations(
    events: Iterable[Mapping[str, object]],
    *,
    min_weighted_support: float,
    candidate_caps: Mapping[str, int],
    temporal_span: int,
    dedup_jaccard_min: float = 0.5,
    max_set_size: int = 3,
) -> Tuple[List[Dict[str, object]], Dict[frozenset[Member], float], Dict[str, Dict[frozenset[Member], float]]]:
    """Rank-weighted (uniform weights in mock) frequent-set mining for size 2–3.

    Returns:
        (candidates, support_index, replicate_support_indices)
    """
    # Build transactions
    tx = _ensure_transactions(list(events), temporal_span)
    n_tx = len(tx)

    # Global supports
    raw_support, window_bounds = _mine_supports(tx, max_set_size=max_set_size)
    ws = _to_weighted_support(raw_support, n_tx=n_tx)

    # Replicates for stability (even/odd sample_id)
    tx_a: Dict[Key, Set[Member]] = {}
    tx_b: Dict[Key, Set[Member]] = {}
    for (sid, layer, w), members in tx.items():
        if sid % 2 == 0:
            tx_a[(sid, layer, w)] = members
        else:
            tx_b[(sid, layer, w)] = members
    ws_a = _to_weighted_support(_mine_supports(tx_a, max_set_size=max_set_size)[0], max(1, len(tx_a)))
    ws_b = _to_weighted_support(_mine_supports(tx_b, max_set_size=max_set_size)[0], max(1, len(tx_b)))

    # Collect candidates of size 2 and 3 with filters
    sizes = {2: candidate_caps.get("size2", 1000), 3: candidate_caps.get("size3", 300)}
    cand_by_size: Dict[int, List[Tuple[frozenset[Member], float, int]]] = {2: [], 3: []}
    for key, w_support in ws.items():
        ksize = len(key)
        if ksize not in (2, 3):
            continue
        if w_support < min_weighted_support:
            continue
        lo, hi = window_bounds.get(key, (0, 0))
        span = (hi - lo + 1)
        if span > max(1, temporal_span):
            continue
        cand_by_size[ksize].append((key, w_support, span))

    # Sort by weighted support desc, lex members as tiebreaker
    for s in cand_by_size:
        cand_by_size[s].sort(key=lambda t: (-t[1], tuple(sorted(t[0]))))

    # Deduplicate by Jaccard similarity and apply caps per size
    accepted: List[Dict[str, object]] = []
    seen_sets: List[Set[Member]] = []
    for s in (2, 3):
        cap = sizes.get(s, 0)
        taken = 0
        for key, w_support, span in cand_by_size[s]:
            members_set = set(key)
            if any(_jaccard(members_set, prev) >= dedup_jaccard_min for prev in seen_sets):
                continue
            accepted.append(
                {
                    "members": sorted(members_set),
                    "size": s,
                    "support": int(round(ws.get(key, 0.0) * n_tx)),
                    "weighted_support": float(w_support),
                    "window_span": int(span),
                }
            )
            seen_sets.append(members_set)
            taken += 1
            if taken >= cap:
                break

    # Final stable sort for determinism
    accepted.sort(key=lambda d: (-float(d["weighted_support"]), d["size"], tuple(d["members"])))  # type: ignore[index]

    replicate_ws = {"A": ws_a, "B": ws_b}
    return accepted, ws, replicate_ws


def mine_coactivations(
    events: Iterable[Mapping[str, object]],
    min_support: int,
    max_set_size: int,
    candidate_cap: int,
) -> List[Dict[str, object]]:
    """Back-compat shim: convert to rank-weighted mining and return top-K across sizes."""
    # Use conservative defaults for Week 1 shim
    candidate_caps = {"size2": candidate_cap, "size3": max(0, candidate_cap // 2)}
    cands, _, _ = mine_rank_weighted_coactivations(
        events,
        min_weighted_support=0.0,
        candidate_caps=candidate_caps,
        temporal_span=3,
        dedup_jaccard_min=0.5,
        max_set_size=max_set_size,
    )
    # Truncate to overall cap for compatibility
    return cands[:candidate_cap]
