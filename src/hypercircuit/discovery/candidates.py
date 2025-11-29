from __future__ import annotations

from typing import Iterable, List, Mapping, Tuple


def prune_and_score(
    candidates: Iterable[Mapping[str, object]],
    max_return: int = 100,
) -> List[Mapping[str, object]]:
    """Deduplicate and assign a trivial stability score (mock)."""
    seen = set()
    out: List[Mapping[str, object]] = []
    for c in candidates:
        feats = tuple(sorted(c["features"]))  # type: ignore[index]
        if feats in seen:
            continue
        seen.add(feats)
        out.append({**c, "stability": 1.0})
        if len(out) >= max_return:
            break
    return out
