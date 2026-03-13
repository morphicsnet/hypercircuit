from __future__ import annotations

import hashlib
from typing import Mapping, Optional, Sequence


def _safe_str(val: object, default: str = "unknown") -> str:
    if val is None:
        return default
    s = str(val)
    return s if s else default


def member_key(
    *,
    feature_space_id: str,
    layer: int,
    node_type: str,
    node_id: int | str,
) -> str:
    """Canonical member key: stable across stages."""
    return f"{_safe_str(feature_space_id)}:{int(layer)}:{_safe_str(node_type)}:{_safe_str(node_id)}"


def feature_key(
    *,
    feature_space_id: str,
    feature_space_version: str | None,
    layer: int,
    node_type: str,
    node_id: int | str,
    dictionary_id: str | None = None,
    dictionary_version: str | None = None,
) -> str:
    """Canonical feature key: identity + lineage."""
    parts = [
        _safe_str(feature_space_id),
        _safe_str(feature_space_version, "v0"),
        str(int(layer)),
        _safe_str(node_type),
        _safe_str(node_id),
    ]
    if dictionary_id:
        parts.append(_safe_str(dictionary_id))
    if dictionary_version:
        parts.append(_safe_str(dictionary_version))
    return ":".join(parts)


def candidate_key(members: Sequence[str], candidate_type: str = "coactivation") -> str:
    """Deterministic candidate key from sorted members."""
    payload = f"{candidate_type}|{','.join(sorted(members))}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def member_key_from_event(ev: Mapping[str, object], granularity: str = "node_id") -> str:
    """Build a member key from an event, honoring granularity."""
    gran = str(granularity or "node_id").lower()
    if gran in {"node_type", "type"}:
        if "node_type" not in ev:
            raise KeyError("Event missing node_type")
        return _safe_str(ev.get("node_type"))

    if gran in {"group", "group_id"}:
        if "node_type" not in ev:
            raise KeyError("Event missing node_type")
        node_type = _safe_str(ev.get("node_type"))
        layer = int(ev.get("layer", 0))
        feature_space_id = _safe_str(ev.get("feature_space_id"))
        group = ev.get("node_group") or ev.get("stable_node_id") or ev.get("consensus_node_id")
        if group is not None:
            if isinstance(group, str) and group.count(":") >= 3:
                return group
            return f"{_safe_str(feature_space_id)}:{int(layer)}:{node_type}:{_safe_str(group)}"
        # fallback to instance if group not present

    # default: instance-level
    if "member_key" in ev and ev.get("member_key"):
        return str(ev["member_key"])
    if "node_type" not in ev:
        raise KeyError("Event missing node_type")
    node_type = _safe_str(ev.get("node_type"))
    node_id = ev.get("node_id")
    layer = int(ev.get("layer", 0))
    feature_space_id = _safe_str(ev.get("feature_space_id"))
    return member_key(
        feature_space_id=feature_space_id,
        layer=layer,
        node_type=node_type,
        node_id=_safe_str(node_id),
    )
