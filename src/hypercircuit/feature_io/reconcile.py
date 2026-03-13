from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from hypercircuit.utils.io import load_jsonl, save_jsonl, write_json


def reconcile_feature_space(
    *,
    events_path: Path,
    out_dir: Path,
    method: str = "identity",
    out_events: str | None = "events_reconciled.jsonl",
) -> Dict[str, Path]:
    """
    Minimal stable-feature reconciliation stage.

    For method="identity", stable_node_id == member_key and consensus is a direct passthrough.
    """
    if method not in {"identity"}:
        raise NotImplementedError(f"Reconciliation method '{method}' is not implemented.")

    rows = load_jsonl(events_path) if events_path.exists() else []
    mappings: Dict[str, Dict[str, Any]] = {}
    for ev in rows:
        member_key = ev.get("member_key")
        feature_key = ev.get("feature_key")
        if not member_key:
            continue
        mk = str(member_key)
        if mk not in mappings:
            mappings[mk] = {
                "member_key": mk,
                "feature_key": feature_key,
                "stable_node_id": mk,
                "feature_role": "raw_member",
            }

    alignment = list(mappings.values())
    consensus = {
        "method": method,
        "features": [m["stable_node_id"] for m in alignment],
        "count": len(alignment),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    alignment_path = out_dir / "feature_alignment.json"
    consensus_path = out_dir / "consensus_features.json"
    write_json(alignment_path, {"method": method, "alignment": alignment})
    write_json(consensus_path, consensus)

    result: Dict[str, Path] = {"alignment": alignment_path, "consensus": consensus_path}
    if out_events:
        out_events_path = out_dir / out_events
        reconciled_rows = []
        for ev in rows:
            rec = dict(ev)
            mk = rec.get("member_key")
            if mk is not None:
                mapping = mappings.get(str(mk))
                if mapping:
                    rec["stable_node_id"] = mapping.get("stable_node_id")
                    rec.setdefault("feature_role", mapping.get("feature_role", "raw_member"))
            reconciled_rows.append(rec)
        save_jsonl(out_events_path, reconciled_rows)
        result["events"] = out_events_path
    return result
