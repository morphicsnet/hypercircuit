from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from hypercircuit.utils.ids import feature_key, member_key


EVENT_SCHEMA_VERSION = "0.2.0"


@dataclass
class EventContext:
    """Shared context for canonical event records."""

    run_id: str
    source_kind: str
    feature_space_id: str
    feature_space_version: str = "v0"
    schema_version: str = EVENT_SCHEMA_VERSION

    # Feature identity lineage
    dictionary_id: Optional[str] = None
    dictionary_version: Optional[str] = None
    dictionary_type: Optional[str] = None
    feature_origin_layer: Optional[int] = None

    # Optional tags / metadata
    run_intent: Optional[str] = None
    task_family: Optional[str] = None
    prompt_family: Optional[str] = None
    split: Optional[str] = None
    label: Optional[str] = None
    capability_tag: Optional[str] = None
    safety_tag: Optional[str] = None


def build_event(
    *,
    ctx: EventContext,
    sample_id: int,
    sequence_id: Optional[int],
    token_index: int,
    layer: int,
    node_type: str,
    node_id: int,
    value: Optional[float] = None,
    step_index: Optional[int] = None,
    timestamp: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a canonical event record from context + local fields."""
    ev: Dict[str, Any] = {
        "schema_version": ctx.schema_version,
        "run_id": ctx.run_id,
        "source_kind": ctx.source_kind,
        "feature_space_id": ctx.feature_space_id,
        "feature_space_version": ctx.feature_space_version,
        "sample_id": int(sample_id),
        "token_index": int(token_index),
        "layer": int(layer),
        "node_type": str(node_type),
        "node_id": int(node_id),
        "step_index": int(step_index if step_index is not None else token_index),
    }
    if sequence_id is not None:
        ev["sequence_id"] = int(sequence_id)
    if value is not None:
        ev["value"] = float(value)

    # Feature identity lineage
    if ctx.dictionary_id:
        ev["dictionary_id"] = ctx.dictionary_id
    if ctx.dictionary_version:
        ev["dictionary_version"] = ctx.dictionary_version
    if ctx.dictionary_type:
        ev["dictionary_type"] = ctx.dictionary_type
    if ctx.feature_origin_layer is not None:
        ev["feature_origin_layer"] = int(ctx.feature_origin_layer)

    # Optional tags
    if ctx.run_intent:
        ev["run_intent"] = ctx.run_intent
    if ctx.task_family:
        ev["task_family"] = ctx.task_family
    if ctx.prompt_family:
        ev["prompt_family"] = ctx.prompt_family
    if ctx.split:
        ev["split"] = ctx.split
    if ctx.label:
        ev["label"] = ctx.label
    if ctx.capability_tag:
        ev["capability_tag"] = ctx.capability_tag
    if ctx.safety_tag:
        ev["safety_tag"] = ctx.safety_tag

    # Canonical identity keys
    ev["member_key"] = member_key(
        feature_space_id=ctx.feature_space_id,
        layer=int(layer),
        node_type=str(node_type),
        node_id=int(node_id),
    )
    ev["feature_key"] = feature_key(
        feature_space_id=ctx.feature_space_id,
        feature_space_version=ctx.feature_space_version,
        layer=int(layer),
        node_type=str(node_type),
        node_id=int(node_id),
        dictionary_id=ctx.dictionary_id,
        dictionary_version=ctx.dictionary_version,
    )

    if extra:
        ev.update(dict(extra))
    if timestamp is not None:
        ev["timestamp"] = timestamp
    return ev
