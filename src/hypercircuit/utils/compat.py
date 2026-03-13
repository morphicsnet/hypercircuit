from __future__ import annotations

from typing import Any, Mapping, Sequence


def assert_schema_version(
    manifest: Mapping[str, Any],
    *,
    key: str,
    supported: Sequence[str],
) -> None:
    ver = str(manifest.get(key, ""))
    if ver not in supported:
        raise ValueError(f"Unsupported {key}: {ver}. Supported: {list(supported)}")


def assert_feature_space_compat(
    manifest: Mapping[str, Any],
    *,
    expected_id: str | None = None,
    expected_version: str | None = None,
) -> None:
    fs = manifest.get("feature_space") or {}
    if expected_id and fs.get("feature_space_id") and fs.get("feature_space_id") != expected_id:
        raise ValueError(f"feature_space_id mismatch: {fs.get('feature_space_id')} != {expected_id}")
    if expected_version and fs.get("feature_space_version") and fs.get("feature_space_version") != expected_version:
        raise ValueError(
            f"feature_space_version mismatch: {fs.get('feature_space_version')} != {expected_version}"
        )


def assert_candidate_manifest(
    manifest: Mapping[str, Any],
    *,
    required_scores: Sequence[str] | None = None,
    allowed_types: Sequence[str] | None = None,
    expected_granularity: str | None = None,
) -> None:
    ctype = str(manifest.get("candidate_type") or "")
    if allowed_types and ctype and ctype not in set(allowed_types):
        raise ValueError(f"candidate_type '{ctype}' not in allowed {list(allowed_types)}")

    if expected_granularity:
        mg = str(manifest.get("member_granularity") or "")
        if mg and mg != expected_granularity:
            raise ValueError(f"member_granularity mismatch: {mg} != {expected_granularity}")

    if required_scores:
        scores = list(manifest.get("scores") or [])
        missing = [s for s in required_scores if s not in scores]
        if missing:
            raise ValueError(f"candidate manifest missing required scores: {missing}")
