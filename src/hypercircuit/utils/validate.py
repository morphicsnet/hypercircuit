from __future__ import annotations

from typing import Iterable, Optional

from hypercircuit.logging.schema import EVENT_SCHEMA_VERSION
from hypercircuit.utils.config import Config


ALLOWED_MEMBER_GRANULARITY = {"node_type", "node_id", "group"}
ALLOWED_SOURCE_KIND = {"mock", "hf_local", "api_trace", "posthoc_import", "hardware_capture"}
ALLOWED_RUN_INTENT = {"demo", "dev", "calibration", "benchmark", "beta", "research"}
ALLOWED_DATA_SOURCE = {"mock", "hf", "jsonl"}
ALLOWED_ACTIVATION_KIND = {"residual", "mlp_out", "attn_out", "mlp", "attn"}


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _ensure_in(value: Optional[str], allowed: Iterable[str], label: str) -> None:
    if value is None:
        return
    if str(value) not in set(allowed):
        raise ValueError(f"{label} must be one of {sorted(set(allowed))}; got '{value}'")


def validate_logging_config(cfg: Config) -> None:
    """Hard checks for logging stage readiness."""
    # Schema/version compatibility
    _require(
        str(cfg.logging.schema_version) == EVENT_SCHEMA_VERSION,
        f"logging.schema_version must be {EVENT_SCHEMA_VERSION} (got {cfg.logging.schema_version})",
    )
    _ensure_in(cfg.logging.member_granularity, ALLOWED_MEMBER_GRANULARITY, "logging.member_granularity")
    _ensure_in(cfg.logging.source_kind, ALLOWED_SOURCE_KIND, "logging.source_kind")
    _ensure_in(cfg.run.intent, ALLOWED_RUN_INTENT, "run.intent")

    # Determine mode (respect legacy run.mock)
    mode = str(getattr(cfg.logging, "mode", "mock"))
    if mode == "mock" and not bool(getattr(cfg.run, "mock", True)):
        mode = "real"

    if mode == "real":
        _require(bool(cfg.model.hf_model), "model.hf_model is required for logging.mode=real")
        _ensure_in(cfg.dataset.source, ALLOWED_DATA_SOURCE, "dataset.source")
        _require(cfg.dataset.source in {"hf", "jsonl"}, "dataset.source must be 'hf' or 'jsonl' for real logging")
        if cfg.dataset.source == "hf":
            _require(bool(cfg.dataset.hf_name), "dataset.hf_name is required when dataset.source=hf")
        if cfg.dataset.source == "jsonl":
            _require(bool(cfg.dataset.path), "dataset.path is required when dataset.source=jsonl")
        _require(
            bool(cfg.sae.path) or bool(cfg.sae.layer_map),
            "sae.path or sae.layer_map is required for real logging",
        )
        if cfg.model.activation_kind:
            _ensure_in(cfg.model.activation_kind, ALLOWED_ACTIVATION_KIND, "model.activation_kind")


def validate_config_for_stage(cfg: Config, stage: str) -> None:
    stage_l = str(stage).lower()
    if stage_l in {"logging", "run_log"}:
        validate_logging_config(cfg)
