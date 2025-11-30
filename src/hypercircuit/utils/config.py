from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

import yaml
from pydantic import BaseModel, Field, ValidationError


# ----------------------
# Schema (Pydantic v2)
# ----------------------

class LoggingThresholds(BaseModel):
    """Threshold and selection knobs for logging/instrumentation."""
    quantile_top_percent: float = 99.5
    zscore_on: float = 3.0
    hysteresis_off_ratio: float = 0.8
    top_k_per_node_type: int = 2


class NodeTypeToggles(BaseModel):
    """Per-node-type enable/disable toggles for logging."""
    sae_features: bool = True
    attn_heads: bool = True
    mlp_channels: bool = True
    residual_dirs: bool = True


class CandidateCaps(BaseModel):
    """Caps for candidate enumeration across sizes and temporal span."""
    size2: int = 1000
    size3: int = 300
    temporal_span: int = 3


class RollbackThresholds(BaseModel):
    """Rollback and specificity thresholds for safety editing."""
    benign_deg_pct: float = 0.5
    specificity_min_ratio: float = 1.2


class RunConfig(BaseModel):
    """Run-level configuration and metadata.

    Attributes:
        output_dir: Base directory where run subdirectories are created.
        run_id: Optional run identifier (auto-generated if None).
        seeds: Per-run seed list for multi-seed policies.
        stage: Optional current stage name (e.g., 'logging','discovery',...)
        model_tag: Optional model identifier tag.
        task_family: Optional task family name.
        split: Optional dataset split (e.g., 'dev', 'test').

    Legacy (back-compat, not used by registry but kept to avoid breaking tests):
        seed: Single seed retained for older paths.
        run_dir: Legacy direct write directory.
        mock: Use mock/deterministic mode.
        timestamped: Legacy timestamp folder hint.
    """
    output_dir: str = "runs"
    run_id: Optional[str] = None
    seeds: List[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    stage: Optional[str] = None
    model_tag: Optional[str] = None
    task_family: Optional[str] = None
    split: Optional[str] = None

    # Legacy/back-compat fields (unused by new registry, but safe to keep)
    seed: int = 0
    run_dir: str = "runs/mock"
    mock: bool = True
    timestamped: bool = True


class LoggingConfig(BaseModel):
    """Activation logging stage configuration."""
    # New instrumentation knobs
    instrumented_layers: List[int] = Field(default_factory=lambda: list(range(-12, 0)))
    token_window: int = 160
    thresholds: LoggingThresholds = LoggingThresholds()
    expected_event_density_range: List[float] = Field(default_factory=lambda: [3.0, 12.0])  # sanity band
    node_types: NodeTypeToggles = NodeTypeToggles()

    # Week 7 expansion toggles
    top_behaviors_only: bool = False
    top_behaviors_families: Optional[List[str]] = None  # None => resolve later from discovery.week2_screening.top_families
    layers_profile_24: Optional[List[int]] = None  # None => default to list(range(-24, 0))

    # Existing mock logger knobs
    tokens_per_sample: int = 8
    threshold: float = 0.5
    hysteresis: float = 0.05
    sparsity: float = 0.1
    storage: str = "jsonl"  # "jsonl" (Parquet stubbed in utils.io)


class Week2ScreeningConfig(BaseModel):
    """Week 2 screening knobs."""
    top_families: List[str] = Field(default_factory=lambda: ["sycophancy", "jailbreak"])
    replicates_k: int = 2
    paraphrase_replicates: int = 1


class DiscoveryConfig(BaseModel):
    """Hyperedge discovery configuration."""
    # Week 1 baselines
    min_weighted_support: float = 0.01
    candidate_caps: CandidateCaps = CandidateCaps()
    synergy_threshold: float = 0.0
    stability_score_min: float = 0.5
    dedup_jaccard_min: float = 0.5

    # Week 2 (optional) aggregated screening
    week2_screening: Week2ScreeningConfig = Week2ScreeningConfig()

    # Back-compat (used by current CLI scaffolding)
    min_support: int = 2
    max_set_size: int = 3
    candidate_cap: int = 100


class DictionaryConfig(BaseModel):
    """Ensemble dictionary selection thresholds/caps (Phase 1 Week 2)."""
    synergy_min: float = 0.0
    stability_min: float = 0.5
    max_per_family: int = 50
    dedup_jaccard_min: float = 0.5
    exemplars_top_k: int = 3
    min_passed_per_top_family: int = 5  # mock threshold
    families: Optional[List[str]] = None  # None => use discovery.week2_screening.top_families


class NecessityConfig(BaseModel):
    """Week 6 necessity toggles (mock/deterministic)."""
    disable_higher_order: bool = True
    retrain_on_subset: bool = False


class SurrogateConfig(BaseModel):
    """Surrogate modeling configuration."""
    # Week 1 baselines
    monotone: bool = True
    max_interaction_order: int = 2
    group_l1: float = 0.0
    l2: float = 0.0
    cv_folds: int = 5
    early_stopping_patience: int = 5
    calibration: str = "isotonic"  # mock calibration knob

    # Back-compat (older CLI fields)
    nonneg: bool = True
    interactions: int = 0
    selection: str = "aic"

    # Week 6 necessity toggles (mock)
    necessity: NecessityConfig = NecessityConfig()

class SafetyEvalConfig(BaseModel):
    """Week 5 safety evaluation defaults (mock/deterministic)."""
    n_prompts_per_family: int = 400
    n_seeds: int = 5
    transfer_scale: float = 0.7
    subset_tests: bool = True


class Gate4RobustnessConfig(BaseModel):
    """Gate 4 robustness evaluation knobs (mock/deterministic)."""
    n_paraphrases: int = 2
    n_adversarial: int = 2
    seeds: int = 5


class Gate4AcceptanceConfig(BaseModel):
    """Gate 4 acceptance thresholds."""
    necessity_drop_min: float = 0.10
    seed_stability_min: float = 0.8
    effect_persistence_min: float = 0.7


class Gate4Config(BaseModel):
    robustness: Gate4RobustnessConfig = Gate4RobustnessConfig()
    acceptance: Gate4AcceptanceConfig = Gate4AcceptanceConfig()


class CausalConfig(BaseModel):
    """Causal evaluation configuration."""
    ablation_strength: float = 1.0
    patch_window: int = 4
    subset_size: int = 16

    # Week 1 baselines
    seeds_count: int = 5
    ablation_noise_mode: str = "var_match"
    activation_transfer_scale: float = 0.7
    subset_tests: bool = True

    # Week 3 targeted eval knobs (mock/deterministic)
    n_ensembles_per_family: int = 10
    n_prompts_per_cell: int = 200
    transfer_scale: float = 0.7

    # Week 5 safety eval defaults
    safety_eval: SafetyEvalConfig = SafetyEvalConfig()

    # Week 6 Gate 4 config
    gate4: Gate4Config = Gate4Config()


class EditingConfig(BaseModel):
    """Safety editing configuration."""
    # Week 1 baselines (kept for back-compat)
    scale: float = 0.3
    clamp: Optional[float] = None
    rollback_delta: float = 0.05
    rollback_thresholds: RollbackThresholds = RollbackThresholds()

    # Week 5 safety edit planning
    max_edit_scale: float = 0.3
    step_schedule: List[float] = Field(default_factory=lambda: [0.1, 0.2, 0.3])
    benign_deg_pct_max: float = 0.5
    specificity_min_ratio: float = 1.2
    families: Optional[List[str]] = None
    target_reduction_pct: float = 0.20


class DatasetConfig(BaseModel):
    """Dataset configuration for mocks."""
    name: str = "synthetic_parity"
    n_samples: int = 64
    n_features: int = 8
    variant: Optional[str] = "tiny"
    # Optional dataset-provided task metadata
    task_family: Optional[str] = None
    split: Optional[str] = None
    # Week 7 overlay flag for per-family 24-layer expansion
    top_behaviors: Optional[bool] = None


# Week 4 matrix evaluation config
class MatrixPerCellConfig(BaseModel):
    n_prompts: int = 500
    n_seeds: int = 5
    ci_method: str = "bootstrap"
    alpha: float = 0.05
    fdr: float = 0.10


class MatrixConfig(BaseModel):
    families: Optional[List[str]] = None  # None => resolve from discovery.week2_screening.top_families and overlays
    methods: List[str] = Field(
        default_factory=lambda: [
            "pairwise_baseline",
            "no_ensembles_patching",
            "auto_circuit_no_hyper",
            "std_steering",
            "hypercircuit_full",
        ]
    )
    half_matrix: bool = True
    per_cell: MatrixPerCellConfig = MatrixPerCellConfig()


# Week 7 semantic labeling configuration
class LabelingConfig(BaseModel):
    """Semantic labeling finalization (mock/deterministic)."""
    exemplars_top_k: int = 3
    agreement_targets: Optional[Mapping[str, float]] = None  # e.g., {"kappa_min": 0.6}
    drift_check_window: int = 1


# Week 7 dashboards export configuration
class DashboardConfig(BaseModel):
    """Dashboard export toggles and filenames."""
    include_ensembles: bool = True
    include_labels: bool = True
    include_summary: bool = True
    out_ensembles: str = "dashboard_ensembles.json"
    out_labels: str = "dashboard_labels.json"
    out_summary: str = "dashboard_summary.json"


# Week 8 release configuration
class ReleaseAcceptanceRequirements(BaseModel):
    """Acceptance requirements for release gating (set None to ignore a gate)."""
    gate1_go: Optional[bool] = None
    gate2_accept_all: Optional[bool] = None
    gate3_accept_all: Optional[bool] = None
    gate4_accept: Optional[bool] = None


class ReleaseOutNames(BaseModel):
    """Override filenames for Week 8 artifacts."""
    frozen_dictionary: str = "ensemble_dictionary_frozen.json"
    release_manifest: str = "release_manifest.json"
    final_report: str = "final_report.json"
    final_summary: str = "final_summary.md"
    scaling_doc: str = "SCALING_RECOMMENDATION.md"


class ReleaseConfig(BaseModel):
    """Week 8 release pipeline configuration."""
    snapshot_tag: str = "v1"
    include_families: Optional[List[str]] = None
    acceptance_requirements: ReleaseAcceptanceRequirements = ReleaseAcceptanceRequirements()
    out_dir_names: ReleaseOutNames = ReleaseOutNames()
    # Optional manifold/version alignment tag (display/provenance only)
    manifold_version: Optional[str] = None


class Config(BaseModel):
    """Top-level configuration across stages."""
    run: RunConfig = RunConfig()
    logging: LoggingConfig = LoggingConfig()
    discovery: DiscoveryConfig = DiscoveryConfig()
    dictionary: DictionaryConfig = DictionaryConfig()
    surrogate: SurrogateConfig = SurrogateConfig()
    causal: CausalConfig = CausalConfig()
    editing: EditingConfig = EditingConfig()
    dataset: DatasetConfig = DatasetConfig()
    matrix: MatrixConfig = MatrixConfig()
    labeling: LabelingConfig = LabelingConfig()
    dashboard: DashboardConfig = DashboardConfig()
    release: ReleaseConfig = ReleaseConfig()


# ----------------------
# Loader & utilities
# ----------------------

def _deep_update(dst: MutableMapping[str, Any], src: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for k, v in src.items():
        if isinstance(v, Mapping) and isinstance(dst.get(k), Mapping):
            _deep_update(dst[k], v)  # type: ignore[index]
        else:
            dst[k] = v  # type: ignore[index]
    return dst


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping.")
    return data


def _apply_override(target: MutableMapping[str, Any], dotted_key: str, value: str) -> None:
    keys = dotted_key.split(".")
    cur: MutableMapping[str, Any] = target
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], Mapping):
            cur[k] = {}
        cur = cur[k]  # type: ignore[assignment]
    # best-effort type casting
    lit: Any = value
    if value.lower() in {"true", "false"}:
        lit = value.lower() == "true"
    else:
        try:
            if "." in value:
                lit = float(value)
            else:
                lit = int(value)
        except ValueError:
            lit = value
    cur[keys[-1]] = lit


def merge_configs(paths: Sequence[Path], overrides: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for p in paths:
        _deep_update(merged, _load_yaml_file(p))
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Bad override format (expected key=val): {ov}")
        k, v = ov.split("=", 1)
        _apply_override(merged, k.strip(), v.strip())
    return merged


def load_config(paths: Sequence[str | Path], overrides: Optional[Sequence[str]] = None) -> Config:
    """Load, merge, and validate config(s) into a typed Config."""
    p_list = [Path(p) for p in paths]
    data = merge_configs(p_list, overrides)
    try:
        return Config.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"Invalid config: {e}") from e


def ensure_dir(path: Path) -> None:
    """Create directory if not exists."""
    path.mkdir(parents=True, exist_ok=True)


def stage_path(run_dir: Path, filename: str) -> Path:
    """Helper to construct an artifact path under a run directory."""
    return run_dir / filename
