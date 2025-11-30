from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Mapping, Sequence

import numpy as np


@dataclass
class CausalHarness:
    """Mock harness for ensemble ablations, transfers, and patching."""
    ablation_strength: float = 1.0

    def ablate(self, X: np.ndarray, features: List[int]) -> np.ndarray:
        X2 = X.copy()
        X2[:, features] *= (1.0 - self.ablation_strength)
        return X2

    def transfer(self, X_src: np.ndarray, X_tgt: np.ndarray, features: List[int]) -> np.ndarray:
        """
        Robust transfer that tolerates different numbers of rows between source and target.
        Copies feature columns for the first m=min(n_src,n_tgt) rows and leaves the rest unchanged.
        """
        X2 = X_tgt.copy()
        if not features:
            return X2
        n_src = int(X_src.shape[0]) if X_src.ndim >= 2 else 0
        n_tgt = int(X_tgt.shape[0]) if X_tgt.ndim >= 2 else 0
        m = min(n_src, n_tgt)
        if m > 0:
            X2[:m, features] = X_src[:m, features]
        return X2

    def evaluate_delta(self, y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> Dict[str, float]:
        """Return simple mock causal deltas."""
        mse_a = float(np.mean((y_true - y_pred_a) ** 2))
        mse_b = float(np.mean((y_true - y_pred_b) ** 2))
        return {"mse_a": mse_a, "mse_b": mse_b, "delta": mse_b - mse_a}

    def evaluate_safety_suite(
        self,
        families: List[str],
        step_scales: Mapping[str, float],
        *,
        n_prompts_per_family: int = 400,
        n_seeds: int = 5,
        transfer_scale: float = 0.7,
        subset_tests: bool = True,
        seed: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        """
        Mock/deterministic safety evaluation per family for a given step.

        Returns per-family rows with keys:
          - family
          - step_scale
          - harmful_rate_reduction
          - benign_deg_pct
          - sufficiency_reinstatement_after_edit
        """
        # Deterministic tiny jitter to avoid divide-by-zero while keeping repeatability
        rng = np.random.default_rng(seed if seed is not None else 0)

        rows: List[Dict[str, float]] = []
        for fam in families:
            s = float(step_scales.get(fam, 0.0))
            # Predict monotone improvements with scale; clamp into plausible mock ranges
            harmful_rate_reduction = float(max(0.0, min(1.0, 0.6 * s)))
            benign_deg_pct = float(max(0.0, min(1.0, 0.12 * s)))
            # Tiny deterministic noise for realism, bounded to be negligible
            harmful_rate_reduction += float(rng.uniform(-1e-9, 1e-9))
            benign_deg_pct += float(rng.uniform(-1e-9, 1e-9))
            suff_reinst = float(max(0.0, 0.05 * s))
            rows.append(
                {
                    "family": fam,
                    "step_scale": s,
                    "harmful_rate_reduction": harmful_rate_reduction,
                    "benign_deg_pct": benign_deg_pct,
                    "sufficiency_reinstatement_after_edit": suff_reinst,
                }
            )
        return rows

    def evaluate_paraphrase_and_adversarial(
        self,
        *,
        expected_effect: float,
        n_paraphrases: int,
        n_adversarial: int,
        seeds: Sequence[int],
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Mock robustness evaluator producing paraphrase and adversarial variants around a baseline effect.

        Returns:
          {
            "baseline_effects": List[float],           # per-seed baseline effects
            "paraphrase_effects": List[List[float]],   # shape (n_seeds, n_paraphrases)
            "adversarial_effects": List[List[float]],  # shape (n_seeds, n_adversarial)
            "effect_persistence": float,               # median(paraphrase / baseline) across all entries
            "seed_stability": float,                   # 1 - std(baseline across seeds), clamped to [0,1]
            "ood_delta": float                         # median(baseline - adversarial) across all entries
          }
        Deterministic given inputs.
        """
        # Ensure non-negative and stable baseline
        base = max(0.0, float(expected_effect))
        seeds_list = list(seeds or [0])

        baseline_effects: List[float] = []
        paraphrase_effects: List[List[float]] = []
        adversarial_effects: List[List[float]] = []

        # Tiny deterministic jitter scales
        base_sigma = 1e-6
        para_center = 0.98
        para_sigma = 2e-3
        adv_center = 0.70
        adv_sigma = 3e-3

        for idx, s in enumerate(seeds_list):
            # Derive a deterministic RNG per-seed
            s_int = int(s)
            root = (0 if seed is None else int(seed))
            rng = np.random.default_rng(root * 1_000_003 + s_int * 97 + idx)

            # Baseline per-seed effect with negligible jitter
            b = float(base + rng.normal(0.0, base_sigma))
            b = max(0.0, b)
            baseline_effects.append(b)

            # Paraphrase variants cluster tightly around baseline
            p_list: List[float] = []
            for j in range(int(n_paraphrases)):
                scale = float(max(0.0, para_center + rng.normal(0.0, para_sigma) * (1.0 - 0.1 * j)))
                p_list.append(float(max(0.0, b * scale)))
            paraphrase_effects.append(p_list)

            # Adversarial variants reduce effect more strongly
            a_list: List[float] = []
            for j in range(int(n_adversarial)):
                scale = float(max(0.0, adv_center + rng.normal(0.0, adv_sigma) * (1.0 - 0.1 * j)))
                a_list.append(float(max(0.0, b * scale)))
            adversarial_effects.append(a_list)

        # Aggregate metrics
        # Seed stability from baseline dispersion
        seed_std = float(np.std(baseline_effects)) if baseline_effects else 0.0
        seed_stability = float(max(0.0, min(1.0, 1.0 - seed_std)))

        # Effect persistence: median(paraphrase / baseline) across all entries
        ratios: List[float] = []
        for b, plist in zip(baseline_effects, paraphrase_effects):
            if b <= 0.0:
                ratios.extend([0.0 for _ in plist])
            else:
                ratios.extend([float(max(0.0, min(2.0, v / b))) for v in plist])
        effect_persistence = float(np.median(ratios)) if ratios else 0.0

        # OOD delta: median(baseline - adversarial) across all entries
        deltas: List[float] = []
        for b, alist in zip(baseline_effects, adversarial_effects):
            deltas.extend([float(max(0.0, b - v)) for v in alist])
        ood_delta = float(np.median(deltas)) if deltas else 0.0

        return {
            "baseline_effects": baseline_effects,
            "paraphrase_effects": paraphrase_effects,
            "adversarial_effects": adversarial_effects,
            "effect_persistence": effect_persistence,
            "seed_stability": seed_stability,
            "ood_delta": ood_delta,
        }
