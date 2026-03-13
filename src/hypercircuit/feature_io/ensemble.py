from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence

import torch

from hypercircuit.feature_io.adapter import FeatureAdapter, FeatureSpaceDescriptor


@dataclass
class EnsembleDescriptor:
    ensemble_id: str
    member_feature_spaces: Sequence[FeatureSpaceDescriptor]
    reconciliation_method: str = "identity"
    agreement_metric: Optional[str] = None
    minimum_support: Optional[float] = None
    consensus_version: str = "v0"


@dataclass
class EnsembleFeatureAdapter(FeatureAdapter):
    """FeatureAdapter stub for ensemble SAE consensus."""

    members: Sequence[FeatureAdapter]
    descriptor: EnsembleDescriptor

    def feature_space_info(self) -> FeatureSpaceDescriptor:
        return FeatureSpaceDescriptor(
            feature_space_id=self.descriptor.ensemble_id,
            feature_space_type="ensemble",
            feature_space_version=self.descriptor.consensus_version,
            producer="ensemble_adapter",
            producer_version=None,
            model_id=None,
            layer_map=None,
            dim=None,
            sparsity_kind=None,
            top_k=None,
            min_activation=None,
            checksum=None,
        )

    def encode_batch(self, activations_by_layer: Mapping[int, torch.Tensor]) -> Mapping[int, torch.Tensor]:
        # Placeholder: consensus logic not implemented yet
        raise NotImplementedError("EnsembleFeatureAdapter.encode_batch not implemented (use identity reconciliation stage).")

    def validate_against_model(self, model_info: Mapping[str, Any]) -> None:
        for m in self.members:
            m.validate_against_model(model_info)

    def feature_names(self, layer: Optional[int] = None) -> List[str]:
        # No consensus feature names by default
        return []

    def member_feature_maps(self) -> Sequence[FeatureSpaceDescriptor]:
        return [m.feature_space_info() for m in self.members]

    def consensus_features(self) -> Sequence[str]:
        return []
