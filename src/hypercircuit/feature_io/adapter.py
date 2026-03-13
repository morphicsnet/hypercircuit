from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Mapping, Optional, Protocol

import torch


@dataclass
class FeatureSpaceDescriptor:
    """Stable descriptor for a feature space/dictionary."""

    feature_space_id: str
    feature_space_type: str
    feature_space_version: str = "v0"

    producer: Optional[str] = None
    producer_version: Optional[str] = None
    model_id: Optional[str] = None
    layer_map: Optional[Mapping[int, str]] = None

    dim: Optional[int] = None
    sparsity_kind: Optional[str] = None
    top_k: Optional[int] = None
    min_activation: Optional[float] = None

    checksum: Optional[str] = None
    dictionary_id: Optional[str] = None
    dictionary_version: Optional[str] = None
    dictionary_type: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FeatureAdapter(Protocol):
    """Interface: activations in → feature events out."""

    def feature_space_info(self) -> FeatureSpaceDescriptor:
        ...

    def encode_batch(self, activations_by_layer: Mapping[int, torch.Tensor]) -> Mapping[int, torch.Tensor]:
        ...

    def validate_against_model(self, model_info: Mapping[str, Any]) -> None:
        ...

    def feature_names(self, layer: Optional[int] = None) -> List[str]:
        ...

