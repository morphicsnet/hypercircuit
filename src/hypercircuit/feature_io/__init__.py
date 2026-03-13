"Feature-space adapters and descriptors."

from .adapter import FeatureAdapter, FeatureSpaceDescriptor
from .ensemble import EnsembleFeatureAdapter, EnsembleDescriptor
from hypercircuit.utils.adapter_registry import register_adapter

register_adapter("feature_adapter", "ensemble", EnsembleFeatureAdapter)

__all__ = [
    "FeatureAdapter",
    "FeatureSpaceDescriptor",
    "EnsembleFeatureAdapter",
    "EnsembleDescriptor",
]
