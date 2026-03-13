"SAE adapters and loaders (includes a fake dictionary for smoke tests)."

from .loaders import SAEFeatureAdapter
from hypercircuit.utils.adapter_registry import register_adapter

register_adapter("feature_adapter", "sae", SAEFeatureAdapter)

__all__ = ["SAEFeatureAdapter"]
