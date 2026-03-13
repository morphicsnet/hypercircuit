"""Model I/O adapters for real-model integration."""

from .adapter import ModelAdapter
from hypercircuit.utils.adapter_registry import register_adapter

# Register default local HF adapter
register_adapter("model_adapter", "hf_local", ModelAdapter)

__all__ = ["ModelAdapter"]
