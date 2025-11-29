"Surrogate modeling scaffold: interpretable monotone combiner."

from .model import MonotoneCombiner
from . import train as train

__all__ = ["MonotoneCombiner", "train"]
