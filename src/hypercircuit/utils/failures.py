from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


FAILURE_MODES = {
    "config_error",
    "model_load_error",
    "tokenization_error",
    "activation_shape_error",
    "dictionary_mismatch",
    "oom",
    "calibration_failure",
    "insufficient_support",
    "unknown",
}


@dataclass
class FailureRecord:
    mode: str
    message: str
    detail: Optional[str] = None


def classify_exception(exc: Exception) -> FailureRecord:
    msg = str(exc)
    lower = msg.lower()
    mode = "unknown"
    if "validation" in lower or "config" in lower:
        mode = "config_error"
    elif "token" in lower and "token" in lower:
        mode = "tokenization_error"
    elif "shape" in lower or "dimension" in lower:
        mode = "activation_shape_error"
    elif "sae" in lower or "dictionary" in lower:
        mode = "dictionary_mismatch"
    elif "out of memory" in lower or "cuda" in lower and "memory" in lower:
        mode = "oom"
    elif "calibration" in lower:
        mode = "calibration_failure"
    elif "support" in lower:
        mode = "insufficient_support"
    elif "model" in lower or "checkpoint" in lower:
        mode = "model_load_error"
    if mode not in FAILURE_MODES:
        mode = "unknown"
    return FailureRecord(mode=mode, message=msg)

