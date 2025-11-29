from __future__ import annotations

import contextlib
import random
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set seeds across Python, NumPy, and Torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Back-compat alias
def set_seed(seed: int) -> None:
    set_global_seed(seed)


def _capture_rng_state() -> Dict[str, object]:
    state: Dict[str, object] = {
        "py": random.getstate(),
        "np": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }
    if torch.cuda.is_available():
        try:
            state["cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            state["cuda"] = None
    return state


def _restore_rng_state(state: Dict[str, object]) -> None:
    try:
        random.setstate(state["py"])  # type: ignore[arg-type]
        np.random.set_state(state["np"])  # type: ignore[arg-type]
        torch.set_rng_state(state["torch"])  # type: ignore[arg-type]
        cuda_state = state.get("cuda")
        if cuda_state is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(cuda_state)  # type: ignore[arg-type]
            except Exception:
                pass
    except Exception:
        # best-effort restore
        pass


@contextlib.contextmanager
def seed_context(seed: Optional[int]) -> Iterator[None]:
    """Context manager to set and restore RNG state for deterministic operations."""
    if seed is None:
        yield
        return
    old = _capture_rng_state()
    set_global_seed(seed)
    try:
        yield
    finally:
        _restore_rng_state(old)


def expand_seeds(seeds: Optional[List[int]]) -> List[int]:
    """Return a normalized seed list with defaults if missing."""
    return list(seeds or [0, 1, 2, 3, 4])


def active_seed(seeds: Optional[List[int]], stage_idx: int = 0) -> int:
    """Select an active seed for a stage index from a seed list."""
    lst = expand_seeds(seeds)
    if not lst:
        return 0
    return lst[stage_idx % len(lst)]
