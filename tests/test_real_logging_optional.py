from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from hypercircuit.logging.real_activations import RealActivationLogger


def test_real_logging_optional(tmp_path: Path) -> None:
    if os.getenv("HYPERCIRCUIT_REAL_TEST") != "1":
        pytest.skip("Set HYPERCIRCUIT_REAL_TEST=1 to run real-model smoke test.")
    try:
        from transformers import AutoModelForCausalLM  # noqa: F401
    except Exception:
        pytest.skip("transformers not installed.")

    # Small jsonl dataset
    data_path = tmp_path / "data.jsonl"
    data_path.write_text('{"text": "Hello world"}\n{"text": "Test sentence"}\n')

    # Load model to get hidden size
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    hidden = int(getattr(model.config, "n_embd", 0) or getattr(model.config, "hidden_size", 0))
    if hidden <= 0:
        pytest.skip("Could not infer hidden size.")

    # Tiny SAE dict
    W = np.random.randn(8, hidden).astype(np.float32)
    b = np.zeros((8,), dtype=np.float32)
    sae_path = tmp_path / "sae.npz"
    np.savez(sae_path, W_enc=W, b_enc=b)

    out_path = tmp_path / "logs.jsonl"
    metrics = RealActivationLogger(
        model_cfg={
            "hf_model": "sshleifer/tiny-gpt2",
            "device": "cpu",
            "dtype": "float32",
            "batch_size": 2,
            "max_length": 16,
            "layers": [-1],
            "activation_kind": "residual",
        },
        sae_cfg={
            "format": "npz",
            "path": str(sae_path),
            "top_k": 2,
            "min_activation": 0.0,
        },
        dataset_cfg={
            "source": "jsonl",
            "path": str(data_path),
            "text_field": "text",
            "max_samples": 2,
        },
        logging_cfg={"member_granularity": "node_id", "source_kind": "hf_local", "run_id": "real-smoke"},
        seed=0,
    ).run(out_path=out_path)

    assert out_path.exists() and out_path.stat().st_size > 0
    assert int(metrics.get("total_events", 0)) >= 0
