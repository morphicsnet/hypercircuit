from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Mapping, Dict

import numpy as np
import torch

from hypercircuit.feature_io.adapter import FeatureAdapter, FeatureSpaceDescriptor

try:
    from safetensors.torch import load_file as _load_safetensors  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _load_safetensors = None


class SAEAdapter:
    """Abstract adapter for loading SAE dictionaries and activations."""

    def feature_names(self) -> List[str]:
        raise NotImplementedError

    def sample_activations(self, n_samples: int) -> np.ndarray:
        """Return mock activations [n_samples, n_features]."""
        raise NotImplementedError

    def encode_activations(self, acts: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Encode dense activations into SAE feature activations."""
        raise NotImplementedError


@dataclass
class FakeSAEDictionary(SAEAdapter):
    """Simple fake SAE dictionary for smoke tests.

    Generates Bernoulli(approx sparsity) activations with small Gaussian noise.
    Deterministic given the numpy RNG state (controlled via utils.seed).
    """
    n_features: int
    sparsity: float = 0.1
    _W_enc: Optional[np.ndarray] = None
    _b_enc: Optional[np.ndarray] = None

    def feature_names(self) -> List[str]:
        return [f"f{i}" for i in range(self.n_features)]

    def sample_activations(self, n_samples: int) -> np.ndarray:
        base = (np.random.rand(n_samples, self.n_features) < self.sparsity).astype(float)
        noise = 0.05 * np.random.randn(n_samples, self.n_features)
        return np.clip(base + noise, 0.0, None)

    def encode_activations(self, acts: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Deterministic mock encoding: linear + ReLU using a fixed random matrix."""
        if self._W_enc is None or self._b_enc is None:
            rng = np.random.default_rng(0)
            in_dim = int(acts.shape[-1]) if hasattr(acts, "shape") else self.n_features
            self._W_enc = rng.standard_normal((self.n_features, in_dim)).astype(np.float32)
            self._b_enc = np.zeros((self.n_features,), dtype=np.float32)
        W = torch.from_numpy(self._W_enc)
        b = torch.from_numpy(self._b_enc)
        x = torch.as_tensor(acts, dtype=W.dtype)
        out = x @ W.T + b
        return torch.relu(out)


@dataclass
class PretrainedSAEDictionary(SAEAdapter):
    """Load a pretrained SAE dictionary from disk (safetensors or npz)."""
    W_enc: torch.Tensor
    b_enc: torch.Tensor
    feature_names_list: Optional[List[str]] = None
    dictionary_id: Optional[str] = None
    dictionary_version: Optional[str] = None
    dictionary_type: Optional[str] = None
    feature_space_id: Optional[str] = None
    feature_space_version: Optional[str] = None
    checksum: Optional[str] = None

    @property
    def n_features(self) -> int:
        return int(self.W_enc.shape[0])

    @property
    def input_dim(self) -> int:
        return int(self.W_enc.shape[1])

    def feature_names(self) -> List[str]:
        if self.feature_names_list:
            return list(self.feature_names_list)
        return [f"f{i}" for i in range(self.n_features)]

    def sample_activations(self, n_samples: int) -> np.ndarray:
        # Not used for real pipelines; return zeros for compatibility.
        return np.zeros((n_samples, self.n_features), dtype=np.float32)

    def encode_activations(self, acts: np.ndarray | torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(acts, dtype=self.W_enc.dtype, device=self.W_enc.device)
        out = x @ self.W_enc.T + self.b_enc
        return torch.relu(out)

    def compute_checksum(self) -> str:
        import hashlib

        h = hashlib.sha256()
        h.update(self.W_enc.detach().cpu().numpy().tobytes())
        h.update(self.b_enc.detach().cpu().numpy().tobytes())
        return h.hexdigest()[:12]

    @classmethod
    def from_path(cls, path: str, fmt: str = "safetensors") -> "PretrainedSAEDictionary":
        fmt_l = fmt.lower()
        names = None
        meta: Dict[str, object] = {}
        if fmt_l == "safetensors":
            if _load_safetensors is None:  # pragma: no cover
                raise ImportError("safetensors is required to load safetensors format.")
            tensors = _load_safetensors(path)
            if "W_enc" not in tensors or "b_enc" not in tensors:
                raise ValueError("safetensors must contain W_enc and b_enc.")
            W = tensors["W_enc"].detach().cpu()
            b = tensors["b_enc"].detach().cpu()
        elif fmt_l == "npz":
            data = np.load(path, allow_pickle=True)
            if "W_enc" not in data or "b_enc" not in data:
                raise ValueError("npz must contain W_enc and b_enc.")
            W = torch.from_numpy(np.asarray(data["W_enc"], dtype=np.float32))
            b = torch.from_numpy(np.asarray(data["b_enc"], dtype=np.float32))
            names = None
            if "feature_names" in data:
                names = [str(x) for x in data["feature_names"].tolist()]
        else:
            raise ValueError(f"Unknown SAE format: {fmt}")

        # Optional sidecar config.json for feature names/metadata
        try:
            import json
            from pathlib import Path
            cfg_path = Path(path).with_suffix(".config.json")
            if cfg_path.exists():
                cfg = json.loads(cfg_path.read_text())
                if isinstance(cfg.get("feature_names"), list):
                    names = [str(x) for x in cfg["feature_names"]]
                meta = {
                    "dictionary_id": cfg.get("dictionary_id"),
                    "dictionary_version": cfg.get("dictionary_version"),
                    "dictionary_type": cfg.get("dictionary_type"),
                    "feature_space_id": cfg.get("feature_space_id"),
                    "feature_space_version": cfg.get("feature_space_version"),
                }
        except Exception:
            pass

        obj = cls(
            W_enc=W,
            b_enc=b,
            feature_names_list=names,
            dictionary_id=str(meta.get("dictionary_id")) if meta.get("dictionary_id") else None,
            dictionary_version=str(meta.get("dictionary_version")) if meta.get("dictionary_version") else None,
            dictionary_type=str(meta.get("dictionary_type")) if meta.get("dictionary_type") else None,
            feature_space_id=str(meta.get("feature_space_id")) if meta.get("feature_space_id") else None,
            feature_space_version=str(meta.get("feature_space_version")) if meta.get("feature_space_version") else None,
        )
        try:
            obj.checksum = obj.compute_checksum()
        except Exception:
            obj.checksum = None
        return obj


@dataclass
class SAEFeatureAdapter(FeatureAdapter):
    """FeatureAdapter implementation for SAE dictionaries."""
    sae_cfg: Mapping[str, object]
    layer_map: Dict[int, PretrainedSAEDictionary] = field(default_factory=dict)

    @classmethod
    def from_config(cls, sae_cfg: Mapping[str, object]) -> "SAEFeatureAdapter":
        return cls(sae_cfg=sae_cfg, layer_map={})

    def _resolve_layer_path(self, layer: int) -> str:
        layer_map = self.sae_cfg.get("layer_map") or {}
        if isinstance(layer_map, dict) and str(layer) in layer_map:
            return str(layer_map[str(layer)])
        if isinstance(layer_map, dict) and layer in layer_map:
            return str(layer_map[layer])
        path = self.sae_cfg.get("path")
        if not path:
            raise ValueError("sae.path is required (or sae.layer_map) for SAEFeatureAdapter")
        return str(path)

    def _get_dict(self, layer: int) -> PretrainedSAEDictionary:
        if layer in self.layer_map:
            return self.layer_map[layer]
        fmt = str(self.sae_cfg.get("format", "safetensors"))
        path = self._resolve_layer_path(layer)
        d = PretrainedSAEDictionary.from_path(path, fmt=fmt)
        # Overlay config-provided metadata if present
        if self.sae_cfg.get("dictionary_id"):
            d.dictionary_id = str(self.sae_cfg.get("dictionary_id"))
        if self.sae_cfg.get("dictionary_version"):
            d.dictionary_version = str(self.sae_cfg.get("dictionary_version"))
        if self.sae_cfg.get("dictionary_type"):
            d.dictionary_type = str(self.sae_cfg.get("dictionary_type"))
        if self.sae_cfg.get("feature_space_id"):
            d.feature_space_id = str(self.sae_cfg.get("feature_space_id"))
        if self.sae_cfg.get("feature_space_version"):
            d.feature_space_version = str(self.sae_cfg.get("feature_space_version"))
        self.layer_map[layer] = d
        return d

    def get_layer_dict(self, layer: int) -> PretrainedSAEDictionary:
        return self._get_dict(layer)

    def feature_space_info(self) -> FeatureSpaceDescriptor:
        # Use first loaded dict if available, else rely on config
        any_dict = next(iter(self.layer_map.values()), None)
        checksum = None
        if any_dict and any_dict.checksum:
            checksum = any_dict.checksum
        feature_space_id = str(self.sae_cfg.get("feature_space_id") or (f"sae-{checksum}" if checksum else "sae-unknown"))
        feature_space_version = str(self.sae_cfg.get("feature_space_version") or "v0")
        dim = int(any_dict.n_features) if any_dict is not None else None
        dict_id = (any_dict.dictionary_id if any_dict else None) or (checksum if checksum else None)
        return FeatureSpaceDescriptor(
            feature_space_id=feature_space_id,
            feature_space_type=str(self.sae_cfg.get("dictionary_type") or "sae"),
            feature_space_version=feature_space_version,
            producer="sae_adapter",
            producer_version=None,
            model_id=None,
            layer_map={int(k): str(v.feature_space_id or feature_space_id) for k, v in self.layer_map.items()} if self.layer_map else None,
            dim=dim,
            sparsity_kind="top_k",
            top_k=int(self.sae_cfg.get("top_k", 16)),
            min_activation=float(self.sae_cfg.get("min_activation", 0.0)),
            checksum=checksum,
            dictionary_id=dict_id,
            dictionary_version=any_dict.dictionary_version if any_dict else None,
            dictionary_type=any_dict.dictionary_type if any_dict else str(self.sae_cfg.get("dictionary_type") or "sae"),
        )

    def encode_batch(self, activations_by_layer: Mapping[int, torch.Tensor]) -> Mapping[int, torch.Tensor]:
        out: Dict[int, torch.Tensor] = {}
        for layer, acts in activations_by_layer.items():
            d = self._get_dict(int(layer))
            if hasattr(acts, "shape") and acts.shape[-1] != d.input_dim:
                raise ValueError(
                    f"SAE input_dim ({d.input_dim}) does not match activation size ({acts.shape[-1]}) for layer {layer}"
                )
            out[int(layer)] = d.encode_activations(acts)
        return out

    def validate_against_model(self, model_info: Mapping[str, object]) -> None:
        hidden = model_info.get("hidden_size")
        if hidden is None:
            return
        for layer, d in self.layer_map.items():
            if d.input_dim != int(hidden):
                raise ValueError(
                    f"SAE input_dim ({d.input_dim}) does not match model hidden_size ({hidden}) for layer {layer}"
                )

    def feature_names(self, layer: Optional[int] = None) -> List[str]:
        if layer is None:
            any_dict = next(iter(self.layer_map.values()), None)
            return any_dict.feature_names() if any_dict else []
        return self._get_dict(int(layer)).feature_names()
