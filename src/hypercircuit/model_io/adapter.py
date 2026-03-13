from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from hypercircuit.utils.config import ActivationTargetSpec

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@dataclass
class ModelAdapter:
    """Adapter for local HF models with activation capture."""

    hf_model: str
    device: str = "cpu"
    dtype: str = "float32"
    batch_size: int = 4
    max_length: int = 256
    layers: List[int] | None = None
    activation_kind: str = "residual"  # residual | mlp | attn
    targets: List[ActivationTargetSpec] | None = None

    def _resolve_dtype(self) -> torch.dtype:
        return _DTYPES.get(self.dtype, torch.float32)

    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy import

        tok = AutoTokenizer.from_pretrained(self.hf_model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.hf_model,
            torch_dtype=self._resolve_dtype(),
        )
        model.to(self.device)
        model.eval()
        return model, tok

    def model_info(self, model) -> Dict[str, object]:
        return {
            "model_id": self.hf_model,
            "model_class": model.__class__.__name__,
            "config_name": getattr(getattr(model, "config", None), "_name_or_path", None),
            "revision": getattr(getattr(model, "config", None), "_commit_hash", None),
            "n_layers": getattr(getattr(model, "config", None), "num_hidden_layers", None),
            "hidden_size": getattr(getattr(model, "config", None), "hidden_size", None),
            "dtype": str(self.dtype),
            "device": str(self.device),
            "fingerprint": self.model_fingerprint(model),
        }

    def tokenizer_info(self, tokenizer) -> Dict[str, object]:
        return {
            "tokenizer_id": getattr(tokenizer, "name_or_path", None),
            "tokenizer_class": tokenizer.__class__.__name__,
        }

    def model_fingerprint(self, model) -> str:
        payload = {
            "name": self.hf_model,
            "config": model.config.to_dict() if hasattr(model, "config") else {},
        }
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:12]

    def _resolve_targets(self, model) -> List[ActivationTargetSpec]:
        if self.targets:
            raw: List[ActivationTargetSpec] = []
            for t in self.targets:
                if isinstance(t, ActivationTargetSpec):
                    raw.append(t)
                elif isinstance(t, dict):
                    raw.append(ActivationTargetSpec.model_validate(t))
                else:
                    raise ValueError(f"Unsupported target spec type: {type(t)}")
            normalized_layers = self._normalize_layers(model, [int(t.layer_index) for t in raw])
            out: List[ActivationTargetSpec] = []
            for t, L in zip(raw, normalized_layers):
                if int(t.layer_index) == int(L):
                    out.append(t)
                else:
                    data = t.model_dump()
                    data["layer_index"] = int(L)
                    out.append(ActivationTargetSpec.model_validate(data))
            return out
        layers = self._normalize_layers(model)
        return [ActivationTargetSpec(target_kind=self.activation_kind, layer_index=L) for L in layers]

    def _normalize_layers(self, model, layers: List[int] | None = None) -> List[int]:
        n_layers = getattr(model.config, "num_hidden_layers", None)
        if n_layers is None:
            # fallback for GPT2
            if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                n_layers = len(model.transformer.h)
            else:
                raise ValueError("Unable to determine number of layers from model config.")
        layers = layers if layers is not None else (self.layers or [-1])
        out: List[int] = []
        for L in layers:
            idx = int(L)
            if idx < 0:
                idx = n_layers + idx
            if idx < 0 or idx >= n_layers:
                raise ValueError(f"Layer index out of range: {L}")
            out.append(idx)
        return out

    def _get_blocks(self, model) -> List[torch.nn.Module]:
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return list(model.transformer.h)
        raise NotImplementedError("Model blocks not found; only GPT2-like models are supported for mlp/attn capture.")

    def _extract_hook_output(self, out):
        if isinstance(out, tuple):
            return out[0]
        return out

    def forward_with_cache(self, model, tokenizer, texts: List[str]) -> Tuple[Dict[str, torch.Tensor], Dict[int, torch.Tensor]]:
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        targets = self._resolve_targets(model)
        kinds = {str(t.target_kind).lower() for t in targets}
        if len(kinds) > 1:
            raise NotImplementedError("Multiple activation target kinds are not yet supported in one pass.")
        kind = list(kinds)[0] if kinds else str(self.activation_kind).lower()
        layers = [int(t.layer_index) for t in targets] if targets else self._normalize_layers(model)
        layer_acts: Dict[int, torch.Tensor] = {}

        if kind in {"residual", "resid", "residual_stream"}:
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states
            if hs is None:
                raise RuntimeError("Model did not return hidden_states.")
            for L in layers:
                layer_acts[L] = hs[L + 1].detach()
        else:
            blocks = self._get_blocks(model)
            handles = []
            for L in layers:
                block = blocks[L]
                if kind in {"mlp", "mlp_out"} and hasattr(block, "mlp"):
                    mod = block.mlp
                elif kind in {"attn", "attn_out"} and hasattr(block, "attn"):
                    mod = block.attn
                else:
                    raise NotImplementedError(f"Activation kind '{kind}' not supported for this model block.")
                def _hook(mod, inp, out, layer=L):
                    layer_acts[layer] = self._extract_hook_output(out).detach()
                handles.append(mod.register_forward_hook(_hook))
            with torch.no_grad():
                _ = model(**inputs, output_hidden_states=False, use_cache=False)
            for h in handles:
                h.remove()
            for L in layers:
                if L not in layer_acts:
                    raise RuntimeError(f"Missing activation for layer {L}.")

        return inputs, layer_acts
