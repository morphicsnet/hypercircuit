from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional
import json

import numpy as np
import torch

from hypercircuit.model_io.adapter import ModelAdapter
from hypercircuit.sae_io.loaders import SAEFeatureAdapter
from hypercircuit.logging.schema import EventContext, build_event
from hypercircuit.utils.io import append_jsonl
from hypercircuit.utils.seed import seed_context


@dataclass
class RealActivationLogger:
    """Real-model activation logger (HF models + pretrained SAE dictionaries)."""

    model_cfg: Mapping[str, object]
    sae_cfg: Mapping[str, object]
    dataset_cfg: Mapping[str, object]
    logging_cfg: Mapping[str, object]
    seed: int
    event_ctx: EventContext | None = None

    def _load_dataset(self) -> List[Dict[str, object]]:
        source = str(self.dataset_cfg.get("source", "hf"))
        text_field = str(self.dataset_cfg.get("text_field", "text"))
        label_field = self.dataset_cfg.get("label_field")
        sequence_field = self.dataset_cfg.get("sequence_field")
        max_samples = self.dataset_cfg.get("max_samples")
        cache_dir = self.dataset_cfg.get("cache_dir")

        rows: List[Dict[str, object]] = []
        if source == "hf":
            from datasets import load_dataset  # lazy import

            hf_name = self.dataset_cfg.get("hf_name")
            hf_split = self.dataset_cfg.get("hf_split", "train")
            if not hf_name:
                raise ValueError("dataset.hf_name is required when dataset.source=hf")
            ds = load_dataset(str(hf_name), split=str(hf_split), cache_dir=cache_dir)
            for i, rec in enumerate(ds):
                if max_samples is not None and i >= int(max_samples):
                    break
                if text_field not in rec:
                    raise KeyError(f"text_field '{text_field}' not found in dataset record")
                row = {"text": rec[text_field]}
                if label_field and label_field in rec:
                    row["label"] = rec[label_field]
                if sequence_field and sequence_field in rec:
                    row["sequence_id"] = rec[sequence_field]
                rows.append(row)
        elif source == "jsonl":
            path = self.dataset_cfg.get("path")
            if not path:
                raise ValueError("dataset.path is required when dataset.source=jsonl")
            p = Path(str(path))
            if not p.exists():
                raise FileNotFoundError(f"dataset.path not found: {p}")
            with p.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if max_samples is not None and i >= int(max_samples):
                        break
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    if text_field not in rec:
                        raise KeyError(f"text_field '{text_field}' not found in jsonl record")
                    row = {"text": rec[text_field]}
                    if label_field and label_field in rec:
                        row["label"] = rec[label_field]
                    if sequence_field and sequence_field in rec:
                        row["sequence_id"] = rec[sequence_field]
                    rows.append(row)
        else:
            raise ValueError(f"Unsupported dataset.source: {source}")

        return rows

    def run(self, out_path: Path) -> Mapping[str, object]:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with seed_context(self.seed):
            adapter = ModelAdapter(
                hf_model=str(self.model_cfg.get("hf_model")),
                device=str(self.model_cfg.get("device", "cpu")),
                dtype=str(self.model_cfg.get("dtype", "float32")),
                batch_size=int(self.model_cfg.get("batch_size", 4)),
                max_length=int(self.model_cfg.get("max_length", 256)),
                layers=list(self.model_cfg.get("layers", [-1])) if self.model_cfg.get("layers") is not None else None,
                activation_kind=str(self.model_cfg.get("activation_kind", "residual")),
                targets=self.model_cfg.get("targets"),
            )
            model, tok = adapter.load_model()
            model_fingerprint = adapter.model_fingerprint(model)
            model_info = adapter.model_info(model)
            tokenizer_info = adapter.tokenizer_info(tok)
            targets = adapter._resolve_targets(model)
            layers = sorted({int(t.layer_index) for t in targets}) if targets else adapter._normalize_layers(model)

            examples = self._load_dataset()
            if not examples:
                raise ValueError("No dataset rows loaded for real logging.")

            top_k = int(self.sae_cfg.get("top_k", 16))
            min_activation = float(self.sae_cfg.get("min_activation", 0.0))
            member_granularity = str(self.logging_cfg.get("member_granularity", "node_id"))

            # Feature adapter (SAE)
            feature_adapter = SAEFeatureAdapter.from_config(self.sae_cfg)
            if layers:
                # Preload one layer for metadata/shape checks
                feature_adapter.get_layer_dict(int(layers[0]))
            try:
                feature_adapter.validate_against_model(model_info)
            except Exception:
                # allow per-layer validation later if needed
                pass
            feature_info = feature_adapter.feature_space_info()

            if self.event_ctx:
                ctx = EventContext(
                    run_id=self.event_ctx.run_id,
                    source_kind=self.event_ctx.source_kind,
                    feature_space_id=str(feature_info.feature_space_id),
                    feature_space_version=str(feature_info.feature_space_version),
                    dictionary_id=feature_info.dictionary_id or self.event_ctx.dictionary_id,
                    dictionary_version=feature_info.dictionary_version or self.event_ctx.dictionary_version,
                    dictionary_type=feature_info.dictionary_type or self.event_ctx.dictionary_type,
                    run_intent=self.event_ctx.run_intent,
                    task_family=self.event_ctx.task_family,
                    prompt_family=self.event_ctx.prompt_family,
                    split=self.event_ctx.split,
                    label=self.event_ctx.label,
                    capability_tag=self.event_ctx.capability_tag,
                    safety_tag=self.event_ctx.safety_tag,
                )
            else:
                ctx = EventContext(
                    run_id=str(self.logging_cfg.get("run_id", "unknown")),
                    source_kind=str(self.logging_cfg.get("source_kind", "hf_local")),
                    feature_space_id=str(feature_info.feature_space_id),
                    feature_space_version=str(feature_info.feature_space_version),
                    dictionary_id=feature_info.dictionary_id,
                    dictionary_version=feature_info.dictionary_version,
                    dictionary_type=feature_info.dictionary_type,
                    run_intent=str(self.logging_cfg.get("run_intent")) if self.logging_cfg.get("run_intent") else None,
                    task_family=str(self.dataset_cfg.get("task_family")) if self.dataset_cfg.get("task_family") else None,
                    prompt_family=str(self.dataset_cfg.get("prompt_family")) if self.dataset_cfg.get("prompt_family") else None,
                    split=str(self.dataset_cfg.get("split")) if self.dataset_cfg.get("split") else None,
                    label=str(self.dataset_cfg.get("label")) if self.dataset_cfg.get("label") else None,
                    capability_tag=str(self.dataset_cfg.get("capability_tag")) if self.dataset_cfg.get("capability_tag") else None,
                    safety_tag=str(self.dataset_cfg.get("safety_tag")) if self.dataset_cfg.get("safety_tag") else None,
                )

            total_events = 0
            tokens_logged = 0
            counts_by_type: Dict[str, int] = {"sae_features": 0}
            layers_used: set[int] = set()

            # process batches
            bs = int(adapter.batch_size)
            for batch_start in range(0, len(examples), bs):
                batch = examples[batch_start: batch_start + bs]
                texts = [str(r["text"]) for r in batch]
                inputs, layer_acts = adapter.forward_with_cache(model, tok, texts)
                attention_mask = inputs.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones_like(inputs["input_ids"])

                # count valid tokens
                tokens_logged += int(attention_mask.sum().item())

                feats_by_layer = feature_adapter.encode_batch(layer_acts)
                for L in sorted(feats_by_layer.keys()):
                    feats = feats_by_layer[L]  # [B, T, F]
                    layers_used.add(int(L))
                    if top_k <= 0:
                        # emit all features above threshold (not recommended)
                        mask = feats >= min_activation
                        idxs = mask.nonzero(as_tuple=False)
                        for idx in idxs:
                            b, t, f = int(idx[0]), int(idx[1]), int(idx[2])
                            if attention_mask[b, t].item() == 0:
                                continue
                            val = float(feats[b, t, f].item())
                            seq_id = batch[b].get("sequence_id", batch_start + b)
                            extra = {"feature_origin_layer": int(L)}
                            if "label" in batch[b]:
                                extra["label"] = batch[b].get("label")
                            rec = build_event(
                                ctx=ctx,
                                sample_id=batch_start + b,
                                sequence_id=int(seq_id) if seq_id is not None else None,
                                token_index=t,
                                layer=L,
                                node_type="sae_features",
                                node_id=int(f),
                                value=val,
                                extra=extra,
                            )
                            append_jsonl(out_path, [rec])
                            total_events += 1
                            counts_by_type["sae_features"] += 1
                    else:
                        values, idx = torch.topk(feats, k=top_k, dim=-1)
                        values_cpu = values.detach().cpu().numpy()
                        idx_cpu = idx.detach().cpu().numpy()
                        attn_cpu = attention_mask.detach().cpu().numpy()
                        for b in range(values_cpu.shape[0]):
                            for t in range(values_cpu.shape[1]):
                                if attn_cpu[b, t] == 0:
                                    continue
                                for j in range(values_cpu.shape[2]):
                                    v = float(values_cpu[b, t, j])
                                    if v < min_activation:
                                        continue
                                    f = int(idx_cpu[b, t, j])
                                    seq_id = batch[b].get("sequence_id", batch_start + b)
                                    extra = {"feature_origin_layer": int(L)}
                                    if "label" in batch[b]:
                                        extra["label"] = batch[b].get("label")
                                    rec = build_event(
                                        ctx=ctx,
                                        sample_id=batch_start + b,
                                        sequence_id=int(seq_id) if seq_id is not None else None,
                                        token_index=int(t),
                                        layer=int(L),
                                        node_type="sae_features",
                                        node_id=int(f),
                                        value=v,
                                        extra=extra,
                                    )
                                    append_jsonl(out_path, [rec])
                                    total_events += 1
                                    counts_by_type["sae_features"] += 1

            events_per_token = total_events / max(1, tokens_logged)
            metrics: Dict[str, object] = {
                "events_per_token": float(events_per_token),
                "events_by_node_type": counts_by_type,
                "events_per_token_by_node_type": {
                    k: v / max(1, tokens_logged) for k, v in counts_by_type.items()
                },
                "total_events": int(total_events),
                "tokens_logged": int(tokens_logged),
                "n_layers": int(len(layers_used) if layers_used else len(layers)),
                "model_id": str(self.model_cfg.get("hf_model")),
                "model_fingerprint": model_fingerprint,
                "activation_kind": str(self.model_cfg.get("activation_kind", "residual")),
                "layers": sorted(layers_used) if layers_used else list(layers),
                "sae_format": str(self.sae_cfg.get("format", "safetensors")),
                "sae_path": str(self.sae_cfg.get("path", "")),
                "sae_top_k": int(top_k),
                "sae_min_activation": float(min_activation),
                "member_granularity": member_granularity,
                "model_info": model_info,
                "tokenizer_info": tokenizer_info,
                "feature_space": feature_info.as_dict(),
            }
            return metrics
