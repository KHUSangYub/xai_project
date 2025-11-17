"""음성 특징 마스킹 실험 도우미."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch

from .prosody import derive_feature_groups


@dataclass
class MaskingResult:
    """마스킹 실험 결과."""

    baseline_accuracy: float
    masked_accuracy: Dict[str, float]


def evaluate_with_mask(
    model: torch.nn.Module,
    loader,
    device: str,
    mask_groups: Dict[str, Sequence[int]] | None = None,
    feature_names: Sequence[str] | None = None,
) -> MaskingResult:
    if mask_groups is None:
        if feature_names is None:
            raise ValueError("mask_groups 또는 feature_names 중 하나는 제공되어야 합니다.")
        mask_groups = derive_feature_groups(feature_names)

    def forward_kwargs(batch):
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "audio_features": batch["audio_features"],
        }

    def run_inference(mask: Sequence[int] | None = None) -> float:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                if mask:
                    batch["audio_features"][:, mask] = 0.0
                logits = model(**forward_kwargs(batch))
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                total += batch["labels"].size(0)
        return correct / total if total else 0.0

    baseline = run_inference()
    masked_scores = {name: run_inference(indices) for name, indices in mask_groups.items()}
    return MaskingResult(baseline_accuracy=baseline, masked_accuracy=masked_scores)

