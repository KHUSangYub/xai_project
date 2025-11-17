"""평가 및 혼동 행렬, 리포트 생성."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from .constants import EMOTION_ORDER


def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: str, forward_kwargs_fn=None) -> Dict[str, np.ndarray]:
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            kwargs = forward_kwargs_fn(batch) if forward_kwargs_fn else batch
            logits = model(**kwargs)
            preds.append(torch.argmax(logits, dim=-1).cpu().numpy())
            labels.append(batch["labels"].cpu().numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(labels)
    conf = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=EMOTION_ORDER, output_dict=True)
    return {"confusion_matrix": conf, "classification_report": report}


def save_report(report: Dict[str, Dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["class,precision,recall,f1,support"]
    for label in EMOTION_ORDER:
        metrics = report[label]
        lines.append(
            f"{label},{metrics['precision']:.4f},{metrics['recall']:.4f},{metrics['f1-score']:.4f},{metrics['support']}"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")

