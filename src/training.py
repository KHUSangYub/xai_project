"""학습 루프와 평가 루틴."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Literal, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader


@dataclass
class TrainResult:
    """학습 요약."""

    best_val_accuracy: float
    best_checkpoint: Path
    history: Dict[str, list[float]]


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=-1)
    return (preds == labels).float().mean().item()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    train: bool = True,
    forward_kwargs_fn: Optional[Callable[[dict], dict]] = None,
) -> Dict[str, float]:
    """단일 에폭 학습/검증."""

    model.train(train)
    total_loss = 0.0
    total_acc = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        kwargs = forward_kwargs_fn(batch) if forward_kwargs_fn else batch
        logits = model(**kwargs)
        loss = criterion(logits, batch["labels"])
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total_loss += loss.item()
        total_acc += accuracy(logits.detach(), batch["labels"])
    n = len(loader)
    return {"loss": total_loss / n, "accuracy": total_acc / n}


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
    output_dir: Path,
    forward_kwargs_fn: Optional[Callable[[dict], dict]] = None,
) -> TrainResult:
    """일반화된 학습 루틴."""

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_ckpt = output_dir / "best.pt"
    model.to(device)
    for epoch in range(1, num_epochs + 1):
        train_stats = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=True,
            forward_kwargs_fn=forward_kwargs_fn,
        )
        val_stats = run_epoch(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=False,
            forward_kwargs_fn=forward_kwargs_fn,
        )
        history["train_loss"].append(train_stats["loss"])
        history["train_acc"].append(train_stats["accuracy"])
        history["val_loss"].append(val_stats["loss"])
        history["val_acc"].append(val_stats["accuracy"])
        if val_stats["accuracy"] > best_acc:
            best_acc = val_stats["accuracy"]
            torch.save({"model_state": model.state_dict()}, best_ckpt)
    return TrainResult(best_val_accuracy=best_acc, best_checkpoint=best_ckpt, history=history)

