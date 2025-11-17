"""PyTorch Dataset 정의."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from .constants import LABEL_TO_ID
from .data_utils import SampleMetadata
from .feature_extractor import load_audio_feature


def _to_label_id(label: str) -> int:
    if label not in LABEL_TO_ID:
        raise KeyError(f"Unknown label: {label}")
    return LABEL_TO_ID[label]


class TextOnlyDataset(Dataset):
    """텍스트 입력만 사용하는 데이터셋."""

    def __init__(self, samples: Sequence[SampleMetadata], tokenizer: PreTrainedTokenizerBase, max_length: int = 64):
        self.samples = list(samples)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        tokens = self.tokenizer(
            sample.transcript,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        item = {k: v.squeeze(0) for k, v in tokens.items()}
        item["labels"] = torch.tensor(_to_label_id(sample.label), dtype=torch.long)
        return item


class AudioOnlyDataset(Dataset):
    """eGeMAPS 특징만 사용하는 데이터셋."""

    def __init__(self, samples: Sequence[SampleMetadata], feature_cache: Path):
        self.samples = list(samples)
        self.feature_cache = feature_cache

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        features = load_audio_feature(sample, self.feature_cache)
        vector = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(_to_label_id(sample.label), dtype=torch.long)
        return {"features": vector, "labels": label}


class FusionDataset(Dataset):
    """텍스트 + 오디오 입력을 모두 제공."""

    def __init__(
        self,
        samples: Sequence[SampleMetadata],
        tokenizer: PreTrainedTokenizerBase,
        feature_cache: Path,
        max_length: int = 64,
    ):
        self.samples = list(samples)
        self.tokenizer = tokenizer
        self.feature_cache = feature_cache
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        tokens = self.tokenizer(
            sample.transcript,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        features = load_audio_feature(sample, self.feature_cache)
        audio_tensor = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(_to_label_id(sample.label), dtype=torch.long)
        item = {k: v.squeeze(0) for k, v in tokens.items()}
        item["audio_features"] = audio_tensor
        item["labels"] = label
        return item

