"""텍스트, 오디오, 융합 모델 정의."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from transformers import AutoModel

from .constants import DEFAULT_TEXT_MODEL_NAME, ID_TO_LABEL


class TextClassifier(nn.Module):
    """감정으로 이미 미세조정된 BERT를 불러와 마지막 Linear만 새로 학습."""

    def __init__(
        self,
        model_name: str = DEFAULT_TEXT_MODEL_NAME,
        num_classes: int | None = None,
        freeze_encoder: bool = True,
        encoder_checkpoint: Path | None = None,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        if encoder_checkpoint:
            state = torch.load(encoder_checkpoint, map_location="cpu")
            if isinstance(state, dict):
                if "model_state" in state:
                    state = state["model_state"]
                elif "state_dict" in state:
                    state = state["state_dict"]
            missing, unexpected = self.encoder.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(f"[TextClassifier] missing={missing}, unexpected={unexpected}")
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        hidden_size = self.encoder.config.hidden_size
        num_classes = num_classes or len(ID_TO_LABEL)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls)
        return logits


class AudioClassifier(nn.Module):
    """eGeMAPS 특징 전용 분류기."""

    def __init__(self, input_dim: int = 88, num_classes: int | None = None) -> None:
        super().__init__()
        num_classes = num_classes or len(ID_TO_LABEL)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class FusionClassifier(nn.Module):
    """텍스트 CLS 임베딩과 오디오 특징을 결합."""

    def __init__(
        self,
        text_model_name: str = DEFAULT_TEXT_MODEL_NAME,
        audio_dim: int = 88,
        hidden_dim: int = 256,
        num_classes: int | None = None,
        freeze_text_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        text_dim = self.text_encoder.config.hidden_size
        num_classes = num_classes or len(ID_TO_LABEL)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(text_dim + audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = text_outputs.last_hidden_state[:, 0]
        fusion = torch.cat([cls, audio_features], dim=-1)
        logits = self.fusion_mlp(fusion)
        return logits

