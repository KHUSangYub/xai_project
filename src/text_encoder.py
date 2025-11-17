"""BERT 텍스트 임베딩 도우미."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from transformers import AutoModel, AutoTokenizer

from .constants import DEFAULT_TEXT_MODEL_NAME


@dataclass
class TextEncoderOutput:
    """텍스트 임베딩 결과."""

    embeddings: torch.Tensor
    attention_mask: torch.Tensor


class TextEncoder:
    """사전 학습된 BERT를 활용한 텍스트 인코더."""

    def __init__(self, model_name: str = DEFAULT_TEXT_MODEL_NAME, device: str | None = None) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: list[str], max_length: int = 64) -> TextEncoderOutput:
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        with torch.no_grad():
            outputs = self.model(**tokens)
            cls_embeddings = outputs.last_hidden_state[:, 0]
        return TextEncoderOutput(embeddings=cls_embeddings, attention_mask=tokens["attention_mask"])

    def freeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False


def load_tokenizer(model_name: str = DEFAULT_TEXT_MODEL_NAME) -> AutoTokenizer:
    """토크나이저만 필요할 때 헬퍼."""

    return AutoTokenizer.from_pretrained(model_name)

