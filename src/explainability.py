"""SHAP/LIME 및 Attention 기반 XAI 분석."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
import torch
from lime.lime_text import LimeTextExplainer

from .constants import EMOTION_ORDER
from .prosody import group_feature_importance


def _text_predict_fn(model, tokenizer, device: str):
    def predict(texts: List[str]) -> np.ndarray:
        tokenized = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        model.eval()
        with torch.no_grad():
            logits = model(**tokenized)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    return predict


def generate_text_shap_report(model, tokenizer, texts: List[str], output_dir: Path, device: str) -> None:
    """텍스트 입력에 대한 SHAP 시각화를 HTML로 저장."""

    output_dir.mkdir(parents=True, exist_ok=True)
    predict = _text_predict_fn(model, tokenizer, device)
    masker = shap.maskers.Text(tokenizer=tokenizer)
    explainer = shap.Explainer(predict, masker)
    shap_values = explainer(texts)
    for idx, text in enumerate(texts):
        html_path = output_dir / f"text_shap_{idx:02d}.html"
        shap.save_html(str(html_path), shap.plots.text(shap_values[idx], display=False))


def generate_text_lime_report(model, tokenizer, texts: List[str], output_dir: Path, device: str, num_features: int = 8) -> None:
    """LIME 기반 텍스트 중요도 막대 그래프 저장."""

    output_dir.mkdir(parents=True, exist_ok=True)
    predict = _text_predict_fn(model, tokenizer, device)
    explainer = LimeTextExplainer(class_names=EMOTION_ORDER)
    for idx, text in enumerate(texts):
        explanation = explainer.explain_instance(text, predict, num_features=num_features)
        fig = explanation.as_pyplot_figure()
        fig.suptitle(f"LIME Tokens #{idx}", fontsize=12)
        fig.savefig(output_dir / f"text_lime_{idx:02d}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_attention_heatmap(model, tokenizer, text: str, output_path: Path, layer: int = -1, head: int | None = None) -> None:
    """BERT self-attention 행렬을 시각화."""

    device = next(model.parameters()).device
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = model.encoder(
        input_ids=tokens["input_ids"],
        attention_mask=tokens["attention_mask"],
        output_attentions=True,
    )
    attentions = outputs.attentions[layer]  # [batch, heads, seq, seq]
    if head is None:
        attn_matrix = attentions.mean(dim=1)[0].detach().cpu().numpy()
    else:
        attn_matrix = attentions[0, head].detach().cpu().numpy()
    token_labels = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
    plt.figure(figsize=(8, 6))
    sns.heatmap(attn_matrix, xticklabels=token_labels, yticklabels=token_labels, cmap="magma", square=True)
    plt.title(f"Self-attention Layer {layer}{' Head '+str(head) if head is not None else ' (avg)'}")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def generate_audio_shap_report(
    audio_model,
    background: torch.Tensor,
    samples: torch.Tensor,
    feature_names: Sequence[str],
    output_dir: Path,
    device: str,
) -> None:
    """eGeMAPS 각 feature의 shap 중요도 및 그룹별 합계를 시각화."""

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_model.to(device).eval()
    background = background.to(device).requires_grad_(True)
    samples = samples.to(device).requires_grad_(True)
    explainer = shap.GradientExplainer(audio_model, background)
    shap_values = explainer.shap_values(samples)
    shap_array = np.mean(np.abs(np.stack(shap_values, axis=0)), axis=0)  # (samples, features)
    importance = shap_array.mean(axis=0)
    top_indices = np.argsort(importance)[::-1][:20]
    plt.figure(figsize=(8, 6))
    labels = [feature_names[i] for i in top_indices]
    plt.barh(labels[::-1], importance[top_indices][::-1])
    plt.title("Top eGeMAPS features by SHAP")
    plt.tight_layout()
    plt.savefig(output_dir / "audio_shap_top_features.png", dpi=200)
    plt.close()

    group_scores = group_feature_importance(feature_names, importance)
    plt.figure(figsize=(6, 4))
    plt.bar(group_scores.keys(), group_scores.values())
    plt.title("Prosody group importance (SHAP)")
    plt.tight_layout()
    plt.savefig(output_dir / "audio_shap_groups.png", dpi=200)
    plt.close()

