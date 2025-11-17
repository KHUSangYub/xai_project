"""Typer 기반 CLI 진입점."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch
import typer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .ambiguous_analysis import analyze_cases, load_cases
from .constants import DEFAULT_TEXT_MODEL_NAME
from .data_utils import collect_ravdess_samples, load_manifest, save_split_manifest, split_train_valid_test
from .datasets import AudioOnlyDataset, FusionDataset, TextOnlyDataset
from .evaluation import evaluate_model, save_report
from .explainability import (
    generate_audio_shap_report,
    generate_text_lime_report,
    generate_text_shap_report,
    plot_attention_heatmap,
)
from .feature_extractor import build_audio_feature_cache, load_audio_feature, load_feature_names
from .models import AudioClassifier, FusionClassifier, TextClassifier
from .prosody import derive_feature_groups, ensure_feature_names
from .training import train_model
from .masking import evaluate_with_mask

app = typer.Typer(add_completion=False, help="멀티모달 감정 분석 CLI")

FEATURE_CACHE_DIR = Path("artifacts/features/eGeMAPS")
MANIFEST_DIR_DEFAULT = Path("artifacts/manifests")


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_checkpoint(model, model_dir: Path) -> None:
    checkpoint_path = model_dir / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{checkpoint_path} 파일이 없습니다. 먼저 학습을 수행하세요.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)


def _serialize_eval(stats: Dict[str, object]) -> Dict[str, object]:
    return {
        "confusion_matrix": stats["confusion_matrix"].tolist(),
        "classification_report": stats["classification_report"],
    }


@app.command("prepare-data")
def prepare_data(
    dataset_root: Path = typer.Argument(..., exists=True, file_okay=False),
    manifest_dir: Path = typer.Option(Path("artifacts/manifests")),
    feature_cache: Path = typer.Option(Path("artifacts/features/eGeMAPS")),
) -> None:
    """RAVDESS 데이터를 스캔하고 manifest 및 오디오 특징 캐시를 생성."""

    typer.echo("RAVDESS 샘플 수집 중...")
    samples = collect_ravdess_samples(dataset_root)
    splits = split_train_valid_test(samples)
    typer.echo("Manifest 저장 중...")
    save_split_manifest(splits, manifest_dir)
    typer.echo("오디오 특징 캐시 생성 중...")
    build_audio_feature_cache(splits["train"] + splits["valid"] + splits["test"], feature_cache)
    typer.echo("준비 완료!")


def _build_dataloaders(manifest_dir: Path, tokenizer_name: str, feature_cache: Path, batch_size: int):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    splits = {split: load_manifest(manifest_dir / f"{split}.jsonl") for split in ["train", "valid", "test"]}
    text_sets = {name: TextOnlyDataset(samples, tokenizer) for name, samples in splits.items()}
    audio_sets = {name: AudioOnlyDataset(samples, feature_cache) for name, samples in splits.items()}
    fusion_sets = {name: FusionDataset(samples, tokenizer, feature_cache) for name, samples in splits.items()}
    loaders = {
        "text": {k: DataLoader(ds, batch_size=batch_size, shuffle=(k == "train")) for k, ds in text_sets.items()},
        "audio": {k: DataLoader(ds, batch_size=batch_size, shuffle=(k == "train")) for k, ds in audio_sets.items()},
        "fusion": {k: DataLoader(ds, batch_size=batch_size, shuffle=(k == "train")) for k, ds in fusion_sets.items()},
        "tokenizer": tokenizer,
    }
    return loaders


@app.command("train-text")
def train_text(
    manifest_dir: Path = typer.Option(MANIFEST_DIR_DEFAULT),
    model_name: str = typer.Option(DEFAULT_TEXT_MODEL_NAME),
    output_dir: Path = typer.Option(Path("artifacts/text_model")),
    batch_size: int = typer.Option(16),
    num_epochs: int = typer.Option(5),
    lr: float = typer.Option(3e-5),
    weight_decay: float = typer.Option(1e-2),
    encoder_checkpoint: Path | None = typer.Option(
        None,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="이미 감정으로 미세조정된 BERT encoder 가중치(.pt)를 불러옵니다.",
    ),
) -> None:
    """텍스트 전용 BERT 분류기 학습."""

    loaders = _build_dataloaders(manifest_dir, model_name, FEATURE_CACHE_DIR, batch_size)
    model = TextClassifier(model_name=model_name, freeze_encoder=True, encoder_checkpoint=encoder_checkpoint)
    result = train_model(
        model=model,
        train_loader=loaders["text"]["train"],
        valid_loader=loaders["text"]["valid"],
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        device=_device(),
        output_dir=output_dir,
        forward_kwargs_fn=lambda batch: {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]},
    )
    typer.echo(f"텍스트 모델 학습 완료 (best={result.best_val_accuracy:.3f})")


@app.command("train-audio")
def train_audio(
    manifest_dir: Path = typer.Option(MANIFEST_DIR_DEFAULT),
    output_dir: Path = typer.Option(Path("artifacts/audio_model")),
    batch_size: int = typer.Option(32),
    num_epochs: int = typer.Option(20),
    lr: float = typer.Option(1e-3),
    weight_decay: float = typer.Option(1e-4),
) -> None:
    """오디오 전용 MLP 학습."""

    loaders = _build_dataloaders(manifest_dir, DEFAULT_TEXT_MODEL_NAME, FEATURE_CACHE_DIR, batch_size)
    model = AudioClassifier()
    result = train_model(
        model=model,
        train_loader=loaders["audio"]["train"],
        valid_loader=loaders["audio"]["valid"],
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        device=_device(),
        output_dir=output_dir,
        forward_kwargs_fn=lambda batch: {"features": batch["features"]},
    )
    typer.echo(f"오디오 모델 학습 완료 (best={result.best_val_accuracy:.3f})")


@app.command("train-fusion")
def train_fusion(
    manifest_dir: Path = typer.Option(MANIFEST_DIR_DEFAULT),
    model_name: str = typer.Option(DEFAULT_TEXT_MODEL_NAME),
    output_dir: Path = typer.Option(Path("artifacts/fusion_model")),
    batch_size: int = typer.Option(16),
    num_epochs: int = typer.Option(10),
    lr: float = typer.Option(1e-4),
    weight_decay: float = typer.Option(1e-4),
) -> None:
    """텍스트+오디오 융합 모델 학습."""

    loaders = _build_dataloaders(manifest_dir, model_name, FEATURE_CACHE_DIR, batch_size)
    model = FusionClassifier(text_model_name=model_name, freeze_text_encoder=True)
    result = train_model(
        model=model,
        train_loader=loaders["fusion"]["train"],
        valid_loader=loaders["fusion"]["valid"],
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        device=_device(),
        output_dir=output_dir,
        forward_kwargs_fn=lambda batch: {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "audio_features": batch["audio_features"],
        },
    )
    typer.echo(f"융합 모델 학습 완료 (best={result.best_val_accuracy:.3f})")


@app.command("evaluate")
def evaluate_models(
    manifest_dir: Path = typer.Option(MANIFEST_DIR_DEFAULT),
    model_dir: Path = typer.Option(Path("artifacts/fusion_model")),
    output_csv: Path = typer.Option(Path("artifacts/reports/fusion_report.csv")),
) -> None:
    """테스트 셋 평가 및 리포트 저장."""

    loaders = _build_dataloaders(manifest_dir, DEFAULT_TEXT_MODEL_NAME, FEATURE_CACHE_DIR, batch_size=32)
    model = FusionClassifier()
    _load_checkpoint(model, model_dir)
    device = _device()
    model.to(device)
    stats = evaluate_model(
        model,
        loaders["fusion"]["test"],
        device=device,
        forward_kwargs_fn=lambda batch: {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "audio_features": batch["audio_features"],
        },
    )
    save_report(stats["classification_report"], output_csv)
    typer.echo(f"리포트를 {output_csv} 에 저장했습니다.")


@app.command("baseline-report")
def baseline_report(
    manifest_dir: Path = typer.Option(MANIFEST_DIR_DEFAULT),
    text_model_dir: Path = typer.Option(Path("artifacts/text_model")),
    audio_model_dir: Path = typer.Option(Path("artifacts/audio_model")),
    fusion_model_dir: Path = typer.Option(Path("artifacts/fusion_model")),
    output_path: Path = typer.Option(Path("artifacts/reports/baseline_summary.json")),
    batch_size: int = typer.Option(32),
) -> None:
    """텍스트/오디오/Fusion 모델 성능을 한 번에 요약."""

    loaders = _build_dataloaders(manifest_dir, DEFAULT_TEXT_MODEL_NAME, FEATURE_CACHE_DIR, batch_size)
    device = _device()
    summaries: Dict[str, Dict[str, object]] = {}

    if (text_model_dir / "best.pt").exists():
        text_model = TextClassifier()
        _load_checkpoint(text_model, text_model_dir)
        text_model.to(device)
        stats = evaluate_model(
            text_model,
            loaders["text"]["test"],
            device=device,
            forward_kwargs_fn=lambda batch: {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]},
        )
        summaries["text"] = _serialize_eval(stats)

    if (audio_model_dir / "best.pt").exists():
        audio_model = AudioClassifier()
        _load_checkpoint(audio_model, audio_model_dir)
        audio_model.to(device)
        stats = evaluate_model(
            audio_model,
            loaders["audio"]["test"],
            device=device,
            forward_kwargs_fn=lambda batch: {"features": batch["features"]},
        )
        summaries["audio"] = _serialize_eval(stats)

    if (fusion_model_dir / "best.pt").exists():
        fusion_model = FusionClassifier()
        _load_checkpoint(fusion_model, fusion_model_dir)
        fusion_model.to(device)
        stats = evaluate_model(
            fusion_model,
            loaders["fusion"]["test"],
            device=device,
            forward_kwargs_fn=lambda batch: {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "audio_features": batch["audio_features"],
            },
        )
        summaries["fusion"] = _serialize_eval(stats)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    typer.echo(f"Baseline 요약을 {output_path} 에 저장했습니다.")


@app.command("masking-study")
def masking_study(
    manifest_dir: Path = typer.Option(MANIFEST_DIR_DEFAULT),
    model_dir: Path = typer.Option(Path("artifacts/fusion_model")),
    batch_size: int = typer.Option(32),
) -> None:
    """Pitch/Energy 등 마스킹 실험."""

    loaders = _build_dataloaders(manifest_dir, DEFAULT_TEXT_MODEL_NAME, FEATURE_CACHE_DIR, batch_size)
    model = FusionClassifier()
    _load_checkpoint(model, model_dir)
    device = _device()
    model.to(device)
    feature_names = ensure_feature_names(FEATURE_CACHE_DIR)
    mask_groups = derive_feature_groups(feature_names)
    result = evaluate_with_mask(
        model,
        loaders["fusion"]["test"],
        device=device,
        mask_groups=mask_groups,
    )
    typer.echo(f"Baseline: {result.baseline_accuracy:.3f}")
    for name, score in result.masked_accuracy.items():
        typer.echo(f"Masked({name}): {score:.3f}")


@app.command("ambiguous-study")
def ambiguous_study(
    config_path: Path = typer.Option(Path("configs/ambiguous_cases.json")),
    dataset_root: Path = typer.Option(Path("Audio_Speech_Actors_01-24")),
    feature_cache: Path = typer.Option(FEATURE_CACHE_DIR),
    text_model_dir: Path = typer.Option(Path("artifacts/text_model")),
    audio_model_dir: Path = typer.Option(Path("artifacts/audio_model")),
    fusion_model_dir: Path = typer.Option(Path("artifacts/fusion_model")),
    output_path: Path = typer.Option(Path("artifacts/reports/ambiguous_report.json")),
) -> None:
    """동일 문장/다른 억양 시나리오 비교."""

    cases = load_cases(config_path, dataset_root)
    if not cases:
        typer.echo("config에 정의된 케이스가 없습니다.")
        raise typer.Exit(code=1)

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TEXT_MODEL_NAME)
    feature_names = ensure_feature_names(feature_cache)
    mask_groups = derive_feature_groups(feature_names)
    device = _device()

    text_model = TextClassifier()
    audio_model = AudioClassifier()
    fusion_model = FusionClassifier()
    _load_checkpoint(text_model, text_model_dir)
    _load_checkpoint(audio_model, audio_model_dir)
    _load_checkpoint(fusion_model, fusion_model_dir)
    text_model.to(device)
    audio_model.to(device)
    fusion_model.to(device)

    result = analyze_cases(
        cases,
        text_model=text_model,
        audio_model=audio_model,
        fusion_model=fusion_model,
        tokenizer=tokenizer,
        cache_dir=feature_cache,
        mask_groups=mask_groups,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.cases, indent=2), encoding="utf-8")
    typer.echo(f"Ambiguous 실험 결과를 {output_path} 에 저장했습니다.")


@app.command("xai-report")
def xai_report(
    manifest_dir: Path = typer.Option(MANIFEST_DIR_DEFAULT),
    feature_cache: Path = typer.Option(FEATURE_CACHE_DIR),
    text_model_dir: Path = typer.Option(Path("artifacts/text_model")),
    audio_model_dir: Path = typer.Option(Path("artifacts/audio_model")),
    output_dir: Path = typer.Option(Path("artifacts/reports/xai")),
    sample_count: int = typer.Option(3, help="텍스트 예시 개수"),
) -> None:
    """SHAP/LIME/Attention 및 오디오 feature 중요도 리포트."""

    samples = load_manifest(manifest_dir / "valid.jsonl")
    if not samples:
        typer.echo("유효한 manifest가 없습니다. prepare-data를 먼저 실행하세요.")
        raise typer.Exit(code=1)
    texts = [sample.transcript for sample in samples[:sample_count]]
    device = _device()
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TEXT_MODEL_NAME)

    text_model = TextClassifier()
    _load_checkpoint(text_model, text_model_dir)
    text_model.to(device)
    generate_text_shap_report(text_model, tokenizer, texts, output_dir / "text_shap", device)
    generate_text_lime_report(text_model, tokenizer, texts, output_dir / "text_lime", device)
    plot_attention_heatmap(text_model, tokenizer, texts[0], output_dir / "attention_heatmap.png")

    audio_model = AudioClassifier()
    _load_checkpoint(audio_model, audio_model_dir)
    audio_model.to(device)
    feature_names = load_feature_names(feature_cache)
    feature_vectors = [load_audio_feature(sample, feature_cache) for sample in samples[: max(64, sample_count * 4)]]
    if len(feature_vectors) < 2:
        typer.echo("오디오 feature 수가 충분하지 않습니다.")
        raise typer.Exit(code=1)
    background = torch.tensor(feature_vectors[: min(32, len(feature_vectors))], dtype=torch.float32)
    target = torch.tensor(feature_vectors[: min(16, len(feature_vectors))], dtype=torch.float32)
    generate_audio_shap_report(
        audio_model,
        background=background,
        samples=target,
        feature_names=feature_names,
        output_dir=output_dir / "audio_shap",
        device=device,
    )
    typer.echo(f"XAI 리포트를 {output_dir} 에 생성했습니다.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

