"""모호한 문장 실험 스크립트."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from .data_utils import SampleMetadata
from .feature_extractor import EGeMAPSExtractor, ensure_audio_feature
from .prosody import derive_feature_groups


@dataclass
class AudioVariant:
    """동일 문장의 서로 다른 억양(감정) 오디오."""

    path: Path
    tag: str


@dataclass
class AmbiguousCase:
    """모호한 문장 한 건."""

    case_id: str
    text: str
    audio_variants: List[AudioVariant] = field(default_factory=list)


@dataclass
class VariantResult:
    """한 오디오 변형에 대한 예측."""

    tag: str
    audio_path: str
    audio_only: List[float]
    fusion: List[float]
    fusion_masked: Dict[str, List[float]]


@dataclass
class AmbiguousResult:
    """모든 모호한 문장 결과."""

    cases: List[Dict]


def load_cases(config_path: Path, dataset_root: Path) -> List[AmbiguousCase]:
    """JSON config를 읽어 모호한 문장 목록을 생성."""

    with config_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    cases: List[AmbiguousCase] = []
    for item in payload:
        variants = [
            AudioVariant(path=(dataset_root / variant["path"]).resolve(), tag=variant["tag"])
            for variant in item.get("audio_variants", [])
        ]
        cases.append(AmbiguousCase(case_id=item["id"], text=item["text"], audio_variants=variants))
    return cases


def analyze_cases(
    cases: Sequence[AmbiguousCase],
    text_model,
    audio_model,
    fusion_model,
    tokenizer,
    cache_dir: Path,
    mask_groups: Dict[str, Sequence[int]],
) -> AmbiguousResult:
    """텍스트-only, 오디오-only, Fusion(+masking) 비교."""

    device = next(fusion_model.parameters()).device
    text_model.to(device).eval()
    audio_model.to(device).eval()
    fusion_model.to(device).eval()

    extractor = EGeMAPSExtractor()
    results: List[Dict] = []

    def predict_text(sentence: str) -> List[float]:
        tokens = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            probs = torch.softmax(text_model(**tokens), dim=-1)[0].cpu().tolist()
        return probs

    def predict_audio(features: torch.Tensor) -> List[float]:
        with torch.no_grad():
            probs = torch.softmax(audio_model(features), dim=-1)[0].cpu().tolist()
        return probs

    def predict_fusion(sentence: str, audio_tensor: torch.Tensor) -> List[float]:
        tokens = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            probs = torch.softmax(
                fusion_model(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                    audio_features=audio_tensor,
                ),
                dim=-1,
            )[0].cpu().tolist()
        return probs

    for case in cases:
        text_scores = predict_text(case.text)
        variant_rows: List[Dict] = []
        for variant in case.audio_variants:
            sample = SampleMetadata(
                audio_path=variant.path,
                transcript=case.text,
                label="unknown",
                speaker_id=variant.path.parent.name,
                statement_id=0,
            )
            vector = ensure_audio_feature(sample, cache_dir, extractor=extractor)
            audio_tensor = torch.tensor(vector, dtype=torch.float32, device=device).unsqueeze(0)
            audio_scores = predict_audio(audio_tensor)
            fusion_scores = predict_fusion(case.text, audio_tensor)
            masked_scores = {}
            for group_name, indices in mask_groups.items():
                masked_tensor = audio_tensor.clone()
                masked_tensor[:, indices] = 0.0
                masked_scores[group_name] = predict_fusion(case.text, masked_tensor)
            variant_rows.append(
                VariantResult(
                    tag=variant.tag,
                    audio_path=str(variant.path),
                    audio_only=audio_scores,
                    fusion=fusion_scores,
                    fusion_masked=masked_scores,
                ).__dict__
            )
        results.append({"case_id": case.case_id, "text": case.text, "text_only": text_scores, "variants": variant_rows})
    return AmbiguousResult(cases=results)

