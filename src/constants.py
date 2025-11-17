"""프로젝트 전역 상수 및 매핑 정의."""

from __future__ import annotations

EMOTION_ID_TO_LABEL = {
    1: "neutral",
    3: "happy",
    4: "sad",
    5: "angry",
}

LABEL_TO_ID = {label: idx for idx, label in enumerate(sorted(set(EMOTION_ID_TO_LABEL.values())))}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

# RAVDESS 파일명 인덱스 → 문장 매핑
STATEMENT_ID_TO_TEXT = {
    1: "Kids are talking by the door.",
    2: "Dogs are sitting by the door.",
}

RAVDESS_SAMPLE_RATE = 48_000
DEFAULT_TEXT_MODEL_NAME = "bert-base-uncased"

EMOTION_ORDER = ["neutral", "happy", "sad", "angry"]

