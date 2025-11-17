"""데이터셋 로딩 및 전처리 유틸리티."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from .constants import EMOTION_ID_TO_LABEL, STATEMENT_ID_TO_TEXT


@dataclass
class SampleMetadata:
    """모든 모달리티에서 재사용할 표준 메타데이터."""

    audio_path: Path
    transcript: str
    label: str
    speaker_id: str
    statement_id: int


def parse_ravdess_filename(file_path: Path) -> Tuple[int, int]:
    """RAVDESS 파일명에서 (emotion_id, statement_id)를 추출."""

    parts = file_path.stem.split("-")
    if len(parts) != 7:
        raise ValueError(f"Unexpected RAVDESS filename: {file_path.name}")
    emotion_id = int(parts[2])
    statement_id = int(parts[4])
    return emotion_id, statement_id


def collect_ravdess_samples(root: Path) -> List[SampleMetadata]:
    """RAVDESS 데이터셋을 순회하며 메타데이터를 구축."""

    samples: List[SampleMetadata] = []
    for wav_path in sorted(root.rglob("*.wav")):
        try:
            emotion_id, statement_id = parse_ravdess_filename(wav_path)
        except ValueError:
            continue
        if emotion_id not in EMOTION_ID_TO_LABEL or statement_id not in STATEMENT_ID_TO_TEXT:
            continue
        label = EMOTION_ID_TO_LABEL[emotion_id]
        transcript = STATEMENT_ID_TO_TEXT[statement_id]
        speaker_id = wav_path.parent.name
        samples.append(
            SampleMetadata(
                audio_path=wav_path,
                transcript=transcript,
                label=label,
                speaker_id=speaker_id,
                statement_id=statement_id,
            )
        )
    return samples


def split_train_valid_test(samples: Sequence[SampleMetadata], seed: int = 42) -> Dict[str, List[SampleMetadata]]:
    """스피커 단위로 데이터를 8/1/1 비율로 분할."""

    rng = random.Random(seed)
    speakers = sorted({sample.speaker_id for sample in samples})
    rng.shuffle(speakers)
    n_total = len(speakers)
    n_train = int(n_total * 0.8)
    n_valid = int(n_total * 0.1)
    splits = {
        "train": speakers[:n_train],
        "valid": speakers[n_train : n_train + n_valid],
        "test": speakers[n_train + n_valid :],
    }
    partition: Dict[str, List[SampleMetadata]] = {k: [] for k in splits}
    for sample in samples:
        for split_name, speaker_ids in splits.items():
            if sample.speaker_id in speaker_ids:
                partition[split_name].append(sample)
                break
    return partition


def write_metadata_table(samples: Sequence[SampleMetadata], output_path: Path) -> None:
    """샘플 메타 정보를 CSV/Parquet으로 덤프."""

    df = pd.DataFrame(
        [
            {
                "audio_path": str(sample.audio_path),
                "transcript": sample.transcript,
                "label": sample.label,
                "speaker_id": sample.speaker_id,
            }
            for sample in samples
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)


def save_split_manifest(splits: Dict[str, List[SampleMetadata]], output_dir: Path) -> None:
    """각 분할을 JSONL 형태로 저장."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, subset in splits.items():
        manifest_path = output_dir / f"{split}.jsonl"
        with manifest_path.open("w", encoding="utf-8") as fp:
            for sample in subset:
                fp.write(
                    json.dumps(
                        {
                            "audio_path": str(sample.audio_path),
                            "transcript": sample.transcript,
                            "label": sample.label,
                            "speaker_id": sample.speaker_id,
                        }
                    )
                    + "\n"
                )


def load_manifest(manifest_path: Path) -> List[SampleMetadata]:
    """JSONL manifest를 SampleMetadata 목록으로 로드."""

    samples: List[SampleMetadata] = []
    with manifest_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            record = json.loads(line)
            samples.append(
                SampleMetadata(
                    audio_path=Path(record["audio_path"]),
                    transcript=record["transcript"],
                    label=record["label"],
                    speaker_id=record["speaker_id"],
                    statement_id=0,
                )
            )
    return samples

