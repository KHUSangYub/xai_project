"""오디오 특징 추출 및 캐싱 로직."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import joblib
import numpy as np
import opensmile

from .data_utils import SampleMetadata

FEATURE_NAME_FILE = "feature_names.json"


class EGeMAPSExtractor:
    """OpenSMILE eGeMAPS 특징 추출기."""

    def __init__(self) -> None:
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        self._feature_names: list[str] | None = None

    def __call__(self, audio_path: Path) -> np.ndarray:
        features = self.smile.process_file(str(audio_path))
        if self._feature_names is None:
            self._feature_names = features.columns.tolist()
        return features.to_numpy().squeeze()

    @property
    def feature_names(self) -> list[str]:
        if self._feature_names is None:
            raise ValueError("추출기를 최소 한 번 호출해야 feature_names를 알 수 있습니다.")
        return self._feature_names


def _cache_key(sample: SampleMetadata) -> str:
    md5 = hashlib.md5(str(sample.audio_path).encode("utf-8")).hexdigest()  # noqa: S324
    return md5


def build_audio_feature_cache(
    samples: Iterable[SampleMetadata],
    cache_dir: Path,
    extractor: EGeMAPSExtractor | None = None,
) -> Path:
    """샘플별 eGeMAPS 벡터를 캐싱."""

    extractor = extractor or EGeMAPSExtractor()
    cache_dir.mkdir(parents=True, exist_ok=True)
    index: Dict[str, str] = {}
    feature_name_path = cache_dir / FEATURE_NAME_FILE
    for sample in samples:
        key = _cache_key(sample)
        cache_path = cache_dir / f"{key}.joblib"
        if not cache_path.exists():
            vector = extractor(sample.audio_path)
            joblib.dump(vector, cache_path)
        elif extractor._feature_names is None:
            # 캐시에 vector만 있는 경우라도 feature 이름 확보를 위해 한 번 로드
            vector = extractor(sample.audio_path)
            joblib.dump(vector, cache_path)
        index[key] = str(sample.audio_path)
        if not feature_name_path.exists() and extractor._feature_names is not None:
            _save_feature_names(feature_name_path, extractor.feature_names)
    index_path = cache_dir / "index.json"
    with index_path.open("w", encoding="utf-8") as fp:
        json.dump(index, fp, indent=2)
    return cache_dir


def load_audio_feature(sample: SampleMetadata, cache_dir: Path) -> np.ndarray:
    """캐시에서 단일 샘플 특징을 로드."""

    key = _cache_key(sample)
    cache_path = cache_dir / f"{key}.joblib"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing cached feature for {sample.audio_path}")
    return joblib.load(cache_path)


def batch_load_features(samples: Iterable[SampleMetadata], cache_dir: Path) -> List[np.ndarray]:
    """여러 샘플 특징을 순차 로드."""

    return [load_audio_feature(sample, cache_dir) for sample in samples]


def _save_feature_names(path: Path, names: Sequence[str]) -> None:
    path.write_text(json.dumps(list(names), indent=2), encoding="utf-8")


def load_feature_names(cache_dir: Path) -> List[str]:
    """캐시 디렉터리에 저장된 eGeMAPS feature 이름을 로드."""

    path = cache_dir / FEATURE_NAME_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Feature name file이 {path} 에 존재하지 않습니다. "
            "build-audio-feature-cache를 다시 실행해 주세요."
        )
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def ensure_audio_feature(
    sample: SampleMetadata,
    cache_dir: Path,
    extractor: EGeMAPSExtractor | None = None,
) -> np.ndarray:
    """해당 샘플의 특징을 캐시에 보장 후 반환."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(sample)
    cache_path = cache_dir / f"{key}.joblib"
    extractor = extractor or EGeMAPSExtractor()
    if not cache_path.exists():
        vector = extractor(sample.audio_path)
        joblib.dump(vector, cache_path)
        feature_name_path = cache_dir / FEATURE_NAME_FILE
        if not feature_name_path.exists() and extractor._feature_names is not None:
            _save_feature_names(feature_name_path, extractor.feature_names)
    return joblib.load(cache_path)

