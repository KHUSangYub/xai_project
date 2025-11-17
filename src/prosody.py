"""eGeMAPS feature 메타데이터와 prosody 그룹 유틸리티."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .feature_extractor import load_feature_names

GROUP_KEYWORDS = {
    "pitch": ["f0", "pitch"],
    "energy": ["loudness", "energy", "pcm", "shimmerLocaldB"],
    "jitter": ["jitter"],
    "shimmer": ["shimmer"],
}


def derive_feature_groups(feature_names: Sequence[str]) -> Dict[str, List[int]]:
    """feature 이름 기반으로 pitch/energy/jitter/shimmer 인덱스 그룹을 추론."""

    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, name in enumerate(feature_names):
        lowered = name.lower()
        for group, keywords in GROUP_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                groups[group].append(idx)
    # 비어있는 그룹 제거
    return {k: v for k, v in groups.items() if v}


def group_feature_importance(
    feature_names: Sequence[str],
    importance: Sequence[float],
    groups: Dict[str, Sequence[int]] | None = None,
) -> Dict[str, float]:
    """각 prosody 그룹별 중요도 합을 계산."""

    groups = groups or derive_feature_groups(feature_names)
    result: Dict[str, float] = {}
    importance = np.asarray(importance)
    for group, indices in groups.items():
        result[group] = float(np.sum(np.abs(importance[np.asarray(indices)])))
    return result


def ensure_feature_names(cache_dir: Path) -> List[str]:
    """캐시 디렉토리에서 feature 이름을 읽고, 없으면 에러."""

    return load_feature_names(cache_dir)

