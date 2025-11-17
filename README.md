# 멀티모달 감정 분석 (텍스트 + 음성)

## 개요

본 저장소는 RAVDESS 및 IEMOCAP을 기반으로 **억양(Prosody) 요소가 감정 판단에 미치는 영향**을 정량적으로 분석하기 위한 연구 코드 전반을 제공합니다. 텍스트(BERT)와 음성(OpenSMILE eGeMAPS)를 결합한 Early-Fusion 모델을 구축하며, baseline 비교·feature masking·ambiguous 문장 실험·XAI(SHAP/LIME)까지 재현할 수 있도록 구성했습니다.

## 디렉터리 구조

- `src/`: 핵심 파이썬 모듈 (데이터 로딩, 모델, 학습, XAI 등)
- `artifacts/`: Manifest, 캐시, 모델 체크포인트, 리포트가 저장되는 디렉터리 (실행 시 자동 생성)
- `Audio_Speech_Actors_01-24/`: RAVDESS 원본 오디오 (사용자가 별도로 준비)

## 실행 방법

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 1) 데이터 준비
python -m src.cli prepare-data /Users/sangyub/Desktop/xai_project/Audio_Speech_Actors_01-24

# 2) Baseline 학습
python -m src.cli train-text --encoder-checkpoint path/to/emotion_finetuned_encoder.pt
python -m src.cli train-audio
python -m src.cli train-fusion

# 3) 평가 및 분석
python -m src.cli baseline-report
python -m src.cli evaluate
python -m src.cli masking-study
python -m src.cli ambiguous-study --config-path configs/ambiguous_cases.json
python -m src.cli xai-report
```

## Ambiguous 문장 및 XAI

- `configs/ambiguous_cases.json`: 동일 문장이 서로 다른 감정으로 발화된 음성 목록. 필요 시 사용자 정의 ambiguous 문장/오디오를 여기에 추가하세요.
- `python -m src.cli ambiguous-study`: 텍스트-only / 오디오-only / Fusion / Prosody 마스킹 결과를 JSON 리포트로 출력.
- `python -m src.cli xai-report`: SHAP/LIME 텍스트 시각화, BERT self-attention heatmap, eGeMAPS feature 중요도 그래프를 `artifacts/reports/xai/`에 저장.

## 재현 실험

1. **텍스트-only**: `train-text`
   - 사전에 감정으로 미세조정된 BERT encoder 가중치를 `--encoder-checkpoint`로 지정하면 본문 파라미터는 그대로 두고 마지막 Linear(4-class)만 학습합니다.
2. **음성-only**: `train-audio`
3. **Fusion**: `train-fusion`
4. **Baseline 비교**: `baseline-report` (정확도/혼동행렬/리포트 일괄 저장)
5. **Feature Masking**: `masking-study` (pitch/energy/jitter/shimmer 자동 추출)
6. **Ambiguous 문장**: `ambiguous-study` (config 기반 실험)
7. **XAI**: `xai-report` (SHAP/LIME/Attention + eGeMAPS 중요도)

각 스크립트는 모두 모듈화되어 있어 Jupyter Notebook, Lightning, Hydra 등 다른 워크플로우로도 손쉽게 이식 가능합니다.

