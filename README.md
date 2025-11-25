# Deepfake Detection Project

Computer Vision 기반 Deepfake 영상/이미지 분류 프로젝트입니다.\
본 프로젝트에서는 **scSE CNN 기반 이미지 모델**과 **TimeSformer
Transformer 기반 영상 모델**을 개발하여 Deepfake 여부를 판별합니다.\
프로젝트는 모델 개발 → 평가 → 앙상블 → 최종 제출까지의 과정을 체계적으로
담고 있습니다.

------------------------------------------------------------------------

## 📁 Project Structure

    project/
    │
    ├── data/                          # 데이터 폴더 (Git에는 비어 있음)
    │
    ├── model/
    │   ├── scse/
    │   │   └── scse_model.py          # scSE 모델 정의
    │   ├── timesformer/
    │   │   └── timesformer_model.py   # TimeSformer 모델 정의
    │   └── weights/
    │       ├── scse_best.pth          # scSE 학습 가중치
    │       └── timesformer_best.pth   # TimeSformer 학습 가중치
    │
    ├── train/
    │   ├── train_scse.ipynb           # 이미지 기반 scSE 학습 코드
    │   └── train_timesformer.ipynb    # 영상 기반 TimeSformer 학습 코드
    │
    ├── task.ipynb                     # 최종 추론 및 submission.csv 생성 스크립트
    │
    └── requirements.txt

------------------------------------------------------------------------

## 🏆 Performance & Model Development History

본 프로젝트는 여러 단계의 실험을 거쳐 성능을 개선하였습니다.

### 1) scSE Model 단독 제출 (초기 단계)

-   이미지 기반 CNN 모델
-   초기 제출 성능: **F1-score ≈ 0.3440**

------------------------------------------------------------------------

### 2) TimeSformer 모델 추가 (학습 없이)

-   영상 기반 Transformer 모델(TimeSformer)을 추가만 하고 학습 없이 실행
-   성능: **F1-score ≈ 0.30**

------------------------------------------------------------------------

### 3) TimeSformer 학습 후 성능 개선

-   RunPod GPU 환경에서 직접 학습 진행
-   성능: **F1-score ≈ 0.48**

------------------------------------------------------------------------

### 4) scSE + TimeSformer Ensemble (최종 제출)

-   이미지(scSE) + 영상(TimeSformer) 결합 Soft Voting
-   전처리 강화
-   최종 제출 성능: **F1-score ≈ 0.4940**

------------------------------------------------------------------------
