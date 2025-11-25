# 💡 Deepfake Detection Project

AIFactory × 국립과학수사연구원(NFS)  
**「딥페이크 범죄 대응을 위한 AI 탐지 모델 경진대회」** 참여 프로젝트입니다.

본 프로젝트는 이미지·영상 기반 딥페이크 판별을 위해  
**scSE CNN 이미지 모델**과 **TimeSformer 영상 Transformer 모델**을 개발하고,  
성능 개선을 위한 여러 실험 및 앙상블 기법을 적용한 프로젝트입니다.

데이터 준비 → 모델 설계 → 학습 → 성능 개선 → 앙상블 → 제출  
전체 AI 모델 개발 파이프라인을 체계적으로 구현하였습니다.

---

## 📌 Competition Information
- **대회명:** 딥페이크 범죄 대응을 위한 AI 탐지 모델 경진대회  
- **주최/주관:** 국립과학수사연구원(NFS)  
- **플랫폼:** AI Factory  
- **목표:** 이미지·영상 기반 딥페이크 판별 모델 개발

---

## 🔍 Project Overview
본 프로젝트에서는 다음 두 가지 모델을 기반으로 다양한 실험을 진행하였습니다:

- **scSE 기반 이미지 분류 모델**  
- **TimeSformer 기반 영상 분류 모델**  
- 두 모델을 결합한 **Soft Voting Ensemble** 적용  

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

## Performance & Model Development History

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
