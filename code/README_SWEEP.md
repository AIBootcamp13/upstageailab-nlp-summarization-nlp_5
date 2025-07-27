# WandB Sweep 통합 및 자동화 시스템

## 개요

이 시스템은 6조의 CV 프로젝트에서 영감을 받아 NLP 대화 요약 프로젝트에 맞게 구현한 실험 자동화 시스템입니다. WandB Sweep을 활용하여 체계적인 하이퍼파라미터 최적화를 수행할 수 있습니다.

## 주요 구성 요소

### 1. trainer.py
- baseline.ipynb의 핵심 학습 로직을 모듈화
- WandB와 완전히 통합된 학습 파이프라인
- 다양한 모델 아키텍처 지원 (KoBART, KoGPT2, T5, mT5)
- ROUGE 메트릭 기반 평가 및 최적화

### 2. sweep_runner.py
- WandB Sweep 실행 및 관리
- 간단한 YAML 설정 로딩 및 동적 파라미터 병합
- 실험 결과 자동 저장 및 분석

### 3. parallel_sweep_runner.py
- 병렬 Sweep 실행 지원
- 여러 Sweep 설정을 순차/병렬로 실행
- 실험 효율성 극대화

## 사용 방법

### 1. 단일 학습 실행

```bash
python trainer.py \
    --config config/base_config.yaml \
    --train-data ../data/train.csv \
    --val-data ../data/dev.csv
```

### 2. Sweep 실행

#### 하이퍼파라미터 튜닝
```bash
python sweep_runner.py \
    --base-config config/base_config.yaml \
    --sweep-config hyperparameter_sweep \
    --count 50
```

#### 모델 비교
```bash
python sweep_runner.py \
    --base-config config/base_config.yaml \
    --sweep-config model_comparison_sweep \
    --count 20
```

### 3. 병렬 Sweep 실행

```bash
python parallel_sweep_runner.py \
    --base-config config/base_config.yaml \
    --single-parallel hyperparameter_sweep \
    --num-workers 4 \
    --runs-per-worker 10
```

### 4. 여러 Sweep 순차 실행

```bash
python parallel_sweep_runner.py \
    --base-config config/base_config.yaml \
    --multiple-serial hyperparameter_sweep model_comparison_sweep \
    --runs-per-sweep 20
```

## 간편 실행 스크립트

scripts 폴더에 자주 사용하는 실행 스크립트들이 준비되어 있습니다:

- `run_hyperparameter_sweep.sh/bat`: 하이퍼파라미터 튜닝
- `run_model_comparison_sweep.sh/bat`: 모델 비교
- `run_parallel_sweep.sh/bat`: 병렬 실행
- `run_all_sweeps.sh/bat`: 모든 Sweep 순차 실행
- `run_single_training.bat`: 단일 학습 (Windows)

## Sweep 설정 파일

config/sweep 폴더에 다음 Sweep 설정들이 준비되어 있습니다:

1. **hyperparameter_sweep.yaml**: 학습률, 배치 크기, 에폭 수 등 최적화
2. **model_comparison_sweep.yaml**: 다양한 모델 아키텍처 비교
3. **generation_params_sweep.yaml**: 생성 파라미터 최적화
4. **ablation_study_sweep.yaml**: 구성 요소별 영향도 분석

## 결과 분석

### 실행 중 모니터링
- WandB 대시보드에서 실시간으로 실험 진행 상황 확인
- 메트릭 시각화 및 비교

### 실행 후 분석
- `outputs/sweep_*/` 폴더에 저장된 결과 확인
- `*_summary.json` 파일에서 최고 성능 파라미터 확인
- 최적 모델은 `models/best_model/` 경로에 저장

## 주요 특징

1. **기존 코드와의 호환성**: baseline.ipynb의 로직을 그대로 활용
2. **유연한 설정 관리**: YAML 기반 계층적 설정 시스템
3. **메모리 최적화**: 긴 시퀀스에 대한 자동 배치 크기 조정
4. **실험 추적**: ExperimentTracker를 통한 체계적인 실험 관리
5. **모델 레지스트리**: 학습된 모델의 버전 관리

## 요구사항

```
transformers>=4.30.0
datasets>=2.10.0
wandb>=0.15.0
torch>=2.0.0
evaluate>=0.4.0
rouge-score>=0.1.2
pyyaml>=6.0
```

## 환경 설정

1. WandB 로그인
```bash
wandb login
```

2. 환경변수 설정 (선택사항)
```bash
export WANDB_PROJECT="nlp-dialogue-summarization"
export WANDB_ENTITY="your-entity"
```

## 문제 해결

### GPU 메모리 부족
- 배치 크기 감소: `per_device_train_batch_size` 조정
- Gradient accumulation 사용: `gradient_accumulation_steps` 증가
- FP16 학습 활성화: `fp16: true`

### Sweep 실패
- 로그 파일 확인: `sweep_results/logs/` 디렉토리
- WandB 대시보드에서 에러 메시지 확인

### 느린 학습 속도
- 데이터 로더 워커 수 증가: `dataloader_num_workers`
- 병렬 Sweep 실행 사용

## 기여 가이드

1. 새로운 Sweep 설정 추가: `config/sweep/` 디렉토리에 YAML 파일 생성
2. 커스텀 메트릭 추가: `utils/metrics.py`에 구현
3. 새로운 모델 지원: `trainer.py`의 `_load_model()` 메서드 수정

## 라이선스

이 프로젝트는 팀 내부 사용을 위해 개발되었습니다.
