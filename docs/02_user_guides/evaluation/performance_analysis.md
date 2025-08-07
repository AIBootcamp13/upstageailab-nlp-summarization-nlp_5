# 📊 성능 분석 가이드

모델 성능을 체계적으로 분석하고 개선 방향을 찾는 방법을 안내합니다.

## 📈 성능 분석 개요

### 주요 평가 지표
- **ROUGE-1**: 단어 수준 유사도
- **ROUGE-2**: 2-gram 수준 유사도  
- **ROUGE-L**: 최장 공통 부분수열 기반 유사도
- **Combined F1**: 전체 ROUGE 점수의 평균

### 성능 목표
- **베이스라인**: 47.12% (ROUGE Combined F1)
- **1차 개선**: 50-52%
- **최종 목표**: 55-60%

## 🔍 성능 분석 방법

### 1. 기본 성능 평가

```python
from utils.metrics import RougeCalculator

calculator = RougeCalculator()
scores = calculator.compute_multi_reference_rouge(predictions, references_list)

print(f"ROUGE-1 F1: {scores['rouge1']['f1']:.4f}")
print(f"ROUGE-2 F1: {scores['rouge2']['f1']:.4f}")
print(f"ROUGE-L F1: {scores['rougeL']['f1']:.4f}")
print(f"Combined F1: {scores['rouge_combined_f1']:.4f}")
```

### 2. 세부 분석

#### 길이별 성능 분석
- **짧은 대화** (< 100 토큰): 높은 성능 예상
- **중간 대화** (100-300 토큰): 표준 성능
- **긴 대화** (> 300 토큰): 성능 저하 가능성

#### 주제별 성능 분석
- **일상 대화**: 기본 성능
- **비즈니스 대화**: 전문 용어로 인한 어려움
- **기술 토론**: 복잡한 개념 요약의 어려움

### 3. 오류 분석

#### 일반적인 문제점
1. **과도한 요약**: 핵심 정보 누락
2. **불충분한 요약**: 불필요한 세부사항 포함
3. **맥락 오해**: 대화 흐름 파악 실패
4. **반복 생성**: 동일한 내용 반복

#### 개선 방향
1. **데이터 품질 향상**: 더 나은 전처리
2. **모델 파라미터 조정**: 하이퍼파라미터 튜닝
3. **생성 설정 최적화**: beam search, length penalty 조정
4. **앙상블 적용**: 여러 모델 결합

## 📊 성능 모니터링

### WandB를 통한 추적

```python
import wandb

# 실험 시작
wandb.init(project="dialogue-summarization")

# 메트릭 로깅
wandb.log({
    "rouge1_f1": scores['rouge1']['f1'],
    "rouge2_f1": scores['rouge2']['f1'],
    "rougeL_f1": scores['rougeL']['f1'],
    "combined_f1": scores['rouge_combined_f1']
})
```

### 비교 분석
- **베이스라인 대비**: 개선 정도 측정
- **이전 실험 대비**: 변경사항 효과 분석
- **목표 대비**: 달성률 확인

## 🎯 성능 개선 전략

### 단계별 개선 계획

#### 1단계: 기본 최적화
- 하이퍼파라미터 튜닝
- 데이터 전처리 개선
- 생성 파라미터 조정

#### 2단계: 고급 기법 적용
- 데이터 증강
- 모델 앙상블
- Post-processing 적용

#### 3단계: 심화 연구
- 아키텍처 개선
- 새로운 평가 지표 도입
- 도메인 특화 기법

### 성능 병목 해결

#### 메모리 최적화
```python
# 배치 크기 조정
BATCH_SIZE = 8  # GPU 메모리에 맞게 조정

# Gradient Accumulation
GRADIENT_ACCUMULATION_STEPS = 4
```

#### 속도 최적화
```python
# FP16 사용
training_args.fp16 = True

# DataLoader 최적화
training_args.dataloader_num_workers = 4
```

## 📈 결과 해석

### 성능 지표 해석
- **ROUGE-1 > 0.4**: 단어 수준 매칭 양호
- **ROUGE-2 > 0.25**: 구문 수준 이해 양호
- **ROUGE-L > 0.35**: 구조적 유사성 양호

### 성능 향상 신호
- **지속적인 상승**: 학습이 잘 진행됨
- **플래토 상태**: 추가 최적화 필요
- **성능 저하**: 과적합 또는 설정 문제

## 🔧 문제 해결

### 성능이 낮은 경우
1. **데이터 품질 확인**: 전처리 재검토
2. **모델 설정 점검**: 하이퍼파라미터 조정
3. **학습 과정 분석**: 로그 및 그래프 확인

### 성능이 정체된 경우
1. **학습률 조정**: 더 작은 학습률 시도
2. **정규화 강화**: Dropout, Weight decay 조정
3. **데이터 증강**: 더 다양한 학습 데이터

## 🔗 관련 자료

- [ROUGE 메트릭 상세](./rouge_metrics.md)
- [하이퍼파라미터 튜닝](../model_training/hyperparameter_tuning.md)
- [실험 추적](../experiment_management/wandb_tracking.md)
- [문제 해결](../../06_troubleshooting/README.md)

---

효과적인 성능 분석을 통해 모델의 강점과 약점을 파악하고 체계적으로 개선해 나가세요.
