# Hyperparameter Tuning 가이드

## 목차
1. [하이퍼파라미터 튜닝이란?](#1-하이퍼파라미터-튜닝이란)
2. [주요 하이퍼파라미터 종류](#2-주요-하이퍼파라미터-종류)
3. [Seq2SeqTrainingArguments 하이퍼파라미터 상세](#3-seq2seqtrainingarguments-하이퍼파라미터-상세)
4. [하이퍼파라미터 최적화 알고리즘](#4-하이퍼파라미터-최적화-알고리즘)
5. [Optuna를 활용한 실전 튜닝](#5-optuna를-활용한-실전-튜닝)
6. [프로젝트 적용 방법](#6-프로젝트-적용-방법)

---

## 1. 하이퍼파라미터 튜닝이란?

하이퍼파라미터는 모델이 학습하면서 자체적으로 최적화하는 것이 아닌, **사용자가 직접 지정해야 하는 설정값**입니다. 최적의 하이퍼파라미터 조합을 찾는 과정을 하이퍼파라미터 튜닝이라고 합니다.

### 왜 중요한가?
- 모델의 **성능을 크게 좌우**
- 같은 모델이라도 하이퍼파라미터에 따라 **10-20% 성능 차이** 발생
- 과적합/과소적합 방지
- 학습 속도 및 안정성 개선

## 2. 주요 하이퍼파라미터 종류

### 2.1 신경망 구조 관련
- **Hidden Layer 개수**: 네트워크의 깊이
- **Dropout 비율**: 과적합 방지
- **Activation Function**: ReLU, GELU, Tanh 등
- **Weight Initialization**: Xavier, He 초기화 등

### 2.2 학습 알고리즘 관련
- **Epoch**: 전체 데이터셋 학습 횟수
- **Batch Size**: 한 번에 처리하는 데이터 개수
- **Learning Rate**: 가중치 업데이트 크기
- **Optimizer**: Adam, AdamW, SGD 등
- **Momentum**: 이전 gradient의 영향력

## 3. Seq2SeqTrainingArguments 하이퍼파라미터 상세

### 3.1 최적화 관련

#### optim (default: adamw_torch)
```python
# 주요 optimizer 옵션
optim_choices = [
    'adamw_torch',      # PyTorch AdamW (권장)
    'adamw_hf',         # HuggingFace AdamW
    'adamw_apex_fused', # APEX fused AdamW (빠름)
    'sgd',              # Stochastic Gradient Descent
    'adafactor'         # 메모리 효율적
]
```

#### learning_rate (default: 5e-05)
- **작은 값 (1e-5)**: 안정적이지만 느린 학습
- **큰 값 (1e-3)**: 빠르지만 불안정할 수 있음
- **권장 범위**: 1e-5 ~ 5e-4

### 3.2 배치 관련

#### per_device_train_batch_size (default: 8)
```python
# GPU 메모리에 따른 권장 배치 크기
# V100 16GB: 16-32
# A100 40GB: 32-64
# RTX 3090 24GB: 24-48
```

#### gradient_accumulation_steps (default: 1)
- 실제 배치 크기 = per_device_train_batch_size × gradient_accumulation_steps
- 메모리 부족 시 이 값을 늘려 큰 배치 효과 달성

### 3.3 학습 스케줄링

#### num_train_epochs (default: 20)
- 대화 요약 태스크: 10-30 epochs 권장
- Early Stopping과 함께 사용 권장

#### warmup_ratio (default: 0.0)
- 학습 초기 learning rate를 점진적으로 증가
- 권장값: 0.05-0.1 (전체 스텝의 5-10%)

#### lr_scheduler_type (default: linear)
```python
scheduler_types = [
    'linear',        # 선형 감소
    'cosine',        # 코사인 감소
    'cosine_with_restarts',  # 주기적 재시작
    'polynomial',    # 다항식 감소
    'constant'       # 고정값
]
```

### 3.4 성능 최적화

#### fp16 / bf16 (default: False)
```python
# FP16 (Half Precision)
fp16=True  # 2배 빠른 속도, 메모리 50% 절약

# BF16 (Brain Float 16)
bf16=True  # FP16보다 안정적, A100에서 권장
```

## 4. 하이퍼파라미터 최적화 알고리즘

### 4.1 Grid Search
```python
# 예시: 모든 조합 탐색
param_grid = {
    'learning_rate': [1e-5, 3e-5, 5e-5],
    'batch_size': [16, 32],
    'epochs': [10, 20, 30]
}
# 총 실험 수: 3 × 2 × 3 = 18
```

**장점**: 
- 직관적이고 구현 쉬움
- 병렬 실행 가능

**단점**: 
- 계산 비용 높음 (지수적 증가)
- 비효율적

### 4.2 Random Search
```python
# 예시: 무작위 샘플링
param_distributions = {
    'learning_rate': uniform(1e-5, 5e-4),
    'batch_size': choice([16, 32, 64]),
    'epochs': randint(10, 50)
}
```

**장점**: 
- Grid Search보다 효율적
- 넓은 범위 탐색 가능

**단점**: 
- 최적값 보장 못함
- 이전 결과 활용 안함

### 4.3 Bayesian Optimization
- 이전 실험 결과를 바탕으로 다음 실험 지점 선택
- 가장 효율적이지만 구현 복잡

![하이퍼파라미터 최적화 비교](https://lh7-us.googleusercontent.com/U3nuIAXtFqub3svMzwqc9uAE4ljdeqUrYB2U8sqS8Is2CsD1vU_V1jEtYFXZqH1-aBXOh_zQzFFyvsJMGtAHz7ohOGm6cK6InZbPoSi6e-Yc8EJzWUstWt7NN3eSI3w9XDxBV_sSHzyk)

## 5. Optuna를 활용한 실전 튜닝

### 5.1 설치
```bash
pip install optuna
```

### 5.2 기본 설정
```python
import optuna
from transformers import Trainer, TrainingArguments

# 평가 메트릭 정의
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # ROUGE 점수 계산
    return {"rouge_score": calculate_rouge(predictions, labels)}

# 모델 초기화 함수
def model_init(trial):
    return AutoModelForSeq2SeqLM.from_pretrained("gogamza/kobart-base-v2")
```

### 5.3 하이퍼파라미터 공간 정의
```python
def optuna_hp_space(trial):
    return {
        # Optimizer 선택
        "optim": trial.suggest_categorical(
            name='optimizer',
            choices=['adamw_torch', 'adamw_hf', 'adafactor']
        ),
        
        # Learning Rate (로그 스케일)
        "learning_rate": trial.suggest_loguniform(
            name='learning_rate', 
            low=1e-5, 
            high=5e-4
        ),
        
        # Batch Size
        "per_device_train_batch_size": trial.suggest_categorical(
            name='batch_size', 
            choices=[8, 16, 32]
        ),
        
        # Epochs
        "num_train_epochs": trial.suggest_int(
            name='epoch', 
            low=10, 
            high=30, 
            step=5
        ),
        
        # Gradient Accumulation
        "gradient_accumulation_steps": trial.suggest_int(
            name='grad_accum', 
            low=1, 
            high=4, 
            step=1
        ),
        
        # Warmup Ratio
        "warmup_ratio": trial.suggest_float(
            name='warmup_ratio', 
            low=0.0, 
            high=0.2, 
            step=0.05
        ),
        
        # LR Scheduler
        "lr_scheduler_type": trial.suggest_categorical(
            name='lr_scheduler',
            choices=['linear', 'cosine', 'cosine_with_restarts']
        ),
        
        # Mixed Precision
        "fp16": trial.suggest_categorical(
            name='fp16', 
            choices=[True, False]
        ),
    }
```

### 5.4 하이퍼파라미터 탐색 실행
```python
# Trainer 초기화
trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Optuna 탐색 실행
best_trials = trainer.hyperparameter_search(
    direction="maximize",          # ROUGE 점수 최대화
    backend="optuna",             # Optuna 사용
    hp_space=optuna_hp_space,     # 탐색 공간
    n_trials=30,                  # 실험 횟수
    pruner=optuna.pruners.MedianPruner()  # 조기 종료
)

# 최적 하이퍼파라미터 출력
print("Best hyperparameters:", best_trials.hyperparameters)
```

## 6. 프로젝트 적용 방법

### 6.1 기존 baseline.ipynb 수정

```python
# 기존 코드
training_args = Seq2SeqTrainingArguments(
    output_dir=f"{output_path}/{model_name.split('/')[-1]}",
    num_train_epochs=20,
    learning_rate=1e-05,
    # ... 기타 설정
)

# Optuna 적용 코드
def create_training_args(trial=None):
    if trial:
        # Optuna 탐색 중
        hp = optuna_hp_space(trial)
    else:
        # 기본값 또는 최적값 사용
        hp = {
            'learning_rate': 3e-5,
            'per_device_train_batch_size': 16,
            'num_train_epochs': 20,
            # ... 최적 하이퍼파라미터
        }
    
    return Seq2SeqTrainingArguments(
        output_dir=f"{output_path}/{model_name.split('/')[-1]}",
        **hp
    )
```

### 6.2 실험 관리 통합

```python
# WandB + Optuna 통합
def objective(trial):
    # WandB 초기화
    wandb.init(
        project="dialogue-summarization",
        name=f"trial_{trial.number}",
        config=trial.params,
        group="optuna_search",
        reinit=True
    )
    
    # 학습 실행
    trainer = create_trainer(trial)
    metrics = trainer.train()
    
    # 결과 기록
    wandb.log({"rouge_score": metrics['eval_rouge']})
    wandb.finish()
    
    return metrics['eval_rouge']

# Optuna 스터디 생성
study = optuna.create_study(
    direction="maximize",
    study_name="dialogue_summarization_hp_search"
)

# 탐색 실행
study.optimize(objective, n_trials=30)
```

### 6.3 권장 하이퍼파라미터 세트

#### 빠른 실험용 (Quick)
```yaml
learning_rate: 5e-5
per_device_train_batch_size: 32
num_train_epochs: 10
warmup_ratio: 0.1
fp16: true
```

#### 균형잡힌 설정 (Balanced)
```yaml
learning_rate: 3e-5
per_device_train_batch_size: 16
num_train_epochs: 20
warmup_ratio: 0.1
gradient_accumulation_steps: 2
lr_scheduler_type: cosine
fp16: true
```

#### 최고 성능용 (Best)
```yaml
learning_rate: 2e-5
per_device_train_batch_size: 8
num_train_epochs: 30
warmup_ratio: 0.15
gradient_accumulation_steps: 4
lr_scheduler_type: cosine_with_restarts
fp16: true
early_stopping_patience: 5
```

### 6.4 실전 팁

1. **단계적 접근**
   - 먼저 learning_rate와 batch_size 최적화
   - 그 다음 epochs와 scheduler 조정
   - 마지막으로 세부 파라미터 튜닝

2. **효율적 탐색**
   - 초기 10-15 trials로 대략적 범위 파악
   - 좋은 범위에서 추가 15-20 trials 집중 탐색

3. **과적합 방지**
   - validation loss 모니터링
   - early stopping 활용
   - dropout, weight decay 조정

4. **재현성 보장**
   - seed 고정
   - 최적 하이퍼파라미터 저장
   - 실험 로그 상세 기록

## 결론

하이퍼파라미터 튜닝은 모델 성능을 크게 향상시킬 수 있는 중요한 과정입니다. Optuna와 WandB를 활용하면 체계적이고 효율적인 튜닝이 가능합니다. 대화 요약 태스크의 경우, learning_rate (2e-5 ~ 5e-5), batch_size (16-32), epochs (15-25) 범위에서 시작하는 것을 권장합니다.
