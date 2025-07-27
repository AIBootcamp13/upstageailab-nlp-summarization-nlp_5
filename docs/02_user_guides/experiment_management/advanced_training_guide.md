# 고급 실험 관리 시스템 사용 가이드

본 가이드는 `trainer.py`에 구현된 고급 실험 관리 기능들을 활용하여 체계적이고 효율적인 모델 학습을 수행하는 방법을 설명합니다.

## 목차

1. [시스템 개요](#시스템-개요)
2. [실험 추적 시스템](#실험-추적-시스템)
3. [모델 등록 시스템](#모델-등록-시스템)
4. [커스텀 콜백 시스템](#커스텀-콜백-시스템)
5. [동적 설정 병합](#동적-설정-병합)
6. [WandB 통합](#wandb-통합)
7. [실습 예제](#실습-예제)
8. [Best Practices](#best-practices)

## 시스템 개요

trainer.py는 NLP 대화 요약 모델 학습을 위한 종합적인 실험 관리 시스템을 제공합니다. 주요 특징:

- **완전 자동화된 실험 추적**: 실험 시작부터 종료까지 모든 단계 자동 기록
- **지능형 모델 등록**: 성능 기반 모델 버전 관리 및 비교
- **WandB Sweep 완전 통합**: 하이퍼파라미터 최적화 자동화
- **다중 디바이스 최적화**: CUDA/MPS/CPU 자동 감지 및 최적화
- **메모리 최적화**: 그래디언트 체크포인팅, 혼합 정밀도 등

### 핵심 컴포넌트

```python
from trainer import DialogueSummarizationTrainer, create_trainer

# 기본 트레이너 생성
trainer = create_trainer("config/base_config.yaml")

# Sweep 모드 트레이너 생성
sweep_trainer = create_trainer("config/sweep_config.yaml", sweep_mode=True)
```

## 실험 추적 시스템

### 자동 실험 추적

모든 실험은 `ExperimentTracker`에 의해 자동으로 추적됩니다:

```python
# 실험 자동 시작 (trainer.train() 호출 시)
experiment_id = tracker.start_experiment(
    name="bart_base_experiment",
    description="KoBART 기본 실험",
    config=config,
    model_type="kobart",
    dataset_info={
        'train_size': 1000,
        'val_size': 200
    },
    wandb_run_id=wandb.run.id
)
```

### 실험 데이터 구조

각 실험은 다음 정보를 자동으로 기록합니다:

```json
{
    "experiment_id": "exp_20250128_001",
    "name": "bart_base_experiment",
    "start_time": "2025-01-28T10:30:00Z",
    "end_time": "2025-01-28T12:45:00Z",
    "status": "completed",
    "config": {
        "model": {"architecture": "kobart"},
        "training": {"learning_rate": 5e-5}
    },
    "metrics": {
        "best_rouge_combined_f1": 0.47,
        "final_loss": 1.23
    },
    "model_info": {
        "model_id": "model_20250128_001",
        "checkpoint_path": "/models/best_model"
    },
    "wandb_run_id": "wandb_run_123",
    "notes": "실험 완료, 좋은 성능"
}
```

### 실험 상태 관리

```python
# 수동으로 실험 상태 업데이트
tracker.update_experiment_status(experiment_id, "in_progress", "학습 진행 중")

# 메트릭 추가 로깅
tracker.log_metrics({
    'learning_rate': 5e-5,
    'batch_size': 16,
    'custom_metric': 0.85
}, step=100)

# 실험 완료
tracker.end_experiment(
    experiment_id=experiment_id,
    final_metrics=final_results,
    best_metrics=best_results,
    status="completed"
)
```

## 모델 등록 시스템

### 자동 모델 등록

성공적으로 학습된 모델은 `ModelRegistry`에 자동 등록됩니다:

```python
# 학습 완료 시 자동 실행
model_id = registry.register_model(
    name="kobart_base_v1",
    architecture="kobart",
    checkpoint="gogamza/kobart-base-v2",
    config=config,
    performance={
        'rouge1_f1': 0.45,
        'rouge2_f1': 0.32,
        'rougeL_f1': 0.41,
        'rouge_combined_f1': 0.39
    },
    training_info={
        'epochs': 3,
        'batch_size': 16,
        'learning_rate': 5e-5
    },
    file_path="/models/kobart_base_v1",
    experiment_id=experiment_id
)
```

### 모델 검색 및 비교

```python
# 최고 성능 모델 찾기
best_models = registry.get_best_models(
    metric='rouge_combined_f1',
    limit=5
)

# 특정 아키텍처 모델 검색
kobart_models = registry.search_models(
    architecture='kobart',
    min_performance={'rouge_combined_f1': 0.35}
)

# 모델 상세 정보 조회
model_info = registry.get_model_info(model_id)
print(f"모델 성능: {model_info['performance']}")
print(f"학습 설정: {model_info['training_info']}")
```

### 모델 버전 관리

```python
# 동일 이름 모델의 새 버전 등록
model_id_v2 = registry.register_model(
    name="kobart_base_v2",  # 버전 명시
    architecture="kobart",
    # ... 기타 정보
)

# 모델 비교
comparison = registry.compare_models([model_id, model_id_v2])
print("성능 비교:")
for metric, values in comparison.items():
    print(f"  {metric}: v1={values[0]:.4f}, v2={values[1]:.4f}")
```

## 커스텀 콜백 시스템

### WandbCallback 기능

`WandbCallback`은 학습 중 실시간 모니터링을 제공합니다:

```python
class WandbCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # ROUGE 점수 결합 계산
        rouge_combined = (
            metrics.get('eval_rouge1', 0) * 0.33 +
            metrics.get('eval_rouge2', 0) * 0.33 +
            metrics.get('eval_rougeL', 0) * 0.34
        )
        
        # WandB 로깅
        log_metrics = {
            'eval/rouge1_f1': metrics.get('eval_rouge1', 0),
            'eval/rouge2_f1': metrics.get('eval_rouge2', 0),
            'eval/rougeL_f1': metrics.get('eval_rougeL', 0),
            'eval/rouge_combined_f1': rouge_combined,
            'eval/loss': metrics.get('eval_loss', 0),
            'epoch': state.epoch,
            'step': state.global_step
        }
        
        wandb.log(log_metrics)
```

### 베스트 메트릭 추적

```python
# 콜백에서 자동으로 최고 성능 추적
if rouge_combined > self.best_metrics.get('rouge_combined_f1', 0):
    self.best_metrics = {
        'rouge1_f1': metrics.get('eval_rouge1', 0),
        'rouge2_f1': metrics.get('eval_rouge2', 0),
        'rougeL_f1': metrics.get('eval_rougeL', 0),
        'rouge_combined_f1': rouge_combined,
        'loss': metrics.get('eval_loss', 0)
    }
```

### 커스텀 콜백 추가

필요에 따라 추가 콜백을 구현할 수 있습니다:

```python
class CustomMetricCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # 커스텀 메트릭 계산
        custom_score = self.calculate_custom_metric(metrics)
        
        # 로깅
        if wandb.run:
            wandb.log({'custom_score': custom_score})
    
    def calculate_custom_metric(self, metrics):
        # 도메인 특화 메트릭 계산 로직
        return (metrics.get('eval_rouge1', 0) + 
                metrics.get('eval_rouge2', 0) * 2) / 3

# 트레이너에 콜백 추가
trainer.trainer.add_callback(CustomMetricCallback())
```

## 동적 설정 병합

### Sweep 파라미터 동적 병합

WandB Sweep 실행 시 파라미터가 기본 설정과 자동으로 병합됩니다:

```python
# ConfigManager의 동적 병합 기능
config_manager = ConfigManager()
config = config_manager.load_config("base_config.yaml")

# Sweep 파라미터 자동 병합 (sweep 모드에서)
if wandb.run and wandb.config:
    merged_config = config_manager.merge_sweep_params(config, wandb.config)
```

### 설정 오버라이드

환경변수를 통한 설정 오버라이드:

```bash
# 환경변수로 설정 오버라이드
export LEARNING_RATE=3e-5
export BATCH_SIZE=32
export NUM_EPOCHS=5

python trainer.py --config config/base_config.yaml
```

```python
# ConfigManager가 자동으로 환경변수 적용
config = config_manager.load_config("base_config.yaml")
# LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS 자동 적용됨
```

### 실험별 설정 조정

```python
# 런타임에 설정 수정
config['training']['learning_rate'] = 1e-4
config['training']['per_device_train_batch_size'] = 8

# 수정된 설정으로 트레이너 생성
trainer = DialogueSummarizationTrainer(config)
```

## WandB 통합

### 기본 WandB 설정

```python
import wandb

# 프로젝트 초기화
wandb.init(
    project="nlp-dialogue-summarization",
    name="kobart_experiment_v1",
    config=config,
    tags=["kobart", "baseline", "v1"]
)

# 트레이너 생성 (WandB 자동 연동)
trainer = create_trainer(config)
```

### Sweep 설정 및 실행

1. **Sweep 설정 파일 작성** (`config/sweep/kobart_sweep.yaml`):

```yaml
program: trainer.py
method: bayes
metric:
  name: rouge_combined_f1
  goal: maximize
parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 1e-6
    max: 1e-3
  per_device_train_batch_size:
    values: [8, 16, 32]
  num_train_epochs:
    values: [2, 3, 4, 5]
  weight_decay:
    distribution: uniform
    min: 0.01
    max: 0.1
```

2. **Sweep 실행**:

```bash
# Sweep 생성
wandb sweep config/sweep/kobart_sweep.yaml

# Agent 실행
wandb agent your-project/sweep-id
```

3. **코드에서 Sweep 파라미터 사용**:

```python
def train_with_sweep():
    # WandB sweep 파라미터 자동 병합
    wandb.init()
    
    # Sweep 모드로 트레이너 생성
    trainer = create_trainer("config/base_config.yaml", sweep_mode=True)
    
    # 데이터 준비 및 학습
    datasets = trainer.prepare_data()
    result = trainer.train(datasets)
    
    # Sweep 결과 기록
    wandb.log(result.best_metrics)
```

### 고급 WandB 기능

```python
# 모델 아티팩트 저장
artifact = wandb.Artifact(
    name="kobart_model_v1",
    type="model",
    description="훈련된 KoBART 모델"
)
artifact.add_dir(str(trainer.model_save_dir))
wandb.log_artifact(artifact)

# 예측 샘플 시각화
predictions = trainer.generate_predictions(test_dataset, max_samples=10)
table = wandb.Table(columns=["Input", "Prediction", "Reference"])

for pred in predictions[:10]:
    table.add_data(
        pred['input'][:100] + "...",
        pred['prediction'],
        pred['reference']
    )

wandb.log({"prediction_samples": table})

# 학습 곡선 시각화
wandb.log({"learning_curves": wandb.plot.line_series(
    xs=epochs,
    ys=[train_losses, val_losses],
    keys=["train_loss", "val_loss"],
    title="Training Progress",
    xname="Epoch"
)})
```

## 실습 예제

### 예제 1: 기본 실험 실행

```python
"""
기본 KoBART 모델로 실험 실행
"""
import wandb
from trainer import create_trainer

# WandB 초기화
wandb.init(
    project="dialogue-summarization",
    name="kobart_baseline",
    tags=["baseline", "kobart"]
)

# 트레이너 생성
trainer = create_trainer("config/models/kobart_base.yaml")

# 데이터 준비
datasets = trainer.prepare_data(
    train_path="data/train.csv",
    val_path="data/validation.csv"
)

# 학습 실행
result = trainer.train(datasets)

# 결과 출력
print(f"Best ROUGE Combined F1: {result.best_metrics['rouge_combined_f1']:.4f}")
print(f"Model saved to: {result.model_path}")
```

### 예제 2: 하이퍼파라미터 Sweep

```python
"""
하이퍼파라미터 최적화 실험
"""
import wandb
from trainer import create_trainer

def sweep_train():
    # Sweep 모드로 WandB 초기화
    wandb.init()
    
    # Sweep 파라미터를 적용한 트레이너 생성
    trainer = create_trainer("config/base_config.yaml", sweep_mode=True)
    
    # 데이터 준비
    datasets = trainer.prepare_data()
    
    # 학습 실행
    result = trainer.train(datasets)
    
    # Sweep 목표 메트릭 기록
    wandb.log({
        "rouge_combined_f1": result.best_metrics['rouge_combined_f1']
    })

# Sweep 설정으로 실행
if __name__ == "__main__":
    sweep_train()
```

### 예제 3: 커스텀 실험 설정

```python
"""
커스텀 설정으로 고급 실험 실행
"""
import wandb
from trainer import DialogueSummarizationTrainer
from utils.config_manager import ConfigManager

# 설정 로딩 및 커스터마이징
config_manager = ConfigManager()
config = config_manager.load_config("config/base_config.yaml")

# 실험별 설정 조정
config['training']['learning_rate'] = 3e-5
config['training']['num_train_epochs'] = 5
config['training']['gradient_accumulation_steps'] = 4
config['meta']['experiment_name'] = "kobart_optimized_v1"

# WandB 초기화
wandb.init(
    project="dialogue-summarization",
    name=config['meta']['experiment_name'],
    config=config,
    tags=["optimized", "kobart", "v1"]
)

# 커스텀 트레이너 생성
trainer = DialogueSummarizationTrainer(
    config=config,
    experiment_name="kobart_optimized_v1"
)

# 컴포넌트 초기화
trainer.initialize_components()

# 데이터 준비
datasets = trainer.prepare_data()

# 학습 실행
result = trainer.train(datasets)

# 테스트 평가 (옵션)
if 'test' in datasets:
    test_results = trainer.evaluate(datasets['test'], metric_key_prefix="test")
    print(f"Test Results: {test_results}")

# 예측 샘플 생성
predictions = trainer.generate_predictions(datasets['validation'], max_samples=5)
for i, pred in enumerate(predictions):
    print(f"\n=== 예측 {i+1} ===")
    print(f"입력: {pred['input'][:100]}...")
    print(f"예측: {pred['prediction']}")
    print(f"참조: {pred['reference']}")
```

### 예제 4: 모델 비교 실험

```python
"""
여러 모델 아키텍처 비교 실험
"""
from trainer import create_trainer
from utils.experiment_utils import ModelRegistry
import wandb

# 실험할 모델 목록
models = [
    ("config/models/kobart_base.yaml", "kobart_base"),
    ("config/models/kogpt2_medium.yaml", "kogpt2_medium"),
    ("config/models/mt5_small.yaml", "mt5_small")
]

results = []

for config_path, model_name in models:
    print(f"\n=== {model_name} 실험 시작 ===")
    
    # WandB 실행 초기화
    wandb.init(
        project="model-comparison",
        name=f"comparison_{model_name}",
        tags=["comparison", model_name],
        reinit=True
    )
    
    # 트레이너 생성
    trainer = create_trainer(config_path)
    
    # 데이터 준비
    datasets = trainer.prepare_data()
    
    # 학습 실행
    result = trainer.train(datasets)
    
    results.append({
        'model': model_name,
        'rouge_combined_f1': result.best_metrics['rouge_combined_f1'],
        'rouge1_f1': result.best_metrics['rouge1_f1'],
        'rouge2_f1': result.best_metrics['rouge2_f1'],
        'rougeL_f1': result.best_metrics['rougeL_f1'],
        'model_path': result.model_path
    })
    
    wandb.finish()

# 결과 비교
print("\n=== 모델 비교 결과 ===")
results.sort(key=lambda x: x['rouge_combined_f1'], reverse=True)

for i, result in enumerate(results):
    print(f"{i+1}. {result['model']}")
    print(f"   ROUGE Combined F1: {result['rouge_combined_f1']:.4f}")
    print(f"   ROUGE-1 F1: {result['rouge1_f1']:.4f}")
    print(f"   ROUGE-2 F1: {result['rouge2_f1']:.4f}")
    print(f"   ROUGE-L F1: {result['rougeL_f1']:.4f}")
    print(f"   Model Path: {result['model_path']}")
    print()

# 최고 성능 모델 정보
best_model = results[0]
print(f"최고 성능 모델: {best_model['model']}")
print(f"성능: {best_model['rouge_combined_f1']:.4f}")
```

## Best Practices

### 1. 실험 명명 규칙

```python
# 추천 명명 규칙
experiment_name = f"{model_arch}_{dataset_version}_{optimization_type}_{timestamp}"
# 예: "kobart_v2_baseline_20250128_001"

# WandB 태그 활용
tags = [
    model_arch,        # "kobart", "kogpt2"
    experiment_type,   # "baseline", "optimized", "ablation"
    dataset_version,   # "v1", "v2", "augmented"
    optimization      # "sweep", "manual", "auto"
]
```

### 2. 설정 관리

```python
# 실험별 설정 파일 분리
configs = {
    'baseline': 'config/experiments/baseline.yaml',
    'optimized': 'config/experiments/optimized.yaml',
    'ablation': 'config/experiments/ablation.yaml'
}

# 환경별 설정 오버라이드
if os.getenv('ENVIRONMENT') == 'production':
    config['training']['fp16'] = True
    config['training']['dataloader_num_workers'] = 8
```

### 3. 리소스 최적화

```python
# 디바이스별 최적화
if torch.cuda.is_available():
    config['training']['per_device_train_batch_size'] = 32
    config['training']['fp16'] = True
elif torch.backends.mps.is_available():
    config['training']['per_device_train_batch_size'] = 16
    config['training']['fp16'] = False  # MPS는 fp16 제한적 지원
else:
    config['training']['per_device_train_batch_size'] = 4
    config['training']['gradient_accumulation_steps'] = 8
```

### 4. 실험 결과 관리

```python
# 실험 결과 체계적 저장
def save_experiment_summary(result, config, experiment_name):
    summary = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'config_hash': hashlib.md5(str(config).encode()).hexdigest(),
        'best_metrics': result.best_metrics,
        'model_info': {
            'architecture': config['model']['architecture'],
            'checkpoint': config['model']['checkpoint'],
            'path': result.model_path
        },
        'training_info': {
            'epochs': config['training']['num_train_epochs'],
            'batch_size': config['training']['per_device_train_batch_size'],
            'learning_rate': config['training']['learning_rate']
        }
    }
    
    # JSON 저장
    with open(f"experiments/{experiment_name}_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
```

### 5. 재현성 보장

```python
# 시드 고정
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 실험 시작 시 시드 설정
set_seed(config['general']['seed'])

# 설정에 시드 정보 포함
config['reproducibility'] = {
    'seed': 42,
    'torch_version': torch.__version__,
    'transformers_version': transformers.__version__
}
```

### 6. 에러 처리 및 복구

```python
# 체크포인트로부터 복구
try:
    result = trainer.train(datasets)
except Exception as e:
    logger.error(f"Training failed: {e}")
    
    # 최신 체크포인트 찾기
    checkpoint_dir = Path(trainer.output_dir) / "checkpoints"
    checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
    
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split('-')[1]))
        logger.info(f"Resuming from {latest_checkpoint}")
        
        result = trainer.train(datasets, resume_from_checkpoint=str(latest_checkpoint))
    else:
        raise
```

이 가이드를 통해 trainer.py의 고급 실험 관리 기능을 효과적으로 활용하여 체계적인 모델 개발이 가능합니다.
