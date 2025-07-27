# Trainer 모듈 API 참조

`trainer.py` 모듈은 NLP 대화 요약 모델의 학습, 평가, 추론을 담당하는 핵심 모듈입니다.

## 목차

1. [DialogueSummarizationTrainer](#dialoguesummarizationtrainer)
2. [TrainingResult](#trainingresult)
3. [WandbCallback](#wandbcallback)
4. [유틸리티 함수](#유틸리티-함수)

## DialogueSummarizationTrainer

메인 트레이너 클래스로 모델 학습의 전체 생명주기를 관리합니다.

### 클래스 정의

```python
class DialogueSummarizationTrainer:
    """
    대화 요약 모델 학습 트레이너
    
    baseline.ipynb의 학습 로직을 모듈화하고 WandB Sweep과 통합
    """
```

### 생성자

```python
def __init__(self, config: Dict[str, Any], 
             sweep_mode: bool = False,
             experiment_name: Optional[str] = None):
```

**Parameters:**
- `config` (Dict[str, Any]): 설정 딕셔너리 (ConfigManager로부터)
- `sweep_mode` (bool, optional): WandB Sweep 모드 여부. 기본값: False
- `experiment_name` (Optional[str]): 실험명. None이면 자동 생성

**Example:**
```python
from trainer import DialogueSummarizationTrainer
from utils.config_manager import ConfigManager

config_manager = ConfigManager()
config = config_manager.load_config("config/base_config.yaml")

trainer = DialogueSummarizationTrainer(
    config=config,
    sweep_mode=False,
    experiment_name="kobart_baseline_v1"
)
```

### 주요 메서드

#### initialize_components()

모든 컴포넌트를 초기화합니다.

```python
def initialize_components(self):
    """모든 컴포넌트 초기화"""
```

**Raises:**
- `RuntimeError`: 컴포넌트 초기화 실패 시

**Example:**
```python
trainer = DialogueSummarizationTrainer(config)
trainer.initialize_components()
```

#### prepare_data()

데이터를 로딩하고 전처리합니다.

```python
def prepare_data(self, train_path: Optional[str] = None, 
                val_path: Optional[str] = None,
                test_path: Optional[str] = None) -> DatasetDict:
```

**Parameters:**
- `train_path` (Optional[str]): 학습 데이터 경로
- `val_path` (Optional[str]): 검증 데이터 경로  
- `test_path` (Optional[str]): 테스트 데이터 경로

**Returns:**
- `DatasetDict`: 처리된 데이터셋 딕셔너리

**Example:**
```python
datasets = trainer.prepare_data(
    train_path="data/train.csv",
    val_path="data/validation.csv",
    test_path="data/test.csv"
)

print(f"Train dataset size: {len(datasets['train'])}")
print(f"Validation dataset size: {len(datasets['validation'])}")
```

#### train()

모델을 학습합니다.

```python
def train(self, dataset: DatasetDict, 
         resume_from_checkpoint: Optional[str] = None) -> TrainingResult:
```

**Parameters:**
- `dataset` (DatasetDict): 학습/검증 데이터셋
- `resume_from_checkpoint` (Optional[str]): 체크포인트 경로 (재개 시)

**Returns:**
- `TrainingResult`: 학습 결과

**Raises:**
- `RuntimeError`: 학습 중 오류 발생 시
- `ValueError`: 잘못된 데이터셋 형식

**Example:**
```python
# 기본 학습
result = trainer.train(datasets)

# 체크포인트에서 재개
result = trainer.train(
    datasets, 
    resume_from_checkpoint="checkpoints/checkpoint-1000"
)

print(f"Best ROUGE Combined F1: {result.best_metrics['rouge_combined_f1']:.4f}")
print(f"Model saved to: {result.model_path}")
```

#### evaluate()

모델을 평가합니다.

```python
def evaluate(self, dataset: Dataset, 
            metric_key_prefix: str = "eval") -> Dict[str, float]:
```

**Parameters:**
- `dataset` (Dataset): 평가 데이터셋
- `metric_key_prefix` (str): 메트릭 키 접두사

**Returns:**
- `Dict[str, float]`: 평가 결과

**Example:**
```python
eval_results = trainer.evaluate(datasets['validation'])
print(f"Validation ROUGE-1: {eval_results['eval_rouge1']:.4f}")

test_results = trainer.evaluate(datasets['test'], metric_key_prefix="test")
print(f"Test ROUGE-L: {test_results['test_rougeL']:.4f}")
```

#### generate_predictions()

예측을 생성합니다.

```python
def generate_predictions(self, dataset: Dataset, 
                       max_samples: Optional[int] = None) -> List[Dict[str, str]]:
```

**Parameters:**
- `dataset` (Dataset): 입력 데이터셋
- `max_samples` (Optional[int]): 최대 샘플 수 (None이면 전체)

**Returns:**
- `List[Dict[str, str]]`: 예측 결과 리스트

**Example:**
```python
# 전체 데이터셋 예측
predictions = trainer.generate_predictions(datasets['test'])

# 샘플링하여 예측
sample_predictions = trainer.generate_predictions(
    datasets['test'], 
    max_samples=10
)

for i, pred in enumerate(sample_predictions[:3]):
    print(f"\n=== 예측 {i+1} ===")
    print(f"입력: {pred['input'][:100]}...")
    print(f"예측: {pred['prediction']}")
    print(f"참조: {pred['reference']}")
```

### 속성

#### 주요 속성

```python
@property
def config(self) -> Dict[str, Any]:
    """설정 딕셔너리"""

@property  
def device(self) -> torch.device:
    """연산 디바이스"""

@property
def model(self) -> PreTrainedModel:
    """학습 모델"""

@property
def tokenizer(self) -> PreTrainedTokenizer:
    """토크나이저"""

@property
def experiment_tracker(self) -> Optional[ExperimentTracker]:
    """실험 추적기"""

@property
def model_registry(self) -> Optional[ModelRegistry]:
    """모델 등록소"""
```

**Example:**
```python
print(f"Using device: {trainer.device}")
print(f"Model architecture: {trainer.config['model']['architecture']}")
print(f"Experiment name: {trainer.experiment_name}")

# 모델 파라미터 수
total_params = sum(p.numel() for p in trainer.model.parameters())
trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
```

## TrainingResult

학습 결과를 담는 데이터 클래스입니다.

### 클래스 정의

```python
@dataclass
class TrainingResult:
    """학습 결과 데이터 클래스"""
    best_metrics: Dict[str, float]
    final_metrics: Dict[str, float]
    model_path: str
    config_used: Dict[str, Any]
    training_history: List[Dict[str, Any]] = field(default_factory=list)
    wandb_run_id: Optional[str] = None
    experiment_id: Optional[str] = None
```

### 속성

- `best_metrics` (Dict[str, float]): 최고 성능 메트릭
- `final_metrics` (Dict[str, float]): 최종 평가 메트릭
- `model_path` (str): 모델 저장 경로
- `config_used` (Dict[str, Any]): 사용된 설정
- `training_history` (List[Dict[str, Any]]): 학습 기록
- `wandb_run_id` (Optional[str]): WandB 실행 ID
- `experiment_id` (Optional[str]): 실험 ID

### 사용 예제

```python
result = trainer.train(datasets)

# 최고 성능 메트릭 확인
print("Best Metrics:")
for metric, value in result.best_metrics.items():
    print(f"  {metric}: {value:.4f}")

# 모델 로딩
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained(result.model_path)
tokenizer = AutoTokenizer.from_pretrained(result.model_path)

# WandB 실행 확인
if result.wandb_run_id:
    print(f"WandB Run: https://wandb.ai/your-entity/your-project/runs/{result.wandb_run_id}")
```

## WandbCallback

WandB 로깅을 위한 커스텀 콜백 클래스입니다.

### 클래스 정의

```python
class WandbCallback(TrainerCallback):
    """WandB 로깅을 위한 커스텀 콜백"""
```

### 생성자

```python
def __init__(self, trainer_instance):
```

**Parameters:**
- `trainer_instance`: DialogueSummarizationTrainer 인스턴스

### 주요 메서드

#### on_evaluate()

평가 시 WandB에 메트릭을 로깅합니다.

```python
def on_evaluate(self, args, state: TrainerState, control: TrainerControl, 
               metrics: Dict[str, float], **kwargs):
```

**Features:**
- ROUGE 점수 결합 계산 (가중 평균)
- 베스트 메트릭 자동 추적
- 실험 추적기 연동
- 실시간 로깅

### 베스트 메트릭 계산

WandbCallback은 다음 공식으로 ROUGE 점수를 결합합니다:

```python
rouge_combined = (
    rouge1_f1 * 0.33 +
    rouge2_f1 * 0.33 + 
    rougeL_f1 * 0.34
)
```

### 로깅되는 메트릭

- `eval/rouge1_f1`: ROUGE-1 F1 점수
- `eval/rouge2_f1`: ROUGE-2 F1 점수  
- `eval/rougeL_f1`: ROUGE-L F1 점수
- `eval/rouge_combined_f1`: 결합된 ROUGE F1 점수
- `eval/loss`: 평가 손실
- `best/rouge_combined_f1`: 현재까지 최고 성능
- `epoch`: 현재 에포크
- `step`: 현재 스텝

## 유틸리티 함수

### create_trainer()

트레이너 생성 편의 함수입니다.

```python
def create_trainer(config: Union[str, Dict[str, Any]], 
                  sweep_mode: bool = False) -> DialogueSummarizationTrainer:
```

**Parameters:**
- `config` (Union[str, Dict[str, Any]]): 설정 파일 경로 또는 설정 딕셔너리
- `sweep_mode` (bool): WandB Sweep 모드 여부

**Returns:**
- `DialogueSummarizationTrainer`: 초기화된 트레이너 인스턴스

**Example:**
```python
from trainer import create_trainer

# 설정 파일로 생성
trainer = create_trainer("config/base_config.yaml")

# 설정 딕셔너리로 생성
config = {
    'model': {'architecture': 'kobart', 'checkpoint': 'gogamza/kobart-base-v2'},
    'training': {'learning_rate': 5e-5, 'per_device_train_batch_size': 16}
}
trainer = create_trainer(config)

# Sweep 모드로 생성
sweep_trainer = create_trainer("config/sweep_config.yaml", sweep_mode=True)
```

## 에러 처리

### 일반적인 예외

```python
try:
    trainer = create_trainer("config/base_config.yaml")
    trainer.initialize_components()
    
    datasets = trainer.prepare_data()
    result = trainer.train(datasets)
    
except FileNotFoundError as e:
    print(f"설정 파일 또는 데이터 파일을 찾을 수 없습니다: {e}")
    
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("GPU 메모리 부족. 배치 크기를 줄여주세요.")
        # 배치 크기 감소 후 재시도
        trainer.config['training']['per_device_train_batch_size'] //= 2
        result = trainer.train(datasets)
    else:
        print(f"런타임 에러: {e}")
        
except ValueError as e:
    print(f"잘못된 설정값: {e}")
    
except Exception as e:
    print(f"예상치 못한 에러: {e}")
    # 에러 로깅 및 정리 작업
    if hasattr(trainer, 'experiment_tracker') and trainer.experiment_tracker:
        trainer.experiment_tracker.end_experiment(
            experiment_id=experiment_id,
            status="failed",
            notes=str(e)
        )
```

## 고급 사용법

### 커스텀 콜백 추가

```python
from transformers import TrainerCallback

class CustomMetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # 커스텀 메트릭 계산
        custom_score = metrics.get('eval_rouge1', 0) * 2
        
        if wandb.run:
            wandb.log({'custom_score': custom_score})

# 콜백 추가
trainer.initialize_components()
trainer.trainer.add_callback(CustomMetricsCallback())
```

### 동적 설정 수정

```python
# 런타임에 설정 수정
trainer.config['training']['learning_rate'] = 3e-5
trainer.config['training']['warmup_ratio'] = 0.1

# 새로운 트레이닝 아규먼트 생성
training_args = trainer._get_training_arguments()
trainer.trainer.args = training_args
```

### 모델 체크포인트 관리

```python
# 체크포인트 디렉토리 확인
checkpoint_dir = trainer.output_dir / "checkpoints"
checkpoints = list(checkpoint_dir.glob("checkpoint-*"))

if checkpoints:
    # 최신 체크포인트로 재개
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split('-')[1]))
    result = trainer.train(datasets, resume_from_checkpoint=str(latest_checkpoint))
```

---

**관련 문서:**
- [Utils API 참조](./utils_api.md)
- [Core API 참조](./core_api.md)
- [실험 관리 가이드](../02_user_guides/experiment_management/advanced_training_guide.md)
