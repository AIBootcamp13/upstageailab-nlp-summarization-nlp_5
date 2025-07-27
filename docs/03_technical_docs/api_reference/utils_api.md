# Utils 모듈 API 참조

`utils` 모듈은 프로젝트 전반에서 사용되는 공통 유틸리티 기능들을 제공합니다.

## 목차

1. [ConfigManager](#configmanager) - 설정 관리
2. [DataProcessor](#dataprocessor) - 데이터 처리
3. [DeviceUtils](#deviceutils) - 디바이스 관리
4. [MetricsUtils](#metricsutils) - 평가 메트릭
5. [ExperimentUtils](#experimentutils) - 실험 관리
6. [PathManager](#pathmanager) - 경로 관리

## ConfigManager

YAML 설정 파일 로딩, 병합, 검증을 담당하는 클래스입니다.

### 클래스 정의

```python
class ConfigManager:
    """
    설정 파일 통합 관리자
    
    기능:
    - 기존 config.yaml과 새로운 base_config.yaml 모두 지원
    - YAML 파일 간 상속 및 병합
    - 환경변수 기반 오버라이드
    - WandB Sweep 파라미터 동적 병합
    - 설정 검증 및 기본값 처리
    - 모델별 특화 설정 로딩
    """
```

### 생성자

```python
def __init__(self, base_dir: Optional[str] = None, validate: bool = True):
```

**Parameters:**
- `base_dir` (Optional[str]): 설정 파일 기본 디렉토리. 기본값: 프로젝트 루트
- `validate` (bool): 설정 검증 활성화 여부. 기본값: True

**Example:**
```python
from utils.config_manager import ConfigManager

# 기본 설정
config_manager = ConfigManager()

# 커스텀 디렉토리 및 검증 비활성화
config_manager = ConfigManager(
    base_dir="/path/to/configs",
    validate=False
)
```

### 주요 메서드

#### load_config()

설정 파일을 로딩합니다.

```python
def load_config(self, config_path: Union[str, Path], 
               model_config: Optional[str] = None,
               sweep_config: Optional[str] = None) -> Dict[str, Any]:
```

**Parameters:**
- `config_path` (Union[str, Path]): 메인 설정 파일 경로
- `model_config` (Optional[str]): 모델별 특화 설정 파일명
- `sweep_config` (Optional[str]): Sweep 설정 파일명

**Returns:**
- `Dict[str, Any]`: 병합된 설정 딕셔너리

**Raises:**
- `FileNotFoundError`: 설정 파일을 찾을 수 없을 때
- `ConfigValidationError`: 설정 검증 실패 시

**Example:**
```python
# 기본 설정 로딩
config = config_manager.load_config("config/base_config.yaml")

# 모델별 설정과 함께 로딩
config = config_manager.load_config(
    config_path="config/base_config.yaml",
    model_config="kobart",  # config/models/kobart.yaml
    sweep_config="basic_sweep"  # config/sweep/basic_sweep.yaml
)

print(f"Model architecture: {config['model']['architecture']}")
print(f"Learning rate: {config['training']['learning_rate']}")
```

#### merge_sweep_params()

WandB Sweep 파라미터를 기본 설정과 병합합니다.

```python
def merge_sweep_params(self, base_config: Dict[str, Any], 
                      sweep_config: Any) -> Dict[str, Any]:
```

**Parameters:**
- `base_config` (Dict[str, Any]): 기본 설정 딕셔너리
- `sweep_config` (Any): WandB sweep 설정

**Returns:**
- `Dict[str, Any]`: 병합된 설정

**Example:**
```python
import wandb

# WandB Sweep 실행 중
wandb.init()

# 기본 설정 로딩
base_config = config_manager.load_config("config/base_config.yaml")

# Sweep 파라미터 병합
merged_config = config_manager.merge_sweep_params(
    base_config, 
    wandb.config
)

# 병합된 설정 사용
print(f"Sweep learning rate: {merged_config['training']['learning_rate']}")
```

#### save_config()

설정을 YAML 파일로 저장합니다.

```python
def save_config(self, config: Dict[str, Any], 
                output_path: Union[str, Path],
                add_metadata: bool = True) -> None:
```

**Parameters:**
- `config` (Dict[str, Any]): 저장할 설정
- `output_path` (Union[str, Path]): 출력 파일 경로
- `add_metadata` (bool): 메타데이터 추가 여부

**Example:**
```python
# 설정 수정
config['training']['learning_rate'] = 3e-5
config['training']['num_train_epochs'] = 5

# 수정된 설정 저장
config_manager.save_config(
    config,
    "config/experiments/modified_config.yaml",
    add_metadata=True
)
```

### 환경변수 오버라이드

환경변수를 통해 설정값을 오버라이드할 수 있습니다.

```bash
# 환경변수 설정
export LEARNING_RATE=1e-4
export BATCH_SIZE=32
export WANDB_PROJECT=my-nlp-project
```

```python
# 자동으로 환경변수가 적용됨
config = config_manager.load_config("config/base_config.yaml")
print(f"Learning rate from env: {config['training']['learning_rate']}")  # 1e-4
print(f"Batch size from env: {config['training']['per_device_train_batch_size']}")  # 32
```

## DataProcessor

데이터 전처리, 토크나이징, 데이터셋 변환을 담당하는 클래스입니다.

### 주요 클래스

#### DataProcessor

```python
class DataProcessor:
    """
    메인 데이터 프로세서
    
    토크나이저와 설정을 받아서 데이터 전처리 및 변환을 수행
    """
```

**생성자:**
```python
def __init__(self, tokenizer: Optional[PreTrainedTokenizer] = None,
             config: Optional[Dict[str, Any]] = None):
```

**주요 메서드:**

##### load_data()

```python
def load_data(self, data_path: Union[str, Path]) -> List[Dict[str, Any]]:
```

**Example:**
```python
from utils.data_utils import DataProcessor
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")
processor = DataProcessor(tokenizer, config)

# 데이터 로딩
data = processor.load_data("data/train.csv")
print(f"Loaded {len(data)} samples")
```

##### process_data()

```python
def process_data(self, data: List[Dict[str, Any]], 
                is_training: bool = True) -> Dataset:
```

**Example:**
```python
# 학습용 데이터 처리
train_dataset = processor.process_data(train_data, is_training=True)

# 평가용 데이터 처리
val_dataset = processor.process_data(val_data, is_training=False)

print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
```

#### TextPreprocessor

텍스트 전처리 전용 클래스입니다.

```python
class TextPreprocessor:
    """
    텍스트 전처리기
    
    한국어 텍스트 정제, 정규화, 특수 토큰 처리
    """
```

**주요 메서드:**

```python
def preprocess_text(self, text: str, 
                   remove_urls: bool = True,
                   remove_emails: bool = True, 
                   normalize_whitespace: bool = True) -> str:
```

**Example:**
```python
from utils.data_utils import TextPreprocessor

preprocessor = TextPreprocessor()

# 텍스트 전처리
raw_text = "안녕하세요!!!   이메일: test@example.com 웹사이트: https://example.com"
cleaned_text = preprocessor.preprocess_text(
    raw_text,
    remove_urls=True,
    remove_emails=True,
    normalize_whitespace=True
)
print(f"Cleaned: {cleaned_text}")
# 출력: "안녕하세요! 웹사이트:"
```

## DeviceUtils

CUDA, MPS, CPU 디바이스 자동 감지 및 최적화를 담당하는 유틸리티 함수들입니다.

### 주요 함수

#### get_optimal_device()

최적의 디바이스를 자동으로 감지합니다.

```python
def get_optimal_device() -> Tuple[torch.device, Dict[str, Any]]:
```

**Returns:**
- `Tuple[torch.device, Dict[str, Any]]`: (디바이스, 디바이스 정보)

**Example:**
```python
from utils.device_utils import get_optimal_device

device, device_info = get_optimal_device()
print(f"Optimal device: {device}")
print(f"Device info: {device_info}")

# 출력 예시:
# Optimal device: cuda:0
# Device info: {
#     'type': 'cuda',
#     'name': 'NVIDIA GeForce RTX 4090',
#     'memory_total': 24564,
#     'memory_available': 23450
# }
```

#### setup_device_config()

디바이스별 최적화 설정을 생성합니다.

```python
def setup_device_config(device_info: Dict[str, Any], 
                        model_size: str = "base") -> DeviceConfig:
```

**Parameters:**
- `device_info` (Dict[str, Any]): 디바이스 정보
- `model_size` (str): 모델 크기 ('small', 'base', 'large')

**Returns:**
- `DeviceConfig`: 최적화 설정 객체

**Example:**
```python
from utils.device_utils import get_optimal_device, setup_device_config

device, device_info = get_optimal_device()
opt_config = setup_device_config(device_info, model_size="base")

print(f"Recommended batch size: {opt_config.batch_size}")
print(f"Mixed precision: {opt_config.fp16}")
print(f"Number of workers: {opt_config.num_workers}")
```

#### detect_cuda_devices()

CUDA 디바이스 정보를 상세히 조회합니다.

```python
def detect_cuda_devices() -> List[Dict[str, Any]]:
```

**Example:**
```python
from utils.device_utils import detect_cuda_devices

cuda_devices = detect_cuda_devices()
for i, device in enumerate(cuda_devices):
    print(f"GPU {i}: {device['name']}")
    print(f"  Memory: {device['memory_total']}MB")
    print(f"  Compute capability: {device['compute_capability']}")
```

## MetricsUtils

ROUGE 점수 계산 및 다중 참조 평가를 담당하는 클래스들입니다.

### RougeCalculator

```python
class RougeCalculator:
    """
    ROUGE 점수 계산기
    
    한국어 토크나이저 지원 및 HuggingFace Trainer 호환
    """
```

**생성자:**
```python
def __init__(self, tokenizer: Optional[PreTrainedTokenizer] = None,
             use_stemmer: bool = True,
             tokenize_korean: bool = True):
```

**주요 메서드:**

#### compute_metrics()

```python
def compute_metrics(self, predictions: List[str], 
                   references: List[str]) -> Dict[str, float]:
```

**Example:**
```python
from utils.metrics import RougeCalculator
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")
calculator = RougeCalculator(tokenizer, tokenize_korean=True)

predictions = ["두 사람이 커피를 마시며 대화했다."]
references = ["두 사람이 카페에서 커피를 마시며 이야기를 나눴다."]

scores = calculator.compute_metrics(predictions, references)
print(f"ROUGE-1: {scores['rouge1']:.4f}")
print(f"ROUGE-2: {scores['rouge2']:.4f}")
print(f"ROUGE-L: {scores['rougeL']:.4f}")
```

### MultiReferenceROUGE

다중 참조 문서에 대한 ROUGE 평가를 지원합니다.

```python
class MultiReferenceROUGE:
    """
    다중 참조 ROUGE 평가기
    """
```

**Example:**
```python
from utils.metrics import MultiReferenceROUGE

multi_rouge = MultiReferenceROUGE(tokenizer)

prediction = "두 사람이 커피 약속을 잡았다."
references = [
    "두 사람이 커피를 마시기로 약속했다.",
    "커피 약속을 정했다.",
    "카페에서 만나기로 했다."
]

scores = multi_rouge.calculate_multi_reference(
    prediction, 
    references
)
print(f"Best ROUGE-1: {scores['best_rouge1']:.4f}")
print(f"Average ROUGE-1: {scores['avg_rouge1']:.4f}")
```

## ExperimentUtils

실험 추적, 모델 등록, 성능 비교를 담당하는 클래스들입니다.

### ExperimentTracker

```python
class ExperimentTracker:
    """
    실험 추적기
    
    실험 생명주기 관리 및 메타데이터 저장
    """
```

**주요 메서드:**

#### start_experiment()

```python
def start_experiment(self, name: str, 
                    description: str = "",
                    config: Optional[Dict[str, Any]] = None,
                    **kwargs) -> str:
```

**Example:**
```python
from utils.experiment_utils import ExperimentTracker

tracker = ExperimentTracker(experiments_dir="experiments/")

experiment_id = tracker.start_experiment(
    name="kobart_baseline_v1",
    description="KoBART 기본 실험",
    config=config,
    model_type="kobart",
    dataset_size=1000
)

print(f"Started experiment: {experiment_id}")
```

#### end_experiment()

```python
def end_experiment(self, experiment_id: str,
                  final_metrics: Optional[Dict[str, float]] = None,
                  status: str = "completed",
                  **kwargs) -> None:
```

**Example:**
```python
# 실험 완료
tracker.end_experiment(
    experiment_id=experiment_id,
    final_metrics={
        "rouge1_f1": 0.45,
        "rouge2_f1": 0.32,
        "rougeL_f1": 0.41
    },
    status="completed"
)
```

### ModelRegistry

```python
class ModelRegistry:
    """
    모델 등록소
    
    모델 버전 관리 및 성능 비교
    """
```

**주요 메서드:**

#### register_model()

```python
def register_model(self, name: str,
                  architecture: str,
                  performance: Dict[str, float],
                  **kwargs) -> str:
```

**Example:**
```python
from utils.experiment_utils import ModelRegistry

registry = ModelRegistry(models_dir="models/")

model_id = registry.register_model(
    name="kobart_baseline_v1",
    architecture="kobart",
    checkpoint="gogamza/kobart-base-v2",
    performance={
        "rouge1_f1": 0.45,
        "rouge2_f1": 0.32,
        "rougeL_f1": 0.41
    },
    training_info={
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 5e-5
    },
    file_path="models/kobart_baseline_v1"
)

print(f"Registered model: {model_id}")
```

#### get_best_models()

```python
def get_best_models(self, metric: str = "rouge_combined_f1",
                   limit: int = 10) -> List[Dict[str, Any]]:
```

**Example:**
```python
# 최고 성능 모델들 조회
best_models = registry.get_best_models(
    metric="rouge_combined_f1",
    limit=5
)

for i, model in enumerate(best_models):
    print(f"{i+1}. {model['name']}: {model['performance']['rouge_combined_f1']:.4f}")
```

## PathManager

크로스 플랫폼 경로 관리 및 디렉토리 자동 생성을 담당하는 클래스입니다.

### 클래스 정의

```python
class PathManager:
    """
    프로젝트 경로 관리자
    
    기능:
    - 프로젝트 루트 자동 감지
    - 플랫폼 독립적 경로 처리
    - 디렉토리 자동 생성
    - 상대 경로 해결
    """
```

### 주요 메서드

#### resolve_path()

```python
def resolve_path(self, path: Union[str, Path]) -> Path:
```

**Example:**
```python
from utils.path_utils import path_manager

# 상대 경로 해결
data_path = path_manager.resolve_path("data/train.csv")
model_path = path_manager.resolve_path("models/best_model")

print(f"Data path: {data_path}")
print(f"Model path: {model_path}")
```

#### get_data_path()

```python
def get_data_path(self, filename: str) -> Path:
```

#### get_output_path()

```python
def get_output_path(self, experiment_name: str) -> Path:
```

#### ensure_dir()

```python
def ensure_dir(self, path: Union[str, Path]) -> Path:
```

**Example:**
```python
# 다양한 경로 생성
data_path = path_manager.get_data_path("train.csv")
output_path = path_manager.get_output_path("experiment_1")
model_path = path_manager.get_model_path("kobart_v1")

# 디렉토리 자동 생성
results_dir = path_manager.ensure_dir("results/analysis")
checkpoints_dir = path_manager.ensure_dir(output_path / "checkpoints")

print(f"Results directory: {results_dir}")
print(f"Checkpoints directory: {checkpoints_dir}")
```

### 전역 인스턴스

프로젝트 전체에서 사용할 수 있는 전역 인스턴스가 제공됩니다.

```python
from utils.path_utils import path_manager

# 바로 사용 가능
project_root = path_manager.project_root
data_dir = path_manager.data_dir
models_dir = path_manager.models_dir

print(f"Project root: {project_root}")
print(f"Data directory: {data_dir}")
print(f"Models directory: {models_dir}")
```

## 통합 사용 예제

### 전체 워크플로우

```python
from utils.config_manager import ConfigManager
from utils.data_utils import DataProcessor
from utils.device_utils import get_optimal_device, setup_device_config
from utils.metrics import RougeCalculator
from utils.experiment_utils import ExperimentTracker, ModelRegistry
from utils.path_utils import path_manager
from transformers import AutoTokenizer

# 1. 설정 관리
config_manager = ConfigManager()
config = config_manager.load_config(
    "config/base_config.yaml",
    model_config="kobart"
)

# 2. 디바이스 최적화
device, device_info = get_optimal_device()
opt_config = setup_device_config(device_info, "base")

# 최적화 설정 적용
config['training']['per_device_train_batch_size'] = opt_config.batch_size
config['training']['fp16'] = opt_config.fp16

# 3. 데이터 처리
tokenizer = AutoTokenizer.from_pretrained(config['model']['checkpoint'])
data_processor = DataProcessor(tokenizer, config)

data_path = path_manager.get_data_path("train.csv")
train_data = data_processor.load_data(data_path)
train_dataset = data_processor.process_data(train_data, is_training=True)

# 4. 평가 메트릭 준비
rouge_calculator = RougeCalculator(tokenizer, tokenize_korean=True)

# 5. 실험 추적 시작
tracker = ExperimentTracker()
experiment_id = tracker.start_experiment(
    name="kobart_optimized_v1",
    description="최적화된 KoBART 실험",
    config=config
)

# 6. 모델 등록 준비
registry = ModelRegistry()

print("All utils initialized successfully!")
print(f"Device: {device}")
print(f"Batch size: {config['training']['per_device_train_batch_size']}")
print(f"Dataset size: {len(train_dataset)}")
print(f"Experiment ID: {experiment_id}")
```

### 에러 처리

```python
try:
    # 설정 로딩
    config = config_manager.load_config("config/base_config.yaml")
    
except FileNotFoundError:
    print("설정 파일을 찾을 수 없습니다.")
    # 기본 설정 사용
    config = config_manager.get_default_config()
    
except ConfigValidationError as e:
    print(f"설정 검증 실패: {e}")
    # 설정 수정 후 재시도
    
except Exception as e:
    print(f"예상치 못한 에러: {e}")
    # 로깅 및 정리 작업
```

---

**관련 문서:**
- [Trainer API 참조](./trainer_api.md)
- [Core API 참조](./core_api.md)
- [시스템 아키텍처](../system_architecture.md)
