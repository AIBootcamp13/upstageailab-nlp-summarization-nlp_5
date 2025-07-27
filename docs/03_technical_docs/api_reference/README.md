# NLP 대화 요약 시스템 API 참조

이 문서는 NLP 대화 요약 시스템의 전체 API를 포괄적으로 다룹니다. 각 모듈별 상세 기능과 사용법을 제공합니다.

## 📚 목차

1. [개요](#개요)
2. [핵심 모듈](#핵심-모듈)
3. [유틸리티 모듈](#유틸리티-모듈)
4. [데이터 증강 모듈](#데이터-증강-모듈)
5. [스크립트 및 자동화](#스크립트-및-자동화)
6. [설치 및 설정](#설치-및-설정)
7. [빠른 시작](#빠른-시작)

## 개요

### 시스템 아키텍처

```
nlp-sum-lyj/
├── code/
│   ├── core/                    # 핵심 추론 엔진
│   ├── utils/                   # 공통 유틸리티
│   ├── data_augmentation/       # 데이터 증강
│   ├── trainer.py               # 메인 트레이너
│   └── auto_experiment_runner.py # 자동 실험 실행
├── config/                      # 설정 파일
├── data/                       # 데이터 디렉토리
└── docs/                       # 문서
```

### 주요 특징

- **모듈식 설계**: 각 기능이 독립적으로 사용 가능
- **자동 디바이스 감지**: CUDA/MPS/CPU 자동 최적화
- **WandB 통합**: 실험 추적 및 하이퍼파라미터 최적화
- **다중 모델 지원**: BART, GPT, T5 등 다양한 아키텍처
- **배치 처리**: 효율적인 대용량 데이터 처리

## 핵심 모듈

### 📖 [trainer.py](./trainer_api.md) - 메인 트레이너 모듈

대화 요약 모델의 학습, 평가, 추론을 담당하는 핵심 모듈입니다.

```python
from trainer import DialogueSummarizationTrainer, create_trainer

# 트레이너 생성
trainer = create_trainer("config/base_config.yaml")

# 학습 실행
datasets = trainer.prepare_data()
result = trainer.train(datasets)
```

**주요 클래스:**
- `DialogueSummarizationTrainer`: 메인 트레이너 클래스
- `TrainingResult`: 학습 결과 데이터 클래스
- `WandbCallback`: WandB 로깅 콜백

**주요 기능:**
- 완전 자동화된 실험 관리
- WandB Sweep 통합
- 실시간 메트릭 추적
- 모델 등록 시스템

### 🔍 [core/inference.py](./core_api.md) - 추론 엔진

독립적인 추론 엔진으로 배치 처리 및 다양한 입력 형식을 지원합니다.

```python
from core.inference import InferenceEngine, InferenceConfig

# 추론 엔진 설정
config = InferenceConfig(
    model_path="models/best_model",
    batch_size=16,
    max_target_length=256
)

# 추론 실행
engine = InferenceEngine(config)
result = engine.predict_single("대화 텍스트")
```

**주요 클래스:**
- `InferenceEngine`: 메인 추론 엔진
- `InferenceConfig`: 추론 설정 클래스

**주요 기능:**
- 단일/배치 예측
- DataFrame 처리
- 캐시 시스템
- 자동 디바이스 최적화

### 🤖 [auto_experiment_runner.py](./automation_api.md) - 자동 실험 실행

YAML 설정 기반의 완전 자동화된 실험 실행 시스템입니다.

```python
from auto_experiment_runner import AutoExperimentRunner

# 자동 실험 실행
runner = AutoExperimentRunner("experiments/")
runner.run_all_experiments()
```

**주요 클래스:**
- `AutoExperimentRunner`: 자동 실험 실행기

**주요 기능:**
- YAML 설정 기반 실험 정의
- 순차적 실험 실행
- 결과 자동 추적
- 에러 처리 및 복구

## 유틸리티 모듈

### ⚙️ [utils/config_manager.py](./utils_api.md#config-manager) - 설정 관리

YAML 설정 파일 로딩, 병합, 검증을 담당합니다.

```python
from utils.config_manager import ConfigManager

config_manager = ConfigManager()
config = config_manager.load_config("config/base_config.yaml")

# Sweep 파라미터 병합
merged_config = config_manager.merge_sweep_params(config, wandb.config)
```

**주요 클래스:**
- `ConfigManager`: 설정 관리 클래스

**주요 기능:**
- YAML 설정 로딩
- 동적 파라미터 병합
- 설정 검증
- 환경변수 오버라이드

### 📊 [utils/data_utils.py](./utils_api.md#data-utils) - 데이터 처리

데이터 전처리, 토크나이징, 데이터셋 변환을 담당합니다.

```python
from utils.data_utils import DataProcessor, TextPreprocessor

processor = DataProcessor(tokenizer, config)
dataset = processor.process_data(raw_data, is_training=True)
```

**주요 클래스:**
- `DataProcessor`: 메인 데이터 프로세서
- `TextPreprocessor`: 텍스트 전처리기
- `DialogueSummarizationDataset`: PyTorch 데이터셋

**주요 기능:**
- 한국어 텍스트 정제
- 토크나이징 및 인코딩
- 데이터 통계 분석
- 배치 처리

### 🔧 [utils/device_utils.py](./utils_api.md#device-utils) - 디바이스 관리

CUDA, MPS, CPU 디바이스 자동 감지 및 최적화를 담당합니다.

```python
from utils.device_utils import get_optimal_device, setup_device_config

device, device_info = get_optimal_device()
opt_config = setup_device_config(device_info, model_size="base")
```

**주요 함수:**
- `get_optimal_device()`: 최적 디바이스 자동 감지
- `setup_device_config()`: 디바이스별 최적화 설정
- `detect_cuda_devices()`: CUDA 디바이스 정보
- `detect_mps_device()`: MPS 디바이스 정보

**주요 기능:**
- 플랫폼별 최적화
- 메모리 기반 배치 크기 조정
- Mixed Precision 설정

### 📈 [utils/metrics.py](./utils_api.md#metrics) - 평가 메트릭

ROUGE 점수 계산 및 다중 참조 평가를 담당합니다.

```python
from utils.metrics import RougeCalculator, MultiReferenceROUGE

calculator = RougeCalculator(tokenizer)
scores = calculator.compute_metrics(predictions, references)
```

**주요 클래스:**
- `RougeCalculator`: ROUGE 점수 계산기
- `MultiReferenceROUGE`: 다중 참조 ROUGE
- `MetricTracker`: 메트릭 추적기

**주요 기능:**
- 단일/다중 참조 ROUGE
- 한국어 토크나이저 지원
- HuggingFace Trainer 호환
- 실시간 메트릭 추적

### 🧪 [utils/experiment_utils.py](./utils_api.md#experiment-utils) - 실험 관리

실험 추적, 모델 등록, 성능 비교를 담당합니다.

```python
from utils.experiment_utils import ExperimentTracker, ModelRegistry

tracker = ExperimentTracker()
experiment_id = tracker.start_experiment(name="test_exp", config=config)

registry = ModelRegistry()
model_id = registry.register_model(name="best_model", performance=metrics)
```

**주요 클래스:**
- `ExperimentTracker`: 실험 추적기
- `ModelRegistry`: 모델 등록소
- `ExperimentConfig`: 실험 설정

**주요 기능:**
- 실험 생명주기 관리
- 모델 버전 관리
- 성능 비교 및 분석
- WandB 연동

### 📂 [utils/path_utils.py](./utils_api.md#path-utils) - 경로 관리

크로스 플랫폼 경로 관리 및 디렉토리 자동 생성을 담당합니다.

```python
from utils.path_utils import PathManager, path_manager

# 전역 경로 관리자 사용
data_path = path_manager.get_data_path("train.csv")
output_path = path_manager.get_output_path("experiment_1")
```

**주요 클래스:**
- `PathManager`: 경로 관리 클래스

**주요 기능:**
- 프로젝트 루트 자동 감지
- 플랫폼 독립적 경로
- 디렉토리 자동 생성
- 상대 경로 해결

## 데이터 증강 모듈

### 🔄 [data_augmentation/simple_augmentation.py](./data_augmentation_api.md#simple) - 기본 증강

동의어 치환, 문장 순서 변경 등의 기본 데이터 증강 기법을 제공합니다.

```python
from data_augmentation.simple_augmentation import SimpleAugmenter

augmenter = SimpleAugmenter()
augmented_data = augmenter.augment_dataset(dataset, augment_ratio=0.3)
```

**주요 클래스:**
- `SimpleAugmenter`: 기본 증강기
- `SynonymReplacement`: 동의어 치환
- `SentenceReorder`: 문장 순서 변경

### 🌐 [data_augmentation/backtranslation.py](./data_augmentation_api.md#backtranslation) - 백번역

다국어 백번역을 통한 고급 데이터 증강을 제공합니다.

```python
from data_augmentation.backtranslation import BackTranslationAugmenter

augmenter = BackTranslationAugmenter(method="google")
augmented_data = augmenter.augment(text, target_lang="en")
```

**주요 클래스:**
- `BackTranslationAugmenter`: 백번역 증강기
- `MultilingualBackTranslation`: 다국어 백번역

## 스크립트 및 자동화

### 🔄 [sweep_runner.py](./scripts_api.md#sweep) - Sweep 실행

WandB Sweep을 통한 하이퍼파라미터 최적화를 실행합니다.

```bash
python sweep_runner.py --config config/sweep/basic_sweep.yaml
```

### ⚡ [parallel_sweep_runner.py](./scripts_api.md#parallel-sweep) - 병렬 Sweep

다중 프로세스를 통한 병렬 Sweep 실행을 제공합니다.

```bash
python parallel_sweep_runner.py --config config/sweep/parallel_sweep.yaml --agents 4
```

### 🎯 [run_inference.py](./scripts_api.md#inference) - 추론 실행

대회 제출용 추론 결과 생성 스크립트입니다.

```bash
python run_inference.py --model models/best_model --input data/test.csv --output submissions/
```

## 설치 및 설정

### 요구사항

```bash
pip install -r requirements.txt
```

**주요 의존성:**
- `torch >= 1.12.0`
- `transformers >= 4.20.0`
- `datasets >= 2.0.0`
- `wandb >= 0.13.0`
- `pandas >= 1.3.0`
- `numpy >= 1.21.0`

### 환경 설정

```bash
# WandB 설정
wandb login

# 환경변수 설정 (선택사항)
export WANDB_PROJECT="nlp-dialogue-summarization"
export CUDA_VISIBLE_DEVICES="0"
```

## 빠른 시작

### 1. 기본 학습

```python
from trainer import create_trainer

# 설정 로딩 및 트레이너 생성
trainer = create_trainer("config/base_config.yaml")

# 데이터 준비
datasets = trainer.prepare_data(
    train_path="data/train.csv",
    val_path="data/validation.csv"
)

# 학습 실행
result = trainer.train(datasets)
print(f"Best ROUGE F1: {result.best_metrics['rouge_combined_f1']:.4f}")
```

### 2. 추론 실행

```python
from core.inference import InferenceEngine, InferenceConfig

# 추론 설정
config = InferenceConfig(
    model_path="models/best_model",
    batch_size=16
)

# 추론 엔진 생성 및 실행
engine = InferenceEngine(config)
result = engine.predict_single("두 사람이 커피숍에서 만나서 이야기를 나눴다.")
print(f"요약: {result}")
```

### 3. 자동 실험 실행

```python
from auto_experiment_runner import AutoExperimentRunner

# 실험 실행기 생성
runner = AutoExperimentRunner("config/experiments/")

# 모든 실험 자동 실행
results = runner.run_all_experiments()
for exp_name, result in results.items():
    print(f"{exp_name}: {result['best_metrics']['rouge_combined_f1']:.4f}")
```

### 4. 데이터 증강

```python
from data_augmentation.simple_augmentation import SimpleAugmenter

# 증강기 생성
augmenter = SimpleAugmenter()

# 데이터 증강 실행
original_data = [
    {"input": "안녕하세요. 오늘 날씨가 좋네요.", "target": "날씨 인사"}
]

augmented_data = augmenter.augment_dataset(
    original_data, 
    augment_ratio=0.5
)
print(f"Original: {len(original_data)}, Augmented: {len(augmented_data)}")
```

## 에러 처리

### 일반적인 에러와 해결방법

1. **CUDA Out of Memory**
   ```python
   # 배치 크기 감소
   config['training']['per_device_train_batch_size'] = 4
   config['training']['gradient_accumulation_steps'] = 4
   ```

2. **모델 로딩 실패**
   ```python
   # 경로 확인
   from utils.path_utils import path_manager
   model_path = path_manager.resolve_path("models/best_model")
   print(f"Resolved path: {model_path}")
   ```

3. **WandB 연결 실패**
   ```bash
   wandb login
   export WANDB_MODE=offline  # 오프라인 모드
   ```

## 성능 최적화

### 권장 설정

```python
# CUDA 사용 시
config = {
    'training': {
        'fp16': True,
        'gradient_checkpointing': True,
        'per_device_train_batch_size': 32
    }
}

# MPS (Apple Silicon) 사용 시
config = {
    'training': {
        'fp16': False,
        'per_device_train_batch_size': 16,
        'dataloader_num_workers': 2
    }
}

# CPU 사용 시
config = {
    'training': {
        'per_device_train_batch_size': 4,
        'gradient_accumulation_steps': 8,
        'dataloader_num_workers': 1
    }
}
```

## 라이센스 및 기여

이 프로젝트는 MIT 라이센스 하에 배포됩니다. 기여를 원하시면 이슈를 등록하거나 풀 리퀘스트를 제출해 주세요.

## 관련 문서

- [사용자 가이드](../02_user_guides/README.md)
- [시스템 아키텍처](./system_architecture.md)
- [성능 최적화](./performance_optimization.md)
- [문제 해결](../06_troubleshooting/README.md)

---

**마지막 업데이트**: 2025-01-28  
**버전**: 2.0.0  
**작성자**: NLP Summarization Team
