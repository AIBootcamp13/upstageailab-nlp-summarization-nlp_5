# 데이터 증강 및 스크립트 API 참조

이 문서는 데이터 증강 모듈과 자동화 스크립트들의 API를 다룹니다.

## 목차

1. [데이터 증강 모듈](#데이터-증강-모듈)
   - [SimpleAugmenter](#simpleaugmenter)
   - [BackTranslationAugmenter](#backtranslationaugmenter)
2. [자동화 스크립트](#자동화-스크립트)
   - [AutoExperimentRunner](#autoexperimentrunner)
   - [SweepRunner](#sweeprunner)
   - [InferenceScript](#inferencescript)

## 데이터 증강 모듈

### SimpleAugmenter

기본적인 데이터 증강 기법을 제공하는 클래스입니다.

#### 클래스 정의

```python
class SimpleAugmenter:
    """
    기본 데이터 증강기
    
    동의어 치환, 문장 순서 변경 등의 기본 증강 기법 제공
    """
```

#### 생성자

```python
def __init__(self, synonym_prob: float = 0.1,
             sentence_shuffle_prob: float = 0.1,
             cache_dir: str = "cache/"):
```

**Parameters:**
- `synonym_prob` (float): 동의어 치환 확률. 기본값: 0.1
- `sentence_shuffle_prob` (float): 문장 순서 변경 확률. 기본값: 0.1
- `cache_dir` (str): 캐시 디렉토리. 기본값: "cache/"

#### 주요 메서드

##### augment()

단일 데이터 샘플을 증강합니다.

```python
def augment(self, dialogue: str, summary: str) -> Tuple[str, str]:
```

**Parameters:**
- `dialogue` (str): 원본 대화
- `summary` (str): 원본 요약

**Returns:**
- `Tuple[str, str]`: (증강된 대화, 증강된 요약)

**Example:**
```python
from data_augmentation.simple_augmentation import SimpleAugmenter

augmenter = SimpleAugmenter(
    synonym_prob=0.15,
    sentence_shuffle_prob=0.1
)

original_dialogue = "A: 오늘 날씨가 좋네요. B: 정말 맑고 화창해요."
original_summary = "날씨가 좋다는 대화"

augmented_dialogue, augmented_summary = augmenter.augment(
    original_dialogue, 
    original_summary
)

print(f"Original: {original_dialogue}")
print(f"Augmented: {augmented_dialogue}")
```

##### augment_dataset()

전체 데이터셋을 증강합니다.

```python
def augment_dataset(self, dataset: List[Dict[str, str]], 
                   augment_ratio: float = 0.5) -> List[Dict[str, str]]:
```

**Parameters:**
- `dataset` (List[Dict[str, str]]): 원본 데이터셋
- `augment_ratio` (float): 증강 비율 (0.5 = 50% 증강)

**Returns:**
- `List[Dict[str, str]]`: 원본 + 증강된 데이터셋

**Example:**
```python
# 원본 데이터셋
dataset = [
    {
        "dialogue": "A: 안녕하세요. B: 안녕하세요!",
        "summary": "인사"
    },
    {
        "dialogue": "A: 점심 뭐 먹을까요? B: 한식 어때요?",
        "summary": "점심 메뉴 상의"
    }
]

# 50% 증강 (1개당 0.5개씩 추가)
augmented_dataset = augmenter.augment_dataset(
    dataset, 
    augment_ratio=0.5
)

print(f"Original size: {len(dataset)}")
print(f"Augmented size: {len(augmented_dataset)}")
```

#### 내부 증강 기법

##### SynonymReplacement

```python
class SynonymReplacement:
    """동의어 치환 증강"""
    
    def replace_synonyms(self, text: str, replace_prob: float = 0.1) -> str:
```

##### SentenceReorder

```python
class SentenceReorder:
    """문장 순서 변경 증강"""
    
    def reorder_sentences(self, text: str, shuffle_prob: float = 0.1) -> str:
```

**Example:**
```python
from data_augmentation.simple_augmentation import SynonymReplacement, SentenceReorder

# 개별 증강 기법 사용
synonym_replacer = SynonymReplacement()
sentence_reorder = SentenceReorder()

text = "오늘 날씨가 좋습니다. 산책하기 좋은 날이에요."

# 동의어 치환
synonym_result = synonym_replacer.replace_synonyms(text, replace_prob=0.2)
print(f"Synonym replacement: {synonym_result}")

# 문장 순서 변경
reorder_result = sentence_reorder.reorder_sentences(text, shuffle_prob=0.5)
print(f"Sentence reorder: {reorder_result}")
```

### BackTranslationAugmenter

백번역을 통한 고급 데이터 증강을 제공하는 클래스입니다.

#### 클래스 정의

```python
class BackTranslationAugmenter:
    """
    백번역 데이터 증강기
    
    한영한 번역을 통한 패러프레이징
    """
```

#### 생성자

```python
def __init__(self, method: str = "google",
             target_langs: List[str] = ["en"],
             quality_threshold: float = 0.7,
             cache_enabled: bool = True):
```

**Parameters:**
- `method` (str): 번역 방법 ("google" 또는 "marian"). 기본값: "google"
- `target_langs` (List[str]): 중간 언어 목록. 기본값: ["en"]
- `quality_threshold` (float): 품질 임계값. 기본값: 0.7
- `cache_enabled` (bool): 캐시 사용 여부. 기본값: True

#### 주요 메서드

##### augment()

백번역을 통한 데이터 증강을 수행합니다.

```python
def augment(self, text: str, target_lang: str = "en") -> str:
```

**Parameters:**
- `text` (str): 원본 텍스트
- `target_lang` (str): 중간 언어. 기본값: "en"

**Returns:**
- `str`: 백번역된 텍스트

**Example:**
```python
from data_augmentation.backtranslation import BackTranslationAugmenter

# Google Translate 사용
google_augmenter = BackTranslationAugmenter(
    method="google",
    target_langs=["en", "ja"],
    quality_threshold=0.8
)

original_text = "오늘은 날씨가 정말 좋습니다."
augmented_text = google_augmenter.augment(original_text, target_lang="en")

print(f"Original: {original_text}")
print(f"Back-translated: {augmented_text}")

# MarianMT 모델 사용 (로컬)
marian_augmenter = BackTranslationAugmenter(
    method="marian",
    target_langs=["en"]
)

local_augmented = marian_augmenter.augment(original_text)
print(f"Marian back-translated: {local_augmented}")
```

##### augment_dataset()

데이터셋 전체에 백번역을 적용합니다.

```python
def augment_dataset(self, dataset: List[Dict[str, str]], 
                   augment_ratio: float = 0.3,
                   max_samples: Optional[int] = None) -> List[Dict[str, str]]:
```

**Example:**
```python
dataset = [
    {"dialogue": "A: 오늘 날씨 어때요? B: 맑고 좋아요.", "summary": "날씨 대화"},
    {"dialogue": "A: 점심 뭐 드실래요? B: 김치찌개 좋겠어요.", "summary": "점심 메뉴 상의"}
]

# 백번역 증강 (30% 증강)
back_translated_dataset = google_augmenter.augment_dataset(
    dataset,
    augment_ratio=0.3,
    max_samples=100
)

print(f"Original: {len(dataset)}")
print(f"Augmented: {len(back_translated_dataset)}")
```

#### 품질 검증

백번역의 품질을 자동으로 검증합니다.

```python
def _validate_quality(self, original: str, back_translated: str) -> float:
    """
    백번역 품질 검증
    
    BLEU 점수와 길이 비율을 종합하여 품질 점수 계산
    """
```

**Example:**
```python
# 품질 검증 예제
original = "오늘은 정말 좋은 날씨입니다."
back_translated = "Today is really nice weather."

quality_score = google_augmenter._validate_quality(original, back_translated)
print(f"Quality score: {quality_score:.2f}")

if quality_score >= google_augmenter.quality_threshold:
    print("High quality back-translation")
else:
    print("Low quality, discarding...")
```

### MultilingualBackTranslation

다중 언어를 거친 백번역을 지원하는 고급 클래스입니다.

```python
class MultilingualBackTranslation:
    """
    다중 언어 백번역
    
    여러 언어를 거쳐 더 다양한 패러프레이징 생성
    """
```

**Example:**
```python
from data_augmentation.backtranslation import MultilingualBackTranslation

multi_augmenter = MultilingualBackTranslation(
    intermediate_langs=["en", "ja", "zh"],
    final_lang="ko"
)

# 한국어 → 영어 → 일본어 → 중국어 → 한국어
multi_result = multi_augmenter.augment_through_languages(
    "안녕하세요. 오늘 날씨가 좋네요.",
    path=["en", "ja", "zh"]
)

print(f"Multi-lingual result: {multi_result}")
```

## 자동화 스크립트

### AutoExperimentRunner

YAML 설정 기반의 완전 자동화된 실험 실행 시스템입니다.

#### 클래스 정의

```python
class AutoExperimentRunner:
    """
    자동 실험 실행기
    
    YAML 설정 파일을 기반으로 실험을 자동으로 발견하고 순차 실행
    """
```

#### 생성자

```python
def __init__(self, experiments_dir: str = "config/experiments/",
             base_config_path: str = "config/base_config.yaml",
             max_concurrent: int = 1):
```

**Parameters:**
- `experiments_dir` (str): 실험 설정 디렉토리
- `base_config_path` (str): 기본 설정 파일 경로
- `max_concurrent` (int): 최대 동시 실행 수

#### 주요 메서드

##### discover_experiment_configs()

실험 설정 파일들을 자동으로 발견합니다.

```python
def discover_experiment_configs(self) -> List[str]:
```

**Returns:**
- `List[str]`: 발견된 실험 설정 파일 목록

##### run_all_experiments()

모든 실험을 순차적으로 실행합니다.

```python
def run_all_experiments(self, filter_pattern: Optional[str] = None) -> Dict[str, Any]:
```

**Parameters:**
- `filter_pattern` (Optional[str]): 실험 필터 패턴

**Returns:**
- `Dict[str, Any]`: 모든 실험 결과

**Example:**
```python
from auto_experiment_runner import AutoExperimentRunner

# 실험 실행기 생성
runner = AutoExperimentRunner(
    experiments_dir="config/experiments/",
    base_config_path="config/base_config.yaml"
)

# 모든 실험 자동 실행
results = runner.run_all_experiments()

# 결과 출력
for exp_name, result in results.items():
    if result['status'] == 'completed':
        metrics = result['best_metrics']
        print(f"{exp_name}: ROUGE F1 = {metrics.get('rouge_combined_f1', 0):.4f}")
    else:
        print(f"{exp_name}: {result['status']} - {result.get('error', 'Unknown error')}")
```

##### run_single_experiment()

단일 실험을 실행합니다.

```python
def run_single_experiment(self, experiment_config_path: str) -> Dict[str, Any]:
```

**Example:**
```python
# 특정 실험만 실행
result = runner.run_single_experiment("config/experiments/kobart_baseline.yaml")

if result['status'] == 'completed':
    print(f"실험 완료: {result['best_metrics']}")
    print(f"모델 저장 위치: {result['model_path']}")
else:
    print(f"실험 실패: {result['error']}")
```

#### 실험 설정 파일 형식

```yaml
# config/experiments/kobart_baseline.yaml
meta:
  experiment_name: "kobart_baseline_v1"
  description: "KoBART 기본 성능 측정"
  tags: ["baseline", "kobart"]

model:
  architecture: "kobart"
  checkpoint: "gogamza/kobart-base-v2"

training:
  learning_rate: 5e-5
  num_train_epochs: 3
  per_device_train_batch_size: 16
  warmup_ratio: 0.1

data:
  train_path: "data/train.csv"
  val_path: "data/validation.csv"
  test_path: "data/test.csv"

wandb:
  project: "nlp-dialogue-summarization"
  tags: ["baseline", "kobart"]
```

### SweepRunner

WandB Sweep을 통한 하이퍼파라미터 최적화를 실행하는 스크립트입니다.

#### 사용법

```bash
# 기본 Sweep 실행
python sweep_runner.py --config config/sweep/basic_sweep.yaml

# 커스텀 프로젝트명으로 실행
python sweep_runner.py --config config/sweep/advanced_sweep.yaml --project my-nlp-project

# 특정 횟수만 실행
python sweep_runner.py --config config/sweep/basic_sweep.yaml --count 10
```

#### 주요 옵션

- `--config`: Sweep 설정 파일 경로
- `--project`: WandB 프로젝트명
- `--entity`: WandB 엔티티명
- `--count`: 실행 횟수
- `--method`: Sweep 방법 (grid, random, bayes)

#### Sweep 설정 파일 예제

```yaml
# config/sweep/basic_sweep.yaml
program: sweep_runner.py
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
  warmup_ratio:
    distribution: uniform
    min: 0.0
    max: 0.2
  weight_decay:
    distribution: uniform
    min: 0.01
    max: 0.3

early_terminate:
  type: hyperband
  min_iter: 3
```

### ParallelSweepRunner

다중 프로세스를 통한 병렬 Sweep 실행을 제공합니다.

#### 사용법

```bash
# 4개 에이전트로 병렬 실행
python parallel_sweep_runner.py --config config/sweep/parallel_sweep.yaml --agents 4

# GPU별 분산 실행
python parallel_sweep_runner.py --config config/sweep/gpu_sweep.yaml --agents 2 --gpus 0,1
```

#### 주요 기능

- **다중 프로세스 실행**: 여러 Sweep 에이전트 동시 실행
- **GPU 분산**: 여러 GPU에 에이전트 분산 배치
- **리소스 모니터링**: CPU/메모리 사용량 실시간 모니터링
- **에러 복구**: 에이전트 실패 시 자동 재시작

### InferenceScript

대회 제출용 추론 결과 생성 스크립트입니다.

#### 사용법

```bash
# 기본 추론 실행
python run_inference.py --model models/best_model --input data/test.csv --output submissions/

# 배치 크기 및 빔 수 조정
python run_inference.py \
    --model models/kobart_v1 \
    --input data/test.csv \
    --output submissions/ \
    --batch-size 32 \
    --num-beams 5 \
    --max-length 256

# GPU 지정
python run_inference.py --model models/best_model --input data/test.csv --output submissions/ --device cuda:1
```

#### 주요 옵션

- `--model`: 모델 경로
- `--input`: 입력 데이터 파일
- `--output`: 출력 디렉토리
- `--batch-size`: 배치 크기
- `--num-beams`: 빔 서치 빔 수
- `--max-length`: 최대 생성 길이
- `--device`: 사용할 디바이스
- `--fp16`: 혼합 정밀도 사용

#### 출력 형식

```csv
fname,summary
test_001.txt,"두 사람이 커피 약속을 잡았다."
test_002.txt,"날씨에 대한 대화를 나누었다."
test_003.txt,"점심 메뉴를 상의했다."
```

## 고급 사용법

### 증강 파이프라인 구성

```python
from data_augmentation.simple_augmentation import SimpleAugmenter
from data_augmentation.backtranslation import BackTranslationAugmenter

class AugmentationPipeline:
    """다중 증강 기법을 조합한 파이프라인"""
    
    def __init__(self):
        self.simple_augmenter = SimpleAugmenter(synonym_prob=0.1)
        self.back_translator = BackTranslationAugmenter(method="google")
    
    def augment_with_pipeline(self, dataset, simple_ratio=0.3, bt_ratio=0.2):
        # 1단계: 기본 증강
        simple_augmented = self.simple_augmenter.augment_dataset(
            dataset, 
            augment_ratio=simple_ratio
        )
        
        # 2단계: 백번역 증강
        final_dataset = self.back_translator.augment_dataset(
            simple_augmented,
            augment_ratio=bt_ratio
        )
        
        return final_dataset

# 파이프라인 사용
pipeline = AugmentationPipeline()
augmented_data = pipeline.augment_with_pipeline(
    original_dataset,
    simple_ratio=0.4,
    bt_ratio=0.2
)
```

### 배치 실험 실행

```python
# 여러 실험을 배치로 실행
experiments = [
    "config/experiments/kobart_baseline.yaml",
    "config/experiments/kogpt2_baseline.yaml",
    "config/experiments/mt5_baseline.yaml"
]

results = {}
for exp_config in experiments:
    print(f"실행 중: {exp_config}")
    result = runner.run_single_experiment(exp_config)
    results[exp_config] = result
    
    # 중간 결과 저장
    with open(f"results/batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

# 최종 결과 비교
best_experiment = max(results.items(), 
                     key=lambda x: x[1].get('best_metrics', {}).get('rouge_combined_f1', 0))

print(f"Best experiment: {best_experiment[0]}")
print(f"Best score: {best_experiment[1]['best_metrics']['rouge_combined_f1']:.4f}")
```

### 에러 처리 및 복구

```python
import time

def robust_experiment_runner(experiment_configs, max_retries=3):
    """에러 발생 시 재시도하는 견고한 실험 실행기"""
    results = {}
    
    for config_path in experiment_configs:
        retries = 0
        success = False
        
        while retries < max_retries and not success:
            try:
                print(f"실행 중 (시도 {retries + 1}): {config_path}")
                result = runner.run_single_experiment(config_path)
                
                if result['status'] == 'completed':
                    results[config_path] = result
                    success = True
                    print(f"✓ 성공: {config_path}")
                else:
                    raise Exception(f"실험 실패: {result.get('error', 'Unknown')}")
                    
            except Exception as e:
                retries += 1
                print(f"✗ 실패 (시도 {retries}): {e}")
                
                if retries < max_retries:
                    wait_time = 2 ** retries  # 지수 백오프
                    print(f"  {wait_time}초 후 재시도...")
                    time.sleep(wait_time)
                else:
                    results[config_path] = {
                        'status': 'failed',
                        'error': str(e),
                        'retries': retries
                    }
    
    return results

# 견고한 실험 실행
robust_results = robust_experiment_runner([
    "config/experiments/kobart_baseline.yaml",
    "config/experiments/kogpt2_baseline.yaml"
])
```

---

**관련 문서:**
- [메인 API 참조](./README.md)
- [실험 관리 가이드](../../02_user_guides/experiment_management/)
- [데이터 증강 가이드](../../02_user_guides/data_analysis/)
