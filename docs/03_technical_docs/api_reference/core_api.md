# Core 모듈 API 참조

`core` 모듈은 독립적인 추론 엔진을 제공하며, 배치 처리와 다양한 입력 형식을 지원합니다.

## 목차

1. [InferenceEngine](#inferenceengine)
2. [InferenceConfig](#inferenceconfig)
3. [유틸리티 함수](#유틸리티-함수)

## InferenceEngine

독립 추론 엔진 클래스로 모델 로딩, 단일/배치 예측, DataFrame 처리 등의 기능을 제공합니다.

### 클래스 정의

```python
class InferenceEngine:
    """
    독립 추론 엔진
    
    모델 로드, 단일/배치 예측, DataFrame 처리 등의 기능을 제공합니다.
    """
```

### 생성자

```python
def __init__(self, config: Union[InferenceConfig, Dict[str, Any]]):
```

**Parameters:**
- `config` (Union[InferenceConfig, Dict[str, Any]]): 추론 설정 객체 또는 딕셔너리

**Example:**
```python
from core.inference import InferenceEngine, InferenceConfig

# InferenceConfig 객체로 생성
config = InferenceConfig(
    model_path="models/best_model",
    batch_size=16,
    max_target_length=256
)
engine = InferenceEngine(config)

# 딕셔너리로 생성
config_dict = {
    "model_path": "models/best_model",
    "batch_size": 16,
    "num_beams": 5,
    "max_target_length": 256
}
engine = InferenceEngine(config_dict)
```

### 주요 메서드

#### predict_single()

단일 대화에 대한 요약을 생성합니다.

```python
def predict_single(self, dialogue: str) -> str:
```

**Parameters:**
- `dialogue` (str): 대화 텍스트

**Returns:**
- `str`: 생성된 요약

**Example:**
```python
dialogue = """
A: 오늘 점심 뭐 먹을까?
B: 한식이 어때? 김치찌개 같은 거.
A: 좋아! 근처에 맛있는 집 알아?
B: 응, 학교 뒤쪽에 할머니가 하시는 집 있어.
"""

summary = engine.predict_single(dialogue)
print(f"요약: {summary}")
# 출력: 요약: 두 사람이 점심 메뉴로 한식을 정하고 맛집을 추천했다.
```

#### predict_batch()

여러 대화에 대한 배치 추론을 수행합니다.

```python
def predict_batch(self, dialogues: List[str], show_progress: bool = True) -> List[str]:
```

**Parameters:**
- `dialogues` (List[str]): 대화 텍스트 리스트
- `show_progress` (bool): 진행률 표시 여부. 기본값: True

**Returns:**
- `List[str]`: 생성된 요약 리스트

**Example:**
```python
dialogues = [
    "A: 오늘 날씨 어때? B: 맑고 좋아!",
    "A: 과제 언제까지야? B: 내일까지 제출해야 해.",
    "A: 저녁 같이 먹을까? B: 좋아, 몇 시에?"
]

summaries = engine.predict_batch(dialogues, show_progress=True)
for i, summary in enumerate(summaries):
    print(f"대화 {i+1}: {summary}")

# 출력:
# 대화 1: 날씨에 대한 대화
# 대화 2: 과제 마감일 확인
# 대화 3: 저녁 약속 제안
```

#### predict_from_dataframe()

DataFrame에서 직접 추론을 수행합니다.

```python
def predict_from_dataframe(self, df: pd.DataFrame, 
                         dialogue_column: str = 'dialogue',
                         output_column: str = 'summary',
                         show_progress: bool = True) -> pd.DataFrame:
```

**Parameters:**
- `df` (pd.DataFrame): 입력 DataFrame
- `dialogue_column` (str): 대화가 포함된 컬럼명. 기본값: 'dialogue'
- `output_column` (str): 생성된 요약을 저장할 컬럼명. 기본값: 'summary'
- `show_progress` (bool): 진행률 표시 여부. 기본값: True

**Returns:**
- `pd.DataFrame`: 요약이 추가된 DataFrame

**Raises:**
- `ValueError`: 지정된 대화 컬럼이 존재하지 않을 때

**Example:**
```python
import pandas as pd

# 입력 데이터 준비
data = {
    'id': [1, 2, 3],
    'dialogue': [
        "A: 안녕하세요. B: 안녕하세요!",
        "A: 오늘 일정 어때요? B: 바쁩니다.",
        "A: 커피 마실까요? B: 좋아요!"
    ]
}
df = pd.DataFrame(data)

# 추론 실행
result_df = engine.predict_from_dataframe(
    df, 
    dialogue_column='dialogue', 
    output_column='generated_summary'
)

print(result_df)
#    id                    dialogue generated_summary
# 0   1        A: 안녕하세요. B: 안녕하세요!              인사
# 1   2  A: 오늘 일정 어때요? B: 바쁩니다.         일정 문의
# 2   3    A: 커피 마실까요? B: 좋아요!         커피 제안
```

#### save_submission()

대회 제출 형식으로 결과를 저장합니다.

```python
def save_submission(self, df: pd.DataFrame, output_path: str,
                   fname_column: str = 'fname',
                   summary_column: str = 'summary'):
```

**Parameters:**
- `df` (pd.DataFrame): 결과 DataFrame
- `output_path` (str): 저장 경로
- `fname_column` (str): 파일명 컬럼. 기본값: 'fname'
- `summary_column` (str): 요약 컬럼. 기본값: 'summary'

**Example:**
```python
# 대회 형식 데이터
submission_data = {
    'fname': ['test_001.txt', 'test_002.txt', 'test_003.txt'],
    'dialogue': ['대화1', '대화2', '대화3']
}
df = pd.DataFrame(submission_data)

# 추론 실행
result_df = engine.predict_from_dataframe(df, dialogue_column='dialogue')

# 제출 파일 저장
engine.save_submission(
    result_df, 
    output_path='submissions/submission.csv',
    fname_column='fname',
    summary_column='summary'
)
```

#### __call__()

다양한 입력 형식을 자동으로 처리하는 호출 가능 인터페이스입니다.

```python
def __call__(self, dialogue: Union[str, List[str], pd.DataFrame], **kwargs):
```

**Parameters:**
- `dialogue` (Union[str, List[str], pd.DataFrame]): 대화 텍스트, 리스트, 또는 DataFrame
- `**kwargs`: 추가 인자

**Returns:**
- 입력 타입에 따른 적절한 결과

**Example:**
```python
# 단일 문자열
result1 = engine("A: 안녕하세요. B: 안녕하세요!")

# 리스트
result2 = engine(["대화1", "대화2", "대화3"])

# DataFrame
df = pd.DataFrame({'dialogue': ["대화1", "대화2"]})
result3 = engine(df, dialogue_column='dialogue')
```

### 속성

#### 주요 속성

```python
@property
def config(self) -> InferenceConfig:
    """추론 설정"""

@property
def device(self) -> torch.device:
    """연산 디바이스"""

@property
def model(self) -> PreTrainedModel:
    """추론 모델"""

@property
def tokenizer(self) -> PreTrainedTokenizer:
    """토크나이저"""

@property
def model_type(self) -> str:
    """모델 타입 ('seq2seq' 또는 'causal')"""
```

**Example:**
```python
print(f"사용 디바이스: {engine.device}")
print(f"모델 타입: {engine.model_type}")
print(f"배치 크기: {engine.config.batch_size}")
print(f"최대 출력 길이: {engine.config.max_target_length}")

# 모델 정보
total_params = sum(p.numel() for p in engine.model.parameters())
print(f"모델 파라미터 수: {total_params:,}")
```

## InferenceConfig

추론 설정을 담는 데이터 클래스입니다.

### 클래스 정의

```python
@dataclass
class InferenceConfig:
    """추론 설정"""
    model_path: str
    batch_size: int = 8
    max_source_length: int = 1024
    max_target_length: int = 256
    num_beams: int = 5
    length_penalty: float = 1.0
    early_stopping: bool = True
    use_cache: bool = True
    device: Optional[str] = None
    fp16: bool = False
    num_workers: int = 0
```

### 파라미터

- `model_path` (str): **필수**. 모델 경로 또는 HuggingFace 모델명
- `batch_size` (int): 배치 크기. 기본값: 8
- `max_source_length` (int): 최대 입력 길이. 기본값: 1024
- `max_target_length` (int): 최대 출력 길이. 기본값: 256
- `num_beams` (int): 빔 서치 빔 수. 기본값: 5
- `length_penalty` (float): 길이 페널티. 기본값: 1.0
- `early_stopping` (bool): 조기 종료 여부. 기본값: True
- `use_cache` (bool): 캐시 사용 여부. 기본값: True
- `device` (Optional[str]): 디바이스 지정 ('cuda', 'cpu', 'mps'). None이면 자동 감지
- `fp16` (bool): 혼합 정밀도 사용 여부. 기본값: False
- `num_workers` (int): DataLoader 워커 수. 기본값: 0

### 사용 예제

```python
from core.inference import InferenceConfig

# 기본 설정
config = InferenceConfig(model_path="models/kobart-base")

# 커스텀 설정
config = InferenceConfig(
    model_path="gogamza/kobart-base-v2",
    batch_size=32,
    max_target_length=128,
    num_beams=3,
    length_penalty=0.8,
    device="cuda:0",
    fp16=True
)

# 고성능 설정 (큰 배치, 빠른 추론)
fast_config = InferenceConfig(
    model_path="models/best_model",
    batch_size=64,
    max_target_length=64,
    num_beams=1,  # Greedy decoding
    early_stopping=False,
    use_cache=True,
    fp16=True
)

# 고품질 설정 (작은 배치, 정확한 추론)
quality_config = InferenceConfig(
    model_path="models/best_model",
    batch_size=4,
    max_target_length=512,
    num_beams=10,
    length_penalty=1.2,
    early_stopping=True
)
```

### 디바이스별 권장 설정

```python
# CUDA 사용 시
cuda_config = InferenceConfig(
    model_path="models/best_model",
    batch_size=32,
    fp16=True,
    device="cuda"
)

# MPS (Apple Silicon) 사용 시
mps_config = InferenceConfig(
    model_path="models/best_model",
    batch_size=16,
    fp16=False,  # MPS는 fp16 제한적 지원
    device="mps"
)

# CPU 사용 시
cpu_config = InferenceConfig(
    model_path="models/best_model",
    batch_size=4,
    fp16=False,
    device="cpu",
    num_workers=4
)
```

## 유틸리티 함수

### create_inference_engine()

추론 엔진 생성을 위한 편의 함수입니다.

```python
def create_inference_engine(model_path: str, **kwargs) -> InferenceEngine:
```

**Parameters:**
- `model_path` (str): 모델 경로
- `**kwargs`: InferenceConfig의 추가 파라미터

**Returns:**
- `InferenceEngine`: 초기화된 추론 엔진

**Example:**
```python
from core.inference import create_inference_engine

# 간단한 생성
engine = create_inference_engine("models/best_model")

# 파라미터와 함께 생성
engine = create_inference_engine(
    model_path="gogamza/kobart-base-v2",
    batch_size=16,
    max_target_length=128,
    num_beams=3
)
```

## 고급 사용법

### 배치 처리 최적화

```python
import time

def benchmark_batch_sizes(engine, dialogues, batch_sizes=[4, 8, 16, 32]):
    """배치 크기별 성능 벤치마크"""
    results = {}
    
    for batch_size in batch_sizes:
        # 설정 업데이트
        engine.config.batch_size = batch_size
        
        # 시간 측정
        start_time = time.time()
        summaries = engine.predict_batch(dialogues, show_progress=False)
        elapsed_time = time.time() - start_time
        
        results[batch_size] = {
            'time': elapsed_time,
            'throughput': len(dialogues) / elapsed_time
        }
        
        print(f"Batch size {batch_size}: {elapsed_time:.2f}s, "
              f"{results[batch_size]['throughput']:.2f} samples/s")
    
    return results

# 벤치마크 실행
test_dialogues = ["테스트 대화"] * 100
benchmark_results = benchmark_batch_sizes(engine, test_dialogues)
```

### 메모리 효율적인 대용량 처리

```python
def process_large_dataset(engine, large_df, chunk_size=1000):
    """대용량 데이터셋을 청크 단위로 처리"""
    results = []
    
    for i in range(0, len(large_df), chunk_size):
        chunk = large_df.iloc[i:i+chunk_size]
        
        print(f"Processing chunk {i//chunk_size + 1}/{len(large_df)//chunk_size + 1}")
        
        # 청크 처리
        chunk_result = engine.predict_from_dataframe(
            chunk, 
            show_progress=False
        )
        
        results.append(chunk_result)
        
        # 메모리 정리
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
    
    # 결과 결합
    final_result = pd.concat(results, ignore_index=True)
    return final_result

# 대용량 처리 실행
# large_df = pd.read_csv("large_dataset.csv")
# result = process_large_dataset(engine, large_df, chunk_size=500)
```

### 캐시 시스템 활용

```python
import pickle
from pathlib import Path

class CachedInferenceEngine:
    """캐시 기능이 있는 추론 엔진 래퍼"""
    
    def __init__(self, engine, cache_dir="cache/"):
        self.engine = engine
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, dialogue):
        """캐시 키 생성"""
        import hashlib
        return hashlib.md5(dialogue.encode()).hexdigest()
    
    def predict_single_cached(self, dialogue):
        """캐시된 단일 예측"""
        cache_key = self._get_cache_key(dialogue)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # 캐시 확인
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # 예측 실행
        result = self.engine.predict_single(dialogue)
        
        # 캐시 저장
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        
        return result

# 캐시 엔진 사용
cached_engine = CachedInferenceEngine(engine)
result = cached_engine.predict_single_cached("반복되는 대화")
```

## 에러 처리

### 일반적인 예외 처리

```python
from core.inference import InferenceEngine, InferenceConfig

try:
    config = InferenceConfig(model_path="models/nonexistent_model")
    engine = InferenceEngine(config)
    
except FileNotFoundError:
    print("모델 파일을 찾을 수 없습니다.")
    
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("GPU 메모리 부족. 배치 크기를 줄여주세요.")
        # 배치 크기 감소 후 재시도
        config.batch_size //= 2
        engine = InferenceEngine(config)
    else:
        print(f"런타임 에러: {e}")
        
except Exception as e:
    print(f"예상치 못한 에러: {e}")

# 추론 중 에러 처리
try:
    result = engine.predict_single("입력 대화")
    
except torch.cuda.OutOfMemoryError:
    print("GPU 메모리 부족으로 CPU로 전환합니다.")
    engine.device = torch.device('cpu')
    engine.model = engine.model.to('cpu')
    result = engine.predict_single("입력 대화")
    
except Exception as e:
    print(f"추론 중 에러 발생: {e}")
    result = "요약 생성 실패"
```

---

**관련 문서:**
- [Trainer API 참조](./trainer_api.md)
- [Utils API 참조](./utils_api.md)
- [추론 최적화 가이드](../02_user_guides/inference_optimization/)
