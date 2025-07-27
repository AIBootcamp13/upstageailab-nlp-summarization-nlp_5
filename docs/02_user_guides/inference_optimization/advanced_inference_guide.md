# 고급 추론 엔진 사용 가이드

## 개요

`core/inference.py`의 InferenceEngine은 단순한 추론을 넘어서는 고급 기능들을 제공합니다. 이 가이드는 효율적이고 확장성 있는 추론을 위한 모든 고급 기능들을 다룹니다.

## 주요 고급 기능

### 1. 다중 입력 형식 지원
- **단일 텍스트**: 즉시 처리가 필요한 경우
- **리스트 배치**: 효율적인 대량 처리
- **DataFrame**: 구조화된 데이터 직접 처리

### 2. 자동 디바이스 최적화
- **플랫폼별 자동 감지**: CUDA, MPS (Apple Silicon), CPU
- **메모리 기반 배치 크기 조정**: 하드웨어에 따른 최적화
- **Mixed Precision 자동 설정**: 성능과 메모리 효율성

### 3. 배치 처리 최적화
- **DataLoader 기반 처리**: 메모리 효율적인 대용량 데이터 처리
- **동적 배치 크기**: 하드웨어 성능에 맞춰 자동 조정
- **진행률 추적**: 대용량 처리 시 상태 모니터링

### 4. 캐시 시스템
- **모델 캐시**: 반복 추론 시 성능 향상
- **토큰화 캐시**: 전처리 단계 최적화

## 상세 사용법

### 기본 설정 및 초기화

```python
from core.inference import InferenceEngine, InferenceConfig

# 기본 설정
config = InferenceConfig(
    model_path="gogamza/kobart-base-v2",
    batch_size=16,  # None으로 설정하면 자동 최적화
    max_source_length=1024,
    max_target_length=256,
    num_beams=5,
    length_penalty=1.0,
    early_stopping=True,
    use_cache=True,
    device=None,  # None으로 설정하면 자동 감지
    fp16=False    # None으로 설정하면 자동 설정
)

# 추론 엔진 초기화
engine = InferenceEngine(config)
```

### 자동 디바이스 최적화 활용

추론 엔진은 실행 환경을 자동으로 감지하고 최적화합니다:

```python
# 자동 최적화 활성화
config = InferenceConfig(
    model_path="your-model-path",
    device=None,        # 자동 감지
    batch_size=None,    # 자동 조정
    fp16=None          # 자동 설정
)

engine = InferenceEngine(config)

# 최적화 결과 확인
print(f"선택된 디바이스: {engine.device}")
print(f"최적화된 배치 크기: {engine.config.batch_size}")
print(f"FP16 활성화: {engine.config.fp16}")
```

**디바이스별 최적화 특징**:
- **CUDA**: 대용량 배치, FP16 활성화
- **MPS (Apple Silicon)**: 중간 배치, 메모리 효율성
- **CPU**: 소용량 배치, FP32 유지

### 다중 입력 형식 처리

#### 1. 단일 텍스트 처리

```python
# 즉시 처리가 필요한 단일 대화
dialogue = "#Person1#: 안녕하세요 오늘 날씨가 정말 좋네요. #Person2#: 네 맞아요. 산책하기 좋은 날씨입니다."

summary = engine.predict_single(dialogue)
print(f"요약: {summary}")
```

#### 2. 배치 리스트 처리

```python
# 여러 대화를 효율적으로 처리
dialogues = [
    "#Person1#: 안녕하세요 오늘 날씨가 정말 좋네요. #Person2#: 네 맞아요.",
    "#Person1#: 회의 시간이 언제였죠? #Person2#: 오후 3시입니다.",
    "#Person1#: 점심 뭐 드실래요? #Person2#: 한식이 좋겠어요."
]

# 진행률과 함께 배치 처리
summaries = engine.predict_batch(dialogues, show_progress=True)

for i, summary in enumerate(summaries):
    print(f"대화 {i+1}: {summary}")
```

#### 3. DataFrame 직접 처리

```python
import pandas as pd

# CSV 파일에서 직접 처리
df = pd.read_csv("data/test.csv")

# DataFrame 처리 (자동으로 배치 최적화 적용)
result_df = engine.predict_from_dataframe(
    df, 
    dialogue_column='dialogue',
    output_column='generated_summary',
    show_progress=True
)

# 결과 확인
print(result_df[['fname', 'generated_summary']].head())
```

#### 4. 통합 인터페이스 활용

```python
# __call__ 메서드로 다양한 입력 자동 처리
summary1 = engine("단일 대화 텍스트")
summaries = engine(["대화1", "대화2", "대화3"])
result_df = engine(dataframe)
```

### 배치 처리 최적화

#### 메모리 효율적인 대용량 처리

```python
# 대용량 데이터 처리를 위한 설정
config = InferenceConfig(
    model_path="your-model-path",
    batch_size=8,       # 메모리에 맞춰 조정
    num_workers=2,      # CPU 코어 활용
    fp16=True          # 메모리 절약
)

engine = InferenceEngine(config)

# 10,000개 대화 처리 예시
large_dialogues = load_large_dataset()  # 가상의 함수

summaries = engine.predict_batch(
    large_dialogues, 
    show_progress=True
)
```

#### 성능 모니터링

```python
import time

# 성능 측정
start_time = time.time()

summaries = engine.predict_batch(dialogues, show_progress=True)

end_time = time.time()
processing_time = end_time - start_time

print(f"처리 시간: {processing_time:.2f}초")
print(f"처리량: {len(dialogues)/processing_time:.2f} 대화/초")
print(f"평균 대화당 시간: {processing_time/len(dialogues):.3f}초")
```

### 생성 파라미터 최적화

#### 품질 중심 설정

```python
# 고품질 요약을 위한 설정
quality_config = InferenceConfig(
    model_path="your-model-path",
    num_beams=8,           # 더 많은 빔 탐색
    length_penalty=1.2,    # 적절한 길이 유도
    early_stopping=True,   # 효율성 유지
    max_target_length=512  # 더 긴 요약 허용
)
```

#### 속도 중심 설정

```python
# 빠른 처리를 위한 설정
speed_config = InferenceConfig(
    model_path="your-model-path",
    num_beams=1,          # 탐욕적 디코딩
    early_stopping=False,  # 조기 종료 비활성화
    use_cache=True,       # 캐시 활용
    max_target_length=128  # 짧은 요약
)
```

### 대회 제출 형식 처리

#### 제출 파일 생성

```python
# 테스트 데이터 처리
test_df = pd.read_csv("data/test.csv")

# 추론 수행
result_df = engine.predict_from_dataframe(
    test_df,
    dialogue_column='dialogue',
    output_column='summary'
)

# 제출 형식으로 저장
engine.save_submission(
    result_df,
    output_path="outputs/submission.csv",
    fname_column='fname',
    summary_column='summary'
)
```

#### 제출 파일 검증

```python
# 제출 파일 형식 확인
submission = pd.read_csv("outputs/submission.csv")

print(f"제출 파일 형태: {submission.shape}")
print(f"필요한 컬럼: {list(submission.columns)}")
print(f"fname 유니크 개수: {submission['fname'].nunique()}")
print(f"빈 요약 개수: {submission['summary'].isna().sum()}")
```

## 성능 최적화 팁

### 1. 하드웨어별 최적 설정

#### CUDA GPU 환경
```python
cuda_config = InferenceConfig(
    model_path="your-model-path",
    batch_size=32,    # 큰 배치 크기
    fp16=True,        # Mixed Precision
    use_cache=True,   # GPU 메모리 캐시 활용
    num_workers=4     # 다중 프로세싱
)
```

#### Apple Silicon (M1/M2) 환경
```python
mps_config = InferenceConfig(
    model_path="your-model-path",
    batch_size=16,    # 중간 배치 크기
    fp16=False,       # MPS에서 FP16 안정성 이슈
    use_cache=True,   # 메모리 효율성
    num_workers=2
)
```

#### CPU 환경
```python
cpu_config = InferenceConfig(
    model_path="your-model-path",
    batch_size=4,     # 작은 배치 크기
    fp16=False,       # CPU는 FP32 사용
    use_cache=True,   
    num_workers=1     # CPU 부하 방지
)
```

### 2. 메모리 관리

#### 메모리 부족 시 대처

```python
# 메모리 부족 시 배치 크기 자동 조정
def adaptive_batch_processing(engine, dialogues, initial_batch_size=32):
    current_batch_size = initial_batch_size
    
    while current_batch_size >= 1:
        try:
            # 배치 크기 업데이트
            engine.config.batch_size = current_batch_size
            
            # 처리 시도
            summaries = engine.predict_batch(dialogues[:10])  # 테스트
            
            # 성공 시 전체 처리
            return engine.predict_batch(dialogues, show_progress=True)
            
        except RuntimeError as e:
            if "memory" in str(e).lower():
                current_batch_size //= 2
                print(f"메모리 부족. 배치 크기를 {current_batch_size}로 줄입니다.")
                torch.cuda.empty_cache()  # GPU 메모리 정리
            else:
                raise e
    
    raise RuntimeError("배치 크기를 1로 줄여도 메모리 부족")
```

### 3. 처리 시간 최적화

#### 병렬 처리 활용

```python
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def parallel_inference(engine, dialogues, n_workers=4):
    """CPU 병렬 처리를 통한 추론 가속화"""
    
    # 데이터를 워커 수만큼 분할
    chunks = np.array_split(dialogues, n_workers)
    
    def process_chunk(chunk):
        return engine.predict_batch(chunk.tolist(), show_progress=False)
    
    # 병렬 처리
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(process_chunk, chunks))
    
    # 결과 병합
    return [summary for chunk_result in results for summary in chunk_result]
```

## 고급 사용 패턴

### 1. 캐시 활용 최적화

```python
# 반복 추론 시 캐시 효과 극대화
class CachedInferenceEngine:
    def __init__(self, config):
        self.engine = InferenceEngine(config)
        self.result_cache = {}
    
    def predict_with_cache(self, dialogue):
        # 해시 기반 캐시
        dialogue_hash = hash(dialogue)
        
        if dialogue_hash in self.result_cache:
            return self.result_cache[dialogue_hash]
        
        summary = self.engine.predict_single(dialogue)
        self.result_cache[dialogue_hash] = summary
        
        return summary
```

### 2. 동적 설정 조정

```python
class AdaptiveInferenceEngine:
    def __init__(self, base_config):
        self.base_config = base_config
        self.engine = InferenceEngine(base_config)
        
    def predict_adaptive(self, dialogues, quality_priority=False):
        """품질 우선도에 따른 동적 설정 조정"""
        
        if quality_priority:
            # 고품질 모드
            self.engine.config.num_beams = 8
            self.engine.config.length_penalty = 1.2
            self.engine.config.batch_size = min(self.engine.config.batch_size, 8)
        else:
            # 속도 모드
            self.engine.config.num_beams = 1
            self.engine.config.length_penalty = 1.0
            self.engine.config.batch_size = self.base_config.batch_size
        
        return self.engine.predict_batch(dialogues)
```

### 3. 실시간 모니터링

```python
import psutil
import GPUtil

class MonitoredInferenceEngine:
    def __init__(self, config):
        self.engine = InferenceEngine(config)
        
    def predict_with_monitoring(self, dialogues):
        """시스템 리소스 모니터링과 함께 추론"""
        
        # 초기 상태 기록
        initial_memory = psutil.virtual_memory().percent
        if torch.cuda.is_available():
            initial_gpu_memory = torch.cuda.memory_allocated() / 1024**3
        
        print(f"추론 시작 - RAM: {initial_memory:.1f}%")
        if torch.cuda.is_available():
            print(f"GPU 메모리: {initial_gpu_memory:.2f}GB")
        
        # 추론 실행
        start_time = time.time()
        summaries = self.engine.predict_batch(dialogues, show_progress=True)
        end_time = time.time()
        
        # 최종 상태 기록
        final_memory = psutil.virtual_memory().percent
        processing_time = end_time - start_time
        
        print(f"추론 완료 - 처리 시간: {processing_time:.2f}초")
        print(f"최종 RAM: {final_memory:.1f}% (증가: {final_memory - initial_memory:.1f}%)")
        print(f"처리량: {len(dialogues)/processing_time:.2f} 대화/초")
        
        return summaries
```

## 문제 해결 가이드

### 일반적인 오류와 해결책

#### 1. CUDA Out of Memory
```python
# 해결 방법 1: 배치 크기 줄이기
config.batch_size = 4

# 해결 방법 2: FP16 사용
config.fp16 = True

# 해결 방법 3: 메모리 정리
torch.cuda.empty_cache()
```

#### 2. MPS (Apple Silicon) 관련 오류
```python
# MPS 특화 설정
if torch.backends.mps.is_available():
    config.fp16 = False  # MPS에서 FP16 이슈 회피
    config.batch_size = min(config.batch_size, 16)
```

#### 3. 모델 로딩 오류
```python
# 경로 확인
from utils.path_utils import PathManager

model_path = PathManager.resolve_path("your-model-path")
if not model_path.exists():
    print(f"모델 경로 오류: {model_path}")
    # Hugging Face Hub에서 자동 다운로드 시도
```

### 성능 벤치마킹

```python
def benchmark_inference_engine(engine, test_dialogues, iterations=3):
    """추론 엔진 성능 벤치마크"""
    
    results = []
    
    for i in range(iterations):
        start_time = time.time()
        summaries = engine.predict_batch(test_dialogues, show_progress=False)
        end_time = time.time()
        
        processing_time = end_time - start_time
        throughput = len(test_dialogues) / processing_time
        
        results.append({
            'iteration': i + 1,
            'processing_time': processing_time,
            'throughput': throughput,
            'avg_time_per_dialogue': processing_time / len(test_dialogues)
        })
    
    # 평균 성능 계산
    avg_throughput = sum(r['throughput'] for r in results) / len(results)
    avg_time_per_dialogue = sum(r['avg_time_per_dialogue'] for r in results) / len(results)
    
    print(f"평균 처리량: {avg_throughput:.2f} 대화/초")
    print(f"평균 대화당 처리 시간: {avg_time_per_dialogue:.3f}초")
    
    return results
```

## 다음 단계

이 가이드를 통해 InferenceEngine의 고급 기능들을 마스터했다면, 다음을 고려해보세요:

1. **[성능 최적화 가이드](../performance_optimization.md)**: 더 깊이 있는 최적화 기법
2. **[실험 관리 가이드](../experiment_management/)**: 체계적인 실험 설계
3. **[API 참조 문서](../../03_technical_docs/api_reference/core_modules.md)**: 전체 API 상세 정보

## 요약

InferenceEngine의 고급 기능들을 활용하면:

- ✅ **다양한 입력 형식 지원**으로 개발 편의성 증대
- ✅ **자동 디바이스 최적화**로 플랫폼별 최적 성능
- ✅ **배치 처리 최적화**로 대용량 데이터 효율적 처리
- ✅ **캐시 시스템**으로 반복 작업 가속화
- ✅ **메모리 관리**로 안정적인 대용량 처리

각 기능을 단계적으로 도입하여 점진적으로 성능을 향상시키세요.
