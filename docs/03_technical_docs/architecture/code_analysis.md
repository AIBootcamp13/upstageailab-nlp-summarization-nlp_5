# 🔍 코드 분석 가이드

프로젝트의 주요 컴포넌트들에 대한 상세한 코드 분석과 구현 설명을 제공합니다.

## 📋 목차
1. [전체 아키텍처 분석](#전체-아키텍처-분석)
2. [데이터 파이프라인 분석](#데이터-파이프라인-분석)
3. [모델 구현 분석](#모델-구현-분석)
4. [학습 시스템 분석](#학습-시스템-분석)
5. [추론 엔진 분석](#추론-엔진-분석)
6. [성능 최적화 분석](#성능-최적화-분석)

---

## 전체 아키텍처 분석

### 레이어드 아키텍처 구조

```
┌─────────────────────────────────┐
│        Presentation Layer       │  ← CLI, API Endpoints
├─────────────────────────────────┤
│        Application Layer        │  ← trainers.py, inference.py
├─────────────────────────────────┤
│          Domain Layer           │  ← core/models, core/training
├─────────────────────────────────┤
│      Infrastructure Layer      │  ← utils/, config/
└─────────────────────────────────┘
```

**설계 원칙:**
- **관심사 분리**: 각 레이어는 명확한 책임
- **의존성 역전**: 상위 레이어가 하위 레이어에 의존
- **상대 경로 강제**: 모든 파일 접근은 프로젝트 루트 기준

### 핵심 디자인 패턴

#### 1. Factory Pattern (모델 생성)
```python
# core/models/model_factory.py
class ModelFactory:
    SUPPORTED_MODELS = {
        "kobart": {
            "base_model": "gogamza/kobart-base-v2",
            "tokenizer_class": "BartTokenizer",
            "model_class": "BartForConditionalGeneration"
        }
    }
    
    @classmethod
    def create_model_and_tokenizer(cls, model_name: str, device: str):
        """팩토리 메서드로 모델과 토크나이저 생성"""
        config = cls.SUPPORTED_MODELS[model_name]
        
        # 동적 클래스 로딩
        tokenizer_class = getattr(transformers, config["tokenizer_class"])
        model_class = getattr(transformers, config["model_class"])
        
        # 인스턴스 생성 및 디바이스 최적화
        tokenizer = tokenizer_class.from_pretrained(config["base_model"])
        model = model_class.from_pretrained(config["base_model"])
        
        return model.to(device), tokenizer, config
```

**장점:**
- 새로운 모델 추가 시 코드 수정 최소화
- 일관된 모델 로딩 인터페이스
- 설정 기반 모델 관리

#### 2. Strategy Pattern (디바이스 최적화)
```python
# utils/device_utils.py
class DeviceStrategy:
    """디바이스별 최적화 전략"""
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def optimize_model(self, model):
        pass

class CudaStrategy(DeviceStrategy):
    def get_config(self):
        return {
            "batch_size": 16,
            "fp16": True,
            "dataloader_pin_memory": True
        }
    
    def optimize_model(self, model):
        return model.cuda().half()  # FP16 최적화

class MPSStrategy(DeviceStrategy):
    def get_config(self):
        return {
            "batch_size": 8,
            "fp16": False,  # MPS FP16 이슈 회피
            "dataloader_pin_memory": False
        }
    
    def optimize_model(self, model):
        return model.to("mps")
```

#### 3. Observer Pattern (실험 추적)
```python
# utils/experiment_utils.py
class ExperimentObserver:
    """실험 상태 변화 감지"""
    
    def __init__(self):
        self.observers = []
    
    def attach(self, observer):
        self.observers.append(observer)
    
    def notify(self, event, data):
        for observer in self.observers:
            observer.update(event, data)

class WandBObserver:
    def update(self, event, data):
        if event == "metrics_updated":
            wandb.log(data)

class FileObserver:
    def update(self, event, data):
        if event == "experiment_completed":
            self.save_results(data)
```

---

## 데이터 파이프라인 분석

### Multi-Reference 데이터 처리

```python
# utils/data_utils.py
class DataProcessor:
    """다중 참조 요약 데이터 전용 프로세서"""
    
    def _detect_and_convert_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """다중 참조 형식 자동 감지 및 표준화"""
        
        # 형식 1: 개별 컬럼 (summary1, summary2, summary3)
        if all(col in df.columns for col in ['summary1', 'summary2', 'summary3']):
            df['summaries'] = df[['summary1', 'summary2', 'summary3']].apply(
                lambda x: [str(val) if pd.notna(val) else "" for val in x], axis=1
            )
            
        # 형식 2: 구분자 분리 (summary 컬럼에 ||| 구분자)
        elif 'summary' in df.columns:
            df['summaries'] = df['summary'].apply(self._parse_multiple_summaries)
            
        return df
```

**핵심 설계 결정:**
- **유연한 입력 형식**: 3가지 다중 참조 형식 자동 감지
- **표준화된 출력**: 항상 `summaries` 컬럼에 리스트 형태로 저장
- **에러 처리**: 잘못된 형식에 대한 명확한 에러 메시지

### 데이터 검증 체계

```python
def validate_submission_format(self, file_path: Union[str, Path]) -> bool:
    """제출 파일 형식 검증"""
    
    try:
        df = pd.read_csv(full_path)
        
        # 1. 필수 컬럼 확인
        required_columns = ['fname', 'summary']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            self.logger.error(f"필수 컬럼 누락: {missing_columns}")
            return False
        
        # 2. 데이터 타입 확인
        if not all(isinstance(fname, str) for fname in df['fname']):
            self.logger.error("fname 컬럼이 문자열 타입이 아닙니다")
            return False
        
        # 3. 빈 값 확인
        if df['summary'].isna().any():
            self.logger.error("summary 컬럼에 빈 값이 있습니다")
            return False
            
        return True
        
    except Exception as e:
        self.logger.error(f"파일 검증 중 오류: {e}")
        return False
```

---

## 모델 구현 분석

### 대화 요약 모델 래퍼

```python
# core/models/summarization_model.py
class DialogueSummarizationModel:
    """대화 요약 전용 모델 래퍼"""
    
    def __init__(self, model_name: str, device: str = "auto"):
        # 디바이스 자동 감지
        if device == "auto":
            device = get_optimal_device()
        
        self.device = device
        self.model, self.tokenizer, self.config = ModelFactory.create_model_and_tokenizer(
            model_name, device
        )
        
        # 대화 요약 특화 설정
        self._setup_generation_config()
        self._setup_special_tokens()
    
    def _setup_generation_config(self):
        """생성 설정 최적화"""
        self.generation_config = {
            "max_length": 128,
            "min_length": 10,
            "num_beams": 4,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
            "do_sample": False
        }
    
    def preprocess_dialogue(self, dialogue: str) -> str:
        """대화 전처리"""
        
        # 기본 정리
        dialogue = dialogue.strip()
        
        # 화자 구분 정규화 (옵션)
        dialogue = re.sub(r'화자\s*(\d+)\s*:', r'<speaker\1>:', dialogue)
        
        # 과도한 공백 정리
        dialogue = re.sub(r'\s+', ' ', dialogue)
        
        return dialogue
```

**설계 특징:**
- **어댑터 패턴**: Transformers 모델을 대화 요약에 특화
- **템플릿 메서드**: 공통 전처리 로직과 모델별 커스터마이징 분리
- **설정 주입**: 생성 파라미터를 외부에서 주입 가능

---

## 학습 시스템 분석

### 적응형 배치 처리

```python
# core/training/adaptive_trainer.py
class AdaptiveBatchProcessor:
    """동적 배치 크기 조정 프로세서"""
    
    def __init__(self, device: str, initial_batch_size: int = 8):
        self.device = device
        self.current_batch_size = initial_batch_size
        self.max_batch_size = self._get_max_batch_size()
        
        # 성능 추적
        self.success_count = 0
        self.oom_count = 0
    
    def process_batch(self, data_loader, process_fn):
        """적응형 배치 처리"""
        
        results = []
        
        for batch in data_loader:
            try:
                # 배치 처리 시도
                batch_results = process_fn(batch)
                results.extend(batch_results)
                
                self.success_count += 1
                
                # 성공 시 배치 크기 증가 고려
                if (self.success_count % 10 == 0 and 
                    self.current_batch_size < self.max_batch_size):
                    self.current_batch_size = min(
                        self.current_batch_size + 1, 
                        self.max_batch_size
                    )
                
            except torch.cuda.OutOfMemoryError:
                # OOM 시 배치 크기 감소
                self._handle_oom()
                continue
        
        return results
    
    def _handle_oom(self):
        """OOM 처리 로직"""
        self.oom_count += 1
        old_size = self.current_batch_size
        self.current_batch_size = max(
            self.current_batch_size // 2, 
            1  # 최소 배치 크기
        )
        
        # 메모리 정리
        if self.device == "cuda":
            torch.cuda.empty_cache()
```

**핵심 알고리즘:**
1. **점진적 증가**: 성공 시 배치 크기를 천천히 증가
2. **급격한 감소**: OOM 시 배치 크기를 절반으로 감소
3. **메모리 관리**: 실패 시 즉시 캐시 정리

---

## 추론 엔진 분석

### 배치 추론 최적화

```python
# core/inference/inference_engine.py
class InferenceEngine:
    """최적화된 추론 엔진"""
    
    def predict_batch(self, dialogues: List[str], batch_size: int = 8, **kwargs) -> List[str]:
        """배치 추론 실행"""
        
        # 디바이스별 배치 크기 자동 조정
        device_config = get_device_config(self.device)
        optimal_batch_size = min(batch_size, device_config['batch_size'])
        
        results = []
        
        for i in tqdm(range(0, len(dialogues), optimal_batch_size)):
            batch_dialogues = dialogues[i:i + optimal_batch_size]
            
            # 배치 토크나이징
            inputs = self.tokenizer(
                batch_dialogues,
                max_length=self.max_input_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # 배치 추론
            with torch.no_grad():
                if self.device == "mps":
                    # MPS 최적화
                    with torch.autocast(device_type="cpu", enabled=False):
                        outputs = self.model.generate(**inputs, **self.generation_config)
                else:
                    outputs = self.model.generate(**inputs, **self.generation_config)
            
            # 배치 디코딩
            batch_summaries = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            
            results.extend([summary.strip() for summary in batch_summaries])
        
        return results
```

**최적화 기법:**
- **디바이스별 배치 크기**: 메모리에 맞는 자동 조정
- **MPS 특별 처리**: Apple Silicon의 autocast 이슈 회피
- **메모리 효율성**: `torch.no_grad()` 컨텍스트 사용

---

## 성능 최적화 분석

### 메모리 프로파일링

```python
# utils/performance.py
def profile_memory_usage(func):
    """메모리 사용량 프로파일링 데코레이터"""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import psutil
        import torch
        
        # 시작 메모리
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        if torch.cuda.is_available():
            gpu_memory_before = torch.cuda.memory_allocated() / 1024 / 1024
        else:
            gpu_memory_before = 0
        
        try:
            # 함수 실행
            result = func(*args, **kwargs)
            
            # 종료 메모리
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_diff = memory_after - memory_before
            
            if torch.cuda.is_available():
                gpu_memory_after = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_memory_diff = gpu_memory_after - gpu_memory_before
            else:
                gpu_memory_diff = 0
            
            print(f"💾 {func.__name__} 메모리 사용량:")
            print(f"  System: {memory_diff:+.1f} MB")
            if gpu_memory_diff:
                print(f"  GPU: {gpu_memory_diff:+.1f} MB")
            
            return result
            
        finally:
            # 가비지 컬렉션
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return wrapper
```

### 성능 병목 지점 분석

**1. 데이터 로딩 병목**
```python
# 개선 전: 순차 처리
def load_data_slow(file_paths):
    data = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        data.append(df)
    return pd.concat(data)

# 개선 후: 병렬 처리
def load_data_fast(file_paths):
    with ThreadPoolExecutor(max_workers=4) as executor:
        dfs = list(executor.map(pd.read_csv, file_paths))
    return pd.concat(dfs)
```

**2. 토크나이징 병목**
```python
# 개선 전: 반복 토크나이징
def tokenize_slow(dialogues, tokenizer):
    tokens = []
    for dialogue in dialogues:
        token = tokenizer(dialogue, return_tensors="pt")
        tokens.append(token)
    return tokens

# 개선 후: 배치 토크나이징
def tokenize_fast(dialogues, tokenizer):
    return tokenizer(
        dialogues, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
```

**3. 모델 추론 병목**
```python
# 개선 전: 개별 추론
def predict_slow(model, inputs):
    results = []
    for input_tensor in inputs:
        with torch.no_grad():
            output = model(input_tensor.unsqueeze(0))
        results.append(output)
    return results

# 개선 후: 배치 추론
def predict_fast(model, inputs):
    with torch.no_grad():
        batch_inputs = torch.stack(inputs)
        batch_outputs = model(batch_inputs)
    return batch_outputs
```

---

## 🔧 코드 품질 분석

### 1. 복잡도 분석

**순환 복잡도 (Cyclomatic Complexity):**
- `DataProcessor._detect_and_convert_format()`: 4 (양호)
- `InferenceEngine.predict_batch()`: 6 (보통)
- `ExperimentTracker.log_metrics()`: 3 (양호)

**개선 권장사항:**
```python
# 복잡도가 높은 함수 분해
def complex_function(data):
    # Before: 높은 복잡도
    if condition1:
        if condition2:
            if condition3:
                # 복잡한 로직
                pass
    
    # After: 함수 분해
    if not self._validate_conditions(data):
        return None
    
    return self._process_data(data)

def _validate_conditions(self, data):
    return condition1 and condition2 and condition3

def _process_data(self, data):
    # 분리된 처리 로직
    pass
```

### 2. 의존성 분석

**의존성 그래프:**
```
core/models -> utils/device_utils
core/training -> core/models, utils/metrics
core/inference -> core/models, utils/path_utils
utils/* -> (순환 의존성 없음)
```

**의존성 주입 패턴:**
```python
# 의존성 주입으로 테스트 용이성 향상
class ModelTrainer:
    def __init__(self, 
                 model_factory: ModelFactory,
                 metric_calculator: RougeCalculator,
                 logger: StructuredLogger):
        self.model_factory = model_factory
        self.metric_calculator = metric_calculator
        self.logger = logger
```

### 3. 테스트 커버리지 분석

**테스트 가능한 설계:**
```python
# 순수 함수로 설계 (테스트 용이)
def normalize_text(text: str) -> str:
    """부작용 없는 순수 함수"""
    return re.sub(r'\s+', ' ', text.strip())

# 의존성 주입으로 모킹 가능
class DataProcessor:
    def __init__(self, path_manager: PathManager = None):
        self.path_manager = path_manager or PathManager()
```

---

## 🚀 성능 개선 권장사항

### 1. 메모리 최적화
- **배치 크기 동적 조정**: OOM 방지
- **그래디언트 체크포인팅**: 메모리 사용량 50% 절약
- **모델 병렬화**: 큰 모델 분산 처리

### 2. 속도 최적화
- **JIT 컴파일**: `torch.jit.script()` 사용
- **ONNX 변환**: 추론 속도 2-3배 향상
- **텐서 연산 최적화**: 불필요한 CPU-GPU 전송 제거

### 3. 코드 품질 개선
- **타입 힌트 완성**: mypy 정적 분석
- **단위 테스트 확대**: 90% 이상 커버리지 목표
- **코드 리뷰 자동화**: pre-commit 훅 활용

---

## 🔗 관련 문서

- [핵심 모듈 API](./core_modules.md) - API 참조
- [유틸리티 함수](./utilities.md) - 헬퍼 함수들
- [프로젝트 구조](../architecture/project_structure.md) - 전체 구조
- [성능 최적화](../../02_user_guides/evaluation/performance_analysis.md) - 성능 가이드

---

이 분석을 바탕으로 코드의 동작 원리를 이해하고 효과적인 확장 및 개선을 진행하세요.
