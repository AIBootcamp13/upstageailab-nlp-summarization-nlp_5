# 🔍 디버깅 가이드

프로젝트 진행 중 발생하는 문제들을 체계적으로 진단하고 해결하는 방법을 안내합니다.

## 🛠️ 디버깅 프로세스

### 1. 문제 파악 단계

#### 에러 메시지 분석
```bash
# 에러 로그 확인
tail -f logs/training.log

# Python 스택 트레이스 분석
python -c "import traceback; traceback.print_exc()"
```

#### 환경 정보 수집
```python
import torch
import platform

print(f"Platform: {platform.system()} {platform.machine()}")
print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

### 2. 일반적인 디버깅 기법

#### 로깅 활용
```python
import logging

# 상세 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 중요 지점에 로그 추가
logger.debug(f"Input shape: {input_tensor.shape}")
logger.info(f"Processing batch {batch_idx}")
```

#### 단계별 검증
```python
# 데이터 로딩 검증
def debug_dataloader(dataloader):
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}: {batch.keys()}")
        if i >= 2:  # 처음 몇 개만 확인
            break

# 모델 출력 검증
def debug_model_output(model, sample_input):
    model.eval()
    with torch.no_grad():
        output = model(sample_input)
        print(f"Output shape: {output.shape}")
        print(f"Output range: {output.min():.4f} ~ {output.max():.4f}")
```

## 🐛 카테고리별 디버깅

### 데이터 관련 문제

#### 데이터 로딩 오류
```python
# 데이터셋 검증
def validate_dataset(dataset):
    print(f"Dataset size: {len(dataset)}")
    
    # 첫 번째 샘플 확인
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    for key, value in sample.items():
        print(f"{key}: {type(value)} - {value}")
```

#### 토크나이징 문제
```python
# 토크나이저 검증
def debug_tokenizer(tokenizer, text):
    tokens = tokenizer.tokenize(text)
    input_ids = tokenizer.encode(text, return_tensors="pt")
    
    print(f"Original text: {text}")
    print(f"Tokens: {tokens}")
    print(f"Input IDs: {input_ids}")
    print(f"Decoded: {tokenizer.decode(input_ids[0])}")
```

### 모델 학습 문제

#### 학습 진행 상황 모니터링
```python
# 학습 과정 디버깅
class DebugCallback:
    def on_epoch_start(self, epoch):
        print(f"Epoch {epoch} started")
    
    def on_batch_end(self, batch_idx, loss):
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}, Loss: {loss:.4f}")
    
    def on_epoch_end(self, epoch, metrics):
        print(f"Epoch {epoch} ended, Metrics: {metrics}")
```

#### 그래디언트 문제 진단
```python
# 그래디언트 확인
def debug_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm()
            print(f"{name}: grad_norm = {grad_norm:.6f}")
        else:
            print(f"{name}: No gradient")
```

### 메모리 관련 문제

#### 메모리 사용량 모니터링
```python
import psutil
import torch

def monitor_memory():
    # 시스템 메모리
    memory = psutil.virtual_memory()
    print(f"System Memory: {memory.percent}% used")
    
    # GPU 메모리
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
```

#### 메모리 누수 탐지
```python
import gc

def debug_memory_leak():
    # 가비지 컬렉션 전후 객체 수 비교
    before = len(gc.get_objects())
    
    # 의심되는 코드 실행
    # your_function()
    
    gc.collect()
    after = len(gc.get_objects())
    
    print(f"Objects before: {before}, after: {after}, diff: {after-before}")
```

## 🔧 도구별 디버깅

### PyTorch 디버깅

#### 모델 구조 확인
```python
from torchsummary import summary

# 모델 요약 출력
summary(model, input_size=(max_length,))

# 모델 그래프 시각화
import torch.utils.tensorboard as tb
with tb.SummaryWriter() as writer:
    writer.add_graph(model, sample_input)
```

#### autograd 디버깅
```python
# 자동 미분 검증
torch.autograd.set_detect_anomaly(True)

# 계산 그래프 확인
x = torch.randn(2, 2, requires_grad=True)
y = x * 2
y.retain_grad()  # 중간 변수의 gradient 보존
```

### Transformers 라이브러리 디버깅

#### 모델 출력 분석
```python
# 모델 출력 상세 분석
outputs = model(**inputs, output_attentions=True, output_hidden_states=True)

print(f"Last hidden state shape: {outputs.last_hidden_state.shape}")
print(f"Number of attention layers: {len(outputs.attentions)}")
print(f"Number of hidden states: {len(outputs.hidden_states)}")
```

#### 토크나이저 특수 토큰 확인
```python
# 특수 토큰 검증
print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
print(f"UNK token: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
print(f"BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
```

## 🚨 성능 디버깅

### 병목 지점 찾기

#### 프로파일링
```python
import cProfile
import pstats

# 코드 프로파일링
profiler = cProfile.Profile()
profiler.enable()

# 프로파일링할 코드
your_function()

profiler.disable()

# 결과 분석
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # 상위 10개 함수
```

#### 시간 측정
```python
import time
from contextlib import contextmanager

@contextmanager
def timer(description):
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"{description}: {elapsed:.4f} seconds")

# 사용 예시
with timer("Data loading"):
    data = load_data()

with timer("Model inference"):
    output = model(data)
```

### 배치 크기 최적화

#### 자동 배치 크기 찾기
```python
def find_optimal_batch_size(model, sample_input, max_batch_size=128):
    batch_size = 1
    
    while batch_size <= max_batch_size:
        try:
            batch_input = sample_input.repeat(batch_size, 1)
            
            with torch.no_grad():
                output = model(batch_input)
            
            print(f"✅ Batch size {batch_size}: Success")
            batch_size *= 2
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ Batch size {batch_size}: OOM")
                optimal_size = batch_size // 2
                print(f"Optimal batch size: {optimal_size}")
                return optimal_size
            else:
                raise e
    
    return max_batch_size
```

## 🔍 고급 디버깅 기법

### 텐서 값 추적
```python
# 텐서 값 변화 추적
class TensorTracker:
    def __init__(self):
        self.values = []
    
    def track(self, tensor, name=""):
        self.values.append({
            'name': name,
            'shape': tensor.shape,
            'mean': tensor.mean().item(),
            'std': tensor.std().item(),
            'min': tensor.min().item(),
            'max': tensor.max().item()
        })
    
    def summary(self):
        for i, v in enumerate(self.values):
            print(f"Step {i} - {v['name']}: "
                  f"shape={v['shape']}, mean={v['mean']:.4f}, "
                  f"std={v['std']:.4f}, range=[{v['min']:.4f}, {v['max']:.4f}]")

# 사용 예시
tracker = TensorTracker()
x = torch.randn(2, 3)
tracker.track(x, "input")
y = torch.relu(x)
tracker.track(y, "after_relu")
tracker.summary()
```

### 모델 웨이트 분석
```python
def analyze_model_weights(model):
    for name, param in model.named_parameters():
        weight_norm = param.norm().item()
        weight_mean = param.mean().item()
        weight_std = param.std().item()
        
        print(f"{name}:")
        print(f"  Norm: {weight_norm:.6f}")
        print(f"  Mean: {weight_mean:.6f}")
        print(f"  Std:  {weight_std:.6f}")
        
        # 이상 값 감지
        if weight_norm > 100:
            print(f"  ⚠️ Large weight norm detected!")
        if abs(weight_mean) > 10:
            print(f"  ⚠️ Large weight mean detected!")
```

## 📋 디버깅 체크리스트

### 환경 설정 확인
- [ ] Python 버전 호환성
- [ ] 패키지 버전 일치
- [ ] CUDA 설정 정상
- [ ] 메모리 충분

### 데이터 확인
- [ ] 데이터 형식 올바름
- [ ] 토크나이징 정상
- [ ] 배치 크기 적절
- [ ] 레이블 정확성

### 모델 확인
- [ ] 아키텍처 설정 올바름
- [ ] 가중치 초기화 정상
- [ ] 그래디언트 흐름 정상
- [ ] 출력 차원 일치

### 학습 과정 확인
- [ ] 손실 함수 적절
- [ ] 학습률 적절
- [ ] 수렴 패턴 정상
- [ ] 오버피팅 확인

## 🆘 긴급 디버깅

### 빠른 문제 해결
1. **재시작**: 커널/프로세스 재시작
2. **메모리 정리**: `torch.cuda.empty_cache()`
3. **간단한 테스트**: 최소 예제로 재현
4. **버전 롤백**: 이전 정상 버전으로 복원

### 도움 요청 전 준비사항
- 에러 메시지 전문
- 재현 가능한 최소 코드
- 환경 정보 (OS, Python, 패키지 버전)
- 시도한 해결 방법들

## 🔗 관련 도구

- **pdb**: Python 내장 디버거
- **ipdb**: IPython 디버거  
- **PyTorch profiler**: 성능 분석
- **TensorBoard**: 시각화
- **WandB**: 실험 추적

---

체계적인 디버깅을 통해 문제를 빠르게 해결하고 안정적인 개발 환경을 구축하세요.
