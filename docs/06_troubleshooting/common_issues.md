# 문제 해결 가이드

## 🎯 개요

이 문서는 NLP 대화 요약 프로젝트에서 발생할 수 있는 **일반적인 문제들과 해결 방법**을 정리합니다. 문제 상황별로 체계적인 해결 방법을 제공합니다.

## 🎆 최신 기술 스택 관련 문제 (2024.12 업데이트)

### torch 2.6.0 호환성 문제

#### transformers 4.54.0 버전 충돌
**증상**:
```bash
VersionConflict: transformers 4.54.0 requires torch>=2.0.0
AttributeError: module 'torch' has no attribute 'compile'
```

**해결 방법**:
```bash
# 1. 체계적 업그레이드 (권장)
uv pip uninstall torch torchvision torchaudio transformers
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
uv pip install transformers==4.54.0

# 2. 호환성 확인
python -c "
import torch
import transformers
print(f'torch: {torch.__version__}')
print(f'transformers: {transformers.__version__}')
print(f'torch.compile 지원: {hasattr(torch, "compile")}')
"
```

#### pytorch_lightning 2.5.2 업그레이드 문제
**증상**:
```bash
ImportError: cannot import name 'LightningModule' from 'pytorch_lightning'
AttributeError: 'Trainer' object has no attribute 'fit_loop'
```

**해결 방법**:
```bash
# 기존 pytorch_lightning 완전 제거
uv pip uninstall pytorch_lightning lightning

# 최신 버전 설치
uv pip install pytorch_lightning==2.5.2

# 또는 대안적으로 lightning 설치
uv pip install lightning==2.5.2
```

### unsloth/QLoRA 설정 문제

#### macOS에서 unsloth 설치 실패
**증상**:
```bash
ERROR: Failed building wheel for sentencepiece
ERROR: Could not build wheels for xformers
```

**해결 방법**:
```yaml
# config.yaml 수정 - QLoRA 모드 사용
qlora:
  use_unsloth: false  # macOS에서는 비활성화
  use_qlora: true     # QLoRA로 대체
  lora_rank: 16
  load_in_4bit: true
```

```python
# 수동 확인
try:
    import unsloth
    print('✅ unsloth 사용 가능')
except ImportError:
    print('⚠️  unsloth 없음, QLoRA 모드 사용')
    
try:
    import peft, bitsandbytes
    print('✅ QLoRA 지원 (peft + bitsandbytes)')
except ImportError:
    print('❌ QLoRA 지원 없음')
```

#### bitsandbytes CUDA 버전 문제
**증상**:
```bash
RuntimeError: CUDA version mismatch: bitsandbytes was compiled with CUDA 11.8
```

**해결 방법**:
```bash
# CUDA 버전 확인
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# 호환 버전 설치
# CUDA 11.8 사용 시
uv pip install bitsandbytes==0.41.1

# CUDA 12.x 사용 시
uv pip install bitsandbytes==0.43.0

# CPU 모드로 대체
uv pip uninstall bitsandbytes
# config.yaml에서 load_in_4bit: false로 설정
```

### gradient checkpointing 문제

#### use_reentrant 경고
**증상**:
```bash
UserWarning: use_reentrant parameter should be passed explicitly
```

**해결 방법**:
```yaml
# config.yaml 수정
training:
  gradient_checkpointing: true
  gradient_checkpointing_kwargs:
    use_reentrant: false  # 최신 버전에서 권장
```

---

## 🚨 긴급 문제 해결

### 시스템이 전혀 작동하지 않는 경우

#### 1. 환경 설정 문제

**증상**: 모듈 임포트 오류, 패키지 없음 오류
```bash
ModuleNotFoundError: No module named 'utils'
ImportError: No module named 'transformers'
```

**진단 단계**:
```bash
# 1. Python 환경 확인
python --version
# 3.8+ 버전이어야 함

# 2. 가상환경 활성화 확인
which python
# 가상환경 경로여야 함

# 3. 패키지 설치 확인
pip list | grep transformers
pip list | grep torch
```

**해결 방법**:
```bash
# Option 1: UV 환경 재설정 (권장)
./scripts/setup_aistages.sh

# Option 2: 수동 재설치
pip install -r code/requirements.txt

# Option 3: 새 환경 생성
python -m venv nlp_env
source nlp_env/bin/activate  # Linux/Mac
# nlp_env\Scripts\activate  # Windows
pip install -r code/requirements.txt
```

#### 2. 경로 문제

**증상**: 파일을 찾을 수 없음, 절대 경로 오류
```bash
FileNotFoundError: [Errno 2] No such file or directory: '/Users/jayden/...'
```

**진단 단계**:
```python
# 현재 작업 디렉토리 확인
import os
print(f"Current directory: {os.getcwd()}")

# 프로젝트 루트 확인
from pathlib import Path
project_files = ['code', 'config', 'docs']
for file in project_files:
    print(f"{file} exists: {Path(file).exists()}")
```

**해결 방법**:
```bash
# 1. 올바른 디렉토리로 이동
cd /path/to/nlp-sum-lyj

# 2. 프로젝트 구조 확인
ls -la
# code/, config/, docs/ 디렉토리가 있어야 함

# 3. Python 경로 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)/code"
```

---

## 🔧 환경 설정 문제

### CUDA/GPU 관련 문제

#### GPU를 인식하지 못하는 경우

**증상**:
```python
torch.cuda.is_available()  # False 반환
```

**진단 단계**:
```bash
# 1. NVIDIA 드라이버 확인
nvidia-smi

# 2. CUDA 버전 확인
nvcc --version

# 3. PyTorch CUDA 지원 확인
python -c "import torch; print(torch.version.cuda)"
```

**해결 방법**:
```bash
# CUDA 호환 PyTorch 재설치
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 또는 CPU 버전 사용
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 메모리 부족 오류

**증상**:
```bash
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**진단 단계**:
```python
import torch
print(f"GPU 메모리 현재 사용량: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print(f"GPU 메모리 최대 사용량: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
print(f"GPU 전체 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

**해결 방법**:
```yaml
# config/base_config.yaml 수정
training:
  per_device_train_batch_size: 2  # 기본값 8에서 2로 감소
  gradient_accumulation_steps: 4  # 배치 크기 감소 보상
  fp16: true  # 혼합 정밀도 학습 활성화

tokenizer:
  encoder_max_len: 512  # 기본값 1024에서 512로 감소
  decoder_max_len: 128  # 기본값 256에서 128로 감소
```

### 패키지 의존성 충돌

#### 버전 충돌 문제

**증상**:
```bash
ERROR: pip's dependency resolver does not currently take into account all the packages
```

**해결 방법**:
```bash
# 1. 모든 패키지 제거 후 재설치
pip freeze > installed_packages.txt
pip uninstall -r installed_packages.txt -y
pip install -r code/requirements.txt

# 2. UV 사용 (권장)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
uv pip install -r code/requirements.txt
```

---

## 📊 데이터 처리 문제

### 데이터 로딩 실패

#### 인코딩 문제

**증상**:
```bash
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc0 in position 0
```

**해결 방법**:
```python
# 자동 인코딩 감지 및 처리
import pandas as pd
import chardet

def load_csv_with_encoding_detection(file_path):
    # 인코딩 감지
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']
    
    print(f"감지된 인코딩: {encoding}")
    
    # 여러 인코딩으로 시도
    encodings = [encoding, 'utf-8', 'cp949', 'euc-kr', 'latin-1']
    
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            print(f"성공한 인코딩: {enc}")
            return df
        except UnicodeDecodeError:
            continue
    
    raise ValueError("지원되는 인코딩을 찾을 수 없습니다")
```

#### CSV 파싱 오류

**증상**:
```bash
pandas.errors.ParserError: Error tokenizing data
```

**해결 방법**:
```python
# 안전한 CSV 읽기
import pandas as pd

def safe_read_csv(file_path):
    try:
        # 기본 시도
        df = pd.read_csv(file_path)
        return df
    except pd.errors.ParserError:
        try:
            # 쿼팅 문제 해결
            df = pd.read_csv(file_path, quoting=3)  # QUOTE_NONE
            return df
        except:
            try:
                # 구분자 문제 해결
                df = pd.read_csv(file_path, sep=None, engine='python')
                return df
            except:
                # 수동 파싱
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                print(f"첫 5줄 확인: {lines[:5]}")
                raise ValueError("CSV 파싱 실패 - 수동 확인 필요")
```

### Multi-Reference 데이터 문제

#### 정답 요약문 형식 오류

**증상**:
```bash
ValueError: summaries column not found or invalid format
```

**진단 단계**:
```python
import pandas as pd

# 데이터 구조 확인
df = pd.read_csv("data/train.csv")
print(f"컬럼명: {df.columns.tolist()}")
print(f"첫 번째 행: {df.iloc[0].to_dict()}")

# summary 관련 컬럼 찾기
summary_cols = [col for col in df.columns if 'summary' in col.lower()]
print(f"Summary 컬럼들: {summary_cols}")
```

**해결 방법**:
```python
# 유연한 multi-reference 처리
def parse_multi_reference_summaries(df):
    """다양한 형식의 multi-reference 데이터 파싱"""
    
    # 방법 1: summary1, summary2, summary3 컬럼
    if all(col in df.columns for col in ['summary1', 'summary2', 'summary3']):
        df['summaries'] = df[['summary1', 'summary2', 'summary3']].values.tolist()
        return df
    
    # 방법 2: summary 컬럼에 구분자로 분리
    elif 'summary' in df.columns:
        def split_summaries(summary_text):
            if pd.isna(summary_text):
                return ["", "", ""]
            
            # 다양한 구분자 시도
            for delimiter in ['|||', ';;', '|', '\n---\n']:
                if delimiter in summary_text:
                    summaries = [s.strip() for s in summary_text.split(delimiter)]
                    # 3개로 맞추기
                    while len(summaries) < 3:
                        summaries.append("")
                    return summaries[:3]
            
            # 구분자가 없으면 단일 요약문 복사
            return [summary_text, summary_text, summary_text]
        
        df['summaries'] = df['summary'].apply(split_summaries)
        return df
    
    else:
        raise ValueError("지원되는 summary 형식을 찾을 수 없습니다")
```

---

## 🤖 모델 학습 문제

### 메모리 관련 문제

#### 훈련 중 메모리 부족

**증상**:
```bash
RuntimeError: CUDA out of memory
torch.cuda.OutOfMemoryError
```

**즉시 해결 방법**:
```python
# 긴급 메모리 정리
import torch
import gc

def emergency_memory_cleanup():
    """긴급 메모리 정리"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    print("메모리 정리 완료")

emergency_memory_cleanup()
```

**설정 최적화**:
```yaml
# config/base_config.yaml - 메모리 절약 설정
training:
  per_device_train_batch_size: 1  # 최소 배치 크기
  gradient_accumulation_steps: 16  # 효과적인 배치 크기 = 1 * 16 = 16
  fp16: true  # 반정밀도 사용
  dataloader_num_workers: 2  # 워커 수 감소
  save_strategy: "epoch"  # 체크포인트 빈도 감소
  logging_steps: 100  # 로깅 빈도 감소

tokenizer:
  encoder_max_len: 256  # 최대 길이 크게 감소
  decoder_max_len: 64   # 출력 길이 감소

model:
  gradient_checkpointing: true  # 그래디언트 체크포인팅 활성화
```

#### 메모리 누수 문제

**진단 방법**:
```python
# 메모리 사용량 모니터링
import psutil
import torch
import time

def monitor_memory(duration=60):
    """메모리 사용량 모니터링"""
    start_time = time.time()
    
    while time.time() - start_time < duration:
        # CPU 메모리
        cpu_memory = psutil.virtual_memory().used / 1024**3
        
        # GPU 메모리
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated(0) / 1024**3
            gpu_cached = torch.cuda.memory_reserved(0) / 1024**3
            print(f"CPU: {cpu_memory:.2f}GB, GPU: {gpu_memory:.2f}GB (Cached: {gpu_cached:.2f}GB)")
        else:
            print(f"CPU: {cpu_memory:.2f}GB")
        
        time.sleep(5)

# 사용법
monitor_memory(60)  # 1분간 모니터링
```

### 학습 수렴 문제

#### 손실이 감소하지 않는 경우

**진단 단계**:
```python
# 학습률 진단
def diagnose_learning_rate(config):
    lr = config['training']['learning_rate']
    
    if lr > 1e-3:
        print(f"⚠️ 학습률이 너무 높을 수 있습니다: {lr}")
        print("권장: 1e-4 ~ 5e-5")
    elif lr < 1e-6:
        print(f"⚠️ 학습률이 너무 낮을 수 있습니다: {lr}")
        print("권장: 1e-4 ~ 5e-5")
    else:
        print(f"✅ 학습률이 적절합니다: {lr}")

# 데이터 진단
def diagnose_data(dataloader):
    batch = next(iter(dataloader))
    
    print(f"배치 크기: {len(batch['input_ids'])}")
    print(f"입력 길이 평균: {batch['input_ids'].shape[1]}")
    print(f"출력 길이 평균: {batch['labels'].shape[1]}")
    
    # 토큰 ID 분포 확인
    print(f"입력 토큰 범위: {batch['input_ids'].min().item()} ~ {batch['input_ids'].max().item()}")
    print(f"라벨 토큰 범위: {batch['labels'].min().item()} ~ {batch['labels'].max().item()}")
```

**해결 방법**:
```yaml
# 학습률 스케줄링 추가
training:
  learning_rate: 5e-5  # 안전한 초기 학습률
  warmup_ratio: 0.1    # 워밍업 추가
  lr_scheduler_type: "cosine"  # 코사인 스케줄러
  weight_decay: 0.01   # 가중치 감쇠 추가
```

#### 과적합 문제

**증상**: 훈련 손실은 감소하지만 검증 손실이 증가

**해결 방법**:
```yaml
# 정규화 강화
training:
  learning_rate: 3e-5  # 학습률 감소
  weight_decay: 0.05   # 가중치 감쇠 증가
  warmup_ratio: 0.1
  num_train_epochs: 3  # 에폭 수 감소
  
  # Early stopping (WandB sweep에서 설정)
  metric_for_best_model: "eval_rouge_combined_f1"
  greater_is_better: true
  load_best_model_at_end: true
  save_total_limit: 2
```

---

## 🧮 ROUGE 계산 문제

### ROUGE 점수가 0 또는 매우 낮은 경우

#### 토큰화 문제

**진단 단계**:
```python
# 토큰화 결과 확인
from utils.metrics import RougeCalculator

calculator = RougeCalculator(use_korean_tokenizer=True)

# 테스트 데이터
prediction = "안녕하세요 좋은 하루입니다"
reference = "안녕하세요 좋은 날이에요"

# 토큰화 확인 (내부 메서드가 있다면)
print(f"예측 토큰: {calculator._tokenize(prediction)}")
print(f"정답 토큰: {calculator._tokenize(reference)}")

# 기본 ROUGE 계산
scores = calculator.compute_korean_rouge([prediction], [reference])
print(f"ROUGE 점수: {scores}")
```

**해결 방법**:
```python
# 토크나이저 설정 조정
calculator = RougeCalculator(
    use_korean_tokenizer=True,
    korean_tokenizer="okt"  # mecab 대신 okt 시도
)

# 또는 영어 토크나이저 사용
calculator_en = RougeCalculator(use_korean_tokenizer=False)
```

#### Multi-Reference 계산 오류

**진단 단계**:
```python
# Multi-reference 데이터 구조 확인
predictions = ["테스트 요약문"]
references_list = [["정답1", "정답2", "정답3"]]

print(f"예측 개수: {len(predictions)}")
print(f"정답 그룹 개수: {len(references_list)}")
print(f"첫 번째 그룹 정답 개수: {len(references_list[0])}")

# 각 정답과 개별 비교
for i, ref in enumerate(references_list[0]):
    single_score = calculator.compute_korean_rouge(predictions, [ref])
    print(f"정답 {i+1} 대비 점수: {single_score['rouge1']['f1']:.4f}")
```

---

## 🚀 추론 및 제출 문제

### 추론 속도가 너무 느린 경우

**진단 단계**:
```python
import time
from core.inference import InferenceEngine

# 추론 속도 측정
engine = InferenceEngine("path/to/model")
test_dialogues = ["테스트 대화"] * 10

start_time = time.time()
predictions = engine.predict_batch(test_dialogues, batch_size=1)
single_time = time.time() - start_time

start_time = time.time()
predictions = engine.predict_batch(test_dialogues, batch_size=5)
batch_time = time.time() - start_time

print(f"단일 처리: {single_time:.2f}초 ({len(test_dialogues)/single_time:.2f} samples/sec)")
print(f"배치 처리: {batch_time:.2f}초 ({len(test_dialogues)/batch_time:.2f} samples/sec)")
```

**최적화 방법**:
```python
# 1. 배치 크기 최적화
optimal_batch_size = 8  # GPU 메모리에 따라 조정

# 2. 생성 파라미터 최적화
generation_config = {
    "num_beams": 3,  # 빔 수 감소 (기본 5 -> 3)
    "max_length": 128,  # 최대 길이 제한
    "early_stopping": True,
    "no_repeat_ngram_size": 2
}

# 3. 모델 최적화 (고급)
import torch
model = torch.jit.script(model)  # JIT 컴파일
```

### 제출 파일 형식 오류

#### 컬럼명 또는 순서 문제

**진단 단계**:
```python
import pandas as pd

# 제출 파일 확인
submission_df = pd.read_csv("submission.csv")
print(f"컬럼명: {submission_df.columns.tolist()}")
print(f"데이터 형태: {submission_df.shape}")
print(f"첫 5행:\n{submission_df.head()}")

# 대회 요구사항 확인
required_columns = ["fname", "summary"]
missing_columns = [col for col in required_columns if col not in submission_df.columns]
if missing_columns:
    print(f"❌ 누락된 컬럼: {missing_columns}")
else:
    print("✅ 컬럼 형식 정확")
```

**해결 방법**:
```python
# 정확한 제출 형식 생성
from utils.data_utils import DataProcessor

processor = DataProcessor()

# 예측 결과와 파일명
predictions = ["요약1", "요약2", "요약3"]
fnames = ["file1.txt", "file2.txt", "file3.txt"]

# 올바른 형식으로 내보내기
submission_df = processor.export_submission_format(
    predictions=predictions,
    fnames=fnames,
    output_path="submission.csv"
)

# 형식 검증
is_valid = processor.validate_submission_format("submission.csv")
print(f"제출 형식 유효성: {is_valid}")
```

---

## 🔍 성능 최적화 문제

### 학습 속도가 너무 느린 경우

#### 데이터 로딩 병목

**진단 방법**:
```python
import time
from torch.utils.data import DataLoader

# 데이터 로딩 속도 측정
def measure_dataloader_speed(dataloader, num_batches=10):
    start_time = time.time()
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        # 실제 처리 없이 로딩만 측정
        pass
    
    end_time = time.time()
    avg_time_per_batch = (end_time - start_time) / num_batches
    
    print(f"배치당 로딩 시간: {avg_time_per_batch:.3f}초")
    return avg_time_per_batch

# 최적화된 DataLoader 설정
optimized_dataloader = DataLoader(
    dataset,
    batch_size=8,
    num_workers=4,  # CPU 코어 수에 따라 조정
    pin_memory=True,  # GPU 사용 시
    persistent_workers=True,  # 워커 재사용
    prefetch_factor=2  # 미리 가져올 배치 수
)
```

#### GPU 활용도 문제

**진단 방법**:
```bash
# GPU 사용률 모니터링
nvidia-smi -l 1  # 1초마다 업데이트

# 또는 Python에서
import pynvml
pynvml.nvmlInit()

def monitor_gpu_utilization():
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    
    for _ in range(60):  # 1분간 모니터링
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        print(f"GPU 사용률: {util.gpu}%, 메모리: {memory.used/memory.total*100:.1f}%")
        time.sleep(1)
```

**최적화 방법**:
```yaml
# 효율적인 학습 설정
training:
  per_device_train_batch_size: 8  # GPU에 맞게 조정
  gradient_accumulation_steps: 2
  fp16: true  # 혼합 정밀도
  dataloader_num_workers: 4
  dataloader_pin_memory: true
  remove_unused_columns: false
  
generation:
  num_beams: 5
  length_penalty: 1.0
  early_stopping: true
```

---

## 🐛 일반적인 버그 해결

### WandB 연동 문제

#### 로그인 실패

**해결 방법**:
```bash
# WandB 재로그인
wandb logout
wandb login

# API 키 직접 설정
export WANDB_API_KEY="your_api_key_here"

# 또는 Python에서
import wandb
wandb.login(key="your_api_key_here")
```

#### 실험 로깅 실패

**진단 및 해결**:
```python
import wandb

# WandB 상태 확인
print(f"WandB 로그인 상태: {wandb.api.api_key is not None}")
print(f"현재 프로젝트: {wandb.run.project if wandb.run else 'None'}")

# 안전한 로깅 함수
def safe_wandb_log(metrics, step=None):
    try:
        if wandb.run is not None:
            wandb.log(metrics, step=step)
        else:
            print(f"WandB 미연결 - 메트릭: {metrics}")
    except Exception as e:
        print(f"WandB 로깅 실패: {e}")
        print(f"메트릭: {metrics}")
```

### 설정 파일 문제

#### YAML 파싱 오류

**증상**:
```bash
yaml.scanner.ScannerError: while parsing a block mapping
```

**해결 방법**:
```python
import yaml

# YAML 파일 검증
def validate_yaml_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        print(f"✅ YAML 파일 유효: {file_path}")
        return data
    except yaml.YAMLError as e:
        print(f"❌ YAML 파싱 오류: {e}")
        print(f"파일: {file_path}")
        
        # 라인별 확인
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines, 1):
                if ':' in line and not line.strip().startswith('#'):
                    if line.count('"') % 2 != 0:
                        print(f"라인 {i}에 따옴표 불일치: {line.strip()}")
        return None

# 사용법
config = validate_yaml_file("config/base_config.yaml")
```

---

## 📞 고급 문제 해결

### 분산 학습 문제

#### 멀티 GPU 설정 오류

**해결 방법**:
```python
# GPU 개수 확인
import torch
num_gpus = torch.cuda.device_count()
print(f"사용 가능한 GPU 개수: {num_gpus}")

# 분산 학습 설정
if num_gpus > 1:
    # config에서 분산 학습 활성화
    training_args = {
        "local_rank": -1,  # 자동 설정
        "ddp_find_unused_parameters": False,
        "dataloader_num_workers": 2,  # GPU당 워커 수
    }
```

### 메모리 프로파일링

#### 상세 메모리 분석

**도구 설치 및 사용**:
```bash
# 메모리 프로파일러 설치
pip install memory-profiler
pip install psutil

# 프로파일링 실행
python -m memory_profiler code/trainer.py
```

```python
# 코드 내 프로파일링
from memory_profiler import profile

@profile
def memory_intensive_function():
    # 메모리 사용량을 확인하고 싶은 함수
    pass

# 또는 라인별 메모리 사용량
from memory_profiler import LineProfiler

profiler = LineProfiler()
profiler.add_function(your_function)
profiler.enable_by_count()
# ... 함수 실행 ...
profiler.print_stats()
```

---

## 🚨 응급 복구 절차

### 완전 초기화

모든 문제가 해결되지 않을 때:

```bash
# 1. 백업 생성
cp -r outputs/ outputs_backup_$(date +%Y%m%d_%H%M%S)/
cp -r logs/ logs_backup_$(date +%Y%m%d_%H%M%S)/

# 2. 환경 완전 재구성
rm -rf .venv/  # 가상환경 삭제
rm -rf __pycache__/  # 캐시 삭제
find . -name "*.pyc" -delete  # 컴파일된 파일 삭제

# 3. 새로운 환경 구성
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r code/requirements.txt

# 4. 설정 초기화
cp config/base_config.yaml config/base_config_backup.yaml
# 기본 설정으로 복원 (Git에서)
git checkout config/base_config.yaml

# 5. 테스트 실행
python -c "from utils.path_utils import PathManager; print('✅ 시스템 복구 완료')"
```

### 데이터 복구

```bash
# 원본 데이터 무결성 확인
python -c "
import pandas as pd
try:
    df = pd.read_csv('data/train.csv')
    print(f'✅ 훈련 데이터 정상: {len(df)} 샘플')
except Exception as e:
    print(f'❌ 훈련 데이터 오류: {e}')

try:
    df = pd.read_csv('data/test.csv')
    print(f'✅ 테스트 데이터 정상: {len(df)} 샘플')
except Exception as e:
    print(f'❌ 테스트 데이터 오류: {e}')
"
```

---

## 📞 지원 요청 가이드

### 효과적인 문제 보고

문제 해결을 위해 다음 정보를 포함하여 보고하세요:

#### 1. 환경 정보
```bash
# 환경 정보 수집 스크립트
python -c "
import sys
import torch
import transformers
import pandas as pd
import platform

print('=== 환경 정보 ===')
print(f'OS: {platform.system()} {platform.release()}')
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'Pandas: {pd.__version__}')
print(f'CUDA 사용 가능: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 버전: {torch.version.cuda}')
    print(f'GPU 개수: {torch.cuda.device_count()}')
print('================')
"
```

#### 2. 오류 재현 단계
1. 정확한 명령어 또는 코드
2. 입력 데이터 샘플
3. 예상 결과 vs 실제 결과
4. 오류 메시지 전문

#### 3. 시도한 해결 방법
- 이미 시도한 해결책들
- 참고한 문서나 자료
- 임시 해결책 여부

### 에스컬레이션 절차

1. **Level 1**: 문서 자체 해결 (30분)
2. **Level 2**: 팀 내 기술 검토 (1시간)
3. **Level 3**: 외부 전문가 상담 (필요시)

---

이 문제 해결 가이드를 통해 대부분의 일반적인 문제들을 체계적으로 해결할 수 있습니다. 새로운 문제가 발견되면 이 문서에 지속적으로 추가됩니다.
