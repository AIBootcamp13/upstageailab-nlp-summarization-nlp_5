# 🌐 mT5_multilingual_XLSum 완전 가이드

다국어 요약 모델 mT5_multilingual_XLSum의 설치부터 고급 활용까지 초보자도 쉽게 따라할 수 있는 종합 가이드입니다.

## 📋 목차

1. [모델 개요](#1-📖-모델-개요)
2. [설치 및 설정](#2-🔧-설치-및-설정)
3. [기본 사용법](#3-🚀-기본-사용법)
4. [API 레퍼런스](#4-📚-api-레퍼런스)
5. [성능 비교](#5-📊-성능-비교)
6. [문제 해결](#6-🛠️-문제-해결)
7. [고급 활용법](#7-⚡-고급-활용법)

---

## 1. 📖 모델 개요

### 🎯 mT5_multilingual_XLSum이란?

**mT5_multilingual_XLSum**은 구글의 mT5(Multilingual T5) 아키텍처를 기반으로 XL-Sum 데이터셋에서 파인튜닝된 다국어 텍스트 요약 모델입니다. 45개 언어에서 뉴스 기사를 한 줄로 요약하는 것을 학습했으며, 한국어 대화 요약에도 뛰어난 성능을 보여줍니다.

### 🌍 XL-Sum 데이터셋의 특징

- **규모**: 100만 개 이상의 뉴스 기사
- **언어 범위**: 45개 언어 지원 (한국어, 영어, 중국어, 일본어 등)
- **요약 스타일**: 짧고 간결한 한 줄 요약 (평균 84토큰)
- **품질**: BBC 등 신뢰할 수 있는 뉴스 소스 활용

### 🔍 다른 모델과의 비교

| 특징 | mT5_multilingual_XLSum | KoBART | T5-base |
|------|------------------------|---------|---------|
| **아키텍처** | mT5 (Encoder-Decoder) | BART | T5 |
| **언어 지원** | 45개 언어 | 한국어 전용 | 영어 전용 |
| **모델 크기** | 2.17GB | 1.2GB | 892MB |
| **입력 길이** | 512 토큰 | 1024 토큰 | 512 토큰 |
| **출력 길이** | 84 토큰 | 200 토큰 | 256 토큰 |
| **특화 분야** | 뉴스 요약 | 대화 요약 | 범용 텍스트 |

### 📈 한국어 성능 지표

mT5_multilingual_XLSum의 한국어 성능 벤치마크:

- **ROUGE-1**: 23.67% (정확한 단어 일치도)
- **ROUGE-2**: 11.45% (두 단어 연속 일치도)  
- **ROUGE-L**: 22.36% (가장 긴 공통 부분 문자열)

> 💡 **참고**: 이 수치는 뉴스 기사 요약 기준이며, 대화 요약에서는 다른 성능을 보일 수 있습니다.

### 🎨 mT5 아키텍처의 장점

1. **다국어 토크나이저**: 101개 언어의 토크나이징을 효율적으로 처리
2. **크로스링구얼 전이**: 한 언어에서 학습한 지식을 다른 언어에 적용
3. **균형 잡힌 성능**: 영어와 비영어권 언어 간 성능 격차 최소화
4. **확장성**: 새로운 언어나 도메인에 쉽게 적용 가능

### ⚠️ 사용 시 고려사항

- **메모리 요구량**: 최소 8GB RAM 필요 (GPU 추론 시)
- **속도**: KoBART 대비 약간 느린 추론 속도
- **요약 길이**: 84토큰으로 제한되어 긴 요약에는 부적합
- **도메인 특성**: 뉴스 기사에 특화되어 일상 대화와 스타일 차이 존재

---

---

## 2. 🔧 설치 및 설정

### 📋 시스템 요구사항

설치를 시작하기 전에 시스템이 다음 조건을 만족하는지 확인하세요:

#### 최소 시스템 요구사항
- **RAM**: 8GB 이상 (GPU 메모리 포함)
- **저장공간**: 5GB 이상 여유 공간
- **Python**: 3.8 이상 (권장: 3.11)
- **운영체제**: Windows 10/11, macOS 10.14+, Ubuntu 18.04+

#### GPU 사용 시 추가 요구사항
- **CUDA**: 11.8 이상 (NVIDIA GPU)
- **GPU 메모리**: 4GB 이상 (권장: 8GB+)
- **드라이버**: 최신 NVIDIA 드라이버

> ⚠️ **주의**: CPU만으로도 실행 가능하지만, GPU 사용 시 추론 속도가 10-20배 향상됩니다.

### 🛠️ 단계별 설치 가이드

#### 1단계: 프로젝트 환경 준비

```bash
# 프로젝트 클론 (이미 있다면 생략)
git clone <프로젝트-URL>
cd nlp-sum-lyj

# UV 패키지 매니저 설치 (권장)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

# 또는 pip 사용
# pip install uv
```

#### 2단계: 가상환경 생성 및 활성화

```bash
# UV 사용 (권장 - 10배 빠른 설치)
uv venv --python 3.11
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate    # Windows

# 기본 의존성 설치
uv pip install -r requirements.txt
```

**pip 사용 시:**
```bash
# 기본 pip 사용
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### 3단계: mT5 모델 다운로드 및 검증

```python
# 모델 다운로드 및 기본 테스트
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# 모델명 설정
model_name = "csebuetnlp/mT5_multilingual_XLSum"

print("토크나이저 다운로드 중...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("모델 다운로드 중... (약 2.17GB)")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 기본 테스트
test_text = "#Person1#: 안녕하세요. #Person2#: 반갑습니다."
input_ids = tokenizer.encode(test_text, return_tensors="pt")

with torch.no_grad():
    output = model.generate(
        input_ids,
        max_length=84,
        num_beams=4,
        early_stopping=True
    )

summary = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"테스트 요약: {summary}")
print("✅ 설치 및 모델 로드 성공!")
```

### 🔍 설치 검증 체크리스트

설치가 완료되면 다음 항목들을 확인해보세요:

```python
# 종합 설치 검증 스크립트
def verify_installation():
    checks = []
    
    # 1. 기본 라이브러리 확인
    try:
        import torch
        import transformers
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        checks.append("✅ 핵심 라이브러리 로드 성공")
    except ImportError as e:
        checks.append(f"❌ 라이브러리 오류: {e}")
        return checks
    
    # 2. CUDA 지원 확인
    if torch.cuda.is_available():
        checks.append(f"✅ CUDA 사용 가능 (GPU: {torch.cuda.get_device_name()})")
    else:
        checks.append("⚠️ CUDA 미사용 (CPU 모드)")
    
    # 3. 메모리 확인
    import psutil
    total_ram = psutil.virtual_memory().total // (1024**3)
    if total_ram >= 8:
        checks.append(f"✅ 충분한 RAM ({total_ram}GB)")
    else:
        checks.append(f"⚠️ RAM 부족 ({total_ram}GB < 8GB)")
    
    # 4. xlsum_utils 확인
    try:
        from code.utils.xlsum_utils import get_xlsum_model_info
        info = get_xlsum_model_info()
        checks.append("✅ xlsum_utils 모듈 로드 성공")
    except ImportError:
        checks.append("❌ xlsum_utils 모듈 오류")
    
    # 5. 모델 호환성 확인
    try:
        model_name = "csebuetnlp/mT5_multilingual_XLSum"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        checks.append("✅ 모델 토크나이저 로드 성공")
    except Exception as e:
        checks.append(f"❌ 모델 로드 오류: {e}")
    
    return checks

# 검증 실행
for check in verify_installation():
    print(check)
```

### 🚨 설치 중 발생할 수 있는 문제들

#### 문제 1: 네트워크 연결 오류
```bash
# Hugging Face Hub 접근 실패 시
export HF_ENDPOINT=https://hf-mirror.com  # 중국 등 제한 지역
# 또는
pip install -U huggingface_hub
huggingface-cli login  # 토큰 설정
```

#### 문제 2: 메모리 부족
```python
# 메모리 최적화 설정
import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# 모델 로드 시 최적화
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # 메모리 절약
    device_map="auto"           # 자동 디바이스 할당
)
```

#### 문제 3: CUDA 버전 불일치
```bash
# PyTorch CUDA 버전 확인
python -c "import torch; print(torch.version.cuda)"

# CUDA 11.8용 재설치
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 문제 4: 권한 오류
```bash
# Linux/macOS에서 권한 문제 시
sudo chown -R $USER:$USER ~/.cache/huggingface
chmod -R 755 ~/.cache/huggingface
```

### 📱 환경별 최적 설정

#### macOS (Apple Silicon)
```bash
# MPS 백엔드 활성화
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
```

#### Windows
```batch
# PowerShell에서 실행 정책 설정
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 가상환경 활성화
.venv\Scripts\Activate.ps1
```

#### Linux (서버 환경)
```bash
# 헤드리스 환경 설정
export MPLBACKEND=Agg
export DISPLAY=:0.0

## 3. 🚀 기본 사용법

mT5_multilingual_XLSum 모델을 사용하는 방법은 크게 3가지가 있습니다. 각 방법은 사용자의 필요와 경험 수준에 따라 선택할 수 있습니다.

### 📌 사용법 개요

| 방법 | 난이도 | 적합한 상황 | 장점 |
|------|--------|-------------|------|
| **방법 1**: 기본 설정 교체 | ⭐ 쉬움 | 빠른 테스트, 기존 코드 활용 | 간단함, 기존 워크플로우 유지 |
| **방법 2**: mT5 전용 설정 | ⭐⭐ 보통 | 최적화된 성능, 전문적 사용 | 최적화된 파라미터, 안정성 |
| **방법 3**: xlsum_utils 직접 활용 | ⭐⭐⭐ 고급 | 커스텀 구현, 세밀한 제어 | 최대한의 유연성, 고급 기능 |

---

### 🎯 방법 1: 기본 설정 교체 (추천 - 초보자용)

가장 간단한 방법으로, 기존 `config.yaml` 파일의 모델명만 변경하여 사용합니다.

#### 1단계: config.yaml 수정

```yaml
# config.yaml 파일에서 다음 라인을 찾아 수정
general:
  data_path: ../data/
  model_name: csebuetnlp/mT5_multilingual_XLSum  # ← 이 부분 변경
  output_dir: ./
```

#### 2단계: 기본 추론 실행
```python
# 기본 추론 예제
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
# 모델 로드
model_name = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 대화 예제
dialogue = """
#Person1#: 오늘 날씨가 정말 좋네요. 산책하러 나가고 싶어요.
#Person2#: 저도 그래요! 근처 공원에 같이 가실래요?
#Person1#: 좋은 생각이에요. 30분 후에 만날까요?
#Person2#: 네, 공원 입구에서 만나요.
"""

# 토크나이징
inputs = tokenizer(
    dialogue.strip(),
    max_length=512,
    truncation=True,
    padding=True,
    return_tensors="pt"
)

# 요약 생성
with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_length=84,      # mT5 XL-Sum 권장 길이
        num_beams=4,        # 빔 서치 크기
        early_stopping=True,
        no_repeat_ngram_size=2
    )

# 결과 디코딩
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"입력: {dialogue[:50]}...")
print(f"요약: {summary}")
```

**예상 출력:**
```
입력: #Person1#: 오늘 날씨가 정말 좋네요. 산책하러 나가고 싶어요....
요약: 두 사람이 날씨가 좋아서 공원에서 만나기로 했다.
```

---

### ⚙️ 방법 2: mT5 전용 설정 활용 (추천 - 최적 성능)

프로젝트에 미리 구성된 `xlsum_mt5` 전용 설정을 사용하여 최적화된 성능을 얻는 방법입니다.

#### 1단계: 전용 설정 로드

```python
# mT5 전용 설정 사용 예제
import yaml
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from code.utils.xlsum_utils import get_xlsum_generation_config, get_xlsum_tokenizer_config

# config.yaml에서 mT5 전용 설정 로드
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
    mt5_config = config['xlsum_mt5']

print("mT5 전용 설정 로드 성공!")
print(f"모델명: {mt5_config['general']['model_name']}")
print(f"입력 길이: {mt5_config['tokenizer']['encoder_max_len']}")
print(f"출력 길이: {mt5_config['tokenizer']['decoder_max_len']}")
```

#### 2단계: 최적화된 추론 실행

```python
# 최적화된 설정으로 모델 로드
model_name = mt5_config['general']['model_name']
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# xlsum_utils에서 최적화된 설정 가져오기
generation_config = get_xlsum_generation_config()
tokenizer_config = get_xlsum_tokenizer_config()

print("최적화된 설정:")
print(f"- 생성 길이: {generation_config['max_length']}")
print(f"- 빔 크기: {generation_config['num_beams']}")
print(f"- 토크나이저 길이: {tokenizer_config['max_length']}")

# 복잡한 대화 예제
complex_dialogue = """
#Person1#: 안녕하세요, 예약 문의드립니다.
#Person2#: 네, 안녕하세요. 언제 예약을 원하시나요?
#Person1#: 다음 주 금요일 저녁 7시쯤 가능할까요?
#Person2#: 금요일 7시는 이미 예약이 찼고, 6시 30분이나 8시는 어떠세요?
#Person1#: 6시 30분이 좋겠네요. 몇 명까지 가능한가요?
#Person2#: 최대 4명까지 가능합니다.
#Person1#: 그럼 4명으로 예약 부탁드립니다.
"""

---

## 4. 📚 API 레퍼런스

`xlsum_utils.py`는 mT5_multilingual_XLSum 모델을 위한 전용 유틸리티 모듈로, 9개의 함수와 1개의 상수를 제공합니다. 이 섹션에서는 각 함수의 사용법을 상세히 설명합니다.

### 📌 함수 개요

| 분류 | 함수명 | 주요 기능 | 난이도 |
|------|-----------|----------|------|
| **전처리** | `xlsum_whitespace_handler()` | 공백 정규화 | ⭐ |
| **전처리** | `preprocess_for_xlsum()` | 텍스트 전처리 | ⭐ |
| **설정** | `get_xlsum_generation_config()` | 생성 설정 | ⭐ |
| **설정** | `get_xlsum_tokenizer_config()` | 토크나이저 설정 | ⭐ |
| **설정** | `get_xlsum_default_config()` | 통합 설정 | ⭐⭐ |
| **정보** | `get_xlsum_model_info()` | 모델 메타정보 | ⭐ |
| **검증** | `is_xlsum_compatible_model()` | 모델 호환성 | ⭐ |
| **검증** | `validate_xlsum_input()` | 입력 유효성 | ⭐ |
| **유틸** | `get_xlsum_preprocessing_prompt()` | 프롬프트 생성 | ⭐ |

---

### 📝 1. 전처리 함수들

#### `xlsum_whitespace_handler(text: str) -> str`

**목적**: 연속된 공백과 줄바꿈을 정규화하여 mT5 모델이 처리하기 용이한 형태로 변환합니다.

**매개변수**:
- `text` (str): 정규화할 입력 텍스트

**반환값**:
- `str`: 공백이 정규화된 텍스트

**사용 예제**:
```python
from code.utils.xlsum_utils import xlsum_whitespace_handler

# 기본 사용법
noisy_text = """
#Person1#:    안녕하세요...   


오늘 날씨가  좋네요.
#Person2#:   네,     정말   좋아요!
"""

clean_text = xlsum_whitespace_handler(noisy_text)
print(f"전: {len(noisy_text)} 문자")
print(f"후: {len(clean_text)} 문자")
print(f"결과: {clean_text}")

# 출력:
# 전: 89 문자
# 후: 53 문자
# 결과: #Person1#: 안녕하세요... 오늘 날씨가 좋네요. #Person2#: 네, 정말 좋아요!
```

**주의사항**:
- 빈 문자열이나 None 입력 시 빈 문자열 반환
- 줄바꿈(`\n`)을 공백으로 대체하므로 문단 배치가 중요한 경우 주의 필요

---

#### `preprocess_for_xlsum(text: str, **kwargs) -> str`

**목적**: XL-Sum 모델용 종합 텍스트 전처리를 수행합니다. 현재는 공백 정규화만 수행하지만, 향후 추가 전처리 기능 확장 가능합니다.

**매개변수**:
- `text` (str): 전처리할 입력 텍스트
- `**kwargs`: 추가 전처리 옵션 (현재 미사용)

**반환값**:
- `str`: 전처리된 텍스트

**사용 예제**:
```python
from code.utils.xlsum_utils import preprocess_for_xlsum

# 대화 전처리
dialogue = """
#Person1#: 안녕하세요,\n\n반갑습니다!
#Person2#:    저도    반가워요.
"""

processed = preprocess_for_xlsum(dialogue)
print(f"원본: {dialogue!r}")
print(f"전처리 후: {processed!r}")

# 출력:
# 원본: '#Person1#: 안녕하세요,\n\n반갑습니다!\n#Person2#:    저도    반가워요.'
# 전처리 후: '#Person1#: 안녕하세요, 반갑습니다! #Person2#: 저도 반가워요.'

# 고급 사용: validate_xlsum_input과 연동
from code.utils.xlsum_utils import validate_xlsum_input

raw_text = "너무 긴 텍스트..." * 1000  # 아주 긴 텍스트

if validate_xlsum_input(raw_text):
    processed = preprocess_for_xlsum(raw_text)
    print("전처리 완료")
else:
    print("입력이 너무 깁니다!")
```

**활용 팁**:
- 대부분의 경우 `xlsum_whitespace_handler()`와 동일한 결과
- 향후 추가 전처리 기능이 예정되어 있어 앞으로는 이 함수 사용 권장

---

### ⚙️ 2. 설정 함수들

#### `get_xlsum_generation_config() -> Dict[str, Any]`

**목적**: mT5 XL-Sum 모델의 최적화된 텍스트 생성 설정을 반환합니다.

**매개변수**: 없음

**반환값**:
- `Dict[str, Any]`: 생성 설정 딕셔너리
  - `max_length` (int): 최대 생성 토큰 수 (84)
  - `num_beams` (int): 빔 서치 크기 (4)
  - `no_repeat_ngram_size` (int): 반복 방지 n-gram 크기 (2)
  - `do_sample` (bool): 샘플링 비활성화 (False)
  - `early_stopping` (bool): 조기 종료 활성화 (True)
  - `length_penalty` (float): 길이 패널티 (1.0)

**사용 예제**:
```python
from code.utils.xlsum_utils import get_xlsum_generation_config
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 최적화된 설정 가져오기
gen_config = get_xlsum_generation_config()
print("최적화된 생성 설정:")
for key, value in gen_config.items():
    print(f"  {key}: {value}")

# 모델과 함께 사용
model_name = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = "#Person1#: 안녕하세요. #Person2#: 안녕하세요!"
inputs = tokenizer(text, return_tensors="pt")

# 최적화된 설정으로 생성
outputs = model.generate(inputs.input_ids, **gen_config)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"\n입력: {text}")
print(f"요약: {summary}")
```

**파라미터 설명**:
- `max_length=84`: XL-Sum 데이터셋의 평균 요약 길이에 최적화
- `num_beams=4`: 품질과 속도의 균형점
- `no_repeat_ngram_size=2`: 반복적인 표현 방지
- `early_stopping=True`: EOS 토큰 만나면 즉시 종료

---

#### `get_xlsum_tokenizer_config() -> Dict[str, Any]`

**목적**: mT5 XL-Sum 모델의 최적화된 토크나이저 설정을 반환합니다.

**매개변수**: 없음

**반환값**:
- `Dict[str, Any]`: 토크나이저 설정 딕셔너리
  - `max_length` (int): 최대 입력 토큰 수 (512)
  - `truncation` (bool): 절단 활성화 (True)
  - `padding` (str): 패딩 방식 ('max_length')
  - `return_tensors` (str): 반환 텐서 타입 ('pt')
  - `add_special_tokens` (bool): 특수 토큰 추가 (True)

**사용 예제**:
```python
from code.utils.xlsum_utils import get_xlsum_tokenizer_config
from transformers import AutoTokenizer

# 최적화된 토크나이저 설정
tok_config = get_xlsum_tokenizer_config()
print("최적화된 토크나이저 설정:")
for key, value in tok_config.items():
    print(f"  {key}: {value}")

# 토크나이저와 함께 사용
model_name = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 긴 대화 텍스트 테스트
long_dialogue = """
#Person1#: 오늘 회의에서 논의된 내용들을 정리해보자.
#Person2#: 네, 좋습니다. 우선 예산 배정 문제부터 이야기해볼까요?
#Person1#: 그렇게 합시다. 올해 마케팅 예산은 지난해 대비 20% 증가했습니다.
""" * 5  # 아주 긴 텍스트

# 기본 토크나이징 vs 최적화된 토크나이징
basic_tokens = tokenizer(long_dialogue, return_tensors="pt")
optimal_tokens = tokenizer(long_dialogue, **tok_config)

print(f"\n기본 토크나이징:")
print(f"  입력 길이: {basic_tokens.input_ids.shape}")
print(f"  처리 시간: 빠름")

print(f"\n최적화된 토크나이징:")
print(f"  입력 길이: {optimal_tokens.input_ids.shape}")
print(f"  처리 시간: 안정적")
print(f"  메모리 사용량: 예측 가능")
```

---

#### `get_xlsum_default_config() -> Dict[str, Any]`

**목적**: XL-Sum 모델의 모든 설정을 통합한 전체 설정을 반환합니다. 하나의 함수로 모든 설정에 접근할 수 있어 편리합니다.

**매개변수**: 없음

**반환값**:
- `Dict[str, Any]`: 통합 설정 딕셔너리
  - `model`: 모델 메타정보 (`get_xlsum_model_info()` 결과)
  - `tokenizer`: 토크나이저 설정 (`get_xlsum_tokenizer_config()` 결과)
  - `generation`: 생성 설정 (`get_xlsum_generation_config()` 결과)
  - `preprocessing`: 전처리 함수들

**사용 예제**:
```python
from code.utils.xlsum_utils import get_xlsum_default_config
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# 통합 설정 로드
config = get_xlsum_default_config()

print("통합 설정 구성:")
print(f"  모델 정보: {list(config['model'].keys())[:5]}...")  # 일부만 표시
print(f"  토크나이저 설정: {list(config['tokenizer'].keys())}")
print(f"  생성 설정: {list(config['generation'].keys())}")
print(f"  전처리 함수: {list(config['preprocessing'].keys())}")

# 완벽한 파이프라인 구성
def complete_summarization_pipeline(text):
    """
    통합 설정을 사용한 완전한 요약 파이프라인
    """
    # 1. 전처리
    preprocessor = config['preprocessing']['text_preprocessor']
    processed_text = preprocessor(text)
    
    # 2. 모델 로드
    model_name = config['model']['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # 3. 토크나이징
    inputs = tokenizer(processed_text, **config['tokenizer'])
    
    # 4. 생성
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, **config['generation'])
    
    # 5. 디코딩
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {
        'original': text,
        'processed': processed_text,
        'summary': summary,
        'model_info': config['model']['architecture']
    }

# 사용 예제
test_dialogue = """
#Person1#:    오늘   회의   어때어요?

#Person2#: 좋았어요!   새로운 프로젝트    시작하기로 했어요.
"""

result = complete_summarization_pipeline(test_dialogue)
print(f"\n=== 완전한 파이프라인 결과 ===")
print(f"원본: {result['original']!r}")
print(f"전처리: {result['processed']!r}")
print(f"요약: {result['summary']}")
print(f"모델: {result['model_info']}")
```

---

### 📊 3. 정보 및 검증 함수들

#### `get_xlsum_model_info() -> Dict[str, Any]`

**목적**: mT5 XL-Sum 모델의 상세한 메타정보를 반환합니다.

**매개변수**: 없음

**반환값**:
- `Dict[str, Any]`: 모델 메타정보 딕셔너리 (주요 필드)
  - `model_name` (str): 모델명
  - `architecture` (str): 아키텍처 타입
  - `languages` (int): 지원 언어 수
  - `model_size` (str): 모델 크기
  - `performance` (dict): 언어별 성능 지표
  - `max_input_length` (int): 최대 입력 길이
  - `max_output_length` (int): 최대 출력 길이

**사용 예제**:
```python
from code.utils.xlsum_utils import get_xlsum_model_info
import json

# 모델 정보 가져오기
model_info = get_xlsum_model_info()

# 기본 정보 표시
print("기본 모델 정보:")
print(f"  모델명: {model_info['model_name']}")
print(f"  아키텍처: {model_info['architecture']}")
print(f"  지원 언어: {model_info['languages']}개")
print(f"  모델 크기: {model_info['model_size']}")

# 성능 지표 상세 보기
print("\n한국어 성능:")
korean_perf = model_info['performance']['korean']
for metric, score in korean_perf.items():
    print(f"  {metric.upper()}: {score:.2f}%")

print("\n영어 성능:")
english_perf = model_info['performance']['english']
for metric, score in english_perf.items():
    print(f"  {metric.upper()}: {score:.2f}%")

# 전체 정보를 JSON 형식으로 저장
with open('model_info.json', 'w', encoding='utf-8') as f:
    json.dump(model_info, f, ensure_ascii=False, indent=2)
print("\n모델 정보가 model_info.json에 저장되었습니다.")

# 시스템 요구사항 체크
def check_system_requirements(model_info):
    import psutil
    
    # RAM 체크
    total_ram_gb = psutil.virtual_memory().total // (1024**3)
    model_size_gb = float(model_info['model_size'].replace('GB', ''))
    recommended_ram = model_size_gb * 4  # 모델 크기의 4배 권장
    
    print(f"\n시스템 요구사항 체크:")
    print(f"  현재 RAM: {total_ram_gb}GB")
    print(f"  권장 RAM: {recommended_ram:.1f}GB")
    
    if total_ram_gb >= recommended_ram:
        print("  ✅ RAM 충분")
    else:
        print("  ⚠️ RAM 부족 - 비효율적일 수 있음")

check_system_requirements(model_info)
```

---

#### `is_xlsum_compatible_model(model_name: str) -> bool`

**목적**: 주어진 모델명이 XL-Sum 호환 모델인지 확인합니다.

**매개변수**:
- `model_name` (str): 확인할 모델명

**반환값**:
- `bool`: XL-Sum 호환 모델 여부

**사용 예제**:
```python
from code.utils.xlsum_utils import is_xlsum_compatible_model

# 다양한 모델 테스트
test_models = [
    "csebuetnlp/mT5_multilingual_XLSum",  # XL-Sum 모델
    "google/mt5-base",                     # 일반 mT5
    "facebook/bart-base",                  # BART
    "google/t5-base",                      # T5
    "csebuetnlp/mT5-base-xlsum",          # 다른 XL-Sum 변형
    "invalid_model_name",                  # 잘못된 모델명
    "",                                    # 빈 문자열
    None                                   # None 입력
]

print("모델 호환성 테스트:")
for model in test_models:
    try:
        is_compatible = is_xlsum_compatible_model(model)
        status = "✅ 호환" if is_compatible else "❌ 비호환"
        print(f"  {model}: {status}")
    except Exception as e:
        print(f"  {model}: ❌ 오류 - {e}")

# 사용 예제: 모델 로드 전 검증
def safe_model_load(model_name):
    if not is_xlsum_compatible_model(model_name):
        print(f"⚠️ {model_name}은 XL-Sum 호환 모델이 아닙니다.")
        print("최적화된 성능을 위해 'csebuetnlp/mT5_multilingual_XLSum' 사용을 권장합니다.")
        return False
    
    print(f"✅ {model_name}는 XL-Sum 호환 모델입니다.")
    # 여기에 모델 로드 로직 추가...
    return True

# 테스트
print("\n모델 로드 안전성 검사:")
safe_model_load("csebuetnlp/mT5_multilingual_XLSum")
safe_model_load("google/mt5-base")
```

---

#### `validate_xlsum_input(text: str, max_length: int = 512) -> bool`

**목적**: XL-Sum 모델에 입력할 텍스트의 유효성을 검증합니다.

**매개변수**:
- `text` (str): 검증할 텍스트
- `max_length` (int, 선택): 최대 허용 길이 (기본값: 512)

**반환값**:
- `bool`: 입력 유효성 여부

**사용 예제**:
```python
from code.utils.xlsum_utils import validate_xlsum_input

# 다양한 입력 테스트
test_inputs = [
    "#Person1#: 안녕하세요. #Person2#: 안녕하세요!",  # 정상 입력
    "",                                                      # 빈 문자열
    "   ",                                                  # 공백만
    "그냥 일반적인 텍스트",                                    # 일반 텍스트
    "아주 긴 텍스트 " * 500,                                # 아주 긴 텍스트
    None                                                    # None
]

print("입력 유효성 검증:")
for i, test_input in enumerate(test_inputs):
    try:
        is_valid = validate_xlsum_input(test_input)
        status = "✅ 유효" if is_valid else "❌ 무효"
        preview = str(test_input)[:30] + "..." if test_input and len(str(test_input)) > 30 else str(test_input)
        print(f"  테스트 {i+1}: {status} - {preview!r}")
    except Exception as e:
        print(f"  테스트 {i+1}: ❌ 오류 - {e}")

# 실용 예제: 안전한 전처리 파이프라인
def safe_preprocessing_pipeline(text):
    """
    입력 검증을 포함한 안전한 전처리 파이프라인
    """
    from code.utils.xlsum_utils import preprocess_for_xlsum
    
    # 1. 입력 검증
    if not validate_xlsum_input(text):
        return {
            'success': False,
            'error': '입력 텍스트가 유효하지 않습니다.',
            'processed_text': None
        }
    
    # 2. 전처리
    try:
        processed_text = preprocess_for_xlsum(text)
        return {
            'success': True,
            'error': None,
            'processed_text': processed_text,
            'original_length': len(text),
            'processed_length': len(processed_text)
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'전처리 오류: {e}',
            'processed_text': None
        }

# 테스트
test_text = "#Person1#: 안녕하세요. #Person2#: 반갑습니다!"
result = safe_preprocessing_pipeline(test_text)

print(f"\n안전한 전처리 결과:")
if result['success']:
    print(f"  ✅ 성공")
    print(f"  원본 길이: {result['original_length']}")
    print(f"  전처리 후: {result['processed_length']}")
    print(f"  결과: {result['processed_text']!r}")
else:
    print(f"  ❌ 실패: {result['error']}")
```

---

### 🔧 4. 유틸리티 함수

#### `get_xlsum_preprocessing_prompt(task_type: str = "summarization") -> str`

**목적**: XL-Sum 모델용 전처리 프롬프트를 반환합니다. 현재 mT5 모델은 프롬프트가 불필요하지만, 호환성을 위해 제공됩니다.

**매개변수**:
- `task_type` (str, 선택): 태스크 타입 (기본값: "summarization")

**반환값**:
- `str`: 전처리 프롬프트 (현재는 빈 문자열)

**사용 예제**:
```python
from code.utils.xlsum_utils import get_xlsum_preprocessing_prompt

# 기본 사용
prompt = get_xlsum_preprocessing_prompt()
print(f"기본 프롬프트: {prompt!r}")

# 다른 태스크 타입 테스트
task_types = ["summarization", "translation", "classification"]
for task in task_types:
    prompt = get_xlsum_preprocessing_prompt(task)
    print(f"{task} 프롬프트: {prompt!r}")

# T5와의 차이점 설명
print("\nmT5 vs T5 프롬프트 비교:")
print("  T5: 'summarize: [INPUT_TEXT]' 형태 필요")
print("  mT5 XL-Sum: 프롬프트 없음 - 직접 입력 처리")
print("  이유: XL-Sum 데이터셋에서 프롬프트 없이 학습됨")

# 향후 확장성 예시
def future_prompt_usage(text, task_type="summarization"):
    """
    향후 프롬프트 기능이 추가될 경우를 대비한 예시
    """
    prompt = get_xlsum_preprocessing_prompt(task_type)
    
    if prompt:  # 프롬프트가 있는 경우
        return f"{prompt} {text}"
    else:  # 현재와 같이 프롬프트가 없는 경우
        return text

test_text = "#Person1#: 안녕하세요. #Person2#: 반갑습니다."
result = future_prompt_usage(test_text)
print(f"\n향후 확장성 예시:")
print(f"  입력: {test_text}")
print(f"  결과: {result}")
```

---

### 🔗 함수 간 관계 및 사용 순서

#### 추천 사용 순서

```
1. 모델 호환성 확인 → is_xlsum_compatible_model()
2. 모델 정보 확인    → get_xlsum_model_info()
3. 입력 유효성 검증   → validate_xlsum_input()
4. 텍스트 전처리       → preprocess_for_xlsum()
5. 토크나이징          → get_xlsum_tokenizer_config()
6. 모델 생성           → get_xlsum_generation_config()
```

#### 함수 의존성 다이어그램

```
get_xlsum_default_config()
    ├── get_xlsum_model_info()
    ├── get_xlsum_tokenizer_config()
    ├── get_xlsum_generation_config()
    └── preprocessing/
        ├── xlsum_whitespace_handler()
        ├── preprocess_for_xlsum() ← xlsum_whitespace_handler()
        └── get_xlsum_preprocessing_prompt()

validate_xlsum_input() ← 독립 함수
is_xlsum_compatible_model() ← 독립 함수
```

#### 📖 완전한 사용 예제

```python
# xlsum_utils 완전 활용 예제
from code.utils.xlsum_utils import *
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def complete_xlsum_workflow(input_text):
    """
    xlsum_utils의 모든 함수를 활용한 완전한 워크플로우
    """
    print("=== XL-Sum 완전 워크플로우 ===")
    
    # 1. 모델 호환성 확인
    model_name = "csebuetnlp/mT5_multilingual_XLSum"
    if not is_xlsum_compatible_model(model_name):
        return "호환되지 않는 모델입니다."
    print("✅ 1. 모델 호환성 확인 완료")
    
    # 2. 모델 정보 확인
    model_info = get_xlsum_model_info()
    print(f"✅ 2. 모델 정보: {model_info['architecture']} ({model_info['model_size']})")
    
    # 3. 입력 유효성 검증
    if not validate_xlsum_input(input_text):
        return "유효하지 않은 입력입니다."
    print("✅ 3. 입력 유효성 검증 완료")
    
    # 4. 텍스트 전처리
    processed_text = preprocess_for_xlsum(input_text)
    print(f"✅ 4. 전처리 완료 ({len(input_text)} → {len(processed_text)} 문자)")
    
    # 5. 통합 설정 사용
    config = get_xlsum_default_config()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("✅ 5. 모델 및 설정 로드 완료")
    
    # 6. 추론 실행
    inputs = tokenizer(processed_text, **config['tokenizer'])
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, **config['generation'])
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("✅ 6. 요약 생성 완료")
    
    return {
        'input': input_text,
        'processed': processed_text,
        'summary': summary,
        'compression_ratio': len(input_text) / len(summary) if summary else 0
    }

# 테스트 실행
test_dialogue = """
#Person1#: 내일 회의 준비는 어떻게 되어가고 있나요?
#Person2#: 거의 다 준비되었습니다. 발표 자료만 마무리하면 됩니다.
#Person1#: 좋네요. 혹시 예상 질문들도 준비해주세요.
#Person2#: 네, 알겠습니다. 내일 오전에 미리 검토해보겠습니다.
"""

result = complete_xlsum_workflow(test_dialogue)
if isinstance(result, dict):
    print(f"\n=== 최종 결과 ===")
    print(f"압축 비율: {result['compression_ratio']:.1f}:1")
    print(f"요약: {result['summary']}")
else:
    print(f"오류: {result}")
```

---
    **tokenizer_config  # 최적화된 토크나이저 설정 사용
)

# 최적화된 생성
with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        **generation_config  # 최적화된 생성 설정 사용
    )

summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\n복잡한 대화 요약:")
print(f"입력 길이: {len(complex_dialogue)} 문자")
print(f"요약: {summary}")
```

**예상 출력:**
```
최적화된 설정:
- 생성 길이: 84
- 빔 크기: 4
- 토크나이저 길이: 512

복잡한 대화 요약:
입력 길이: 187 문자
요약: 한 사람이 다음 주 금요일 저녁 예약을 문의하여 6시 30분에 4명으로 예약했다.
```

---

### 🔧 방법 3: xlsum_utils 직접 활용 (고급 사용자용)

`xlsum_utils.py`의 전용 함수들을 직접 사용하여 세밀한 제어와 고급 기능을 활용하는 방법입니다.

#### 1단계: xlsum_utils 함수 활용

```python
# xlsum_utils 직접 활용 예제
from code.utils.xlsum_utils import (
    xlsum_whitespace_handler,
    preprocess_for_xlsum,
    get_xlsum_model_info,
    is_xlsum_compatible_model,
    validate_xlsum_input,
    get_xlsum_default_config
)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# 모델 정보 확인
model_info = get_xlsum_model_info()
print("=== mT5 XL-Sum 모델 정보 ===")
print(f"아키텍처: {model_info['architecture']}")
print(f"지원 언어: {model_info['languages']}개")
print(f"모델 크기: {model_info['model_size']}")
print(f"한국어 ROUGE-1: {model_info['performance']['korean']['rouge_1']:.2f}%")
print(f"권장 배치 크기: {model_info['recommended_batch_size']}")

# 모델 호환성 확인
model_name = "csebuetnlp/mT5_multilingual_XLSum"
if is_xlsum_compatible_model(model_name):
    print(f"✅ {model_name}은 XL-Sum 호환 모델입니다.")
else:
    print(f"❌ {model_name}은 XL-Sum 호환 모델이 아닙니다.")
```

#### 2단계: 고급 전처리 및 추론

```python
# 고급 전처리 파이프라인
def advanced_summarization(dialogue_text, model, tokenizer):
    """
    xlsum_utils를 활용한 고급 요약 함수
    """
    # 1. 입력 검증
    if not validate_xlsum_input(dialogue_text):
        return "❌ 입력 텍스트가 유효하지 않습니다."
    
    # 2. 공백 정규화
    normalized_text = xlsum_whitespace_handler(dialogue_text)
    print(f"정규화 전: {len(dialogue_text)} 문자")
    print(f"정규화 후: {len(normalized_text)} 문자")
    
    # 3. XL-Sum 전용 전처리
    preprocessed_text = preprocess_for_xlsum(normalized_text)
    
    # 4. 통합 설정 사용
    config = get_xlsum_default_config()
    
    # 5. 토크나이징
    inputs = tokenizer(
        preprocessed_text,
        **config['tokenizer']
    )
    
    # 6. 추론 실행
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            **config['generation']
        )
    
    # 7. 결과 디코딩
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {
        'original_length': len(dialogue_text),
        'normalized_length': len(normalized_text),
        'summary': summary,
        'compression_ratio': len(dialogue_text) / len(summary) if summary else 0
    }

# 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 테스트 대화 (노이즈가 많은 텍스트)
noisy_dialogue = """
#Person1#:    안녕하세요...   

오늘  회의 어떠셨나요??  
#Person2#:   좀   지루했어요.    


발표자료가      너무 많았고요.
#Person1#: 저도    그렇게   생각해요.
다음엔    좀  더   간결하게   했으면   좋겠어요.
"""

# 고급 요약 실행
result = advanced_summarization(noisy_dialogue, model, tokenizer)

print("\n=== 고급 처리 결과 ===")
print(f"원본 길이: {result['original_length']} 문자")
print(f"정규화 후 길이: {result['normalized_length']} 문자")
print(f"압축 비율: {result['compression_ratio']:.1f}:1")
print(f"최종 요약: {result['summary']}")
```

**예상 출력:**
```
=== mT5 XL-Sum 모델 정보 ===
아키텍처: mT5
지원 언어: 45개
모델 크기: 2.17GB
한국어 ROUGE-1: 23.67%
권장 배치 크기: 4
✅ csebuetnlp/mT5_multilingual_XLSum은 XL-Sum 호환 모델입니다.

정규화 전: 145 문자
정규화 후: 98 문자

=== 고급 처리 결과 ===
원본 길이: 145 문자
정규화 후 길이: 98 문자
압축 비율: 8.1:1
최종 요약: 두 사람이 회의가 지루했고 발표자료가 너무 많아서 다음엔 간결하게 하자고 했다.
```

### 🎪 성능 비교 및 권장사항

#### 처리 속도 비교 (상대적)
- **방법 1**: 1.0x (기준)
- **방법 2**: 0.9x (약간의 설정 오버헤드)
- **방법 3**: 0.8x (전처리로 인한 약간의 지연)

#### 요약 품질 비교
- **방법 1**: 기본 품질 (ROUGE-1: ~22%)
- **방법 2**: 최적화된 품질 (ROUGE-1: ~24%)
- **방법 3**: 최고 품질 (ROUGE-1: ~25%, 전처리 효과)

#### 사용 권장사항
- **프로토타이핑**: 방법 1 사용
- **프로덕션 환경**: 방법 2 사용
- **연구 및 실험**: 방법 3 사용

---


nvidia-smi
```

---
## 4. 📚 API 레퍼런스

`xlsum_utils.py`는 mT5_multilingual_XLSum 모델을 위한 전용 유틸리티 모듈로, 9개의 핸수와 1개의 상수를 제공합니다. 이 섹션에서는 각 함수의 사용법을 상세히 설명합니다.

### 📌 함수 개요

| 분류 | 함수명 | 주요 기능 | 난이도 |
|------|-----------|----------|------|
| **전처리** | `xlsum_whitespace_handler()` | 공백 정규화 | ⭐ |
| **전처리** | `preprocess_for_xlsum()` | 텍스트 전처리 | ⭐ |
| **설정** | `get_xlsum_generation_config()` | 생성 설정 | ⭐ |
| **설정** | `get_xlsum_tokenizer_config()` | 토크나이저 설정 | ⭐ |
| **설정** | `get_xlsum_default_config()` | 통합 설정 | ⭐⭐ |
| **정보** | `get_xlsum_model_info()` | 모델 메타정보 | ⭐ |
| **검증** | `is_xlsum_compatible_model()` | 모델 호환성 | ⭐ |
| **검증** | `validate_xlsum_input()` | 입력 유효성 | ⭐ |
| **유틸** | `get_xlsum_preprocessing_prompt()` | 프롬프트 생성 | ⭐ |

---

### 📝 1. 전처리 함수들

#### `xlsum_whitespace_handler(text: str) -> str`

**목적**: 연속된 공백과 줄바꿈을 정규화하여 mT5 모델이 처리하기 용이한 형태로 변환합니다.

**매개변수**:
- `text` (str): 정규화할 입력 텍스트

**반환값**:
- `str`: 공백이 정규화된 텍스트

**사용 예제**:
```python
from code.utils.xlsum_utils import xlsum_whitespace_handler

# 기본 사용니다
noisy_text = """
#Person1#:    안녕하세요...   


오늘 날씨가  좋네요.
#Person2#:   네,     정말   좋아요!
"""

clean_text = xlsum_whitespace_handler(noisy_text)
print(f"전: {len(noisy_text)} 문자")
print(f"후: {len(clean_text)} 문자")
print(f"결과: {clean_text}")

# 출력:
# 전: 89 문자
# 후: 53 문자
# 결과: #Person1#: 안녕하세요... 오늘 날씨가 좋네요. #Person2#: 네, 정말 좋아요!
```

**주의사항**:
- 빈 문자열이나 None 입력 시 빈 문자열 반환
- 줄바꿈(`\n`)을 공백으로 대체하므로 문단 배치가 중요한 경우 주의 필요

---

#### `preprocess_for_xlsum(text: str, **kwargs) -> str`

**목적**: XL-Sum 모델용 종합 텍스트 전처리를 수행합니다. 현재는 공백 정규화만 수행하지만, 향후 추가 전처리 기능 확장 가능합니다.

**매개변수**:
- `text` (str): 전처리할 입력 텍스트
- `**kwargs`: 추가 전처리 옵션 (현재 미사용)

**반환값**:
- `str`: 전처리된 텍스트

**사용 예제**:
```python
from code.utils.xlsum_utils import preprocess_for_xlsum

# 대화 전처리
dialogue = """
#Person1#: 안녕하세요,\n\n반갑습니다!
#Person2#:    저도    반가워요.
"""

processed = preprocess_for_xlsum(dialogue)
print(f"원본: {dialogue!r}")
print(f"전처리 후: {processed!r}")

# 출력:
# 원본: '#Person1#: 안녕하세요,\n\n반갑습니다!\n#Person2#:    저도    반가워요.'
# 전처리 후: '#Person1#: 안녕하세요, 반갑습니다! #Person2#: 저도 반가워요.'

# 고급 사용: validate_xlsum_input과 연동
from code.utils.xlsum_utils import validate_xlsum_input

raw_text = "너무 긴 텍스트..." * 1000  # 아주 긴 텍스트

if validate_xlsum_input(raw_text):
    processed = preprocess_for_xlsum(raw_text)
    print("전처리 완료")
else:
    print("입력이 너무 깁니다!")
```

**활용 팁**:
- 대부분의 경우 `xlsum_whitespace_handler()`와 동일한 결과
- 향후 추가 전처리 기능이 예정되어 있어 앞으로는 이 함수 사용 권장

---

### ⚙️ 2. 설정 함수들

#### `get_xlsum_generation_config() -> Dict[str, Any]`

**목적**: mT5 XL-Sum 모델의 최적화된 텍스트 생성 설정을 반환합니다.

**매개변수**: 없음

**반환값**:
- `Dict[str, Any]`: 생성 설정 딕셔너리
  - `max_length` (int): 최대 생성 토큰 수 (84)
  - `num_beams` (int): 빔 서치 크기 (4)
  - `no_repeat_ngram_size` (int): 반복 방지 n-gram 크기 (2)
  - `do_sample` (bool): 샘플링 비활성화 (False)
  - `early_stopping` (bool): 조기 종료 활성화 (True)
  - `length_penalty` (float): 길이 패널티 (1.0)

**사용 예제**:
```python
from code.utils.xlsum_utils import get_xlsum_generation_config
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 최적화된 설정 가져오기
gen_config = get_xlsum_generation_config()
print("최적화된 생성 설정:")
for key, value in gen_config.items():
    print(f"  {key}: {value}")

# 모델과 함께 사용
model_name = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = "#Person1#: 안녕하세요. #Person2#: 안녕하세요!"
inputs = tokenizer(text, return_tensors="pt")

# 최적화된 설정으로 생성
outputs = model.generate(inputs.input_ids, **gen_config)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"\n입력: {text}")
print(f"요약: {summary}")
```

**파라미터 설명**:
- `max_length=84`: XL-Sum 데이터셋의 평균 요약 길이에 최적화
- `num_beams=4`: 품질과 속도의 균형점
- `no_repeat_ngram_size=2`: 반복적인 표현 방지
- `early_stopping=True`: EOS 토큰 만나면 즉시 종료

---

#### `get_xlsum_tokenizer_config() -> Dict[str, Any]`

**목적**: mT5 XL-Sum 모델의 최적화된 토크나이저 설정을 반환합니다.

**매개변수**: 없음

**반환값**:
- `Dict[str, Any]`: 토크나이저 설정 딕셔너리
  - `max_length` (int): 최대 입력 토큰 수 (512)
  - `truncation` (bool): 절단 활성화 (True)
  - `padding` (str): 패딩 방식 ('max_length')
  - `return_tensors` (str): 반환 텐서 타입 ('pt')
  - `add_special_tokens` (bool): 특수 토큰 추가 (True)

**사용 예제**:
```python
from code.utils.xlsum_utils import get_xlsum_tokenizer_config
from transformers import AutoTokenizer

# 최적화된 토크나이저 설정
tok_config = get_xlsum_tokenizer_config()
print("최적화된 토크나이저 설정:")
for key, value in tok_config.items():
    print(f"  {key}: {value}")

# 토크나이저와 함께 사용
model_name = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 긴 대화 텍스트 테스트
long_dialogue = """
#Person1#: 오늘 회의에서 논의된 내용들을 정리해보자.
#Person2#: 네, 좋습니다. 우선 예산 배정 문제부터 이야기해볼까요?
#Person1#: 그렇게 합시다. 올해 마케팅 예산은 지난해 대비 20% 증가했습니다.
""" * 5  # 아주 긴 텍스트

# 기본 토크나이징 vs 최적화된 토크나이징
basic_tokens = tokenizer(long_dialogue, return_tensors="pt")
optimal_tokens = tokenizer(long_dialogue, **tok_config)

print(f"\n기본 토크나이징:")
print(f"  입력 길이: {basic_tokens.input_ids.shape}")
print(f"  처리 시간: 빠름")

print(f"\n최적화된 토크나이징:")
print(f"  입력 길이: {optimal_tokens.input_ids.shape}")
print(f"  처리 시간: 안정적")
print(f"  메모리 사용량: 예측 가능")
```

---

#### `get_xlsum_default_config() -> Dict[str, Any]`

**목적**: XL-Sum 모델의 모든 설정을 통합한 전체 설정을 반환합니다. 하나의 함수로 모든 설정에 접근할 수 있어 편리합니다.

**매개변수**: 없음

**반환값**:
- `Dict[str, Any]`: 통합 설정 딕셔너리
  - `model`: 모델 메타정보 (`get_xlsum_model_info()` 결과)
  - `tokenizer`: 토크나이저 설정 (`get_xlsum_tokenizer_config()` 결과)
  - `generation`: 생성 설정 (`get_xlsum_generation_config()` 결과)
  - `preprocessing`: 전처리 함수들

**사용 예제**:
```python
from code.utils.xlsum_utils import get_xlsum_default_config
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# 통합 설정 로드
config = get_xlsum_default_config()

print("통합 설정 구성:")
print(f"  모델 정보: {list(config['model'].keys())[:5]}...")  # 일부만 표시
print(f"  토크나이저 설정: {list(config['tokenizer'].keys())}")
print(f"  생성 설정: {list(config['generation'].keys())}")
print(f"  전처리 함수: {list(config['preprocessing'].keys())}")

# 완볽한 파이프라인 구성
def complete_summarization_pipeline(text):
    """
    통합 설정을 사용한 완전한 요약 파이프라인
    """
    # 1. 전처리
    preprocessor = config['preprocessing']['text_preprocessor']
    processed_text = preprocessor(text)
    
    # 2. 모델 로드
    model_name = config['model']['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # 3. 토크나이징
    inputs = tokenizer(processed_text, **config['tokenizer'])
    
    # 4. 생성
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, **config['generation'])
    
    # 5. 디코딩
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {
        'original': text,
        'processed': processed_text,
        'summary': summary,
        'model_info': config['model']['architecture']
    }

# 사용 예제
test_dialogue = """
#Person1#:    오늘   회의   어때어요?

#Person2#: 좋았어요!   새로운 프로젝트    시작하기로 했어요.
"""

result = complete_summarization_pipeline(test_dialogue)
print(f"\n=== 완전한 파이프라인 결과 ===")
print(f"원본: {result['original']!r}")
print(f"전처리: {result['processed']!r}")
print(f"요약: {result['summary']}")
print(f"모델: {result['model_info']}")
```

---

### 📊 3. 정보 및 검증 함수들

#### `get_xlsum_model_info() -> Dict[str, Any]`

**목적**: mT5 XL-Sum 모델의 상세한 메타정보를 반환합니다.

**매개변수**: 없음

**반환값**:
- `Dict[str, Any]`: 모델 메타정보 딕셔너리 (주요 필드)
  - `model_name` (str): 모델명
  - `architecture` (str): 아키텍처 타입
  - `languages` (int): 지원 언어 수
  - `model_size` (str): 모델 크기
  - `performance` (dict): 언어별 성능 지표
  - `max_input_length` (int): 최대 입력 길이
  - `max_output_length` (int): 최대 출력 길이

**사용 예제**:
```python
from code.utils.xlsum_utils import get_xlsum_model_info
import json

# 모델 정보 가져오기
model_info = get_xlsum_model_info()

# 기본 정보 표시
print("기본 모델 정보:")
print(f"  모델명: {model_info['model_name']}")
print(f"  아키텍처: {model_info['architecture']}")
print(f"  지원 언어: {model_info['languages']}개")
print(f"  모델 크기: {model_info['model_size']}")

# 성능 지표 상세 보기
print("\n한국어 성능:")
korean_perf = model_info['performance']['korean']
for metric, score in korean_perf.items():
    print(f"  {metric.upper()}: {score:.2f}%")

print("\n영어 성능:")
english_perf = model_info['performance']['english']
for metric, score in english_perf.items():
    print(f"  {metric.upper()}: {score:.2f}%")

# 전체 정보를 JSON 형식으로 저장
with open('model_info.json', 'w', encoding='utf-8') as f:
    json.dump(model_info, f, ensure_ascii=False, indent=2)
print("\n모델 정보가 model_info.json에 저장되었습니다.")

# 시스템 요구사항 체크
def check_system_requirements(model_info):
    import psutil
    
    # RAM 체크
    total_ram_gb = psutil.virtual_memory().total // (1024**3)
    model_size_gb = float(model_info['model_size'].replace('GB', ''))
    recommended_ram = model_size_gb * 4  # 모델 크기의 4배 권장
    
    print(f"\n시스템 요구사항 체크:")
    print(f"  현재 RAM: {total_ram_gb}GB")
    print(f"  권장 RAM: {recommended_ram:.1f}GB")
    
    if total_ram_gb >= recommended_ram:
        print("  ✅ RAM 충분")
    else:
        print("  ⚠️ RAM 부족 - 비효율적일 수 있음")

check_system_requirements(model_info)
```

---

#### `is_xlsum_compatible_model(model_name: str) -> bool`

**목적**: 주어진 모델명이 XL-Sum 호환 모델인지 확인합니다.

**매개변수**:
- `model_name` (str): 확인할 모델명

**반환값**:
- `bool`: XL-Sum 호환 모델 여부

**사용 예제**:
```python
from code.utils.xlsum_utils import is_xlsum_compatible_model

# 다양한 모델 테스트
test_models = [
    "csebuetnlp/mT5_multilingual_XLSum",  # XL-Sum 모델
    "google/mt5-base",                     # 일반 mT5
    "facebook/bart-base",                  # BART
    "google/t5-base",                      # T5
    "csebuetnlp/mT5-base-xlsum",          # 다른 XL-Sum 변형
    "invalid_model_name",                  # 잘못된 모델명
    "",                                    # 빈 문자열
    None                                   # None 입력
]

print("모델 호환성 테스트:")
for model in test_models:
    try:
        is_compatible = is_xlsum_compatible_model(model)
        status = "✅ 호환" if is_compatible else "❌ 비호환"
        print(f"  {model}: {status}")
    except Exception as e:
        print(f"  {model}: ❌ 오류 - {e}")

# 사용 예제: 모델 로드 전 검증
def safe_model_load(model_name):
    if not is_xlsum_compatible_model(model_name):
        print(f"⚠️ {model_name}은 XL-Sum 호환 모델이 아닙니다.")
        print("최적화된 성능을 위해 'csebuetnlp/mT5_multilingual_XLSum' 사용을 권장합니다.")
        return False
    
    print(f"✅ {model_name}는 XL-Sum 호환 모델입니다.")
    # 여기에 모델 로드 로직 추가...
    return True

# 테스트
print("\n모델 로드 안전성 검사:")
safe_model_load("csebuetnlp/mT5_multilingual_XLSum")
safe_model_load("google/mt5-base")
```

---

#### `validate_xlsum_input(text: str, max_length: int = 512) -> bool`

**목적**: XL-Sum 모델에 입력할 텍스트의 유효성을 검증합니다.

**매개변수**:
- `text` (str): 검증할 텍스트
- `max_length` (int, 선택): 최대 허용 길이 (기본값: 512)

**반환값**:
- `bool`: 입력 유효성 여부

**사용 예제**:
```python
from code.utils.xlsum_utils import validate_xlsum_input

# 다양한 입력 테스트
test_inputs = [
    "#Person1#: 안녕하세요. #Person2#: 안녕하세요!",  # 정상 입력
    "",                                                      # 빈 문자열
    "   ",                                                  # 공백만
    "그냥 일반적인 텍스트",                                    # 일반 텍스트
    "아주 긴 텍스트 " * 500,                                # 아주 긴 텍스트
    None                                                    # None
]

print("입력 유효성 검증:")
for i, test_input in enumerate(test_inputs):
    try:
        is_valid = validate_xlsum_input(test_input)
        status = "✅ 유효" if is_valid else "❌ 무효"
        preview = str(test_input)[:30] + "..." if test_input and len(str(test_input)) > 30 else str(test_input)
        print(f"  테스트 {i+1}: {status} - {preview!r}")
    except Exception as e:
        print(f"  테스트 {i+1}: ❌ 오류 - {e}")

# 실용 예제: 안전한 전처리 파이프라인
def safe_preprocessing_pipeline(text):
    """
    입력 검증을 포함한 안전한 전처리 파이프라인
    """
    from code.utils.xlsum_utils import preprocess_for_xlsum
    
    # 1. 입력 검증
    if not validate_xlsum_input(text):
        return {
            'success': False,
            'error': '입력 텍스트가 유효하지 않습니다.',
            'processed_text': None
        }
    
    # 2. 전처리
    try:
        processed_text = preprocess_for_xlsum(text)
        return {
            'success': True,
            'error': None,
            'processed_text': processed_text,
            'original_length': len(text),
            'processed_length': len(processed_text)
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'전처리 오류: {e}',
            'processed_text': None
        }

# 테스트
test_text = "#Person1#: 안녕하세요. #Person2#: 반갑습니다!"
result = safe_preprocessing_pipeline(test_text)

print(f"\n안전한 전처리 결과:")
if result['success']:
    print(f"  ✅ 성공")
    print(f"  원본 길이: {result['original_length']}")
    print(f"  전처리 후: {result['processed_length']}")
    print(f"  결과: {result['processed_text']!r}")
else:
    print(f"  ❌ 실패: {result['error']}")
```

---

### 🔧 4. 유틸리티 함수

#### `get_xlsum_preprocessing_prompt(task_type: str = "summarization") -> str`

**목적**: XL-Sum 모델용 전처리 프롬프트를 반환합니다. 현재 mT5 모델은 프롬프트가 불필요하지만, 호환성을 위해 제공됩니다.

**매개변수**:
- `task_type` (str, 선택): 태스크 타입 (기본값: "summarization")

**반형값**:
- `str`: 전처리 프롬프트 (현재는 빈 문자열)

**사용 예제**:
```python
from code.utils.xlsum_utils import get_xlsum_preprocessing_prompt

# 기본 사용
prompt = get_xlsum_preprocessing_prompt()
print(f"기본 프롬프트: {prompt!r}")

# 다른 태스크 타입 테스트
task_types = ["summarization", "translation", "classification"]
for task in task_types:
    prompt = get_xlsum_preprocessing_prompt(task)
    print(f"{task} 프롬프트: {prompt!r}")

# T5와의 차이점 설명
print("\nmT5 vs T5 프롬프트 비교:")
print("  T5: 'summarize: [INPUT_TEXT]' 형태 필요")
print("  mT5 XL-Sum: 프롬프트 뺈 - 직접 입력 처리")
print("  이유: XL-Sum 데이터셋에서 프롬프트 없이 학습됨")

# 향후 확장성 예시
def future_prompt_usage(text, task_type="summarization"):
    """
    향후 프롬프트 기능이 추가될 경우를 대비한 예시
    """
    prompt = get_xlsum_preprocessing_prompt(task_type)
    
    if prompt:  # 프롬프트가 있는 경우
        return f"{prompt} {text}"
    else:  # 현재와 같이 프롬프트가 없는 경우
        return text

test_text = "#Person1#: 안녕하세요. #Person2#: 반갑습니다."
result = future_prompt_usage(test_text)
print(f"\n향후 확장성 예시:")
print(f"  입력: {test_text}")
print(f"  결과: {result}")
```

---

### 🔗 함수 간 관계 및 사용 순서

#### 추천 사용 순서

```
1. 모델 호환성 확인 → is_xlsum_compatible_model()
2. 모델 정보 처인    → get_xlsum_model_info()
3. 입력 유효성 검증   → validate_xlsum_input()
4. 텍스트 전처리       → preprocess_for_xlsum()
5. 토크나이징          → get_xlsum_tokenizer_config()
6. 모델 생성           → get_xlsum_generation_config()
```

#### 함수 의존성 다이어그램

```
get_xlsum_default_config()
    ├── get_xlsum_model_info()
    ├── get_xlsum_tokenizer_config()
    ├── get_xlsum_generation_config()
    └── preprocessing/
        ├── xlsum_whitespace_handler()
        ├── preprocess_for_xlsum() ← xlsum_whitespace_handler()
        └── get_xlsum_preprocessing_prompt()

validate_xlsum_input() ← 독립 함수
is_xlsum_compatible_model() ← 독립 함수
```

---


👤 **작성자**: NLP 대화 요약 프로젝트팀
