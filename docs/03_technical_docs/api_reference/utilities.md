# 🛠️ 유틸리티 함수 참조

프로젝트에서 공통으로 사용되는 유틸리티 함수 및 헬퍼 클래스들의 API 참조 문서입니다.

## 📋 목차
1. [경로 관리 유틸리티](#경로-관리-유틸리티)
2. [디바이스 최적화 유틸리티](#디바이스-최적화-유틸리티)
3. [데이터 처리 유틸리티](#데이터-처리-유틸리티)
4. [메트릭 계산 유틸리티](#메트릭-계산-유틸리티)
5. [로깅 유틸리티](#로깅-유틸리티)
6. [실험 관리 유틸리티](#실험-관리-유틸리티)

---

## 경로 관리 유틸리티

### `PathManager` 클래스

프로젝트 내 상대 경로 관리를 담당하는 핵심 유틸리티입니다.

```python
class PathManager:
    """프로젝트 경로 관리 유틸리티"""
    
    @staticmethod
    def resolve_path(relative_path: Union[str, Path]) -> Path:
        """상대 경로를 절대 경로로 변환"""
        
    @staticmethod
    def ensure_dir(dir_path: Union[str, Path]) -> Path:
        """디렉토리 존재 보장"""
        
    @staticmethod
    def get_project_root() -> Path:
        """프로젝트 루트 디렉토리 반환"""
```

**주요 기능:**
- 프로젝트 루트 기준 상대 경로 강제
- 크로스 플랫폼 호환성 보장
- 디렉토리 자동 생성

**사용 예시:**
```python
from utils.path_utils import PathManager

# 상대 경로 해결
data_path = PathManager.resolve_path("data/train.csv")

# 디렉토리 생성
PathManager.ensure_dir("outputs/experiments")
```

---

## 디바이스 최적화 유틸리티

### `get_optimal_device()` 함수

시스템에서 사용 가능한 최적의 디바이스를 자동으로 감지합니다.

```python
def get_optimal_device() -> str:
    """
    최적 디바이스 자동 감지
    
    Returns:
        str: "mps", "cuda", "cpu" 중 하나
    """
```

**감지 순서:**
1. CUDA 사용 가능 → "cuda"
2. MPS 사용 가능 (Apple Silicon) → "mps"  
3. 기본값 → "cpu"

### `get_device_config()` 함수

디바이스별 최적화 설정을 반환합니다.

```python
def get_device_config(device: str) -> Dict[str, Any]:
    """
    디바이스별 최적화 설정 반환
    
    Args:
        device: 디바이스 타입
        
    Returns:
        Dict: 최적화 설정
    """
```

**반환 설정:**
- `batch_size`: 권장 배치 크기
- `fp16`: Float16 사용 여부
- `dataloader_pin_memory`: Pin memory 사용 여부
- `torch_dtype`: 권장 데이터 타입

**사용 예시:**
```python
from utils.device_utils import get_optimal_device, get_device_config

device = get_optimal_device()
config = get_device_config(device)

print(f"디바이스: {device}")
print(f"권장 배치 크기: {config['batch_size']}")
```

---

## 데이터 처리 유틸리티

### `validate_dialogue_input()` 함수

대화 입력 데이터의 유효성을 검증하고 정제합니다.

```python
def validate_dialogue_input(dialogue: str) -> str:
    """
    대화 입력 검증 및 정제
    
    Args:
        dialogue: 입력 대화 텍스트
        
    Returns:
        str: 검증 및 정제된 대화 텍스트
        
    Raises:
        ValueError: 유효하지 않은 입력인 경우
    """
```

**검증 항목:**
- 길이 제한 확인 (10~10,000자)
- 악성 패턴 검사
- HTML 태그 제거
- 공백 정리

### `parse_multiple_summaries()` 함수

구분자로 분리된 다중 요약문을 파싱합니다.

```python
def parse_multiple_summaries(summary_text: str) -> List[str]:
    """
    구분자 분리된 요약문 파싱
    
    지원 구분자: |||, ##, ---, 줄바꿈
    """
```

**사용 예시:**
```python
from utils.data_utils import validate_dialogue_input, parse_multiple_summaries

# 입력 검증
clean_dialogue = validate_dialogue_input(raw_dialogue)

# 다중 요약문 파싱
summaries = parse_multiple_summaries("요약1|||요약2|||요약3")
```

---

## 메트릭 계산 유틸리티

### `compute_rouge_scores()` 함수

ROUGE 점수를 계산하는 핵심 함수입니다.

```python
def compute_rouge_scores(
    predictions: List[str],
    references: List[str],
    use_korean_tokenizer: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    ROUGE 점수 계산
    
    Returns:
        Dict: rouge1, rouge2, rougeL 점수
    """
```

### `normalize_text()` 함수

텍스트 정규화를 수행합니다.

```python
def normalize_text(text: str, korean: bool = True) -> str:
    """
    텍스트 정규화
    
    - 공백 정리
    - 특수문자 처리
    - 한국어 토크나이징 (선택)
    """
```

**사용 예시:**
```python
from utils.metrics import compute_rouge_scores, normalize_text

# 텍스트 정규화
clean_pred = normalize_text(prediction)
clean_ref = normalize_text(reference)

# ROUGE 점수 계산
scores = compute_rouge_scores([clean_pred], [clean_ref])
```

---

## 로깅 유틸리티

### `StructuredLogger` 클래스

구조화된 JSON 로깅을 제공합니다.

```python
class StructuredLogger:
    """구조화된 JSON 로거"""
    
    def __init__(self, name: str, level: str = "INFO"):
        """로거 초기화"""
        
    def info(self, message: str, **kwargs):
        """정보 로그"""
        
    def error(self, message: str, **kwargs):
        """에러 로그"""
        
    def warning(self, message: str, **kwargs):
        """경고 로그"""
```

### `setup_logging()` 함수

전역 로깅 시스템을 초기화합니다.

```python
def setup_logging(
    log_file: str = "logs/app.log",
    level: str = "INFO"
) -> None:
    """로깅 시스템 초기화"""
```

**사용 예시:**
```python
from utils.logging import StructuredLogger, setup_logging

# 로깅 시스템 초기화
setup_logging()

# 구조화된 로거 사용
logger = StructuredLogger("training")
logger.info("학습 시작", epoch=1, batch_size=16)
```

---

## 실험 관리 유틸리티

### `generate_experiment_id()` 함수

고유한 실험 ID를 생성합니다.

```python
def generate_experiment_id(prefix: str = "exp") -> str:
    """
    고유 실험 ID 생성
    
    Format: {prefix}_{timestamp}_{random}
    """
```

### `save_experiment_config()` 함수

실험 설정을 저장합니다.

```python
def save_experiment_config(
    config: Dict[str, Any],
    experiment_id: str,
    output_dir: str = "outputs/experiments"
) -> Path:
    """실험 설정 JSON 파일로 저장"""
```

### `load_experiment_config()` 함수

저장된 실험 설정을 로딩합니다.

```python
def load_experiment_config(experiment_id: str) -> Dict[str, Any]:
    """실험 설정 로딩"""
```

**사용 예시:**
```python
from utils.experiment_utils import (
    generate_experiment_id,
    save_experiment_config,
    load_experiment_config
)

# 실험 ID 생성
exp_id = generate_experiment_id("kobart")

# 설정 저장
config = {"model": "kobart", "lr": 0.001}
save_experiment_config(config, exp_id)

# 설정 로딩
loaded_config = load_experiment_config(exp_id)
```

---

## 🔧 유틸리티 활용 패턴

### 1. 기본 초기화 패턴

```python
from utils.device_utils import get_optimal_device, get_device_config
from utils.path_utils import PathManager
from utils.logging import setup_logging

# 기본 초기화
setup_logging()
device = get_optimal_device()
config = get_device_config(device)

# 경로 설정
data_path = PathManager.resolve_path("data/train.csv")
output_dir = PathManager.ensure_dir("outputs/models")
```

### 2. 데이터 처리 패턴

```python
from utils.data_utils import validate_dialogue_input, parse_multiple_summaries
from utils.metrics import normalize_text, compute_rouge_scores

# 데이터 검증 및 정제
clean_dialogue = validate_dialogue_input(raw_dialogue)
summaries = parse_multiple_summaries(multi_summary)

# 평가 시 텍스트 정규화
norm_pred = normalize_text(prediction)
norm_refs = [normalize_text(ref) for ref in references]
scores = compute_rouge_scores([norm_pred], norm_refs)
```

### 3. 실험 추적 패턴

```python
from utils.experiment_utils import generate_experiment_id, save_experiment_config
from utils.logging import StructuredLogger

# 실험 초기화
exp_id = generate_experiment_id("baseline")
logger = StructuredLogger(exp_id)

# 설정 저장
config = {"model": "kobart", "batch_size": 16}
save_experiment_config(config, exp_id)

# 진행 상황 로깅
logger.info("학습 시작", experiment_id=exp_id, config=config)
```

---

## 🚀 성능 최적화 팁

### 1. 메모리 효율성
- `PathManager`는 정적 메서드로 메모리 효율적
- `StructuredLogger`는 싱글톤 패턴으로 사용 권장

### 2. 디바이스 최적화
- `get_optimal_device()`는 앱 시작 시 한 번만 호출
- 디바이스별 설정은 전역 변수로 캐싱

### 3. 파일 I/O 최적화
- 배치 단위로 로그 기록
- 실험 설정은 JSON으로 캐싱

---

## 🔗 관련 문서

- [핵심 모듈 API](./core_modules.md) - 메인 클래스 참조
- [프로젝트 구조](../architecture/project_structure.md) - 전체 아키텍처
- [사용자 가이드](../../02_user_guides/README.md) - 실제 사용법

---

이 유틸리티들을 활용하여 일관되고 효율적인 코드를 작성하세요.
