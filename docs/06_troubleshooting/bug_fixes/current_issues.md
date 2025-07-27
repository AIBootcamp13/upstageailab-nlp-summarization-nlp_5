# 버그 수정 및 코드 개선 사항

## 🎯 개요

현재 코드에서 발견된 **버그**와 **개선이 필요한 사항**들을 정리합니다. 이러한 수정들은 시스템 안정성과 유지보수성을 위해 **반드시 처리**되어야 합니다.

---

## 🔴 Critical Bugs (즉시 수정 필요)

### 1. trainer.py - 절대 경로 하드코딩

#### 문제 위치
```python
# code/trainer.py, 라인 약 80-90
def setup_paths(self):
    base_output_dir = Path(self.config['general']['output_dir'])
    
    # 문제: 절대 경로가 하드코딩될 가능성
    if self.sweep_mode and wandb.run:
        self.output_dir = base_output_dir / f"sweep_{wandb.run.id}"
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = base_output_dir / f"{self.experiment_name}_{timestamp}"
```

#### 수정 방안
```python
# 수정된 코드
from utils.path_utils import PathManager

def setup_paths(self):
    # PathManager를 사용한 안전한 경로 처리
    base_output_dir = PathManager.resolve_path(self.config['general']['output_dir'])
    
    if self.sweep_mode and wandb.run:
        self.output_dir = base_output_dir / f"sweep_{wandb.run.id}"
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = base_output_dir / f"{self.experiment_name}_{timestamp}"
    
    # 디렉토리 존재 보장
    PathManager.ensure_dir(self.output_dir)
```

### 2. config_manager.py - 환경변수 매핑 부족

#### 문제 위치
```python
# code/utils/config_manager.py, 라인 약 40-50
self._env_mapping = {
    'WANDB_PROJECT': 'wandb.project',
    'WANDB_ENTITY': 'wandb.entity', 
    'MODEL_NAME': 'general.model_name',
    'OUTPUT_DIR': 'general.output_dir',
    'BATCH_SIZE': 'training.per_device_train_batch_size',
    'LEARNING_RATE': 'training.learning_rate',
    'NUM_EPOCHS': 'training.num_train_epochs'
}
```

#### 수정 방안
```python
# 확장된 환경변수 매핑
self._env_mapping = {
    # 기존 매핑
    'WANDB_PROJECT': 'wandb.project',
    'WANDB_ENTITY': 'wandb.entity',
    'MODEL_NAME': 'general.model_name',
    'OUTPUT_DIR': 'general.output_dir',
    'BATCH_SIZE': 'training.per_device_train_batch_size',
    'LEARNING_RATE': 'training.learning_rate',
    'NUM_EPOCHS': 'training.num_train_epochs',
    
    # 추가 매핑 (크로스 플랫폼 지원)
    'DATA_PATH': 'general.data_path',
    'MODEL_ARCHITECTURE': 'model.architecture',
    'MODEL_CHECKPOINT': 'model.checkpoint',
    'ENCODER_MAX_LEN': 'tokenizer.encoder_max_len',
    'DECODER_MAX_LEN': 'tokenizer.decoder_max_len',
    'NUM_BEAMS': 'generation.num_beams',
    'FP16': 'training.fp16',
    'SEED': 'general.seed',
    'DEVICE': 'general.device',
    
    # AI Stages 특화 환경변수
    'AISTAGES_DATA_DIR': 'general.data_path',
    'AISTAGES_OUTPUT_DIR': 'general.output_dir',
    'CUDA_VISIBLE_DEVICES': 'general.visible_devices'
}
```

### 3. scripts/setup_aistages.sh - 하드코딩된 경로

#### 문제 위치
```bash
# code/scripts/setup_aistages.sh, 라인 약 20-25
export PATH="/data/ephemeral/home/.local/bin:$PATH"
echo 'export PATH="/data/ephemeral/home/.local/bin:$PATH"' >> ~/.bashrc
```

#### 수정 방안
```bash
# 동적 경로 탐지
USER_HOME=$(eval echo ~$USER)
LOCAL_BIN_DIR="$USER_HOME/.local/bin"

# UV 설치 경로 동적 탐지
if [ -f "$LOCAL_BIN_DIR/uv" ]; then
    UV_PATH="$LOCAL_BIN_DIR"
elif command -v uv &> /dev/null; then
    UV_PATH=$(dirname $(which uv))
else
    echo "UV 설치 중..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    UV_PATH="$USER_HOME/.local/bin"
fi

# PATH 업데이트
export PATH="$UV_PATH:$PATH"
if ! grep -q "$UV_PATH" ~/.bashrc; then
    echo "export PATH=\"$UV_PATH:\$PATH\"" >> ~/.bashrc
fi
```

---

## 🟡 중요한 개선 사항

### 4. data_utils.py - 에러 처리 부족

#### 문제 위치
```python
# code/utils/data_utils.py 전반
def load_dataset(self, file_path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(file_path)  # 에러 처리 부족
    return df
```

#### 수정 방안
```python
def load_dataset(self, file_path: Union[str, Path]) -> pd.DataFrame:
    """
    데이터셋 로딩 (개선된 에러 처리 포함)
    """
    from utils.path_utils import PathManager
    
    # 경로 해결
    file_path = PathManager.resolve_path(file_path)
    
    # 파일 존재 확인
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    # 파일 권한 확인
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    
    try:
        # 인코딩 자동 감지 시도
        encodings = ['utf-8', 'cp949', 'euc-kr', 'latin-1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                self.logger.info(f"Successfully loaded with encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise ValueError(f"Could not read file with any supported encoding: {encodings}")
        
        # 빈 파일 확인
        if df.empty:
            raise ValueError(f"Dataset file is empty: {file_path}")
        
        self.logger.info(f"Loaded dataset: {len(df)} samples from {file_path}")
        return df
        
    except pd.errors.EmptyDataError:
        raise ValueError(f"Dataset file is empty or invalid: {file_path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Failed to parse CSV file: {e}")
    except Exception as e:
        self.logger.error(f"Unexpected error loading dataset: {e}")
        raise
```

### 5. sweep_runner.py - 메모리 관리 부족

#### 문제 위치
```python
# code/sweep_runner.py, train_function 메서드
def train_function(self):
    # ... 학습 코드 ...
    # 문제: 메모리 정리 없음
    result = trainer.train(datasets)
    return result  # 메모리 누수 가능성
```

#### 수정 방안
```python
def train_function(self):
    """
    단일 Sweep 실행을 위한 학습 함수 (메모리 최적화)
    """
    run = wandb.run
    
    if run is None:
        raise RuntimeError("WandB run not initialized")
    
    trainer = None
    try:
        # Sweep 파라미터 가져오기
        sweep_params = dict(wandb.config)
        
        logger.info(f"Starting sweep run: {run.id}")
        logger.info(f"Sweep parameters: {sweep_params}")
        
        # 기본 설정에 Sweep 파라미터 병합
        config = self.config_manager.merge_sweep_params(sweep_params)
        
        # 실험명 생성
        experiment_name = self._generate_experiment_name(sweep_params)
        
        # 트레이너 생성
        trainer = DialogueSummarizationTrainer(
            config=config,
            sweep_mode=True,
            experiment_name=experiment_name
        )
        
        # 컴포넌트 초기화
        trainer.initialize_components()
        
        # 데이터 준비
        datasets = trainer.prepare_data()
        
        # 학습 실행
        result = trainer.train(datasets)
        
        # WandB에 최종 결과 로깅
        wandb.run.summary.update({
            'best_rouge1_f1': result.best_metrics.get('rouge1_f1', 0),
            'best_rouge2_f1': result.best_metrics.get('rouge2_f1', 0),
            'best_rougeL_f1': result.best_metrics.get('rougeL_f1', 0),
            'best_rouge_combined_f1': result.best_metrics.get('rouge_combined_f1', 0),
            'final_loss': result.final_metrics.get('eval_loss', 0),
            'model_path': result.model_path
        })
        
        # 결과 저장
        self._save_sweep_result(run.id, sweep_params, result)
        
        logger.info(f"Sweep run {run.id} completed successfully")
        logger.info(f"Best ROUGE combined F1: {result.best_metrics.get('rouge_combined_f1', 0):.4f}")
        
        return result
        
    except Exception as e:
        logger.error(f"Sweep run failed: {str(e)}")
        wandb.run.summary['status'] = 'failed'
        wandb.run.summary['error'] = str(e)
        raise
    
    finally:
        # 메모리 정리
        if trainer is not None:
            # 모델 메모리 해제
            if hasattr(trainer, 'model') and trainer.model is not None:
                del trainer.model
            if hasattr(trainer, 'tokenizer') and trainer.tokenizer is not None:
                del trainer.tokenizer
            del trainer
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Python 가비지 컬렉션
        import gc
        gc.collect()
```

### 6. utils/__init__.py 파일 누락

#### 문제
```
code/utils/ 디렉토리에 __init__.py 파일이 없어 Python 모듈로 인식되지 않음
```

#### 수정 방안
```python
# code/utils/__init__.py (신규 생성)
"""
NLP Dialogue Summarization Utils Package

이 패키지는 NLP 대화 요약 프로젝트의 유틸리티 모듈들을 포함합니다.
"""

from .config_manager import ConfigManager, load_config
from .data_utils import DataProcessor, TextPreprocessor, DialogueSummarizationDataset
from .metrics import RougeCalculator

# 버전 정보
__version__ = "1.0.0"
__author__ = "NLP Team 5"

# 공통 유틸리티 함수들
def get_project_info():
    """프로젝트 정보 반환"""
    return {
        "name": "NLP Dialogue Summarization",
        "version": __version__,
        "author": __author__,
        "description": "AI 부트캠프 13기 NLP Advanced 대화 요약 프로젝트"
    }

# 모듈 임포트 시 자동 실행되는 설정
import logging
import warnings

# 로깅 설정
logging.getLogger(__name__).addHandler(logging.NullHandler())

# 불필요한 경고 억제
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

__all__ = [
    'ConfigManager',
    'load_config', 
    'DataProcessor',
    'TextPreprocessor',
    'DialogueSummarizationDataset',
    'RougeCalculator',
    'get_project_info'
]
```

### 7. requirements.txt 의존성 버전 이슈

#### 문제 위치
```txt
# code/requirements.txt
pandas==2.1.4
numpy==1.23.5
wandb==0.16.1
tqdm==4.66.1
pytorch_lightning==2.1.2  # 실제로는 사용하지 않음
transformers[torch]==4.35.2
rouge==1.0.1
jupyter==1.0.0
jupyterlab==4.0.9
```

#### 수정 방안
```txt
# 수정된 requirements.txt
# 데이터 처리
pandas>=2.1.0,<3.0.0
numpy>=1.23.0,<2.0.0

# 딥러닝 프레임워크
torch>=2.0.0,<3.0.0
transformers[torch]>=4.35.0,<5.0.0

# 평가 및 모니터링
evaluate>=0.4.0  # rouge 대신 사용
wandb>=0.16.0,<1.0.0

# 유틸리티
tqdm>=4.60.0
pyyaml>=6.0
pathlib2>=2.3.0; python_version < "3.4"

# 한국어 처리 (선택적)
konlpy>=0.6.0; extra == "korean"

# 개발 환경
jupyter>=1.0.0
jupyterlab>=4.0.0

# 추가 의존성
scipy>=1.9.0  # 통계 계산용
matplotlib>=3.5.0  # 그래프 생성용
seaborn>=0.11.0  # 데이터 시각화용
psutil>=5.8.0  # 시스템 모니터링용

# 선택적 의존성 (성능 최적화)
accelerate>=0.20.0; extra == "accelerate"
deepspeed>=0.9.0; extra == "deepspeed"
```

---

## 🟢 개선 권장 사항

### 8. 로깅 시스템 표준화

#### 현재 문제
각 모듈마다 다른 로깅 방식 사용

#### 개선 방안
```python
# code/utils/logging_utils.py (신규 생성)
import logging
import sys
from pathlib import Path
from typing import Optional
from utils.path_utils import PathManager

def setup_logger(name: str, 
                log_file: Optional[str] = None,
                level: str = "INFO",
                format_string: Optional[str] = None) -> logging.Logger:
    """
    표준화된 로거 설정
    
    Args:
        name: 로거 이름
        log_file: 로그 파일 경로 (None이면 콘솔만)
        level: 로깅 레벨
        format_string: 커스텀 포맷 문자열
    """
    logger = logging.getLogger(name)
    
    # 기존 핸들러 제거 (중복 방지)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # 기본 포맷
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 (선택적)
    if log_file:
        log_path = PathManager.resolve_path(log_file)
        PathManager.ensure_dir(log_path.parent)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# 프로젝트 전체에서 사용할 로거 팩토리
def get_logger(module_name: str) -> logging.Logger:
    """모듈별 표준 로거 반환"""
    output_dir = PathManager.get_output_dir()
    log_file = output_dir / "logs" / f"{module_name}.log"
    
    return setup_logger(
        name=module_name,
        log_file=str(log_file),
        level="INFO"
    )
```

### 9. 설정 검증 강화

#### 개선 방안
```python
# code/utils/config_validator.py (신규 생성)
from typing import Dict, Any, List, Tuple
import os
from pathlib import Path

class ConfigValidator:
    """설정 검증기"""
    
    def __init__(self):
        self.validation_rules = self._load_validation_rules()
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """검증 규칙 정의"""
        return {
            'training': {
                'learning_rate': {
                    'type': (int, float),
                    'range': (1e-7, 1e-2),
                    'required': True
                },
                'per_device_train_batch_size': {
                    'type': int,
                    'range': (1, 512),
                    'required': True
                },
                'num_train_epochs': {
                    'type': int,
                    'range': (1, 1000),
                    'required': True
                },
                'warmup_ratio': {
                    'type': (int, float),
                    'range': (0.0, 1.0),
                    'required': False
                }
            },
            'tokenizer': {
                'encoder_max_len': {
                    'type': int,
                    'range': (1, 8192),
                    'required': True
                },
                'decoder_max_len': {
                    'type': int,
                    'range': (1, 2048),
                    'required': True
                }
            },
            'generation': {
                'num_beams': {
                    'type': int,
                    'range': (1, 20),
                    'required': False
                },
                'length_penalty': {
                    'type': (int, float),
                    'range': (0.1, 3.0),
                    'required': False
                }
            }
        }
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        설정 검증
        
        Returns:
            (검증 통과 여부, 오류 메시지 리스트)
        """
        errors = []
        
        for section_name, section_rules in self.validation_rules.items():
            if section_name not in config:
                errors.append(f"Missing required section: {section_name}")
                continue
            
            section_config = config[section_name]
            
            for param_name, rules in section_rules.items():
                # 필수 파라미터 확인
                if rules.get('required', False) and param_name not in section_config:
                    errors.append(f"Missing required parameter: {section_name}.{param_name}")
                    continue
                
                if param_name not in section_config:
                    continue  # 선택적 파라미터
                
                value = section_config[param_name]
                
                # 타입 검증
                if 'type' in rules:
                    expected_types = rules['type']
                    if not isinstance(expected_types, tuple):
                        expected_types = (expected_types,)
                    
                    if not isinstance(value, expected_types):
                        errors.append(f"Invalid type for {section_name}.{param_name}: "
                                    f"expected {expected_types}, got {type(value)}")
                        continue
                
                # 범위 검증
                if 'range' in rules and isinstance(value, (int, float)):
                    min_val, max_val = rules['range']
                    if not (min_val <= value <= max_val):
                        errors.append(f"Value out of range for {section_name}.{param_name}: "
                                    f"{value} not in [{min_val}, {max_val}]")
        
        # 교차 검증 (예: encoder_max_len > decoder_max_len)
        cross_validation_errors = self._cross_validate(config)
        errors.extend(cross_validation_errors)
        
        return len(errors) == 0, errors
    
    def _cross_validate(self, config: Dict[str, Any]) -> List[str]:
        """교차 검증"""
        errors = []
        
        # 토크나이저 길이 검증
        if 'tokenizer' in config:
            tokenizer_config = config['tokenizer']
            encoder_len = tokenizer_config.get('encoder_max_len')
            decoder_len = tokenizer_config.get('decoder_max_len')
            
            if encoder_len and decoder_len and decoder_len > encoder_len:
                errors.append("decoder_max_len should not be greater than encoder_max_len")
        
        # 배치 크기와 메모리 검증
        if 'training' in config and 'tokenizer' in config:
            batch_size = config['training'].get('per_device_train_batch_size', 0)
            seq_len = config['tokenizer'].get('encoder_max_len', 0)
            
            # 간단한 메모리 추정 (GPU 메모리 기반)
            estimated_memory_gb = (batch_size * seq_len * 4) / (1024**3)  # 매우 단순화
            
            if estimated_memory_gb > 16:  # 16GB 임계값
                errors.append(f"Configuration may require too much memory (~{estimated_memory_gb:.1f}GB). "
                            f"Consider reducing batch_size or encoder_max_len")
        
        return errors
```

---

## 📋 수정 우선순위 및 일정

### 🔴 Critical (즉시 수정 - Week 1)
1. **경로 처리 개선** (trainer.py, config_manager.py) - 2시간
2. **setup_aistages.sh 경로 수정** - 1시간
3. **utils/__init__.py 추가** - 30분

### 🟡 중요 (Week 1-2 중)
4. **data_utils.py 에러 처리 강화** - 3시간
5. **sweep_runner.py 메모리 관리** - 2시간
6. **requirements.txt 정리** - 1시간

### 🟢 개선 (Week 2-3 중)
7. **로깅 시스템 표준화** - 4시간
8. **설정 검증 강화** - 3시간

---

## 🛠️ 수정 체크리스트

### Phase 1: 긴급 수정 (Day 1-2)
- [ ] PathManager 클래스 구현 완료
- [ ] trainer.py 경로 처리 수정
- [ ] config_manager.py 환경변수 매핑 확장
- [ ] setup_aistages.sh 동적 경로 처리
- [ ] utils/__init__.py 추가

### Phase 2: 안정성 개선 (Day 3-5)
- [ ] data_utils.py 에러 처리 강화
- [ ] sweep_runner.py 메모리 관리 개선
- [ ] requirements.txt 의존성 정리
- [ ] 크로스 플랫폼 테스트 수행

### Phase 3: 품질 향상 (Week 2)
- [ ] 로깅 시스템 표준화
- [ ] 설정 검증 시스템 구현
- [ ] 코드 스타일 통일
- [ ] 문서화 업데이트

---

## 🧪 테스트 시나리오

### 크로스 플랫폼 테스트
```bash
# Windows에서 테스트
python -c "from utils.path_utils import PathManager; print(PathManager.get_project_root())"

# macOS에서 테스트  
python -c "from utils.path_utils import PathManager; print(PathManager.get_project_root())"

# Linux에서 테스트
python -c "from utils.path_utils import PathManager; print(PathManager.get_project_root())"
```

### 메모리 누수 테스트
```python
# 메모리 사용량 모니터링
import psutil
import torch

def test_memory_leak():
    initial_memory = psutil.virtual_memory().used
    initial_gpu = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    # 여러 번 학습 실행
    for i in range(5):
        # 학습 실행
        result = run_training()
        
        # 메모리 체크
        current_memory = psutil.virtual_memory().used
        current_gpu = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        print(f"Iteration {i}: CPU +{(current_memory - initial_memory) / 1024**2:.1f}MB, "
              f"GPU +{(current_gpu - initial_gpu) / 1024**2:.1f}MB")
```

---

## 📈 개선 효과 예상

### 안정성 향상
- **크로스 플랫폼 호환성**: 100% → Windows, macOS, Linux 모두 지원
- **에러 발생률**: 50% 감소 → 명확한 에러 메시지와 처리
- **메모리 효율성**: 30% 향상 → 메모리 누수 방지

### 개발 효율성 향상
- **디버깅 시간**: 40% 단축 → 표준화된 로깅
- **설정 오류**: 80% 감소 → 자동 검증 시스템
- **환경 설정 시간**: 70% 단축 → 자동 설정 스크립트

### 유지보수성 향상
- **코드 일관성**: 크게 향상 → 표준화된 구조
- **새 기능 추가**: 용이 → 모듈화된 아키텍처
- **팀 협업**: 향상 → 명확한 컨벤션

이러한 수정들을 통해 프로젝트의 안정성과 확장성을 크게 향상시킬 수 있습니다.