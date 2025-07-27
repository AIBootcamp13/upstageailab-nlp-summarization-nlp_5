# 🍎 Mac/Linux 환경 자동 감지 및 MPS/CUDA 지원 구현 가이드

## 📊 개요

이 문서는 Mac(Apple Silicon)과 Linux 환경을 자동으로 감지하여 적절한 GPU 가속(MPS/CUDA)을 사용하도록 설정하는 방법을 설명합니다.

**작성일**: 2025-07-26  
**현재 상태**: ❌ MPS 미지원 (CUDA만 지원)  
**목표**: Mac M1/M2의 MPS와 Linux/Windows의 CUDA 자동 감지 및 최적화

---

## 🔴 현재 문제점

### trainer.py의 현재 디바이스 설정 코드
```python
def _setup_device(self) -> torch.device:
    """디바이스 설정"""
    device_config = self.config['general'].get('device', 'auto')
    
    if device_config == 'auto':
        # 문제: MPS를 전혀 고려하지 않음
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_config)
    
    logger.info(f"Using device: {device}")
    return device
```

**영향**:
- Mac M1/M2 사용자는 GPU 가속을 받을 수 없음
- CPU 대비 10배 이상 느린 학습 속도

---

## ✅ 해결 방안

### 1. device_utils.py 생성

```python
# code/utils/device_utils.py
import torch
import platform
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def get_system_info() -> Dict[str, Any]:
    """시스템 정보 수집"""
    return {
        'platform': platform.system(),  # 'Darwin' (Mac), 'Linux', 'Windows'
        'platform_release': platform.release(),
        'platform_version': platform.version(),
        'architecture': platform.machine(),  # 'arm64' (M1/M2), 'x86_64', etc.
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'mps_available': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
    }

def get_optimal_device() -> str:
    """
    환경에 따른 최적 디바이스 자동 선택
    
    우선순위:
    1. CUDA (NVIDIA GPU) - Linux/Windows
    2. MPS (Apple Silicon GPU) - Mac M1/M2
    3. CPU (fallback)
    
    Returns:
        str: 'cuda', 'mps', 또는 'cpu'
    """
    system_info = get_system_info()
    
    # CUDA 확인 (주로 Linux/Windows)
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"CUDA device detected: {device_name} ({memory_gb:.1f}GB)")
        logger.info(f"CUDA version: {torch.version.cuda}")
        
    # MPS 확인 (Mac M1/M2)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Apple MPS (Metal Performance Shaders) device detected")
        logger.info(f"Platform: {system_info['platform']} {system_info['architecture']}")
        
        # MPS 사용 시 주의사항 로깅
        logger.warning("MPS Notes: Some operations may be slower than CUDA. "
                      "Mixed precision (fp16) is not fully supported.")
        
    # CPU fallback
    else:
        device = "cpu"
        logger.info("No GPU detected, using CPU")
        logger.info(f"Platform: {system_info['platform']} {system_info['architecture']}")
        
        # 추천사항 제공
        if system_info['platform'] == 'Darwin' and system_info['architecture'] == 'arm64':
            logger.warning("You appear to be on Apple Silicon but MPS is not available. "
                          "Please ensure PyTorch is installed with MPS support: "
                          "pip install torch torchvision torchaudio")
    
    return device

def setup_device_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    디바이스별 최적화된 설정 자동 구성
    
    Args:
        config: 기존 설정 딕셔너리
        
    Returns:
        Dict[str, Any]: 디바이스 최적화가 적용된 설정
    """
    device = get_optimal_device()
    
    # 디바이스별 기본 최적화 설정
    device_optimizations = {
        "cuda": {
            "fp16": True,  # CUDA는 mixed precision 완벽 지원
            "dataloader_num_workers": 4,
            "gradient_accumulation_steps": 1,
            "per_device_train_batch_size": 8,
            "gradient_checkpointing": False,  # 메모리 충분하면 비활성화
        },
        "mps": {
            "fp16": False,  # MPS는 현재 fp16 미지원
            "dataloader_num_workers": 0,  # MPS는 multiprocessing과 호환성 이슈
            "gradient_accumulation_steps": 4,  # 작은 배치 크기 보상
            "per_device_train_batch_size": 4,  # 안정성을 위해 작게
            "gradient_checkpointing": True,  # 메모리 절약
        },
        "cpu": {
            "fp16": False,  # CPU는 fp16 비효율적
            "dataloader_num_workers": 2,
            "gradient_accumulation_steps": 8,
            "per_device_train_batch_size": 2,
            "gradient_checkpointing": True,
        }
    }
    
    # 기존 설정에 디바이스 최적화 적용
    optimizations = device_optimizations[device]
    
    if "training" not in config:
        config["training"] = {}
    
    # 사용자가 명시적으로 설정하지 않은 값만 업데이트
    for key, value in optimizations.items():
        if key not in config["training"]:
            config["training"][key] = value
            logger.info(f"Auto-configured {key}: {value} for {device}")
    
    # 디바이스 정보 추가
    config["device"] = device
    config["device_info"] = get_system_info()
    
    return config

def log_device_info():
    """디바이스 정보 상세 로깅"""
    system_info = get_system_info()
    device = get_optimal_device()
    
    logger.info("="*50)
    logger.info("System Information:")
    logger.info(f"  Platform: {system_info['platform']}")
    logger.info(f"  Architecture: {system_info['architecture']}")
    logger.info(f"  Python: {system_info['python_version']}")
    logger.info(f"  PyTorch: {system_info['torch_version']}")
    
    if device == "cuda":
        logger.info(f"  CUDA Version: {system_info['cuda_version']}")
        logger.info(f"  GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f}GB)")
    
    elif device == "mps":
        logger.info("  MPS: Available (Apple Silicon GPU)")
    
    logger.info(f"  Selected Device: {device}")
    logger.info("="*50)
```

### 2. trainer.py 수정

```python
# trainer.py의 _setup_device 메서드를 다음과 같이 수정

from utils.device_utils import get_optimal_device, setup_device_config, log_device_info

def _setup_device(self) -> torch.device:
    """디바이스 설정 (MPS/CUDA 자동 감지)"""
    # 디바이스 정보 로깅
    log_device_info()
    
    # 설정에서 명시적 디바이스 지정 확인
    device_config = self.config['general'].get('device', 'auto')
    
    if device_config == 'auto':
        # 자동 감지
        device_str = get_optimal_device()
        
        # 설정 자동 최적화
        self.config = setup_device_config(self.config)
    else:
        # 사용자 지정 디바이스
        device_str = device_config
        logger.info(f"Using user-specified device: {device_str}")
    
    device = torch.device(device_str)
    logger.info(f"Final device selection: {device}")
    
    # MPS 사용 시 추가 설정
    if device_str == "mps":
        # MPS 관련 환경 변수 설정 (선택사항)
        import os
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        logger.info("MPS environment variables configured")
    
    return device
```

### 3. 설치 요구사항 업데이트

```bash
# requirements.txt에 추가
torch>=2.0.0  # MPS는 PyTorch 2.0 이상에서 안정적

# Mac M1/M2 사용자를 위한 설치 명령 (README.md에 추가)
# For Mac M1/M2 (Apple Silicon):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For Linux/Windows with CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🧪 테스트 코드

```python
# test_device_setup.py
import sys
sys.path.append('code')

from utils.device_utils import get_optimal_device, get_system_info, setup_device_config

def test_device_detection():
    """디바이스 감지 테스트"""
    print("=== Device Detection Test ===")
    
    # 시스템 정보
    info = get_system_info()
    print(f"Platform: {info['platform']}")
    print(f"Architecture: {info['architecture']}")
    print(f"CUDA Available: {info['cuda_available']}")
    print(f"MPS Available: {info['mps_available']}")
    
    # 최적 디바이스 선택
    device = get_optimal_device()
    print(f"\nSelected Device: {device}")
    
    # 설정 최적화 테스트
    config = {"training": {}}
    optimized_config = setup_device_config(config)
    
    print("\nOptimized Configuration:")
    for key, value in optimized_config['training'].items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    test_device_detection()
```

---

## 📋 구현 체크리스트

- [ ] `code/utils/device_utils.py` 파일 생성
- [ ] `get_optimal_device()` 함수 구현
- [ ] `setup_device_config()` 함수 구현
- [ ] trainer.py의 `_setup_device()` 메서드 수정
- [ ] requirements.txt 업데이트
- [ ] README.md에 플랫폼별 설치 가이드 추가
- [ ] 테스트 스크립트 실행 및 검증

---

## ⚠️ 주의사항

### MPS 사용 시 제한사항
1. **Mixed Precision (fp16) 미지원**: 현재 MPS는 fp16 연산을 완전히 지원하지 않음
2. **일부 연산 느림**: 특정 연산은 CUDA보다 느릴 수 있음
3. **DataLoader 멀티프로세싱 이슈**: num_workers는 0으로 설정 필요

### 성능 비교
- **CUDA**: 가장 빠름, 모든 기능 지원
- **MPS**: CUDA의 70-80% 성능, 일부 제한
- **CPU**: 가장 느림 (GPU 대비 10-20배)

---

## 🚀 예상 효과

1. **Mac M1/M2 사용자**: CPU 대비 5-10배 빠른 학습
2. **자동 최적화**: 플랫폼별 최적 설정 자동 적용
3. **호환성**: 모든 플랫폼에서 코드 수정 없이 실행

---

**작성자**: AI Assistant  
**최종 검토**: 2025-07-26  
**문서 버전**: 1.0