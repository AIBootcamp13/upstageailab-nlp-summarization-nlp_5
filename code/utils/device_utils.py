"""
디바이스 관리 유틸리티

Mac M1/M2의 MPS와 Linux/Windows의 CUDA를 자동 감지하여 
최적의 디바이스를 선택하고 플랫폼별 최적화 설정을 제공합니다.
"""

import os
import platform
import subprocess
import logging
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import torch

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    """디바이스 정보를 담는 데이터 클래스"""
    device_type: str  # 'cuda', 'mps', 'cpu'
    device_name: str
    device_index: int = 0
    memory_gb: Optional[float] = None
    compute_capability: Optional[str] = None
    
    def __str__(self) -> str:
        return f"{self.device_type}:{self.device_index} ({self.device_name})"


@dataclass 
class OptimizationConfig:
    """디바이스별 최적화 설정"""
    fp16: bool = False
    fp16_opt_level: str = "O1"
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    num_workers: int = 0
    pin_memory: bool = False
    mixed_precision: str = "no"  # 'no', 'fp16', 'bf16'
    gradient_checkpointing: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'fp16': self.fp16,
            'fp16_opt_level': self.fp16_opt_level,
            'per_device_train_batch_size': self.batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'dataloader_num_workers': self.num_workers,
            'dataloader_pin_memory': self.pin_memory,
            'mixed_precision': self.mixed_precision,
            'gradient_checkpointing': self.gradient_checkpointing
        }


def get_system_info() -> Dict[str, Any]:
    """
    시스템 정보를 수집합니다.
    
    Returns:
        시스템 정보 딕셔너리
    """
    info = {
        'platform': platform.system(),
        'platform_release': platform.release(),
        'platform_version': platform.version(),
        'architecture': platform.machine(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
    }
    
    # PyTorch 빌드 정보
    info['torch_cuda_available'] = torch.cuda.is_available()
    info['torch_cuda_version'] = torch.version.cuda if torch.cuda.is_available() else None
    info['torch_cudnn_version'] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
    
    # MPS 지원 (Mac M1/M2)
    info['torch_mps_available'] = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    
    # CPU 정보
    info['cpu_count'] = os.cpu_count()
    
    logger.info(f"시스템 정보: {info}")
    return info


def detect_cuda_devices() -> Dict[int, DeviceInfo]:
    """
    CUDA 디바이스를 감지합니다.
    
    Returns:
        디바이스 인덱스를 키로 하는 DeviceInfo 딕셔너리
    """
    devices = {}
    
    if not torch.cuda.is_available():
        return devices
    
    device_count = torch.cuda.device_count()
    logger.info(f"CUDA 디바이스 {device_count}개 감지됨")
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        
        # 메모리 크기 (GB)
        memory_gb = props.total_memory / (1024**3)
        
        # 컴퓨트 능력
        compute_capability = f"{props.major}.{props.minor}"
        
        device_info = DeviceInfo(
            device_type='cuda',
            device_name=props.name,
            device_index=i,
            memory_gb=memory_gb,
            compute_capability=compute_capability
        )
        
        devices[i] = device_info
        logger.info(f"  GPU {i}: {device_info}")
    
    return devices


def detect_mps_device() -> Optional[DeviceInfo]:
    """
    MPS (Metal Performance Shaders) 디바이스를 감지합니다.
    
    Returns:
        MPS 디바이스 정보 또는 None
    """
    if not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available():
        return None
    
    # macOS 버전 확인
    try:
        macos_version = platform.mac_ver()[0]
        logger.info(f"macOS 버전: {macos_version}")
    except:
        macos_version = "Unknown"
    
    # Apple Silicon 확인
    processor = platform.processor()
    if 'arm' in processor.lower():
        chip_name = "Apple Silicon (M1/M2)"
    else:
        chip_name = "Unknown"
    
    device_info = DeviceInfo(
        device_type='mps',
        device_name=f"{chip_name} - macOS {macos_version}",
        device_index=0
    )
    
    logger.info(f"MPS 디바이스 감지됨: {device_info}")
    return device_info


def get_optimal_device() -> Tuple[torch.device, DeviceInfo]:
    """
    사용 가능한 최적의 디바이스를 선택합니다.
    우선순위: CUDA > MPS > CPU
    
    Returns:
        (torch.device, DeviceInfo) 튜플
    """
    # CUDA 확인
    cuda_devices = detect_cuda_devices()
    if cuda_devices:
        # 메모리가 가장 큰 GPU 선택
        best_gpu = max(cuda_devices.values(), key=lambda d: d.memory_gb or 0)
        device = torch.device(f'cuda:{best_gpu.device_index}')
        logger.info(f"최적 디바이스로 CUDA 선택: {best_gpu}")
        return device, best_gpu
    
    # MPS 확인
    mps_device = detect_mps_device()
    if mps_device:
        device = torch.device('mps')
        logger.info(f"최적 디바이스로 MPS 선택: {mps_device}")
        return device, mps_device
    
    # CPU 사용
    cpu_info = DeviceInfo(
        device_type='cpu',
        device_name=platform.processor() or 'CPU',
        device_index=0
    )
    device = torch.device('cpu')
    logger.info(f"GPU를 사용할 수 없어 CPU 사용: {cpu_info}")
    return device, cpu_info


def setup_device_config(device_info: DeviceInfo, model_size: str = 'base') -> OptimizationConfig:
    """
    디바이스별 최적화 설정을 생성합니다.
    
    Args:
        device_info: 디바이스 정보
        model_size: 모델 크기 ('small', 'base', 'large')
        
    Returns:
        최적화 설정
    """
    config = OptimizationConfig()
    
    # 모델 크기별 기본 배치 크기
    base_batch_sizes = {
        'small': {'cuda': 16, 'mps': 8, 'cpu': 4},
        'base': {'cuda': 8, 'mps': 4, 'cpu': 2},
        'large': {'cuda': 4, 'mps': 2, 'cpu': 1}
    }
    
    base_batch_size = base_batch_sizes.get(model_size, base_batch_sizes['base'])
    
    if device_info.device_type == 'cuda':
        # CUDA 최적화 설정
        config.fp16 = True
        config.mixed_precision = 'fp16'
        config.batch_size = base_batch_size['cuda']
        config.num_workers = 4
        config.pin_memory = True
        
        # 메모리 크기에 따른 조정
        if device_info.memory_gb and device_info.memory_gb < 8:
            config.batch_size = max(1, config.batch_size // 2)
            config.gradient_checkpointing = True
        elif device_info.memory_gb and device_info.memory_gb >= 16:
            config.batch_size = min(32, config.batch_size * 2)
        
        # 컴퓨트 능력에 따른 조정
        if device_info.compute_capability:
            major, minor = map(int, device_info.compute_capability.split('.'))
            if major >= 7:  # Volta 이상
                config.fp16_opt_level = "O2"
            if major >= 8:  # Ampere 이상
                config.mixed_precision = 'bf16'  # BF16 지원
                
    elif device_info.device_type == 'mps':
        # MPS 최적화 설정
        config.fp16 = False  # MPS는 현재 fp16 지원이 제한적
        config.mixed_precision = 'no'
        config.batch_size = base_batch_size['mps']
        config.num_workers = 0  # MPS에서는 멀티프로세싱 이슈 있음
        config.pin_memory = False
        
        # MPS 특화 환경변수 설정
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
    else:  # CPU
        # CPU 최적화 설정
        config.fp16 = False
        config.mixed_precision = 'no'
        config.batch_size = base_batch_size['cpu']
        config.num_workers = min(2, os.cpu_count() or 1)
        config.pin_memory = False
        config.gradient_checkpointing = True  # 메모리 절약
    
    # 그래디언트 누적으로 유효 배치 크기 유지
    target_batch_size = base_batch_sizes['base']['cuda']
    if config.batch_size < target_batch_size:
        config.gradient_accumulation_steps = target_batch_size // config.batch_size
    
    logger.info(f"{device_info.device_type} 최적화 설정: {config}")
    return config


def print_device_summary():
    """디바이스 요약 정보를 출력합니다."""
    print("\n" + "="*60)
    print("디바이스 감지 및 설정 요약")
    print("="*60)
    
    # 시스템 정보
    sys_info = get_system_info()
    print(f"\n플랫폼: {sys_info['platform']} {sys_info['platform_release']}")
    print(f"아키텍처: {sys_info['architecture']}")
    print(f"PyTorch 버전: {sys_info['torch_version']}")
    
    # CUDA 정보
    if sys_info['torch_cuda_available']:
        print(f"\nCUDA 사용 가능: Yes")
        print(f"CUDA 버전: {sys_info['torch_cuda_version']}")
        cuda_devices = detect_cuda_devices()
        for device in cuda_devices.values():
            print(f"  - {device}")
    else:
        print(f"\nCUDA 사용 가능: No")
    
    # MPS 정보
    if sys_info['torch_mps_available']:
        print(f"\nMPS 사용 가능: Yes")
        mps_device = detect_mps_device()
        if mps_device:
            print(f"  - {mps_device}")
    else:
        print(f"\nMPS 사용 가능: No")
    
    # 최적 디바이스
    device, device_info = get_optimal_device()
    print(f"\n선택된 디바이스: {device}")
    
    # 최적화 설정
    for model_size in ['small', 'base', 'large']:
        config = setup_device_config(device_info, model_size)
        print(f"\n{model_size.upper()} 모델 최적화 설정:")
        print(f"  - 배치 크기: {config.batch_size}")
        print(f"  - Mixed Precision: {config.mixed_precision}")
        print(f"  - Gradient Accumulation: {config.gradient_accumulation_steps}")
        print(f"  - DataLoader Workers: {config.num_workers}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 디바이스 요약 출력
    print_device_summary()
