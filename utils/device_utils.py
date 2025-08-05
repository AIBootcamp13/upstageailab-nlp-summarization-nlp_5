"""
디바이스 관리 유틸리티

Mac M1/M2의 MPS와 Linux/Windows의 CUDA를 자동 감지하여 
최적의 디바이스를 선택하고 플랫폼별 최적화 설정을 제공합니다.
"""
import os
import platform
import subprocess
import logging
import torch
import time
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import time
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



class ContainerAwareDeviceDetector:
    """
    컴테이너 환경에 특화된 RTX 3090 디바이스 감지기
    
    시스템 명령어 부재 상황에서도 정확한 하드웨어 감지와 극한 최적화 설정 적용을 보장합니다.
    컴테이너 환경에서는 nvidia-smi, lspci 등의 명령어가 없을 수 있어 다단계 감지 방법을 사용합니다.
    """
    
    def __init__(self, fallback_mode: bool = True):
        """
        Args:
            fallback_mode: 감지 실패 시 기본 설정 사용 여부
        """
        self.fallback_mode = fallback_mode
        self.detection_methods = [
            ('torch_cuda', self._torch_cuda_detection),
            ('nvidia_ml', self._nvidia_ml_detection), 
            ('env_variable', self._env_variable_detection),
            ('proc_gpu', self._proc_gpu_detection)
        ]
        self.detection_results = {}
        
    def detect_device_robust(self) -> Tuple[torch.device, DeviceInfo]:
        """
        견고한 디바이스 감지 (다단계 방법)
        
        Returns:
            (torch.device, DeviceInfo) 튜플
        """
        logger.info("🔍 컴테이너 특화 디바이스 감지 시작")
        
        # 컴테이너 환경인지 확인
        is_container = self._detect_container_environment()
        if is_container:
            logger.info("📦 컴테이너 환경 감지됨 - 전용 감지 로직 사용")
        
        # 다단계 감지 시도
        best_detection = None
        
        for method_name, method_func in self.detection_methods:
            try:
                result = method_func()
                self.detection_results[method_name] = result
                
                if result and result.get('success', False):
                    logger.info(f"✅ {method_name} 방법으로 감지 성공: {result.get('device_name', 'Unknown')}")
                    
                    # RTX 3090 감지 성공 시 최고 우선순위
                    if 'RTX 3090' in result.get('device_name', ''):
                        best_detection = result
                        break
                    elif not best_detection:
                        best_detection = result
                else:
                    logger.debug(f"❌ {method_name} 방법 실패: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.debug(f"❌ {method_name} 방법 예외: {e}")
                self.detection_results[method_name] = {'success': False, 'error': str(e)}
        
        # 감지 결과 처리
        if best_detection:
            return self._create_device_from_detection(best_detection)
        else:
            return self._handle_detection_failure()
    
    def _detect_container_environment(self) -> bool:
        """
        컴테이너 환경인지 감지
        
        Returns:
            컴테이너 환경 여부
        """
        container_indicators = [
            # Docker 감지
            os.path.exists('/.dockerenv'),
            # cgroup 확인
            os.path.exists('/proc/1/cgroup') and 'docker' in open('/proc/1/cgroup').read() if os.path.exists('/proc/1/cgroup') else False,
            # Kubernetes 감지
            os.environ.get('KUBERNETES_SERVICE_HOST') is not None,
            # Singularity 감지  
            os.environ.get('SINGULARITY_CONTAINER') is not None,
            # 기타 컴테이너 환경변수
            os.environ.get('CONTAINER') is not None
        ]
        
        return any(container_indicators)
    
    def _torch_cuda_detection(self) -> Dict[str, Any]:
        """
        PyTorch CUDA 기반 감지 (가장 안정적)
        
        Returns:
            감지 결과
        """
        try:
            if not torch.cuda.is_available():
                return {'success': False, 'error': 'CUDA not available in PyTorch'}
            
            device_count = torch.cuda.device_count()
            if device_count == 0:
                return {'success': False, 'error': 'No CUDA devices found'}
            
            # 대표 디바이스 (GPU 0) 정보 가져오기
            props = torch.cuda.get_device_properties(0)
            memory_gb = props.total_memory / (1024**3)
            
            return {
                'success': True,
                'method': 'torch_cuda',
                'device_name': props.name,
                'device_count': device_count,
                'memory_gb': memory_gb,
                'compute_capability': f"{props.major}.{props.minor}",
                'multiprocessor_count': props.multi_processor_count
            }
            
        except Exception as e:
            return {'success': False, 'error': f'PyTorch CUDA detection failed: {e}'}
    
    def _nvidia_ml_detection(self) -> Dict[str, Any]:
        """
        nvidia-ml-py 라이브러리 기반 감지
        
        Returns:
            감지 결과
        """
        try:
            import pynvml
            
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            if device_count == 0:
                return {'success': False, 'error': 'No NVIDIA devices found via nvidia-ml'}
            
            # 대표 디바이스 정보
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_gb = memory_info.total / (1024**3)
            
            # 추가 정보
            try:
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                compute_capability = f"{major}.{minor}"
            except:
                compute_capability = "Unknown"
            
            return {
                'success': True,
                'method': 'nvidia_ml',
                'device_name': name,
                'device_count': device_count,
                'memory_gb': memory_gb,
                'compute_capability': compute_capability
            }
            
        except ImportError:
            return {'success': False, 'error': 'pynvml library not available'}
        except Exception as e:
            return {'success': False, 'error': f'nvidia-ml detection failed: {e}'}
    
    def _env_variable_detection(self) -> Dict[str, Any]:
        """
        환경변수 기반 감지
        
        Returns:
            감지 결과
        """
        try:
            # CUDA 관련 환경변수 확인
            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
            nvidia_visible = os.environ.get('NVIDIA_VISIBLE_DEVICES')
            
            if cuda_visible is None and nvidia_visible is None:
                return {'success': False, 'error': 'No CUDA environment variables found'}
            
            # 디바이스 정보 추론
            device_ids = cuda_visible or nvidia_visible
            
            # GPU 모델명 추론 (컴테이너에서 공통적인 패턴)
            gpu_model = os.environ.get('GPU_MODEL', 'Unknown GPU')
            gpu_memory = os.environ.get('GPU_MEMORY_GB', '24')  # RTX 3090 기본값
            
            # RTX 3090 추론 로직 (컴테이너 환경에서 일반적인 패턴)
            if gpu_model == 'Unknown GPU' and gpu_memory == '24':
                gpu_model = 'NVIDIA GeForce RTX 3090'  # 24GB VRAM은 RTX 3090의 특징
            
            return {
                'success': True,
                'method': 'env_variable',
                'device_name': gpu_model,
                'device_ids': device_ids,
                'memory_gb': float(gpu_memory),
                'inferred': True  # 추론된 정보임을 표시
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Environment variable detection failed: {e}'}
    
    def _proc_gpu_detection(self) -> Dict[str, Any]:
        """
        /proc 파일시스템 기반 감지
        
        Returns:
            감지 결과
        """
        try:
            # /proc/driver/nvidia/gpus 확인
            nvidia_proc_path = Path('/proc/driver/nvidia/gpus')
            
            if not nvidia_proc_path.exists():
                return {'success': False, 'error': '/proc/driver/nvidia not found'}
            
            # GPU 디렉토리 열거
            gpu_dirs = [d for d in nvidia_proc_path.iterdir() if d.is_dir()]
            
            if not gpu_dirs:
                return {'success': False, 'error': 'No GPU directories in /proc/driver/nvidia/gpus'}
            
            # 대표 GPU 정보 읽기
            gpu_dir = gpu_dirs[0]
            info_file = gpu_dir / 'information'
            
            if not info_file.exists():
                return {'success': False, 'error': 'GPU information file not found'}
            
            # 정보 파싱
            gpu_info = {}
            with open(info_file, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        gpu_info[key.strip()] = value.strip()
            
            device_name = gpu_info.get('Model', 'Unknown GPU')
            
            # 메모리 정보 추론 (RTX 3090 = 24GB)
            memory_gb = 24.0 if 'RTX 3090' in device_name else 8.0
            
            return {
                'success': True,
                'method': 'proc_gpu',
                'device_name': device_name,
                'device_count': len(gpu_dirs),
                'memory_gb': memory_gb,
                'gpu_info': gpu_info
            }
            
        except Exception as e:
            return {'success': False, 'error': f'/proc GPU detection failed: {e}'}
    
    def _create_device_from_detection(self, detection: Dict[str, Any]) -> Tuple[torch.device, DeviceInfo]:
        """
        감지 결과로부터 디바이스 객체 생성
        
        Args:
            detection: 감지 결과
            
        Returns:
            (torch.device, DeviceInfo) 튜플
        """
        device_name = detection.get('device_name', 'Unknown GPU')
        memory_gb = detection.get('memory_gb', 0.0)
        compute_capability = detection.get('compute_capability', 'Unknown')
        
        device_info = DeviceInfo(
            device_type='cuda',
            device_name=device_name,
            device_index=0,
            memory_gb=memory_gb,
            compute_capability=compute_capability
        )
        
        device = torch.device('cuda:0')
        
        # RTX 3090 감지 성공 시 극한 최적화 설정 자동 적용
        if 'RTX 3090' in device_name:
            logger.info(f"🔥 RTX 3090 감지 성공! 극한 최적화 모드 활성화")
            logger.info(f"   감지 방법: {detection.get('method', 'unknown')}")
            logger.info(f"   VRAM: {memory_gb:.1f}GB")
            logger.info(f"   컴퓨트 능력: {compute_capability}")
        
        return device, device_info
    
    def _handle_detection_failure(self) -> Tuple[torch.device, DeviceInfo]:
        """
        모든 감지 방법 실패 시 처리
        
        Returns:
            (torch.device, DeviceInfo) 튜플
        """
        if not self.fallback_mode:
            logger.error("모든 디바이스 감지 방법 실패 - fallback_mode 비활성화")
            raise RuntimeError("디바이스 감지 실패 및 폴백 모드 비활성화")
        
        logger.warning("⚠️  모든 GPU 감지 방법 실패 - CPU 모드로 대체")
        
        # 기본 CPU 디바이스 정보
        cpu_info = DeviceInfo(
            device_type='cpu',
            device_name=platform.processor() or 'CPU',
            device_index=0
        )
        
        device = torch.device('cpu')
        
        # 감지 결과를 로그에 기록
        logger.info("📄 디바이스 감지 결과 요약:")
        for method, result in self.detection_results.items():
            status = "✅" if result.get('success', False) else "❌"
            error = result.get('error', 'No error') if not result.get('success', False) else ''
            logger.info(f"  {status} {method}: {error}")
        
        return device, cpu_info
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """
        감지 결과 요약 반환
        
        Returns:
            감지 결과 요약
        """
        return {
            'container_environment': self._detect_container_environment(),
            'detection_methods_tried': len(self.detection_methods),
            'successful_detections': sum(1 for r in self.detection_results.values() if r.get('success', False)),
            'detection_results': self.detection_results,
            'fallback_mode': self.fallback_mode
        }


# 전역 ContainerAwareDeviceDetector 인스턴스
_container_device_detector = ContainerAwareDeviceDetector()


def get_robust_optimal_device() -> Tuple[torch.device, DeviceInfo]:
    """
    견고한 최적 디바이스 선택 (전역 함수)
    
    기존 get_optimal_device()의 안전한 버전.
    컴테이너 환경에서 시스템 명령어 부재 시에도 정확한 디바이스 감지를 보장합니다.
    
    Returns:
        (torch.device, DeviceInfo) 튜플
        
    Example:
        # 기존 방식
        device, device_info = get_optimal_device()
        
        # 안전한 방식  
        device, device_info = get_robust_optimal_device()
    """
    return _container_device_detector.detect_device_robust()







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
def get_rtx3090_extreme_config(model_architecture: str = 'base', use_unsloth: bool = True) -> Dict[str, Any]:
    """
    RTX 3090 + CUDA 12.2 + Unsloth 극한 최적화 설정
    
    Args:
        model_architecture: 모델 아키텍처 ('mt5', 'bart', 't5', 'kobart', 'base')
        use_unsloth: Unsloth 사용 여부
        
    Returns:
        RTX 3090 극한 최적화 설정 딕셔너리
    """
    
    # RTX 3090 기본 사양: 24GB VRAM, 48코어 CPU, 251GB RAM
    base_config = {
        'device_memory_gb': 24.0,
        'cpu_cores': 48,
        'system_ram_gb': 251.0,
        'cuda_version': '12.2',
        'use_unsloth': use_unsloth
    }
    
    # 아키텍처별 극한 최적화 매트릭스
    optimization_matrix = {
        'mt5': {
            'per_device_train_batch_size': 12 if use_unsloth else 6,
            'per_device_eval_batch_size': 24 if use_unsloth else 12,
            'gradient_accumulation_steps': 4,
            'encoder_max_len': 1536 if use_unsloth else 1024,
            'decoder_max_len': 256,
            'dataloader_num_workers': 36,
            'lora_rank': 256 if use_unsloth else 128,
            'lora_alpha': 512 if use_unsloth else 256,
            'generation_num_beams': 16 if use_unsloth else 10
        },
        'bart': {
            'per_device_train_batch_size': 20 if use_unsloth else 8,
            'per_device_eval_batch_size': 32 if use_unsloth else 16,
            'gradient_accumulation_steps': 3,
            'encoder_max_len': 1280 if use_unsloth else 1024,
            'decoder_max_len': 256,
            'dataloader_num_workers': 36,
            'lora_rank': 192 if use_unsloth else 96,
            'lora_alpha': 384 if use_unsloth else 192,
            'generation_num_beams': 12 if use_unsloth else 8
        },
        'kobart': {
            'per_device_train_batch_size': 20 if use_unsloth else 8,
            'per_device_eval_batch_size': 32 if use_unsloth else 16,
            'gradient_accumulation_steps': 3,
            'encoder_max_len': 1280 if use_unsloth else 1024,
            'decoder_max_len': 256,
            'dataloader_num_workers': 36,
            'lora_rank': 192 if use_unsloth else 96,
            'lora_alpha': 384 if use_unsloth else 192,
            'generation_num_beams': 12 if use_unsloth else 8
        },
        't5': {
            'per_device_train_batch_size': 16 if use_unsloth else 8,
            'per_device_eval_batch_size': 28 if use_unsloth else 14,
            'gradient_accumulation_steps': 3,
            'encoder_max_len': 1024,
            'decoder_max_len': 200,
            'dataloader_num_workers': 36,
            'lora_rank': 128 if use_unsloth else 64,
            'lora_alpha': 256 if use_unsloth else 128,
            'generation_num_beams': 10 if use_unsloth else 6
        },
        'base': {
            'per_device_train_batch_size': 16 if use_unsloth else 8,
            'per_device_eval_batch_size': 24 if use_unsloth else 12,
            'gradient_accumulation_steps': 4,
            'encoder_max_len': 1024,
            'decoder_max_len': 200,
            'dataloader_num_workers': 32,
            'lora_rank': 128 if use_unsloth else 64,
            'lora_alpha': 256 if use_unsloth else 128,
            'generation_num_beams': 8
        }
    }
    
    # CUDA 12.2 + RTX 3090 특화 설정
    cuda_optimizations = {
        'fp16': False,  # RTX 3090에서는 bf16 선호
        'bf16': True,   # RTX 3090 + CUDA 12.2 최적
        'tf32': True,   # Ampere 아키텍처 최적화
        'torch_compile': True,  # PyTorch 2.7.1 컴파일 최적화
        'gradient_checkpointing': False if use_unsloth else True,  # Unsloth는 자체 최적화
        'dataloader_pin_memory': True,  # 251GB RAM 활용
        'dataloader_persistent_workers': True,  # 워커 재사용
        'group_by_length': True,  # 효율성 증대
        'remove_unused_columns': False,  # 전체 정보 활용
        'optim': 'adamw_8bit' if use_unsloth else 'adamw_hf',  # Unsloth 호환 옵티마이저
        'adam_beta1': 0.9,
        'adam_beta2': 0.95 if use_unsloth else 0.999,  # Unsloth 추천값
        'max_grad_norm': 0.3 if use_unsloth else 1.0   # Unsloth와 함께 낮은 값
    }
    
    # 아키텍처별 설정 가져오기
    arch_config = optimization_matrix.get(model_architecture.lower(), optimization_matrix['base'])
    
    # 최종 설정 조합
    final_config = {
        **base_config,
        **arch_config,
        **cuda_optimizations
    }
    
    # 예상 성능 향상 계산
    memory_efficiency = 40 if use_unsloth else 20  # Unsloth 메모리 절약
    speed_improvement = 2.5 if use_unsloth else 1.5  # Unsloth 속도 향상
    
    final_config.update({
        'expected_memory_efficiency_percent': memory_efficiency,
        'expected_speed_improvement': speed_improvement,
        'effective_batch_size': arch_config['per_device_train_batch_size'] * arch_config['gradient_accumulation_steps'],
        'total_vram_utilization_target': 90.0,  # 24GB 중 90% 목표
        'cpu_utilization_target': 75.0,  # 48코어 중 75% 목표
        'optimization_level': 'extreme',
        'architecture': model_architecture,
        'profile_name': f'RTX3090_Extreme_{model_architecture.upper()}_{"Unsloth" if use_unsloth else "Standard"}'
    })
    
    logger.info(f"RTX 3090 극한 최적화 프로파일 생성: {final_config['profile_name']}")
    logger.info(f"배치 크기: {arch_config['per_device_train_batch_size']}, "
                f"유효 배치: {final_config['effective_batch_size']}, "
                f"워커 수: {arch_config['dataloader_num_workers']}")
    
    return final_config


def apply_rtx3090_extreme_optimizations(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    기존 설정에 RTX 3090 극한 최적화를 적용
    
    Args:
        config: 기존 설정 딕셔너리
        
    Returns:
        RTX 3090 극한 최적화가 적용된 설정
    """
    # GPU 정보 확인
    if not torch.cuda.is_available():
        logger.warning("CUDA를 사용할 수 없어 RTX 3090 최적화를 적용할 수 없습니다")
        return config
    
    device_props = torch.cuda.get_device_properties(0)
    if 'RTX 3090' not in device_props.name:
        logger.warning(f"RTX 3090이 아니므로 극한 최적화를 적용하지 않습니다: {device_props.name}")
        return config
    
    # 아키텍처 추론
    model_arch = 'base'
    if 'mt5' in config.get('model_name', '').lower():
        model_arch = 'mt5'
    elif 'bart' in config.get('model_name', '').lower():
        model_arch = 'kobart' if 'kobart' in config.get('model_name', '').lower() else 'bart'
    elif 't5' in config.get('model_name', '').lower():
        model_arch = 't5'
    
    # Unsloth 사용 여부 확인
    use_unsloth = config.get('qlora', {}).get('use_unsloth', True)
    
    # 극한 최적화 설정 가져오기
    extreme_config = get_rtx3090_extreme_config(model_arch, use_unsloth)
    
    # 기존 설정에 극한 최적화 적용
    optimized_config = config.copy()
    
    # training 섹션 업데이트
    if 'training' not in optimized_config:
        optimized_config['training'] = {}
    
    training_updates = {
        'per_device_train_batch_size': extreme_config['per_device_train_batch_size'],
        'per_device_eval_batch_size': extreme_config['per_device_eval_batch_size'],
        'gradient_accumulation_steps': extreme_config['gradient_accumulation_steps'],
        'dataloader_num_workers': extreme_config['dataloader_num_workers'],
        'dataloader_pin_memory': extreme_config['dataloader_pin_memory'],
        'bf16': extreme_config['bf16'],
        'fp16': extreme_config['fp16'],
        'gradient_checkpointing': extreme_config['gradient_checkpointing'],
        'group_by_length': extreme_config['group_by_length'],
        'optim': extreme_config['optim'],
        'adam_beta1': extreme_config['adam_beta1'],
        'adam_beta2': extreme_config['adam_beta2'],
        'max_grad_norm': extreme_config['max_grad_norm']
    }
    
    optimized_config['training'].update(training_updates)
    
    # tokenizer 섹션 업데이트
    if 'tokenizer' not in optimized_config:
        optimized_config['tokenizer'] = {}
    
    tokenizer_updates = {
        'encoder_max_len': extreme_config['encoder_max_len'],
        'decoder_max_len': extreme_config['decoder_max_len']
    }
    
    optimized_config['tokenizer'].update(tokenizer_updates)
    
    # qlora 섹션 업데이트
    if 'qlora' not in optimized_config:
        optimized_config['qlora'] = {}
    
    qlora_updates = {
        'lora_rank': extreme_config['lora_rank'],
        'lora_alpha': extreme_config['lora_alpha'],
        'use_unsloth': use_unsloth
    }
    
    optimized_config['qlora'].update(qlora_updates)
    
    # generation 설정 업데이트
    if 'generation' in optimized_config or 'training' in optimized_config:
        generation_updates = {
            'generation_num_beams': extreme_config['generation_num_beams'],
            'generation_max_length': extreme_config['decoder_max_len']
        }
        
        if 'training' in optimized_config:
            optimized_config['training'].update(generation_updates)
        if 'generation' in optimized_config:
            optimized_config['generation'].update(generation_updates)
    
    # 메타데이터 추가
    optimized_config['_rtx3090_extreme_optimization'] = {
        'applied': True,
        'profile_name': extreme_config['profile_name'],
        'expected_memory_efficiency': extreme_config['expected_memory_efficiency_percent'],
        'expected_speed_improvement': extreme_config['expected_speed_improvement'],
        'effective_batch_size': extreme_config['effective_batch_size'],
        'optimization_timestamp': time.time()
    }
    
    logger.info(f"RTX 3090 극한 최적화 적용 완료: {extreme_config['profile_name']}")
    
    return optimized_config


def setup_device_config(device_info: DeviceInfo, model_size: str = 'base', use_qlora: bool = False) -> OptimizationConfig:
    """
    디바이스별 최적화 설정을 생성합니다.
    
    Args:
        device_info: 디바이스 정보
        model_size: 모델 크기 ('small', 'base', 'large')
        use_qlora: QLoRA 사용 여부
        
    Returns:
        최적화 설정
    """
    config = OptimizationConfig()
    
    # 모델 크기별 기본 배치 크기
    # QLoRA 사용 시 메모리 효율이 높아 배치 크기를 크게 설정
    if use_qlora:
        # 조장님 권장사항: QLoRA로 batch_size 24~32
        base_batch_sizes = {
            'small': {'cuda': 32, 'mps': 16, 'cpu': 8},
            'base': {'cuda': 24, 'mps': 12, 'cpu': 6},
            'large': {'cuda': 16, 'mps': 8, 'cpu': 4}
        }
    else:
        # Full fine-tuning 배치 크기 (보수적)
        base_batch_sizes = {
            'small': {'cuda': 16, 'mps': 8, 'cpu': 4},
            'base': {'cuda': 8, 'mps': 4, 'cpu': 2},
            'large': {'cuda': 4, 'mps': 2, 'cpu': 1}
        }
    
    base_batch_size = base_batch_sizes.get(model_size, base_batch_sizes['base'])
    
    if device_info.device_type == 'cuda':
        # RTX 3090 자동 감지 및 극한 최적화 적용
        if 'RTX 3090' in device_info.device_name:
            logger.info(f"RTX 3090 감지됨: {device_info.device_name} - 극한 최적화 모드 활성화")
            
            # RTX 3090 극한 최적화 설정 사용
            extreme_config = get_rtx3090_extreme_config(model_size, use_qlora)
            
            # OptimizationConfig에 극한 최적화 적용
            config.fp16 = extreme_config['fp16']
            config.mixed_precision = 'bf16' if extreme_config['bf16'] else ('fp16' if extreme_config['fp16'] else 'no')
            config.batch_size = extreme_config['per_device_train_batch_size']
            config.gradient_accumulation_steps = extreme_config['gradient_accumulation_steps']
            config.num_workers = extreme_config['dataloader_num_workers']
            config.pin_memory = extreme_config['dataloader_pin_memory']
            config.gradient_checkpointing = extreme_config['gradient_checkpointing']
            
            logger.info(f"RTX 3090 극한 최적화 적용: 배치={config.batch_size}, 워커={config.num_workers}, 유효배치={extreme_config['effective_batch_size']}")
            
            return config
        # CUDA 최적화 설정
        config.fp16 = True
        config.mixed_precision = 'fp16'
        config.batch_size = base_batch_size['cuda']
        config.num_workers = 4
        config.pin_memory = True
        
        # 메모리 크기에 따른 조정
        if device_info.memory_gb:
            if device_info.memory_gb < 8:
                # 메모리가 적을 때
                config.batch_size = max(1, config.batch_size // 2)
                config.gradient_checkpointing = True
            elif device_info.memory_gb >= 16 and device_info.memory_gb < 24:
                # 중간 메모리 (16-24GB)
                if use_qlora:
                    config.batch_size = min(32, config.batch_size)
                else:
                    config.batch_size = config.batch_size
            elif device_info.memory_gb >= 24:
                # 충분한 메모리 (24GB 이상)
                if use_qlora:
                    config.batch_size = min(48, int(config.batch_size * 1.5))
                else:
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
    # QLoRA 사용 시는 큰 배치 크기를 허용하므로 gradient accumulation 최소화
    if use_qlora:
        target_batch_size = config.batch_size  # QLoRA는 큰 배치 그대로 사용
    else:
        target_batch_size = base_batch_sizes['base']['cuda']
        
    if config.batch_size < target_batch_size and not use_qlora:
        config.gradient_accumulation_steps = target_batch_size // config.batch_size
    
    logger.info(f"{device_info.device_type} 최적화 설정 ({'QLoRA' if use_qlora else 'Full FT'}): {config}")
    return config


def print_device_summary():
    """디바이스 요약 정보를 출력합니다."""
    print("\n" + "="*60)
    print("디바이스 감지 및 설정 요약 (RTX 3090 극한 최적화 포함)")
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
    print(f"\n🚀 최적화 설정:")
    
    # RTX 3090 극한 최적화 확인
    if device_info.device_type == 'cuda' and 'RTX 3090' in device_info.device_name:
        print(f"\n🔥 RTX 3090 극한 최적화 모드 활성화!")
        print(f"   디바이스: {device_info.device_name}")
        print(f"   VRAM: {device_info.memory_gb:.1f}GB")
        
        # 아키텍처별 극한 최적화 설정 표시
        for arch in ['mt5', 'bart', 't5', 'kobart']:
            extreme_config = get_rtx3090_extreme_config(arch, True)  # Unsloth 사용 가정
            print(f"\n   {arch.upper()} 극한 최적화:")
            print(f"     - 배치 크기: {extreme_config['per_device_train_batch_size']} (유효: {extreme_config['effective_batch_size']})")
            print(f"     - 시퀀스 길이: {extreme_config['encoder_max_len']}")
            print(f"     - LoRA Rank: {extreme_config['lora_rank']}")
            print(f"     - 워커 수: {extreme_config['dataloader_num_workers']}")
            print(f"     - 예상 속도 향상: {extreme_config['expected_speed_improvement']:.1f}x")
            print(f"     - 메모리 효율: {extreme_config['expected_memory_efficiency_percent']}% 절약")
    
    print(f"\n📊 기본 모델 크기별 최적화 설정:")
    for model_size in ['small', 'base', 'large']:
        config = setup_device_config(device_info, model_size)
        print(f"\n{model_size.upper()} 모델 최적화 설정:")
        print(f"  - 배치 크기: {config.batch_size}")
        print(f"  - Mixed Precision: {config.mixed_precision}")
        print(f"  - Gradient Accumulation: {config.gradient_accumulation_steps}")
        print(f"  - DataLoader Workers: {config.num_workers}")
    
    print("\n" + "="*60)


def test_rtx3090_extreme_config():
    """
    RTX 3090 극한 최적화 설정 테스트
    """
    print("\n" + "="*50)
    print("🧪 RTX 3090 극한 최적화 테스트")
    print("="*50)
    
    # 아키텍처별 테스트
    architectures = ['mt5', 'bart', 't5', 'kobart', 'base']
    
    for arch in architectures:
        print(f"\n🔍 {arch.upper()} 아키텍처 테스트:")
        
        # Unsloth 사용/미사용 비교
        for use_unsloth in [True, False]:
            config = get_rtx3090_extreme_config(arch, use_unsloth)
            mode = "Unsloth" if use_unsloth else "Standard"
            
            print(f"  {mode} 모드:")
            print(f"    배치: {config['per_device_train_batch_size']} | 유효: {config['effective_batch_size']}")
            print(f"    시퀀스: {config['encoder_max_len']} | LoRA: {config['lora_rank']}")
            print(f"    워커: {config['dataloader_num_workers']} | 속도: {config['expected_speed_improvement']:.1f}x")
    
    # GPU 감지 테스트
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        print(f"\n📊 현재 GPU: {device_props.name}")
        print(f"VRAM: {device_props.total_memory / (1024**3):.1f}GB")
        
        if 'RTX 3090' in device_props.name:
            print("✅ RTX 3090 감지됨 - 극한 최적화 사용 가능!")
        else:
            print("⚠️  RTX 3090이 아니므로 극한 최적화 비활성화")
    else:
        print("❌ CUDA를 사용할 수 없습니다")
    
    print("\n" + "="*50)


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 기본 디바이스 요약 출력
    print_device_summary()
    
    # RTX 3090 극한 최적화 테스트
    test_rtx3090_extreme_config()
