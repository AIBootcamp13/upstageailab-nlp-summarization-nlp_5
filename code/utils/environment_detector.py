"""
환경 자동 감지 및 Unsloth 활성화 시스템

이 모듈은 현재 실행 환경을 자동으로 감지하고,
Ubuntu + CUDA 환경에서는 자동으로 Unsloth를 활성화합니다.
"""

import platform
import subprocess
import os
import logging
from typing import Dict, Any, Tuple
import torch

logger = logging.getLogger(__name__)

class EnvironmentDetector:
    """환경 자동 감지 및 최적화 설정 클래스"""
    
    def __init__(self):
        self.os_type = platform.system()
        self.os_release = platform.release()
        self.machine = platform.machine()
        self.python_version = platform.python_version()
        
    def detect_environment(self) -> Dict[str, Any]:
        """현재 환경을 자동 감지합니다."""
        env_info = {
            'os': self.os_type,
            'os_release': self.os_release,
            'machine': self.machine,
            'python_version': self.python_version,
            'is_cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'is_ubuntu': self._is_ubuntu(),
            'is_macos': self._is_macos(),
            'is_windows': self._is_windows(),
            'cuda_version': self._get_cuda_version(),
            'gpu_info': self._get_gpu_info(),
            'available_memory_gb': self._get_available_memory_gb(),
            'cpu_count': os.cpu_count(),
        }
        
        # Unsloth 사용 가능 여부 판단
        env_info['unsloth_recommended'] = self._should_use_unsloth(env_info)
        env_info['unsloth_available'] = self._check_unsloth_availability()
        
        return env_info
    
    def _is_ubuntu(self) -> bool:
        """Ubuntu 환경인지 확인"""
        if self.os_type != 'Linux':
            return False
        
        try:
            with open('/etc/os-release', 'r') as f:
                content = f.read().lower()
                return 'ubuntu' in content
        except FileNotFoundError:
            return False
    
    def _is_macos(self) -> bool:
        """macOS 환경인지 확인"""
        return self.os_type == 'Darwin'
    
    def _is_windows(self) -> bool:
        """Windows 환경인지 확인"""
        return self.os_type == 'Windows'
    
    def _get_cuda_version(self) -> str:
        """CUDA 버전 확인"""
        if not torch.cuda.is_available():
            return "N/A"
        
        try:
            result = subprocess.run(['nvidia-smi'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'CUDA Version:' in line:
                        return line.split('CUDA Version:')[1].strip().split()[0]
            
            # PyTorch에서 CUDA 버전 확인
            return torch.version.cuda or "Unknown"
        except Exception as e:
            logger.warning(f"CUDA 버전 확인 실패: {e}")
            return "Unknown"
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """GPU 정보 수집"""
        if not torch.cuda.is_available():
            return {"count": 0, "devices": []}
        
        gpu_info = {
            "count": torch.cuda.device_count(),
            "devices": []
        }
        
        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            gpu_info["devices"].append({
                "id": i,
                "name": device_props.name,
                "memory_gb": device_props.total_memory / (1024**3),
                "compute_capability": f"{device_props.major}.{device_props.minor}"
            })
        
        return gpu_info
    
    def _get_available_memory_gb(self) -> float:
        """사용 가능한 시스템 메모리(GB) 확인"""
        try:
            if self._is_ubuntu() or self.os_type == 'Linux':
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if 'MemTotal:' in line:
                            # kB to GB 변환
                            memory_kb = int(line.split()[1])
                            return memory_kb / (1024**2)
            elif self._is_macos():
                result = subprocess.run(['sysctl', 'hw.memsize'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    memory_bytes = int(result.stdout.split()[1])
                    return memory_bytes / (1024**3)
        except Exception as e:
            logger.warning(f"메모리 정보 확인 실패: {e}")
        
        return 0.0
    
    def _should_use_unsloth(self, env_info: Dict[str, Any]) -> bool:
        """Unsloth 사용 권장 여부 판단"""
        # 기본 조건: Ubuntu + CUDA
        if not (env_info['is_ubuntu'] and env_info['is_cuda_available']):
            return False
        
        # GPU 메모리 확인 (최소 6GB 권장)
        if env_info['gpu_info']['count'] > 0:
            max_gpu_memory = max(
                device['memory_gb'] for device in env_info['gpu_info']['devices']
            )
            if max_gpu_memory < 6.0:
                logger.warning(f"GPU 메모리가 부족합니다: {max_gpu_memory:.1f}GB < 6GB")
                return False
        
        # CUDA 버전 확인 (11.8+ 권장)
        cuda_version = env_info['cuda_version']
        if cuda_version not in ['Unknown', 'N/A']:
            try:
                major, minor = map(int, cuda_version.split('.')[:2])
                if major < 11 or (major == 11 and minor < 8):
                    logger.warning(f"CUDA 버전이 낮습니다: {cuda_version} < 11.8")
                    return False
            except ValueError:
                logger.warning(f"CUDA 버전 파싱 실패: {cuda_version}")
        
        return True
    
    def _check_unsloth_availability(self) -> bool:
        """Unsloth 패키지 설치 여부 확인"""
        try:
            import unsloth
            return True
        except ImportError:
            return False
    
    def get_recommended_config(self) -> Dict[str, Any]:
        """환경에 맞는 권장 설정 반환"""
        env_info = self.detect_environment()
        
        config = {
            'use_unsloth': False,
            'use_qlora': True,
            'fp16': True,
            'bf16': False,
            'gradient_checkpointing': True,
            'dataloader_num_workers': min(4, env_info['cpu_count'] // 2),
            'recommended_batch_size': 4,
        }
        
        # Ubuntu + CUDA 환경에서 Unsloth 자동 활성화
        if env_info['unsloth_recommended']:
            config['use_unsloth'] = True
            config['gradient_checkpointing'] = False  # Unsloth는 자체 최적화
            
            # GPU 메모리에 따른 배치 크기 조정
            if env_info['gpu_info']['count'] > 0:
                max_gpu_memory = max(
                    device['memory_gb'] for device in env_info['gpu_info']['devices']
                )
                
                if max_gpu_memory >= 24:  # RTX 3090/4090급
                    config['recommended_batch_size'] = 12
                    config['dataloader_num_workers'] = min(8, env_info['cpu_count'] // 2)
                    config['bf16'] = True
                    config['fp16'] = False
                elif max_gpu_memory >= 16:  # RTX 4080급
                    config['recommended_batch_size'] = 8
                    config['dataloader_num_workers'] = 6
                elif max_gpu_memory >= 12:  # RTX 4070급
                    config['recommended_batch_size'] = 6
                elif max_gpu_memory >= 8:   # RTX 4060급
                    config['recommended_batch_size'] = 4
        
        # macOS에서는 특별 설정
        elif env_info['is_macos']:
            config['use_unsloth'] = False
            config['fp16'] = False  # macOS MPS는 fp16 이슈
            config['bf16'] = False
            config['dataloader_num_workers'] = 0  # macOS 멀티프로세싱 이슈
            config['recommended_batch_size'] = 2
        
        # Windows에서는 보수적 설정
        elif env_info['is_windows']:
            config['dataloader_num_workers'] = 0  # Windows 멀티프로세싱 이슈
            config['recommended_batch_size'] = 2
        
        return config
    
    def print_environment_summary(self):
        """환경 정보 요약 출력"""
        env_info = self.detect_environment()
        config = self.get_recommended_config()
        
        print("🔍 환경 자동 감지 결과")
        print("=" * 50)
        print(f"OS: {env_info['os']} ({env_info['os_release']})")
        print(f"Architecture: {env_info['machine']}")
        print(f"Python: {env_info['python_version']}")
        print(f"CPU Cores: {env_info['cpu_count']}")
        print(f"System Memory: {env_info['available_memory_gb']:.1f}GB")
        
        print(f"\n🎮 GPU 정보")
        if env_info['is_cuda_available']:
            print(f"CUDA: Available (v{env_info['cuda_version']})")
            print(f"GPU Count: {env_info['gpu_info']['count']}")
            for device in env_info['gpu_info']['devices']:
                print(f"  - {device['name']}: {device['memory_gb']:.1f}GB")
        else:
            print("CUDA: Not Available")
        
        print(f"\n⚡ Unsloth 지원")
        print(f"권장 여부: {'✅ 예' if env_info['unsloth_recommended'] else '❌ 아니오'}")
        print(f"설치 상태: {'✅ 설치됨' if env_info['unsloth_available'] else '❌ 미설치'}")
        
        print(f"\n🚀 권장 설정")
        print(f"use_unsloth: {config['use_unsloth']}")
        print(f"recommended_batch_size: {config['recommended_batch_size']}")
        print(f"fp16: {config['fp16']}, bf16: {config['bf16']}")
        print(f"dataloader_num_workers: {config['dataloader_num_workers']}")
        print("=" * 50)


def get_auto_config() -> Dict[str, Any]:
    """환경에 맞는 자동 설정 반환 (전역 함수)"""
    detector = EnvironmentDetector()
    return detector.get_recommended_config()


def should_use_unsloth() -> bool:
    """Unsloth 사용 여부 자동 판단 (전역 함수)"""
    detector = EnvironmentDetector()
    env_info = detector.detect_environment()
    return env_info['unsloth_recommended'] and env_info['unsloth_available']


if __name__ == "__main__":
    # 테스트 실행
    detector = EnvironmentDetector()
    detector.print_environment_summary()
