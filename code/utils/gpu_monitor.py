"""
GPU 메모리 모니터링 유틸리티

학습 중 GPU 메모리 사용률을 추적하고 동적으로 배치 크기를 조정합니다.
"""

import torch
import logging
from typing import Dict, Any, Optional, Tuple
import wandb
import time

logger = logging.getLogger(__name__)


class GPUMemoryMonitor:
    """GPU 메모리 사용률 모니터링 및 최적화"""
    
    def __init__(self, device: torch.device, log_to_wandb: bool = True):
        """
        Args:
            device: 모니터링할 디바이스
            log_to_wandb: WandB에 로깅 여부
        """
        self.device = device
        self.log_to_wandb = log_to_wandb
        self.is_cuda = device.type == 'cuda'
        self.memory_history = []
        
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(device)
    
    def get_memory_stats(self) -> Dict[str, float]:
        """현재 GPU 메모리 사용 통계 반환"""
        if not self.is_cuda:
            return {}
        
        # 메모리 통계 수집
        allocated = torch.cuda.memory_allocated(self.device) / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved(self.device) / (1024**3)  # GB
        max_allocated = torch.cuda.max_memory_allocated(self.device) / (1024**3)  # GB
        
        # 전체 메모리 크기
        total_memory = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)  # GB
        
        # 사용률 계산
        utilization = (allocated / total_memory) * 100 if total_memory > 0 else 0
        max_utilization = (max_allocated / total_memory) * 100 if total_memory > 0 else 0
        
        stats = {
            'gpu_memory_allocated_gb': allocated,
            'gpu_memory_reserved_gb': reserved,
            'gpu_memory_max_allocated_gb': max_allocated,
            'gpu_memory_total_gb': total_memory,
            'gpu_memory_utilization_percent': utilization,
            'gpu_memory_max_utilization_percent': max_utilization
        }
        
        # 이력 저장
        self.memory_history.append({
            'timestamp': time.time(),
            **stats
        })
        
        return stats
    
    def log_memory_stats(self, step: Optional[int] = None, prefix: str = "train"):
        """메모리 통계를 로깅"""
        stats = self.get_memory_stats()
        
        if not stats:
            return
        
        # 콘솔 로깅
        logger.info(f"GPU 메모리 사용: {stats['gpu_memory_allocated_gb']:.2f}/{stats['gpu_memory_total_gb']:.2f}GB "
                   f"({stats['gpu_memory_utilization_percent']:.1f}%)")
        
        # WandB 로깅
        if self.log_to_wandb and wandb.run is not None:
            wandb_stats = {f"{prefix}/{k}": v for k, v in stats.items()}
            if step is not None:
                wandb.log(wandb_stats, step=step)
            else:
                wandb.log(wandb_stats)
    
    def suggest_batch_size_adjustment(self, current_batch_size: int, 
                                    target_utilization: float = 80.0) -> Tuple[int, str]:
        """
        현재 메모리 사용률을 기반으로 배치 크기 조정 제안
        
        Args:
            current_batch_size: 현재 배치 크기
            target_utilization: 목표 메모리 사용률 (%)
            
        Returns:
            (제안 배치 크기, 조정 이유)
        """
        if not self.is_cuda or not self.memory_history:
            return current_batch_size, "No GPU memory data"
        
        # 최근 메모리 사용률 평균
        recent_stats = self.memory_history[-5:] if len(self.memory_history) >= 5 else self.memory_history
        avg_utilization = sum(s['gpu_memory_utilization_percent'] for s in recent_stats) / len(recent_stats)
        max_utilization = max(s['gpu_memory_max_utilization_percent'] for s in recent_stats)
        
        # 안전 마진 (10%)
        safe_target = target_utilization - 10
        
        if max_utilization < 50:
            # 메모리 사용률이 너무 낮음 - 배치 크기 증가
            scale_factor = safe_target / max_utilization
            suggested_size = int(current_batch_size * min(scale_factor, 2.0))  # 최대 2배까지
            reason = f"메모리 사용률이 낮음 ({max_utilization:.1f}% < 50%)"
            
        elif max_utilization > 90:
            # 메모리 사용률이 너무 높음 - 배치 크기 감소
            scale_factor = safe_target / max_utilization
            suggested_size = int(current_batch_size * max(scale_factor, 0.5))  # 최소 절반까지
            reason = f"메모리 사용률이 높음 ({max_utilization:.1f}% > 90%)"
            
        else:
            # 적절한 범위
            suggested_size = current_batch_size
            reason = f"메모리 사용률이 적절함 ({max_utilization:.1f}%)"
        
        # 2의 배수로 조정
        suggested_size = max(1, (suggested_size // 2) * 2)
        
        return suggested_size, reason
    
    def reset_stats(self):
        """통계 초기화"""
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        self.memory_history.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """전체 학습 과정의 메모리 사용 요약"""
        if not self.memory_history:
            return {}
        
        allocated_values = [s['gpu_memory_allocated_gb'] for s in self.memory_history]
        utilization_values = [s['gpu_memory_utilization_percent'] for s in self.memory_history]
        
        return {
            'gpu_memory_avg_allocated_gb': sum(allocated_values) / len(allocated_values),
            'gpu_memory_max_allocated_gb': max(allocated_values),
            'gpu_memory_avg_utilization_percent': sum(utilization_values) / len(utilization_values),
            'gpu_memory_max_utilization_percent': max(utilization_values),
            'gpu_memory_measurements': len(self.memory_history)
        }


def monitor_gpu_memory_simple() -> float:
    """
    RTX 3090 극한 최적화용 간단한 GPU 메모리 모니터링
    
    Returns:
        현재 GPU 메모리 사용량 (GB)
    """
    if not torch.cuda.is_available():
        return 0.0
    
    memory_info = torch.cuda.mem_get_info()
    used_gb = (memory_info[1] - memory_info[0]) / (1024**3)
    return used_gb


def check_rtx3090_memory_critical(threshold_gb: float = 22.0) -> bool:
    """
    RTX 3090 메모리 임계점 확인
    
    Args:
        threshold_gb: 임계점 (GB, 기본값 22GB)
        
    Returns:
        True if 메모리 사용량이 임계점 초과
    """
    current_usage = monitor_gpu_memory_simple()
    return current_usage > threshold_gb


def get_rtx3090_memory_status() -> Dict[str, Any]:
    """
    RTX 3090 메모리 상태 상세 정보
    
    Returns:
        메모리 상태 딕셔너리
    """
    if not torch.cuda.is_available():
        return {'available': False}
    
    memory_info = torch.cuda.mem_get_info()
    device_props = torch.cuda.get_device_properties(0)
    
    total_gb = memory_info[1] / (1024**3)
    free_gb = memory_info[0] / (1024**3)
    used_gb = total_gb - free_gb
    usage_percent = (used_gb / total_gb) * 100
    
    return {
        'available': True,
        'device_name': device_props.name,
        'total_gb': total_gb,
        'used_gb': used_gb,
        'free_gb': free_gb,
        'usage_percent': usage_percent,
        'is_rtx3090': 'RTX 3090' in device_props.name,
        'is_critical': used_gb > 22.0,
        'safe_for_optimization': used_gb < 20.0
    }


def optimize_batch_size_for_memory(model, tokenizer, sample_data, 
                                 device: torch.device,
                                 min_batch_size: int = 1,
                                 max_batch_size: int = 64) -> int:
    """
    OOM 없이 사용 가능한 최대 배치 크기 찾기
    
    Args:
        model: 테스트할 모델
        tokenizer: 토크나이저
        sample_data: 샘플 데이터
        device: 디바이스
        min_batch_size: 최소 배치 크기
        max_batch_size: 최대 배치 크기
        
    Returns:
        최적 배치 크기
    """
    logger.info("배치 크기 최적화 시작...")
    
    optimal_size = min_batch_size
    
    # 이진 탐색으로 최적 배치 크기 찾기
    left, right = min_batch_size, max_batch_size
    
    while left <= right:
        mid = (left + right) // 2
        
        try:
            # 메모리 캐시 초기화
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # 테스트 배치 생성
            test_batch = tokenizer(
                [sample_data] * mid,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            
            # Forward pass 테스트
            with torch.no_grad():
                _ = model(**test_batch)
            
            # 성공하면 더 큰 배치 시도
            optimal_size = mid
            left = mid + 1
            logger.info(f"배치 크기 {mid} 성공")
            
        except torch.cuda.OutOfMemoryError:
            # OOM 발생 시 더 작은 배치 시도
            right = mid - 1
            logger.info(f"배치 크기 {mid} 실패 (OOM)")
            
        except Exception as e:
            # 다른 에러 발생 시
            logger.warning(f"배치 크기 {mid} 테스트 중 에러: {e}")
            right = mid - 1
        
        finally:
            # 메모리 정리
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    logger.info(f"최적 배치 크기: {optimal_size}")
    return optimal_size
