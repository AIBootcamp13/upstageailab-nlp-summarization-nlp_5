"""
메모리 사용량 모니터링 유틸리티

QLoRA 및 unsloth의 메모리 절약 효과를 측정하고 모니터링하는 유틸리티입니다.
"""

import torch
import psutil
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """메모리 스냅샷 데이터 클래스"""
    timestamp: str
    cpu_memory_mb: float
    gpu_memory_mb: Optional[float]
    gpu_memory_reserved_mb: Optional[float]
    gpu_memory_allocated_mb: Optional[float]
    system_memory_percent: float
    model_type: str  # "standard", "qlora", "unsloth"
    

class MemoryMonitor:
    """
    메모리 사용량 모니터링 클래스
    
    QLoRA와 unsloth의 메모리 절약 효과를 측정하고 기록합니다.
    """
    
    def __init__(self, save_path: Optional[str] = None):
        """
        MemoryMonitor 초기화
        
        Args:
            save_path: 메모리 로그 저장 경로
        """
        self.snapshots: Dict[str, MemorySnapshot] = {}
        self.save_path = Path(save_path) if save_path else Path("memory_usage.json")
        self.baseline_memory = None
        
    def take_snapshot(self, label: str, model_type: str = "unknown") -> MemorySnapshot:
        """
        현재 메모리 상태 스냅샷 생성
        
        Args:
            label: 스냅샷 라벨
            model_type: 모델 타입 ("standard", "qlora", "unsloth")
            
        Returns:
            MemorySnapshot: 메모리 스냅샷
        """
        # CPU 메모리
        process = psutil.Process()
        cpu_memory_mb = process.memory_info().rss / 1024 / 1024
        system_memory_percent = psutil.virtual_memory().percent
        
        # GPU 메모리 (사용 가능한 경우)
        gpu_memory_mb = None
        gpu_memory_reserved_mb = None  
        gpu_memory_allocated_mb = None
        
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_usage() / 1024 / 1024
            gpu_memory_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
            gpu_memory_allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
        elif torch.backends.mps.is_available():
            # MPS 메모리는 직접 측정이 어려움
            gpu_memory_mb = 0
            
        snapshot = MemorySnapshot(
            timestamp=datetime.now().isoformat(),
            cpu_memory_mb=cpu_memory_mb,
            gpu_memory_mb=gpu_memory_mb,
            gpu_memory_reserved_mb=gpu_memory_reserved_mb,
            gpu_memory_allocated_mb=gpu_memory_allocated_mb,
            system_memory_percent=system_memory_percent,
            model_type=model_type
        )
        
        self.snapshots[label] = snapshot
        
        logger.info(f"메모리 스냅샷 '{label}' 생성:")
        logger.info(f"  - CPU: {cpu_memory_mb:.1f}MB")
        logger.info(f"  - 시스템 메모리: {system_memory_percent:.1f}%")
        if gpu_memory_mb is not None:
            logger.info(f"  - GPU: {gpu_memory_mb:.1f}MB")
            
        return snapshot
    
    def set_baseline(self, label: str = "baseline"):
        """
        베이스라인 메모리 설정
        
        Args:
            label: 베이스라인 라벨
        """
        self.baseline_memory = self.take_snapshot(label, "standard")
        logger.info(f"베이스라인 메모리 설정: {label}")
    
    def compare_with_baseline(self, current_label: str) -> Dict[str, float]:
        """
        현재 메모리와 베이스라인 비교
        
        Args:
            current_label: 현재 스냅샷 라벨
            
        Returns:
            Dict[str, float]: 메모리 변화율 (음수면 감소)
        """
        if self.baseline_memory is None:
            logger.warning("베이스라인 메모리가 설정되지 않았습니다.")
            return {}
            
        if current_label not in self.snapshots:
            logger.warning(f"스냅샷 '{current_label}'을 찾을 수 없습니다.")
            return {}
            
        current = self.snapshots[current_label]
        baseline = self.baseline_memory
        
        comparison = {}
        
        # CPU 메모리 변화율
        if baseline.cpu_memory_mb > 0:
            cpu_change = ((current.cpu_memory_mb - baseline.cpu_memory_mb) / baseline.cpu_memory_mb) * 100
            comparison['cpu_memory_change_percent'] = cpu_change
        
        # GPU 메모리 변화율
        if baseline.gpu_memory_mb is not None and current.gpu_memory_mb is not None:
            if baseline.gpu_memory_mb > 0:
                gpu_change = ((current.gpu_memory_mb - baseline.gpu_memory_mb) / baseline.gpu_memory_mb) * 100
                comparison['gpu_memory_change_percent'] = gpu_change
                
        return comparison
    
    def generate_report(self) -> Dict[str, Any]:
        """
        메모리 사용량 리포트 생성
        
        Returns:
            Dict[str, Any]: 메모리 리포트
        """
        report = {
            'snapshots': {},
            'comparisons': {},
            'summary': {}
        }
        
        # 스냅샷 정보
        for label, snapshot in self.snapshots.items():
            report['snapshots'][label] = {
                'timestamp': snapshot.timestamp,
                'cpu_memory_mb': snapshot.cpu_memory_mb,
                'gpu_memory_mb': snapshot.gpu_memory_mb,
                'system_memory_percent': snapshot.system_memory_percent,
                'model_type': snapshot.model_type
            }
        
        # 베이스라인과 비교
        if self.baseline_memory:
            for label, snapshot in self.snapshots.items():
                if label != 'baseline':
                    comparison = self.compare_with_baseline(label)
                    if comparison:
                        report['comparisons'][label] = comparison
        
        # 요약 통계
        if self.baseline_memory and len(self.snapshots) > 1:
            cpu_changes = []
            gpu_changes = []
            
            for label, comparison in report['comparisons'].items():
                if 'cpu_memory_change_percent' in comparison:
                    cpu_changes.append(comparison['cpu_memory_change_percent'])
                if 'gpu_memory_change_percent' in comparison:
                    gpu_changes.append(comparison['gpu_memory_change_percent'])
            
            if cpu_changes:
                report['summary']['avg_cpu_memory_change'] = sum(cpu_changes) / len(cpu_changes)
                report['summary']['max_cpu_memory_reduction'] = min(cpu_changes)  # 음수가 감소
                
            if gpu_changes:
                report['summary']['avg_gpu_memory_change'] = sum(gpu_changes) / len(gpu_changes)
                report['summary']['max_gpu_memory_reduction'] = min(gpu_changes)  # 음수가 감소
        
        return report
    
    def save_report(self, filepath: Optional[str] = None):
        """
        메모리 리포트를 파일로 저장
        
        Args:
            filepath: 저장 파일 경로
        """
        save_path = Path(filepath) if filepath else self.save_path
        report = self.generate_report()
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        logger.info(f"메모리 리포트 저장됨: {save_path}")
    
    def print_summary(self):
        """메모리 사용량 요약 출력"""
        report = self.generate_report()
        
        print("\n=== 메모리 사용량 요약 ===")
        print()
        
        # 스냅샷 정보
        print("📊 메모리 스냅샷:")
        for label, data in report['snapshots'].items():
            print(f"  {label}:")
            print(f"    - CPU: {data['cpu_memory_mb']:.1f}MB")
            print(f"    - 모델 타입: {data['model_type']}")
            if data['gpu_memory_mb'] is not None:
                print(f"    - GPU: {data['gpu_memory_mb']:.1f}MB")
        
        print()
        
        # 비교 결과
        if report['comparisons']:
            print("📈 베이스라인 대비 변화:")
            for label, comparison in report['comparisons'].items():
                print(f"  {label}:")
                if 'cpu_memory_change_percent' in comparison:
                    change = comparison['cpu_memory_change_percent']
                    symbol = "📉" if change < 0 else "📈"
                    print(f"    {symbol} CPU: {change:+.1f}%")
                if 'gpu_memory_change_percent' in comparison:
                    change = comparison['gpu_memory_change_percent']
                    symbol = "📉" if change < 0 else "📈"
                    print(f"    {symbol} GPU: {change:+.1f}%")
        
        # 요약 통계
        if report['summary']:
            print()
            print("🎯 요약 통계:")
            summary = report['summary']
            if 'max_cpu_memory_reduction' in summary:
                reduction = abs(summary['max_cpu_memory_reduction'])
                print(f"  - 최대 CPU 메모리 절약: {reduction:.1f}%")
            if 'max_gpu_memory_reduction' in summary:
                reduction = abs(summary['max_gpu_memory_reduction'])
                print(f"  - 최대 GPU 메모리 절약: {reduction:.1f}%")


# 편의 함수들
def quick_memory_check() -> Dict[str, float]:
    """빠른 메모리 확인"""
    monitor = MemoryMonitor()
    snapshot = monitor.take_snapshot("quick_check")
    
    return {
        'cpu_memory_mb': snapshot.cpu_memory_mb,
        'gpu_memory_mb': snapshot.gpu_memory_mb or 0,
        'system_memory_percent': snapshot.system_memory_percent
    }


def log_memory_usage(label: str, model_type: str = "unknown"):
    """메모리 사용량 로깅 데코레이터"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = MemoryMonitor()
            monitor.take_snapshot(f"{label}_start", model_type)
            
            try:
                result = func(*args, **kwargs)
                monitor.take_snapshot(f"{label}_end", model_type)
                return result
            except Exception as e:
                monitor.take_snapshot(f"{label}_error", model_type)
                raise e
                
        return wrapper
    return decorator
