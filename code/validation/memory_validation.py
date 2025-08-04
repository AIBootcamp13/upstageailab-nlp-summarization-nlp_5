"""
메모리 검증 및 관리 모듈

실험 전 메모리 요구사항을 추정하고 실험간 메모리를 완전히 정리합니다.
"""

import gc
import logging
import torch
import psutil
from typing import Dict, Any, Optional, Tuple
import subprocess
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class MemoryValidator:
    """메모리 요구사항 검증 및 관리 클래스"""
    
    def __init__(self):
        """MemoryValidator 초기화"""
        self.gpu_available = torch.cuda.is_available()
        self.device_info = self._get_device_info()
        
    def _get_device_info(self) -> Dict[str, Any]:
        """디바이스 정보 수집"""
        info = {
            "cpu_cores": psutil.cpu_count(),
            "ram_total_gb": psutil.virtual_memory().total / (1024**3),
            "ram_available_gb": psutil.virtual_memory().available / (1024**3),
            "gpu_available": self.gpu_available
        }
        
        if self.gpu_available:
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
        
        return info
    
    def cleanup_gpu_memory(self) -> Dict[str, Any]:
        """
        GPU 메모리 완전 정리
        
        Returns:
            정리 결과
        """
        cleanup_result = {
            "success": False,
            "memory_before_mb": 0,
            "memory_after_mb": 0,
            "memory_freed_mb": 0
        }
        
        if not self.gpu_available:
            cleanup_result["success"] = True
            cleanup_result["message"] = "GPU 없음 - 정리 불필요"
            return cleanup_result
        
        try:
            # 정리 전 메모리 사용량
            memory_before = torch.cuda.memory_allocated() / (1024**2)
            cleanup_result["memory_before_mb"] = memory_before
            
            # 1. PyTorch 캐시 정리
            torch.cuda.empty_cache()
            
            # 2. GPU 동기화
            torch.cuda.synchronize()
            
            # 3. Python 가비지 컬렉션
            gc.collect()
            
            # 4. PyTorch 메모리 강제 해제
            if torch.cuda.is_available():
                # 모든 GPU 디바이스의 메모리 정리
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
            
            # 정리 후 메모리 사용량
            memory_after = torch.cuda.memory_allocated() / (1024**2)
            cleanup_result["memory_after_mb"] = memory_after
            cleanup_result["memory_freed_mb"] = memory_before - memory_after
            cleanup_result["success"] = True
            
            logger.info(f"🧹 GPU 메모리 정리 완료: {cleanup_result['memory_freed_mb']:.1f}MB 해제")
            
        except Exception as e:
            cleanup_result["error"] = str(e)
            logger.error(f"GPU 메모리 정리 실패: {e}")
        
        return cleanup_result
    
    def estimate_memory_requirements(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        실험 설정에 따른 메모리 요구사항 추정
        
        Args:
            config: 실험 설정
            
        Returns:
            메모리 요구사항 추정 결과
        """
        model_name = config.get('general', {}).get('model_name', '')
        
        # 모델별 기본 메모리 요구량 (GB)
        model_base_memory = {
            'kobart': 2.0,      # KoBART (작은 모델)
            't5-base': 3.0,     # T5-base
            't5-large': 8.0,    # T5-large  
            'mt5': 4.0,         # mT5
            'eenzeenee': 3.0    # eenzeenee T5-base Korean
        }
        
        # 모델 타입 감지
        model_type = 'kobart'
        model_name_lower = model_name.lower()
        if 'eenzeenee' in model_name_lower:
            model_type = 'eenzeenee'
        elif 'mt5' in model_name_lower:
            model_type = 'mt5'
        elif 't5-large' in model_name_lower:
            model_type = 't5-large'
        elif 't5' in model_name_lower:
            model_type = 't5-base'
        elif 'bart' in model_name_lower or 'kobart' in model_name_lower:
            model_type = 'kobart'
        
        base_memory = model_base_memory.get(model_type, 3.0)
        
        # 설정별 메모리 증가 계산
        training_config = config.get('training', {})
        batch_size = training_config.get('per_device_train_batch_size', 8)
        encoder_len = config.get('tokenizer', {}).get('encoder_max_len', 1024)
        decoder_len = config.get('tokenizer', {}).get('decoder_max_len', 200)
        
        # 메모리 요구량 계산
        # 공식: base_memory * batch_multiplier * sequence_multiplier
        batch_multiplier = max(1.0, batch_size / 8)  # 배치 8 기준
        sequence_multiplier = max(1.0, (encoder_len + decoder_len) / 1200)  # 1200 토큰 기준
        
        estimated_memory_gb = base_memory * batch_multiplier * sequence_multiplier
        
        # QLoRA/Unsloth 사용시 메모리 절약
        qlora_config = config.get('qlora', {})
        if qlora_config.get('use_qlora', False):
            estimated_memory_gb *= 0.4  # QLoRA는 약 60% 메모리 절약
        if qlora_config.get('use_unsloth', False):
            estimated_memory_gb *= 0.7  # Unsloth는 추가 30% 절약
        
        # 결과 생성
        result = {
            "model_type": model_type,
            "model_name": model_name,
            "base_memory_gb": base_memory,
            "batch_size": batch_size,
            "sequence_length": encoder_len + decoder_len,
            "estimated_memory_gb": estimated_memory_gb,
            "batch_multiplier": batch_multiplier,
            "sequence_multiplier": sequence_multiplier,
            "use_qlora": qlora_config.get('use_qlora', False),
            "use_unsloth": qlora_config.get('use_unsloth', False),
            "available_memory_gb": self.device_info.get("gpu_memory_gb", 0),
            "memory_sufficient": False,
            "recommendations": []
        }
        
        # 메모리 충분성 판단
        available_memory = self.device_info.get("gpu_memory_gb", 0)
        if available_memory > 0:
            # 80% 안전 마진 적용
            safe_memory = available_memory * 0.8
            result["memory_sufficient"] = estimated_memory_gb <= safe_memory
            result["memory_utilization_percent"] = (estimated_memory_gb / available_memory) * 100
            
            # 권장사항 생성
            if not result["memory_sufficient"]:
                overage = estimated_memory_gb - safe_memory
                result["recommendations"].append(
                    f"🚨 메모리 부족: {overage:.1f}GB 초과 (요구: {estimated_memory_gb:.1f}GB, 안전 한계: {safe_memory:.1f}GB)"
                )
                
                # 배치 크기 조정 권장
                safe_batch_size = max(1, int(batch_size * safe_memory / estimated_memory_gb))
                result["recommendations"].append(
                    f"💡 배치 크기를 {batch_size} → {safe_batch_size}로 줄이세요"
                )
                
                # 시퀀스 길이 조정 권장
                if encoder_len > 512:
                    result["recommendations"].append(
                        f"💡 encoder_max_len을 {encoder_len} → 512로 줄이세요"
                    )
                
                # QLoRA 사용 권장
                if not qlora_config.get('use_qlora', False):
                    result["recommendations"].append(
                        "💡 QLoRA를 활성화하여 메모리를 60% 절약하세요"
                    )
            else:
                result["recommendations"].append(
                    f"✅ 메모리 충분: {result['memory_utilization_percent']:.1f}% 사용예정"
                )
        
        return result
    
    def get_safe_batch_size(self, config: Dict[str, Any]) -> int:
        """
        안전한 배치 크기 계산
        
        Args:
            config: 실험 설정
            
        Returns:
            안전한 배치 크기
        """
        memory_est = self.estimate_memory_requirements(config)
        
        if memory_est["memory_sufficient"]:
            return config.get('training', {}).get('per_device_train_batch_size', 8)
        
        # 메모리 부족시 안전한 배치 크기 계산
        current_batch = config.get('training', {}).get('per_device_train_batch_size', 8)
        available_memory = self.device_info.get("gpu_memory_gb", 8) * 0.8
        required_memory = memory_est["estimated_memory_gb"]
        
        if required_memory > 0:
            safe_batch = max(1, int(current_batch * available_memory / required_memory))
            return safe_batch
        
        return max(1, current_batch // 2)
    
    def auto_adjust_config_for_memory(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        메모리에 맞게 설정 자동 조정
        
        Args:
            config: 원본 설정
            
        Returns:
            조정된 설정
        """
        adjusted_config = config.copy()
        memory_est = self.estimate_memory_requirements(config)
        
        if memory_est["memory_sufficient"]:
            logger.info("✅ 메모리 충분 - 설정 조정 불필요")
            return adjusted_config
        
        logger.warning("⚠️ 메모리 부족 감지 - 설정 자동 조정")
        
        # 1. 배치 크기 조정
        safe_batch = self.get_safe_batch_size(config)
        if 'training' not in adjusted_config:
            adjusted_config['training'] = {}
        adjusted_config['training']['per_device_train_batch_size'] = safe_batch
        
        # 평가 배치는 더 크게 설정 가능
        eval_batch = min(safe_batch * 2, 32)
        adjusted_config['training']['per_device_eval_batch_size'] = eval_batch
        
        logger.info(f"📉 배치 크기 조정: {config.get('training', {}).get('per_device_train_batch_size', 8)} → {safe_batch}")
        
        # 2. 시퀀스 길이 조정 (필요시)
        tokenizer_config = adjusted_config.get('tokenizer', {})
        current_encoder_len = tokenizer_config.get('encoder_max_len', 1024)
        
        if current_encoder_len > 1024:
            adjusted_config['tokenizer']['encoder_max_len'] = 512
            logger.info(f"📉 Encoder 길이 조정: {current_encoder_len} → 512")
        
        # 3. 데이터로더 워커 수 조정
        cpu_cores = self.device_info.get("cpu_cores", 4)
        safe_workers = min(cpu_cores // 2, 16)  # 보수적 설정
        adjusted_config['training']['dataloader_num_workers'] = safe_workers
        
        # 4. QLoRA 강제 활성화 (메모리 절약)
        if 'qlora' not in adjusted_config:
            adjusted_config['qlora'] = {}
        
        adjusted_config['qlora']['use_qlora'] = True
        adjusted_config['qlora']['load_in_4bit'] = True
        
        logger.info("🔧 QLoRA 강제 활성화로 메모리 절약")
        
        return adjusted_config


def estimate_memory_requirements(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    메모리 요구사항 추정 편의 함수
    
    Args:
        config: 실험 설정
        
    Returns:
        메모리 추정 결과
    """
    validator = MemoryValidator()
    return validator.estimate_memory_requirements(config)


def cleanup_between_experiments() -> bool:
    """
    실험간 메모리 완전 정리
    
    Returns:
        정리 성공 여부
    """
    validator = MemoryValidator()
    result = validator.cleanup_gpu_memory()
    return result["success"]


def auto_fix_memory_config(config: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """
    메모리 문제 자동 수정
    
    Args:
        config: 원본 설정
        
    Returns:
        (수정된 설정, 수정 여부)
    """
    validator = MemoryValidator()
    original_sufficient = validator.estimate_memory_requirements(config)["memory_sufficient"]
    
    if original_sufficient:
        return config, False
    
    fixed_config = validator.auto_adjust_config_for_memory(config)
    return fixed_config, True
